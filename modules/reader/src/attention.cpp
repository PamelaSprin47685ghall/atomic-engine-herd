#include <tq/reader/attention.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace tq::reader {
namespace {

// One head of one cell, as a contiguous run. The layout is the one attention.h
// documents; asking for it in one place keeps the three loops below from disagreeing
// about the stride.
//
// `head` is a KEY/VALUE head, not a query head. The stride uses n_head_kv, so a caller
// that passes the query's larger head count here would address a different cell's
// key -- the defect this signature exists to make unrepresentable.
const float * cell_head(const float * base, const kv_buffers & kv,
                        std::size_t cell, std::size_t head) {
    return base + (cell * kv.kv_heads() + head) * kv.head_dim;
}

} // namespace

attention_result attend(const float * q,
                        const kv_buffers & kv,
                        const query_view & view,
                        float kq_scale,
                        const rope_fn & rope) {
    attention_result result;
    // A failed call returns an EMPTY result, not the partially filled buffer it happened
    // to build.
    //
    // `out` is sized up front and written head by head, so a failure at head 3 -- an
    // overflowing dot product, a non-finite score, a NaN in V -- used to leave heads 0..2
    // in the vector while reporting ok == false. A caller that checks only `out.size()`
    // (the natural thing to do with a vector-returning API, and what a `if (!r.out.empty())`
    // guard looks like) then reads real numbers from a call that reported failure. Every
    // exit below goes through this instead, so "no answer" and "a partial answer" cannot
    // be confused for one another.
    const auto failure = [] { return attention_result{}; };
    if (!q || !kv.valid() || !rope) {
        return failure();
    }

    // Gather the visible cells in span order, and refuse anything the buffers cannot
    // address. Checking here rather than at each access is what keeps a bad cell index
    // from becoming a read of a neighbouring key -- which would produce plausible
    // numbers instead of a failure.
    std::vector<std::size_t> cells;
    std::vector<std::size_t> span_of;      // parallel to cells: which span each came from
    std::vector<std::int32_t> span_pos;    // the rotation for each span, by index
    for (const span_keys & span : view.spans) {
        if (span.cells.empty()) {
            continue; // an empty span contributes nothing and must not open a rotation
        }
        span_pos.push_back(span.effective_pos);
        for (const std::int32_t cell : span.cells) {
            if (cell < 0 || std::size_t(cell) >= kv.n_cell) {
                return failure();
            }
            cells.push_back(std::size_t(cell));
            span_of.push_back(span_pos.size() - 1);
        }
    }
    if (cells.empty()) {
        // Nothing to normalize. The softmax of an empty set is undefined, not zero, so
        // this is a failure the caller must see rather than a zero vector that looks
        // like "the query saw nothing".
        return failure();
    }

    const std::size_t n_head = kv.n_head;
    const std::size_t kv_heads = kv.kv_heads();
    const std::size_t group = n_head / kv_heads;   // >= 1; valid() guarantees it divides
    const std::size_t head_dim = kv.head_dim;

    // "every visible key, once" -- checked, not assumed.
    //
    // views.h's resolve() never emits the same physical cell twice, but this function
    // takes a query_view directly, and a repeated cell is weighted twice: two cells
    // presented as three answered 1.333 where the correct value was 1.5. That looks like
    // an ordinary result, which is the failure mode this whole file exists to avoid.
    //
    // One sort of the gathered indices: O(n log n) beside the n * head_dim * n_head
    // multiply-adds below, i.e. ~100k comparisons against 32M flops for a large view.
    {
        std::vector<std::size_t> ordered = cells;
        std::sort(ordered.begin(), ordered.end());
        if (std::adjacent_find(ordered.begin(), ordered.end()) != ordered.end()) {
            return failure();
        }
    }

    // The span-rotation buffer is spans x head_dim, and the span count comes from the
    // caller's view -- unlike n_cell it is not bounded by the buffers, so this product
    // has to be checked on its own. A wrapped value here allocates a short buffer and the
    // copy below writes past its end.
    if (!kv_buffers::product_fits(span_pos.size(), head_dim, 1)) {
        return failure();
    }

    // Filled locally and moved into the result only on success.
    //
    // This is what makes every `return failure()` above honest without touching each
    // site: `out` used to be sized up front and written head by head, so a failure at
    // head 3 left heads 0..2 in the vector while reporting ok == false -- a partial
    // answer that a caller checking `out.size()` would read as a result. Now a failed
    // call cannot carry anything, by construction rather than by remembering to clear it.
    std::vector<float> values(n_head * head_dim, 0.0f);

    std::vector<float> q_rot(head_dim);
    std::vector<float> scores(cells.size());
    std::vector<float> weights(cells.size());

    // One rotation per (head, span), shared by every key in the span. That is the
    // fork's structure too (build_attn_reader ropes once per span and multiplies the
    // gathered keys against it), and it is why span_keys groups cells by position.
    std::vector<float> rotated_for_span(span_pos.size() * head_dim);

    for (std::size_t h = 0; h < n_head; ++h) {
        // GQA: query head h reads kv head h / group. `98 -> 24` on a Llama-3 style 4:1
        // model, and identity when the counts match.
        const std::size_t kv_head = h / group;

        for (std::size_t s = 0; s < span_pos.size(); ++s) {
            std::copy(q + h * head_dim, q + h * head_dim + head_dim, q_rot.begin());
            rope(q_rot.data(), head_dim, span_pos[s]);
            std::copy(q_rot.begin(), q_rot.end(),
                      rotated_for_span.begin() + std::ptrdiff_t(s * head_dim));
        }

        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < cells.size(); ++i) {
            const float * k = cell_head(kv.k, kv, cells[i], kv_head);
            const float * qh = rotated_for_span.data() + span_of[i] * head_dim;
            double dot = 0.0;
            for (std::size_t d = 0; d < head_dim; ++d) {
                dot += double(k[d]) * double(qh[d]);
            }
            const float scaled = float(dot) * kq_scale;
            // Checked per score, not only on the maximum. std::max drops a NaN (every
            // comparison against NaN is false, so it keeps the accumulator), which means
            // a NaN score would leave `maximum` finite and slip past a check on it alone.
            if (!std::isfinite(scaled)) {
                return result;
            }
            scores[i] = scaled;
            maximum = std::max(maximum, scaled);
        }

        double denominator = 0.0;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            const double w = std::exp(double(scores[i]) - double(maximum));
            weights[i] = float(w);
            denominator += w;
        }
        if (!(denominator > 0.0)) {
            return result;
        }

        for (std::size_t d = 0; d < head_dim; ++d) {
            double sum = 0.0;
            for (std::size_t i = 0; i < cells.size(); ++i) {
                sum += double(weights[i]) * double(cell_head(kv.v, kv, cells[i], kv_head)[d]);
            }
            const float value = float(sum / denominator);
            // The value reduction is where NaN or infinity in V lands. Failing here is the
            // same rule the score check follows: a caller that reads ok == false reports a
            // failure, while one that reads ok == true with a NaN in hand uses it.
            if (!std::isfinite(value)) {
                return result;
            }
            values[h * head_dim + d] = value;
        }
    }

    // The only place a result becomes non-empty. Everything above either filled `values`
    // and fell through, or returned through `failure()` with nothing in it -- so
    // `out.empty() == !ok` holds for every caller without anyone having to remember it.
    result.out       = std::move(values);
    result.n_visible = cells.size();
    result.ok        = true;
    return result;
}

query_view identity_view(std::int32_t first_cell, std::size_t n_keys,
                         std::int32_t first_pos) {
    // One span per position: distinct effective positions, ascending, which is the
    // shape views.h's validator accepts (strictly ascending virtual positions) and the
    // shape a plain causal attention has.
    query_view view;
    view.spans.reserve(n_keys);
    for (std::size_t i = 0; i < n_keys; ++i) {
        span_keys span;
        span.effective_pos = first_pos + std::int32_t(i);
        span.cells.push_back(first_cell + std::int32_t(i));
        view.spans.push_back(std::move(span));
    }
    return view;
}

} // namespace tq::reader
