// The physical attention core against three things: exact arithmetic, ggml's own kernels, and its
// refusals.
//
// Why three
// ---------
// A core compared only with a hand reference agrees with itself: if the reference implements the
// same misunderstanding, both are wrong the same way and the test passes. A core compared only with
// ggml agrees with one implementation, which says nothing about whether it is the fork's semantics.
// So this file does both, and each covers what the other cannot:
//
//   * the DOUBLE reference states the semantics in the least ambiguous form available: scatter to
//     the physical length with -infinity, scale, max, exp, one global sum, then reduce in increasing
//     cell order. If the core and this disagree, the core has a defect.
//   * the GGML path builds the same operations as tensors — `ggml_fill(-INFINITY)`,
//     `ggml_set_rows`, `ggml_soft_max_ext`, `ggml_mul_mat` — which is the fork's own graph. If the
//     core and this disagree, the core does not reproduce what the product would compute, whatever
//     the reference says.
//
// A third comparison is printed rather than asserted, because it is the reason this file exists and
// it is not a pass/fail fact: the COMPACT union sum (attention.h) versus the physical form. They are
// equal in real arithmetic and differ in floating point, and the fork chose the physical form for
// exactly that reason. Printing the difference makes the fork's claim visible; asserting on it would
// be asserting a rounding result.
#include <tq/reader/physical.h>

#include <ggml.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr std::size_t kHeadDim = 8;
constexpr std::size_t kKvHeads = 2;
constexpr std::size_t kHeads = 4;
constexpr std::size_t kCells = 12;
constexpr float kScale = 0.70710678f;

// Cells the reader may see: a gap in the middle and a gap at the end, so "the physical length" and
// "the visible set" are visibly different numbers and a compact implementation cannot pass by luck.
std::vector<std::int32_t> span_a_cells() { return {0, 1, 3}; }
std::vector<std::int32_t> span_b_cells() { return {5, 10}; }

// Cache values, deterministic and asymmetric so a wrong gather or a wrong head pairing cannot
// produce the right answer by coincidence.
std::vector<float> cache_values() {
    std::vector<float> v(kHeadDim * kKvHeads * kCells, 0.0f);
    unsigned state = 0x9e3779b9u;
    for (float & value : v) {
        state = state * 1664525u + 1013904223u;
        value = float((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
    return v;
}

std::vector<float> span_scores(const std::vector<std::int32_t> & cells, float offset) {
    std::vector<float> scores;
    scores.reserve(cells.size());
    for (std::size_t i = 0; i < cells.size(); ++i) {
        scores.push_back(0.7f * float(i) - 0.4f + offset + 0.03f * float(cells[i]));
    }
    return scores;
}

std::vector<tq::reader::physical_span> the_spans() {
    std::vector<tq::reader::physical_span> spans(2);
    spans[0].cells = span_a_cells();
    spans[0].scores = span_scores(spans[0].cells, 0.0f);
    spans[1].cells = span_b_cells();
    spans[1].scores = span_scores(spans[1].cells, 1.1f);
    return spans;
}

// The semantics, in double, written the most literal way: a full-length array, -infinity for
// hidden, scale, max, exp, one sum, divide, then a reduction in increasing cell order.
struct reference_result {
    std::vector<double> weights;   // [kCells]
    std::vector<double> out;       // [kHeads][kHeadDim]
};

reference_result reference(const std::vector<tq::reader::physical_span> & spans,
                           const std::vector<float> & v, double scale) {
    reference_result r;
    r.weights.assign(kCells, -std::numeric_limits<double>::infinity());
    for (const auto & span : spans) {
        for (std::size_t j = 0; j < span.cells.size(); ++j) {
            r.weights[std::size_t(span.cells[j])] = double(span.scores[j]);
        }
    }

    double maximum = -std::numeric_limits<double>::infinity();
    for (double & w : r.weights) {
        if (std::isinf(w) && w < 0) continue;
        w *= scale;
        if (w > maximum) maximum = w;
    }
    double total = 0.0;
    for (double & w : r.weights) {
        if (std::isinf(w) && w < 0) { w = 0.0; continue; }
        w = std::exp(w - maximum);
        total += w;
    }
    for (double & w : r.weights) w /= total;

    const std::size_t group = kHeads / kKvHeads;
    r.out.assign(kHeads * kHeadDim, 0.0);
    for (std::size_t head = 0; head < kHeads; ++head) {
        const std::size_t kv_head = head / group;
        for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
            double acc = 0.0;
            for (std::size_t cell = 0; cell < kCells; ++cell) {
                acc += r.weights[cell] *
                       double(v[dim + kHeadDim * (kv_head + kKvHeads * cell)]);
            }
            r.out[head * kHeadDim + dim] = acc;
        }
    }
    return r;
}

double max_abs_diff(const std::vector<float> & a, const std::vector<double> & b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double delta = std::fabs(double(a[i]) - b[i]);
        if (delta > worst) worst = delta;
    }
    return worst;
}

} // namespace

int main() {
    const std::vector<tq::reader::physical_span> spans = the_spans();
    const std::vector<float> v = cache_values();
    const reference_result expected = reference(spans, v, double(kScale));

    // ---- 1. the core against exact arithmetic ----
    std::vector<float> weights;
    std::vector<float> out;
    const auto attended = tq::reader::attend_physical(
        spans.data(), spans.size(), kCells, kScale, v.data(), kHeadDim, kKvHeads, kHeads, weights,
        out);
    check(attended == tq::reader::physical_refusal::ok, "core: the fixture is a valid emission");
    const double vs_reference = max_abs_diff(out, expected.out);
    std::fprintf(stderr, "core vs double reference: max|diff| = %.3e\n", vs_reference);
    check(vs_reference < 1e-6,
          "core: the physical attention matches exact arithmetic (float accumulation only)");

    // ---- 2. the invariant the -infinity fill exists for ----
    {
        std::size_t visible = 0;
        std::size_t hidden = 0;
        double mass = 0.0;
        for (std::size_t cell = 0; cell < kCells; ++cell) {
            const bool named = std::find(spans[0].cells.begin(), spans[0].cells.end(),
                                         std::int32_t(cell)) != spans[0].cells.end() ||
                               std::find(spans[1].cells.begin(), spans[1].cells.end(),
                                         std::int32_t(cell)) != spans[1].cells.end();
            if (named) {
                ++visible;
                check(weights[cell] > 0.0f, "a visible cell has a positive weight");
            } else {
                ++hidden;
                check(weights[cell] == 0.0f,
                      "a hidden cell has weight EXACTLY zero: that is what -infinity buys, and it is "
                      "why the full length and the compact set agree in real arithmetic");
            }
            mass += double(weights[cell]);
        }
        check(visible == 5 && hidden == 7, "fixture: 5 of 12 cells are visible");
        check(std::fabs(mass - 1.0) < 1e-6, "core: the weights sum to one over the full length");
    }

    // ---- 3. the core against ggml's own kernels, on the same scatter ----
    //
    // The scatter is applied on the host and copied into a leaf, then ggml runs the two operations
    // that decide the numbers: one global softmax over the full length, and the value multiply.
    // Validating the scatter NODE separately (case 4) keeps this case about the maths rather than
    // about `ggml_set_rows`.
    {
        ggml_context * ctx = ggml_init({64u * 1024u * 1024u, nullptr, /*no_alloc=*/false});
        check(ctx != nullptr, "ggml: context created");

        // The scatter, done by the core and written into the graph's input by hand.
        std::vector<float> scattered;
        check(tq::reader::scatter(spans.data(), spans.size(), kCells, -INFINITY, scattered) ==
                  tq::reader::physical_refusal::ok,
              "ggml: the core scattered the scores");

        ggml_tensor * logits = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kCells, 1, kHeads);
        // Broadcast the single scattered row across the heads: every head sees the same logits in
        // this fixture (the scores are per-cell, not per-head), which is the fork's shape when a
        // span's partial scores are written into one physical vector per head.
        for (std::size_t head = 0; head < kHeads; ++head) {
            std::memcpy(static_cast<float *>(logits->data) + head * kCells, scattered.data(),
                        kCells * sizeof(float));
        }

        ggml_tensor * weights_t = ggml_soft_max_ext(ctx, logits, nullptr, kScale, 0.0f);

        ggml_tensor * v_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
        std::memcpy(v_t->data, v.data(), v.size() * sizeof(float));

        // The fork's V handling, re-expressed: permute, then transpose into the multiply's layout.
        ggml_tensor * v_perm = ggml_permute(ctx, v_t, 0, 2, 1, 3);   // [hd, n_kv, kvh]
        ggml_tensor * v_mult = ggml_cont(ctx, ggml_transpose(ctx, v_perm));   // [n_kv, hd, kvh]
        ggml_tensor * product = ggml_mul_mat(ctx, v_mult, weights_t);   // [hd, 1, nh]
        ggml_tensor * result = ggml_cont(ctx, ggml_permute(ctx, product, 0, 2, 1, 3));  // [hd, nh, 1]

        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, result);
        const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 1);
        check(status == GGML_STATUS_SUCCESS, "ggml: the graph computed");

        // ggml lays out [hd, nh] as dim + hd*head; the core uses head*hd + dim. Compare by value.
        const float * got = static_cast<const float *>(result->data);
        double worst = 0.0;
        for (std::size_t head = 0; head < kHeads; ++head) {
            for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
                const double delta = std::fabs(double(got[dim + kHeadDim * head]) -
                                               double(out[head * kHeadDim + dim]));
                if (delta > worst) worst = delta;
            }
        }
        std::fprintf(stderr, "core vs ggml kernels: max|diff| = %.3e\n", worst);
        check(worst < 1e-6,
              "core: matches ggml's soft_max_ext + mul_mat on the same scattered logits, so the "
              "core computes what the product's graph would");

        // And ggml's weights agree with the core's, cell by cell -- the normalisation is the part
        // the fork's comment is about, so it should be checked directly rather than inferred from
        // a downstream product.
        ggml_tensor * weights_only = ggml_cont(ctx, weights_t);
        ggml_cgraph * graph2 = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph2, weights_only);
        ggml_graph_compute_with_ctx(ctx, graph2, 1);
        const float * ggml_w = static_cast<const float *>(weights_only->data);
        double worst_w = 0.0;
        for (std::size_t head = 0; head < kHeads; ++head) {
            for (std::size_t cell = 0; cell < kCells; ++cell) {
                const double delta =
                    std::fabs(double(ggml_w[cell + kCells * head]) - double(weights[cell]));
                if (delta > worst_w) worst_w = delta;
            }
        }
        std::fprintf(stderr, "core vs ggml weights: max|diff| = %.3e\n", worst_w);
        check(worst_w < 1e-6,
              "core: the weights match ggml_soft_max_ext with the scale applied, including the "
              "exact zeros for hidden cells");

        ggml_free(ctx);
    }

    // ---- 4. the scatter NODE: ggml_set_rows against the core's scatter ----
    //
    // The fork writes each span's scores into the physical vector with `ggml_set_rows`, so a core
    // that computes the right answer from a host scatter still has to agree with what that node
    // does. `all_cells` is a permutation, deliberately: a scatter that ignored the index tensor and
    // wrote in order would pass an identity permutation.
    {
        ggml_context * ctx = ggml_init({16u * 1024u * 1024u, nullptr, /*no_alloc=*/false});
        ggml_tensor * physical = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kCells, 1);
        physical = ggml_fill(ctx, physical, -INFINITY);

        std::vector<std::int32_t> cells;
        std::vector<float> scores;
        for (const auto & span : spans) {
            cells.insert(cells.end(), span.cells.begin(), span.cells.end());
            scores.insert(scores.end(), span.scores.begin(), span.scores.end());
        }
        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, std::int64_t(cells.size()));
        std::memcpy(idx->data, cells.data(), cells.size() * sizeof(std::int32_t));
        ggml_tensor * rows = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, std::int64_t(cells.size()), 1);
        for (std::size_t i = 0; i < scores.size(); ++i) {
            static_cast<float *>(rows->data)[i] = scores[i];
        }

        ggml_tensor * scattered_t = ggml_set_rows(ctx, physical, rows, idx);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, scattered_t);
        const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 1);
        check(status == GGML_STATUS_SUCCESS, "set_rows: the graph computed");

        std::vector<float> host;
        check(tq::reader::scatter(spans.data(), spans.size(), kCells, -INFINITY, host) ==
                  tq::reader::physical_refusal::ok,
              "set_rows: the core scattered the same spans");

        const float * node = static_cast<const float *>(scattered_t->data);
        std::size_t mismatched = 0;
        for (std::size_t cell = 0; cell < kCells; ++cell) {
            const float a = node[cell];
            const float b = host[cell];
            const bool both_hidden = (std::isinf(a) && a < 0) && (std::isinf(b) && b < 0);
            if (!both_hidden && a != b) ++mismatched;
        }
        std::fprintf(stderr, "set_rows vs core scatter: %zu cell(s) differ\n", mismatched);
        check(mismatched == 0,
              "core: ggml_set_rows writes the same physical vector as the core's scatter, so the "
              "fork's node and this rule agree cell for cell");
        ggml_free(ctx);
    }

    // ---- 5. the difference the fork's comment is about, measured in the same precision ----
    //
    // The first version of this block compared the core's FLOAT output against a DOUBLE compact
    // sum and printed 1.451e-08 as "the difference between the physical and compact forms". That
    // number is the float storage floor, not the accumulation order: it would have appeared even
    // if the two orders agreed bit for bit. Printing it under that label was misleading, and the
    // assertion built on it (agree to 1e-12) failed for the same reason.
    //
    // So both are now accumulated in float and the comparison counts BIT-level differences. That
    // isolates the one thing this case is about, and it can report "zero" honestly -- which would
    // mean the order made no difference on this fixture, and that is what the log should say.
    {
        std::vector<float> compact;
        float maximum = -INFINITY;
        for (const auto & span : spans) {
            for (float s : span.scores) {
                const float scaled = s * kScale;
                if (scaled > maximum) maximum = scaled;
            }
        }
        float total = 0.0f;
        for (const auto & span : spans) {
            for (float s : span.scores) {
                const float e = std::exp(s * kScale - maximum);
                compact.push_back(e);
                total += e;
            }
        }
        for (float & e : compact) e /= total;

        std::vector<float> compact_out(kHeads * kHeadDim, 0.0f);
        const std::size_t group = kHeads / kKvHeads;
        for (std::size_t head = 0; head < kHeads; ++head) {
            const std::size_t kv_head = head / group;
            for (std::size_t dim = 0; dim < kHeadDim; ++dim) {
                float acc = 0.0f;
                std::size_t k = 0;
                for (const auto & span : spans) {
                    for (std::int32_t cell : span.cells) {
                        acc += compact[k++] * v[dim + kHeadDim * (kv_head + kKvHeads * cell)];
                    }
                }
                compact_out[head * kHeadDim + dim] = acc;
            }
        }

        std::size_t differing = 0;
        for (std::size_t i = 0; i < out.size() && i < compact_out.size(); ++i) {
            if (std::memcmp(&out[i], &compact_out[i], sizeof(float)) != 0) ++differing;
        }
        double worst = 0.0;
        for (std::size_t i = 0; i < out.size() && i < compact_out.size(); ++i) {
            const double delta = std::fabs(double(out[i]) - double(compact_out[i]));
            if (delta > worst) worst = delta;
        }
        std::fprintf(stderr,
                     "physical vs compact (both float): %zu of %zu values differ, max|diff| = %.3e\n",
                     differing, out.size(), worst);
        check(worst < 1e-6,
              "the physical and compact forms agree within float rounding on this fixture -- equal "
              "in real arithmetic, as they must be, because a hidden cell contributes exactly zero");

        // The fork's reason is about accumulation at scale, and this fixture is 12 cells with 5
        // visible. Saying "the order matters here" would be a claim the measurement does not
        // support, so the case records what it found instead of asserting a difference it did not
        // observe.
        if (differing == 0) {
            std::fprintf(stderr,
                         "  (on this fixture the two orders happened to agree bit for bit; the "
                         "fork's concern is a longer sum, and this case does not manufacture one)\n");
        }
    }

    // ---- 6. refusals: each names a cause, and no partial result escapes ----
    {
        std::vector<float> w;
        std::vector<float> o;

        std::vector<tq::reader::physical_span> duplicate(2);
        duplicate[0].cells = {0, 1};
        duplicate[0].scores = {1.0f, 2.0f};
        duplicate[1].cells = {1, 2};   // cell 1 named twice
        duplicate[1].scores = {3.0f, 4.0f};
        check(tq::reader::scatter(duplicate.data(), duplicate.size(), kCells, -INFINITY, w) ==
                  tq::reader::physical_refusal::duplicate_cell,
              "refusal: a doubly named cell is refused, because one of its scores would be "
              "discarded according to iteration order");

        std::vector<tq::reader::physical_span> outside(1);
        outside[0].cells = {0, kCells};
        outside[0].scores = {1.0f, 2.0f};
        check(tq::reader::scatter(outside.data(), outside.size(), kCells, -INFINITY, w) ==
                  tq::reader::physical_refusal::cell_out_of_range,
              "refusal: a cell at n_kv is out of range, not silently dropped");

        std::vector<tq::reader::physical_span> mismatched(1);
        mismatched[0].cells = {0, 1, 2};
        mismatched[0].scores = {1.0f, 2.0f};
        check(tq::reader::scatter(mismatched.data(), mismatched.size(), kCells, -INFINITY, w) ==
                  tq::reader::physical_refusal::length_mismatch,
              "refusal: a span whose scores and cells disagree is refused, because the pairing is "
              "positional");

        std::vector<float> all_hidden(kCells, -INFINITY);
        check(tq::reader::softmax_scaled(all_hidden, kScale) ==
                  tq::reader::physical_refusal::all_hidden,
              "refusal: every cell hidden has no mass to normalise, so it is named rather than "
              "returned as 0/0");

        std::vector<tq::reader::physical_span> empty(0);
        check(tq::reader::scatter(empty.data(), 0, kCells, -INFINITY, w) ==
                  tq::reader::physical_refusal::no_spans,
              "refusal: no spans is a refusal, not a normalised zero vector");

        check(tq::reader::scatter(spans.data(), spans.size(), 0, -INFINITY, w) ==
                  tq::reader::physical_refusal::empty_cache,
              "refusal: an empty cache is a refusal");

        // `attend_physical` clears its output on every refusal path, so a caller cannot compute
        // with a half-built result.
        o.assign(4, 123.0f);
        check(tq::reader::attend_physical(duplicate.data(), duplicate.size(), kCells, kScale,
                                          v.data(), kHeadDim, kKvHeads, kHeads, w, o) ==
                  tq::reader::physical_refusal::duplicate_cell,
              "refusal: the combined call reports the same cause");
        check(o.empty(), "refusal: and leaves no output behind, so a caller cannot use half of one");
        for (const auto r : {tq::reader::physical_refusal::duplicate_cell,
                             tq::reader::physical_refusal::all_hidden,
                             tq::reader::physical_refusal::not_normalisable}) {
            check(tq::reader::describe(r) != nullptr && tq::reader::describe(r)[0] != '\0',
                  "every refusal explains itself");
        }
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "physical core: scatter to the full cache length, one global softmax, native "
                 "reduction order -- matching exact arithmetic and ggml's kernels\n");
    return 0;
}
