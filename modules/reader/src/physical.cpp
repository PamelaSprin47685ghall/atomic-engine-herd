#include "tq/reader/physical.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace tq::reader {
namespace {

// A cell is "hidden" when its weight is still the fill we put there. -INFINITY is the only fill
// that makes the check exact under a float comparison, which is why the fork uses it too.
bool is_hidden(float value) { return std::isinf(value) && value < 0.0f; }

} // namespace

const char * describe(physical_refusal refusal) {
    switch (refusal) {
        case physical_refusal::ok:               return "ok";
        case physical_refusal::empty_cache:      return "the cache reports no cells, so there is no physical length";
        case physical_refusal::no_spans:         return "no span was supplied, so there is nothing to normalise";
        case physical_refusal::length_mismatch:  return "a span's scores and cells disagree in count; the pairing is "
                                                         "positional, so a mismatch would write scores against the wrong cells";
        case physical_refusal::cell_out_of_range:return "a cell index lies outside the cache; writing it would put a "
                                                         "score in another sequence's slot";
        case physical_refusal::duplicate_cell:   return "two spans name the same physical cell; one of their scores "
                                                         "would be discarded according to iteration order";
        case physical_refusal::all_hidden:       return "every cell is hidden, so the softmax has no mass and the "
                                                         "result would be 0/0";
        case physical_refusal::not_normalisable: return "the normalisation produced a non-finite weight, so the sum "
                                                         "was degenerate";
        case physical_refusal::null_buffer:      return "a required buffer was null, so nothing was computed";
        case physical_refusal::degenerate_shape: return "a shape argument was zero (n_head, n_head_kv, head_dim "
                                                         "or n_kv)";
        case physical_refusal::indivisible_heads:return "n_head is not a multiple of n_head_kv: the group size is "
                                                         "integer division, so head/group would index past the "
                                                         "cache's head axis and read the next cell's data";
        case physical_refusal::size_overflow:    return "n_head * head_dim overflows size_t, so the output size "
                                                         "cannot be computed";
    }
    return "unknown refusal";
}

physical_refusal scatter(const physical_span * spans, std::size_t n_spans, std::size_t n_kv,
                         float hidden_fill, std::vector<float> & weights) {
    weights.clear();
    if (n_kv == 0) {
        return physical_refusal::empty_cache;
    }
    if (n_spans == 0 || !spans) {
        return physical_refusal::no_spans;
    }

    // Filled first, written second. The fill is what makes a cell the reader may not see weigh
    // exactly nothing; a caller that passed 0.0f instead would give hidden cells a softmax share.
    weights.assign(n_kv, hidden_fill);

    // A duplicate check needs to know which cells were written, and `weights` cannot answer that:
    // a legitimate score can equal the fill (a logit of -infinity is improbable but a NaN-free
    // -inf is representable), and then "was it written?" becomes undecidable. So the set of written
    // cells is tracked separately — a byte per cell, which for a 2048-cell cache is 2 KiB and for
    // the whole point of the operation is not worth optimising away.
    std::vector<unsigned char> written(n_kv, 0);

    for (std::size_t s = 0; s < n_spans; ++s) {
        const physical_span & span = spans[s];
        if (span.cells.size() != span.scores.size()) {
            return physical_refusal::length_mismatch;
        }
        for (std::size_t j = 0; j < span.cells.size(); ++j) {
            const std::int32_t cell = span.cells[j];
            if (cell < 0 || std::size_t(cell) >= n_kv) {
                return physical_refusal::cell_out_of_range;
            }
            if (written[std::size_t(cell)] != 0) {
                return physical_refusal::duplicate_cell;
            }
            written[std::size_t(cell)] = 1;
            weights[std::size_t(cell)] = span.scores[j];
        }
    }
    return physical_refusal::ok;
}

physical_refusal softmax_scaled(std::vector<float> & logits, float scale) {
    if (logits.empty()) {
        return physical_refusal::no_spans;
    }

    // ATOMIC: on any refusal, `logits` is exactly as it arrived.
    //
    // The previous version transformed in place as it went, so `not_normalisable` left the vector
    // holding scaled exponents that summed to zero or to a non-finite total. `attend_physical`
    // clears `out` on a refusal, so nothing downstream consumed that state -- but `softmax_scaled`
    // is part of the public interface (physical.h), and a caller that used it directly, got a
    // refusal, and then retried with a corrected scale would have been normalising already-scaled
    // values. A refusal that leaves the input in a state only a full recompute can repair is not a
    // refusal; it is a partial success with a status code.
    //
    // Three passes, no scratch allocation:
    //   1. scale and find the maximum, writing nothing
    //   2. exponentiate and total, writing nothing
    //   3. write the normalised weights
    // `exp` is therefore evaluated twice per visible cell. For a 2048-cell cache that is 4096
    // calls per layer; the alternative -- a scratch vector the size of the cache -- is the same
    // order of work plus an allocation on a path whose whole point is to avoid surprises.
    //
    // ggml's order is preserved exactly: scale first, then the max subtraction, then exp, then one
    // global denominator. Scaling after the max would round differently, which is the property the
    // fork's reader attention is built around.
    float maximum = -INFINITY;
    std::size_t visible = 0;
    for (const float value : logits) {
        if (is_hidden(value)) {
            continue;
        }
        const float scaled = value * scale;
        if (!std::isfinite(scaled)) {
            // +inf or NaN after scaling is not a weight the rest of the computation can carry: an
            // inf max would make every other cell exp(-inf) = 0 and silently concentrate all mass on
            // one key, which is a wrong answer that looks like a sharp one. -inf is treated as
            // hidden above (a scale that underflows a finite logit is a legitimate way to hide it).
            if (scaled != -INFINITY) {
                return physical_refusal::not_normalisable;
            }
            continue;
        }
        ++visible;
        if (scaled > maximum) {
            maximum = scaled;
        }
    }
    if (visible == 0) {
        return physical_refusal::all_hidden;
    }

    double total = 0.0;
    for (const float value : logits) {
        if (is_hidden(value)) {
            continue;
        }
        const float scaled = value * scale;
        if (!std::isfinite(scaled)) {
            continue;   // -infinity: contributes exp(-inf) = 0, already counted as not visible
        }
        total += std::exp(double(scaled) - double(maximum));
    }
    // THIS GUARD CANNOT FIRE, and saying so is more useful than pretending it protects something.
    //
    // The previous comment claimed `total > 0` "fails when every exponent underflowed to zero -- a
    // real possibility with a large negative scale". That is false, and the reason is one line of
    // arithmetic: `maximum` is taken over exactly the same finite scaled values this loop
    // exponentiates, so the cell that produced it contributes `exp(x - x) == exp(0) == 1` exactly,
    // in double, with no rounding available to lose it. Therefore
    //
    //     1 <= total <= visible
    //
    // for every input that reaches this point -- never 0, never infinite, never NaN. A scale of
    // -1e30 does not make the exponent underflow; it makes every scaled logit -1e30, their maximum
    // -1e30, and every exponent exp(0) = 1. Measured on eight adversarial inputs (huge negative
    // scale, FLT_MAX, FLT_MAX..-FLT_MAX, denormals, tiny/2, 12 terms at +-1e38, a single cell, one
    // hidden): total was 1, 2, or 3 in every case, and the guard fired zero times.
    //
    // So there is no path to `not_normalisable` here. The only refusals this function can produce
    // are `no_spans` (empty input), `all_hidden` (nothing visible), and `not_normalisable` from
    // PASS 1 (a NaN or +infinity scaled logit). All three precede every write, which is what makes
    // the atomicity property structural rather than tested-into-place.
    //
    // The check stays, and it is not cargo: it is the assertion that the invariant above holds. If
    // a future edit changes how `maximum` is chosen, or exponentiates a different set than it
    // ranged over, `total` could leave [1, visible] and this line would be the thing that reports
    // it instead of a silent division. What it must NOT do is claim a reachable failure mode it
    // does not have -- that is how an unreachable branch becomes a reason to believe the code is
    // safe somewhere it is not.
    if (!(total > 0.0) || !std::isfinite(total)) {
        return physical_refusal::not_normalisable;
    }

    for (float & value : logits) {
        if (is_hidden(value)) {
            // Exactly zero, written explicitly: exp(-inf) is zero, and a hidden cell's weight must
            // be zero rather than a value downstream has to reason about -inf to obtain.
            value = 0.0f;
            continue;
        }
        const float scaled = value * scale;
        if (!std::isfinite(scaled)) {
            value = 0.0f;   // -infinity after scaling: also exactly zero mass
            continue;
        }
        value = float(std::exp(double(scaled) - double(maximum)) / total);
    }
    return physical_refusal::ok;
}

physical_refusal reduce_values(const float * v, std::size_t head_dim, std::size_t n_head_kv,
                              std::size_t n_head, std::size_t n_kv, const float * weights,
                              std::vector<float> & out) {
    out.clear();

    // VALIDATED, not assumed. The previous version returned `void` and answered a bad call by
    // filling `out` with zeros -- so `attend_physical` reported `ok` with an all-zero answer, which
    // is the worst possible failure: a caller that checks only the refusal code cannot tell a
    // legitimate zero from an unperformed reduction.
    //
    // The head grouping is the sharp one. `n_head / n_head_kv` is integer division, so 7 query
    // heads over 2 kv heads gives group == 3, and `head / group` reaches kv_head == 2 for head 6 --
    // one past the end of the cache's head axis. The read lands in the next cell's data, is finite,
    // and is wrong. Divisibility is therefore a precondition, not a detail.
    if (!v || !weights) {
        return physical_refusal::null_buffer;
    }
    if (n_head_kv == 0 || n_head == 0 || head_dim == 0 || n_kv == 0) {
        return physical_refusal::degenerate_shape;
    }
    if (n_head % n_head_kv != 0) {
        return physical_refusal::indivisible_heads;
    }

    // `out.assign(n_head * head_dim, ...)` is a size computation before it is an allocation, and a
    // product that wraps would allocate a small buffer and then index past it. The check costs one
    // division.
    if (head_dim > (SIZE_MAX / n_head)) {
        return physical_refusal::size_overflow;
    }
    // THE ELEMENT INDEX OVERFLOWS SEPARATELY, and it is the more dangerous of the two.
    //
    // The largest index this loop forms is
    //     dim + head_dim * (kv_head + n_head_kv * (n_kv - 1))
    //   = head_dim * n_head_kv * n_kv - 1
    // so `head_dim * n_head_kv * n_kv` must not wrap. If it does, the index lands back inside the
    // buffer, the read succeeds, and the answer is a sum over another cell's values: finite,
    // plausible, and inside bounds -- so neither a sanitizer nor a bounds check downstream would
    // catch it. `out` can be small and entirely valid while `v` is read out of range, which is why
    // the output-size check above does not cover this.
    //
    // Two divisions rather than one product-then-compare, so no intermediate can wrap before it is
    // tested.
    if (n_kv > SIZE_MAX / n_head_kv) {
        return physical_refusal::size_overflow;
    }
    if (head_dim > SIZE_MAX / (n_head_kv * n_kv)) {
        return physical_refusal::size_overflow;
    }
    const std::size_t group = n_head / n_head_kv;
    out.assign(n_head * head_dim, 0.0f);

    for (std::size_t head = 0; head < n_head; ++head) {
        const std::size_t kv_head = head / group;
        for (std::size_t dim = 0; dim < head_dim; ++dim) {
            // FLOAT, NOT DOUBLE, and the measurement is why.
            //
            // The previous version accumulated in `double` and carried a comment claiming this was
            // "the native physical reduction order the fork preserves". Both halves of that were
            // wrong, and the numbers say so. `ggml_vec_dot_f32` (ggml-cpu/vec.cpp) is
            //
            //     float sumf = 0.0f;
            //     ... GGML_F32_VEC_FMA into 8 accumulators, np = n & ~(step-1), step = 8*epr
            //
            // so ggml's reduction is a FLOAT sum in a blocked SIMD order whose block length
            // depends on the build's ISA. Accumulating in double is a different arithmetic at every
            // size -- not a different order, a different type -- and it was the single largest
            // source of disagreement with ggml's own mul_mat.
            //
            // Measured against `ggml_mul_mat` on this host (AVX2, so step == 64), for a single kv
            // head with head_dim 8, weights non-negative, 32..2048 cells:
            //
            //   n_kv     double/ascending          float/ascending
            //     4      6 bits differ  6.0e-08     0 bits   0.0e+00   <- exact
            //     8      4 bits differ  6.0e-08     0 bits   0.0e+00   <- exact
            //    12      4 bits differ  6.0e-08     0 bits   0.0e+00   <- exact
            //    16      5 bits differ  6.0e-08     0 bits   0.0e+00   <- exact
            //    24      7 bits differ  1.2e-07     0 bits   0.0e+00   <- exact
            //    32      5 bits differ  1.5e-07     0 bits   0.0e+00   <- exact
            //    64      6 bits differ  1.8e-07     6 bits   3.0e-07
            //   128      8 bits differ  4.8e-07     7 bits   7.2e-07
            //   512      6 bits differ  9.5e-07     8 bits   1.9e-06
            //  2048      6 bits differ  2.4e-06     7 bits   1.4e-05
            //
            // Below 64 cells ggml takes its scalar path, which IS plain ascending float, and this
            // loop is then bit-identical. At 64 and above ggml's blocked SIMD sum takes over and
            // neither variant matches: the order differs, double is closer in magnitude but still
            // not equal, and no portable kernel can reproduce an order chosen by the ISA.
            //
            // So the honest position is bounded, not absolute:
            //
            //   n_kv < 64 (this host)  bit-identical to ggml's mul_mat, verified.
            //   n_kv >= 64             the sums differ in the low bits. This kernel computes the
            //                          mathematical value in the cache's order; ggml computes a
            //                          blocked float sum. Neither is "the" answer, and a product
            //                          path built on this kernel does NOT reproduce a ggml run at
            //                          real cache sizes (2048 measured earlier from a model).
            //
            // That last line is a constraint on the product path, not a footnote: it is why this
            // core is a reference and not yet a substitute. Recording it here is cheaper than
            // rediscovering it as a logits difference nobody can attribute.
            float acc = 0.0f;
            for (std::size_t cell = 0; cell < n_kv; ++cell) {
                acc += weights[cell] * v[dim + head_dim * (kv_head + n_head_kv * cell)];
            }
            out[head * head_dim + dim] = acc;
        }
    }
    return physical_refusal::ok;
}

physical_refusal attend_physical(const physical_span * spans, std::size_t n_spans,
                                 std::size_t n_kv, float kq_scale, const float * v,
                                 std::size_t head_dim, std::size_t n_head_kv, std::size_t n_head,
                                 std::vector<float> & weights, std::vector<float> & out) {
    const physical_refusal scattered = scatter(spans, n_spans, n_kv, -INFINITY, weights);
    if (scattered != physical_refusal::ok) {
        out.clear();
        return scattered;
    }
    const physical_refusal normalised = softmax_scaled(weights, kq_scale);
    if (normalised != physical_refusal::ok) {
        out.clear();
        return normalised;
    }
    // THE REDUCTION'S ANSWER IS CARRIED, not dropped. It used to return void and answer a bad call
    // by filling `out` with zeros, so this function reported `ok` with an all-zero result -- a
    // caller checking only the refusal code could not tell a legitimate zero from a reduction that
    // never ran.
    const physical_refusal reduced =
        reduce_values(v, head_dim, n_head_kv, n_head, n_kv, weights.data(), out);
    if (reduced != physical_refusal::ok) {
        out.clear();
        return reduced;
    }
    return physical_refusal::ok;
}

} // namespace tq::reader
