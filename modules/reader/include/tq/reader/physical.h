// tq/reader/physical.h — the reader's attention over the PHYSICAL cache, as arithmetic.
//
// Why this exists next to `attention.h`
// -------------------------------------
// `attention.h` computes attention over the union of a view's spans: gather the named cells, one
// softmax over exactly those, reduce. Mathematically that is the same integral as attention over
// the whole cache with a -infinity mask, because `exp(-inf) == 0`. Numerically it is not.
//
// The fork this project re-implements says so in its own words (llama-graph.cpp, the reader
// attention builder):
//
//     "Preserve the native physical reduction order and padding. A tiny change in a compact sum
//      can cross a later quantization boundary."
//
// and then does exactly that: it scatters each span's scores into a cache-length vector filled
// with -INFINITY, normalises ONCE over that full length, and multiplies by V in the native order.
// So "compact union sum" is not an equal substitute for it — by the fork authors' design, not by
// an accident of implementation. A re-implementation that only computes the compact form cannot
// be called equivalent, and this file is the part that can.
//
// The station split
// -----------------
// These are rules, not a graph. Nothing here knows what ggml is: the station holds the three
// operations that decide the numbers, and a caller that wants tensors builds them from the same
// three. That is what makes them testable without a context — and testable against one afterwards,
// which is the only way "matches ggml" means anything.
//
//   scatter   full length, -INFINITY for cells the reader may not see
//   softmax   ONE normalisation over that length, with the kq scale applied inside (as ggml does)
//   reduce    the native order: sum over cells 0..n-1, including the zeros
//
// What it refuses, and why refusal is the only safe answer
// --------------------------------------------------------
// * a cell outside [0, n_kv) — the score would land in another sequence's slot;
// * the same cell named twice — two spans writing one position means one of them is silently
//   discarded, and which one depends on iteration order;
// * a span whose score count differs from its cell count — the pairing is positional, so a
//   mismatch writes scores against the wrong cells;
// * every cell hidden — the softmax denominator is zero and the result is 0/0. The fork's callers
//   guarantee a visible cell ("every view includes its current key"), so this is a caller bug, and
//   returning a NaN would be the worst possible report of it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tq::reader {

// One span as the physical form needs it: the scores it computed, and where they go.
//
// `cells[j]` is the physical cache index of `scores[j]`. The two vectors are parallel, and equal
// length is a precondition the scatter checks rather than assumes.
struct physical_span {
    std::vector<std::int32_t> cells;
    std::vector<float>        scores;
};

// Why a step refused. Named, because the caller's next move differs per cause.
enum class physical_refusal {
    ok,
    empty_cache,        // n_kv == 0: there is no physical length to scatter into
    no_spans,           // nothing was computed, so there is nothing to normalise
    length_mismatch,    // a span's scores and cells disagree in count
    cell_out_of_range,  // a cell index outside [0, n_kv)
    duplicate_cell,     // two spans (or one span twice) name the same physical cell
    all_hidden,         // every cell is -infinity: the softmax has no mass to normalise
    not_normalisable,   // a finite weight came out as nan/inf, so the sum was degenerate
    null_buffer,        // a required pointer (v, weights) was null
    degenerate_shape,   // n_head, n_head_kv, head_dim or n_kv was zero
    indivisible_heads,  // n_head is not a multiple of n_head_kv, so head/group would index past the
                        // cache's head axis
    size_overflow,      // n_head * head_dim does not fit in size_t
};

const char * describe(physical_refusal refusal);

// Scatters span scores into a cache-length weight vector.
//
// `weights` is resized to `n_kv` and filled with `hidden_fill` (the fork uses -INFINITY), then each
// span's scores are written at that span's cells. A caller that wants attention over the whole
// cache passes -INFINITY; a caller that wants to see the raw scatter can pass any fill.
//
// The fill is a parameter rather than a constant because the -INFINITY is the SEMANTIC part of the
// contract (hidden cells must receive exactly zero softmax weight) and a test should be able to
// show what happens without it.
physical_refusal scatter(const physical_span * spans, std::size_t n_spans, std::size_t n_kv,
                         float hidden_fill, std::vector<float> & weights);

// One normalisation over the whole physical length, with the scale applied to the logits first.
//
// Matches ggml_soft_max_ext(x, mask=nullptr, scale, max_bias=0): for each row, m = max(x[i]*scale),
// then x[i] = exp(x[i]*scale - m) / sum_j exp(x[j]*scale - m). The order matters — scaling after
// the max subtraction gives different rounding — and the scale is part of this operation rather
// than a separate pass because that is where ggml puts it.
//
// A cell at -INFINITY contributes exp(-inf) = 0 exactly, which is why the full-length form and the
// compact form agree in real arithmetic. What differs between them is the sum's composition, and
// that is the whole point of the physical form.
physical_refusal softmax_scaled(std::vector<float> & logits, float scale);

// The value reduction: the term SEQUENCE is the contract; bit-equality with ggml is not.
//
//   out[head][dim] = sum over cell 0..n_kv-1 of weights[cell] * v[dim][head/group][cell]
//
// `v` is the physical cache in the layout the emitter documents as dim-major:
// v[d + head_dim*(kv_head + n_head_kv*cell)]. The sum runs over the FULL length, zeros included, in
// increasing cell order -- that is the term sequence the fork preserves, and skipping the zeros
// would be the compact form again. That much is portable: it is what this function does, on every
// build.
//
// WHAT IS NOT PORTABLE, and this paragraph used to claim otherwise: agreement to the BIT with
// `ggml_vec_dot_f32`. That function (ggml-cpu/vec.cpp) is
//
//     float sumf = 0.0f;
//     ... GGML_F32_VEC_FMA into GGML_F32_ARR vector accumulators, np = n & ~(GGML_F32_STEP-1)
//
// and `GGML_F32_STEP` is chosen per ISA in ggml-cpu/simd-mappings.h: 16 on some targets, 32 on
// others, 64 for AVX2, and the non-SIMD branch keeps the same blocking over scalar lanes. So the
// size at which a build stops agreeing with a plain ascending float sum is a property of that
// build -- of its ISA, and of whether the compiler contracted the tail loop into FMA.
//
// MEASURED ON ONE BUILD (this host: AVX2, so STEP == 64), reported because a number is more useful
// than a hedge, and explicitly NOT a contract:
//
//   n_kv     float truncation of this loop vs ggml mul_mat
//     4-32   0 of 32 outputs differ          (ggml's blocking does not run below STEP)
//     64     6 of 32 differ                  (blocking begins)
//    128     7 of 32 differ
//    512     8 of 32 differ, max|diff| 1.9e-06
//   2048     7 of 32 differ, max|diff| 1.4e-05
//
// On a STEP == 16 build the first row would stop at 16, and a scalar-only build could agree at
// every size. Nothing in this file may depend on which of those you have.
//
// Two consequences, both previously got wrong here:
//
//   * an earlier version accumulated in DOUBLE. That is a different arithmetic at EVERY size, not
//     a wider version of the same one: at 12 cells it differed from ggml in 4 of 32 outputs while
//     the float loop differs in none. The wider accumulator was the deviation, not the safeguard.
//
//   * at a real cache size (2048 cells, measured from a model earlier in this lane) NEITHER variant
//     equals ggml. So this function reproduces ggml's VALUE within float rounding at any size, and
//     its bits only on builds whose blocking happens to match -- which is a constraint on any
//     product path that would compare against a ggml run, and it belongs here rather than in a
//     changelog.
//
// `n_head` must be a multiple of `n_head_kv`; the group size is integer division, so 7 heads over
// 2 kv heads would compute kv_head == 2 for head 6 and read the next cell's data. That is refused,
// as is any shape whose element index would overflow size_t (see the implementation): an index that
// wrapped would read outside `v` and produce a plausible number from another cell.
physical_refusal reduce_values(const float * v, std::size_t head_dim, std::size_t n_head_kv,
                              std::size_t n_head, std::size_t n_kv, const float * weights,
                              std::vector<float> & out);

// The whole operation, in one call: scatter, normalise, reduce.
//
// Exists so a caller cannot accidentally run the steps out of order — and so the tests that compare
// against a graph and against a double reference exercise the same sequence the product would.
//
// ON REFUSAL, `out` IS CLEARED AND `weights` IS WHATEVER THE FAILING STEP LEFT IT AS. The second
// half of that is not an oversight; it is the one place in this file where a partial result is
// visible, and a caller must know which:
//
//   scatter fails      `weights` is left resized to n_kv with the fill in it (or partially
//                      written, if the failure was a mid-span duplicate). It is not a normalised
//                      vector and must not be consumed as one.
//   softmax fails      `weights` is bit-for-bit what scatter wrote -- `softmax_scaled` is atomic
//                      (see its declaration), so nothing was rescaled.
//   reduce fails       `weights` is a VALID normalised vector; `out` is empty.
//
// `out` is the only output that matters, and it is empty on every refusal, so a caller that checks
// the return code before reading `out` cannot be misled. The note exists for a caller that inspects
// `weights` for diagnostics: at the third failure it is meaningful, at the first two it is not.
//
// NOT IDEMPOTENT, and the trap is worth naming: a successful call overwrites every hidden cell with
// 0.0f. A second call on the same vector therefore sees those cells as VISIBLE — a finite 0.0 —
// and gives them weight. Normalising twice does not converge to the same answer; it resurrects the
// cells the scatter hid. Compute the weights once per scatter, and re-scatter to renormalise.
physical_refusal attend_physical(const physical_span * spans, std::size_t n_spans,
                                 std::size_t n_kv, float kq_scale, const float * v,
                                 std::size_t head_dim, std::size_t n_head_kv, std::size_t n_head,
                                 std::vector<float> & weights, std::vector<float> & out);

} // namespace tq::reader
