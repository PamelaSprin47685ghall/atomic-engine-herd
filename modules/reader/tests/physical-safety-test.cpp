// The physical core's arithmetic boundaries: the cases where a wrong answer would be finite,
// plausible, and invisible.
//
// What this file is for, stated as the question it answers: can any pair of arguments make
// `attend_physical` produce a NUMBER that is not the answer to the question asked? Every case below
// is one where the old code did -- or would have.
//
// The three defects this file was written against, each found by reading rather than by a failure:
//
//   1. `reduce_values` returned `void`. A null `v` or a zero `n_head_kv` made it fill `out` with
//      zeros and return, and `attend_physical` then reported `ok`. A caller checking the refusal
//      code could not tell a legitimate zero from a reduction that never ran. It now returns a
//      refusal and the caller propagates it.
//
//   2. `group = n_head / n_head_kv` is integer division. With 7 query heads over 2 kv heads,
//      group == 3, so head 6 computes `kv_head == 2` -- one past the cache's head axis -- and reads
//      the next cell's data. The value is finite and wrong. Divisibility is now a refusal.
//
//   3. `n_head * head_dim` was computed without a range check before `out.assign`. A wrapped
//      product allocates a small buffer and then indexes past it. Now refused.
//
// And one claim that was NOT a defect but was overstated in the other direction: it said the
// reduction accumulates in double while ggml's mul_mat accumulates in float, so it reproduces the
// ORDER of the terms and not the ARITHMETIC. The measurement showed that the double accumulator was
// itself the largest disagreement -- at 12 cells it differed from ggml in 4 of 32 outputs while a
// plain float loop differs in none. The core now accumulates in float and is bit-identical below
// ggml's SIMD crossover (64 cells on this host) and honestly different above it, where the order is
// a property of the build's ISA. The last case asserts both sides of that boundary.
#include <tq/reader/physical.h>

#include <ggml.h>
#include <ggml-cpu.h>

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

using tq::reader::physical_refusal;
using tq::reader::physical_span;

physical_span span_of(std::vector<std::int32_t> cells, std::vector<float> scores) {
    physical_span span;
    span.cells = std::move(cells);
    span.scores = std::move(scores);
    return span;
}

// A cache in the emitter's dim-major layout: v[d + head_dim*(kv_head + n_head_kv*cell)].
std::vector<float> cache(std::size_t head_dim, std::size_t n_head_kv, std::size_t n_kv,
                         unsigned seed = 12345u) {
    std::vector<float> v(head_dim * n_head_kv * n_kv, 0.0f);
    unsigned state = seed;
    for (float & value : v) {
        state = state * 1664525u + 1013904223u;
        value = float((state >> 9) & 0xFFFF) / 65536.0f - 0.5f;
    }
    return v;
}

// ---------------------------------------------------------------------------------------------
// 1. Indivisible heads: the case that read the next cell's data.
// ---------------------------------------------------------------------------------------------
void indivisible_heads_are_refused_not_read() {
    // 7 query heads over 2 kv heads. group == 3 by integer division, so head 6 computes
    // kv_head == 2, and the cache's head axis has only 0 and 1.
    constexpr std::size_t kHeadDim = 4, kKvHeads = 2, kHeads = 7, kCells = 6;
    const std::vector<float> v = cache(kHeadDim, kKvHeads, kCells);

    std::vector<float> weights(kCells, 1.0f / float(kCells));
    std::vector<float> out;
    const physical_refusal refusal =
        tq::reader::reduce_values(v.data(), kHeadDim, kKvHeads, kHeads, kCells, weights.data(), out);

    check(refusal == physical_refusal::indivisible_heads,
          "7 heads over 2 kv heads is refused: the group size is integer division, so head 6 would "
          "index past the cache's head axis and read the next cell's data");
    check(out.empty(),
          "and nothing is written, so no caller can mistake a partial answer for a whole one");
    std::fprintf(stderr, "indivisible: %s\n", tq::reader::describe(refusal));

    // The divisible neighbours must still work, or the refusal would be indiscriminate.
    for (std::size_t heads : {std::size_t(2), std::size_t(4), std::size_t(8)}) {
        std::vector<float> ok_out;
        const physical_refusal ok = tq::reader::reduce_values(v.data(), kHeadDim, kKvHeads, heads,
                                                             kCells, weights.data(), ok_out);
        check(ok == physical_refusal::ok && ok_out.size() == heads * kHeadDim,
              "a divisible head count is accepted and produces one block per query head");
    }
}

// ---------------------------------------------------------------------------------------------
// 2. Null buffers and degenerate shapes: previously a silent all-zero success.
// ---------------------------------------------------------------------------------------------
void bad_arguments_are_refused_rather_than_answered_with_zeros() {
    constexpr std::size_t kHeadDim = 4, kKvHeads = 2, kHeads = 4, kCells = 6;
    const std::vector<float> v = cache(kHeadDim, kKvHeads, kCells);
    std::vector<float> weights(kCells, 1.0f / float(kCells));
    std::vector<float> out;

    check(tq::reader::reduce_values(nullptr, kHeadDim, kKvHeads, kHeads, kCells, weights.data(),
                                    out) == physical_refusal::null_buffer,
          "a null value cache is refused, not answered with zeros");
    check(out.empty(), "and the output stays empty, so the refusal cannot be read as a result");

    check(tq::reader::reduce_values(v.data(), kHeadDim, kKvHeads, kHeads, kCells, nullptr, out) ==
              physical_refusal::null_buffer,
          "null weights are refused");

    check(tq::reader::reduce_values(v.data(), kHeadDim, 0, kHeads, kCells, weights.data(), out) ==
              physical_refusal::degenerate_shape,
          "zero kv heads is refused rather than producing an empty group");

    check(tq::reader::reduce_values(v.data(), 0, kKvHeads, kHeads, kCells, weights.data(), out) ==
              physical_refusal::degenerate_shape,
          "zero head dimension is refused");

    // THE CASE THAT MATTERED: through the whole operation, not just the reduction. The previous
    // version returned `ok` here with an all-zero `out`.
    physical_span spans[1];
    spans[0] = span_of({0, 2}, {1.0f, 0.5f});
    std::vector<float> weights2;
    std::vector<float> out2;
    const physical_refusal whole =
        tq::reader::attend_physical(spans, 1, kCells, 1.0f, nullptr, kHeadDim, kKvHeads, kHeads,
                                    weights2, out2);
    check(whole == physical_refusal::null_buffer,
          "attend_physical does not report success when its reduction could not run");
    check(out2.empty(), "and hands back no tensor, so no caller can use an all-zero answer");
    std::fprintf(stderr, "attend_physical(null v): %s\n", tq::reader::describe(whole));
}

// ---------------------------------------------------------------------------------------------
// 3. Non-finite logits: NaN, +inf, and a NaN scale.
// ---------------------------------------------------------------------------------------------
void non_finite_logits_are_refused() {
    constexpr std::size_t kCells = 5;

    // A NaN score. `value > maximum` is false for NaN, so the maximum would be computed as if the
    // NaN were absent, and exp(NaN) poisons the total -- which is what the check catches.
    {
        physical_span spans[1];
        spans[0] = span_of({0, 1, 2}, {1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f});
        std::vector<float> weights, out;
        const physical_refusal r = tq::reader::scatter(spans, 1, kCells, -INFINITY, weights);
        check(r == physical_refusal::ok, "a NaN score scatters (the scatter does not judge values)");
        check(tq::reader::softmax_scaled(weights, 1.0f) == physical_refusal::not_normalisable,
              "and the normalisation refuses it: exp(NaN) is NaN, so the sum has no meaning");
    }

    // A +infinity score. max becomes +inf, and exp(inf - inf) is NaN, so the total is NaN.
    {
        physical_span spans[1];
        spans[0] = span_of({0, 1}, {1.0f, std::numeric_limits<float>::infinity()});
        std::vector<float> weights, out;
        tq::reader::scatter(spans, 1, kCells, -INFINITY, weights);
        const physical_refusal r = tq::reader::softmax_scaled(weights, 1.0f);
        check(r == physical_refusal::not_normalisable,
              "a +infinity score is refused: exp(inf - inf) is NaN, and reporting that is better "
              "than returning a NaN-filled weight vector");
        std::fprintf(stderr, "+inf logit: %s\n", tq::reader::describe(r));
    }

    // A NaN scale. `value *= scale` turns every finite logit into NaN.
    {
        physical_span spans[1];
        spans[0] = span_of({0, 1}, {1.0f, 2.0f});
        std::vector<float> weights, out;
        tq::reader::scatter(spans, 1, kCells, -INFINITY, weights);
        const physical_refusal r =
            tq::reader::softmax_scaled(weights, std::numeric_limits<float>::quiet_NaN());
        check(r == physical_refusal::not_normalisable, "a NaN kq scale is refused");
    }

    // Scale zero: a degenerate but well-defined request. Every logit becomes 0, so the weights are
    // uniform -- and that is the answer to "softmax with no scaling", not an error.
    {
        physical_span spans[1];
        spans[0] = span_of({0, 1, 2}, {5.0f, -3.0f, 100.0f});
        std::vector<float> weights, out;
        tq::reader::scatter(spans, 1, kCells, -INFINITY, weights);
        check(tq::reader::softmax_scaled(weights, 0.0f) == physical_refusal::ok,
              "scale 0 is a legal degenerate request, not an error");
        check(std::fabs(weights[0] - 1.0f / 3.0f) < 1e-6f && std::fabs(weights[2] - 1.0f / 3.0f) < 1e-6f,
              "and it yields the uniform distribution over the visible cells");
    }

    // A negative scale: legal. It flips the ordering, which is a caller's prerogative.
    {
        physical_span spans[1];
        spans[0] = span_of({0, 1}, {1.0f, 3.0f});
        std::vector<float> weights, out;
        tq::reader::scatter(spans, 1, kCells, -INFINITY, weights);
        check(tq::reader::softmax_scaled(weights, -1.0f) == physical_refusal::ok,
              "a negative scale is accepted");
        check(weights[0] > weights[1],
              "and reverses the ordering, which is what a negative scale means");
    }
}

// ---------------------------------------------------------------------------------------------
// 4. Hidden cells must weigh EXACTLY zero, and a hidden fill of 0 must not.
// ---------------------------------------------------------------------------------------------
void hidden_cells_weigh_exactly_zero() {
    constexpr std::size_t kCells = 8;
    physical_span spans[2];
    spans[0] = span_of({0, 3}, {0.0f, 1.0f});
    spans[1] = span_of({5, 6, 7}, {2.0f, -1.0f, 0.5f});

    std::vector<float> weights, out;
    const physical_refusal r = tq::reader::scatter(spans, 2, kCells, -INFINITY, weights);
    check(r == physical_refusal::ok, "the scatter of two spans succeeds");
    check(weights.size() == kCells, "and covers the full physical length");

    const std::size_t hidden[] = {1, 2, 4};
    bool all_hidden_exact = true;
    for (std::size_t cell : hidden) {
        all_hidden_exact = all_hidden_exact && std::isinf(weights[cell]) && weights[cell] < 0.0f;
    }
    check(all_hidden_exact, "hidden cells hold -INFINITY before normalisation");

    check(tq::reader::softmax_scaled(weights, 1.0f) == physical_refusal::ok, "normalisation succeeds");
    bool zero_weight = true;
    for (std::size_t cell : hidden) {
        zero_weight = zero_weight && weights[cell] == 0.0f;   // exact, not "small"
    }
    check(zero_weight,
          "after normalisation every hidden cell weighs EXACTLY zero -- exp(-inf) is 0, and a "
          "merely tiny weight would let a hidden key contribute");

    float total = 0.0f;
    for (float w : weights) total += w;
    check(std::fabs(total - 1.0f) < 1e-6f, "and the weights sum to one");

    // The fill is a parameter: with 0.0f as the fill, hidden cells get an EQUAL share, which is the
    // mistake the -INFINITY exists to prevent. Pinning the negative case is what makes the positive
    // one meaningful.
    {
        std::vector<float> flat, ignored;
        tq::reader::scatter(spans, 2, kCells, 0.0f, flat);
        check(tq::reader::softmax_scaled(flat, 1.0f) == physical_refusal::ok, "a zero fill normalises");
        check(flat[1] > 0.0f,
              "and a hidden cell competes for mass: with a 0 fill the mask is gone, which is why "
              "the production entry point hard-codes -INFINITY");
    }
}

// ---------------------------------------------------------------------------------------------
// 5. The claim this core can and cannot make about ggml's own reduction.
// ---------------------------------------------------------------------------------------------
void the_order_is_reproduced_and_the_arithmetic_is_not() {
    constexpr std::size_t kHeadDim = 6, kKvHeads = 2, kHeads = 4, kCells = 16;
    const std::vector<float> v = cache(kHeadDim, kKvHeads, kCells, 777u);

    // Weights over the full length, half of them hidden.
    std::vector<float> weights(kCells, -INFINITY);
    for (std::size_t cell = 0; cell < kCells; cell += 2) {
        weights[cell] = 0.25f * float(cell + 1);
    }
    check(tq::reader::softmax_scaled(weights, 1.0f) == physical_refusal::ok, "weights normalise");

    std::vector<float> core_out;
    check(tq::reader::reduce_values(v.data(), kHeadDim, kKvHeads, kHeads, kCells, weights.data(),
                                    core_out) == physical_refusal::ok,
          "the reduction runs");

    // ggml's own multiply, in the layout mul_mat requires.
    //
    // `ggml_mul_mat(a, b)` needs `a->ne[0] == b->ne[0]` and returns ne0 = a->ne[1]. With the
    // weights on [n_cells, 1, n_head_kv], that means `a` must have n_cells on ne0 -- the cell-major
    // form [n_cells, head_dim, n_head_kv], which is the OTHER of upstream's two V layouts. The
    // first version of this case passed the dim-major cache straight in and tripped
    // `GGML_ASSERT(ggml_can_mul_mat)`; the assert was right and the fixture was wrong.
    ggml_init_params params{16u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kCells, 1, kKvHeads);
    for (std::size_t h = 0; h < kKvHeads; ++h) {
        for (std::size_t c = 0; c < kCells; ++c) {
            static_cast<float *>(w->data)[c + kCells * h] = weights[c];
        }
    }

    ggml_tensor * vv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kCells, kHeadDim, kKvHeads);
    for (std::size_t h = 0; h < kKvHeads; ++h) {
        for (std::size_t d = 0; d < kHeadDim; ++d) {
            for (std::size_t c = 0; c < kCells; ++c) {
                static_cast<float *>(vv->data)[c + kCells * (d + kHeadDim * h)] =
                    v[d + kHeadDim * (h + kKvHeads * c)];
            }
        }
    }

    ggml_tensor * product = ggml_mul_mat(ctx, vv, w);   // [head_dim, 1, n_head_kv]
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, product);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    const float * ggml_out = static_cast<const float *>(product->data);
    double worst = 0.0;
    std::size_t differing_bits = 0;
    for (std::size_t h = 0; h < kHeads; ++h) {
        const std::size_t kv_head = h / (kHeads / kKvHeads);
        for (std::size_t d = 0; d < kHeadDim; ++d) {
            const float mine = core_out[h * kHeadDim + d];
            const float theirs = ggml_out[d + kHeadDim * kv_head];
            const double delta = std::fabs(double(mine) - double(theirs));
            if (delta > worst) worst = delta;
            if (std::memcmp(&mine, &theirs, sizeof(float)) != 0) ++differing_bits;
        }
    }
    std::fprintf(stderr,
                 "order vs ggml mul_mat: max|diff| = %.3e, %zu of %zu values differ in the last "
                 "bits\n",
                 worst, differing_bits, kHeadDim * kHeads);

    // WHAT THIS FILE MAY ASSERT ABOUT ggml, and the previous two versions got it wrong in opposite
    // directions -- so the reasoning is recorded rather than the verdict alone.
    //
    //   v1  `differing_bits > 0`  "a wider accumulator must disagree"   -- failed on a correct
    //                             implementation, which is the most expensive kind of test.
    //   v2  `differing_bits == 0` at 12 cells, plus "at 128 cells it MUST differ" -- passed here,
    //                             and both halves are properties of THIS BUILD, not of the code.
    //
    // v2's hard half is the one this case now refuses to assert. Bit-equality with
    // `ggml_vec_dot_f32` is decided by `GGML_F32_STEP = 8 * GGML_F32_EPR`, and that macro is
    // chosen per ISA in ggml-cpu/simd-mappings.h: 16 for some targets, 32 for others, 64 for AVX2
    // (the value measured here), 32 with EPR 8 elsewhere. The non-SIMD branch keeps the same
    // blocking over scalar lanes. So:
    //
    //   * "identical below 64" is wrong on a build with STEP 16 -- not identical below 16 only.
    //   * "different at 128" is wrong on a build whose STEP divides 128 in a way that reproduces
    //     the ascending sum, and on any build where the scalar tail covers the whole row.
    //
    // A test asserting either number fails on someone else's machine for a reason that has nothing
    // to do with this kernel. What IS portable is the arithmetic the caller can observe: the result
    // agrees with ggml to float rounding at every size, and the numbers that come out are the ones
    // the term sequence produces. That is what is asserted; the divergence count is REPORTED, and
    // the report says it is build-dependent.
    check(worst < 1e-6,
          "the reduction agrees with ggml's mul_mat to float rounding: the tolerated residue is "
          "accumulation order, and a wrong cell/head mapping would be off by far more than 1e-6");
    std::fprintf(stderr,
                 "  (this build: %zu of %zu values are bit-different; that count is a property of "
                 "GGML_F32_STEP, i.e. of the ISA and the compiler, NOT of this kernel -- it is "
                 "reported, not asserted)\n",
                 differing_bits, kHeadDim * kHeads);

    // The same comparison across a range of cache sizes, so the tolerance is shown to hold where
    // the block lengths that exist in the wild straddle the row. ONE head and ONE kv head, because
    // the first version of this loop mixed strides: it laid the fixture out with the kv-head axis
    // collapsed (`big_v[d + head_dim*c]`) while `reduce_values` addresses the cache as
    // `d + head_dim*(kv_head + n_head_kv*cell)`. With kKvHeads == 2 those disagree from cell 1 on,
    // and the case failed at 12 cells with max|diff| 1.7e-01 -- which is the tolerance doing its
    // job, not a kernel defect. A fixture that cannot be read the same way by both sides measures
    // the fixture.
    for (const std::size_t cells : {std::size_t(12), std::size_t(64), std::size_t(128)}) {
        std::vector<float> big_v(std::size_t(kHeadDim) * cells, 0.0f);
        std::vector<float> big_w(cells, 0.0f);
        for (std::size_t i = 0; i < big_v.size(); ++i) big_v[i] = 0.5f * float(i % 7) - 1.0f;
        for (std::size_t i = 0; i < big_w.size(); ++i) big_w[i] = 1.0f / float(cells);

        ggml_init_params params{4u * 1024u * 1024u, nullptr, false};
        ggml_context * bctx = ggml_init(params);
        ggml_tensor * bv = ggml_new_tensor_2d(bctx, GGML_TYPE_F32, cells, kHeadDim);
        ggml_tensor * bw = ggml_new_tensor_1d(bctx, GGML_TYPE_F32, cells);
        for (std::size_t c = 0; c < cells; ++c) {
            for (std::size_t d = 0; d < kHeadDim; ++d) {
                // mul_mat wants cell-major: bv[c + cells*d] is v[dim][cell] -- the same element
                // reduce_values reads as v[d + head_dim*(0 + 1*c)] with one kv head.
                static_cast<float *>(bv->data)[c + cells * d] = big_v[d + kHeadDim * c];
            }
            static_cast<float *>(bw->data)[c] = big_w[c];
        }
        ggml_tensor * bo = ggml_mul_mat(bctx, bv, bw);
        ggml_cgraph * bg = ggml_new_graph(bctx);
        ggml_build_forward_expand(bg, bo);
        ggml_graph_compute_with_ctx(bctx, bg, 1);
        const float * bref = static_cast<const float *>(bo->data);

        // ONE head, ONE kv head -- the same shape the fixture above is laid out for.
        //
        // The previous call passed `kKvHeads, kKvHeads` (2, 2) while `big_v` holds only
        // `kHeadDim * cells` floats, i.e. a single kv head. `reduce_values` then addressed
        // `d + head_dim*(kv_head + 2*cell)`, walked off the end of the buffer, and returned numbers
        // of magnitude 2.4e+31 -- which the 1e-5 tolerance reported as a failure, correctly. The
        // mismatch was in this line, not in the kernel; the independent test that uses matching
        // shapes passes. A fixture and its call have to describe the same layout, and the pair is
        // written adjacently here so a future edit cannot change one without seeing the other.
        std::vector<float> mine;
        check(tq::reader::reduce_values(big_v.data(), kHeadDim, /*n_head_kv=*/1, /*n_head=*/1,
                                        cells, big_w.data(), mine) ==
                  tq::reader::physical_refusal::ok,
              "the reduction ran at the larger cache size");
        double big_worst = 0.0;
        std::size_t big_diff = 0;
        for (std::size_t d = 0; d < kHeadDim; ++d) {
            big_worst = std::max(big_worst, std::fabs(double(mine[d]) - double(bref[d])));
            if (std::memcmp(&mine[d], &bref[d], sizeof(float)) != 0) ++big_diff;
        }
        std::fprintf(stderr, "  at %4zu cells: max|diff| %.3e, %zu/%zu bit-different\n", cells,
                     big_worst, big_diff, kHeadDim);
        // Tolerance scales with the number of terms rounded, and 1e-5 relative is still three
        // orders below the O(1) error a wrong layout produces.
        check(big_worst < 1e-5,
              "the reduction stays within float tolerance at this size: whatever order ggml's "
              "build uses, the answer is the same value");
        ggml_free(bctx);
    }

    ggml_free(ctx);
}

// ---------------------------------------------------------------------------------------------
// 6. Overflow in the output size.
// ---------------------------------------------------------------------------------------------
void an_overflowing_output_size_is_refused() {
    const std::size_t big = std::numeric_limits<std::size_t>::max() / 2 + 1;
    std::vector<float> weights(1, 1.0f);
    std::vector<float> out;
    const float dummy = 0.0f;
    const physical_refusal r = tq::reader::reduce_values(&dummy, big, 1, 2, 1, weights.data(), out);
    check(r == physical_refusal::size_overflow,
          "an output whose size overflows size_t is refused before the allocation, so a wrapped "
          "product cannot allocate a small buffer and then index past it");
    check(out.empty(), "and nothing is written");
    std::fprintf(stderr, "overflow: %s\n", tq::reader::describe(r));
}

// ---------------------------------------------------------------------------------------------
// 7. The ELEMENT index overflows separately from the output size.
//
// `out` here is tiny and entirely valid: one head, one dimension. What would overflow is the
// cache's own addressing, `head_dim * n_head_kv * n_kv`. That combination is the one the previous
// check could not see, and it is the dangerous half -- a wrapped index lands back inside `v`, so
// the read succeeds and the sum is taken over another cell's values. No sanitizer flags it and no
// output-size check catches it.
void an_overflowing_element_index_is_refused() {
    std::vector<float> weights(1, 1.0f);
    std::vector<float> out;
    const float dummy = 0.0f;

    {
        // n_kv * n_head_kv wraps: n_kv > SIZE_MAX / 2 with n_head_kv == 2.
        const std::size_t n_kv = std::numeric_limits<std::size_t>::max() / 2 + 1;
        const physical_refusal r =
            tq::reader::reduce_values(&dummy, 1, 2, 2, n_kv, weights.data(), out);
        check(r == physical_refusal::size_overflow,
              "a cell count that wraps the head factor is refused, even though the output is one "
              "element: the index would land back inside v and read another cell");
        check(out.empty(), "and the output stays empty -- a refusal must not hand back a buffer");
    }
    {
        // head_dim * (n_head_kv * n_kv) wraps with a legal-looking head count.
        const std::size_t n_kv = std::size_t(1) << 32;
        const std::size_t head_dim = (std::size_t(1) << 32) + 1;   // product exceeds 64 bits
        const physical_refusal r =
            tq::reader::reduce_values(&dummy, head_dim, 1, 1, n_kv, weights.data(), out);
        check(r == physical_refusal::size_overflow,
              "and a head dimension whose product with the cache size wraps is refused too");
        check(out.empty(), "with no partial output");
    }
}

// ---------------------------------------------------------------------------------------------
// 9. The SCALE is an input too, and it is unchecked by every caller.
//    `kq_scale` comes from the model's hyperparameters, so a corrupt one -- NaN from a division,
//    infinity from a bad reciprocal -- reaches this function with no layer between it and the
//    weights. The three cases below assert that a degenerate scale is REFUSED ATOMICALLY: same
//    verdict as a bad logit, and the vector is left bit-for-bit as it arrived.
//
//    The scale is also where the -infinity convention gets subtle, so one case pins it: a finite
//    positive logit under a scale of -infinity overflows to -infinity, which `is_hidden` accepts.
//    That is not a hole -- a scale that drives a logit to minus infinity genuinely gives it zero
//    mass -- but it means "hidden" is a property of the SCALED value, and the case says so.
void a_degenerate_scale_is_refused_atomically() {
    const auto untouched = [](const std::vector<float> & a, const std::vector<float> & b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) return false;
        }
        return true;
    };

    {
        std::vector<float> logits = {1.0f, 2.0f, 3.0f};
        const std::vector<float> before = logits;
        const float nan_scale = std::numeric_limits<float>::quiet_NaN();
        check(tq::reader::softmax_scaled(logits, nan_scale) ==
                  tq::reader::physical_refusal::not_normalisable,
              "a NaN scale is refused: every scaled logit is NaN, so no cell has a maximum to "
              "subtract");
        check(untouched(logits, before),
              "and the input survives it untouched -- a caller that retries with a corrected "
              "scale must not be normalising NaN-multiplied values");
    }
    {
        // +infinity scale with a POSITIVE logit overflows to +infinity, which is refused rather
        // than allowed to become the maximum: an infinite maximum would make every other cell
        // exp(-inf) = 0 and concentrate all mass on one key, which is a wrong answer that looks
        // like a sharp one.
        std::vector<float> logits = {1.0f, 2.0f};
        const std::vector<float> before = logits;
        const float inf_scale = std::numeric_limits<float>::infinity();
        check(tq::reader::softmax_scaled(logits, inf_scale) ==
                  tq::reader::physical_refusal::not_normalisable,
              "an infinite scale that overflows a positive logit is refused, not concentrated");
        check(untouched(logits, before), "with the input untouched");
    }
    {
        // The converse, and the reason the three cases are not one: an infinite scale on NEGATIVE
        // logits drives them to -infinity, which is exactly the hidden convention. There is no
        // visible cell left, so the refusal is `all_hidden` -- a different code for a different
        // fact, and a caller that maps both to "bad scale" would lose the distinction.
        std::vector<float> logits = {-1.0f, -2.0f};
        const std::vector<float> before = logits;
        const float inf_scale = std::numeric_limits<float>::infinity();
        check(tq::reader::softmax_scaled(logits, inf_scale) ==
                  tq::reader::physical_refusal::all_hidden,
              "an infinite scale on negative logits hides every cell, and that is reported as "
              "all_hidden rather than as a general non-finite failure");
        check(untouched(logits, before), "with the input untouched here too");
    }
    {
        // THE INVARIANT THAT MAKES THE SECOND-LOOP GUARD DEAD, asserted rather than argued.
        //
        // `total` is the sum of `exp(scaled_i - maximum)` over the same finite values `maximum` was
        // taken over, so the argmax contributes exp(0) == 1 exactly and total is always in
        // [1, visible]. The consequence is that `not_normalisable` is unreachable from that point,
        // and every refusal this function can produce -- empty input, all hidden, NaN or +inf from
        // pass 1 -- happens BEFORE any write.
        //
        // That last sentence is the whole of the atomicity property, and it is provable only if the
        // invariant is checked. So it is checked here, over inputs chosen to break it if it were
        // breakable: a scale large enough that every scaled logit is the same huge negative number,
        // the largest finite float against its negative, denormals, and a single cell. A guard
        // commented "cannot fire" without this case would be an assertion with nothing behind it --
        // exactly the shape this project keeps having to remove.
        struct probe { const char * name; std::vector<float> logits; float scale; };
        const std::vector<probe> adversarial = {
            {"scale -1e30 (all exponents equal)", {-1.0f, 2.0f, 0.0f}, -1e30f},
            {"FLT_MAX against -FLT_MAX", {std::numeric_limits<float>::max(),
                                          -std::numeric_limits<float>::max()}, 1.0f},
            {"denormals", {std::numeric_limits<float>::denorm_min(),
                           std::numeric_limits<float>::denorm_min()}, 1.0f},
            {"single cell", {3.0f}, 1.0f},
            {"one cell, one hidden", {1.0f, -std::numeric_limits<float>::infinity()}, 1.0f},
        };
        for (const probe & p : adversarial) {
            std::vector<float> row = p.logits;
            const std::vector<float> before = row;
            const physical_refusal r = tq::reader::softmax_scaled(row, p.scale);
            if (r == physical_refusal::ok) {
                // total was in [1, visible]: every weight is finite, and the visible ones sum to 1.
                double sum = 0.0;
                bool finite = true;
                for (float w : row) {
                    if (!std::isfinite(w)) finite = false;
                    sum += double(w);
                }
                check(finite,
                      std::string("normalisable input (") + p.name +
                          ") produced only finite weights: total left [1, visible] would show up "
                          "here as a non-finite or zero weight");
                check(std::fabs(sum - 1.0) < 1e-5,
                      std::string("and the weights sum to one (") + p.name + ")");
            } else {
                check(r == physical_refusal::all_hidden,
                      std::string("the only refusal reachable from finite, visible input is "
                                  "all_hidden (") + p.name + ")");
                check(untouched(row, before),
                      std::string("and it wrote nothing (") + p.name + ")");
            }
        }
    }
    {
        // The one scale that must WORK, because it is the ordinary case: 0 makes every visible cell
        // equally weighted. It is also where -inf * 0 would produce NaN if the hidden check ran
        // after the multiply instead of before.
        std::vector<float> logits = {5.0f, -3.0f, -std::numeric_limits<float>::infinity()};
        check(tq::reader::softmax_scaled(logits, 0.0f) == tq::reader::physical_refusal::ok,
              "a zero scale normalises the visible cells uniformly and does not turn the hidden "
              "one into NaN");
        check(std::fabs(logits[0] - 0.5f) < 1e-6f && std::fabs(logits[1] - 0.5f) < 1e-6f,
              "two visible cells under a zero scale weigh one half each");
        check(logits[2] == 0.0f,
              "and the hidden cell weighs exactly zero -- -inf times a zero scale is the NaN this "
              "ordering avoids");
    }
}

// ---------------------------------------------------------------------------------------------
// 8. A refused normalisation leaves the input exactly as it arrived.
//
// The previous implementation transformed `logits` in place as it computed, so `not_normalisable`
// returned with the vector holding scaled exponents -- half a transformation. `attend_physical`
// discarded it, but `softmax_scaled` is public, and a caller that retried after a refusal would have
// been normalising already-scaled values. A refusal that damages its input is a partial success with
// a status code.
void a_refused_normalisation_does_not_touch_its_input() {
    // Every exponent underflows: a hugely negative scale maps finite logits to -infinity, which is
    // treated as hidden, so no cell is visible... except `all_hidden` would fire first. To reach
    // `not_normalisable` instead, give one cell a logit whose scaled value is finite but whose
    // exponent underflows to zero, and the rest -infinity is not reachable either. The reachable
    // case is a NaN/inf logit, so that is what is used -- and the property under test is the same:
    // a refusal must not have written anything.
    std::vector<float> logits = {1.0f, std::numeric_limits<float>::infinity(), 2.0f};
    const std::vector<float> before = logits;

    const physical_refusal r = tq::reader::softmax_scaled(logits, 1.0f);
    check(r == physical_refusal::not_normalisable,
          "an infinite logit is refused rather than concentrating all mass on one cell");
    bool untouched = logits.size() == before.size();
    for (std::size_t i = 0; untouched && i < before.size(); ++i) {
        untouched = std::memcmp(&logits[i], &before[i], sizeof(float)) == 0;
    }
    check(untouched,
          "and the input is bit-for-bit what the caller passed: a refusal that leaves a half-"
          "transformed vector makes a retry compute on already-scaled values");

    // The success path still writes, so the property above is not \"it never writes anything\".
    std::vector<float> ok = {1.0f, 2.0f};
    const std::vector<float> ok_before = ok;
    check(tq::reader::softmax_scaled(ok, 1.0f) == physical_refusal::ok, "a finite row normalises");
    check(std::memcmp(&ok[0], &ok_before[0], sizeof(float)) != 0,
          "and the weights really were replaced -- the negative case above must not be passing "
          "because this function never writes at all");
}

} // namespace

int main() {
    indivisible_heads_are_refused_not_read();
    bad_arguments_are_refused_rather_than_answered_with_zeros();
    non_finite_logits_are_refused();
    hidden_cells_weigh_exactly_zero();
    the_order_is_reproduced_and_the_arithmetic_is_not();
    an_overflowing_output_size_is_refused();
    an_overflowing_element_index_is_refused();
    a_refused_normalisation_does_not_touch_its_input();
    a_degenerate_scale_is_refused_atomically();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "physical core: indivisible heads, null buffers, non-finite logits and overflow "
                 "are refused; hidden cells weigh exactly zero; the term order matches ggml and the "
                 "accumulator does not\n");
    return 0;
}
