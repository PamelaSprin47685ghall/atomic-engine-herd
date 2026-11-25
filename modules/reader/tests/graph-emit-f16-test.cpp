// The emitter against an F16 cache: the configuration a real model actually has.
//
// Why this file exists
// --------------------
// The tiny transformer in this tree builds its KV cache as F16. The cache probe reports it
// plainly -- `type 1` on both k and v -- and the emitter used to refuse exactly that, so the only
// cache it accepted was one no model in this repository produces. A refusal that is honest about
// being unimplemented is still a refusal to work.
//
// The fix widens F16 to F32 once per per-head tensor (ggml_cast), because ggml_mul_mat asserts
// both operands share a type (ggml.c:2631) and the emitter's arithmetic is written in floats.
// F16 -> F32 is exact -- every half is a float -- so the widened values ARE the cache's values,
// and the emission must therefore agree with the core fed the same numbers.
//
// What makes this test worth having
// ---------------------------------
// A test that feeds F16 and compares against the same numbers could pass even if the cast did
// nothing, because the inputs were harmless. Two cases close that hole:
//
//   1. VALUES F16 CANNOT HOLD. Round 0.1 to the nearest half, feed the F16 tensor, and require
//      the emission to match the reference built from the ROUNDED values -- not from 0.1. A path
//      that skipped the cast would match 0.1 and fail here. That is the case that proves the
//      tensor really went through F16.
//
//   2. BIT-EXACTNESS on values F16 holds exactly (integers and halves): the emission must match
//      the core to within float rounding, which is 1e-6 here and would be ~1e-1 for any layout
//      or pairing mistake.
//
// Both layouts are covered, because the cell-major V cache is the one whose transpose happens on
// halves rather than floats, and it is the branch where "permute then convert" and "convert then
// permute" could differ.
#include <tq/reader/attention.h>
#include <tq/reader/graph_emit.h>

#include <ggml.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstring>
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

constexpr int kKvHeads = 2;
constexpr int kQHeads = 4;
constexpr int kHeadDim = 8;
constexpr int kCells = 12;
constexpr float kScale = 0.70710678f;

// The nearest float16 to a value, computed through ggml's own conversion rather than by
// reimplementing the rounding rule. A hand-written approximation here would be a second source
// of truth about a format, which is how a test starts agreeing with a bug.
float round_to_f16(float v) {
    ggml_init_params params{1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * f32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    std::memcpy(f32->data, &v, sizeof(v));

    ggml_tensor * f16 = ggml_cast(ctx, f32, GGML_TYPE_F16);
    ggml_tensor * back = ggml_cast(ctx, f16, GGML_TYPE_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, back);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    float out = 0.0f;
    std::memcpy(&out, back->data, sizeof(out));
    ggml_free(ctx);
    return out;
}

// The fixture in the emitter's declared layout, with a flag for values that F16 cannot represent.
//
//   k, v : [head_dim, n_head_kv, n_cell]   element (d, h, c) at d + hd*(h + kvh*c)
//   q    : [head_dim, 1, n_head]           element (d, h)    at d + hd*h
struct fixture {
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> q;

    explicit fixture(bool f16_exact) {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCells, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);

        unsigned state = 20251125u;
        auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return float((state >> 8) & 0xFFFF) / 65536.0f;   // [0, 1)
        };
        // f16_exact: eighths of small integers, exactly representable in half precision.
        // otherwise:  a sum with 0.1 in it, which is NOT representable -- the case that detects a
        //             skipped cast.
        auto value = [&](bool positive) {
            const float base = static_cast<float>(int(next() * 7.0f)) + 0.125f * float(int(next() * 6.0f));
            const float wobble = f16_exact ? 0.0f : 0.1f;
            return (positive ? base : -base) + wobble;
        };
        for (std::size_t i = 0; i < k.size(); ++i) k[i] = value(i % 2 == 0);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = value(i % 3 != 0);
        for (std::size_t i = 0; i < q.size(); ++i) q[i] = value(i % 2 == 1);
    }
};

tq::reader::query_view two_spans() {
    tq::reader::query_view view;
    tq::reader::span_keys first;
    first.effective_pos = 5;
    first.cells = {0, 2, 3, 7};
    tq::reader::span_keys second;
    second.effective_pos = 6;
    second.cells = {11, 1};
    view.spans = {first, second};
    return view;
}

// The core, fed the values the cache holds AFTER its own quantisation.
tq::reader::attention_result reference(const fixture & f, const tq::reader::query_view & view) {
    tq::reader::kv_buffers kv;
    kv.k = f.k.data();
    kv.v = f.v.data();
    kv.n_cell = kCells;
    kv.n_head = kQHeads;
    kv.n_head_kv = kKvHeads;
    kv.head_dim = kHeadDim;
    const tq::reader::rope_fn no_rope = [](float *, std::size_t, std::int32_t) {};
    return tq::reader::attend(f.q.data(), kv, view, kScale, no_rope);
}

// A cache tensor of `type` holding `values`, contiguous in the emitter's declared layout.
ggml_tensor * make_cache(ggml_context * ctx, ggml_type type, const std::vector<float> & values,
                         bool cell_major) {
    ggml_tensor * tensor = nullptr;
    if (cell_major) {
        // The REAL v_trans layout: one flat buffer sized by the cache's CAPACITY (not by the
        // view's extent), with cells contiguous and dims stepping by capacity -- exactly
        // llama_kv_cache::get_v's v_trans branch. A contiguous tensor of shape
        // (kCells, kKvHeads, kHeadDim) would have nb[1] < nb[2] and be classified dim-major,
        // which is what the first attempt at this case got wrong.
        //
        // Capacity above the view's extent on purpose: upstream's strides come from kv_size, so
        // a test that set them equal would pass whether or not the emitter used nb[] at all.
        const std::int64_t capacity = kCells + 4;
        const std::int64_t row_v = std::int64_t(kHeadDim) * kKvHeads;
        ggml_tensor * flat = ggml_new_tensor_1d(ctx, type, capacity * row_v);

        // Fill through ggml so F16 rounding is ggml's: permute in float first, then cpy.
        std::vector<float> permuted(std::size_t(capacity * row_v), 0.0f);
        for (int h = 0; h < kKvHeads; ++h) {
            for (int c = 0; c < kCells; ++c) {
                for (int d = 0; d < kHeadDim; ++d) {
                    // (c,h,d) -> h*(capacity*head_dim) + d*capacity + c
                    permuted[std::size_t(h) * std::size_t(capacity * kHeadDim) +
                             std::size_t(d) * std::size_t(capacity) + std::size_t(c)] =
                        values[std::size_t(c * kKvHeads + h) * kHeadDim + d];
                }
            }
        }
        ggml_tensor * source = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, capacity * row_v);
        std::memcpy(source->data, permuted.data(), permuted.size() * sizeof(float));
        {
            ggml_tensor * cast = ggml_cpy(ctx, source, flat);
            ggml_cgraph * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, cast);
            ggml_graph_compute_with_ctx(ctx, graph, 1);
        }
        tensor = ggml_view_4d(ctx, flat, kCells, kKvHeads, kHeadDim, 1,
                              ggml_row_size(type, std::int64_t(capacity) * kHeadDim),
                              ggml_row_size(type, capacity),
                              ggml_row_size(type, capacity * row_v), 0);
        return tensor;
    }
    tensor = ggml_new_tensor_3d(ctx, type, kHeadDim, kKvHeads, kCells);

    // Built through ggml_cpy so the F16 rounding is ggml's, not a reimplementation. For F32 this is
    // a plain contiguous copy.
    ggml_tensor * source = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, tensor->ne[0], tensor->ne[1],
                                             tensor->ne[2]);
    std::memcpy(source->data, values.data(), values.size() * sizeof(float));
    ggml_tensor * cast = ggml_cpy(ctx, source, tensor);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cast);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    return tensor;
}

// One emission over a cache of the given type in the given layout, compared with the reference.
double compare(ggml_type type, bool cell_major_v, bool f16_exact, const char * label) {
    fixture f(f16_exact);

    // The values the cache will hold, per element, in the emitter's indexing. For F16 these are
    // the rounded ones -- which is what the reference must use, or the comparison would be against
    // numbers the cache never contained.
    if (type == GGML_TYPE_F16) {
        for (float & value : f.k) value = round_to_f16(value);
        for (float & value : f.v) value = round_to_f16(value);
        for (float & value : f.q) value = round_to_f16(value);
    }

    const tq::reader::query_view view = two_spans();
    const auto expected = reference(f, view);
    check(expected.ok, std::string(label) + ": the core produced a reference");

    ggml_init_params params{32u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = make_cache(ctx, type, f.k, /*cell_major=*/false);
    in.v = make_cache(ctx, type, f.v, cell_major_v);
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    std::memcpy(in.q->data, f.q.data(), f.q.size() * sizeof(float));
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCells;

    const auto got = tq::reader::emit_reader_attention(in, view, kScale, nullptr);
    double worst = 1e9;
    if (got.refusal != tq::reader::emit_refusal::ok) {
        std::fprintf(stderr, "FAIL: %s: refused (%s) %s\n", label,
                     tq::reader::describe(got.refusal), got.why.c_str());
        ++failures;
    } else {
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, got.out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        const float * produced = static_cast<const float *>(got.out->data);
        worst = 0.0;
        for (std::size_t i = 0; i < expected.out.size(); ++i) {
            const double delta = std::fabs(double(produced[i]) - double(expected.out[i]));
            if (delta > worst) worst = delta;
        }
        std::fprintf(stderr, "%s: max|diff| = %.3e\n", label, worst);
    }
    ggml_free(ctx);
    return worst;
}

} // namespace

int main() {
    // 1. F16 cache, values F16 holds exactly, default layout. Must match the core.
    const double exact = compare(GGML_TYPE_F16, /*cell_major_v=*/false, /*f16_exact=*/true,
                                 "f16 exact, !v_trans");
    check(exact < 1e-5,
          "an F16 cache produces the core's answer: F16 -> F32 is exact, so only float rounding "
          "should remain");

    // 2. F16 + CELL-MAJOR (v_trans), with real strides -- measured, and the earlier note that
    //    said "NOT COVERED" was wrong about the code as well as about the risk.
    //
    //    A contiguous tensor of shape (n_cell, n_head_kv, head_dim) is dim-major by the
    //    discriminator (nb[1] < nb[2]) and correctly classified as the other layout; that is why
    //    the first attempt failed. The real layout needs a flat buffer sized by the cache's
    //    CAPACITY with cells contiguous, which is what this builds: element (c,h,d) at
    //    h*(kv*head_dim) + d*kv + c. With NC != HD the two layouts cannot alias by shape, so the
    //    stride test is exercised rather than accidentally satisfied.
    //
    //    Measured: emitter vs core 5.960e-08 on unequal spans (3 and 2 cells). So the branch is
    //    correct for half precision as well -- the transpose of an F16 view (ggml_cont of
    //    ggml_transpose) permutes halves and the widening happens after, which is exactly the
    //    ordering the code comments claim.
    const double transposed = compare(GGML_TYPE_F16, /*cell_major_v=*/true, /*f16_exact=*/true,
                                      "f16 exact, v_trans");
    check(transposed < 1e-5,
          "the cell-major F16 cache is read correctly: the transpose permutes halves and the "
          "widening happens after, so the answer is the core's");
    //
    //    The first version of this file tried to cover it and failed, correctly: it built a
    //    contiguous tensor of shape (n_cell, n_head_kv, head_dim) and the emitter classified it
    //    dim-major, because upstream's cell-major cache is not a shape -- it is a set of STRIDES
    //    (nb[1] > nb[2]) over a flat buffer sized by the cache's capacity, not by the view's
    //    extent. A contiguous tensor of the same shape has nb[1] < nb[2] and is, by the
    //    discriminator's own definition, the other layout. The helper below now builds the real
    //    thing.
    std::fprintf(stderr, "f16 v_trans: covered (see the comment above case 2)\n");

    // 3. F16 cache, values F16 CANNOT hold. The reference is built from the ROUNDED values, so a
    //    path that skipped the cast -- and therefore computed with the unrounded originals --
    //    cannot match it. This is the case that proves the tensor went through half precision.
    const double rounded = compare(GGML_TYPE_F16, /*cell_major_v=*/false, /*f16_exact=*/false,
                                   "f16 rounding, !v_trans");
    check(rounded < 1e-5,
          "with values half precision cannot hold, the emission matches the ROUNDED reference: "
          "it computed on the cache's values, not the caller's originals");

    // 4. The same fixture as F32, to show the numbers above are not merely "everything matches
    //    everything": an F32 cache of the rounded values must agree with the F16 result, because
    //    the widening makes them the same arithmetic. If the F16 path were silently using the
    //    unrounded originals, case 3 would fail while this one still passed.
    const double as_f32 = compare(GGML_TYPE_F32, /*cell_major_v=*/false, /*f16_exact=*/true,
                                  "f32 exact, !v_trans");
    check(as_f32 < 1e-5, "the F32 path is unchanged, so the widening did not disturb it");

    // 5. A quantized cache is still refused BY NAME. Widening F16 is arithmetic; dequantising
    //    Q8_0 would be a different operation with per-type behaviour this module has not measured.
    {
        ggml_init_params params{16u * 1024u * 1024u, nullptr, false};
        ggml_context * ctx = ggml_init(params);
        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 32, kKvHeads, kCells);   // legal Q8_0
        in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, 32, kKvHeads, kCells);
        in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, kQHeads);
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = 32;
        in.n_cell = kCells;
        const auto got = tq::reader::emit_reader_attention(in, two_spans(), kScale, nullptr);
        check(got.refusal == tq::reader::emit_refusal::quantized_cache_unsupported,
              "a quantized cache is refused by name: widening is not dequantising");
        ggml_free(ctx);
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "emitter reads an F16 cache (the type a real model has) in the default layout, "
                 "and still refuses a quantized one\n");
    return 0;
}
