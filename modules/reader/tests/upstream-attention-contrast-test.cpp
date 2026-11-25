// The emitter's output against the UPSTREAM OPERATOR SEQUENCE, not against a reference we wrote.
//
// `graph-emit-physical-test` compares the graph with a hand-written double reference. That proves the
// graph computes attention; it cannot prove the graph's LAYOUT matches what `llm_graph_context::
// build_attn_mha` returns, because both sides were written by us. The claim in the header -- that
// upstream's return is flat `d + head_dim*head` for one token -- used to be a reading of the source
// (permute `(0,2,1,3)`, then `ggml_cont_2d`). This test replaces that reading with a computation:
//
//   kq   = ggml_mul_mat(k, q)                      // [n_cell, 1, n_head]
//   kq   = ggml_soft_max_ext(kq, mask, scale, 0)   // mask carries -INFINITY where not visible
//   kqv  = ggml_mul_mat(v, kq)                     // [head_dim, 1, n_head]
//   kqv  = ggml_permute(kqv, 0, 2, 1, 3)
//   kqv  = ggml_cont_2d(kqv, ne0*ne1, ne2*ne3)     // [head_dim*n_head, 1] -- the upstream tail
//
// Those are upstream's own operators in upstream's own order. If the emitter's `out` agrees with the
// result value-by-value, then the emitter can be plugged into `build_attn_mha` by `ggml_reshape_1d`
// and nothing else. If it disagrees, the header's claim is wrong and this file says so.
//
// What this is NOT: a hook replacement. Nothing here installs a seam or touches llama.cpp. It grades
// the tensor the emitter hands back, in the same arithmetic the CPU backend runs upstream.
//
// F16 cache, mixed with an F32 query: the mixture a real graph has. `round_through_f16` narrows the
// fixture first, so both sides see exactly the numbers the half-precision cache holds.
#include <tq/reader/attention.h>
#include <tq/reader/graph_emit_physical.h>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
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

constexpr int kHeadDim = 8;
constexpr int kKvHeads = 2;
constexpr int kQHeads  = 4;
constexpr int kCell    = 16;
constexpr float kScale = 0.70710678f;
constexpr std::size_t kArena = 1024u * 1024u;

// 5 visible cells out of 16, neither a prefix nor contiguous.
tq::reader::query_view the_view(int n_spans) {
    tq::reader::query_view view;
    tq::reader::span_keys a;
    a.effective_pos = 5;
    a.cells = {0, 7, 3};
    tq::reader::span_keys b;
    b.effective_pos = 5;
    b.cells = {12, 15};
    view.spans.push_back(a);
    if (n_spans > 1) view.spans.push_back(b);
    return view;
}

struct fixture {
    // emitter layouts, as `graph_emit_physical.h` takes them:
    //   k, v : [head_dim, n_head_kv, n_cell]   idx = d + HD*(kvh + KVH*cell)
    //   q    : [head_dim, 1, n_head]           idx = d + HD*h
    std::vector<float> k, v, q;

    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        unsigned state = 0x2468aceu;
        auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return float((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
    }
};

float round_through_f16(float x) {
    ggml_fp16_t h{};
    ggml_fp32_to_fp16_row(&x, &h, 1);
    float y = 0.0f;
    ggml_fp16_to_fp32_row(&h, &y, 1);
    return y;
}

std::vector<int> visible_cells(const tq::reader::query_view & view) {
    std::vector<int> visible(kCell, 0);
    for (const auto & span : view.spans) {
        for (std::int32_t cell : span.cells) visible[std::size_t(cell)] = 1;
    }
    return visible;
}

// THE STOCK SIDE: upstream's operator sequence and upstream's layouts.
//
//   k2 : [head_dim, n_cell, n_head_kv]   idx = d + HD*(i + NK*kvh)   -- what llama builds
//   v2 : [n_cell,   head_dim, n_head_kv] idx = i + NK*(d + HD*kvh)   -- mul_mat's ne0 is n_cell
//
// Returns the flat `d + head_dim*head` vector exactly as `build_attn_mha`'s tail produces it.
std::vector<float> upstream_stock_out(const fixture & f, const tq::reader::query_view & view) {
    std::vector<float> k2(std::size_t(kHeadDim) * kCell * kKvHeads);
    std::vector<float> v2(std::size_t(kCell) * kHeadDim * kKvHeads);
    for (int kvh = 0; kvh < kKvHeads; ++kvh) {
        for (int i = 0; i < kCell; ++i) {
            for (int d = 0; d < kHeadDim; ++d) {
                k2[std::size_t(d) + kHeadDim * (i + kCell * kvh)] =
                    f.k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * i)];
                v2[std::size_t(i) + kCell * (d + kHeadDim * kvh)] =
                    f.v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * i)];
            }
        }
    }

    const std::vector<int> visible = visible_cells(view);
    std::vector<float> mask(std::size_t(kCell) * kQHeads);
    for (int h = 0; h < kQHeads; ++h) {
        for (int i = 0; i < kCell; ++i) {
            mask[std::size_t(i) + kCell * h] = visible[std::size_t(i)] ? 0.0f : -INFINITY;
        }
    }

    std::vector<ggml_fp16_t> kh(k2.size()), vh(v2.size());
    ggml_fp32_to_fp16_row(k2.data(), kh.data(), std::int64_t(k2.size()));
    ggml_fp32_to_fp16_row(v2.data(), vh.data(), std::int64_t(v2.size()));

    // An allocating context is enough here: this graph is small and not the emitter's admission test.
    ggml_context * ctx = ggml_init({/*mem_size=*/4u * 1024u * 1024u, /*mem_buffer=*/nullptr,
                                    /*no_alloc=*/false});
    std::vector<float> flat;
    if (!ctx) return flat;

    ggml_tensor * kt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, kCell, kKvHeads);
    ggml_tensor * vt = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kCell, kHeadDim, kKvHeads);
    ggml_tensor * qt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    ggml_tensor * mt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kCell, 1, kQHeads);
    std::memcpy(kt->data, kh.data(), kh.size() * sizeof(ggml_fp16_t));
    std::memcpy(vt->data, vh.data(), vh.size() * sizeof(ggml_fp16_t));
    std::memcpy(qt->data, f.q.data(), f.q.size() * sizeof(float));
    std::memcpy(mt->data, mask.data(), mask.size() * sizeof(float));

    ggml_tensor * kq = ggml_mul_mat(ctx, kt, qt);
    kq = ggml_soft_max_ext(ctx, kq, mt, kScale, 0.0f);
    ggml_tensor * kqv = ggml_mul_mat(ctx, vt, kq);
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
    kqv = ggml_cont_2d(ctx, kqv, kqv->ne[0] * kqv->ne[1], kqv->ne[2] * kqv->ne[3]);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, kqv);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) == GGML_STATUS_SUCCESS) {
        const float * p = static_cast<const float *>(kqv->data);
        flat.assign(p, p + std::size_t(kHeadDim) * kQHeads);
    }
    ggml_free(ctx);
    return flat;
}

// THE EMITTER SIDE: the same fixture and view, through `emit_physical_attention`.
std::vector<float> emitter_out(const fixture & f, const tq::reader::query_view & view,
                               std::string & why) {
    ggml_context * ctx = ggml_init({/*mem_size=*/kArena, /*mem_buffer=*/nullptr, /*no_alloc=*/true});
    std::vector<float> flat;
    if (!ctx) { why = "ggml_init(no_alloc) failed"; return flat; }

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, kKvHeads, kCell);
    in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, kKvHeads, kCell);
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCell;
    in.collect_indices = true;
    ggml_set_input(in.k);
    ggml_set_input(in.v);
    ggml_set_input(in.q);

    const auto emitted = tq::reader::emit_physical_attention(in, view, kScale);
    if (emitted.refusal != tq::reader::physical_emit_refusal::ok) {
        why = std::string(tq::reader::describe(emitted.refusal)) + " -- " + emitted.why;
        ggml_free(ctx);
        return flat;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, emitted.out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = backend ? ggml_backend_alloc_ctx_tensors(ctx, backend) : nullptr;
    if (!backend || !buf) {
        why = "placement failed";
        if (buf) ggml_backend_buffer_free(buf);
        if (backend) ggml_backend_free(backend);
        ggml_free(ctx);
        return flat;
    }

    for (const auto & entry : emitted.indices) {
        // `entry.span` beyond the view means the emitter wants the bulk list, and that list is only
        // as long as it is. Copying `n_cells` from a shorter source reads past the end; the test
        // would then compare against garbage rather than report a contract violation.
        const std::vector<std::int32_t> * src =
            entry.span < view.spans.size() ? &view.spans[entry.span].cells : &emitted.index_cells;
        if (src->size() < entry.n_cells) {
            why = "index entry (span " + std::to_string(static_cast<long long>(entry.span)) +
                  ") needs " + std::to_string(entry.n_cells) + " cells, source holds " +
                  std::to_string(src->size());
            ggml_backend_buffer_free(buf);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return flat;
        }
        std::memcpy(entry.index->data, src->data(), entry.n_cells * sizeof(std::int32_t));
    }

    std::vector<ggml_fp16_t> half(std::max(f.k.size(), f.v.size()));
    ggml_fp32_to_fp16_row(f.k.data(), half.data(), std::int64_t(f.k.size()));
    std::memcpy(in.k->data, half.data(), f.k.size() * sizeof(ggml_fp16_t));
    ggml_fp32_to_fp16_row(f.v.data(), half.data(), std::int64_t(f.v.size()));
    std::memcpy(in.v->data, half.data(), f.v.size() * sizeof(ggml_fp16_t));
    std::memcpy(in.q->data, f.q.data(), f.q.size() * sizeof(float));

    if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
        const float * p = static_cast<const float *>(emitted.out->data);
        flat.assign(p, p + std::size_t(kHeadDim) * kQHeads);
        why = "ok";
    } else {
        why = "compute failed";
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return flat;
}

void contrast(int n_spans) {
    fixture f;
    // Narrow the fixture through F16 first: both sides must see the numbers the cache holds.
    for (float & x : f.k) x = round_through_f16(x);
    for (float & x : f.v) x = round_through_f16(x);

    const tq::reader::query_view view = the_view(n_spans);

    std::string why;
    const std::vector<float> emitter = emitter_out(f, view, why);
    const std::vector<float> stock = upstream_stock_out(f, view);

    // Both must produce something before anything is compared. The earlier shape of this function
    // checked `emitter.size() != stock.size()` and returned, which is true for 0 != 0 as well as
    // for 32 != 32, so two empty vectors fell through: the diff loop ran zero times, max|diff| was
    // 0, and the run looked like agreement until the non-vacuity checks happened to catch it.
    if (emitter.empty() || stock.empty()) {
        check(false, "no output (emitter " + std::to_string(emitter.size()) + " [" + why +
                         "], stock " + std::to_string(stock.size()) + ")");
        return;
    }
    if (emitter.size() != stock.size()) {
        check(false, "sizes differ: emitter " + std::to_string(emitter.size()) + " vs stock " +
                         std::to_string(stock.size()));
        return;
    }

    double max_abs = 0.0;
    int worst = -1;
    for (std::size_t i = 0; i < emitter.size(); ++i) {
        // NaN is the case this loop cannot see: `NaN > max_abs` is false, so a NaN element never
        // raises max_abs and a tensor full of NaNs reports max|diff| = 0. A test that cannot fail
        // on NaN is not measuring anything, so non-finite values are a failure of their own.
        if (!std::isfinite(emitter[i]) || !std::isfinite(stock[i])) {
            check(false, "non-finite value at index " + std::to_string(i) + ": emitter " +
                             std::to_string(emitter[i]) + ", stock " + std::to_string(stock[i]));
            return;
        }
        const double d = std::fabs(double(emitter[i]) - double(stock[i]));
        if (d > max_abs) { max_abs = d; worst = int(i); }
    }

    // A layout mismatch is not a rounding difference: a permuted order swaps values that differ on
    // the order of the values themselves, while the same arithmetic agrees to float precision.
    check(max_abs < 1e-5, "emitter vs upstream operator sequence: max|diff| = " +
                              std::to_string(max_abs) + " at index " + std::to_string(worst));

    // Non-vacuity: the output must be non-zero and head-dependent, and reversing the head order must
    // NOT agree -- otherwise "max|diff| tiny" could come from zeros on both sides.
    double magnitude = 0.0;
    for (float x : stock) magnitude = std::max(magnitude, std::fabs(double(x)));
    check(magnitude > 1e-3, "stock output is degenerate (max |value| = " +
                                std::to_string(magnitude) + ")");
    double swapped = 0.0;
    for (int h = 0; h < kQHeads; ++h) {
        for (int d = 0; d < kHeadDim; ++d) {
            const std::size_t a = std::size_t(d) + kHeadDim * h;
            const std::size_t b = std::size_t(d) + kHeadDim * (kQHeads - 1 - h);
            swapped = std::max(swapped, std::fabs(double(stock[a]) - double(stock[b])));
        }
    }
    check(swapped > 1e-4, "heads are indistinguishable; the comparison cannot detect a wrong order");

    std::fprintf(stderr,
                 "contrast(%d span): max|diff| = %.3e, worst index %d, |stock|max = %.3e, "
                 "head-swap distance = %.3e\n",
                 n_spans, max_abs, worst, magnitude, swapped);
}

}  // namespace

int main() {
    contrast(1);
    contrast(2);

    if (failures == 0) {
        std::fprintf(stderr, "upstream-attention contrast: OK\n");
        return 0;
    }
    std::fprintf(stderr, "upstream-attention contrast: %d failure(s)\n", failures);
    return 1;
}
