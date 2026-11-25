// The `q` stride contract: what the emitter accepts, what it refuses, and the numbers that decide it.
//
// THE HISTORY THIS FILE REPLACES, because both mistakes are worth not repeating.
//
//   1. An earlier revision refused every non-contiguous `q`, on the stated ground that "`mul_mat`
//      assumes a packed right operand". Measurement says that is false: a `[hd,1,nh]` query with
//      `nb[2]` padded 2x and 3x still matches the definition exactly. The kernel walks the strides.
//   2. The test written to guard that refusal was green and worthless: it never wrote any data into
//      the tensors, so "`ggml_cont` is faithful" compared uninitialised memory with uninitialised
//      memory, and zero-filled pages made it pass. It also created a `ggml_cont` node AFTER
//      `ggml_backend_alloc_ctx_tensors`, and in a no_alloc context such a node has no storage -- that
//      was the SIGSEGV that ended the previous run.
//
// So this file does three things: it builds EVERY node before placement, it writes real data
// (including a sentinel in the padding, so a kernel that ignored a stride would read 999 and be
// caught), and it checks the case the previous version skipped -- `n_head > n_head_kv`, where the
// emitter's per-head view has `ne[2] == group > 1` and is therefore the only place a padded `nb[2]`
// could actually reach `mul_mat`.
//
// The one thing NOT asserted here, stated rather than silently omitted: the emitter also refuses a
// `q` whose `nb[0]` is not one element. No legal ggml construction reaches that state with a valid
// shape (`ggml_view_*` does not take `nb[0]`; `ggml_transpose` changes `ne` as well; `ggml_reshape`
// requires contiguity), so it has no test and is defence-in-depth only. Writing a fixture by poking
// `tensor->nb[0]` would assert something about a tensor whose bytes do not mean what its strides say,
// which is exactly the kind of fake fixture this pass removed.
#include <tq/reader/graph_emit_physical.h>
#include <tq/reader/upstream_view.h>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const std::string & what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; }
}

constexpr int kHeadDim = 8, kKvHeads = 2, kHeads = 4, kCell = 16, kTokens = 3;
constexpr float kScale = 0.35355339f;
constexpr float kSentinel = 999.0f;   // padding garbage: a stride-ignoring kernel would read this

struct fixture {
    std::vector<float> k, v, q;   // q: [hd, T, nh], d + hd*(t + T*h)
    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kTokens * kHeads, 0.0f);
        unsigned s = 0x51ed270bu;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
    }
    float K(int d, int kvh, int c) const { return k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float V(int d, int kvh, int c) const { return v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float Q(int d, int t, int h) const { return q[std::size_t(d) + kHeadDim * (t + kTokens * h)]; }
};

// One token, all cells visible, GQA grouping h/(nh/kvh) -- the definition, written independently.
void reference(const fixture & f, int t, std::vector<float> & out) {
    const int group = kHeads / kKvHeads;
    out.assign(std::size_t(kHeadDim) * kHeads, 0.0f);
    for (int h = 0; h < kHeads; ++h) {
        const int kvh = h / group;
        std::vector<double> score(static_cast<std::size_t>(kCell), 0.0);
        double max_s = -INFINITY;
        for (int c = 0; c < kCell; ++c) {
            double s = 0.0;
            for (int d = 0; d < kHeadDim; ++d)
                s += double(f.Q(d, t, h)) * double(f.K(d, kvh, c));
            score[std::size_t(c)] = s * double(kScale);
            if (score[std::size_t(c)] > max_s) max_s = score[std::size_t(c)];
        }
        double sum = 0.0;
        for (double & s : score) { s = std::exp(s - max_s); sum += s; }
        for (int d = 0; d < kHeadDim; ++d) {
            double acc = 0.0;
            for (int c = 0; c < kCell; ++c) acc += (score[std::size_t(c)] / sum) * double(f.V(d, kvh, c));
            out[std::size_t(d) + std::size_t(kHeadDim) * h] = float(acc);
        }
    }
}

bool publish_cb(ggml_tensor * t, const void * d, std::size_t o, std::size_t s) {
    if (t == nullptr || t->buffer == nullptr) return false;
    ggml_backend_tensor_set(t, d, o, s);
    return true;
}

}  // namespace

int main() {
    const fixture f;
    std::vector<float> want;
    reference(f, 0, want);

    ggml_init_params ip{};
    ip.mem_size = 8u * 1024u * 1024u;
    ip.no_alloc = true;   // EVERY node below is created before placement; none after
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { std::fprintf(stderr, "FAIL: ggml_init\n"); return 1; }

    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, 1, 1, 1);
    ggml_set_input(k); ggml_set_input(v); ggml_set_input(mask);

    // Packed references, one F32 and one F16.
    ggml_tensor * q32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads);
    ggml_tensor * q16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, 1, kHeads);
    ggml_set_input(q32); ggml_set_input(q16);

    // Padded views: same numbers, head stride 4x (F32) and 3x (F16) the packed one.
    ggml_tensor * wide32 = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, std::size_t(kHeadDim) * 4 * kHeads);
    ggml_tensor * wide16 = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, std::size_t(kHeadDim) * 3 * kHeads);
    ggml_set_input(wide32); ggml_set_input(wide16);
    ggml_tensor * q32_padded = ggml_view_3d(ctx, wide32, kHeadDim, 1, kHeads,
                                            std::size_t(kHeadDim) * sizeof(float),
                                            std::size_t(kHeadDim) * 4 * sizeof(float), 0);
    ggml_tensor * q16_padded = ggml_view_3d(ctx, wide16, kHeadDim, 1, kHeads,
                                            std::size_t(kHeadDim) * sizeof(ggml_fp16_t),
                                            std::size_t(kHeadDim) * 3 * sizeof(ggml_fp16_t), 0);

    // A `cont` node over the padded view, so its faithfulness can be read back. Created NOW, before
    // placement: a node created afterwards has no storage and the read would fault (the previous
    // revision's crash).
    ggml_tensor * cont32 = ggml_cont(ctx, q32_padded);
    ggml_cgraph * gf_cont = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf_cont, cont32);

    const tq::reader::query_view full = tq::reader::full_coverage_view(kCell);

    auto emit = [&](ggml_tensor * q) {
        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = k; in.v = v; in.q = q; in.mask = mask;
        in.coverage = tq::reader::graph_inputs::mask_coverage::intersect;
        in.n_head = kHeads; in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim; in.n_cell = kCell;
        in.collect_indices = true;
        return tq::reader::emit_physical_attention(in, full, kScale);
    };

    // ---- (0) the stride facts of the fixtures themselves, so the cases below mean what they say ----
    check(q32->nb[2] == kHeadDim * sizeof(float), "the packed F32 query has head stride hd*element");
    check(q32_padded->nb[2] == 4 * std::size_t(kHeadDim) * sizeof(float),
          "the padded F32 query has a head stride four times the packed one");
    check(!ggml_is_contiguous(q32_padded), "the padded F32 query is (correctly) not contiguous");
    check(!ggml_is_contiguous(q16_padded), "the padded F16 query is (correctly) not contiguous");
    check(q32_padded->nb[0] == sizeof(float) && q16_padded->nb[0] == sizeof(ggml_fp16_t),
          "both padded fixtures keep nb[0] == one element, which is the only stride rule the emitter states");

    // ---- (1) emissions. `group = nh/kvh = 2`, so the per-head view has ne[2] == 2: this is the one
    //         shape where a padded nb[2] genuinely reaches mul_mat instead of collapsing to ne[2]==1.
    tq::reader::physical_emit_result em32 = emit(q32);
    tq::reader::physical_emit_result em32p = emit(q32_padded);
    tq::reader::physical_emit_result em16 = emit(q16);
    tq::reader::physical_emit_result em16p = emit(q16_padded);
    for (const auto * pair : {&em32, &em32p, &em16, &em16p}) {
        check(pair->refusal == tq::reader::physical_emit_refusal::ok,
              std::string("emission accepted, got ") + tq::reader::describe(pair->refusal) + " -- " +
                  pair->why);
    }
    if (em32.refusal != tq::reader::physical_emit_refusal::ok ||
        em32p.refusal != tq::reader::physical_emit_refusal::ok ||
        em16.refusal != tq::reader::physical_emit_refusal::ok ||
        em16p.refusal != tq::reader::physical_emit_refusal::ok) {
        ggml_free(ctx);
        return 1;
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = be ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
    if (buf == nullptr) {
        std::fprintf(stderr, "FAIL: placement\n");
        if (buf != nullptr) ggml_backend_buffer_free(buf);
        if (be != nullptr) ggml_backend_free(be);
        ggml_free(ctx);
        return 1;
    }

    // ---- (2) real data, written after placement, padding filled with a sentinel -------------------
    ggml_backend_tensor_set(k, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(v, f.v.data(), 0, f.v.size() * sizeof(float));
    {
        const std::vector<float> zero_bias(std::size_t(kCell), 0.0f);
        ggml_backend_tensor_set(mask, zero_bias.data(), 0, zero_bias.size() * sizeof(float));

        std::vector<float> qv(std::size_t(kHeadDim) * kHeads), half(std::size_t(kHeadDim) * kHeads);
        for (int h = 0; h < kHeads; ++h)
            for (int d = 0; d < kHeadDim; ++d) qv[std::size_t(d + kHeadDim * h)] = f.Q(d, 0, h);
        ggml_backend_tensor_set(q32, qv.data(), 0, qv.size() * sizeof(float));
        {
            std::vector<ggml_fp16_t> hv(qv.size());
            ggml_fp32_to_fp16_row(qv.data(), hv.data(), std::int64_t(qv.size()));
            ggml_backend_tensor_set(q16, hv.data(), 0, hv.size() * sizeof(ggml_fp16_t));
        }

        std::vector<float> w32(std::size_t(kHeadDim) * 4 * kHeads, kSentinel);
        std::vector<ggml_fp16_t> w16(std::size_t(kHeadDim) * 3 * kHeads);
        for (int h = 0; h < kHeads; ++h)
            for (int d = 0; d < kHeadDim; ++d) {
                w32[std::size_t(d) + std::size_t(kHeadDim) * 4 * h] = f.Q(d, 0, h);
                w16[std::size_t(d) + std::size_t(kHeadDim) * 3 * h] =
                    ggml_fp32_to_fp16(f.Q(d, 0, h));
            }
        ggml_backend_tensor_set(wide32, w32.data(), 0, w32.size() * sizeof(float));
        ggml_backend_tensor_set(wide16, w16.data(), 0, w16.size() * sizeof(ggml_fp16_t));
    }

    for (auto * pair : {&em32, &em32p, &em16, &em16p}) {
        std::string why;
        const auto r = tq::reader::publish_index_inputs(*pair, &publish_cb, &why, nullptr);
        check(r == tq::reader::index_publish_refusal::ok,
              std::string("publish: ") + tq::reader::describe(r) + " (" + why + ")");
        if (r != tq::reader::index_publish_refusal::ok) {
            check(false, "not computing: an emission's gather indices were never written");
            ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx);
            return 1;
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, em32.out);
    ggml_build_forward_expand(gf, em32p.out);
    ggml_build_forward_expand(gf, em16.out);
    ggml_build_forward_expand(gf, em16p.out);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        check(false, "compute failed, so no number below was read");
        ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx);
        return 1;
    }

    auto read = [](ggml_tensor * t) {
        std::vector<float> out(std::size_t(kHeadDim) * kHeads, 0.0f);
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
        return out;
    };
    auto compare = [&](const std::vector<float> & got, const char * label, double tol) {
        double worst = 0.0;
        int at = -1;
        bool finite = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) { finite = false; break; }
            const double d = std::fabs(double(got[i]) - double(want[i]));
            if (d > worst) { worst = d; at = int(i); }
        }
        check(finite, std::string(label) + ": finite output");
        check(worst <= tol, std::string(label) + ": matches the definition, max|diff| = " +
                                std::to_string(worst) + " at " + std::to_string(at) +
                                " -- a large difference means the padding sentinel was read, i.e. a "
                                "stride was ignored");
        std::fprintf(stderr, "  %-14s max|diff| = %.3e\n", label, worst);
    };

    // ---- (3) the numbers, with group = 2 -----------------------------------------------------------
    const std::vector<float> o32 = read(em32.out), o32p = read(em32p.out);
    const std::vector<float> o16 = read(em16.out), o16p = read(em16p.out);
    compare(o32, "packed F32", 1e-6);
    compare(o32p, "padded F32", 1e-6);
    compare(o16, "packed F16", 5e-3);
    compare(o16p, "padded F16", 5e-3);
    check(o32 == o32p, "padded and packed F32 agree bit for bit: the head stride is honoured, so "
                       "layout is not part of the result");
    {
        double gap = 0.0;
        for (std::size_t i = 0; i < o16.size(); ++i) gap = std::max(gap, std::fabs(double(o16[i]) - double(o16p[i])));
        check(gap < 5e-3, "padded and packed F16 agree within half precision: max|diff| = " + std::to_string(gap));
    }

    // ---- (4) cont over a padded view is faithful, read element-wise through each tensor's strides ---
    if (ggml_backend_graph_compute(be, gf_cont) != GGML_STATUS_SUCCESS) {
        check(false, "compute of cont(padded q) failed");
    } else {
        int differ = 0;
        std::string first;
        for (int h = 0; h < kHeads; ++h) {
            for (int d = 0; d < kHeadDim; ++d) {
                float src = 0, copy = 0;
                // Element-wise through each tensor's OWN strides. `ggml_backend_tensor_get` on a
                // strided VIEW copies raw bytes and ignores nb -- the mistake that once produced
                // "8 of 16 differ" and a wrong conclusion about cont.
                ggml_backend_tensor_get(q32_padded, &src,
                    std::size_t(d) * q32_padded->nb[0] + std::size_t(h) * q32_padded->nb[2],
                    sizeof(float));
                ggml_backend_tensor_get(cont32, &copy,
                    std::size_t(d) * cont32->nb[0] + std::size_t(h) * cont32->nb[2], sizeof(float));
                if (src != copy) {
                    if (differ == 0) first = "h=" + std::to_string(h) + " d=" + std::to_string(d) +
                                              ": view " + std::to_string(src) + " cont " + std::to_string(copy);
                    ++differ;
                }
            }
        }
        check(differ == 0, "ggml_cont is faithful on the padded view: " + std::to_string(differ) +
                               " of " + std::to_string(kHeadDim * kHeads) + " differ" +
                               (first.empty() ? "" : "; first " + first));
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);

    if (failures == 0) { std::fprintf(stderr, "q stride contract: OK\n"); return 0; }
    std::fprintf(stderr, "q stride contract: %d failure(s)\n", failures);
    return 1;
}
