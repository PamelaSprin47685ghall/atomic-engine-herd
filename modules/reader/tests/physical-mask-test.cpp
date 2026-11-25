// The physical emitter's mask path: the first test that ever runs it.
//
// `emit_physical_attention` grew a `graph_inputs::mask` and passes it to `ggml_soft_max_ext`, which is
// the operator `build_attn_mha` gives upstream's `kq_mask` to. Until this file existed, that pointer
// had been read by the compiler and by nothing else -- so "the mask is honoured" was a claim about a
// line of code, not about a number.
//
// What makes the mask's SEMANTICS worth asserting rather than assuming:
//
//   * It is a BIAS, not a boolean. A finite value is a sliding-window or ALiBi term, so a test that
//     only checked `-inf` would pass on an implementation that discarded finite values.
//   * It CANNOT WIDEN. The emitter fills the full cache length with `-inf` and writes the named cells
//     over it, so a position no span names keeps its `-inf`; adding a finite bias to `-inf` is still
//     `-inf`. That is a real limitation of this design, and the last case asserts it as a limitation
//     rather than leaving a reader to discover it. If it ever becomes false, that case fails and the
//     reader is told the design changed.
//   * Its axis is CACHE POSITION, which is the axis this emitter's softmax runs over -- the reason
//     the physical form can consume a mask at all, and the reason the compact one refuses.
//
// Every numeric assertion is against a reference written from that description, not from the graph.
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

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int kHeadDim = 4;
constexpr int kKvHeads = 2;
constexpr int kQHeads  = 4;   // group 2
constexpr int kCell    = 8;
constexpr float kScale = 0.5f;

// Three named cells, non-monotonic, with five positions left unseen by the view.
tq::reader::query_view the_view() {
    tq::reader::query_view v;
    tq::reader::span_keys a;
    a.effective_pos = 5;
    a.cells = {6, 2};
    tq::reader::span_keys b;
    b.effective_pos = 5;
    b.cells = {4};
    v.spans.push_back(a);
    v.spans.push_back(b);
    return v;
}

struct fixture {
    std::vector<float> k, v, q;   // k,v: [hd, kvh, cell]   q: [hd, 1, nh]
    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        unsigned s = 0x51ed270bu;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
    }
    float K(int d, int kvh, int cell) const {
        return k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * cell)];
    }
    float V(int d, int kvh, int cell) const {
        return v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * cell)];
    }
    float Q(int d, int h) const { return q[std::size_t(d) + kHeadDim * h]; }
};

// Softmax over the FULL cache with the given bias per position; unnamed positions hold -inf and a
// finite bias there cannot revive them (-inf + finite == -inf), which is exactly what the emitter's
// fill-then-scatter does. Returns weights [kCell * kQHeads] and out [kHeadDim * kQHeads].
void reference(const fixture & f, const std::vector<int> & cells, const std::vector<float> & bias,
               std::vector<float> & weights, std::vector<float> & out) {
    const int group = kQHeads / kKvHeads;
    weights.assign(std::size_t(kCell) * kQHeads, 0.0f);
    out.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);

    for (int h = 0; h < kQHeads; ++h) {
        const int kvh = h / group;
        std::vector<double> score(kCell, -INFINITY);
        for (int c : cells) {
            double s = 0.0;
            for (int d = 0; d < kHeadDim; ++d) s += double(f.Q(d, h)) * double(f.K(d, kvh, c));
            s = s * double(kScale) + double(bias[std::size_t(c)]);
            // -inf + finite stays -inf: a bias cannot open a position the view never named.
            score[std::size_t(c)] = (s == -INFINITY) ? -INFINITY : s;
        }
        double max_s = -INFINITY;
        for (int c = 0; c < kCell; ++c) if (score[std::size_t(c)] > max_s) max_s = score[std::size_t(c)];
        double sum = 0.0;
        for (int c = 0; c < kCell; ++c) {
            if (!std::isfinite(score[std::size_t(c)])) continue;
            const double e = std::exp(score[std::size_t(c)] - max_s);
            weights[std::size_t(c) + std::size_t(kCell) * h] = float(e);
            sum += e;
        }
        for (int c = 0; c < kCell; ++c) weights[std::size_t(c) + std::size_t(kCell) * h] /= float(sum);
        for (int d = 0; d < kHeadDim; ++d) {
            double acc = 0.0;
            for (int c = 0; c < kCell; ++c) {
                if (weights[std::size_t(c) + std::size_t(kCell) * h] == 0.0f) continue;
                acc += double(weights[std::size_t(c) + std::size_t(kCell) * h]) * double(f.V(d, kvh, c));
            }
            out[std::size_t(d) + std::size_t(kHeadDim) * h] = float(acc);
        }
    }
}

// Runs the emitter with the given mask type and bias. `mask_type == GGML_TYPE_COUNT` means no mask.
bool run(const fixture & f, const std::vector<float> & bias, ggml_type mask_type,
         std::vector<float> & out, std::vector<float> & weights, std::string & why,
         tq::reader::physical_emit_refusal * refusal = nullptr,
         std::int64_t mask_ne0 = -1,
         const tq::reader::query_view * view_override = nullptr) {
    ggml_init_params ip{};
    ip.mem_size = 8u * 1024u * 1024u;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { why = "ggml_init"; return false; }

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCell;
    in.collect_indices = true;
    ggml_set_input(in.k);
    ggml_set_input(in.v);
    ggml_set_input(in.q);

    ggml_tensor * mt = nullptr;
    if (mask_type != GGML_TYPE_COUNT) {
        const std::int64_t ne0 = mask_ne0 > 0 ? mask_ne0 : kCell;
        mt = ggml_new_tensor_4d(ctx, mask_type, ne0, 1, 1, 1);
        ggml_set_input(mt);
        in.mask = mt;
        // The INTENT, stated. These cases want "these cells, filtered by this bias", which is
        // exactly `intersect`; several of them deliberately let the mask leave visible cells
        // outside the view (an all-zero mask over a three-cell view), and under `intersect` that
        // narrowing is the request rather than a discrepancy. The case that supplies a mask
        // WITHOUT stating an intent lives in `unstated_coverage_is_refused`.
        in.coverage = tq::reader::graph_inputs::mask_coverage::intersect;
    }

    const tq::reader::query_view view = view_override ? *view_override : the_view();
    const auto emitted = tq::reader::emit_physical_attention(in, view, kScale);
    if (refusal) *refusal = emitted.refusal;
    if (emitted.refusal != tq::reader::physical_emit_refusal::ok) {
        why = std::string(tq::reader::describe(emitted.refusal)) + " -- " + emitted.why;
        ggml_free(ctx);
        return false;
    }

    // Checked here, before any computation: a mask this wide would make the tensor's own `ne`
    // disagree with the softmax axis, and a silent reinterpretation is the failure mode the
    // emitter's `bad_mask` refusal exists to prevent.
    if (mt != nullptr && mask_ne0 > 0 && mask_ne0 != kCell) {
        why = "emitter accepted a mask of ne0 " + std::to_string(mask_ne0);
        ggml_free(ctx);
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, emitted.out);
    ggml_build_forward_expand(gf, emitted.weights);

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { why = "backend"; ggml_free(ctx); return false; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { why = "alloc"; ggml_backend_free(be); ggml_free(ctx); return false; }

    ggml_backend_tensor_set(in.k, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(in.v, f.v.data(), 0, f.v.size() * sizeof(float));
    ggml_backend_tensor_set(in.q, f.q.data(), 0, f.q.size() * sizeof(float));
    for (const auto & entry : emitted.indices) {
        const std::vector<std::int32_t> * src =
            entry.span < view.spans.size() ? &view.spans[entry.span].cells : &emitted.index_cells;
        if (src->size() < entry.n_cells) { why = "index source too short"; 
            ggml_backend_buffer_free(buf); ggml_backend_free(be); ggml_free(ctx); return false; }
        std::memcpy(entry.index->data, src->data(), entry.n_cells * sizeof(std::int32_t));
    }
    if (mt != nullptr) {
        if (mask_type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(mt, bias.data(), 0, std::size_t(kCell) * sizeof(float));
        } else {
            std::vector<ggml_fp16_t> half(static_cast<std::size_t>(kCell), ggml_fp16_t{});
            ggml_fp32_to_fp16_row(bias.data(), half.data(), std::int64_t(kCell));
            ggml_backend_tensor_set(mt, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        }
    }

    bool ok = false;
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        why = "compute";
    } else {
        out.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        weights.assign(std::size_t(kCell) * kQHeads, 0.0f);
        std::memcpy(out.data(), emitted.out->data, out.size() * sizeof(float));
        std::memcpy(weights.data(), emitted.weights->data, weights.size() * sizeof(float));
        ok = true;
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return ok;
}

void expect_close(const std::vector<float> & got, const std::vector<float> & want, double tol,
                  const std::string & what) {
    check(got.size() == want.size(), what + ": size");
    if (got.size() != want.size()) return;
    double worst = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            check(false, what + ": non-finite at " + std::to_string(i) + " (" +
                             std::to_string(got[i]) + " vs " + std::to_string(want[i]) + ")");
            return;
        }
        worst = std::max(worst, std::fabs(double(got[i]) - double(want[i])));
    }
    check(worst <= tol, what + ": max|diff| = " + std::to_string(worst));
}

// Runs with an explicit view. A thin wrapper rather than a second copy of `run`, because a copy
// would drift: the mask/index plumbing is the part that must stay identical between the two, and it
// is exactly the part the full-coverage case needs to share.
bool run_with_view(const fixture & f, const std::vector<float> & bias, ggml_type mask_type,
                   const tq::reader::query_view & view, std::vector<float> & out,
                   std::vector<float> & weights, std::string & why) {
    return run(f, bias, mask_type, out, weights, why, nullptr, -1, &view);
}

const std::vector<int> kNamed{6, 2, 4};

// An all-zero mask must change nothing. This is the case that proves the mask is USED but not
// INVENTED: an implementation that reshaped or reinterpreted it would move numbers here.
void zero_mask_changes_nothing() {
    const fixture f;
    std::vector<float> base_out, base_w, zero_out, zero_w;
    std::string why;
    if (!run(f, {}, GGML_TYPE_COUNT, base_out, base_w, why)) { check(false, "no mask: " + why); return; }
    const std::vector<float> zeros(std::size_t(kCell), 0.0f);
    if (!run(f, zeros, GGML_TYPE_F32, zero_out, zero_w, why)) { check(false, "zero mask: " + why); return; }
    check(base_out == zero_out, "an all-zero mask leaves `out` bit-identical");
    check(base_w == zero_w, "and leaves `weights` bit-identical");

    std::vector<float> want_w, want_out;
    reference(f, kNamed, zeros, want_w, want_out);
    expect_close(base_out, want_out, 1e-6, "no-mask out matches the reference");
    expect_close(base_w, want_w, 1e-6, "no-mask weights match the reference");
}

// A finite bias must MOVE the weights the way the definition says. Without this, an implementation
// that treated any non-zero mask as "hidden" would still pass `zero_mask_changes_nothing`.
void finite_bias_shifts_the_distribution() {
    const fixture f;
    std::vector<float> bias(std::size_t(kCell), 0.0f);
    bias[4] = 1.5f;    // a named cell, strongly preferred
    bias[2] = -0.75f;  // another named cell, discouraged

    std::vector<float> got_out, got_w, want_w, want_out, plain_w, plain_out;
    std::string why;
    if (!run(f, bias, GGML_TYPE_F32, got_out, got_w, why)) { check(false, "biased: " + why); return; }
    reference(f, kNamed, bias, want_w, want_out);
    expect_close(got_w, want_w, 1e-6, "biased weights match the reference");
    expect_close(got_out, want_out, 1e-6, "biased out matches the reference");

    const std::vector<float> zeros(std::size_t(kCell), 0.0f);
    if (!run(f, zeros, GGML_TYPE_F32, plain_out, plain_w, why)) { check(false, "plain: " + why); return; }

    // Non-vacuity: the bias must actually do something, and cell 4 must gain weight.
    double moved = 0.0;
    for (std::size_t i = 0; i < got_w.size(); ++i) moved = std::max(moved, std::fabs(double(got_w[i]) - double(plain_w[i])));
    check(moved > 1e-3, "a finite bias changes the weights by " + std::to_string(moved));
    check(got_w[4] > plain_w[4], "cell 4 gains weight with a positive bias (" +
                                     std::to_string(got_w[4]) + " > " + std::to_string(plain_w[4]) + ")");
    check(got_w[2] < plain_w[2], "cell 2 loses weight with a negative bias");
}

// -inf must hide a named cell exactly, i.e. its weight becomes exactly zero and the rest renormalises.
void negative_infinity_hides_a_named_cell() {
    const fixture f;
    std::vector<float> bias(std::size_t(kCell), 0.0f);
    bias[4] = -INFINITY;

    std::vector<float> got_out, got_w, want_w, want_out;
    std::string why;
    if (!run(f, bias, GGML_TYPE_F32, got_out, got_w, why)) { check(false, "hidden: " + why); return; }
    check(got_w[4] == 0.0f, "a -inf mask gives cell 4 exactly zero weight, got " +
                                std::to_string(got_w[4]));
    reference(f, kNamed, bias, want_w, want_out);
    expect_close(got_w, want_w, 1e-6, "hidden-cell weights match the reference");
    expect_close(got_out, want_out, 1e-6, "hidden-cell out matches the reference");
    check(got_w[6] > 0.0f && got_w[2] > 0.0f, "the remaining cells still carry the mass");
}

// F16 is the type upstream's `kq_mask` actually has (measured type 1), so the same numbers must
// survive that width.
void f16_mask_agrees_with_f32() {
    const fixture f;
    std::vector<float> bias(std::size_t(kCell), 0.0f);
    bias[4] = 1.5f;
    bias[2] = -0.75f;

    std::vector<float> f32_out, f32_w, f16_out, f16_w;
    std::string why;
    if (!run(f, bias, GGML_TYPE_F32, f32_out, f32_w, why)) { check(false, "f32 mask: " + why); return; }
    if (!run(f, bias, GGML_TYPE_F16, f16_out, f16_w, why)) { check(false, "f16 mask: " + why); return; }

    double worst = 0.0;
    for (std::size_t i = 0; i < f32_out.size(); ++i)
        worst = std::max(worst, std::fabs(double(f16_out[i]) - double(f32_out[i])));
    check(worst < 1e-4, "F16 and F32 masks agree: max|diff| = " + std::to_string(worst));

    std::vector<float> want_w, want_out;
    reference(f, kNamed, bias, want_w, want_out);
    expect_close(f16_w, want_w, 1e-4, "F16 mask weights match the reference");
}

// THE LIMITATION, asserted rather than discovered. A finite bias on a position the view never names
// cannot make it visible: the emitter filled it with -inf, and -inf + finite is -inf. If this ever
// passes as "visible", the emitter's construction changed and every reader of `mask` needs to know.
void a_bias_cannot_open_an_unnamed_position() {
    const fixture f;
    std::vector<float> bias(std::size_t(kCell), 0.0f);
    bias[0] = 100.0f;   // position 0 is NOT in the view
    bias[7] = 100.0f;   // nor is 7

    std::vector<float> got_out, got_w, want_w, want_out;
    std::string why;
    if (!run(f, bias, GGML_TYPE_F32, got_out, got_w, why)) { check(false, "open attempt: " + why); return; }
    check(got_w[0] == 0.0f, "a +100 bias on an unnamed position leaves weight 0, got " +
                                std::to_string(got_w[0]));
    check(got_w[7] == 0.0f, "and the same for position 7, got " + std::to_string(got_w[7]));
    reference(f, kNamed, bias, want_w, want_out);
    expect_close(got_w, want_w, 1e-6, "unnamed-position weights match the reference");
    expect_close(got_out, want_out, 1e-6, "and so does the output");
}

// The refusals, named. A mask that cannot be honoured must not be broadcast into agreement.
void bad_masks_are_refused() {
    const fixture f;
    const std::vector<float> zeros(std::size_t(kCell), 0.0f);
    std::vector<float> a, b;
    std::string why;

    // Wrong column count: a mask over 4 positions cannot index 8 cache columns.
    tq::reader::physical_emit_refusal r = tq::reader::physical_emit_refusal::ok;
    check(!run(f, zeros, GGML_TYPE_F32, a, b, why, &r, 4),
          "a mask with ne0 4 is refused, but the emitter accepted it");
    check(r == tq::reader::physical_emit_refusal::bad_mask,
          std::string("refusal is bad_mask, got ") + tq::reader::describe(r));

    // Wrong type: I32 is an index, not a bias.
    r = tq::reader::physical_emit_refusal::ok;
    check(!run(f, zeros, GGML_TYPE_I32, a, b, why, &r),
          "an I32 mask is refused, but the emitter accepted it");
    check(r == tq::reader::physical_emit_refusal::bad_mask,
          std::string("I32 refusal is bad_mask, got ") + tq::reader::describe(r));
}

// Two mask shapes that a broadcast-tolerant reading would accept and this emitter must reject, each
// checked against the cost of the refusal as well as its name.
//
//   * A STRIDED mask. The softmax kernel walks `mp[i]` over `ne00` elements and never consults
//     nb[0], so a transposed view is read from the wrong bytes with no error at all -- the worst
//     outcome available, and one that no numeric tolerance can catch because the numbers are simply
//     some other row's.
//   * A head count that does not DIVIDE n_head. ggml broadcasts by `i02 % ne12` (softmax kernel,
//     ggml-cpu/ops.cpp:5876), so 3 rows over 4 heads is not a shape ggml rejects; it is a shape
//     where heads 0 and 3 both read row 0 and head 1 reads row 1. An interleaved bias is not what
//     "per-head bias" says, and silently reinterpreting it is the failure this refusal prevents.
//
// Both are also checked for the arena contract: a refusal must leave the scratch context exactly as
// it found it. This emitter publishes an admission bound measured in OBJECTS, so nodes created by a
// refused call are residue that makes the bound wrong -- the reason the validation now runs before
// the first `ggml_new_*` rather than beside the softmax.
void refusals_cost_nothing_and_name_the_shape() {
    const auto scratch = [](std::int64_t ne0, std::int64_t ne1, std::int64_t ne2,
                            ggml_tensor ** mask_out, ggml_context ** ctx_out) {
        ggml_init_params ip{};
        ip.mem_size = 4u * 1024u * 1024u;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        *ctx_out = ctx;
        *mask_out = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, 1);
        return ctx != nullptr;
    };

    // (a) A strided mask: transposing a [kCell, 2] tensor yields [2, kCell] with nb[0] = 8 bytes.
    {
        ggml_context * ctx = nullptr;
        ggml_tensor * ignored = nullptr;
        if (!scratch(kCell, 2, 1, &ignored, &ctx)) { check(false, "scratch for the strided mask"); return; }
        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim;
        in.n_cell = kCell;
        in.mask = ggml_transpose(ctx, ignored);   // [2, kCell]: ne0 != n_cell AND nb[0] != 4
        check(!ggml_is_contiguous(in.mask), "the fixture's mask really is strided");

        const std::size_t before = ggml_used_mem(ctx);
        const auto r = tq::reader::emit_physical_attention(in, the_view(), kScale);
        const std::size_t after = ggml_used_mem(ctx);
        check(r.refusal == tq::reader::physical_emit_refusal::bad_mask,
              std::string("a strided mask is refused as bad_mask, got ") +
                  tq::reader::describe(r.refusal) + " (" + r.why + ")");
        check(after == before, "the refusal created " + std::to_string(after - before) +
                                   " bytes of nodes; a refusal must cost nothing");
        ggml_free(ctx);
    }

    // (b) Three head rows over four heads: ggml would broadcast it, and this emitter will not.
    {
        ggml_context * ctx = nullptr;
        ggml_tensor * mask = nullptr;
        if (!scratch(kCell, 1, 3, &mask, &ctx)) { check(false, "scratch for the 3-row mask"); return; }
        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim;
        in.n_cell = kCell;
        in.mask = mask;

        const std::size_t before = ggml_used_mem(ctx);
        const auto r = tq::reader::emit_physical_attention(in, the_view(), kScale);
        const std::size_t after = ggml_used_mem(ctx);
        check(r.refusal == tq::reader::physical_emit_refusal::bad_mask,
              std::string("3 head rows over 4 heads is refused as bad_mask, got ") +
                  tq::reader::describe(r.refusal) + " (" + r.why + ")");
        check(after == before, "the head-count refusal also created no nodes");
        ggml_free(ctx);
    }
}

// Two refusals that had never been observed: the compact emitter rejecting a mask at all, and the
// physical emitter rejecting a mask with more than one head row.
//
// The compact one matters because ignoring the pointer was the behaviour before the check existed:
// nothing in `graph_emit.cpp` read `graph_inputs::mask`, so a caller that supplied a visibility rule
// got the result of a different configuration, reported as success. A refusal that only a compiler
// has seen is not yet a refusal.
//
// The head-row case is the counterpart: it used to be ACCEPTED for any divisor of n_head, on the
// reading that ggml broadcasts by `i02 % ne12`. It does -- and that makes `ne2 = 2` mean "heads 0
// and 2 read row 0, heads 1 and 3 read row 1": an even/odd INTERLEAVE, which is neither a per-head
// bias nor the GQA grouping. (An earlier version of this comment called it a block split, "heads
// 0,1 -> row 0"; `%` interleaves rather than partitioning, so that description was wrong.) Rather
// than document a second interpretation, one shared row is now the only shape.
void compact_refuses_a_mask_and_head_rows_are_single() {
    // (a) The compact emitter, with a mask it cannot honour.
    {
        ggml_init_params ip{};
        ip.mem_size = 4u * 1024u * 1024u;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (ctx == nullptr) { check(false, "compact: ggml_init"); return; }

        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        in.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, 1, 1, 1);
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim;
        in.n_cell = kCell;

        const std::size_t before = ggml_used_mem(ctx);
        const auto r = tq::reader::emit_reader_attention(in, the_view(), kScale);
        const std::size_t after = ggml_used_mem(ctx);

        check(r.refusal == tq::reader::emit_refusal::mask_not_supported,
              std::string("the compact emitter refuses a mask, got ") +
                  tq::reader::describe(r.refusal) + " (" + r.why + ")");
        check(!r.why.empty(), "and says which emitter can honour it");
        check(after == before, "the compact refusal created " + std::to_string(after - before) +
                                   " bytes; a refusal must cost nothing");
        ggml_free(ctx);
    }

    // (b) Two head rows, which used to pass as a divisor of four.
    {
        ggml_init_params ip{};
        ip.mem_size = 4u * 1024u * 1024u;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (ctx == nullptr) { check(false, "ne2: ggml_init"); return; }

        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        in.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, 1, 2, 1);
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim;
        in.n_cell = kCell;

        const std::size_t before = ggml_used_mem(ctx);
        const auto r = tq::reader::emit_physical_attention(in, the_view(), kScale);
        const std::size_t after = ggml_used_mem(ctx);

        check(r.refusal == tq::reader::physical_emit_refusal::bad_mask,
              std::string("two head rows are refused, got ") +
                  tq::reader::describe(r.refusal) + " (" + r.why + ")");
        check(after == before, "the head-row refusal created no nodes");
        ggml_free(ctx);
    }
}

// The mask-coverage contract, in both directions: the emitter refuses to assume it, and the helper
// answers it when the values are readable.
//
// These two belong together because they are one decision seen from two sides. The emitter cannot
// look at the mask's values (they do not exist at build time), so it requires the caller to state
// the claim; the helper is how a caller earns the right to state it, at the one moment the numbers
// exist. A test of only the refusal would leave the helper unverified; a test of only the helper
// would leave the emitter free to proceed on an unstated claim.
void unstated_coverage_is_refused() {
    ggml_init_params ip{};
    ip.mem_size = 4u * 1024u * 1024u;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { check(false, "coverage: ggml_init"); return; }

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    in.mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, 1, 1, 1);
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCell;
    // `in.coverage` deliberately left at its default. The mask is VALID in shape and type, so the
    // only thing that can refuse it is the unstated claim -- which is what makes this case about
    // the claim and nothing else.
    check(in.coverage == tq::reader::graph_inputs::mask_coverage::unstated,
          "the default coverage is `unstated`; if the default were `intersect`, nothing would be asked");

    const std::size_t before = ggml_used_mem(ctx);
    const auto r = tq::reader::emit_physical_attention(in, the_view(), kScale);
    const std::size_t after = ggml_used_mem(ctx);

    check(r.refusal == tq::reader::physical_emit_refusal::bad_mask,
          std::string("a mask with unstated coverage is refused, got ") +
              tq::reader::describe(r.refusal));
    check(r.why.find("coverage") != std::string::npos,
          "the refusal names the missing claim, so a reader is not sent hunting: " + r.why);
    check(after == before, "the coverage refusal created " + std::to_string(after - before) +
                               " bytes of nodes; a refusal must cost nothing");
    ggml_free(ctx);
}

void coverage_check_answers_the_question() {
    const std::size_t n = std::size_t(kCell);

    // (a) Every position invisible: nothing to cover, so the view trivially covers it.
    {
        std::vector<float> mask(n, -INFINITY);
        const auto r = tq::reader::view_covers_finite_mask(the_view(), mask.data(), n);
        check(r.complete, "an all -inf mask needs no coverage");
        check(r.finite_positions == 0 && r.named == 0, "and reports no visible positions");
    }

    // (b) A finite value on a cell the view names (6, 2, 4 are named): covered.
    {
        std::vector<float> mask(n, -INFINITY);
        mask[6] = 0.5f;
        mask[2] = -0.75f;
        const auto r = tq::reader::view_covers_finite_mask(the_view(), mask.data(), n);
        check(r.complete, "finite values on named cells are covered");
        check(r.finite_positions == 2 && r.named == 2,
              "two visible, two named, got " + std::to_string(r.finite_positions) + "/" +
                  std::to_string(r.named));
    }

    // (c) A finite value on a cell the view does NOT name (0): this is the case that would otherwise
    //     produce a narrower attention reported as the caller's own. It must be caught, and the
    //     position reported so the caller can fix the view rather than guess.
    {
        std::vector<float> mask(n, -INFINITY);
        mask[0] = 0.5f;
        const auto r = tq::reader::view_covers_finite_mask(the_view(), mask.data(), n);
        check(!r.complete, "a finite value outside the view is NOT covered");
        check(r.first_uncovered == 0, "and the uncovered position is reported as 0, got " +
                                          std::to_string(r.first_uncovered));
    }

    // (d) +inf and NaN. Neither is a bias the softmax can use; both are refusals rather than
    //     "visible enough", because a mask no reference can reproduce must not be certified.
    {
        std::vector<float> plus_inf(n, -INFINITY);
        plus_inf[6] = INFINITY;
        check(!tq::reader::view_covers_finite_mask(the_view(), plus_inf.data(), n).complete,
              "+inf is refused even on a named cell");

        std::vector<float> nan_mask(n, -INFINITY);
        nan_mask[6] = std::nanf("");
        check(!tq::reader::view_covers_finite_mask(the_view(), nan_mask.data(), n).complete,
              "NaN is refused even on a named cell");
    }

    // (e) A null mask: the question does not APPLY, and the two fields now say that separately.
    //     The previous assertion was `complete == true`, which read as "the view covers the mask"
    //     while there was no mask at all -- true, and meaningless. A caller branching on `complete`
    //     alone would then believe coverage had been verified without anything being examined. So
    //     `applicable` is false AND `complete` is false: there is no answer, and the absence of an
    //     answer cannot be mistaken for a pass.
    {
        const auto r = tq::reader::view_covers_finite_mask(the_view(), nullptr, n);
        check(!r.applicable, "a null mask makes the coverage question inapplicable");
        check(!r.complete,
              "and `complete` is false -- a vacuous true here is what a caller mistakes for a pass");
        check(r.finite_positions == 0 && r.named == 0, "with nothing counted");
    }

    // (f) An EMPTY cache with a mask that IS given. This is the case the code and its comment
    //     disagreed about: `applicable` was set and `complete` was left at its default `false`,
    //     while the comment claimed "vacuously complete". Neither answer is arbitrary here --
    //     there is no position, so no finite bias can be missing from a view, so coverage holds.
    //     The distinction that matters is the one from case (e): `applicable` tells a caller the
    //     question was asked, `complete` answers it. An empty domain answers "yes" rather than
    //     "unknown", and a caller is not sent to widen a view that has nothing in it.
    {
        const float one[] = {-INFINITY};
        const auto r = tq::reader::view_covers_finite_mask(the_view(), one, 0);
        check(r.applicable, "an empty cache with a mask still asks the question");
        check(r.complete, "and coverage holds vacuously -- there is no position to miss");
        check(r.finite_positions == 0 && r.named == 0, "with nothing counted either");
    }

    // (g) The two ways of having nothing to say stay distinguishable: no mask (inapplicable)
    //     versus an empty domain (applicable and complete). A caller that meant one and got the
    //     other will branch wrongly, so the pair is asserted together.
    {
        const float unused[1] = {0.0f};   // never read: the domain is empty
        check(!tq::reader::view_covers_finite_mask(the_view(), nullptr, n).applicable,
              "no mask is inapplicable, not complete");
        check(tq::reader::view_covers_finite_mask(the_view(), unused, 0).applicable,
              "an empty domain is applicable, not inapplicable");
    }
}

// THE VIEW THAT MAKES THE MASK SUFFICIENT, and the only one whose coverage holds without reading
// the mask's values.
//
// The emitter can only NARROW: it fills the cache with `-inf` and writes the named cells over it.
// So the coverage question -- "does the view name every position the mask leaves visible?" -- is
// unanswerable for a sparse view when the values do not exist yet. `full_coverage_view` sidesteps
// it by naming every position, after which `view ∩ mask == mask` and visibility is decided by the
// mask alone, exactly as upstream decides it. The claim stops being about unknown values and becomes
// a statement about shape.
//
// Both halves are asserted, because either alone is worthless:
//   * a position OUTSIDE a sparse view keeps zero weight even with a finite bias (the limitation);
//   * the SAME bias on the SAME position carries weight under `full_coverage_view` (the fix).
// If only the first held, the fix would not be a fix; if only the second, nothing would show why the
// sparse case is a trap.
void a_full_view_makes_the_mask_sufficient() {
    const fixture f;
    std::vector<float> bias(std::size_t(kCell), 0.0f);
    bias[0] = 1.0f;   // named by neither the sparse view's spans nor ... well, not by the_view()
    bias[7] = 0.5f;

    const tq::reader::query_view full = tq::reader::full_coverage_view(kCell);
    // Guarded, not assumed: the three assertions below index `spans[0]`, and a view that came back
    // empty would have made them read past the end of the vector rather than fail. The size check
    // is not a formality here -- it is the difference between a red test and undefined behaviour.
    check(full.spans.size() == 1, "a full view is one span, got " +
                                      std::to_string(full.spans.size()));
    if (full.spans.size() != 1) return;
    check(full.spans[0].cells.size() == std::size_t(kCell),
          "naming every cache position, got " + std::to_string(full.spans[0].cells.size()));
    check(full.spans[0].cells.front() == 0 && full.spans[0].cells.back() == std::int32_t(kCell) - 1,
          "ascending from 0 to n_cell - 1, so the softmax axis is the cache axis");

    // The coverage question, answered by shape alone.
    const auto report = tq::reader::view_covers_finite_mask(full, bias.data(), std::size_t(kCell));
    check(report.applicable && report.complete,
          "a full view covers every finite bias without knowing the values");
    check(report.named == report.finite_positions,
          "and reports " + std::to_string(report.named) + " of " +
              std::to_string(report.finite_positions) + " visible positions as named");

    // Sparse view: the same bias on positions 0 and 7 carries no weight.
    std::vector<float> sparse_out, sparse_w, full_out, full_w;
    std::string why;
    if (!run(f, bias, GGML_TYPE_F32, sparse_out, sparse_w, why)) {
        check(false, "sparse with bias: " + why);
        return;
    }
    check(sparse_w[0] == 0.0f && sparse_w[7] == 0.0f,
          "under the sparse view a finite bias on an unnamed position keeps weight 0");

    // Full view: the same numbers, and now those positions contribute.
    const bool ok = run_with_view(f, bias, GGML_TYPE_F32, full, full_out, full_w, why);
    if (!ok) { check(false, "full view with bias: " + why); return; }
    check(full_w[0] > 0.0f && full_w[7] > 0.0f,
          "under the full view the same bias carries weight (0 -> " + std::to_string(full_w[0]) +
              ", 7 -> " + std::to_string(full_w[7]) + ")");

    // And it agrees with the definition over the whole cache, which is the property that makes the
    // full view a substitute for upstream's own mask handling rather than merely a different answer.
    std::vector<int> all_cells;
    for (int c = 0; c < kCell; ++c) all_cells.push_back(c);
    std::vector<float> want_w, want_out;
    reference(f, all_cells, bias, want_w, want_out);
    expect_close(full_w, want_w, 1e-6, "full-view weights match the definition over the cache");
    expect_close(full_out, want_out, 1e-6, "full-view out matches the definition");
}

// The boundaries of `full_coverage_view`, and the precondition that no check can enforce.
//
//   * `n_cell == 0` must give a view with NO spans. A single empty span would be accepted by the
//     emitter's shape checks and then produce a gather over no cells -- and `spans[0]` would exist,
//     so a caller reading it would get a legal-looking empty list instead of an obvious absence.
//   * `n_cell == 1` must give exactly one cell, `0`. The degenerate case is where an off-by-one
//     (`i <= n_cell`) hides: it would name cell 1, which is outside a one-cell cache.
//   * `effective_pos` is 0 on every span, and that is INERT only for the emitter that does not
//     rotate. Asserting the value documents the contract for the emitter that does, where it would
//     mean "rotate q to position 0" and be wrong. Nothing can enforce the distinction at runtime,
//     so the test pins the number that a reader will see and the comment says what it may be used
//     with.
void full_view_boundaries_and_its_rotation_precondition() {
    {
        const tq::reader::query_view none = tq::reader::full_coverage_view(0);
        check(none.spans.empty(), "n_cell 0 gives no span at all, got " +
                                      std::to_string(none.spans.size()));

        const tq::reader::query_view negative = tq::reader::full_coverage_view(-4);
        check(negative.spans.empty(), "a negative count gives no span");
    }
    {
        const tq::reader::query_view one = tq::reader::full_coverage_view(1);
        check(one.spans.size() == 1, "n_cell 1 is one span");
        if (one.spans.size() != 1) return;
        check(one.spans[0].cells == std::vector<std::int32_t>({0}),
              "naming cell 0 only; `i <= n_cell` would name cell 1, outside a one-cell cache");
    }
    {
        // A non-trivial size, checked cell by cell rather than by front/back alone: a view that
        // started at 0, ended at n-1 and skipped the middle would pass those two assertions.
        const tq::reader::query_view v = tq::reader::full_coverage_view(5);
        check(v.spans.size() == 1 && v.spans[0].cells.size() == 5, "five cells for n_cell 5");
        if (v.spans.size() != 1 || v.spans[0].cells.size() != 5) return;
        for (std::int32_t i = 0; i < 5; ++i) {
            check(v.spans[0].cells[std::size_t(i)] == i,
                  "cell " + std::to_string(i) + " is " + std::to_string(i) + ", got " +
                      std::to_string(v.spans[0].cells[std::size_t(i)]));
        }
        check(v.spans[0].effective_pos == 0,
              "effective_pos is 0: inert for `emit_physical_attention` (which rotates nothing and "
              "takes an already-rotated q), and NOT usable with an emitter that rotates q per span");
    }
}

// THE PER-ROW FORM: what the product actually calls, where the mask arrives F16 and one row per
// token. Asserted through the same entry point production uses, so a broken widening or a wrong
// element width shows up here rather than as a plausible number in a graph.
void row_form_agrees_and_reads_from_the_row_it_is_given() {
    const std::size_t n = std::size_t(kCell);

    // IEEE 754 binary16, which is what `ggml_fp16_t` holds bit for bit. Written as bit patterns
    // rather than converted, so an error in the widening cannot hide behind a value that was never
    // what the mask said.
    const std::uint16_t f16_neg_inf = 0xFC00u;
    const std::uint16_t f16_zero    = 0x0000u;
    const std::uint16_t f16_pos_inf = 0x7C00u;
    const std::uint16_t f16_nan     = 0x7E00u;

    // (a) The same three verdicts as the F32 path, on the same cells.
    {
        std::vector<std::uint16_t> row(n, f16_neg_inf);
        auto r = tq::reader::view_covers_finite_mask_row(the_view(), row.data(),
                                                         tq::reader::mask_width::f16, n);
        check(r.applicable && r.complete, "an all -inf F16 row needs no coverage");
        check(r.finite_positions == 0 && r.named == 0, "and reports no visible positions");

        row[6] = f16_zero;
        row[2] = f16_zero;
        r = tq::reader::view_covers_finite_mask_row(the_view(), row.data(),
                                                    tq::reader::mask_width::f16, n);
        check(r.complete, "finite values in F16 on named cells are covered");
        check(r.finite_positions == 2 && r.named == 2,
              "two visible, two named in F16, got " + std::to_string(r.finite_positions) + "/" +
                  std::to_string(r.named));

        row[0] = f16_zero;   // cell 0 is NOT named by the_view(): the case the emitter cannot see
        r = tq::reader::view_covers_finite_mask_row(the_view(), row.data(),
                                                    tq::reader::mask_width::f16, n);
        check(!r.complete, "a finite F16 value outside the view is NOT covered");
        check(r.first_uncovered == 0, "and the uncovered position is reported as 0, got " +
                                          std::to_string(r.first_uncovered));
    }

    // (b) +inf and NaN in F16 are refusals, not "visible enough". A widening that turned a
    //     non-finite F16 into a finite F32 would make both of these PASS, which is exactly why they
    //     are asserted here instead of only on the F32 path.
    {
        std::vector<std::uint16_t> row(n, f16_neg_inf);
        row[6] = f16_pos_inf;
        check(!tq::reader::view_covers_finite_mask_row(the_view(), row.data(),
                                                       tq::reader::mask_width::f16, n).complete,
              "F16 +inf is refused even on a named cell");
        row[6] = f16_nan;
        check(!tq::reader::view_covers_finite_mask_row(the_view(), row.data(),
                                                       tq::reader::mask_width::f16, n).complete,
              "F16 NaN is refused even on a named cell");
    }

    // (c) THE ROW START IS THE CALLER'S. Two rows in one buffer: row 0 all-invisible, row 1 visible
    //     on a cell the view does not name. Reading the pointer's first row blindly would certify
    //     both; reading the row that was handed over reports the violation. This is the arithmetic
    //     the prefill path does with `t * n_cell * elem`; only that multiplication lives outside
    //     this function, and this case pins what the function must do with the result.
    {
        std::vector<std::uint16_t> rows(2 * n, f16_neg_inf);
        rows[n] = f16_zero;   // row 1, cell 0: outside the_view()

        check(tq::reader::view_covers_finite_mask_row(the_view(), rows.data(),
                                                      tq::reader::mask_width::f16, n).complete,
              "row 0 is all -inf and covers");
        const auto r1 = tq::reader::view_covers_finite_mask_row(
            the_view(), rows.data() + n, tq::reader::mask_width::f16, n);
        check(!r1.complete, "row 1 has a finite value outside the view and does not cover");
        check(r1.first_uncovered == 0,
              "and the row's own position 0 is reported, got " + std::to_string(r1.first_uncovered));
    }

    // (d) The three cases stay distinguishable through the row form: no row (inapplicable), and an
    //     empty domain with a row (applicable and complete). The empty case is answered by hand
    //     because a zero-length widened buffer's `data()` may be null, and passing that on would
    //     silently answer the "no mask" question instead of this one.
    {
        const auto none = tq::reader::view_covers_finite_mask_row(the_view(), nullptr,
                                                                  tq::reader::mask_width::f16, n);
        check(!none.applicable && !none.complete, "a null row is inapplicable, not complete");

        const std::uint16_t one[1] = {f16_zero};   // never read: the domain is empty
        const auto empty = tq::reader::view_covers_finite_mask_row(
            the_view(), one, tq::reader::mask_width::f16, 0);
        check(empty.applicable, "an empty domain with a row still asks the question");
        check(empty.complete, "and coverage holds vacuously");
        check(empty.finite_positions == 0 && empty.named == 0, "with nothing counted");
    }
}

// THE ROW STRIDE: the last arithmetic between the product and the tested predicate. Prefill's
// `kq_mask` is `[n_cell, n_tokens, 1, 1]`, so token `t` starts at `t * n_cell * sizeof(element)`. A
// stride computed with the wrong element width, or a product that dropped `t`, reads one token's mask
// as another's and reports a confident verdict about the wrong bytes. Both are pinned here, and then
// the result is driven through the row form, so the composition production runs is what this test
// runs.
void the_row_stride_follows_the_width_and_the_row_index() {
    const std::size_t n = std::size_t(kCell);

    check(tq::reader::mask_element_bytes(tq::reader::mask_width::f16) == sizeof(std::uint16_t),
          "an F16 mask element is two bytes");
    check(tq::reader::mask_element_bytes(tq::reader::mask_width::f32) == sizeof(float),
          "an F32 mask element is four bytes");

    // (a) F32: three rows, only the last one violates. The pointer for each `t` is compared against
    //     the byte offset the caller would compute by hand, so a stride that used the wrong width or
    //     ignored `t` fails here rather than as a plausible number in a graph.
    {
        std::vector<float> rows(3 * n, -INFINITY);
        rows[2 * n] = 0.0f;   // row 2, cell 0: outside the_view()

        const auto * base = static_cast<const void *>(rows.data());
        check(tq::reader::finite_mask_row(base, 0, n, tq::reader::mask_width::f32) == base,
              "row 0 starts at the base pointer");
        check(tq::reader::finite_mask_row(base, 2, n, tq::reader::mask_width::f32) ==
                  static_cast<const void *>(rows.data() + 2 * n),
              "row 2 of an F32 mask starts 2*n floats in");

        for (std::size_t t = 0; t < 3; ++t) {
            const auto cov = tq::reader::view_covers_finite_mask_row(
                the_view(), tq::reader::finite_mask_row(base, t, n, tq::reader::mask_width::f32),
                tq::reader::mask_width::f32, n);
            check(cov.complete == (t < 2),
                  "F32 row " + std::to_string(t) + (t < 2 ? " covers" : " does not cover"));
        }
    }

    // (b) The same three rows as F16: HALF the stride, so reusing the F32 arithmetic would land
    //     inside row 1 while asking for row 2. The byte offsets are compared directly, because
    //     reading the wrong row can leave the verdict unchanged by luck.
    {
        std::vector<std::uint16_t> rows(3 * n, std::uint16_t(0xFC00u));
        rows[2 * n] = std::uint16_t(0x0000u);   // row 2, cell 0: outside the_view()

        const auto * base = static_cast<const void *>(rows.data());
        const auto * row2 = static_cast<const void *>(rows.data() + 2 * n);
        check(tq::reader::finite_mask_row(base, 2, n, tq::reader::mask_width::f16) == row2,
              "row 2 of an F16 mask starts 2*n shorts in");
        check(tq::reader::finite_mask_row(base, 2, n, tq::reader::mask_width::f32) != row2,
              "and the F32 stride does NOT name that row in an F16 buffer");

        const auto cov = tq::reader::view_covers_finite_mask_row(
            the_view(), tq::reader::finite_mask_row(base, 2, n, tq::reader::mask_width::f16),
            tq::reader::mask_width::f16, n);
        check(!cov.complete, "F16 row 2 does not cover");
        check(cov.first_uncovered == 0,
              "and reports its OWN position 0, got " + std::to_string(cov.first_uncovered));
    }

    // (c) The three cases survive the stride: no base stays no mask, and an empty row makes every
    //     `t` the base pointer with the empty domain's answer.
    {
        check(tq::reader::finite_mask_row(nullptr, 3, n, tq::reader::mask_width::f16) == nullptr,
              "a null base has no row 3");
        const std::uint16_t one[1] = {std::uint16_t(0x0000u)};   // never read

        const void * empty_row =
            tq::reader::finite_mask_row(one, 7, 0, tq::reader::mask_width::f16);
        check(empty_row == static_cast<const void *>(one),
              "with n_cell 0 every row index is the base pointer");

        const auto empty = tq::reader::view_covers_finite_mask_row(
            the_view(), empty_row, tq::reader::mask_width::f16, 0);
        check(empty.applicable && empty.complete,
              "an empty domain through the stride asks the question and covers");
    }
}

}  // namespace

int main() {
    the_row_stride_follows_the_width_and_the_row_index();
    row_form_agrees_and_reads_from_the_row_it_is_given();
    a_full_view_makes_the_mask_sufficient();
    full_view_boundaries_and_its_rotation_precondition();
    zero_mask_changes_nothing();
    unstated_coverage_is_refused();
    coverage_check_answers_the_question();
    finite_bias_shifts_the_distribution();
    negative_infinity_hides_a_named_cell();
    f16_mask_agrees_with_f32();
    a_bias_cannot_open_an_unnamed_position();
    bad_masks_are_refused();
    refusals_cost_nothing_and_name_the_shape();
    compact_refuses_a_mask_and_head_rows_are_single();

    if (failures == 0) {
        std::fprintf(stderr, "physical mask: OK\n");
        return 0;
    }
    std::fprintf(stderr, "physical mask: %d failure(s)\n", failures);
    return 1;
}
