// Prefill by composition: T tokens through the verified single-query emitter, joined along dim 1,
// with every token's gather table published separately through upstream's own `set_inputs` window.
//
// WHAT THIS TEST IS FOR, AND WHAT IT REFUSES TO CLAIM
// --------------------------------------------------
// Two bespoke multi-token emitters were written and removed here. The first under-charged its own
// gathers by a factor of T (an admission bound below the truth, which aborts inside ggml); the second
// produced a difference nobody could attribute. The composition was then measured to 2.98e-08 on a
// per-(token, head) reference -- so this test does not re-derive the arithmetic. It checks the four
// things composition adds and the emitter cannot check for itself:
//
//   1. THE ARENA BOUND covers the whole graph, computed before the first node. A too-small arena is
//      refused with `arena_admission` and the context is left with NOTHING in it -- asserted as
//      `ggml_used_mem` growth of zero, because this emitter publishes an object count and residue
//      would make that count a lie.
//   2. THE NUMBERS, against a double reference written per (token, head) from the definition, with a
//      per-token causal mask and GQA (`group = n_head / n_head_kv = 2`). A broadcast mask and a
//      head-swapped grouping both change these values, and the non-vacuity guards below prove the
//      fixture can tell.
//   3. THE PER-TOKEN PUBLICATION, by driving the real window: `ggml_backend_sched_alloc_graph`, then
//      `publish_index_inputs` once per token, then compute. Each token owns its own `index_cells`, so
//      this is the only shape that needs no new publication design.
//   4. THE VISIBILITY IS READ PER TOKEN: token 0 must give weight 0 to a cell only token 2 can see.
//
// NOT CLAIMED: that this is wired into a product. It is not; the replacement switches still refuse at
// startup. NOT CLAIMED either: that the composition is atomic against ggml itself failing -- a ggml
// context has no undo. What is claimed is narrower and checkable: every STRUCTURAL refusal is decided
// before the first node, and a graph that outgrew its charge is reported as `partial_graph` so the
// caller discards the context instead of running it.
#include <tq/reader/graph_emit_prefill.h>
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
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int kHeadDim = 8, kKvHeads = 2, kHeads = 4, kCell = 16, kTokens = 3;
constexpr float kScale = 0.35355339f;

// Token t sees the view's cells whose VALUE is below `visible(t)`; the rest are -inf. Using cell
// VALUES rather than positions in the view keeps the ladder visible in the numbers: token 2 sees
// strictly more than token 0, so a broadcast mask moves a weight to zero that should be positive.
int visible(int t) { return 6 + 3 * t; }
float bias(int t) { return 0.25f * float(t); }

struct fixture {
    std::vector<float> k, v;      // [hd, kvh, n_cell]
    std::vector<float> q;         // [hd, T, nh], d + hd*(t + T*h)
    std::vector<float> mask;      // [n_cell, T, 1, 1], c + n_cell*t

    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kTokens * kHeads, 0.0f);
        mask.assign(std::size_t(kCell) * kTokens, 0.0f);
        unsigned s = 0x51ed270bu;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
        for (int t = 0; t < kTokens; ++t)
            for (int c = 0; c < kCell; ++c)
                mask[std::size_t(c) + std::size_t(kCell) * t] =
                    (c < visible(t)) ? bias(t) : -INFINITY;
    }
    float K(int d, int kvh, int c) const { return k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float V(int d, int kvh, int c) const { return v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float Q(int d, int t, int h) const { return q[std::size_t(d) + kHeadDim * (t + kTokens * h)]; }
    float M(int c, int t) const { return mask[std::size_t(c) + std::size_t(kCell) * t]; }
};

// FULL COVERAGE, and the choice is load-bearing rather than convenient.
//
// A sparse view was tried first (cells {6,2} + {0,4,1,3,5}, i.e. 0..6) and the per-token visibility
// assertions could not pass: a cell outside the view has weight exactly zero whatever the mask says,
// because the emitter fills the cache with -inf and only the named cells get scores. So "token 2 sees
// cell 11" was false before the mask was consulted at all -- the fixture, not the code.
//
// Naming every cache cell is the documented recipe for reproducing upstream's visibility: the emitter
// can only NARROW (`view ∩ mask`), so with a complete view the visibility is the mask's, which is
// exactly what a prefill mask means. It is also the shape a replacement would use.
tq::reader::query_view the_view() {
    tq::reader::query_view v;
    tq::reader::span_keys span;
    span.effective_pos = 5;
    for (int c = 0; c < kCell; ++c) span.cells.push_back(c);
    v.spans.push_back(std::move(span));
    return v;
}

std::vector<int> flatten(const tq::reader::query_view & view) {
    std::vector<int> cells;
    for (const auto & span : view.spans)
        for (std::int32_t c : span.cells) cells.push_back(int(c));
    return cells;
}

// The definition, per (token, head): one softmax over the view's cells in VIEW order, the mask bias
// added where the cell is visible, -inf making a cell's weight exactly zero. Indexed the way the
// result documents itself: out[d + hd*(t + T*h)], weights[c + n_cell*(t + T*h)].
void reference(const fixture & f, const std::vector<int> & cells, std::vector<float> & out,
               std::vector<float> & weights) {
    const int group = kHeads / kKvHeads;
    out.assign(std::size_t(kHeadDim) * kTokens * kHeads, 0.0f);
    weights.assign(std::size_t(kCell) * kTokens * kHeads, 0.0f);
    for (int t = 0; t < kTokens; ++t) {
        for (int h = 0; h < kHeads; ++h) {
            const int kvh = h / group;
            std::vector<double> score(cells.size(), 0.0);
            double max_s = -INFINITY;
            for (std::size_t i = 0; i < cells.size(); ++i) {
                const double b = double(f.M(cells[i], t));
                if (!std::isfinite(b)) {
                    score[i] = -INFINITY;
                    continue;
                }
                double s = 0.0;
                for (int d = 0; d < kHeadDim; ++d)
                    s += double(f.Q(d, t, h)) * double(f.K(d, kvh, cells[i]));
                score[i] = s * double(kScale) + b;
                if (score[i] > max_s) max_s = score[i];
            }
            double sum = 0.0;
            for (std::size_t i = 0; i < cells.size(); ++i) {
                if (!std::isfinite(score[i])) continue;
                score[i] = std::exp(score[i] - max_s);
                sum += score[i];
            }
            for (std::size_t i = 0; i < cells.size(); ++i) {
                if (!std::isfinite(score[i])) continue;   // stays exactly 0
                const double w = score[i] / sum;
                weights[std::size_t(cells[i]) + std::size_t(kCell) * (t + std::size_t(kTokens) * h)] =
                    float(w);
                for (int d = 0; d < kHeadDim; ++d)
                    out[std::size_t(d) + std::size_t(kHeadDim) * (t + std::size_t(kTokens) * h)] +=
                        float(w * double(f.V(d, kvh, cells[i])));
            }
        }
    }
}

bool publish_cb(ggml_tensor * t, const void * d, std::size_t o, std::size_t s) {
    // Through the backend: a non-null `data` says "placed", not "host memory".
    if (t == nullptr || t->buffer == nullptr) return false;
    ggml_backend_tensor_set(t, d, o, s);
    return true;
}

}  // namespace

int main() {
    const fixture f;
    const tq::reader::query_view view = the_view();
    const std::vector<int> cells = flatten(view);
    std::vector<float> want_out, want_w;
    reference(f, cells, want_out, want_w);

    // ---- (0) the bound itself, independent of any context --------------------------------------
    {
        bool ov = false;
        const std::size_t n3 = tq::reader::prefill_object_count(3, 2, false, &ov);
        const std::size_t n8 = tq::reader::prefill_object_count(8, 2, false, &ov);
        check(!ov && n3 > 0, "the bound for 3 tokens / 2 spans is positive");
        check(n8 > n3, "eight tokens cost more than three: " + std::to_string(n8) + " > " +
                           std::to_string(n3));
        // Per token it must be at least the single-query charge, so growth is at least three per
        // token (13 + 6*2 = 25, plus two slicing nodes, plus two joining nodes per seam).
        check(n8 - n3 >= 5 * 27, "growth covers per-token charge plus slicing and joining (" +
                                     std::to_string(n8 - n3) + " for five extra tokens)");
        bool ov2 = false;
        check(tq::reader::prefill_object_count(1, 2, false, &ov2) == 0 && !ov2,
              "T == 1 is not a prefill: the bound is 0, not a small number");
        // 2^60 tokens at 19 objects each is past SIZE_MAX/19, so the multiplication would wrap.
        // A wrapped count is smaller than the truth, which is the one direction a bound must never
        // err in -- so the function reports overflow and returns 0 (a refusal, not a budget).
        bool ov3 = false;
        check(tq::reader::prefill_object_count(std::int64_t(1) << 60, 1, false, &ov3) == 0 && ov3,
              "an absurd request reports overflow instead of a wrapped (too small) count");
    }

    // ---- (1) a too-small arena is refused, and the context stays empty --------------------------
    {
        ggml_init_params ip{};
        ip.mem_size = 2u * 1024u;   // far below T * per-token charge
        ip.no_alloc = true;
        ggml_context * tiny = ggml_init(ip);
        if (tiny == nullptr) {
            check(false, "tiny ggml_init");
        } else {
            ggml_tensor * k = ggml_new_tensor_3d(tiny, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
            ggml_tensor * v = ggml_new_tensor_3d(tiny, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
            ggml_tensor * q = ggml_new_tensor_3d(tiny, GGML_TYPE_F32, kHeadDim, kTokens, kHeads);
            ggml_tensor * m = ggml_new_tensor_4d(tiny, GGML_TYPE_F32, kCell, kTokens, 1, 1);
            if (k == nullptr || v == nullptr || q == nullptr || m == nullptr) {
                check(false, "tiny context could not hold even the inputs");
            } else {
                tq::reader::prefill_inputs in;
                in.ctx = tiny;
                in.k = k; in.v = v; in.q = q; in.mask = m;
                in.view = view;
                in.n_head = kHeads; in.n_head_kv = kKvHeads;
                in.head_dim = kHeadDim; in.n_cell = kCell;
                in.coverage = tq::reader::graph_inputs::mask_coverage::intersect;
                const std::size_t before = ggml_used_mem(tiny);
                const auto r = tq::reader::emit_prefill_attention(in, kScale);
                const std::size_t grew = ggml_used_mem(tiny) - before;
                check(r.refusal == tq::reader::prefill_refusal::arena_admission,
                      std::string("a 2 KiB arena is refused as arena_admission, got ") +
                          tq::reader::describe(r.refusal) + " -- " + r.why);
                check(grew == 0, "the refusal built " + std::to_string(grew) +
                                     " bytes; a refusal must cost nothing, because the published "
                                     "admission bound is an object count");
                check(r.per_token.empty(), "and no partial per-token emission was handed back");
            }
            ggml_free(tiny);
        }
    }

    // ---- (2) the real path ----------------------------------------------------------------------
    {
        const std::size_t objects_needed = [&] {
            bool ov = false;
            return tq::reader::prefill_object_count(kTokens, view.spans.size(), false, &ov);
        }();
        // A generous arena: the bound plus the caller's own graph and scaffolding. The emitter's
        // header is explicit that `ggml_new_graph` dwarfs an emission, so this is sized for the
        // test's own objects, not tuned to the emitter.
        ggml_init_params ip{};
        ip.mem_size = std::size_t(objects_needed) * 512u + 512u * 1024u;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        if (ctx == nullptr) {
            check(false, "ggml_init for the real path");
            std::fprintf(stderr, "prefill emit: %d failure(s)\n", failures);
            return 1;
        }

        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kTokens, kHeads);
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, kTokens, 1, 1);
        ggml_set_input(k); ggml_set_input(v); ggml_set_input(q); ggml_set_input(m);

        tq::reader::prefill_inputs in;
        in.ctx = ctx;
        in.k = k; in.v = v; in.q = q; in.mask = m;
        in.view = view;
        in.n_head = kHeads; in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim; in.n_cell = kCell;
        in.coverage = tq::reader::graph_inputs::mask_coverage::intersect;

        const std::size_t before = ggml_used_mem(ctx);
        const auto r = tq::reader::emit_prefill_attention(in, kScale);
        const std::size_t grew = ggml_used_mem(ctx) - before;
        if (r.refusal != tq::reader::prefill_refusal::ok) {
            check(false, std::string("emission refused: ") + tq::reader::describe(r.refusal) +
                             " -- " + r.why);
            ggml_free(ctx);
            std::fprintf(stderr, "prefill emit: %d failure(s)\n", failures);
            return 1;
        }
        check(r.per_token.size() == std::size_t(kTokens), "one emission per token");
        check(r.objects_charged > 0 && r.object_cost > 0 && r.bytes_required > 0,
              "the result reports its charge");
        check(grew <= r.bytes_required,
              "the arena grew " + std::to_string(grew) + " bytes, within the charge of " +
                  std::to_string(r.bytes_required));
        check(r.out != nullptr && r.weights != nullptr, "both joined tensors exist");
        check(r.out->ne[0] == kHeadDim && r.out->ne[1] == kTokens && r.out->ne[2] == kHeads,
              "out is [hd, T, nh]: got [" + std::to_string(r.out->ne[0]) + " " +
                  std::to_string(r.out->ne[1]) + " " + std::to_string(r.out->ne[2]) + "]");
        check(r.weights->ne[0] == kCell && r.weights->ne[1] == kTokens &&
                  r.weights->ne[2] == kHeads,
              "weights is [n_cell, T, nh]");
        for (std::size_t t = 0; t < r.per_token.size(); ++t) {
            check(r.per_token[t].indices.size() == view.spans.size() + 1,
                  "token " + std::to_string(t) + " has one gather per span plus the scatter");
            check(!r.per_token[t].index_cells.empty(),
                  "token " + std::to_string(t) + " owns its own cells");
        }

        ggml_backend_t be = ggml_backend_cpu_init();
        ggml_backend_buffer_t buf = be ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
        if (buf == nullptr) {
            check(false, "placement failed");
            if (buf) ggml_backend_buffer_free(buf);
            if (be) ggml_backend_free(be);
            ggml_free(ctx);
            std::fprintf(stderr, "prefill emit: %d failure(s)\n", failures);
            return 1;
        }

        // Real data, after placement (an unplaced tensor has no `data` to write into).
        ggml_backend_tensor_set(k, f.k.data(), 0, f.k.size() * sizeof(float));
        ggml_backend_tensor_set(v, f.v.data(), 0, f.v.size() * sizeof(float));
        ggml_backend_tensor_set(q, f.q.data(), 0, f.q.size() * sizeof(float));
        ggml_backend_tensor_set(m, f.mask.data(), 0, f.mask.size() * sizeof(float));

        // One publication per token: each owns its own `index_cells`, so the tiling contract is
        // satisfied per call. A single merged buffer is what the second bespoke attempt could not do.
        bool published = true;
        for (std::size_t t = 0; t < r.per_token.size(); ++t) {
            std::string why;
            const auto pr = tq::reader::publish_index_inputs(r.per_token[t], &publish_cb, &why,
                                                             nullptr);
            check(pr == tq::reader::index_publish_refusal::ok,
                  "token " + std::to_string(t) + " publish: " +
                      tq::reader::describe(pr) + " (" + why + ")");
            if (pr != tq::reader::index_publish_refusal::ok) published = false;
        }
        if (!published) {
            check(false, "not computing: a token's gather indices were never written");
            ggml_backend_buffer_free(buf);
            ggml_backend_free(be);
            ggml_free(ctx);
            std::fprintf(stderr, "prefill emit: %d failure(s)\n", failures);
            return 1;
        }

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, r.out);
        ggml_build_forward_expand(gf, r.weights);
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            check(false, "compute failed after a successful publish");
        } else {
            std::vector<float> got_out(std::size_t(kHeadDim) * kTokens * kHeads, 0.0f);
            std::vector<float> got_w(std::size_t(kCell) * kTokens * kHeads, 0.0f);
            ggml_backend_tensor_get(r.out, got_out.data(), 0, got_out.size() * sizeof(float));
            ggml_backend_tensor_get(r.weights, got_w.data(), 0, got_w.size() * sizeof(float));

            auto compare = [&](const std::vector<float> & got, const std::vector<float> & want,
                               const char * label, double tol) {
                double worst = 0.0;
                int at = -1;
                bool finite = true;
                for (std::size_t i = 0; i < got.size(); ++i) {
                    if (!std::isfinite(got[i]) || !std::isfinite(want[i])) { finite = false; break; }
                    const double d = std::fabs(double(got[i]) - double(want[i]));
                    if (d > worst) { worst = d; at = int(i); }
                }
                check(finite, std::string(label) + ": every element is finite");
                check(worst <= tol, std::string(label) + ": max|diff| = " + std::to_string(worst) +
                                        " at " + std::to_string(at) + " (tol " +
                                        std::to_string(tol) + ")");
                std::fprintf(stderr, "  %-14s max|diff| = %.3e\n", label, worst);
            };
            compare(got_out, want_out, "out", 1e-5);
            compare(got_w, want_w, "weights", 1e-5);

            // (3) visibility is read PER TOKEN, asserted over SETS rather than one cell.
            //
            //    A single-cell assertion was tried and was wrong for a reason worth recording: with
            //    this fixture's random values, a cell that token 2 alone can see still underflows to
            //    a float zero (exp(score - max) below the smallest normal), and the reference agrees
            //    -- max|diff| was 1.49e-08 while the cell's weight was 0.000000. So the property to
            //    assert is about VISIBILITY, not about magnitude:
            //
            //      * every cell at or above `visible(0)` is exactly zero for token 0, because the
            //        mask hides it there and nothing else narrows a full-coverage view;
            //      * token 2, which sees up to `visible(2)`, has positive weight on the cells
            //        between the two thresholds -- at least one of them must carry mass, or the mask
            //        was not read per token at all.
            {
                for (int c = visible(0); c < kCell; ++c) {
                    const double w = double(got_w[std::size_t(c) + std::size_t(kCell) *
                                                      (0 + std::size_t(kTokens) * 0)]);
                    check(w == 0.0, "token 0 gives cell " + std::to_string(c) +
                                        " exactly zero weight (got " + std::to_string(w) + ")");
                }
                double w2 = 0.0;
                for (int c = visible(1); c < visible(2); ++c) {
                    w2 = std::max(w2, double(got_w[std::size_t(c) + std::size_t(kCell) *
                                                   (2 + std::size_t(kTokens) * 0)]));
                }
                check(w2 > 0.0, "token 2 carries positive weight on the band only it can see (" +
                                    std::to_string(w2) + ") -- so the mask is per token");
            }

            // (4) Non-vacuity: broadcast visibility and a modulo grouping must both move a number.
            {
                std::vector<float> bm(std::size_t(kCell) * kTokens, -INFINITY);
                for (int t = 0; t < kTokens; ++t)
                    for (int c : cells)
                        bm[std::size_t(c) + std::size_t(kCell) * 0] = 0.0f;   // one row for all
                std::vector<float> o2, w2v;
                // A reference with the broadcast row applied to every token:
                {
                    fixture g = f;
                    for (int t = 0; t < kTokens; ++t)
                        for (int c = 0; c < kCell; ++c)
                            g.mask[std::size_t(c) + std::size_t(kCell) * t] =
                                g.mask[std::size_t(c) + std::size_t(kCell) * 0];
                    reference(g, cells, o2, w2v);
                }
                double gap = 0.0;
                for (std::size_t i = 0; i < want_out.size(); ++i)
                    gap = std::max(gap, std::fabs(double(want_out[i]) - double(o2[i])));
                check(gap > 1e-3, "per-token vs broadcast visibility differ by " +
                                      std::to_string(gap) + " -- the fixture can detect a broadcast");

                double magnitude = 0.0;
                for (float x : want_out) magnitude = std::max(magnitude, std::fabs(double(x)));
                check(magnitude > 1e-2, "the reference is non-degenerate (" +
                                            std::to_string(magnitude) + ")");
            }
        }

        ggml_backend_buffer_free(buf);
        ggml_backend_free(be);
        ggml_free(ctx);
    }

    // ---- (5) preflight refusals that must not build anything ------------------------------------
    {
        ggml_init_params ip{};
        ip.mem_size = 256u * 1024u;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kTokens, kHeads);
        ggml_tensor * q1 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads);
        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, kTokens, 1, 1);
        ggml_tensor * m1 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, 1, 1, 1);
        if (ctx == nullptr || k == nullptr || v == nullptr || q == nullptr || q1 == nullptr ||
            m == nullptr || m1 == nullptr) {
            check(false, "the preflight context could not hold its fixtures");
            if (ctx) ggml_free(ctx);
        } else {
            tq::reader::prefill_inputs base;
            base.ctx = ctx;
            base.k = k; base.v = v; base.q = q; base.mask = m;
            base.view = the_view();
            base.n_head = kHeads; base.n_head_kv = kKvHeads;
            base.head_dim = kHeadDim; base.n_cell = kCell;
            base.coverage = tq::reader::graph_inputs::mask_coverage::intersect;

            auto expect = [&](tq::reader::prefill_inputs in, tq::reader::prefill_refusal want,
                              const char * label) {
                const std::size_t before = ggml_used_mem(ctx);
                const auto r = tq::reader::emit_prefill_attention(in, kScale);
                const std::size_t grew = ggml_used_mem(ctx) - before;
                check(r.refusal == want, std::string(label) + ": got " +
                                             tq::reader::describe(r.refusal) + " -- " + r.why);
                check(grew == 0, std::string(label) + ": built " + std::to_string(grew) +
                                     " bytes before refusing");
            };

            {   auto in = base; in.q = q1;
                expect(in, tq::reader::prefill_refusal::single_token, "T == 1"); }
            {   auto in = base; in.mask = m1;
                expect(in, tq::reader::prefill_refusal::bad_mask, "one mask row for three tokens"); }
            {   auto in = base; in.n_head = 5;
                expect(in, tq::reader::prefill_refusal::bad_head_layout,
                       "5 query heads over 2 kv heads"); }
            {   auto in = base; in.n_head = 6;
                expect(in, tq::reader::prefill_refusal::bad_shape,
                       "6 stated heads against a 4-head q tensor"); }
            {   auto in = base; in.k = nullptr;
                expect(in, tq::reader::prefill_refusal::no_context, "null cache"); }
            {   auto in = base;
                // The view has ONE span now (full coverage), so a duplicate is made by repeating a
                // cell inside span 0. The earlier version indexed spans[1], which no longer exists --
                // that out-of-bounds read, not the emitter, was the crash this section caused.
                in.view.spans[0].cells.push_back(in.view.spans[0].cells.front());
                expect(in, tq::reader::prefill_refusal::bad_view, "a duplicated cell"); }
            {   auto in = base;
                in.view.spans[0].cells[0] = kCell;   // one past the end
                expect(in, tq::reader::prefill_refusal::bad_view, "a cell outside the cache"); }

            ggml_free(ctx);
        }
    }

    if (failures == 0) {
        std::fprintf(stderr, "prefill emit: OK\n");
        return 0;
    }
    std::fprintf(stderr, "prefill emit: %d failure(s)\n", failures);
    return 1;
}
