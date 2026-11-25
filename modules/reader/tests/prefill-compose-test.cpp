// Prefill by COMPOSITION: T tokens, each through the already-verified single-query emitter, joined
// along dim 1 — with the join axis and the slice strides MEASURED, not inferred.
//
// WHY THIS EXISTS AS A TEST AND NOT AN EMITTER
// --------------------------------------------
// Two multi-token emitters were written and both were removed. The first mis-charged its own gathers
// (an admission bound below the truth). The second produced `out` differing from a reference by
// 0.49 and the cause was never diagnosed, so the number could not be attributed to anything. Both
// times the argument was about layout, and neither time was the layout printed.
//
// This file takes the smallest step that can settle the numeric question without writing a new
// emitter: run the existing one T times, join the results the way an adapter would, and compare
// against a reference written per (token, head) pair. Nothing here is new graph code; if the numbers
// agree, the composition is a verified recipe, and if they do not, the disagreement is localised to
// a join or a slice rather than to an unknown quantity of new arithmetic.
//
// WHAT IS MEASURED RATHER THAN ARGUED
// -----------------------------------
//   1. The JOIN axis. `concat(out_0, ...)` along dim 1 of [hd,1,nh] operands must read back as
//      `d + hd*(t + T*h)`. `tensor-layout-probe-test` asserts this for ggml in general; here it is
//      asserted again on the ACTUAL emitter outputs, because the operand being a graph result rather
//      than a hand-filled tensor is exactly the difference that matters.
//   2. The MASK slice. A [n_cell, T, 1, 1] mask's token-t column must be readable as a contiguous
//      [n_cell,1,1,1] view at offset t*nb[1] — printed, then compared against the standalone tensor
//      the emitter also consumes, so "the slice is the same data" is a fact rather than a hope.
//   3. The QUERY slice. q arrives as [hd, T, nh]; a token's [hd,1,nh] block is NOT a natural view
//      (the head stride is T*hd*element). The working recipe is one whole-tensor permute to
//      [hd,nh,T] then cont, after which token t is contiguous at t*hd*nh. Its read-back values are
//      compared to the standalone per-token tensor.
//   4. The NUMBER. `out` for all T tokens against a double reference whose softmax is written per
//      (token, head) directly from the definition.
//
// SAFETY. Under no_alloc the index tensors have no storage until the scheduler places the graph, so
// publication happens after `ggml_backend_alloc_ctx_tensors` and before compute — one call per token,
// since each emission owns its own `index_cells`. Every failure path releases buffer, backend and
// context, and a compute failure is a `check(false, ...)` so it cannot exit 0.
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

constexpr int kHeadDim = 8;
constexpr int kKvHeads = 2;
constexpr int kQHeads  = 2;   // group = 1: each query head reads its own kv head
constexpr int kCell    = 16;
constexpr int kTokens  = 3;
constexpr float kScale = 0.35355339f;

// Token t may see the first `visible(t)` cells only: a causal ladder with a finite per-token bias, so
// a broadcast mask (one row for all tokens) gives a different answer and cannot pass by accident.
int visible(int t) { return 6 + 3 * t; }
float bias(int t) { return 0.25f * float(t); }

struct fixture {
    std::vector<float> k, v;              // [hd, kvh, n_cell]
    std::vector<float> q;                 // [hd, T, nh],   d + hd*(t + T*h)
    std::vector<float> mask;              // [n_cell, T, 1, 1], c + n_cell*t

    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kTokens * kQHeads, 0.0f);
        mask.assign(std::size_t(kCell) * kTokens, 0.0f);
        unsigned s = 0x9e3779b9u;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
        for (int t = 0; t < kTokens; ++t) {
            for (int c = 0; c < kCell; ++c) {
                mask[std::size_t(c) + std::size_t(kCell) * t] =
                    (c < visible(t)) ? bias(t) : -INFINITY;
            }
        }
    }

    float K(int d, int kvh, int c) const { return k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float V(int d, int kvh, int c) const { return v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float Q(int d, int t, int h) const { return q[std::size_t(d) + kHeadDim * (t + kTokens * h)]; }
    float M(int c, int t) const { return mask[std::size_t(c) + std::size_t(kCell) * t]; }
};

// The definition, per (token, head): one softmax over the whole cache, -inf hidden, finite bias added.
void reference(const fixture & f, std::vector<float> & out) {
    const int group = kQHeads / kKvHeads;
    out.assign(std::size_t(kHeadDim) * kTokens * kQHeads, 0.0f);
    for (int t = 0; t < kTokens; ++t) {
        for (int h = 0; h < kQHeads; ++h) {
            const int kvh = h / group;
            std::vector<double> score(static_cast<std::size_t>(kCell), 0.0);
            double max_s = -INFINITY;
            for (int c = 0; c < kCell; ++c) {
                const double b = double(f.M(c, t));
                if (!std::isfinite(b)) {
                    score[std::size_t(c)] = -INFINITY;
                    continue;
                }
                double s = 0.0;
                for (int d = 0; d < kHeadDim; ++d)
                    s += double(f.Q(d, t, h)) * double(f.K(d, kvh, c));
                score[std::size_t(c)] = s * double(kScale) + b;
                if (score[std::size_t(c)] > max_s) max_s = score[std::size_t(c)];
            }
            double sum = 0.0;
            for (int c = 0; c < kCell; ++c) {
                if (!std::isfinite(score[std::size_t(c)])) continue;
                score[std::size_t(c)] = std::exp(score[std::size_t(c)] - max_s);
                sum += score[std::size_t(c)];
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (int c = 0; c < kCell; ++c) {
                    if (!std::isfinite(score[std::size_t(c)]) || score[std::size_t(c)] == 0.0) continue;
                    acc += (score[std::size_t(c)] / sum) * double(f.V(d, kvh, c));
                }
                out[std::size_t(d) + std::size_t(kHeadDim) * (t + kTokens * h)] = float(acc);
            }
        }
    }
}

bool publish_cb(ggml_tensor * t, const void * d, std::size_t o, std::size_t s) {
    // Through the backend: `data != nullptr` means placed, not host-addressable.
    if (t == nullptr || t->buffer == nullptr) return false;
    ggml_backend_tensor_set(t, d, o, s);
    return true;
}

}  // namespace

int main() {
    const fixture f;
    std::vector<float> want;
    reference(f, want);

    ggml_init_params ip{};
    ip.mem_size = 8u * 1024u * 1024u;
    ip.no_alloc = true;   // every tensor created BEFORE placement; one created after has no data
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: ggml_init\n");
        return 1;
    }

    // k/v as upstream passes them: [hd, n_head_kv, n_cell].
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    ggml_set_input(k);
    ggml_set_input(v);

    // One mask per token, and the combined tensor the slice test reads from.
    std::vector<ggml_tensor *> mask_t(kTokens);
    for (int t = 0; t < kTokens; ++t) {
        mask_t[t] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, 1, 1, 1);
        ggml_set_input(mask_t[t]);
    }
    ggml_tensor * mask_all = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, kTokens, 1, 1);
    ggml_set_input(mask_all);

    // One query per token, plus the combined [hd,T,nh] the slice test works on.
    std::vector<ggml_tensor *> q_t(kTokens);
    for (int t = 0; t < kTokens; ++t) {
        q_t[t] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        ggml_set_input(q_t[t]);
    }
    ggml_tensor * q_all = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kTokens, kQHeads);
    ggml_set_input(q_all);

    // The slice recipes under test, built before placement like everything else.
    //   mask: [n_cell, T, 1, 1] -> token t is contiguous at t*nb[1]
    ggml_tensor * mask_slice1 = ggml_view_4d(ctx, mask_all, kCell, 1, 1, 1,
                                             mask_all->nb[1], mask_all->nb[2], mask_all->nb[3],
                                             std::size_t(1) * mask_all->nb[1]);
    //   q: whole-tensor permute to [hd,nh,T] then cont, after which token t is contiguous
    ggml_tensor * q_head_major = ggml_cont(ctx, ggml_permute(ctx, q_all, 0, 2, 1, 3));
    // TOKEN t IS A PACKED [hd,1,nh] BLOCK AT t*hd*nh*element -- and the strides below say so.
    //
    // The previous revision passed the PARENT's strides (`nb[1]`, `nb[2]` of a `[hd,nh,T]` tensor).
    // For a view whose `ne[2] == nh` that is wrong in exactly one place: `nb[2]` should be `hd*element`
    // (the stride of a dim of extent 1 is one row, not `nh` rows), so `h=1` read the NEXT TOKEN's head
    // 0 -- 200 where token 1's head 1 dim 0 is 110. A legal tensor describing the wrong bytes, which
    // no shape or contiguity check could distinguish from a correct one.
    //
    // With packed strides the slice is contiguous, and `q_stride_contract` proves the emitter accepts
    // it and returns the same numbers as a hand-built per-token tensor (padded F32 bit-identical,
    // padded F16 within half precision) even with `group = 2`.
    ggml_tensor * q_slice1 = ggml_view_3d(ctx, q_head_major, kHeadDim, 1, kQHeads,
                                          std::size_t(kHeadDim) * sizeof(float),
                                          std::size_t(kHeadDim) * sizeof(float),
                                          std::size_t(1) * kHeadDim * kQHeads * sizeof(float));

    const tq::reader::query_view full = tq::reader::full_coverage_view(kCell);

    std::vector<tq::reader::physical_emit_result> em(kTokens);
    for (int t = 0; t < kTokens; ++t) {
        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = k;
        in.v = v;
        // TOKEN 1 IS EMITTED FROM THE SLICES, so the join and the numeric comparison below cover the
        // whole recipe -- view, stride, emitter, concat -- rather than two disjoint halves.
        //
        // The earlier revision fed per-token hand-built tensors and verified the slices only by
        // reading them back, which proved the slice CONTENTS and nothing about whether the emitter
        // composes with them. That gap is where a 2.164e-02 disagreement hid, and its cause was the
        // slice's strides (see the note above `q_slice1`), not the emitter: with packed strides the
        // same slice now produces identical numbers through the emitter (`q_stride_contract`), so the
        // two feeds must agree here, and a token-1-only failure in the number below would mean the
        // prefill recipe broke again.
        in.q = (t == 1) ? q_slice1 : q_t[t];
        in.mask = (t == 1) ? mask_slice1 : mask_t[t];
        in.coverage = tq::reader::graph_inputs::mask_coverage::intersect;
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim;
        in.n_cell = kCell;
        in.collect_indices = true;
        em[t] = tq::reader::emit_physical_attention(in, full, kScale);
        if (em[t].refusal != tq::reader::physical_emit_refusal::ok) {
            check(false, "token " + std::to_string(t) + ": emit refused: " +
                             tq::reader::describe(em[t].refusal) + " -- " + em[t].why);
            ggml_free(ctx);
            return 1;
        }
        if (em[t].out == nullptr || em[t].weights == nullptr ||
            em[t].indices.size() != 2) {   // one span + the scatter
            check(false, "token " + std::to_string(t) + ": emission is not 2 entries with outputs");
            ggml_free(ctx);
            return 1;
        }
    }

    // Join the per-token outputs along dim 1. NOTHING ELSE is joined: each emission keeps its own
    // `index_cells`, so the publications stay separate (that part was already measured to work).
    ggml_tensor * joined = em[0].out;
    for (int t = 1; t < kTokens; ++t) joined = ggml_concat(ctx, joined, em[t].out, 1);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = be ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
    if (be == nullptr || buf == nullptr) {
        check(false, "placement failed");
        if (buf != nullptr) ggml_backend_buffer_free(buf);
        if (be != nullptr) ggml_backend_free(be);
        ggml_free(ctx);
        return 1;
    }

    // Inputs, through the backend: `data != nullptr` means placed, not host-addressable.
    ggml_backend_tensor_set(k, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(v, f.v.data(), 0, f.v.size() * sizeof(float));
    ggml_backend_tensor_set(q_all, f.q.data(), 0, f.q.size() * sizeof(float));
    ggml_backend_tensor_set(mask_all, f.mask.data(), 0, f.mask.size() * sizeof(float));
    for (int t = 0; t < kTokens; ++t) {
        std::vector<float> qv(static_cast<std::size_t>(kHeadDim) * kQHeads, 0.0f);
        std::vector<float> mv(static_cast<std::size_t>(kCell), 0.0f);
        for (int h = 0; h < kQHeads; ++h)
            for (int d = 0; d < kHeadDim; ++d) qv[std::size_t(d + kHeadDim * h)] = f.Q(d, t, h);
        for (int c = 0; c < kCell; ++c) mv[std::size_t(c)] = f.M(c, t);
        ggml_backend_tensor_set(q_t[t], qv.data(), 0, qv.size() * sizeof(float));
        ggml_backend_tensor_set(mask_t[t], mv.data(), 0, mv.size() * sizeof(float));
    }

    // Publishing, one call per token, because each emission owns its own `index_cells`. A single
    // merged buffer is what a previous attempt could not do.
    //
    // A FAILED PUBLISH STOPS THE TEST. Recording it and computing anyway would gather through index
    // tensors whose contents are whatever placement left in them -- uninitialised host memory on the
    // CPU -- so the graph would read past k/v and could crash rather than report. A crash inside a
    // red test replaces the diagnosis with a signal, which is the opposite of what a test is for.
    // This is the same hard-stop the host-lifecycle test needed, and it is here for the same reason:
    // the check is only meaningful if the thing it checks is a precondition of what follows.
    for (int t = 0; t < kTokens; ++t) {
        std::string why;
        const auto r = tq::reader::publish_index_inputs(em[t], &publish_cb, &why, nullptr);
        check(r == tq::reader::index_publish_refusal::ok,
              "token " + std::to_string(t) + ": publish: " + tq::reader::describe(r) + " (" + why + ")");
        if (r != tq::reader::index_publish_refusal::ok) {
            check(false, "not computing: token " + std::to_string(t) +
                             "'s gather indices were never written, so the joined result would be "
                             "built from uninitialised memory");
            ggml_backend_buffer_free(buf);
            ggml_backend_free(be);
            ggml_free(ctx);
            return 1;
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, joined);
    ggml_build_forward_expand(gf, mask_slice1);
    ggml_build_forward_expand(gf, q_slice1);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        check(false, "compute failed, so no layout or number below was read");
        ggml_backend_buffer_free(buf);
        ggml_backend_free(be);
        ggml_free(ctx);
        return 1;
    }

    // ---------------------------------------------------------------- (1) the join axis, on real
    // emitter outputs. `joined` is [hd, T, nh]; element (d,t,h) must be at d + hd*(t + T*h).
    {
        check(joined->ne[0] == kHeadDim && joined->ne[1] == kTokens && joined->ne[2] == kQHeads,
              "joined is [hd, T, nh]: got [" + std::to_string(joined->ne[0]) + " " +
                  std::to_string(joined->ne[1]) + " " + std::to_string(joined->ne[2]) + "]");
        std::vector<float> j(static_cast<std::size_t>(kHeadDim) * kTokens * kQHeads, 0.0f);
        ggml_backend_tensor_get(joined, j.data(), 0, j.size() * sizeof(float));
        int mismatches = 0;
        std::string first;
        for (int h = 0; h < kQHeads; ++h) {
            for (int t = 0; t < kTokens; ++t) {
                for (int d = 0; d < kHeadDim; ++d) {
                    const float got = j[std::size_t(d + kHeadDim * (t + kTokens * h))];
                    // An emission's own token-0 head-h value, read back from that emission, is the
                    // ground truth for this slot: the join must not move it.
                    std::vector<float> single(static_cast<std::size_t>(kHeadDim) * kQHeads, 0.0f);
                    ggml_backend_tensor_get(em[t].out, single.data(), 0, single.size() * sizeof(float));
                    const float there = single[std::size_t(d + kHeadDim * h)];
                    if (got != there) {
                        if (mismatches == 0)
                            first = "d=" + std::to_string(d) + " t=" + std::to_string(t) +
                                    " h=" + std::to_string(h) + ": joined " + std::to_string(got) +
                                    " vs emission " + std::to_string(there);
                        ++mismatches;
                    }
                }
            }
        }
        check(mismatches == 0, "joined[d + hd*(t + T*h)] is NOT the emission's own value at " +
                                   std::to_string(mismatches) + " of " +
                                   std::to_string(kHeadDim * kTokens * kQHeads) +
                                   " slots; first: " + first);
    }

    // ---------------------------------------------------------------- (2) the mask slice
    {
        std::vector<float> s(static_cast<std::size_t>(kCell), 0.0f);
        ggml_backend_tensor_get(mask_slice1, s.data(), 0, s.size() * sizeof(float));
        int bad = 0;
        std::string first;
        for (int c = 0; c < kCell; ++c) {
            const float expect = f.M(c, 1);
            if (s[std::size_t(c)] != expect) {
                if (bad == 0) first = "c=" + std::to_string(c) + ": slice " +
                                      std::to_string(s[std::size_t(c)]) + " vs token-1 mask " +
                                      std::to_string(expect);
                ++bad;
            }
        }
        // -inf == -inf is true, so a hidden column compares equal; a wrong ROW does not, because the
        // three rows differ (6, 9 and 12 visible cells).
        check(bad == 0, "the [n_cell,T,1,1] mask slice at t=1 is not token 1's column: " +
                            std::to_string(bad) + " of " + std::to_string(kCell) +
                            " entries differ; first: " + first);
    }

    // ---------------------------------------------------------------- (3) the query slice
    {
        check(q_slice1->ne[0] == kHeadDim && q_slice1->ne[1] == 1 && q_slice1->ne[2] == kQHeads,
              "the query slice has shape [hd,1,nh]: got [" + std::to_string(q_slice1->ne[0]) + " " +
                  std::to_string(q_slice1->ne[1]) + " " + std::to_string(q_slice1->ne[2]) + "]");
        std::vector<float> s(static_cast<std::size_t>(kHeadDim) * kQHeads, 0.0f);
        ggml_backend_tensor_get(q_slice1, s.data(), 0, s.size() * sizeof(float));
        int bad = 0;
        std::string first;
        for (int h = 0; h < kQHeads; ++h) {
            for (int d = 0; d < kHeadDim; ++d) {
                const float expect = f.Q(d, 1, h);
                if (s[std::size_t(d + kHeadDim * h)] != expect) {
                    if (bad == 0) first = "d=" + std::to_string(d) + " h=" + std::to_string(h) +
                                          ": slice " + std::to_string(s[std::size_t(d + kHeadDim * h)]) +
                                          " vs token-1 q " + std::to_string(expect);
                    ++bad;
                }
            }
        }
        check(bad == 0, "the permute+cont query slice at t=1 is not token 1's row: " +
                            std::to_string(bad) + " of " + std::to_string(kHeadDim * kQHeads) +
                            " entries differ; first: " + first);
    }

    // ---------------------------------------------------------------- (4) the number
    {
        std::vector<float> got(static_cast<std::size_t>(kHeadDim) * kTokens * kQHeads, 0.0f);
        ggml_backend_tensor_get(joined, got.data(), 0, got.size() * sizeof(float));
        double worst = 0.0;
        int worst_at = -1;
        bool finite = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) { finite = false; break; }
            const double d = std::fabs(double(got[i]) - double(want[i]));
            if (d > worst) { worst = d; worst_at = int(i); }
        }
        check(finite, "out is finite at every element");
        check(worst < 1e-5, "composed prefill matches the per-(token,head) definition: max|diff| = " +
                                std::to_string(worst) + " at index " + std::to_string(worst_at));

        // Non-vacuity, three ways, so a broken fixture cannot pass:
        double magnitude = 0.0;
        for (float x : want) magnitude = std::max(magnitude, std::fabs(double(x)));
        check(magnitude > 1e-2, "the reference is non-degenerate (max |value| = " +
                                    std::to_string(magnitude) + ")");
        // Tokens must differ from each other (a broadcast mask and a shared query would erase this).
        double token_gap = 0.0;
        for (int h = 0; h < kQHeads; ++h)
            for (int d = 0; d < kHeadDim; ++d) {
                const double a = double(want[std::size_t(d + kHeadDim * (0 + kTokens * h))]);
                const double b = double(want[std::size_t(d + kHeadDim * (2 + kTokens * h))]);
                token_gap = std::max(token_gap, std::fabs(a - b));
            }
        check(token_gap > 1e-3, "token 0 and token 2 differ by " + std::to_string(token_gap) +
                                    " -- per-token visibility is observable in the fixture");
        // Heads must differ too, or a head mix-up would be invisible.
        double head_gap = 0.0;
        for (int t = 0; t < kTokens; ++t)
            for (int d = 0; d < kHeadDim; ++d) {
                const double a = double(want[std::size_t(d + kHeadDim * (t + kTokens * 0))]);
                const double b = double(want[std::size_t(d + kHeadDim * (t + kTokens * 1))]);
                head_gap = std::max(head_gap, std::fabs(a - b));
            }
        check(head_gap > 1e-3, "the two heads differ by " + std::to_string(head_gap) +
                                   " -- a head swap would move a number");

        std::fprintf(stderr,
                     "prefill compose: max|diff| = %.3e (worst %d), |ref|max = %.3e, "
                     "token gap = %.3e, head gap = %.3e, %d tokens x %d heads over %d cells\n",
                     worst, worst_at, magnitude, token_gap, head_gap, kTokens, kQHeads, kCell);
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);

    if (failures == 0) {
        std::fprintf(stderr, "prefill compose: OK\n");
        return 0;
    }
    std::fprintf(stderr, "prefill compose: %d failure(s)\n", failures);
    return 1;
}
