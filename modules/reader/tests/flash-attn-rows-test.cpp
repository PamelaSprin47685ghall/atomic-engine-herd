// Upstream's own row-table attention operator, executed on the CPU and checked against arithmetic
// written from the definition.
//
// WHY THIS TEST EXISTS. Every earlier reader test either computed attention with our own operators
// (`graph_emit_physical`, `upstream-attention-contrast`) or validated a row list without computing
// anything (`rows_attention`). What was never established is whether `ggml_flash_attn_ext_rows` --
// the ONE operator upstream uses when `kv_rows` is present -- actually computes the documented
// contract on this backend. That question has to be answered before any replacement is designed
// around it, because the contract has three traps that a plausible implementation can each get
// wrong in a way that still produces numbers:
//
//   1. `kv_rows` is an INDIRECTION, not a position map. Mask column i reads row `kv_rows[i]` of k
//      and v. The rows here are deliberately non-monotonic (3, 0, 6, 5, 2) and the fixture is built
//      so that reading position i instead of row kv_rows[i] gives visibly different numbers.
//   2. A NEGATIVE row is padding, and ggml.h requires its mask entry to be -inf. Padding columns
//      are interleaved, not trailing, so an implementation that assumed "the valid ones come
//      first" fails here.
//   3. The MASK is a per-column bias, not a boolean. Two columns carry FINITE values, stored in
//      F16 -- upstream's actual `kq_mask` type (measured `type 1`). An implementation that treats
//      any non-zero mask as "hidden" loses the bias; one that ignores finite bias entirely gets
//      the wrong answer on those columns only, which is easy to miss.
//
// GQA is exercised (n_head 4 over n_head_kv 2) because the grouping rule `h / (n_head/n_head_kv)`
// and the wrong-but-plausible `h % n_head_kv` differ on this fixture -- asserted, not assumed.
//
// WHAT THIS IS NOT: a replacement, a seam, or a product path. It runs one operator on the CPU in a
// test process. Nothing here is installed, and the reader lane still changes no number in a decode.
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <algorithm>
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
constexpr int kQHeads  = 4;   // group = 2
// 8, not 5. The row list below names cells 0..6 (max 6), so a cache of 5 rows made `rows[i] == 6`
// index past the end of k/v -- in the REFERENCE as well as in the operator. That is what produced
// `-3.12e+35` / `-nan` on kv head 1 while kv head 0 happened to land in bounds: the failure was in
// the fixture, not in the operator. Every non-negative row must be a valid row of k/v, and the
// assertion below now proves it instead of trusting the numbers to line up.
constexpr int kNRows    = 8;  // k/v rows available; rows may name any of them
constexpr int kNkv      = 8;  // mask columns; 3 of them are padding
constexpr float kScale  = 0.35355339f;

// Mask columns, paired with kv_rows entries. -1 with -INFINITY is padding; finite bias is real.
struct plan {
    std::vector<std::int32_t> rows;
    std::vector<float>        bias;
};

// Deliberately awkward: non-monotonic rows, leading *and* interleaved padding, and a finite
// POSITIVE bias followed later by a finite NEGATIVE one. Ordering the padding last, or making every
// finite bias zero, would let a wrong implementation agree on this fixture.
plan the_plan() {
    return plan{
        {3, -1, 0, 6, -1, 5, 2, -1},
        {0.75f, -INFINITY, -0.5f, 0.25f, -INFINITY, 0.0f, 1.25f, -INFINITY},
    };
}

struct fixture {
    std::vector<float> q;   // [kHeadDim, 1, kQHeads]
    std::vector<float> k;   // [kHeadDim, kNRows, kKvHeads]
    std::vector<float> v;   // [kHeadDim, kNRows, kKvHeads]
    std::vector<float> mask;  // [kNkv, 1, 1, 1]

    explicit fixture(const plan & p) {
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        k.assign(std::size_t(kHeadDim) * kNRows * kKvHeads, 0.0f);
        v.assign(k.size(), 0.0f);
        mask.assign(std::size_t(kNkv), 0.0f);
        unsigned s = 0x1234567u;
        auto next = [&s] {
            s = s * 1103515245u + 12345u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : q) x = next();
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 2.0f;
        for (int i = 0; i < kNkv; ++i) mask[std::size_t(i)] = p.bias[std::size_t(i)];
    }

    // k/v as the operator lays them out: [n_embd, n_rows, n_head_kv].
    float K(int d, int row, int kvh) const {
        return k[std::size_t(d) + kHeadDim * (row + kNRows * kvh)];
    }
    float V(int d, int row, int kvh) const {
        return v[std::size_t(d) + kHeadDim * (row + kNRows * kvh)];
    }
    float Q(int d, int h) const { return q[std::size_t(d) + kHeadDim * h]; }
};

// The contract, written out: for query head h, kv head = h / group; score_i = scale * q.k(row_i)
// + mask_i; softmax over columns whose mask is finite; out = sum p_i * v(row_i).
void reference(const fixture & f, const plan & p, bool wrong_grouping, std::vector<float> & out) {
    const int group = kQHeads / kKvHeads;
    out.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
    std::vector<double> score(static_cast<std::size_t>(kNkv), 0.0);

    for (int h = 0; h < kQHeads; ++h) {
        const int kvh = wrong_grouping ? (h % kKvHeads) : (h / group);
        double max_s = -INFINITY;
        for (int i = 0; i < kNkv; ++i) {
            const double bias = double(f.mask[std::size_t(i)]);
            if (!std::isfinite(bias)) {
                score[std::size_t(i)] = -INFINITY;
                continue;
            }
            double s = 0.0;
            for (int d = 0; d < kHeadDim; ++d)
                s += double(f.Q(d, h)) * double(f.K(d, p.rows[std::size_t(i)], kvh));
            s = s * double(kScale) + bias;
            score[std::size_t(i)] = s;
            if (s > max_s) max_s = s;
        }
        double sum = 0.0;
        std::vector<double> w(std::size_t(kNkv), 0.0);
        for (int i = 0; i < kNkv; ++i) {
            if (!std::isfinite(score[std::size_t(i)])) continue;
            w[std::size_t(i)] = std::exp(score[std::size_t(i)] - max_s);
            sum += w[std::size_t(i)];
        }
        for (int d = 0; d < kHeadDim; ++d) {
            double acc = 0.0;
            for (int i = 0; i < kNkv; ++i) {
                if (w[std::size_t(i)] == 0.0) continue;
                acc += (w[std::size_t(i)] / sum) * double(f.V(d, p.rows[std::size_t(i)], kvh));
            }
            out[std::size_t(d) + std::size_t(kHeadDim) * h] = float(acc);
        }
    }
}

// Runs `ggml_flash_attn_ext_rows` over the fixture. The mask is stored F16 (upstream's own type);
// every other operand is F32 so that a mismatch cannot be blamed on quantisation.
bool run_operator(const fixture & f, const plan & p, std::vector<float> & out, std::string & why) {
    ggml_init_params ip{};
    ip.mem_size  = 16u * 1024u * 1024u;
    ip.no_alloc  = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) { why = "ggml_init"; return false; }

    ggml_tensor * qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads, 1);
    ggml_tensor * kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, kNRows, kKvHeads, 1);
    ggml_tensor * vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, kNRows, kKvHeads, 1);
    ggml_tensor * mt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kNkv, 1, 1, 1);
    ggml_tensor * rt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kNkv, 1);
    ggml_set_input(qt);
    ggml_set_input(kt);
    ggml_set_input(vt);
    ggml_set_input(mt);
    ggml_set_input(rt);

    ggml_tensor * o = ggml_flash_attn_ext_rows(ctx, qt, kt, vt, mt, rt, kScale, 0.0f, 0.0f);
    if (!o) { why = "ggml_flash_attn_ext_rows returned null"; ggml_free(ctx); return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) { why = "backend"; ggml_free(ctx); return false; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { why = "alloc"; ggml_backend_free(be); ggml_free(ctx); return false; }

    ggml_backend_tensor_set(qt, f.q.data(), 0, f.q.size() * sizeof(float));
    ggml_backend_tensor_set(kt, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(vt, f.v.data(), 0, f.v.size() * sizeof(float));
    ggml_backend_tensor_set(rt, p.rows.data(), 0, p.rows.size() * sizeof(std::int32_t));

    std::vector<ggml_fp16_t> half(static_cast<std::size_t>(kNkv), ggml_fp16_t{});
    ggml_fp32_to_fp16_row(f.mask.data(), half.data(), std::int64_t(kNkv));
    ggml_backend_tensor_set(mt, half.data(), 0, half.size() * sizeof(ggml_fp16_t));

    bool ok = false;
    const ggml_status st = ggml_backend_graph_compute(be, gf);
    if (st != GGML_STATUS_SUCCESS) {
        why = "compute status " + std::to_string(int(st));
    } else if (o->ne[0] != kHeadDim || o->ne[1] != kQHeads || o->ne[2] != 1 || o->ne[3] != 1) {
        // MEASURED, not assumed: the operator returns [n_embd_v, n_head, n_batch, ne3] -- the head
        // axis is SECOND. An earlier version of this check expected [n_embd, n_batch, n_head] and
        // the run reported [8 4 1 1], which is exactly the transposition of that expectation. The
        // flat order is therefore `d + head_dim*head`, which is the emitter's layout; a caller that
        // assumed batch-before-head would read the right numbers in the wrong order.
        why = "output is [" + std::to_string(o->ne[0]) + " " + std::to_string(o->ne[1]) + " " +
              std::to_string(o->ne[2]) + " " + std::to_string(o->ne[3]) + "]";
    } else {
        out.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        std::memcpy(out.data(), o->data, out.size() * sizeof(float));
        ok = true;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return ok;
}

// The check whose absence made this test red for the wrong reason. With a 5-row cache and a row
// list naming cell 6, both the reference and the operator indexed past k/v -- kv head 0 happened
// to land in bounds and kv head 1 did not, so the failure looked like an operator that mishandles
// GQA. It was the fixture. A row list is data; the bound is a property of the cache, and nothing
// else in this file connects them.
//
// It RETURNS the complaint instead of recording it, and the caller stops. That is not style: a
// `check` that only tallies a failure leaves this function returning normally, and the very next
// line builds the fixture and runs the operator -- with the out-of-range row still in the plan.
// The guard would then have reported the danger and permitted it anyway, which is worse than not
// checking, because the log would say the fixture was known-bad while k/v was still read past its
// end. Nothing downstream may run on a plan this function rejects.
std::string rows_fit_the_cache(const plan & p) {
    if (p.rows.size() != std::size_t(kNkv)) {
        return "the row list and the mask must describe the same columns: " +
               std::to_string(p.rows.size()) + " vs " + std::to_string(kNkv);
    }
    if (p.bias.size() != std::size_t(kNkv)) {
        return "the bias vector and the mask must describe the same columns: " +
               std::to_string(p.bias.size()) + " vs " + std::to_string(kNkv);
    }
    for (std::size_t i = 0; i < p.rows.size(); ++i) {
        const std::int32_t r = p.rows[i];
        const float bias = p.bias[i];
        // "Non-finite" is not the test, and "+inf or NaN is fine because it is not finite" is the
        // exact mistake this line used to make. ggml.h says a padding column's mask "must be -inf",
        // so -inf is the ONLY admissible value: +inf would mean "infinitely preferred", and NaN
        // would make every comparison in the softmax false. Both are accepted by `!isfinite` and
        // both would be carried into the reference, whose own `isfinite` test would then silently
        // treat the column as padding too -- so the two sides would agree while the plan violated
        // the contract. The check is therefore `isinf && < 0`.
        if (r < 0) {
            const bool is_neg_inf = std::isinf(bias) && bias < 0.0f;
            if (!is_neg_inf) {
                return "row list entry " + std::to_string(i) + " is padding (" + std::to_string(r) +
                       ") but its mask is " + std::to_string(bias) + ", not -inf";
            }
            continue;
        }
        // A non-negative row is a real cell, so its mask must be a real number: a bias may be
        // positive or negative, but inf or NaN is not a bias and the reference cannot represent it.
        // Without this, a NaN here would make one column's score NaN, `max_s` NaN, and every weight
        // NaN -- and the operator's output would differ from the reference for a reason that has
        // nothing to do with the operator.
        if (!std::isfinite(bias)) {
            return "row list entry " + std::to_string(i) + " is a real cell (" + std::to_string(r) +
                   ") but its mask is " + std::to_string(bias) + ", not a finite bias";
        }
        if (r >= kNRows) {
            return "row list entry " + std::to_string(i) + " names cell " + std::to_string(r) +
                   " but the cache holds " + std::to_string(kNRows) +
                   "; the reference and the operator would both read past k/v";
        }
    }
    return {};
}

void the_operator_honours_the_documented_contract() {
    const plan p = the_plan();
    // Hard gate: no fixture, no reference, no operator until the plan is known to be in bounds.
    const std::string plan_problem = rows_fit_the_cache(p);
    if (!plan_problem.empty()) {
        check(false, "fixture rejected before any computation: " + plan_problem);
        return;
    }
    fixture f(p);

    std::vector<float> want, want_wrong;
    reference(f, p, /*wrong_grouping=*/false, want);
    reference(f, p, /*wrong_grouping=*/true, want_wrong);

    std::vector<float> got;
    std::string why;
    if (!run_operator(f, p, got, why)) {
        check(false, "ggml_flash_attn_ext_rows on the CPU: " + why);
        return;
    }

    // Both sides must be the shape the operator promises BEFORE anything is compared. The compare
    // loop below is bounded by `min(got.size(), want.size())`, so a short or empty reference makes
    // it visit fewer elements and return a small worst-difference -- a green verdict from a
    // comparison that never happened. Checking the sizes first turns that into a named failure, and
    // it also makes "every element was compared" a consequence rather than a hope.
    const std::size_t expected = std::size_t(kHeadDim) * kQHeads;
    check(got.size() == expected, "operator produced " + std::to_string(got.size()) +
                                      " values, expected " + std::to_string(expected));
    check(want.size() == expected, "reference produced " + std::to_string(want.size()) +
                                       " values, expected " + std::to_string(expected));
    if (got.size() != expected || want.size() != expected) return;

    double worst = 0.0;
    int worst_at = -1;
    bool reported = false;
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
        if (!std::isfinite(got[i])) {
            if (!reported) {
                reported = true;
                std::fprintf(stderr, "non-finite at %zu; operator:", i);
                for (float x : got) std::fprintf(stderr, " %.6g", double(x));
                std::fprintf(stderr, "\n  reference:");
                for (float x : want) std::fprintf(stderr, " %.6g", double(x));
                std::fprintf(stderr, "\n");
            }
            check(false, "non-finite output at " + std::to_string(i));
            break;
        }
        const double d = std::fabs(double(got[i]) - double(want[i]));
        if (d > worst) { worst = d; worst_at = int(i); }
    }
    check(worst < 1e-5, "rows-indirected attention matches the definition: max|diff| = " +
                            std::to_string(worst) + " at index " + std::to_string(worst_at));

    // Non-vacuity, three ways. Without these the comparison could pass on a degenerate fixture.
    double magnitude = 0.0, grouping_gap = 0.0;
    for (float x : want) magnitude = std::max(magnitude, std::fabs(double(x)));
    for (std::size_t i = 0; i < want.size(); ++i)
        grouping_gap = std::max(grouping_gap, std::fabs(double(want[i]) - double(want_wrong[i])));
    check(magnitude > 1e-2, "output is not degenerate (max |value| = " + std::to_string(magnitude) + ")");
    check(grouping_gap > 1e-3, "h/group differs from h%kvh here by " + std::to_string(grouping_gap) +
                                   " -- a wrong grouping would be visible");

    // The mask must have been USED as a bias: recompute with the two finite biases zeroed. If the
    // operator ignored finite bias, both runs would agree and this assertion would fail.
    plan no_bias = p;
    for (float & x : no_bias.bias) if (std::isfinite(x)) x = 0.0f;
    fixture f2(no_bias);
    std::vector<float> want2, got2;
    reference(f2, no_bias, false, want2);
    if (!run_operator(f2, no_bias, got2, why)) {
        check(false, "second run: " + why);
        return;
    }
    // Both sizes again, and each element checked finite on BOTH sides. The earlier version compared
    // `min(got2.size(), want2.size())` elements and never looked at finiteness: a NaN in either array
    // makes every `fabs(a - b) > max` comparison false, so `second_worst` stays 0 and the run reads
    // as agreement. A bias-free reference that produced 4 values instead of 32 would also have
    // passed, because the loop simply stopped early.
    check(got2.size() == expected, "bias-free operator produced " + std::to_string(got2.size()) +
                                       " values, expected " + std::to_string(expected));
    check(want2.size() == expected, "bias-free reference produced " + std::to_string(want2.size()) +
                                        " values, expected " + std::to_string(expected));
    if (got2.size() != expected || want2.size() != expected) return;
    double bias_effect = 0.0;
    for (std::size_t i = 0; i < got2.size(); ++i) {
        if (!std::isfinite(got2[i]) || !std::isfinite(got[i])) {
            check(false, "bias comparison needs finite values at index " + std::to_string(i) +
                             ": bias-free " + std::to_string(got2[i]) + ", biased " +
                             std::to_string(got[i]));
            return;
        }
        bias_effect = std::max(bias_effect, std::fabs(double(got2[i]) - double(got[i])));
    }
    check(bias_effect > 1e-4, "finite mask entries change the result by " +
                                  std::to_string(bias_effect) +
                                  " -- so the mask is a bias, not a boolean");
    double second_worst = 0.0;
    int second_worst_at = -1;
    for (std::size_t i = 0; i < got2.size(); ++i) {
        if (!std::isfinite(got2[i]) || !std::isfinite(want2[i])) {
            check(false, "bias-free comparison needs finite values at index " + std::to_string(i) +
                             ": operator " + std::to_string(got2[i]) + ", reference " +
                             std::to_string(want2[i]));
            return;
        }
        const double d = std::fabs(double(got2[i]) - double(want2[i]));
        if (d > second_worst) { second_worst = d; second_worst_at = int(i); }
    }
    check(second_worst < 1e-5, "and the bias-free run matches the definition too: max|diff| = " +
                                   std::to_string(second_worst) + " at index " +
                                   std::to_string(second_worst_at));

    std::fprintf(stderr,
                 "flash-attn rows: max|diff| = %.3e (worst %d), |out|max = %.3e, "
                 "grouping gap = %.3e, bias effect = %.3e\n",
                 worst, worst_at, magnitude, grouping_gap, bias_effect);
}

}  // namespace

int main() {
    the_operator_honours_the_documented_contract();

    if (failures == 0) {
        std::fprintf(stderr, "flash-attn rows: OK\n");
        return 0;
    }
    std::fprintf(stderr, "flash-attn rows: %d failure(s)\n", failures);
    return 1;
}
