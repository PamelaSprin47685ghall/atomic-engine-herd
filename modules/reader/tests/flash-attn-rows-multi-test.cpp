// Several query tokens over a RUNTIME row table, through upstream's own operator, against the
// definition written per (token, head).
//
// WHY THIS IS A SEPARATE FILE FROM `flash-attn-rows-test`
// -------------------------------------------------------
// That one is single-token: `q` is `[head_dim, 1, n_head]`, the mask is `[n_kv, 1, 1, 1]`, and the
// output is `[n_embd_v, n_head, 1, 1]`. Every property it establishes holds for exactly one query
// row. A prefill is T rows, and the row table's indirection is shared by all of them while the MASK
// is per (column, token) -- so the two things a multi-token call can get wrong (reading one token's
// visibility for another; indexing the row table per token instead of per column) are invisible to a
// single-token test.
//
// WHAT IS MEASURED RATHER THAN ARGUED
// -----------------------------------
//   1. Two tokens see DIFFERENT visible sets, and a column visible to token 2 alone carries weight
//      for token 2 and exactly zero for token 0. A broadcast row cannot produce that.
//   2. The row list is NON-IDENTITY and NON-MONOTONIC (`{5,-1,2,7,-1,0}`) with padding INTERLEAVED
//      (columns 1 and 4). A reader that treated column i as row i would read different values, and
//      the numeric comparison would fail.
//   3. Padding columns carry `-inf` for every token, which is the contract the CPU kernel relies on
//      when it clamps a negative row to row 0 (`ggml-cpu/ops.cpp:` `kv_row(i) = max(rows[i], 0)`).
//   4. The output layout is `[n_embd_v, n_head, n_batch, ne3]` -- head axis SECOND, batch THIRD --
//      with flat index `d + n_embd_v*(h + n_head*t)`. That was measured on the single-token case and
//      is asserted again here for T > 1, because the two orders coincide when T == 1 and therefore a
//      single-token test cannot distinguish them.
//
// WHAT THIS IS NOT
// ----------------
// Not a replacement, not a seam, and not evidence that a lane is wired. It drives one operator in a
// test process. It also cannot make the operator safe for arbitrary row VALUES: a non-negative row at
// or past `k->ne[1]` is an out-of-bounds read the kernel does not check, which is why the emitter
// that wrapped this operator was removed (HANDOFF §7.26) and why validation exists at all (§7.27-29).
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
constexpr int kQHeads  = 4;   // group = 2
constexpr int kRows     = 8;  // k/v rows available; the table names any of them
constexpr int kNkv      = 6;  // mask columns == kv_rows length
constexpr int kTokens   = 3;
constexpr float kScale  = 0.5f;

// Non-identity, non-monotonic, padding interleaved. Every non-negative entry is < kRows, and the two
// negative entries are padding whose mask columns must be -inf for every token.
const std::int32_t kRowTable[kNkv] = {5, -1, 2, 7, -1, 0};
const bool kIsPadding[kNkv] = {false, true, false, false, true, false};

struct fixture {
    std::vector<float> q;     // [head_dim, T, n_head]   idx d + hd*(t + T*h)
    std::vector<float> k;     // [head_dim, kRows, n_head_kv] idx d + hd*(row + kRows*kvh)
    std::vector<float> v;     // same
    std::vector<float> mask;  // [kNkv, T] F32 here; stored as F16 in the tensor   idx c + kNkv*t

    fixture() {
        q.assign(std::size_t(kHeadDim) * kTokens * kQHeads, 0.0f);
        k.assign(std::size_t(kHeadDim) * kRows * kKvHeads, 0.0f);
        v.assign(k.size(), 0.0f);
        mask.assign(std::size_t(kNkv) * kTokens, 0.0f);
        unsigned s = 0x9e3779b9u;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : q) x = next();
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 2.0f;

        // Per-token visibility, chosen so the tokens are NOT nested: token 0 sees columns {0,2},
        // token 1 sees {2,3,5}, token 2 sees {0,3,5}. Nested sets would let a "sees at least what
        // token 0 sees" implementation pass; these do not nest.
        const bool visible[kTokens][kNkv] = {
            {true,  false, true,  false, false, false},
            {false, false, true,  true,  false, true},
            {true,  false, false, true,  false, true},
        };
        // Finite biases differ per token so "the mask's token axis is read" is observable even for a
        // column both tokens can see.
        const float bias[kTokens] = {0.25f, -0.5f, 1.0f};
        for (int t = 0; t < kTokens; ++t) {
            for (int c = 0; c < kNkv; ++c) {
                if (kIsPadding[c]) {
                    // The contract: padding is invisible to EVERY token. A finite value here would
                    // be a violation the validator refuses, and the kernel would mix row 0's value in.
                    mask[std::size_t(c) + std::size_t(kNkv) * t] = -INFINITY;
                } else {
                    mask[std::size_t(c) + std::size_t(kNkv) * t] =
                        visible[t][c] ? bias[t] + 0.1f * float(c) : -INFINITY;
                }
            }
        }
        // THE MASK IS ROUNDED THROUGH F16 BEFORE ANYONE READS IT.
        //
        // The tensor holds F16 (the width the CPU kernel asserts), so the operator sees the half-
        // precision value of each bias. A reference that used the F32 value would compare against
        // numbers the operator never had, and the difference is not noise: a bias near 1.0 carries
        // about 5e-4 relative error in F16, which moves the softmax weights and shows up as ~6e-5 in
        // the output. Measured on the first run of this file: max|diff| = 6.032e-05 with the F32
        // reference, which is exactly that rounding and nothing else.
        for (float & x : mask) {
            if (!std::isfinite(x)) continue;   // -inf survives the round trip and must stay -inf
            ggml_fp16_t h{};
            ggml_fp32_to_fp16_row(&x, &h, 1);
            ggml_fp16_to_fp32_row(&h, &x, 1);
        }
    }
    float Q(int d, int t, int h) const { return q[std::size_t(d) + kHeadDim * (t + kTokens * h)]; }
    float K(int d, int row, int kvh) const { return k[std::size_t(d) + kHeadDim * (row + kRows * kvh)]; }
    float V(int d, int row, int kvh) const { return v[std::size_t(d) + kHeadDim * (row + kRows * kvh)]; }
    float M(int c, int t) const { return mask[std::size_t(c) + std::size_t(kNkv) * t]; }
};

// The definition: for (token, head), softmax over the COLUMNS whose mask is finite, with the value
// read through the row table. `d + n_embd_v*(h + n_head*t)` is the operator's measured order.
void reference(const fixture & f, std::vector<float> & out) {
    const int group = kQHeads / kKvHeads;
    out.assign(std::size_t(kHeadDim) * kQHeads * kTokens, 0.0f);
    for (int t = 0; t < kTokens; ++t) {
        for (int h = 0; h < kQHeads; ++h) {
            const int kvh = h / group;
            std::vector<double> score(static_cast<std::size_t>(kNkv), 0.0);
            double max_s = -INFINITY;
            for (int c = 0; c < kNkv; ++c) {
                const double b = double(f.M(c, t));
                if (!std::isfinite(b)) {
                    score[std::size_t(c)] = -INFINITY;
                    continue;
                }
                // THE INDIRECTION: column c uses row kRowTable[c]. A test that read row c instead
                // would compare against a different tensor, and the numbers would move.
                const int row = int(kRowTable[c]);
                double s = 0.0;
                for (int d = 0; d < kHeadDim; ++d) s += double(f.Q(d, t, h)) * double(f.K(d, row, kvh));
                score[std::size_t(c)] = s * double(kScale) + b;
                if (score[std::size_t(c)] > max_s) max_s = score[std::size_t(c)];
            }
            double sum = 0.0;
            for (int c = 0; c < kNkv; ++c) {
                if (!std::isfinite(score[std::size_t(c)])) continue;
                score[std::size_t(c)] = std::exp(score[std::size_t(c)] - max_s);
                sum += score[std::size_t(c)];
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (int c = 0; c < kNkv; ++c) {
                    if (!std::isfinite(score[std::size_t(c)]) || score[std::size_t(c)] == 0.0) continue;
                    acc += (score[std::size_t(c)] / sum) * double(f.V(d, int(kRowTable[c]), kvh));
                }
                out[std::size_t(d) + std::size_t(kHeadDim) *
                                         (std::size_t(h) + std::size_t(kQHeads) * std::size_t(t))] =
                    float(acc);
            }
        }
    }
}

bool run_operator(const fixture & f, const std::vector<std::int32_t> & rows,
                  std::vector<float> & out, std::string & why) {
    ggml_init_params ip{};
    ip.mem_size = 16u * 1024u * 1024u;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { why = "ggml_init"; return false; }

    // q: [n_embd_k, n_batch, n_head, ne3] -- n_batch is the TOKEN axis, per the operator's contract.
    ggml_tensor * qt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, kTokens, kQHeads, 1);
    ggml_tensor * kt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, kRows, kKvHeads, 1);
    ggml_tensor * vt = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, kRows, kKvHeads, 1);
    // mask: [n_kv, n_batch, ne32, ne3], F16 -- the CPU kernel asserts that width.
    ggml_tensor * mt = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kNkv, kTokens, 1, 1);
    // kv_rows: [n_kv, ne3] I32; ne[0] must equal the mask's ne[0].
    ggml_tensor * rt = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kNkv, 1);
    ggml_set_input(qt);
    ggml_set_input(kt);
    ggml_set_input(vt);
    ggml_set_input(mt);
    ggml_set_input(rt);

    ggml_tensor * o = ggml_flash_attn_ext_rows(ctx, qt, kt, vt, mt, rt, kScale, 0.0f, 0.0f);
    if (o == nullptr) { why = "operator returned null"; ggml_free(ctx); return false; }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);

    ggml_backend_t be = ggml_backend_cpu_init();
    if (be == nullptr) { why = "backend"; ggml_free(ctx); return false; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == nullptr) { why = "alloc"; ggml_backend_free(be); ggml_free(ctx); return false; }

    ggml_backend_tensor_set(qt, f.q.data(), 0, f.q.size() * sizeof(float));
    ggml_backend_tensor_set(kt, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(vt, f.v.data(), 0, f.v.size() * sizeof(float));
    ggml_backend_tensor_set(rt, rows.data(), 0, rows.size() * sizeof(std::int32_t));
    {
        std::vector<ggml_fp16_t> half(f.mask.size());
        ggml_fp32_to_fp16_row(f.mask.data(), half.data(), std::int64_t(f.mask.size()));
        ggml_backend_tensor_set(mt, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    }

    bool ok = false;
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        why = "compute failed";
    } else if (o->ne[0] != kHeadDim || o->ne[1] != kQHeads || o->ne[2] != kTokens || o->ne[3] != 1) {
        // MEASURED LAYOUT: [n_embd_v, n_head, n_batch, ne3]. When kTokens == 1 the head and batch
        // axes coincide in extent, which is why only a multi-token run can tell them apart.
        why = "output is [" + std::to_string(o->ne[0]) + " " + std::to_string(o->ne[1]) + " " +
              std::to_string(o->ne[2]) + " " + std::to_string(o->ne[3]) +
              "], expected [head_dim, n_head, n_tokens, 1]";
    } else {
        out.assign(std::size_t(kHeadDim) * kQHeads * kTokens, 0.0f);
        std::memcpy(out.data(), o->data, out.size() * sizeof(float));
        ok = true;
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return ok;
}

void multi_token_rows_match_the_definition() {
    const fixture f;
    std::vector<std::int32_t> rows(std::begin(kRowTable), std::end(kRowTable));
    std::vector<float> want;
    reference(f, want);

    std::vector<float> got;
    std::string why;
    if (!run_operator(f, rows, got, why)) {
        check(false, "multi-token rows attention: " + why);
        return;
    }

    double worst = 0.0;
    int at = -1;
    for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
        if (!std::isfinite(got[i])) { check(false, "non-finite at " + std::to_string(i)); return; }
        const double d = std::fabs(double(got[i]) - double(want[i]));
        if (d > worst) { worst = d; at = int(i); }
    }
    check(worst < 1e-5, "multi-token rows attention matches the definition: max|diff| = " +
                            std::to_string(worst) + " at " + std::to_string(at));

    // Non-vacuity, four ways. Without these the comparison could pass on a degenerate fixture.
    double magnitude = 0.0;
    for (float x : want) magnitude = std::max(magnitude, std::fabs(double(x)));
    check(magnitude > 1e-2, "the reference is non-degenerate (max |value| = " +
                                std::to_string(magnitude) + ")");

    // (i) tokens differ: the token axis is real, not a broadcast.
    double token_gap = 0.0;
    for (int h = 0; h < kQHeads; ++h)
        for (int d = 0; d < kHeadDim; ++d) {
            const double a = double(want[std::size_t(d) + kHeadDim * (h + kQHeads * 0)]);
            const double c = double(want[std::size_t(d) + kHeadDim * (h + kQHeads * 2)]);
            token_gap = std::max(token_gap, std::fabs(a - c));
        }
    check(token_gap > 1e-3, "token 0 and token 2 differ by " + std::to_string(token_gap) +
                                " -- their visible sets differ, so a broadcast mask would erase this");

    // (ii) heads differ under GQA: h/group vs h%kvh would swap two heads' answers.
    double head_gap = 0.0;
    for (int t = 0; t < kTokens; ++t)
        for (int d = 0; d < kHeadDim; ++d) {
            const double a = double(want[std::size_t(d) + kHeadDim * (1 + kQHeads * t)]);
            const double c = double(want[std::size_t(d) + kHeadDim * (2 + kQHeads * t)]);
            head_gap = std::max(head_gap, std::fabs(a - c));
        }
    check(head_gap > 1e-3, "heads 1 and 2 differ by " + std::to_string(head_gap) +
                               " -- a grouped-vs-modulo mistake would move a number");

    // (iii) THE INDIRECTION IS USED. Recomputing with row c instead of row kRowTable[c] must give a
    //       different answer; otherwise "matches the definition" says nothing about the row table.
    {
        fixture wrong(f);
        // Rewrite the table as identity over the visible columns only: same shapes, same mask, and a
        // genuinely different set of value rows.
        std::vector<std::int32_t> identity(kNkv);
        for (int c = 0; c < kNkv; ++c) identity[std::size_t(c)] = kIsPadding[c] ? -1 : c;
        std::vector<float> other;
        std::string why2;
        if (!run_operator(wrong, identity, other, why2)) {
            check(false, "identity-row control run: " + why2);
        } else {
            double gap = 0.0;
            for (std::size_t i = 0; i < other.size() && i < want.size(); ++i)
                gap = std::max(gap, std::fabs(double(other[i]) - double(want[i])));
            check(gap > 1e-3, "reading row c instead of row kv_rows[c] changes the result by " +
                                  std::to_string(gap) + " -- so the indirection is exercised");
        }
    }

    std::fprintf(stderr,
                 "flash-attn rows multi: max|diff| = %.3e (worst %d), |out|max = %.3e, "
                 "token gap = %.3e, head gap = %.3e, %d tokens x %d heads over %d columns\n",
                 worst, at, magnitude, token_gap, head_gap, kTokens, kQHeads, kNkv);
}

} // namespace

int main() {
    multi_token_rows_match_the_definition();

    if (failures == 0) {
        std::fprintf(stderr, "flash-attn rows multi: OK\n");
        return 0;
    }
    std::fprintf(stderr, "flash-attn rows multi: %d failure(s)\n", failures);
    return 1;
}
