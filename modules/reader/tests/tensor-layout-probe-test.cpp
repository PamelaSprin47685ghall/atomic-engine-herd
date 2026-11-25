// Two ggml layout facts a multi-token emission depends on, asserted instead of argued.
//
// Why this file exists. A prefill emitter was written twice and removed twice. The first removal
// was a collapsed object count (an admission bound that under-charged its own gathers). The second
// was an unexplained 0.49 difference in `out`, and in that case the LAYOUT had been inferred from
// ggml's documentation rather than printed. So the layout was measured, once, and the two claims
// that had been treated as one are separate:
//
//   concat(a, b, dim = 1) on two [hd, 1, nh] operands
//        gives `d + hd*(t + T*h)`  -- TOKEN-major across the seam. For THIS shape and THESE
//        operands. ggml defines dim ordering per axis, so this is expected to generalise, but the
//        measurement covers one shape; it is not a claim about every operand.
//
//   cont(permute(q, 0, 2, 1, 3)) on a [hd, T, nh] tensor
//        gives `d + hd*(h + nh*t)`  -- HEAD-major, i.e. token blocks are contiguous. It is NOT
//        `d + hd*(t + T*h)`.
//
// THE THIRD PARAGRAPH THIS FILE USED TO HAVE, AND MUST NOT HAVE. It said the second fact "explains"
// the 0.49 difference. It cannot: that emitter is deleted, so no experiment can now attribute the
// difference to it, and an explanation that cannot be checked is exactly the kind of confident
// prose this repository keeps having to retract. What is true is narrower and still useful -- the
// two orders DIFFER while the tensors have identical `ne`, so any code that mixes them produces a
// wrong number from the wrong token and no assertion fires. That is a hazard worth pinning with a
// test; it is not a diagnosis of a program nobody can run.
//
// The `differs` check below therefore asserts the hazard, not the cause: if these two ever became
// the same order, every adapter that distinguishes them would be wrong, and the test says so rather
// than letting the distinction live on in prose.
#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <cstdio>
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

// token t, head h, dim d -> 100*t + 10*h + d. Every value distinct, so a wrong axis is a wrong
// number rather than a coincidence.
float cell_value(int t, int h, int d) { return float(100 * t + 10 * h + d); }

constexpr int kHeadDim = 2;
constexpr int kHeads   = 2;
constexpr int kTokens  = 3;

void concat_dim1_is_token_major(const ggml_tensor * cat) {
    // Expected flat order: d + hd*(t + T*h)  =>  for h, then t, then d.
    std::vector<float> want;
    for (int h = 0; h < kHeads; ++h)
        for (int t = 0; t < kTokens; ++t)
            for (int d = 0; d < kHeadDim; ++d) want.push_back(cell_value(t, h, d));

    const float * got = static_cast<const float *>(cat->data);
    for (std::size_t i = 0; i < want.size(); ++i) {
        check(got[i] == want[i], "concat dim 1: element " + std::to_string(i) + " is " +
                                     std::to_string(got[i]) + ", expected " +
                                     std::to_string(want[i]) +
                                     " -- the seam is not token-major for (h,t,d) order");
    }
}

void permute_then_cont_is_head_major(const ggml_tensor * pc) {
    // Expected flat order: d + hd*(h + nh*t)  =>  for t, then h, then d.
    std::vector<float> want;
    for (int t = 0; t < kTokens; ++t)
        for (int h = 0; h < kHeads; ++h)
            for (int d = 0; d < kHeadDim; ++d) want.push_back(cell_value(t, h, d));

    const float * got = static_cast<const float *>(pc->data);
    for (std::size_t i = 0; i < want.size(); ++i) {
        check(got[i] == want[i], "permute+cont: element " + std::to_string(i) + " is " +
                                     std::to_string(got[i]) + ", expected " +
                                     std::to_string(want[i]));
    }

    // And the counter-claim, asserted so that "the two orders are the same" can never be assumed:
    // the token-major order concat produces must NOT match this tensor.
    std::vector<float> token_major;
    for (int h = 0; h < kHeads; ++h)
        for (int t = 0; t < kTokens; ++t)
            for (int d = 0; d < kHeadDim; ++d) token_major.push_back(cell_value(t, h, d));
    bool differs = false;
    for (std::size_t i = 0; i < token_major.size(); ++i) {
        if (got[i] != token_major[i]) differs = true;
    }
    check(differs, "permute+cont produces the SAME flat order as a dim-1 concat; if that ever "
                   "becomes true, every adapter that treats them as different is wrong");
}

}  // namespace

int main() {
    ggml_init_params ip{};
    ip.mem_size = 8u * 1024u * 1024u;
    ip.no_alloc = true;  // create every tensor, THEN place: a tensor made after allocation has no data
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        std::fprintf(stderr, "FAIL: ggml_init\n");
        return 1;
    }

    // One [hd,1,nh] operand per token, the shape a single-token emission returns.
    std::vector<ggml_tensor *> per_token;
    for (int t = 0; t < kTokens; ++t) {
        ggml_tensor * o = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kHeads);
        ggml_set_input(o);
        per_token.push_back(o);
    }
    ggml_tensor * cat = per_token[0];
    for (int t = 1; t < kTokens; ++t) cat = ggml_concat(ctx, cat, per_token[t], 1);

    ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kTokens, kHeads);
    ggml_set_input(q);
    ggml_tensor * pc = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));

    // The buffer handle is KEPT. `ggml_backend_alloc_ctx_tensors` returns the buffer it created, and
    // the earlier version discarded that return value in a `!` test: the memory was then never
    // released, and the failure path freed the context while leaving the backend alive. A test that
    // leaks on success and on failure is a test whose resource behaviour nobody has looked at.
    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = be ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
    if (be == nullptr || buf == nullptr) {
        std::fprintf(stderr, "FAIL: placement\n");
        // `buf` is null here by construction (the probe is what failed), so only the two live
        // handles are released -- and the backend is released even when the allocation failed,
        // because it was successfully created before the failure.
        if (buf != nullptr) ggml_backend_buffer_free(buf);
        if (be != nullptr) ggml_backend_free(be);
        ggml_free(ctx);
        return 1;
    }

    for (int t = 0; t < kTokens; ++t) {
        std::vector<float> v(std::size_t(kHeadDim) * kHeads);
        for (int h = 0; h < kHeads; ++h)
            for (int d = 0; d < kHeadDim; ++d) v[std::size_t(d + kHeadDim * h)] = cell_value(t, h, d);
        ggml_backend_tensor_set(per_token[t], v.data(), 0, v.size() * sizeof(float));
    }
    {   // q as the single-query emitter's caller would receive it: [hd, T, nh], head-major inside a token
        std::vector<float> v(std::size_t(kHeadDim) * kTokens * kHeads);
        for (int h = 0; h < kHeads; ++h)
            for (int t = 0; t < kTokens; ++t)
                for (int d = 0; d < kHeadDim; ++d)
                    v[std::size_t(d + kHeadDim * (t + kTokens * h))] = cell_value(t, h, d);
        ggml_backend_tensor_set(q, v.data(), 0, v.size() * sizeof(float));
    }

    // A compute failure is a FAILURE, not a note. The earlier version printed and continued, so a
    // graph that never ran left `failures` at 0 and the process exited 0 -- a green result from a
    // comparison that never happened. `check` is what makes the exit code carry it.
    ggml_cgraph * g1 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g1, cat);
    if (ggml_backend_graph_compute(be, g1) != GGML_STATUS_SUCCESS) {
        check(false, "the concat graph did not compute, so its layout was never read");
    } else {
        concat_dim1_is_token_major(cat);
    }

    ggml_cgraph * g2 = ggml_new_graph(ctx);
    ggml_build_forward_expand(g2, pc);
    if (ggml_backend_graph_compute(be, g2) != GGML_STATUS_SUCCESS) {
        check(false, "the permute graph did not compute, so its layout was never read");
    } else {
        permute_then_cont_is_head_major(pc);
    }

    // Buffer first, then the backend that owns its allocation, then the context whose tensors the
    // buffer held. Each step releases what the previous one still refers to.
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);

    if (failures == 0) {
        std::fprintf(stderr, "tensor layout probe: OK\n");
        return 0;
    }
    std::fprintf(stderr, "tensor layout probe: %d failure(s)\n", failures);
    return 1;
}
