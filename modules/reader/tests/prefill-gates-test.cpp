// The three preflight gates that exist to keep a STRUCTURAL refusal out of the node-building phase,
// each driven with a real, legal ggml fixture.
//
// WHY THESE THREE, AND WHY THEY NEED A TEST AT ALL
// ------------------------------------------------
// `emit_prefill_attention` builds `cont(permute(q, 0, 2, 1, 3))` before it emits anything, so a
// condition the per-token emitter would refuse is refused too late: `q_head_major` already exists,
// the caller owns a half-built context, and the verdict arrives as `inner_refusal` -- which names a
// symptom rather than the cause. Three such conditions were added to the preflight after that was
// noticed, and none of them had a test. A gate nobody has driven is a gate nobody knows works:
//
//   no_alloc_context    an allocating context. The per-token emitter refuses it because it can bound
//                       OBJECTS but not the data regions an allocating graph also owns.
//   bad_view            spans disagreeing on `effective_pos`. One view is shared by every token, so
//                       the disagreement would be refused identically T times, after the reorder.
//   bad_shape           a CELL-MAJOR `v` (`nb[1] > nb[2]`). The value reduction permutes v into
//                       `[head_dim, n_cell, n_head_kv]`, which needs a dim-major cache.
//
// Each case asserts the refusal CODE and that the call grew the arena by ZERO BYTES. The second half
// is the actual claim: "refused before a node existed" is a statement about `ggml_used_mem`, and a
// gate placed one line too late would still return the right code while having built half a graph.
//
// A well-formed control case runs first in the same file. Without it, a fixture that accidentally
// made EVERY case refuse (a mis-sized q, say) would satisfy all three assertions.
//
// ONE DIVERGENCE IS ASSERTED RATHER THAN HIDDEN: for a cell-major `v`, this entry point reports
// `bad_shape` while `emit_physical_attention` reports its own `v_layout_cell_major`. The two codes
// describe one condition, and a caller that switches entry points sees a different value. It is
// recorded here (and in the header's enum comment) because silently accepting that two entry points
// disagree is how a caller ends up retrying the wrong thing.
#include <tq/reader/graph_emit_prefill.h>
#include <tq/reader/upstream_view.h>

#include <ggml.h>

#include <cstdio>
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
constexpr int kHeads   = 4;   // group = 2
constexpr int kCell    = 16;
constexpr int kTokens  = 3;
constexpr float kScale = 0.35355339f;

// A view that names every cell exactly once, so it is legal under every gate below -- the cases
// differ only in the one field under test.
tq::reader::query_view full_view(int effective_pos = 5) {
    tq::reader::query_view v;
    tq::reader::span_keys span;
    span.effective_pos = effective_pos;
    for (int c = 0; c < kCell; ++c) span.cells.push_back(std::int32_t(c));
    v.spans.push_back(std::move(span));
    return v;
}

// The four fixture tensors, created in `ctx`. `v_cell_major` builds a v whose kill stride is larger
// than its cell stride: the one property the gate tests, and a layout ggml accepts as a legal view.
struct fixtures {
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * m = nullptr;
    ggml_tensor * v_parent = nullptr;   // keeps the cell-major view's storage alive

    bool build(ggml_context * ctx, bool v_cell_major) {
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kTokens, kHeads);
        m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, kTokens, 1, 1);
        if (k == nullptr || q == nullptr || m == nullptr) return false;
        if (!v_cell_major) {
            v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
            if (v == nullptr) return false;
        } else {
            // ne stays [head_dim, n_head_kv, n_cell]; the STRIDES are the cell-major ones:
            //   nb[1] = one cell plane (so the kv-head step dwarfs the cell step), nb[2] = one float.
            // A parent long enough for the resulting view; `ggml_view_3d` bounds-checks it, so a
            // mistake here is an assertion, not a silently wrong tensor.
            const std::size_t el = sizeof(float);
            v_parent = ggml_new_tensor_1d(
                ctx, GGML_TYPE_F32,
                std::int64_t(std::size_t(kCell) * std::size_t(kHeadDim) * std::size_t(kKvHeads)));
            if (v_parent == nullptr) return false;
            v = ggml_view_3d(ctx, v_parent, kHeadDim, kKvHeads, kCell,
                             std::size_t(kCell) * std::size_t(kHeadDim) * el, el, 0);
            if (v == nullptr) return false;
        }
        ggml_set_input(k);
        // NOT on `v` when it is a view: `ggml_set_input` asserts `op == GGML_OP_NONE`, and a
        // `ggml_view_3d` result carries `GGML_OP_VIEW`. The view's storage is the parent's, and the
        // parent is what a caller would fill -- so the parent is marked instead. (Measured: marking
        // the view aborted inside ggml_set_input, which is how this line was found.)
        if (!v_cell_major) ggml_set_input(v);
        ggml_set_input(q);
        ggml_set_input(m);
        return true;
    }
};

void fill(tq::reader::prefill_inputs & in, const fixtures & f, ggml_context * ctx) {
    in.ctx = ctx;
    in.k = f.k;
    in.v = f.v;
    in.q = f.q;
    in.mask = f.m;
    in.n_head = std::size_t(kHeads);
    in.n_head_kv = std::size_t(kKvHeads);
    in.head_dim = std::size_t(kHeadDim);
    in.n_cell = std::size_t(kCell);
    in.coverage = tq::reader::graph_inputs::mask_coverage::intersect;
}

// Runs one case in an allocating or no_alloc context and returns the growth across the call.
struct outcome {
    tq::reader::prefill_refusal refusal = tq::reader::prefill_refusal::ok;
    std::string why;
    std::size_t grew = 0;
    bool grew_valid = false;   // false when the context refused to be created at all
};

outcome run_case(bool v_cell_major, const tq::reader::query_view & view, bool no_alloc,
                 tq::reader::graph_inputs::mask_coverage coverage =
                     tq::reader::graph_inputs::mask_coverage::intersect) {
    outcome out;
    ggml_init_params ip{};
    ip.mem_size = 2u * 1024u * 1024u;
    ip.no_alloc = no_alloc;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        check(false, "ggml_init");
        return out;
    }
    fixtures f;
    if (!f.build(ctx, v_cell_major)) {
        check(false, "the fixture tensors could not be created");
        ggml_free(ctx);
        out.grew_valid = true;
        return out;
    }
    tq::reader::prefill_inputs in;
    fill(in, f, ctx);
    in.view = view;
    // AFTER `fill`, which sets its own default: the parameter is what this case is about, so it has
    // to win. (The first version applied it before the call and `fill` overwrote it, so the
    // unstated-intent case measured a call that had stated the intent -- the gate looked absent
    // while the mutation below showed it working. Ordering, not the gate.)
    in.coverage = coverage;

    const std::size_t before = ggml_used_mem(ctx);
    const tq::reader::prefill_result r = tq::reader::emit_prefill_attention(in, kScale);
    out.grew = ggml_used_mem(ctx) - before;
    out.grew_valid = true;
    out.refusal = r.refusal;
    out.why = r.why;
    ggml_free(ctx);
    return out;
}

// The shared assertion: the verdict, the code, and the fact that nothing was built.
void expect_refusal(const outcome & r, tq::reader::prefill_refusal want, const char * label) {
    check(r.grew_valid, std::string(label) + ": the context could not hold the fixture");
    check(r.refusal == want,
          std::string(label) + ": expected " + tq::reader::describe(want) + ", got " +
              tq::reader::describe(r.refusal) + " -- " + r.why);
    check(r.grew == 0,
          std::string(label) + ": the refusal built " + std::to_string(r.grew) +
              " bytes of nodes. This gate exists so that a structural refusal costs nothing; " +
              "a gate one line too late returns the same code after building half a graph.");
}

}  // namespace

// THE OFFSET ARITHMETIC IS GUARDED, and the guards are driven directly -- they are pure functions
// over numbers, so a case is a call rather than a constructed graph.
//
// Three properties, each of which a plausible implementation gets wrong:
//
//   `prefill_scale` stays in `size_t`. The query's per-token stride is `head_dim * n_head *
//   element`; computing the first factor and then narrowing it to `int64_t` (to reach
//   `prefill_slice_offset`'s token parameter) is IMPLEMENTATION-DEFINED once the product exceeds
//   `INT64_MAX`, and on this target it wraps negative. A test that only checked "a huge product is
//   refused" would pass on both implementations, because the wrapped value happens to trip the
//   helper's `t < 0`. So the assertions pin the ACCEPTING side at the exact boundary instead:
//   `SIZE_MAX / 8` squared times 8 is representable, so it must be accepted.
//
//   Both helpers refuse a null `out` rather than dereference it. Latent in every current caller
//   (each passes a local), which is why it would have surfaced as a crash in a future test rather
//   than as a wrong number.
//
//   A refusal leaves `*out` untouched. A caller that inspects the output after a false return sees
//   its own value, not a half-computed one.
void offset_arithmetic_is_guarded() {
    const std::size_t L = std::numeric_limits<std::size_t>::max();

    // prefill_scale: product, zero, boundary, overflow, null.
    {
        std::size_t v = 0;
        check(tq::reader::prefill_scale(8, 4, &v) && v == 32, "prefill_scale: 8 * 4 == 32");
        check(tq::reader::prefill_scale(0, L, &v) && v == 0,
              "prefill_scale: zero times anything is zero, even at SIZE_MAX");
        check(!tq::reader::prefill_scale(2, L / 2 + 2, &v), "prefill_scale: 2 * (SIZE_MAX/2 + 2) wraps");
        check(!tq::reader::prefill_scale(L / 8 + 1, 8, &v),
              "prefill_scale: (SIZE_MAX/8 + 1) * 8 wraps");
        // THE BOUNDARY THAT DISTINGUISHES THE TWO IMPLEMENTATIONS. A product above INT64_MAX but
        // below SIZE_MAX is a legal size_t multiply and must be ACCEPTED: only a version that
        // narrows to int64_t would refuse it, and it would do so by wrapping.
        check(tq::reader::prefill_scale(L / 8, 8, &v) && v == (L / 8) * 8,
              "prefill_scale: the largest representable product is accepted, not narrowed away");
        // THE CASE THAT DISTINGUISHES the two implementations is 2^63 -- above INT64_MAX (so a
        // narrowing version cannot represent it) yet a perfectly legal size_t. My first draft
        // multiplied it by 2 and expected acceptance: that was arithmetic, not code, that was wrong,
        // since 2^64 exceeds SIZE_MAX and must be refused. Both halves are asserted now.
        check(tq::reader::prefill_scale((L / 2) + 1, 1, &v) && v == (L / 2) + 1,
              "prefill_scale: 2^63 is accepted -- a version that narrowed to int64_t could not "
              "represent it");
        check(!tq::reader::prefill_scale((L / 2) + 1, 2, &v), "prefill_scale: 2^64 is refused");
        v = 12345;
        check(!tq::reader::prefill_scale(3, 4, nullptr), "prefill_scale: a null out is refused");
        check(v == 12345, "prefill_scale: and nothing was written");
    }

    // prefill_slice_offset: the multiply and its own preconditions.
    {
        std::size_t v = 0;
        check(tq::reader::prefill_slice_offset(3, 8, &v) && v == 24, "slice_offset: 3 * 8 == 24");
        check(tq::reader::prefill_slice_offset(0, 999, &v) && v == 0, "slice_offset: 0 * stride == 0");
        v = 7;
        check(!tq::reader::prefill_slice_offset(-1, 8, &v), "slice_offset: a negative token is refused");
        check(v == 7, "slice_offset: a refusal leaves out untouched");
        check(!tq::reader::prefill_slice_offset(2, L / 2 + 2, &v), "slice_offset: the product wraps");
        check(!tq::reader::prefill_slice_offset(std::int64_t(L / 4 + 1), 8, &v),
              "slice_offset: a token count whose product wraps is refused");
        check(tq::reader::prefill_slice_offset(std::int64_t(L / 8), 8, &v) && v == (L / 8) * 8,
              "slice_offset: the largest accepted offset is exact");
        v = 99;
        check(!tq::reader::prefill_slice_offset(1, 1, nullptr), "slice_offset: a null out is refused");
        check(v == 99, "slice_offset: and nothing was written");
    }
}

int main() {
    // (0) THE CONTROL. A well-formed call must be ADMITTED, or every refusal below proves only that
    //     the fixtures are broken. The arena is generous because this case builds a graph.
    {
        const outcome r = run_case(false, full_view(), true);
        check(r.grew_valid && r.refusal == tq::reader::prefill_refusal::ok,
              std::string("control: a well-formed call is admitted, got ") +
                  tq::reader::describe(r.refusal) + " -- " + r.why);
        check(r.grew > 0, "control: the admitted call built nodes (growth " +
                              std::to_string(r.grew) + " bytes), so the zero-growth refusals below " +
                              "measure a real difference");
    }

    // (1) AN ALLOCATING CONTEXT. `ggml_init` with no_alloc = false; the fixtures themselves allocate,
    //     which is why the growth is measured around the call and not around the context.
    {
        const outcome r = run_case(false, full_view(), /*no_alloc=*/false);
        expect_refusal(r, tq::reader::prefill_refusal::no_alloc_context, "allocating context");
    }

    // (2) SPANS THAT DISAGREE ON effective_pos. Two disjoint spans covering the cache, positions 3
    //     and 7. Each span alone is legal; the pair is what the gate refuses.
    {
        tq::reader::query_view mixed;
        tq::reader::span_keys a;
        a.effective_pos = 3;
        for (int c = 0; c < kCell / 2; ++c) a.cells.push_back(std::int32_t(c));
        tq::reader::span_keys b;
        b.effective_pos = 7;
        for (int c = kCell / 2; c < kCell; ++c) b.cells.push_back(std::int32_t(c));
        mixed.spans.push_back(std::move(a));
        mixed.spans.push_back(std::move(b));

        const outcome r = run_case(false, mixed, true);
        expect_refusal(r, tq::reader::prefill_refusal::bad_view, "mixed effective_pos");
        check(r.why.find("effective_pos") != std::string::npos,
              "mixed effective_pos: the message names the field, so a caller is not sent hunting: " +
                  r.why);

        // And the SAME cells with one position are admitted -- so the refusal is about the
        // disagreement, not about having two spans.
        tq::reader::query_view same = mixed;
        same.spans[1].effective_pos = 3;
        const outcome ok_run = run_case(false, same, true);
        check(ok_run.refusal == tq::reader::prefill_refusal::ok,
              std::string("two spans agreeing on effective_pos are admitted, got ") +
                  tq::reader::describe(ok_run.refusal) + " -- " + ok_run.why);
    }

    // (3) A CELL-MAJOR `v`. `nb[1] > nb[2]`, which the per-token emitter calls
    //     `v_layout_cell_major`. This entry point folds it into `bad_shape` -- the divergence the
    //     file header records -- so the code asserted here is the one prefill actually produces.
    {
        const outcome r = run_case(/*v_cell_major=*/true, full_view(), true);
        expect_refusal(r, tq::reader::prefill_refusal::bad_shape, "cell-major v");
        check(r.why.find("cell-major") != std::string::npos,
              "cell-major v: the message names the layout, not just a shape mismatch: " + r.why);
    }

    // (4) THE MASK INTENT. `unstated` must be REFUSED here exactly as the single-query emitter
    //     refuses it, and refused BEFORE the reorder nodes exist. The permissive behaviour this
    //     replaces was worse than a missing check: the call was admitted, the intersection was
    //     computed anyway, and the caller got a narrower answer than its mask described with nothing
    //     in the numbers to say so.
    {
        const outcome r = run_case(false, full_view(), true,
                                   tq::reader::graph_inputs::mask_coverage::unstated);
        expect_refusal(r, tq::reader::prefill_refusal::mask_intent_unstated, "unstated mask intent");
        check(r.why.find("coverage") != std::string::npos,
              "unstated intent: the message names the missing claim, so a caller is not sent "
              "hunting: " + r.why);
        check(r.why.find("intersect") != std::string::npos,
              "unstated intent: and names the value that resolves it: " + r.why);
    }

    // (6) THE OFFSET ARITHMETIC. Pure numbers, so this needs no context -- and the boundary
    //     assertions are what distinguish a size_t multiply from one that narrows to int64_t.
    offset_arithmetic_is_guarded();

    if (failures == 0) {
        std::fprintf(stderr, "prefill gates: OK\n");
        return 0;
    }
    std::fprintf(stderr, "prefill gates: %d failure(s)\n", failures);
    return 1;
}
