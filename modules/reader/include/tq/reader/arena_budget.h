// tq/reader/arena_budget.h — what an emission will cost the graph's arena, before it runs.
//
// Why this is not "two tensor headers"
// ------------------------------------
// The cache probe learned that a view which does not fit ends the process: ggml_new_object's
// out-of-space branch returns NULL under NDEBUG but ggml_new_tensor_impl follows it with
// GGML_ASSERT(obj_new) (ggml.c:1832), so every build aborts. The reader emission is a far larger
// allocation than the two cache views that lesson came from -- it creates a node per head per
// operand per span, and in an allocating context it also writes real data for the gathered rows,
// the concatenations, the scores and the widening casts.
//
// A budget of "2 * ggml_tensor_overhead()" would therefore be a budget for the wrong thing: it
// covers the two views the probe asks for and nothing the emitter creates. That mistake is worth
// naming because it is tempting -- the numbers are right there, and 368 * 2 looks like a
// measurement. It is a measurement of a different operation.
//
// The two regimes, and why they need different arithmetic
// -------------------------------------------------------
//   no_alloc = true   (what llama.cpp's graph uses)   every tensor object costs its header and
//                                                     NO data: ggml.c:1825 allocates data only
//                                                     when `!ctx->no_alloc`.
//   no_alloc = false  (what this module's tests use)  objects plus data, where a view still
//                                                     allocates nothing (view_src != NULL) but
//                                                     every cont/cast/concat/soft_max result does.
//
// The estimator takes those as INSTRUCTIONS rather than inferring them, because the difference
// between the regimes is a factor of ~40 on a 2048-cell cache: getting it wrong in the safe
// direction wastes memory, and getting it wrong in the other direction ends the process.
//
// What it does not do: predict the exact byte count, or gate an emission.
//
// The number is ADVISORY, and the qualification is not modesty. A measured sweep of 30 shapes put
// it 1.17x to 2.50x above `ggml_used_mem` growth, and the object count 1.24x to 2.50x above the
// measured maximum -- which means the margin, not the arithmetic, is what stands between this
// estimate and a counterexample. A number whose safety rests on margin cannot decide whether an
// operation that ABORTS ON FAILURE may run: see kEstimateAdmitsEmission below.
//
// Its honest uses are two: sizing an arena generously before an emission, and detecting after one
// (see estimate_drift) that the emitter stopped matching the inventory enumerated here.
#pragma once

#include <cstddef>
#include <cstdint>

namespace tq::reader {

// The shapes and types an emission will run on, plus the two cost facts only ggml can supply.
//
// `tensor_overhead` must be ggml_tensor_overhead(); `elem_*` must be ggml_type_size() of the
// respective cache element type. They are parameters instead of calls because this module has no
// ggml dependency by design (its core is arithmetic and its tests are what link ggml), and because
// a caller that fills them from ggml cannot disagree with ggml about them.
struct budget_inputs {
    // Shapes. The emitter's declared layout, not the tensors': a caller that passes the cache's
    // raw ne values has already made the layout mistake, so the budget asks in the emitter's terms.
    std::size_t head_dim = 0;
    std::size_t n_head_kv = 0;
    std::size_t n_cell = 0;
    std::size_t n_head = 0;

    // Per-span key counts, in emission order: `cells_per_span[0]` is the first span's count.
    //
    // A pointer and a count, NOT a fixed array with a limit. The previous version held
    // `cells_per_span[8]` and computed with `min(n_spans, 8)`, so a caller with nine spans was
    // silently costed as if it had eight -- a bound BELOW the real cost, which is the one
    // direction a bound must never err in. The emitter itself has no span limit (`for (const
    // span_keys & span : view.spans)`), so the two disagreed about the same input.
    //
    // It also cannot be a `std::vector`: this header is included by code that must not allocate,
    // and a budget that could throw bad_alloc while deciding whether there is room would be its
    // own counterexample. The caller owns the storage and states the count; the estimator reads
    // exactly that many entries and refuses if the count is zero.
    const std::size_t * cells_per_span = nullptr;
    std::size_t n_spans = 0;

    // The rotation capability, and why the budget must be told about it.
    //
    // `span_rope_fn` is injected: `rope(ctx, q, pos, user)` is arbitrary caller code, and it can
    // create any number of nodes -- a yarn correction, a whole rotary graph, nothing at all. No
    // arithmetic over shapes can bound that, so an estimator that ignored it would be claiming an
    // upper bound on a computation it cannot see.
    //
    // A rope therefore makes the estimate impossible, and that is the whole of this field's
    // meaning: `has_rope == true` produces `budget_status::unbounded_rope` and a saturated total,
    // which reads as "do not emit".
    //
    // An earlier version offered a `rope_nodes_per_span` escape: the caller declared a node count
    // and the estimator charged that many headers plus one query-sized copy each. That was not a
    // bound, and the flaw is worth stating because the field looked reasonable. The callback's
    // signature is
    //
    //     ggml_tensor * (*)(ggml_context * ctx, ggml_tensor * q, int32_t pos, void * user)
    //
    // -- it is handed the CONTEXT, so a single call may do
    // `ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100000, 100000)`, which is a legal rotation
    // implementation and allocates forty gigabytes. A node COUNT cannot bound BYTES when each
    // node's shape is chosen by code the estimator cannot see. The field is gone rather than
    // defaulted to zero, because a field that can only ever mean "I promise" is an invitation to
    // believe the promise.
    bool has_rope = false;

    // Element sizes, in bytes, of the cache as the emitter will read it: ggml_type_size() gives
    // 2 for F16 and 4 for F32. The estimator uses the WIDER of k/v/q per operand, because a
    // widened operand allocates four bytes per element regardless of what the cache held.
    std::size_t k_elem = 4;
    std::size_t v_elem = 4;
    std::size_t q_elem = 4;

    // True when the emission will broaden the operand to F32 (the emitter does this for any
    // non-F32 cache or query). Affects data cost only, never node count.
    bool widen_k = false;
    bool widen_v = false;
    bool widen_q = false;

    // The context the emission will run in.
    bool no_alloc = false;

    // Cost facts from ggml.
    std::size_t tensor_overhead = 0;
};

// The estimate, split so a caller can see which half it is spending.
//
// `objects` is what the emission adds to the arena's object list; `data_bytes` is what it adds to
// its data region. Both matter under no_alloc == false, only the first under no_alloc == true.
// Why an estimate is not a proven bound. Only `ok` means the numbers below may be used as one.
//
// The three failures are different facts and a caller acts on them differently:
//
//   unsupported       the inputs do not describe an emission (a zero shape, no spans, no cost
//                     fact). Nothing was computed; "do not emit" is still the answer, but it is
//                     not a statement about cost.
//   overflow          a product or sum would exceed size_t. The true cost is larger than any
//                     number this type can hold, so the total saturates to SIZE_MAX and every
//                     arena refuses it -- which is the correct outcome, reached deliberately
//                     rather than by wrapping into a small number that looks affordable.
//   unbounded_rope    the injected rotation's node count was not declared, so no upper bound
//                     exists. Saturated for the same reason as overflow.
enum class budget_status { ok, unsupported, overflow, unbounded_rope };

struct arena_budget {
    budget_status status = budget_status::unsupported;
    std::size_t objects = 0;
    std::size_t data_bytes = 0;
    std::size_t total_bytes = 0;
};

// Estimates the emission's arena cost.
//
// What `budget_status::ok` CLAIMS, stated exactly, because the following three sentences are not
// the same claim and only the first is provable from this file:
//
//   1. DERIVED. The total is an upper bound over the node inventory this function enumerates
//      (kTailNodes, kSetupNodesPerHead, kSpanNodesFixed, kSpanNodesPerHead) applied to the shapes
//      given, with every multiplication and addition checked for size_t overflow and every
//      refusal named. That much follows from the arithmetic.
//
//   2. NOT DERIVED. That the inventory matches what graph_emit.cpp actually creates. The emitter
//      is a different file that can gain a node without this one noticing, and no arithmetic here
//      can detect that. Constants kSetupNodesPerHead = 11 and kSpanNodesPerHead = 8 are therefore
//      generous rather than proven: an audit that counted the calls found fewer, and a measured
//      sweep of 30 shapes found the estimate 1.17x to 2.50x above `ggml_used_mem` growth, with the
//      object count 1.3x to 2.5x above the measured maximum. The margin is the whole of the safety.
//
//   3. THEREFORE: `ok` means "this is a bound if the emitter still creates the nodes this file
//      enumerates". The maintenance is a POST-HOC check -- see estimate_drift() -- and it is not an
//      admission gate: `kEstimateAdmitsEmission` is false, because an over-full arena aborts inside
//      ggml before any check can run. A caller that must not crash allocates an arena large enough
//      that exhaustion is impossible, using this number as an advisory size, and treats exhaustion
//      as a defect.
//
// Saying "the result is an upper bound" without that qualifier would be the defect this header
// spends its length refusing: a claim that reads as a guarantee and is in fact a maintained
// agreement between two files.
//
// A budget whose status is not ok carries no usable number: unsupported computes nothing,
// overflow and unbounded_rope saturate to SIZE_MAX. A caller must read those as "do not emit".
arena_budget estimate(const budget_inputs & in);

// WHETHER THIS ESTIMATE MAY ADMIT AN EMISSION. It may not, and that is a compile-time constant
// rather than a comment, because the previous version of this file put the same fact in prose and
// a reader still treated the post-hoc check as protection.
//
// The reason is an ordering fact, not a matter of confidence:
//
//   emission runs -> a tensor does not fit -> ggml_new_object returns NULL under NDEBUG but
//   ggml_new_tensor_impl follows it with GGML_ASSERT(obj_new) (ggml.c:1832) -> GGML_ABORT ->
//   ggml_abort -> the process is gone.
//
// GGML_ASSERT is compiled into every build, so there is no configuration in which the allocation
// failure is returned to a caller. A check that runs AFTER the emission therefore observes nothing
// in the only case that matters: by the time it could return a value, the value's owner no longer
// exists. Calling such a check a safety mechanism inverts the causality -- it would be a smoke
// detector that reports after the house has burnt down.
//
// Two further reasons, each sufficient on its own:
//
//   * the inventory is not proven (see point 2 on estimate()). A number computed from constants
//     chosen by inspection cannot gate an operation whose failure mode is process death.
//   * nothing here reads the arena's actual free bytes. Even a proven inventory would only be a
//     lower bound on what the emission needs; whether the arena HAS that much is a different
//     question, and it is the one that decides.
//
// So a caller that must not crash has exactly one sound option: allocate an arena large enough
// that the emission cannot exhaust it, and treat any exhaustion as a defect. This module supplies
// a size to allocate, marked advisory, and no gate.
constexpr bool kEstimateAdmitsEmission = false;

// What the emission actually cost, compared with the estimate. A POST-HOC regression check.
//
//     const std::size_t before = ggml_used_mem(ctx);
//     const auto out = emit_reader_attention(...);       // may abort; see above
//     const std::size_t after  = ggml_used_mem(ctx);
//     if (estimate_drift(budget, before, after) != drift::within) { /* the inventory drifted */ }
//
// It answers one question -- "did the emitter stop matching the inventory this estimate
// enumerates?" -- and it answers it only for emissions that survived. A drifted estimate that
// allocates too little does not reach this function; it kills the process at ggml.c:1832, which is
// a louder signal but not one this function can produce.
//
// `unusable` covers the two ways the comparison itself means nothing: a budget that is not `ok`
// (nothing to compare against, and a saturated SIZE_MAX must never read as a large allowance),
// and `used_after < used_before` (a context's used memory does not shrink, so that combination
// means the two measurements came from different contexts, and the subtraction would wrap to a
// huge number and pass every test -- the failure shape this module keeps removing).
enum class drift {
    within,    // the emission fit, and the estimate was large enough
    exceeded,  // the emission cost MORE than the estimate: the inventory has drifted
    unusable,  // the comparison cannot be made; see above
};

drift estimate_drift(const arena_budget & budget, std::size_t used_before, std::size_t used_after);

const char * describe(drift value);

} // namespace tq::reader
