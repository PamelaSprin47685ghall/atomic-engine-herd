// tq/reader/graph_emit_prefill.h — prefill attention as a COMPOSITION of the verified single-query
// emission, one per token, joined along the token axis.
//
// WHY COMPOSITION AND NOT A BESPOKE MULTI-TOKEN GRAPH
// ---------------------------------------------------
// Two bespoke attempts were written and both were removed. The first charged `spans` gathers while
// building `T * spans` of them -- an UNDER-estimate, the one direction an admission bound must never
// err in, and `ggml_new_object` aborts half-way leaving a partially built context. The second died on
// a 0.49 difference nobody could attribute. Meanwhile the composition was measured: `T` calls to
// `emit_physical_attention`, joined along dim 1, matched a per-(token, head) double reference to
// 2.98e-08, with a real slice for one token and a per-token causal mask. So the numbers do not need
// re-deriving here; only the composition and its budget do.
//
// WHAT THIS FILE OWNS, AND WHAT IT DELIBERATELY DOES NOT
// ------------------------------------------------------
//   owns      the preflight (every refusal the single-query emitter can raise is decided before the
//             first node exists), the whole-graph object bound, the per-token slicing, the join, and
//             the per-token index hand-back.
//   not       the arithmetic (that is `graph_emit_physical.cpp`, pinned by its own tests), the index
//             write (that is `publish_index_inputs`, after placement), or any llama type. This
//             header has no `llama-*` include: it is the ggml half, testable without a model.
//
// THE PREFlight IS THE ATOMICITY STORY, AND IT IS STATED HONESTLY
// --------------------------------------------------------------
// "构图失败不得部分生成" cannot be achieved by deleting nodes -- a ggml context has no undo. What can
// be achieved is that NOTHING STRUCTURAL can fail after the first node. The structural set is decided
// up here, and it mirrors what the single-query emitter would otherwise refuse PER TOKEN -- after
// `q_head_major` already exists:
//
//   shapes            k/v [head_dim, n_head_kv, n_cell], q [head_dim, T, n_head], ne3 == 1
//   v layout          dim-major (`nb[1] <= nb[2]`); a cell-major cache is refused here rather than
//                     arriving as `v_layout_cell_major` on token 0
//   head layout       n_head_kv divides n_head
//   mask              contiguous F16/F32, exactly [n_cell, T, 1, 1]
//   view              non-empty, no empty span, every cell in range, no duplicate, and every span
//                     agreeing on `effective_pos` (a disagreement is `mixed_effective_pos` in the
//                     inner emitter -- again a post-node refusal)
//   allocation model  the context is no_alloc; an allocating one is refused because the bound counts
//                     objects and cannot size data regions
//   types, streams    k/v/q in {F32,F16}, one stream on ne3
//   budget            the whole-graph object bound fits the free bytes, computed before any node
//
// With those decided, the remaining failure modes are `inner_refusal`, `partial_graph` and
// `overflow`, and each is named for what it is rather than left as "ggml disagreed":
//
//   inner_refusal      a per-token emission or a join returned null, i.e. ggml could not create a
//                      node. Two sources: the arena shrank between the admission check and the build
//                      (another tenant of the same context), or a view/permute/cont allocation
//                      failed. Only reachable when the bound's own assumption -- that the free bytes
//                      measured at entry are still free when spent -- does not hold.
//   partial_graph      ggml grew the arena past the charge. The bound is `objects * per_object` with
//                      `per_object` measured from a one-object probe, so it carries alignment slack;
//                      growth above it means ggml's per-object cost changed under this build.
//   overflow           the bound's arithmetic wraps, or a per-token slice offset would. The offset
//                      case is reachable only at token counts far beyond any real cache, because the
//                      slice offset is a bare `t * nb[2]` multiply (see the note at the loop).
//
// WHAT THIS PREFLIGHT DOES **NOT** ESTABLISH, so that the sentence above is not read as more than it
// says. Each of these was checked against the inner emitter's own refusals and found either covered
// by a different route or genuinely outside:
//
//   `q->nb[0]`         NOT checked. The inner emitter refuses a non-unit `nb[0]` because `mul_mat`
//                      reads its right operand along dim 0 -- but this function makes its own copy
//                      with `cont(permute(q))`, and a copy's `nb[0]` is always the element size. So a
//                      source tensor with a wider `nb[0]` would be copied according to that stride
//                      and hand the inner emitter a LEGAL tensor: a wrong answer, not a refusal. No
//                      legal ggml construction reaches that state (`ggml_view_*` inherits `nb[0]`),
//                      which is why there is no test -- and why this is written down rather than
//                      assumed away.
//   `q->ne[1] == T`    vacuous by definition (`T` is read from it). The old code tested it anyway;
//                      that check is gone and its absence is explained at the site.
//   slice offsets      `std::size_t(t) * nb` with no overflow guard. Reachable only if `T * nb`
//                      exceeds `size_t`, and `T` is bounded by the tensor that holds it.
//
// Everything else the inner emitter can refuse -- null context, zero layout dimension, k/v/q shape,
// head divisibility, cache type, `ne3` streams, mask type/contiguity/axes, empty view, empty span,
// cell range, duplicates, `mixed_effective_pos`, `v_layout_cell_major`, `index_data_unavailable`
// (this function always sets `collect_indices`) and `no_alloc_context` -- is decided above.
//
// Nothing here is wired into a product path: the replacement switches still refuse at startup, and
// this function has no caller outside its tests.
//
// THE BOUND
// ---------
//   per token   13 + q_cast + 6*spans      the single-query emitter's own charge, reused verbatim
//   slicing     2 + 2*T                    one permute+cont pair for the queries (so each token's
//                                          block is contiguous), then one view for the query and one
//                                          for the mask, per token
//   joining     2*(T-1)                    one concat chain for `out`, one for `weights`
//   total       ~~~~~~~~~~~~~~~~~~~~~~~~~~ everything above, summed with overflow checks
//
// Every term is checked for overflow before it is added: a wrapped count is smaller than the truth,
// and a bound that is too small is the failure this whole file is about.
//
// ADMISSION IS MEASURED, NOT ASSUMED: the per-object byte cost comes from a 16 KiB `no_alloc` probe
// (one tensor, read as `ggml_used_mem` growth), the free bytes come from the caller's context, and the
// comparison happens before the first node. A caller with less room gets `arena_admission` and a
// context with nothing in it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <ggml.h>

#include <tq/reader/attention.h>
#include <tq/reader/graph_emit_physical.h>

namespace tq::reader {

enum class prefill_refusal {
    ok,
    no_context,          // ctx, k, v, q or mask is null
    bad_shape,           // k/v/q/mask disagree with each other or with the stated head counts
    bad_head_layout,     // n_head_kv does not divide n_head
    single_token,        // T == 1: the single-query emitter is cheaper and identical in value
    bad_mask,            // mask absent, wrong type, not contiguous, or not [n_cell, T, 1, 1]
    bad_view,            // no spans, a span with no cells, a cell outside the cache, a duplicate,
                         // or spans that disagree on effective_pos
    no_alloc_context,    // the context allocates: object bounds cannot cover its data regions
    mask_intent_unstated, // a mask was supplied without stating that view ∩ mask is intended

    // NOTE ON TWO FOLDED CODES, because a caller that switches entry points will notice.
    //
    //   cell-major `v`  is `bad_shape` here and `v_layout_cell_major` in `emit_physical_attention`.
    //   mixed effective_pos  is `bad_view` here and `mixed_effective_pos` there.
    //
    // The conditions are identical; only the vocabulary differs, because this enum is a PREFLIGHT
    // verdict ("these arguments cannot form a graph") while the single-query enum is a per-emission
    // one ("this call cannot be honoured"). Folding them keeps this enum from growing two codes whose
    // distinction a caller cannot act on -- but it does mean the same mistake reports two different
    // strings depending on which function saw it. Recorded rather than silently accepted: a caller
    // that retries on one code and not the other has been misled by the shape of this list.
    unsupported_cache_type, // k or v is not F32/F16 -- the layouts here are measured on floats
    multi_stream,        // k, v or q carries more than one stream on ne3
    arena_admission,     // the bound does not fit the arena; NOTHING was built
    overflow,            // the bound's own arithmetic would wrap
    inner_refusal,       // a per-token emission refused although the preflight passed
    partial_graph,       // ggml grew the arena past the charge: the context must be discarded
};

const char * describe(prefill_refusal refusal);

// The whole-graph object count, for a caller that wants the number before it asks for an emission.
// `q_cast` is 1 when the queries are not F32 (the single-query emitter casts its right operand).
// Returns 0 with `overflow` set rather than a small number: 0 is a refusal, not a budget.
std::size_t prefill_object_count(std::int64_t n_tokens, std::size_t spans, bool q_cast,
                                 bool * overflow);

// `t * stride`, refused instead of wrapped. Returns false and leaves `*out` untouched on overflow.
//
// WHY THIS EXISTS AS A FUNCTION. The per-token views below are built with
// `ggml_view_*(..., std::size_t(t) * nb)` -- a BARE multiply whose product feeds an offset that
// ggml then checks against the parent's byte extent. If the product wraps, the check sees a SMALL
// offset: it passes, the view points at the wrong bytes, and nothing anywhere reports it. That is the
// one failure shape the rest of this file is organised to avoid, so the arithmetic gets the same
// treatment as the object bound: checked first, named when it fails.
//
// The honest scope of the guard, stated because the alternative is a false sense of coverage: with a
// tensor ggml itself built, `t` is below `ne[1]` and `stride` is derived from the same `ne`, so the
// product is bounded by the tensor's own `nbytes` -- and a placed tensor's `nbytes` fits in memory.
// The wrap is therefore not reachable THROUGH a legal tensor. It is reachable through a caller that
// hands over a hand-built one (a `ggml_view_4d` with fabricated strides, a struct filled in by
// something other than ggml), and this file does not require its callers to be well-behaved. Guarding
// costs one comparison per token; not guarding costs a wrong answer that looks like a number.
// `out` MUST NOT BE NULL, and the guard is a refusal rather than a crash. The first version wrote
// `*out` unconditionally; every caller in this file passes a local, so the bug was latent -- but a
// helper that dereferences a caller's pointer without asking is the kind of latent defect that
// surfaces the day someone passes `nullptr` "just to test the overflow path". Refusing keeps the
// function total on its stated inputs.
inline bool prefill_slice_offset(std::int64_t t, std::size_t stride, std::size_t * out) {
    if (out == nullptr || t < 0) return false;
    const std::size_t n = static_cast<std::size_t>(t);
    if (stride != 0 && n > std::numeric_limits<std::size_t>::max() / stride) return false;
    *out = n * stride;
    return true;
}

// `a * b` over `size_t`, refused instead of wrapped, with the same null contract.
//
// WHY THIS EARNS ITS OWN FUNCTION. The query slice's per-token stride is `head_dim * n_head *
// element`, and the previous revision computed the first factor as
// `std::int64_t(std::size_t(head_dim) * std::size_t(n_head))` to feed `prefill_slice_offset`, which
// takes an `int64_t` token count. That conversion is IMPLEMENTATION-DEFINED as soon as the product
// exceeds `INT64_MAX`, even though it is a perfectly representable `size_t`: on this target it wraps
// negative and the helper's `t < 0` branch happens to refuse. "Happens to refuse because a narrowing
// conversion wrapped" is the same accident the `int64_limit` check exists to remove elsewhere in this
// file -- so the product is kept in `size_t` and never narrowed. Multiplying two sizes has nothing to
// do with the token count, and conflating them was the defect.
inline bool prefill_scale(std::size_t a, std::size_t b, std::size_t * out) {
    if (out == nullptr) return false;
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) return false;
    *out = a * b;
    return true;
}

struct prefill_inputs {
    ggml_context * ctx = nullptr;

    // Cache views, dim-major, as upstream passes them: [head_dim, n_head_kv, n_cell]. Shared by every
    // token -- the tokens differ in their queries and their masks, not in the cache they read.
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;

    // Queries for ALL tokens, rotation already applied: [head_dim, n_tokens, n_head], laid out
    // `d + head_dim*(t + n_tokens*head)`. That is the order upstream's own q has (measured
    // `ne [8 4 1]` for one token), so an adapter passing upstream's tensor needs no reorder.
    ggml_tensor * q = nullptr;

    // Per (cache position, token) bias: [n_cell, n_tokens, 1, 1], F16 or F32, contiguous. A finite
    // entry is a real bias; `-inf` means invisible. There is deliberately no one-row form: a
    // broadcast row would give a causal token sight of its own future, which is the whole reason a
    // prefill mask has a token axis.
    ggml_tensor * mask = nullptr;

    // THE CELLS EVERY TOKEN ATTENDS TO, AND THE INTENT THAT MAKES A MASK MEANINGFUL.
    //
    // The emitter fills every cache position with `-inf` and writes the named cells over it, so a
    // mask can only NARROW that set: `-inf + finite` is still `-inf`. Visibility is therefore
    // `view ∩ mask` -- never "the mask alone", and never "whatever upstream's mask meant". A caller
    // that supplied a mask expecting the view to be irrelevant has a narrower answer than it asked
    // for, and nothing about the numbers would say so.
    //
    // `coverage` is where the caller says which of those it wants, and it is REQUIRED: `unstated`
    // is refused before the first node. The field exists because the single-query emitter already
    // requires it (`graph_inputs::mask_coverage`) and this entry point used to accept the same call
    // silently -- the same arguments were refused by one function and admitted by the other, with
    // the admitted one computing the intersection either way. Two entry points disagreeing about
    // whether a caller has to know what it is asking for is not a convenience, it is a trap.
    //
    // `view_covers_finite_mask` (in `graph_emit_physical.h`) is the host-side check for the one
    // moment the mask's VALUES are readable; nothing here can read them, because under `no_alloc`
    // they do not exist until the graph is placed.
    query_view view;
    graph_inputs::mask_coverage coverage = graph_inputs::mask_coverage::unstated;

    std::size_t n_head = 0;
    std::size_t n_head_kv = 0;
    std::size_t head_dim = 0;
    std::size_t n_cell = 0;
};

struct prefill_result {
    prefill_refusal refusal = prefill_refusal::ok;
    std::string why;

    // [head_dim, n_tokens, n_head], `d + head_dim*(t + n_tokens*head)`.
    //
    // NOT upstream's order for T > 1. `build_attn_mha` returns `[head_dim*n_head, n_tokens]`, whose
    // flat index is `d + head_dim*head + head_dim*n_head*t`; an adapter must permute to
    // [head_dim, n_head, n_tokens] and make it contiguous FIRST. For T == 1 the two orders coincide,
    // which is why the single-token adapter could get away with a reshape -- that coincidence ends
    // here, and this comment is the place a caller finds out.
    ggml_tensor * out = nullptr;

    // [n_cell, n_tokens, n_head], post-softmax, same join. Exposed because hidden cells must be
    // exactly zero, which is the strongest single comparison a caller can make.
    ggml_tensor * weights = nullptr;

    // One entry per token, in token order. Each is a COMPLETE single-query emission with its own
    // `indices` and `index_cells`, and each must be published SEPARATELY: `publish_index_inputs`
    // requires entries to tile ONE buffer, so T sets of per-token entries -- each starting at offset
    // 0 -- cannot be merged without a new publication design. Publishing them one by one is that
    // design, and it needs no change to either function.
    //
    // The tensors these entries name are owned by `prefill_inputs::ctx`; the result owns only the
    // values.
    std::vector<physical_emit_result> per_token;

    std::size_t objects_charged = 0;
    std::size_t object_cost = 0;
    std::size_t bytes_required = 0;
};

// Emits prefill attention for `T >= 2` tokens.
//
// Every refusal is decided before the first node except `inner_refusal` and `partial_graph`, which by
// construction mean ggml disagreed with the bound; both are loud and both tell the caller to discard
// the context. Nothing here is wired into a product path: the replacement switches still refuse at
// startup, and this function has no caller outside its test.
prefill_result emit_prefill_attention(const prefill_inputs & in, float kq_scale);

} // namespace tq::reader
