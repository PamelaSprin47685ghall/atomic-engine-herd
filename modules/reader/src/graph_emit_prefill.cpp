#include <tq/reader/graph_emit_prefill.h>

#include <cmath>
#include <cstring>
#include <limits>

namespace tq::reader {

const char * describe(prefill_refusal refusal) {
    switch (refusal) {
        case prefill_refusal::ok:            return "ok";
        case prefill_refusal::no_context:    return "ctx, k, v, q or mask is null";
        case prefill_refusal::bad_shape:     return "k/v/q/mask disagree with each other or with the stated head counts";
        case prefill_refusal::bad_head_layout: return "n_head_kv does not divide n_head";
        case prefill_refusal::single_token:  return "T == 1: the single-query emitter is cheaper and identical in value";
        case prefill_refusal::bad_mask:      return "the mask is absent, the wrong type, not contiguous, or not [n_cell, T, 1, 1]";
        case prefill_refusal::bad_view:      return "the view is empty, has an empty span, names a cell outside the cache, or names one twice";
        case prefill_refusal::unsupported_cache_type: return "k or v is a type this composition cannot widen exactly";
        case prefill_refusal::multi_stream:  return "k, v or q carries more than one stream on ne3";
        case prefill_refusal::arena_admission: return "the composition's object bound does not fit the arena; nothing was built";
        case prefill_refusal::overflow:      return "the object bound's own arithmetic would wrap size_t";
        case prefill_refusal::no_alloc_context:
            return "the context allocates; object bounds cannot cover its data regions";
        case prefill_refusal::mask_intent_unstated:
            return "a mask was supplied with `coverage = unstated`: this function computes "
                   "view ∩ mask, so the intent has to be stated";
        // PARTIAL is part of the meaning, not a footnote: this refusal is reachable only after the
        // reorder nodes exist, so the caller owns a context it must throw away. Saying so in the
        // enum's own text is what keeps a reader from treating the code as a mere rejection.
        case prefill_refusal::inner_refusal:
            return "a per-token emission refused although the preflight passed; the graph is PARTIAL "
                   "and the context must be discarded";
        case prefill_refusal::partial_graph: return "ggml grew the arena past the charge: the context must be discarded";
    }
    return "?";
}

std::size_t prefill_object_count(std::int64_t n_tokens, std::size_t spans, bool q_cast,
                                 bool * overflow) {
    if (overflow) *overflow = false;
    if (n_tokens < 2) {
        // A prefill of fewer than two tokens is not a prefill; 0 is a refusal, not a budget.
        return 0;
    }
    const std::size_t t = std::size_t(n_tokens);
    const std::size_t limit = std::numeric_limits<std::size_t>::max();

    // per token: the single-query emitter's own charge
    if (spans > (limit - 13 - (q_cast ? 1u : 0u)) / 6) {
        if (overflow) *overflow = true;
        return 0;
    }
    const std::size_t per_token = 13 + (q_cast ? 1u : 0u) + 6 * spans;
    if (per_token != 0 && t > limit / per_token) {
        if (overflow) *overflow = true;
        return 0;
    }
    std::size_t total = t * per_token;

    // slicing: one permute + one cont for the queries, then a query view and a mask view per token
    const std::size_t slicing = t > (limit - 2) / 2 ? 0 : 2 + 2 * t;
    if (slicing == 0 || total > limit - slicing) {
        if (overflow) *overflow = true;
        return 0;
    }
    total += slicing;

    // joining: one concat chain for `out`, one for `weights`
    if (t - 1 > (limit - total) / 2) {
        if (overflow) *overflow = true;
        return 0;
    }
    total += 2 * (t - 1);
    return total;
}

namespace {

// The per-object cost of this ggml build, measured rather than assumed. Same 16 KiB probe the
// single-query emitter uses; the two must agree, and if they ever disagree the emitter's own
// post-charge check fires first, so the failure is a named refusal rather than an abort inside
// `ggml_new_object`.
bool measure_object_bytes(std::size_t & per_object) {
    ggml_init_params ip{};
    ip.mem_size = 16u * 1024u;
    ip.no_alloc = true;
    ggml_context * probe = ggml_init(ip);
    if (probe == nullptr) return false;
    const std::size_t before = ggml_used_mem(probe);
    ggml_tensor * t = ggml_new_tensor_1d(probe, GGML_TYPE_I32, 4);
    if (t == nullptr) {
        ggml_free(probe);
        return false;
    }
    const std::size_t after = ggml_used_mem(probe);
    ggml_free(probe);
    if (after <= before) return false;
    per_object = after - before;
    return true;
}

prefill_refusal refuse(prefill_refusal r, std::string why, prefill_result * out) {
    out->refusal = r;
    out->why = std::move(why);
    return r;
}

} // namespace

prefill_result emit_prefill_attention(const prefill_inputs & in, float kq_scale) {
    prefill_result result;

    // ---------------------------------------------------------------------------------------------
    // PREFLIGHT. Everything the per-token emitter can refuse is decided here, before a node exists,
    // so that "a structural failure leaves half a graph" cannot happen. The refusals are listed in
    // the same order the single-query emitter checks them, because a caller should get the same
    // verdict from either entry point.
    // ---------------------------------------------------------------------------------------------
    if (in.ctx == nullptr || in.k == nullptr || in.v == nullptr || in.q == nullptr ||
        in.mask == nullptr) {
        refuse(prefill_refusal::no_context, "one of ctx, k, v, q, mask is null", &result);
        return result;
    }

    // THE INTENT, ASKED BEFORE ANYTHING ELSE ABOUT THE MASK -- because it is the one fact about a
    // mask this function cannot check. The mask's values do not exist at build time under `no_alloc`
    // (measured: `data == nullptr`), so "does the view name every cell the mask leaves visible?" has
    // no answer here. What the caller CAN state is which semantics it wants, and an unstated intent
    // means this function would silently pick one: `view ∩ mask`.
    //
    // The single-query emitter already refuses `unstated` with its own code. Accepting it here made
    // the two entry points disagree about whether a caller must know what it is asking for, and the
    // permissive side was the one computing the intersection anyway -- so a caller porting a call
    // from one to the other got a different answer about whether its arguments were acceptable, and
    // the same numbers either way. That is the shape of a trap, not a default.
    if (in.coverage != graph_inputs::mask_coverage::intersect) {
        refuse(prefill_refusal::mask_intent_unstated,
               "a mask is supplied but `coverage` is `unstated`. This function computes visibility "
               "as view ∩ mask: it fills every cache position with -INFINITY and writes the named "
               "cells over it, so a mask can only NARROW that set. If that is what you want, say so "
               "with coverage = intersect. If your mask leaves visible cells the view does not name "
               "and you need them to contribute, the view is incomplete and no flag can repair it -- "
               "widen the view, and use view_covers_finite_mask where the mask's values are readable",
               &result);
        return result;
    }
    if (in.n_head == 0 || in.n_head_kv == 0 || in.head_dim == 0 || in.n_cell == 0) {
        refuse(prefill_refusal::bad_shape, "n_head/n_head_kv/head_dim/n_cell must be positive",
               &result);
        return result;
    }
    // AND THEY MUST FIT IN `int64_t`, WHICH IS THE TYPE EVERY `ne` IS COMPARED AGAINST.
    //
    // The four fields are `size_t`; the shape checks below convert each one with
    // `std::int64_t(in.head_dim)` and compare it to `ne[i]`, which is `int64_t`. When the value
    // exceeds `INT64_MAX` that conversion is IMPLEMENTATION-DEFINED, not a compile error: on this
    // target it wraps to a negative number. The comparison then fails and the call is refused as
    // `bad_shape` -- so the outcome is still a refusal, but by accident, on a path nobody designed
    // for. A refusal that happens because a conversion wrapped is one refactor away from a refusal
    // that does not happen.
    //
    // Checked explicitly, and reported as `overflow` rather than `bad_shape`: a caller that passed
    // 2^63 as `head_dim` did not describe a wrong shape, it described one that does not fit the
    // width the tensors describe their shapes in.
    const std::size_t int64_limit = std::size_t(std::numeric_limits<std::int64_t>::max());
    if (in.n_head > int64_limit || in.n_head_kv > int64_limit || in.head_dim > int64_limit ||
        in.n_cell > int64_limit) {
        refuse(prefill_refusal::overflow,
               "a layout dimension exceeds INT64_MAX, so `std::int64_t(...)` on it is "
               "implementation-defined rather than a comparison: n_head " +
                   std::to_string(in.n_head) + ", n_head_kv " + std::to_string(in.n_head_kv) +
                   ", head_dim " + std::to_string(in.head_dim) + ", n_cell " +
                   std::to_string(in.n_cell),
               &result);
        return result;
    }
    if (in.n_head % in.n_head_kv != 0) {
        refuse(prefill_refusal::bad_head_layout,
               "n_head_kv " + std::to_string(in.n_head_kv) + " does not divide n_head " +
                   std::to_string(in.n_head),
               &result);
        return result;
    }

    const std::int64_t T = in.q->ne[1];
    if (T < 2) {
        refuse(prefill_refusal::single_token,
               "q has " + std::to_string(T) + " token(s); T == 1 belongs to emit_physical_attention",
               &result);
        return result;
    }

    // Shapes, exactly as the single-query emitter checks them per call.
    const bool k_ok = in.k->ne[0] == std::int64_t(in.head_dim) &&
                      in.k->ne[1] == std::int64_t(in.n_head_kv) &&
                      in.k->ne[2] == std::int64_t(in.n_cell) && in.k->ne[3] == 1;
    const bool v_ok = in.v->ne[0] == std::int64_t(in.head_dim) &&
                      in.v->ne[1] == std::int64_t(in.n_head_kv) &&
                      in.v->ne[2] == std::int64_t(in.n_cell) && in.v->ne[3] == 1;
    const bool q_ok = in.q->ne[0] == std::int64_t(in.head_dim) &&
                      in.q->ne[2] == std::int64_t(in.n_head) && in.q->ne[3] == 1;
    if (!k_ok || !v_ok || !q_ok) {
        refuse(prefill_refusal::bad_shape,
               "expected k/v [head_dim, n_head_kv, n_cell] and q [head_dim, T, n_head]; got k [" +
                   std::to_string(in.k->ne[0]) + " " + std::to_string(in.k->ne[1]) + " " +
                   std::to_string(in.k->ne[2]) + " " + std::to_string(in.k->ne[3]) + "], v [" +
                   std::to_string(in.v->ne[0]) + " " + std::to_string(in.v->ne[1]) + " " +
                   std::to_string(in.v->ne[2]) + "], q [" + std::to_string(in.q->ne[0]) + " " +
                   std::to_string(in.q->ne[1]) + " " + std::to_string(in.q->ne[2]) + "]",
               &result);
        return result;
    }
    // NO CHECK THAT `q->ne[1] == T`, BECAUSE `T` IS `q->ne[1]`.
    //
    // The previous revision had
    //
    //     if (in.q->ne[1] != T) refuse(bad_shape, "q changed shape during the preflight");
    //
    // and it could never fire: `T` was assigned from that very field a few lines above, so the
    // comparison is a tautology. It read as a guard, which is worse than no guard -- a reviewer
    // counting the refusals would count one that does not exist.
    //
    // What the shape of `q` actually needs is already established elsewhere, and the two facts are
    // distinct: `q->ne[0] == head_dim` and `q->ne[2] == n_head` are checked in `q_ok` (so the token
    // axis is the only one left, i.e. `ne[1]`), and the mask's own token count is checked against
    // `T` in the mask shape test. A caller cannot make the token axis disagree with `T` because `T`
    // is read from it; what it CAN get wrong is the mask's axis, and that is where the check is.

    // v's LAYOUT, not just its shape -- the third of the same defect class as the two above.
    //
    // The single-query emitter refuses a cell-major value cache with its own code
    // (`v_layout_cell_major`, judged as `nb[1] > nb[2]`), because `ggml_permute(v,0,2,1,3)` needs a
    // dim-major cache to become `[head_dim, n_cell, n_head_kv]`. `v_ok` above checks only `ne`, so a
    // cell-major `v` passes the preflight here and is refused per token -- again after the reorder
    // nodes exist. The judgement is duplicated deliberately: it is one comparison, and the
    // alternative is calling into the emitter to ask, which builds nodes.
    if (in.v->nb[1] > in.v->nb[2]) {
        refuse(prefill_refusal::bad_shape,
               "v has cell-major rows (nb[1] = " + std::to_string(in.v->nb[1]) + " > nb[2] = " +
                   std::to_string(in.v->nb[2]) +
                   "); the value reduction needs a dim-major cache, and that refusal would otherwise "
                   "arrive after this function built its reorder nodes",
               &result);
        return result;
    }

    // The operand types the single-query emitter admits: F32 or F16 caches, and a query it can widen.
    auto type_ok = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16; };
    if (!type_ok(in.k->type) || !type_ok(in.v->type) || !type_ok(in.q->type)) {
        refuse(prefill_refusal::unsupported_cache_type,
               "k/v/q must be F32 or F16; got " + std::to_string(int(in.k->type)) + "/" +
                   std::to_string(int(in.v->type)) + "/" + std::to_string(int(in.q->type)),
               &result);
        return result;
    }

    // The mask: per token, so `ne[1]` must be T. One row for T tokens is the case ggml would
    // broadcast and the one that gives a causal token sight of its own future.
    if ((in.mask->type != GGML_TYPE_F16 && in.mask->type != GGML_TYPE_F32) ||
        !ggml_is_contiguous(in.mask)) {
        refuse(prefill_refusal::bad_mask,
               "the mask must be a contiguous F16 or F32 tensor; its type is " +
                   std::to_string(int(in.mask->type)) +
                   (ggml_is_contiguous(in.mask) ? "" : " and it is not contiguous"),
               &result);
        return result;
    }
    if (in.mask->ne[0] != std::int64_t(in.n_cell) || in.mask->ne[1] != T ||
        in.mask->ne[2] != 1 || in.mask->ne[3] != 1) {
        refuse(prefill_refusal::bad_mask,
               "the mask must be [n_cell, n_tokens, 1, 1]; got [" +
                   std::to_string(in.mask->ne[0]) + " " + std::to_string(in.mask->ne[1]) + " " +
                   std::to_string(in.mask->ne[2]) + " " + std::to_string(in.mask->ne[3]) + "]",
               &result);
        return result;
    }

    // The view: non-empty, no empty span, every cell inside the cache, no duplicates. A duplicate is
    // refused rather than deduplicated because two scatter writes to one destination make the result
    // order-dependent.
    // THE ALLOCATION MODEL, CHECKED HERE BECAUSE THE INNER EMITTER CHECKS IT TOO LATE.
    //
    // `emit_physical_attention` refuses an allocating context (`no_alloc_context`) because it can
    // bound the objects it creates but not the data regions an allocating graph also owns. This
    // function passes `in.ctx` straight through, so an allocating context would be admitted here,
    // the two reorder nodes would be built, and the FIRST per-token emission would then refuse --
    // leaving a half-built graph and an `inner_refusal` that names a symptom rather than the cause.
    // Asking the same question first costs one call and removes that path entirely.
    if (!ggml_get_no_alloc(in.ctx)) {
        refuse(prefill_refusal::no_alloc_context,
               "the context allocates: the emitters here bound object counts only, and an "
               "allocating graph also owns data regions this function cannot size",
               &result);
        return result;
    }

    const std::size_t spans = in.view.spans.size();
    if (spans == 0) {
        refuse(prefill_refusal::bad_view, "the view has no spans", &result);
        return result;
    }

    // EVERY SPAN MUST AGREE ON `effective_pos`, AND THIS IS THE SAME DEFECT CLASS AS THE ONE ABOVE.
    //
    // The single-query emitter refuses a view whose spans disagree (`mixed_effective_pos`): it uses
    // the value to keep the rotation of the query consistent across spans. One view is shared by
    // every token here, so a disagreement would be refused identically T times -- but only AFTER
    // `q_head_major` exists. Refusing now means the caller learns about a malformed view without
    // paying for a graph it must throw away.
    {
        const std::int32_t first_pos = in.view.spans.front().effective_pos;
        for (std::size_t s = 1; s < spans; ++s) {
            if (in.view.spans[s].effective_pos != first_pos) {
                refuse(prefill_refusal::bad_view,
                       "span 0 has effective_pos " + std::to_string(first_pos) + " but span " +
                           std::to_string(s) + " has " +
                           std::to_string(in.view.spans[s].effective_pos) +
                           "; the single-query emitter would refuse this view after the reorder "
                           "nodes were built",
                       &result);
                return result;
            }
        }
    }
    std::vector<char> seen(std::size_t(in.n_cell), 0);
    std::size_t total_cells = 0;
    for (std::size_t s = 0; s < spans; ++s) {
        if (in.view.spans[s].cells.empty()) {
            refuse(prefill_refusal::bad_view, "span " + std::to_string(s) + " has no cells",
                   &result);
            return result;
        }
        for (std::int32_t cell : in.view.spans[s].cells) {
            if (cell < 0 || std::size_t(cell) >= std::size_t(in.n_cell)) {
                refuse(prefill_refusal::bad_view,
                       "span " + std::to_string(s) + " names cell " + std::to_string(cell) +
                           ", outside [0, " + std::to_string(in.n_cell) + ")",
                       &result);
                return result;
            }
            if (seen[std::size_t(cell)] != 0) {
                refuse(prefill_refusal::bad_view,
                       "cell " + std::to_string(cell) + " appears twice in the view",
                       &result);
                return result;
            }
            seen[std::size_t(cell)] = 1;
            ++total_cells;
        }
    }

    // THE SLICE OFFSETS ARE CHECKED BEFORE THE FIRST NODE, for the same reason the object bound is:
    // they are arithmetic that can wrap, and a wrapped offset is SMALLER than the truth -- so ggml's
    // own `view_size + view_offs <= ggml_nbytes` check accepts it and the view reads the wrong bytes.
    // A check after the node exists would be a check on a graph that must be discarded anyway.
    //
    // The maximum offset is at `t == T-1`, because `t` ascends; testing the last one covers all of
    // them. Two strides matter:
    //
    //   the mask's  nb[1]  -- a `[n_cell, T, 1, 1]` mask, whose token stride is ggml's own
    //   the query's        -- computed from `hd * nh * element`, which is what `cont(permute(q))`
    //                         allocates per token; taking it from `in.q->nb[2]` would be wrong,
    //                         because that is the SOURCE stride and the loop slices the COPY
    std::size_t probe = 0;
    if (!prefill_slice_offset(T - 1, in.mask->nb[1], &probe)) {
        refuse(prefill_refusal::overflow,
               "the last token's mask slice offset wraps: (" + std::to_string(T - 1) +
                   ") * nb[1] = (" + std::to_string(T - 1) + ") * " +
                   std::to_string(in.mask->nb[1]) + " exceeds size_t. ggml would then accept a " +
                   "wrapped (small) offset as an in-bounds view and read another token's row",
               &result);
        return result;
    }
    {
        // THE PRODUCT IS KEPT IN `size_t` AND NEVER NARROWED, which is the difference between a
        // checked multiply and one that only looks checked. The previous revision wrote
        //
        //     prefill_slice_offset(std::int64_t(size_t(head_dim) * size_t(n_head)), el_q, &per_head)
        //
        // so the product was computed in `size_t` and then CONVERTED to the helper's `int64_t` token
        // parameter. A product above `INT64_MAX` (but still a legal `size_t`) makes that conversion
        // implementation-defined: on this target it wraps negative and the helper's `t < 0` branch
        // refuses, which is a refusal produced by a wrapped conversion rather than by a rule. Two
        // numbers being multiplied are sizes, not a token count, and `prefill_scale` multiplies them
        // as sizes.
        const std::size_t el_q = ggml_element_size(in.q);
        std::size_t per_head = 0;
        std::size_t per_token_q = 0;
        if (!prefill_scale(std::size_t(in.head_dim), std::size_t(in.n_head), &per_head) ||
            !prefill_scale(per_head, el_q, &per_head) ||
            !prefill_slice_offset(T - 1, per_head, &per_token_q)) {
            refuse(prefill_refusal::overflow,
                   "the last token's query slice offset wraps size_t: (" + std::to_string(T - 1) +
                       ") * head_dim (" + std::to_string(in.head_dim) + ") * n_head (" +
                       std::to_string(in.n_head) + ") * " + std::to_string(el_q) +
                       " bytes exceeds size_t",
                   &result);
            return result;
        }
    }

    // ---------------------------------------------------------------------------------------------
    // THE BOUND, before the first node. Its arithmetic is checked for overflow first: a wrapped count
    // is smaller than the truth, and too small is the failure that ends the process inside
    // `ggml_new_object`.
    // ---------------------------------------------------------------------------------------------
    const bool q_cast = in.q->type != GGML_TYPE_F32;
    bool overflow = false;
    const std::size_t objects = prefill_object_count(T, spans, q_cast, &overflow);
    if (overflow || objects == 0) {
        refuse(prefill_refusal::overflow,
               "the object bound overflows size_t at " + std::to_string(T) + " tokens and " +
                   std::to_string(spans) + " spans",
               &result);
        return result;
    }

    std::size_t per_object = 0;
    if (!measure_object_bytes(per_object) || per_object == 0) {
        refuse(prefill_refusal::arena_admission,
               "this ggml build did not report a per-object cost, so no bound can be established",
               &result);
        return result;
    }
    if (objects > std::numeric_limits<std::size_t>::max() / per_object) {
        refuse(prefill_refusal::overflow, "objects * per_object overflows size_t", &result);
        return result;
    }
    const std::size_t need = objects * per_object;
    const std::size_t used_before = ggml_used_mem(in.ctx);
    const std::size_t size_total = ggml_get_mem_size(in.ctx);
    if (used_before > size_total || size_total - used_before < need) {
        refuse(prefill_refusal::arena_admission,
               "the composition needs " + std::to_string(need) + " bytes for " +
                   std::to_string(objects) + " objects at " + std::to_string(per_object) +
                   " each, but the arena has " +
                   std::to_string(used_before <= size_total ? size_total - used_before : 0) +
                   " free; nothing was built",
               &result);
        return result;
    }

    // ---------------------------------------------------------------------------------------------
    // BUILD. From here on, every failure is ggml disagreeing with the bound -- there is no structural
    // refusal left, because the preflight decided them all.
    // ---------------------------------------------------------------------------------------------
    const std::int64_t hd = std::int64_t(in.head_dim);
    const std::int64_t nh = std::int64_t(in.n_head);
    const std::int64_t kvh = std::int64_t(in.n_head_kv);
    const std::int64_t n_cell = std::int64_t(in.n_cell);
    // No `sizeof(float)` here any more: the query slice takes its strides from `q_head_major`, so
    // an F16 query is sliced in F16 width. Keeping a byte width constant was what made the F16 path
    // abort inside `ggml_view_3d`.

    // One permute + one cont for the queries: [hd, T, nh] -> [hd, nh, T], after which token t's block
    // is CONTIGUOUS at t*hd*nh. Doing it once is cheaper than per token, and it is the recipe that
    // was measured (`q_stride_contract`: padded and packed agree bit for bit), so the per-token view
    // below is contiguous by construction rather than by a stride argument.
    ggml_tensor * q_head_major = ggml_cont(in.ctx, ggml_permute(in.ctx, in.q, 0, 2, 1, 3));
    if (q_head_major == nullptr) {
        refuse(prefill_refusal::inner_refusal, "ggml could not reorder the queries by head", &result);
        result.refusal = prefill_refusal::inner_refusal;
        return result;
    }

    result.per_token.reserve(std::size_t(T));
    for (std::int64_t t = 0; t < T; ++t) {
        // The query slice: [hd, 1, nh], packed strides, token block at t*hd*nh*element.
        // THE STRIDES COME FROM THE TENSOR, NOT FROM `sizeof(float)`.
        //
        // The previous version computed them as `hd * sizeof(float)`, which is the F32 layout. The
        // preflight admits an F16 query (`type_ok` accepts it, and `q_cast` is charged for the cast
        // the single-query emitter performs), and for F16 that arithmetic puts the last token's
        // offset at `(T-1)*hd*nh*4` inside a tensor of `hd*nh*T*2` bytes -- so `ggml_view_3d` failed
        // its own `view_size + view_offs <= ggml_nbytes` assertion and ABORTED the process. Measured
        // on the F16 grid case, and it is the exact failure mode this lane refuses everywhere:
        // a configuration the preflight allowed, dying inside ggml with no refusal to read.
        //
        // `q_head_major` is contiguous by construction (`cont` of a permute), so `nb[1]` is one head
        // row and `nb[2]` one token block, in the tensor's OWN element width. Using them keeps the
        // view packed for F16 and F32 alike, and removes the assumption that made the two disagree.
        std::size_t q_off = 0;
        std::size_t mask_off = 0;
        // BOTH OFFSETS GO THROUGH THE CHECKED HELPER, and the refusal names the token. The preflight
        // proved the LAST token's offset cannot wrap, and `t` ascends, so neither of these can fail --
        // which is exactly why they must still be able to: if that reasoning ever stops holding (a
        // stride read from a different tensor, a token axis that is not monotone), the failure is a
        // named refusal here instead of a view onto another token's bytes.
        if (!prefill_slice_offset(t, q_head_major->nb[2], &q_off) ||
            !prefill_slice_offset(t, in.mask->nb[1], &mask_off)) {
            refuse(prefill_refusal::overflow,
                   "token " + std::to_string(t) + "'s slice offset wraps size_t (q stride " +
                       std::to_string(q_head_major->nb[2]) + ", mask stride " +
                       std::to_string(in.mask->nb[1]) + ")",
                   &result);
            return result;
        }
        ggml_tensor * q_t = ggml_view_3d(in.ctx, q_head_major, hd, 1, nh,
                                         q_head_major->nb[1], q_head_major->nb[1], q_off);
        // The mask slice: [n_cell, 1, 1, 1], contiguous at t*nb[1] of the [n_cell, T, 1, 1] mask.
        ggml_tensor * mask_t = ggml_view_4d(in.ctx, in.mask, n_cell, 1, 1, 1,
                                            in.mask->nb[1], in.mask->nb[2], in.mask->nb[3],
                                            mask_off);
        if (q_t == nullptr || mask_t == nullptr) {
            refuse(prefill_refusal::inner_refusal,
                   "ggml could not slice token " + std::to_string(t) + "'s query or mask", &result);
            return result;
        }

        graph_inputs one;
        one.ctx = in.ctx;
        one.k = in.k;
        one.v = in.v;
        one.q = q_t;
        one.mask = mask_t;
        // `intersect`: the view already names every cell the mask can leave visible (a precondition
        // the caller owns), so view ∩ mask is the mask's visibility -- exactly what a prefill means.
        one.coverage = graph_inputs::mask_coverage::intersect;
        one.n_head = in.n_head;
        one.n_head_kv = in.n_head_kv;
        one.head_dim = in.head_dim;
        one.n_cell = in.n_cell;
        one.collect_indices = true;

        physical_emit_result em = emit_physical_attention(one, in.view, kq_scale);
        if (em.refusal != physical_emit_refusal::ok) {
            refuse(prefill_refusal::inner_refusal,
                   "token " + std::to_string(t) + ": " + describe(em.refusal) + " -- " + em.why,
                   &result);
            return result;
        }
        if (em.out == nullptr || em.weights == nullptr ||
            em.indices.size() != spans + 1) {
            refuse(prefill_refusal::inner_refusal,
                   "token " + std::to_string(t) + " returned " +
                       std::to_string(em.indices.size()) + " entries for " +
                       std::to_string(spans) + " span(s), or a null output",
                   &result);
            return result;
        }
        result.per_token.push_back(std::move(em));
    }

    // Join along dim 1, twice: one chain for `out`, one for `weights`. NOT along dim 2 -- that would
    // produce a tensor whose shape looks right and whose values mix tokens and heads. Two chains
    // because the weights are the strongest comparison and joining only `out` would leave them
    // unexposed.
    ggml_tensor * out = result.per_token[0].out;
    ggml_tensor * weights = result.per_token[0].weights;
    for (std::size_t t = 1; t < result.per_token.size(); ++t) {
        out = ggml_concat(in.ctx, out, result.per_token[t].out, 1);
        weights = ggml_concat(in.ctx, weights, result.per_token[t].weights, 1);
        if (out == nullptr || weights == nullptr) {
            refuse(prefill_refusal::inner_refusal,
                   "ggml could not join token " + std::to_string(t) + "'s output", &result);
            return result;
        }
    }

    // The bound is charged, and then CHECKED against what the arena actually grew. A count that is
    // too high only over-reserves; one that is too low lets `ggml_new_object` abort mid-build, so the
    // growth is compared and a graph that outgrew the charge is reported rather than returned.
    const std::size_t grew = ggml_used_mem(in.ctx) - used_before;
    if (grew > need) {
        result.refusal = prefill_refusal::partial_graph;
        result.why = "the graph grew " + std::to_string(grew) + " bytes against a charge of " +
                     std::to_string(need) + " for " + std::to_string(objects) +
                     " objects; ggml disagrees with the bound, so this context must be discarded " +
                     "rather than used";
        result.objects_charged = objects;
        result.object_cost = per_object;
        result.bytes_required = need;
        return result;
    }

    result.out = out;
    result.weights = weights;
    result.objects_charged = objects;
    result.object_cost = per_object;
    result.bytes_required = need;
    result.refusal = prefill_refusal::ok;
    (void) total_cells;
    (void) kvh;
    (void) nh;
    return result;
}

} // namespace tq::reader
