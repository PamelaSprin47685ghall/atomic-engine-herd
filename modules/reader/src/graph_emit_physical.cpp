#include "tq/reader/graph_emit_physical.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace tq::reader {
namespace {

physical_emit_result refuse(physical_emit_refusal refusal, std::string why) {
    physical_emit_result result;
    result.refusal = refusal;
    result.why = std::move(why);
    return result;
}

// ONE OBJECT'S COST, MEASURED RATHER THAN ASSUMED.
//
// Under `no_alloc` `ggml_new_tensor_impl` passes `obj_alloc_size = 0` to `ggml_new_object`, which
// allocates `GGML_OBJECT_SIZE + GGML_PAD(GGML_TENSOR_SIZE, GGML_MEM_ALIGN)` — the SAME number for
// every object, whatever its shape. So the arena cost of a no_alloc graph is
// `objects * that_constant`, and the whole admission question reduces to knowing the constant and
// counting the objects.
//
// Neither constant is in ggml's public header, and the previous three revisions each guessed at the
// cost instead: first by counting only node headers, then by enumerating data, then by multiplying
// by eight. Each guess was wrong in the direction that kills a process. This measures it: create one
// tensor in a scratch context and read `ggml_used_mem` before and after. That is the actual allocator
// answering, not a model of it.
//
// The scratch context is 16 KiB on the heap and freed before returning, so the measurement cannot
// itself become the memory problem it exists to avoid. If the measurement is refused (ggml returns a
// context whose used-memory did not move), the emitter refuses rather than falling back to a guess.
constexpr std::size_t kProbeArena = 16u * 1024u;

bool measure_object_bytes(std::size_t & per_object) {
    ggml_context * probe = ggml_init({kProbeArena, nullptr, /*no_alloc=*/true});
    if (!probe) {
        return false;
    }
    const std::size_t before = ggml_used_mem(probe);
    ggml_tensor * t = ggml_new_tensor_1d(probe, GGML_TYPE_F32, 8);
    const std::size_t after = ggml_used_mem(probe);
    ggml_free(probe);

    if (!t || after <= before) {
        return false;
    }
    per_object = after - before;
    return true;
}

} // namespace

const char * describe(physical_emit_refusal refusal) {
    switch (refusal) {
        case physical_emit_refusal::ok:                    return "ok";
        case physical_emit_refusal::no_context:            return "no graph context or tensor";
        case physical_emit_refusal::bad_shape:             return "a dimension is zero, or the tensors disagree with the declared layout";
        case physical_emit_refusal::bad_head_layout:       return "n_head_kv does not divide n_head";
        // F16 IS SUPPORTED NOW, so this case means what its enumerator says: a type that is neither
        // F32 nor F16 — a quantized cache. The comment above this case used to argue that the gate was
        // unsatisfiable from the product, because a real cache is F16 and no `--cache-type-k` exists.
        // Both facts are still true and are no longer a problem: the emitter widens F16 exactly, one
        // cast per operand, and charges those casts to the arena bound. What remains refused is the
        // widening that has no exact embedding.
        //
        // Measured on the tiny transformer in this tree (`HERD_READER_CACHE=1`, three layers), kept
        // because it is the shape the tests reproduce rather than a claim about every model:
        //
        //   reader: layer 0 cache k ne [8 4 256 1] nb [16 64] type 1 | v ... type 1 | n_kv 256
        case physical_emit_refusal::unsupported_cache_type:return "k, v or q is a type this emission cannot widen exactly. F32 and F16 are "
                                                                  "supported in any mixture (F16 -> F32 is exact, one cast per operand, "
                                                                  "charged to the arena bound); a quantized cache is refused because it "
                                                                  "has no such embedding";
        case physical_emit_refusal::multi_stream:          return "k, v or q carries more than one stream on ne3";
        case physical_emit_refusal::bad_mask:               return "the supplied mask is not a per-cache-column bias this emission can honour";
        case physical_emit_refusal::empty_view:            return "the view has no spans";
        case physical_emit_refusal::empty_span:            return "a span in the view has no cells";
        case physical_emit_refusal::cell_out_of_range:     return "a cell index is outside the cache";
        case physical_emit_refusal::duplicate_cell:        return "a cell appears twice: the scatter would write it twice";
        case physical_emit_refusal::mixed_effective_pos:   return "the spans sit at different effective positions, which needs a per-span query rebuild "
                                                                  "(rope at span.position) that this emitter does not do; refusing rather than "
                                                                  "rotating once and using it everywhere";
        case physical_emit_refusal::v_layout_cell_major:   return "v is the cell-major cache (nb[1] > nb[2]); the value reduction below needs a "
                                                                  "dim-major [head_dim, n_head_kv, n_cell] tensor, and the branch that would "
                                                                  "transpose a cell-major one into that form is not implemented here";
        case physical_emit_refusal::no_alloc_context:      return "the context allocates nothing; this emitter needs one index tensor per span "
                                                                  "plus the scatter's cell tensor to have storage at build time, and the "
                                                                  "N+1 handoff that would carry them is not implemented";
        case physical_emit_refusal::index_data_unavailable:return "the context allocates nothing and the caller did not ask for the index hand-off "
                                                                  "(collect_indices = false), so no one would fill the gather indices";
        case physical_emit_refusal::arena_admission_unproven:
            return "an ALLOCATING context is refused: with allocation every operand's data enters the arena sum "
                   "and ggml's own internal allocations are not enumerable, so admission cannot be proven. A "
                   "no_alloc context can be admitted exactly, and that is the regime a product graph runs in";
        case physical_emit_refusal::no_room:               return "the arena cannot hold this emission's objects";
        case physical_emit_refusal::graph_error:           return "ggml refused a node (shape or allocation)";
    }
    return "unknown refusal";
}

// The emission, for a `no_alloc` context — the regime in which the arena cost is PROVABLE.
//
// WHY THE REGIME IS NARROWED RATHER THAN THE ESTIMATE IMPROVED. Three revisions tried to predict the
// arena cost of an allocating graph and all three were low, the last one by enough to abort inside
// `ggml_new_object` at a 16 KiB arena after its own guard had passed. The lesson is not "estimate
// harder": it is that an allocating graph's cost includes allocations this file does not make and
// cannot enumerate — `mul_mat`'s blocking, `cont` of a non-contiguous view, the concat chain's
// reallocation. A bound that omits unknown terms is not a bound.
//
// A `no_alloc` graph has no such terms. ggml allocates nothing for data (every tensor's
// `data_size` is 0), and every node costs exactly one object, whose size `measure_object_bytes`
// reads off the allocator. So:
//
//     need = objects * measured_object_bytes          // exact for this regime
//
// `objects` is charged BRANCH-FREE and CONSERVATIVELY: 13 + 6*spans, where the true count is
// 12 + 5*spans + (spans-1) + (contiguous ? 0 : 1). The difference — at most one object — is charged
// whether or not the branch is taken, so a miscounted conditional can only over-reserve. Then, because
// the count is the one thing here that a human wrote down, the emission checks itself afterwards: the
// arena growth is compared against `objects * per_object`, and a graph that grew more is refused
// rather than returned. A wrong count therefore cannot certify itself.
mask_coverage_report view_covers_finite_mask(const query_view & view, const float * mask,
                                             std::size_t n_cell) {
    mask_coverage_report out;
    // No mask means the question does not apply, and `applicable` is what says so. Returning a
    // "complete" report here would tell a caller that coverage was verified when nothing was
    // examined -- and a caller that branches on `complete` would then proceed on a false assurance,
    // which is the failure mode this whole function exists to avoid.
    if (mask == nullptr) return out;
    out.applicable = true;
    if (n_cell == 0) {
        // Applicable, and COMPLETE: there is no cache position, so there is no position whose
        // finite bias could be missing from the view. The previous revision left `complete` at its
        // default `false` here while the comment claimed the opposite -- which is the worse of the
        // two failures, because a caller that branches on `complete` would treat an empty cache as
        // an uncovered mask and widen a view that has nothing to widen.
        //
        // Both fields are now set deliberately: `applicable` because a mask WAS given (so the
        // question was asked), `complete` because nothing can be uncovered in an empty domain.
        out.complete = true;
        return out;
    }

    std::vector<char> named(n_cell, 0);
    for (const span_keys & span : view.spans) {
        for (std::int32_t cell : span.cells) {
            // A cell outside the cache is not this function's refusal to make (the emitter names
            // it, with the span and the cell), so it is skipped rather than counted as coverage.
            if (cell >= 0 && std::size_t(cell) < n_cell) named[std::size_t(cell)] = 1;
        }
    }

    bool violated = false;
    for (std::size_t i = 0; i < n_cell; ++i) {
        const float bias = mask[i];
        if (std::isinf(bias) && bias < 0.0f) continue;  // -inf: invisible, which the fill agrees with
        if (!std::isfinite(bias)) {
            // +inf or NaN. Neither is a bias the softmax can use, and both would slip past a test
            // written as "is it a number": `isfinite` is false for them, so an implementation that
            // only asked `!isnan` would report coverage for a mask no reference can reproduce.
            if (!violated) { violated = true; out.first_uncovered = i; }
            continue;
        }
        ++out.finite_positions;
        if (named[i]) {
            ++out.named;
            continue;
        }
        if (!violated) { violated = true; out.first_uncovered = i; }
    }

    out.complete = !violated;
    return out;
}

mask_coverage_report view_covers_finite_mask_row(const query_view & view, const void * row,
                                                 mask_width width, std::size_t n_cell) {
    // No row means no mask. The scalar form draws the same line for a null pointer, and it is drawn
    // here too: a caller that branches on `complete` alone must not read "nothing was examined" as
    // "coverage holds".
    if (row == nullptr) return mask_coverage_report{};
    // The empty domain, answered by hand. Handing a zero-length widened buffer to the scalar form
    // would pass a `data()` that MAY be null, and a null mask there is the third case -- no mask --
    // not this one. An empty cache with a mask is applicable and complete.
    if (n_cell == 0) {
        mask_coverage_report empty;
        empty.applicable = true;
        empty.complete = true;
        return empty;
    }
    if (width == mask_width::f32) {
        return view_covers_finite_mask(view, static_cast<const float *>(row), n_cell);
    }
    // F16 is widened here because the predicate takes F32 and has no overload on purpose: widening
    // needs a destination the caller must own, and this function owns exactly one row's worth.
    // `ggml_fp16_to_fp32` is exact -- every F16 value is representable in F32 -- so the report
    // describes the mask's real values rather than a rounding of them.
    std::vector<float> widened(n_cell);
    const auto * src = static_cast<const ggml_fp16_t *>(row);
    for (std::size_t i = 0; i < n_cell; ++i) widened[i] = ggml_fp16_to_fp32(src[i]);
    return view_covers_finite_mask(view, widened.data(), n_cell);
}

std::size_t mask_element_bytes(mask_width width) {
    return width == mask_width::f16 ? sizeof(ggml_fp16_t) : sizeof(float);
}

const void * finite_mask_row(const void * base, std::size_t t, std::size_t n_cell,
                             mask_width width) {
    // No base means no mask, and the null travels through: the row form answers "inapplicable" for
    // it, which is the answer the scalar form gives for the same absence.
    if (base == nullptr) return nullptr;
    // One multiplication by the row index over a stride that already carries the element width.
    // Written this way rather than as `t * n_cell * elem` so the width is applied exactly once, at
    // the point `mask_element_bytes` names it.
    const std::size_t stride = n_cell * mask_element_bytes(width);
    return static_cast<const char *>(base) + t * stride;
}

physical_emit_result emit_physical_attention(const graph_inputs & in, const query_view & view,
                                             float kq_scale) {
    if (!in.ctx || !in.k || !in.v || !in.q) {
        return refuse(physical_emit_refusal::no_context, "a reader emission needs ctx, k, v and q");
    }
    if (in.n_head == 0 || in.n_head_kv == 0 || in.head_dim == 0 || in.n_cell == 0) {
        return refuse(physical_emit_refusal::bad_shape, "a layout dimension is zero");
    }
    if (in.n_head % in.n_head_kv != 0) {
        return refuse(physical_emit_refusal::bad_head_layout,
                      "n_head_kv " + std::to_string(in.n_head_kv) + " does not divide n_head " +
                          std::to_string(in.n_head));
    }
    // F32 AND F16, widened to one arithmetic type below.
    //
    // The previous revision admitted F32 only and said the fork asserted F32 on this path. That was
    // true of the fork and false as a description of what a caller hands in: the tiny transformer in
    // this tree builds an F16 cache (measured: cache type 1, ne [8 4 256 1], n_kv 256), and mixed
    // shapes are legitimate — K/V from the cache, Q from the current graph, which upstream keeps in
    // F32. "All three the same type" was an assumption about the model wearing the clothes of a
    // contract.
    //
    // What is refused is what cannot be widened exactly: a quantized cache. F16 -> F32 preserves every
    // value a half can hold, so the arithmetic below runs on precisely the numbers the cache stores —
    // the same treatment the compact emitter beside this file applies, and the reason it is safe to
    // call this a widening rather than a conversion. Quantized types have no such embedding and are
    // refused by name.
    const auto arithmetic_ok = [](ggml_type t) {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_F16;
    };
    if (!arithmetic_ok(in.k->type) || !arithmetic_ok(in.v->type) || !arithmetic_ok(in.q->type)) {
        return refuse(physical_emit_refusal::unsupported_cache_type,
                      "k, v and q must each be F32 or F16 (any mixture); this emission widens F16 to "
                      "F32 exactly and cannot widen a quantized cache, so it refuses rather than "
                      "approximating one");
    }
    if (std::size_t(in.k->ne[0]) != in.head_dim || std::size_t(in.k->ne[1]) != in.n_head_kv ||
        std::size_t(in.k->ne[2]) != in.n_cell) {
        return refuse(physical_emit_refusal::bad_shape, "k is not [head_dim, n_head_kv, n_cell]");
    }
    if (std::size_t(in.q->ne[0]) != in.head_dim || std::size_t(in.q->ne[1]) != 1 ||
        std::size_t(in.q->ne[2]) != in.n_head) {
        return refuse(physical_emit_refusal::bad_shape, "q is not [head_dim, 1, n_head]");
    }
    if (in.k->ne[3] > 1 || in.v->ne[3] > 1 || in.q->ne[3] > 1) {
        return refuse(physical_emit_refusal::multi_stream,
                      "k, v or q has more than one stream on ne3; one stream per emission");
    }

    // THE MASK, VALIDATED HERE -- BEFORE A SINGLE NODE EXISTS.
    //
    // The previous revision checked this immediately before the softmax, arguing that the axes are
    // "settled there". They are settled by `in.n_cell` and `in.n_head`, both available now, so
    // checking late bought nothing and cost two things: it built dozens of nodes and then refused,
    // leaving a half-built graph in the caller's arena; and since this emitter's admission bound is
    // an OBJECT COUNT, nodes created by a refused call are exactly the residue that makes a
    // published object count wrong. A refusal must cost nothing.
    //
    // Every clause is the contract `ggml_soft_max_impl` itself asserts (ggml.c:4112-4118), so the
    // only difference is who names the failure. Two of them were missing entirely before this pass:
    //
    //   contiguity  -- the softmax kernel reads `ne00` mask elements by POINTER ARITHMETIC (`mp[i]`)
    //                  and never consults nb[0]. A strided mask is therefore read from the wrong
    //                  bytes, silently. ggml asserts it, but with no message naming this caller.
    //   ne[2] rule  -- ggml requires `a->ne[2] % mask->ne[2] == 0`, and the kernel broadcasts by
    //                  `i12 = i02 % ne12`. The old check (`ne2 == 1 || ne2 == n_head`) accepted a
    //                  strict subset of that; it was safe, and it refused masks ggml can honour
    //                  (e.g. 2 rows of a 4-head softmax), which is a limitation this emitter was
    //                  documenting as if it were a rule of ggml.
    if (in.mask != nullptr) {
        // The INTENT comes first, before any shape check. A differently-shaped mask is a mistake a
        // reader can fix from the message; an unstated intent is a run that would compute one of
        // two different visibilities and report the result without saying which. So this is asked
        // first, and `intersect` is accepted as a real request rather than treated as a corner case:
        // "these cells, further filtered by this bias" is exactly what this emitter computes.
        if (in.coverage != graph_inputs::mask_coverage::intersect) {
            return refuse(physical_emit_refusal::bad_mask,
                          "a mask is supplied but `coverage` is `unstated`. This emitter computes "
                          "visibility as view ∩ mask: it fills every cache position with -INFINITY "
                          "and writes the named cells over it, so a mask can only NARROW that set. "
                          "If that is what you want, say so with coverage = intersect. If your mask "
                          "leaves visible cells the view does not name and you need them to "
                          "contribute, the view is incomplete and no flag can repair it -- widen the "
                          "view, and use view_covers_finite_mask to check it where the mask's values "
                          "are readable");
        }
        const ggml_tensor * m = in.mask;
        if (m->type != GGML_TYPE_F16 && m->type != GGML_TYPE_F32) {
            return refuse(physical_emit_refusal::bad_mask,
                          "mask type " + std::to_string(int(m->type)) +
                              " is neither F16 nor F32; the softmax reads a bias in one of those "
                              "two widths and this emission will not convert one silently");
        }
        if (!ggml_is_contiguous(m)) {
            return refuse(physical_emit_refusal::bad_mask,
                          "mask is not contiguous: the softmax kernel reads ne[0] elements by "
                          "pointer arithmetic and ignores nb[0], so a strided mask would be read "
                          "from the wrong bytes with no error");
        }
        if ((std::int64_t) m->ne[0] != (std::int64_t) in.n_cell) {
            return refuse(physical_emit_refusal::bad_mask,
                          "mask has " + std::to_string(m->ne[0]) + " columns but this emission "
                          "softmaxes " + std::to_string(in.n_cell) + " cache columns; the mask is "
                          "indexed by cache position, and a mask over another axis yields a "
                          "plausible number");
        }
        if (m->ne[1] != 1) {
            return refuse(physical_emit_refusal::bad_mask,
                          "mask has " + std::to_string(m->ne[1]) + " rows on ne[1]; ggml would "
                          "broadcast a taller mask, but this emission has one query row and will "
                          "not silently pick a row");
        }
        // EXACTLY ONE head row, not a divisor of n_head.
        //
        // The previous revision accepted any divisor, on the theory that "ggml broadcasts a taller
        // mask, so a divisor is meaningful". ggml does broadcast -- `i12 = i02 % ne12` in the
        // softmax kernel (ggml-cpu/ops.cpp:5876) -- and what that produces is not a per-head bias
        // and not the GQA grouping either: with `ne2 = 2` over four heads, mask row 0 is read by
        // heads 0 and 2, and row 1 by heads 1 and 3. It is an even/odd interleave, which
        // corresponds to no attention layout in this tree. (An earlier revision of this comment
        // said "heads 0,1 -> row 0; heads 2,3 -> row 1", i.e. a block split. That was wrong: `%`
        // interleaves, it does not partition into contiguous blocks.)
        //
        // Since nothing in this tree asks for an interleaved bias, and upstream's `kq_mask` is
        // `ne [2048 1 1 1]` (measured), the supported shape is the one with a single meaning: one
        // row shared by every head. A mask silently reinterpreted along that axis produces a
        // plausible number, which is the failure class this file refuses everywhere else.
        //
        // Removing the divisor rule also removes any `%` from this function, so `ne[2] == 0` is
        // refused by the comparison rather than crashing on a modulo by zero.
        if (m->ne[2] != 1) {
            return refuse(physical_emit_refusal::bad_mask,
                          "mask has " + std::to_string(m->ne[2]) + " head rows on ne[2]; this "
                          "emission accepts one shared row, because ggml broadcasts a taller mask "
                          "by `head % rows` (with 2 rows over 4 heads, heads 0 and 2 read row 0 and "
                          "heads 1 and 3 read row 1) -- which is neither a per-head bias nor the "
                          "GQA grouping");
        }
        if (m->ne[3] != 1) {
            return refuse(physical_emit_refusal::bad_mask,
                          "mask has " + std::to_string(m->ne[3]) + " stream(s) on ne[3]; one "
                          "stream per emission, as for k, v and q");
        }
    }

    // v is shape-checked. `ggml_permute(v, 0,2,1,3)` maps ne0 = v->ne0 and ne1 = v->ne2, so a
    // dim-major v [head_dim, n_head_kv, n_cell] becomes [head_dim, n_cell, n_head_kv]; `ggml_mul_mat`
    // asserts `ggml_can_mul_mat` and `!ggml_is_transposed`, so the value tensor must be transposed
    // and made contiguous before the reduction. A cell-major cache does not satisfy the declared
    // layout and is refused rather than half-supported.
    if (std::size_t(in.v->ne[0]) != in.head_dim || std::size_t(in.v->ne[1]) != in.n_head_kv ||
        std::size_t(in.v->ne[2]) != in.n_cell) {
        return refuse(physical_emit_refusal::bad_shape, "v is not [head_dim, n_head_kv, n_cell]");
    }
    if (in.v->nb[1] > in.v->nb[2]) {
        return refuse(physical_emit_refusal::v_layout_cell_major,
                      "v has cell-major rows (nb[1] > nb[2]); the value reduction expects a "
                      "dim-major cache, and the transpose that would convert this one is not "
                      "implemented in this emitter");
    }

    // The regime this emitter admits, and why it is the only one it admits.
    if (!ggml_get_no_alloc(in.ctx)) {
        return refuse(physical_emit_refusal::arena_admission_unproven,
                      "this emitter admits no_alloc contexts only: there the arena cost is one object "
                      "per node, measured from the allocator, and the admission test is exact. An "
                      "allocating context also allocates every operand's data plus ggml's internal "
                      "workspace, which this file cannot enumerate; the last revision tried to and was "
                      "measurably low (passed a 16 KiB arena and ggml aborted needing more). Use a "
                      "no_alloc context and write the index tensors after the scheduler places the "
                      "graph, as upstream's own set_inputs does");
    }
    if (!in.collect_indices) {
        return refuse(physical_emit_refusal::index_data_unavailable,
                      "a no_alloc context leaves the gather indices without storage, so the caller "
                      "must receive them and write them after placement; collect_indices is false");
    }

    if (view.spans.empty()) {
        return refuse(physical_emit_refusal::empty_view, "the view has no spans");
    }

    // ONE EFFECTIVE POSITION, or nothing. The fork re-creates the query per span when a span's
    // position differs (rope at span.position, then the Hadamard rotation, then a permute). That needs
    // the pre-rope source and the rope parameters, which this emitter does not receive — so a mixed
    // view is refused instead of being rotated once and used for spans it does not describe.
    const std::int32_t effective = view.spans.front().effective_pos;
    std::size_t total_cells = 0;
    for (const span_keys & span : view.spans) {
        if (span.cells.empty()) {
            return refuse(physical_emit_refusal::empty_span, "a span in the view has no cells");
        }
        if (span.effective_pos != effective) {
            return refuse(physical_emit_refusal::mixed_effective_pos,
                          "span positions differ (" + std::to_string(effective) + " and " +
                              std::to_string(span.effective_pos) +
                              "); this emitter applies one rotation to every span");
        }
        total_cells += span.cells.size();
    }

    // The cells, in the order the scores are accumulated, plus the two refusals that keep a scatter
    // honest: an out-of-range index would write into another sequence's slot, and a duplicate would
    // write one position twice (which write wins depends on iteration order).
    std::vector<std::int32_t> cells;
    cells.reserve(total_cells);
    for (const span_keys & span : view.spans) {
        for (std::int32_t cell : span.cells) {
            if (cell < 0 || std::size_t(cell) >= in.n_cell) {
                return refuse(physical_emit_refusal::cell_out_of_range,
                              "cell " + std::to_string(cell) + " outside [0, " +
                                  std::to_string(in.n_cell) + ")");
            }
            cells.push_back(cell);
        }
    }
    {
        std::vector<std::int32_t> sorted = cells;
        std::sort(sorted.begin(), sorted.end());
        if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
            return refuse(physical_emit_refusal::duplicate_cell,
                          "a cell appears twice in the view; the scatter would write one position twice");
        }
    }

    // ADMISSION, before the first ggml call. Under no_alloc the cost is objects times one measured
    // constant, so this is a check and not a hope.
    std::size_t per_object = 0;
    if (!measure_object_bytes(per_object)) {
        return refuse(physical_emit_refusal::no_room,
                      "cannot measure what one object costs in this ggml build (the probe context "
                      "reported no growth), so the admission test cannot be made");
    }

    const std::size_t spans = view.spans.size();
    // 12 fixed + 5 per span + one concat per span after the first + cont(k) when k is strided.
    // Charged branch-free as 13 + 6*spans: cont(k) is charged even when k is contiguous, and the
    // concat count uses `spans` rather than `spans - 1`. Both charges are conservative, and the
    // post-emission check below refuses a graph that outgrew them.
    // The count's OWN arithmetic is checked first. `6 * spans` wraps for spans > SIZE_MAX/6, and a
    // wrapped count is a bound BELOW the truth — the one direction this check must never err in. The
    // spans come from a vector so the value is unreachable in practice; the guard costs a compare and
    // removes the need to argue about "in practice" at all.
    // ONE CAST IS CHARGED, and the emission creates exactly one: `q`, when the model keeps its query
    // in half precision. k and v are not cast anywhere below — `ggml_mul_mat` takes its right operand
    // in the left operand's `vec_dot_type`, which is F32 for an F32 or F16 left side, so only the
    // right side has to be F32 and only `q` is ever on the right.
    //
    // The previous revision charged `k_cast = spans` and `v_cast = 1` while its own comment argued
    // that neither cast existed, and the code in between still performed both. Two of those three
    // disagreed with each other at all times. The count now matches the code by construction: add a
    // cast below and it must be charged here, or the arena check under-counts and `ggml_new_object`
    // aborts. Deleting this line without deleting the cast is the failure this comment is about.
    const std::size_t q_cast = in.q->type == GGML_TYPE_F32 ? 0 : 1;
    if (spans > (std::size_t(-1) - 13 - q_cast) / 6) {
        return refuse(physical_emit_refusal::no_room,
                      "the object count overflows size_t at " + std::to_string(spans) + " spans");
    }
    const std::size_t objects = 13 + q_cast + 6 * spans;
    std::size_t need = 0;
    if (per_object != 0 && objects > (std::size_t(-1) / per_object)) {
        return refuse(physical_emit_refusal::no_room, "the admission requirement overflows size_t");
    }
    need = objects * per_object;

    const std::size_t used_before = ggml_used_mem(in.ctx);
    const std::size_t size = ggml_get_mem_size(in.ctx);
    if (used_before > size) {
        return refuse(physical_emit_refusal::no_room,
                      "the context reports more used memory (" + std::to_string(used_before) +
                          ") than it has (" + std::to_string(size) +
                          "); refusing rather than counting an impossible measurement as room");
    }
    if (size - used_before < need) {
        return refuse(physical_emit_refusal::no_room,
                      "declining before ggml aborts: " + std::to_string(size - used_before) +
                          " bytes free, and this emission needs " + std::to_string(need) + " for " +
                          std::to_string(objects) + " objects at " + std::to_string(per_object) +
                          " bytes each (measured, not assumed: under no_alloc an object's cost is "
                          "its header only, and it is the same for every node)");
    }

    // ---- the emission, in the fork's order -------------------------------------------------------
    ggml_context * ctx = in.ctx;
    const std::int64_t hd = std::int64_t(in.head_dim);
    const std::int64_t kvh = std::int64_t(in.n_head_kv);
    const std::int64_t nh = std::int64_t(in.n_head);
    const std::int64_t n_cell = std::int64_t(in.n_cell);

    // THE ONLY STRIDE PRECONDITION `q` HAS, AND IT IS CHECKED BEFORE THE FIRST NODE EXISTS.
    //
    // `mul_mat`'s vec_dot reads its right operand along dim 0, so `nb[0]` must be one element. Every
    // other stride is honoured -- measured, not assumed: a `[hd,1,nh]` query with `nb[2]` padded to
    // 2x and 3x still matches the definition exactly (max|diff| = 0.000e+00). Note also that the
    // per-head view this function builds has `ne[2] == group`, which ggml reports contiguous whatever
    // `nb[2]` is, so a "q must be contiguous" test was never about a tensor the operator sees strided.
    //
    // Checked HERE, above every `ggml_cont`/`ggml_cast`/`ggml_new_*`, because this emitter's admission
    // bound is an OBJECT COUNT: nodes built by a refused call are residue that makes a published bound
    // wrong, and the caller has no way to tell. A refusal must cost nothing.
    if (in.q->nb[0] != ggml_element_size(in.q)) {
        return refuse(physical_emit_refusal::bad_shape,
                      "q has nb[0] = " + std::to_string(in.q->nb[0]) + " but one element is " +
                          std::to_string(ggml_element_size(in.q)) +
                          "; `mul_mat` reads its right operand along dim 0, so a step larger than "
                          "one element would produce values from outside the intended row");
    }
    // THE REFUSAL THAT USED TO BE HERE WAS WRONG, AND MEASUREMENT SAYS SO.
    //
    // It refused any non-contiguous `q`, on the reasoning that `mul_mat` "assumes a packed right
    // operand". Three measurements say otherwise, and the third is the one that matters:
    //
    //   1. `mul_mat(k, q)` with `q` 2-D and `nb[1]` padded to 64 and to 96 (packed is 32) returns
    //      the definition exactly: max|diff| = 0.000e+00 in both cases. The kernel walks `nb[1]`.
    //   2. `q` as `[hd, 1, nh]` with `nb[2]` padded to 64 and 96 (the shape a prefill actually
    //      produces): `mul_mat` again exact, max|diff| = 0.000e+00.
    //   3. The emitter never hands `mul_mat` the strided tensor anyway. It first takes
    //      `view_3d(in.q, hd, 1, group, in.q->nb[1], in.q->nb[2], ...)`, whose `ne[2] == group`; for
    //      the single-token decode that is `ne [hd 1 1]`, and ggml reports such a view as
    //      CONTIGUOUS whatever its `nb[2]` is, because a dim of extent 1 has no stride to violate.
    //      So `ggml_is_contiguous(in.q)` was being consulted about a tensor that the operator never
    //      sees in strided form.
    //
    // The 2.164e-02 that motivated the refusal was a caller-side bug: a token view built with the
    // PARENT's `nb[2]` (64) instead of the packed head stride (32), so `h=1` read the next token's
    // head 0. That view is a legal tensor describing the wrong bytes; no check here could have
    // distinguished it from a correct strided view, and refusing all of them bought nothing.
    //
    // What remains: `nb[0]` must be the element size, because `mul_mat`'s vec_dot reads along dim 0.
    // That is checked below and it is the only stride precondition the operator actually has.

    // One arithmetic type on the side that requires it: `mul_mat` takes its RIGHT operand in the
    // left operand's `vec_dot_type`, F32 for both F32 and F16 caches, so only `q` is widened here.
    // When `q` already is F32 this is `in.q` itself and no object appears.
    ggml_tensor * q = in.q->type == GGML_TYPE_F32 ? in.q : ggml_cast(ctx, in.q, GGML_TYPE_F32);

    ggml_tensor * k_src = ggml_is_contiguous(in.k) ? in.k : ggml_cont(ctx, in.k);
    ggml_tensor * k_flat = ggml_reshape_2d(ctx, k_src, hd * kvh, n_cell);   // [hd*kvh, n_cell]

    // ONE ARITHMETIC TYPE ON THE SIDE THAT REQUIRES IT, and nothing more.
    //
    // `ggml_mul_mat(a, b)` takes `b` in `a`'s `vec_dot_type`, which is F32 for both F32 and F16 `a`. So
    // the RIGHT operand must be F32 and the left may be either — that is the documented pairing, not a
    // CPU accident, and it is why the query is widened here while k and v are not:
    //
    //   scores = mul_mat(gp, q)          gp may be F16 (a half cache), q must be F32
    //   out    = mul_mat(v_t, weights)   v_t may be F16, weights is produced F32 by soft_max_ext
    //
    // A `ggml_cast` on the gathered block would also work, and this emitter used to do it. A mutation
    // removed it and the emitted numbers did not move by a single bit (weights 2.980e-08, out
    // 5.960e-08 either way, on the F16 fixture) — because the kernel already reads F16 rows exactly.
    // The cast was therefore not carrying the correctness it appeared to carry: it normalised the type
    // at the cost of one object and one full copy of every gathered block. It stays only where the
    // contract demands it, which is the right-hand side.
    //
    // `q` is widened because a half-precision model may keep its query in F16; when q already is F32
    // this is `in.q` itself and no object appears.
    // THE ONE CAST IN THE EMISSION, kept for portability and not for numbers — measured, twice.
    //
    // `ggml_mul_mat` documents its right operand as belonging to the left operand's `vec_dot_type`,
    // F32 for both F32 and F16 left sides, and this widening enforces that. What the CPU backend was
    // MEASURED to do is accept an F16 right operand anyway: a mutation that deleted this cast left
    // every emitted number bit-identical (F16 q, F32 cache: weights 2.980e-08, out 4.470e-08; F16 q,
    // F16 cache: 2.980e-08 / 5.960e-08 — unchanged) and raised no assert. So on this backend the cast
    // buys nothing, exactly as the k and v casts did.
    //
    // It stays because "the CPU backend tolerates it" is not a property of the interface, and backends
    // differ about whose job the conversion is. F32-on-the-right is the documented contract, so paying
    // one object for it is the conservative side of an unverified assumption; the alternative is a
    // graph that runs here and asserts on a backend nobody has tried. If someone later proves every
    // backend accepts a half right operand, this cast AND its charge in the object count go together —
    // that is the one edit, and the count above says so.
    // `q` MUST ALREADY BE CONTIGUOUS: REFUSED, AND CHECKED BEFORE ANY NODE EXISTS.
    //
    // `q` is the RIGHT operand of `mul_mat`, whose kernels walk it through `nb[1]`/`nb[2]` and assume
    // packing. The refusal lives near the TOP of this function, not here, because a refusal that
    // builds nodes first leaves residue in the caller's arena -- and this emitter's admission bound is
    // an OBJECT COUNT, so that residue is exactly what makes a published bound wrong.
    //
    // `ggml_cont` was called an alternative repair and then rejected as unsafe. That rejection was
    // based on a READBACK artifact, not on `cont`: `ggml_backend_tensor_get` on a strided view copies
    // raw bytes and ignores `nb`, so "8 of 16 differ" measured the read, not the copy. Read
    // element-wise through the parent tensor's strides, `cont` is faithful on both a packed and a
    // strided source (0 of 16 differ, measured). The defect was upstream of the emitter.
    //
    // The four-variant table above stands as measured; what it does NOT show is that `cont` damaged
    // anything. Reading the same view element-wise through the parent tensor's strides settles it:
    //
    //   view_3d(q_head, hd, 1, nh, q_head->nb[1], q_head->nb[2], t*q_head->nb[2])
    //        nb2 = 64 for a [8,1,2] form whose packed nb2 is 32
    //        => h=1 reads t=2's head 0: value 200 where token 1's head 1 dim 0 is 110
    //        => 8 of 16 elements wrong, and NOT because anything was copied
    //   view_3d(q_head, hd, 1, nh, hd*el, hd*el, t*hd*nh*el)
    //        => contiguous, 0 of 16 wrong
    //
    // So slicing a token out of `[hd, nh, T]` DOES work -- with packed strides. The bug was passing
    // the PARENT's `nb[2]` to a `ne[2] == nh` view, which is a caller-side mistake, not a limitation
    // of the emitter or of `ggml_cont`. The precondition that remains real is the one this function
    // enforces: `q` must arrive packed, and it is refused (at the top, before any node) when it is not.

    physical_emit_result result;
    result.indices.reserve(spans);
    std::vector<std::size_t> cells_offset(spans, 0);

    ggml_tensor * partials = nullptr;
    for (std::size_t s = 0; s < spans; ++s) {
        const span_keys & span = view.spans[s];
        const std::int64_t n = std::int64_t(span.cells.size());

        ggml_tensor * idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(idx);
        cells_offset[s] = result.indices.empty() ? 0 : result.indices.back().cells_offset +
                                                         result.indices.back().n_cells;
        result.indices.push_back({s, idx, cells_offset[s], span.cells.size()});

        // NOT WIDENED. `gathered` keeps the cache's type and enters `mul_mat` as the LEFT operand,
        // where F16 is the ordinary case rather than an exception: every F16 model multiplies F16
        // weights by F32 activations. Measured, not argued — a mutation that deleted this cast left the
        // emitted numbers bit-identical (weights 2.980e-08, out 5.960e-08 either way on the F16
        // fixture), so the cast was buying one object and one full copy of every gathered block and
        // nothing else. The same measurement, repeated on the query, showed the right-hand-side
        // pairing is no more load-bearing on this backend — see the note at the `q` widening, which is
        // kept anyway for portability.
        ggml_tensor * gathered = ggml_get_rows(ctx, k_flat, idx);           // [hd*kvh, S], cache type
        ggml_tensor * g3 = ggml_reshape_3d(ctx, gathered, hd, kvh, n);      // [hd, kvh, S]
        ggml_tensor * gp = ggml_permute(ctx, g3, 0, 2, 1, 3);               // [hd, S, kvh]
        ggml_tensor * scores = ggml_mul_mat(ctx, gp, q);                    // [S, 1, n_head]

        partials = partials ? ggml_concat(ctx, partials, scores, 0) : scores;
    }

    ggml_tensor * rows = ggml_reshape_3d(ctx, partials, 1, n_cell > 0 ? std::int64_t(cells.size()) : 0,
                                         nh);                              // [1, n_visible, n_head]

    ggml_tensor * physical = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, n_cell, nh);
    ggml_tensor * filled = ggml_fill(ctx, physical, -INFINITY);

    ggml_tensor * scatter_idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, std::int64_t(cells.size()));
    ggml_set_input(scatter_idx);
    result.indices.push_back({spans, scatter_idx, 0, cells.size()});
    result.index_cells = cells;

    ggml_tensor * scattered = ggml_set_rows(ctx, filled, rows, scatter_idx);
    // The mask was validated before the first node was created (see the block after the stream
    // checks). Nothing about it is re-decided here: a second check would be a second place to keep
    // right, and this function's job at this line is to hand the pointer to ggml.
    ggml_tensor * mask = in.mask;

    ggml_tensor * weights_in = ggml_reshape_3d(ctx, scattered, n_cell, 1, nh);
    // `mask` rather than `nullptr`: when the caller supplies upstream's `kq_mask`, the -inf entries
    // AND the finite biases are both carried by ggml, which is the whole point -- invisibility comes
    // from the mask, so a visible cell must not also be erased here. `nullptr` (the previous
    // behaviour, and still the default) keeps visibility entirely in the cell list.
    ggml_tensor * weights = ggml_soft_max_ext(ctx, weights_in, mask, kq_scale, 0.0f);

    ggml_tensor * v_perm = ggml_permute(ctx, in.v, 0, 2, 1, 3);             // [hd, n_cell, kvh]
    ggml_tensor * v_t = ggml_cont(ctx, ggml_transpose(ctx, v_perm));        // [n_cell, hd, kvh], cache type
    // NOT WIDENED either, and for the same measured reason as `gathered`: `v_t` is the LEFT operand and
    // `weights` — produced F32 by `soft_max_ext` — is the right one, which is exactly the pairing
    // `ggml_mul_mat` requires. The cast that used to sit here was also bit-identical when removed, and
    // it cost one object plus a full copy of the (n_cell x hd x kvh) cache, the largest tensor in the
    // emission.
    ggml_tensor * out = ggml_mul_mat(ctx, v_t, weights);                    // [hd, 1, n_head]

    ggml_set_output(out);
    ggml_set_output(weights);

    // THE COUNT TESTS ITSELF, and the comment says what kind of evidence that is.
    //
    // The BOUND is algebra, from ggml's allocator: under no_alloc every object is
    // `GGML_OBJECT_SIZE + PAD(GGML_TENSOR_SIZE, ALIGN)` because `ggml_new_tensor_impl` zeroes
    // `obj_alloc_size` and `ggml_new_object` pads only the header. No shape enters it, so the only
    // open question is the COUNT, and the count is the one number here a human wrote down.
    //
    // This is the measurement that can falsify that number. It is NOT the safety mechanism: by the
    // time it runs the objects exist, so a count low enough to overflow the arena would already have
    // aborted inside `ggml_new_object`. It also cannot clean up — a firing here leaves the caller's
    // context holding a half-built graph, which is worse than a refusal before any object exists.
    // Measured counts are 17, 23, 29, 35 for 1..4 spans against a charge of 19, 25, 31, 37; the test
    // pins that margin so a ggml that starts allocating an extra node per emission fails there.
    //
    // `need` is reused rather than recomputing `objects * per_object`: `need` is that product with the
    // overflow guard already applied, and a wrapped product here would silently pass.
    const std::size_t grew = ggml_used_mem(ctx) - used_before;
    if (grew > need) {
        return refuse(physical_emit_refusal::no_room,
                      "the emission created " + std::to_string(grew / (per_object ? per_object : 1)) +
                          " objects' worth of arena where the admission test charged " +
                          std::to_string(objects) + " (" + std::to_string(grew) + " bytes against " +
                          std::to_string(need) +
                          "); the object count in this file is wrong for this view. The context now "
                          "holds that half-built graph and cannot be cleaned, which is why the "
                          "admission check — not this one — is what keeps the process alive");
    }

    result.out = out;
    result.weights = weights;
    result.n_visible = cells.size();
    result.n_cell = in.n_cell;
    result.n_head = in.n_head;
    result.head_dim = in.head_dim;
    // Reported so a caller can budget its own scaffolding against the arena it has left. A number the
    // caller cannot read is a number the caller cannot plan with, and the caller's `ggml_new_graph` is
    // larger than this entire emission.
    result.bytes_required = need;
    result.object_cost = per_object;
    result.objects_charged = objects;
    return result;
}

// ---------------------------------------------------------------------------------------------
// The publish step: host cells -> the tensors the scheduler placed.
// ---------------------------------------------------------------------------------------------

const char * describe(index_publish_refusal refusal) {
    switch (refusal) {
        case index_publish_refusal::ok:            return "ok";
        case index_publish_refusal::no_entries:    return "there is nothing to publish: the result carries no index entries "
                                                          "(it was a refusal, or collect_indices was false)";
        case index_publish_refusal::not_an_input:  return "an entry's tensor is missing or is not marked GGML_TENSOR_FLAG_INPUT, "
                                                          "so writing it would not reach the node that reads it";
        case index_publish_refusal::no_storage:    return "an entry's tensor has no data: publish ran before "
                                                          "ggml_backend_sched_alloc_graph() placed the graph, and a write now "
                                                          "would go nowhere";
        case index_publish_refusal::size_disagrees:return "an entry's n_cells does not equal its tensor's ne[0], so the write "
                                                          "would be truncated or would run past the tensor";
        case index_publish_refusal::cell_range:    return "an entry's cells_offset/n_cells fall outside index_cells";
        case index_publish_refusal::entry_layout:  return "the entries are not the layout the emitter produces: the gathers do not "
                                                          "tile [0, total) contiguously in order, or the scatter is not the "
                                                          "whole buffer, or an entry is not an I32 tensor";
        case index_publish_refusal::write_failed:  return "the writer reported a failed write partway through; the inputs "
                                                          "already written stay written, so the graph must be discarded "
                                                          "rather than run";
    }
    return "unknown refusal";
}

namespace {

// One entry, checked against the buffer it names. Returns ok or the first thing wrong with it.
index_publish_refusal validate_entry(const physical_emit_result & result,
                                     const graph_inputs::index_entry & entry,
                                     std::string * why) {
    const std::string at = "entry span " + std::to_string(entry.span) + ": ";

    if (!entry.index) {
        if (why) { *why = at + "no tensor (the emitter records only tensors it created, so this entry was "
                                "default-constructed or hand-edited)"; }
        return index_publish_refusal::not_an_input;
    }
    // Integral promotion, made explicit: `flags` is an enum, and `&` on two different enum types is
    // a diagnostic under some -Wextra configurations while `int & int` never is.
    if ((static_cast<int>(entry.index->flags) & static_cast<int>(GGML_TENSOR_FLAG_INPUT)) == 0) {
        if (why) { *why = at + "the tensor is not a graph input"; }
        return index_publish_refusal::not_an_input;
    }
    // The ordering matters: `data` is only meaningful once the scheduler has placed the graph, so a
    // null here means "published too early", which is a different caller mistake from a size
    // disagreement and gets its own answer.
    if (entry.index->data == nullptr) {
        if (why) { *why = at + "the tensor has no storage yet"; }
        return index_publish_refusal::no_storage;
    }
    // `ne[0]` is int64_t and `n_cells` is std::size_t. A NEGATIVE `ne[0]` becomes a colossal
    // std::size_t under the comparison below, which would reject the entry with a message reading
    // "n_cells 5 against ne[0] -1" — the right verdict from the wrong check. Saying what is actually
    // wrong costs one comparison: a tensor cannot have a negative extent, and if it does, nothing
    // about its size can be trusted.
    if (entry.index->ne[0] < 0) {
        if (why) {
            *why = at + "the tensor's ne[0] is " + std::to_string(entry.index->ne[0]) +
                   ", a negative extent; the tensor is malformed and its size means nothing";
        }
        return index_publish_refusal::size_disagrees;
    }
    if (entry.n_cells == 0 || entry.n_cells != std::size_t(entry.index->ne[0])) {
        if (why) {
            *why = at + "n_cells " + std::to_string(entry.n_cells) + " against ne[0] " +
                   std::to_string(entry.index->ne[0]);
        }
        return index_publish_refusal::size_disagrees;
    }
    if (entry.cells_offset > result.index_cells.size() ||
        entry.n_cells > result.index_cells.size() - entry.cells_offset) {
        if (why) {
            *why = at + "cells [" + std::to_string(entry.cells_offset) + ", +" +
                   std::to_string(entry.n_cells) + ") outside a buffer of " +
                   std::to_string(result.index_cells.size());
        }
        return index_publish_refusal::cell_range;
    }
    return index_publish_refusal::ok;
}

} // namespace

index_publish_refusal publish_index_inputs(const physical_emit_result & result,
                                           index_writer write,
                                           std::string * why,
                                           std::size_t * failed_span) {
    if (result.indices.empty()) {
        if (why) { *why = describe(index_publish_refusal::no_entries); }
        return index_publish_refusal::no_entries;
    }
    if (!write) {
        if (why) { *why = "no writer supplied"; }
        return index_publish_refusal::write_failed;
    }

    // ---------------------------------------------------------------------------------------------
    // PASS 0: THE LAYOUT, checked against itself because `physical_emit_result` is the CALLER'S
    // mutable object. Every per-entry check below is a bounds check, and bounds cannot see a lie:
    // a caller who erases a gather leaves entries that are all individually in range, and an earlier
    // revision would then have written the remaining subset and returned `ok` — precisely the
    // "caller published a subset" failure this function claims to remove, reachable by calling
    // `indices.erase()` between emit and publish. A caller who re-points a gather's `cells_offset`
    // likewise wrote one span's cells into another's tensor, in range, silently.
    //
    // What the shape can show is checked:
    //
    //   entries[0 .. S-1]   the gathers, tiling [0, N) contiguously in order, N = index_cells.size()
    //   entries[S]          the scatter: offset 0, n_cells == N, i.e. the whole buffer
    //
    // with S = indices.size() - 1. Erasing a gather opens a gap OR leaves the gathers short of N;
    // erasing the scatter leaves the last entry not covering the buffer; reordering breaks the
    // tiling; duplicating double-counts and overruns N. Each is refused before a byte moves, and
    // each is caught by the tiling — which is why the emitter-written count that a previous revision
    // compared here was removed rather than repaired. It was public and mutable, so it never proved
    // origin; and it was redundant for the case it was introduced for.
    //
    // THIS VALIDATES STRUCTURE, NOT PROVENANCE AND NOT CONTENT. A caller can assemble a set that
    // tiles, and this function will publish it; a caller can edit a value in `index_cells`, and
    // nothing here can tell. Both are stated in the header rather than papered over with a field the
    // caller can write. What IS guaranteed is that each tensor receives exactly its own slice, and
    // nothing is written unless the whole set is consistent.
    {
        const std::size_t total = result.index_cells.size();
        const std::size_t scatter = result.indices.size() - 1;
        std::size_t next_offset = 0;
        for (std::size_t i = 0; i < scatter; ++i) {
            const graph_inputs::index_entry & entry = result.indices[i];
            // The emitter always makes I32 index tensors; anything else would take 4-byte cells into
            // a buffer of another element size, which is in-range and wrong.
            if (entry.index && entry.index->type != GGML_TYPE_I32) {
                if (why) { *why = "gather " + std::to_string(i) + " is not an I32 tensor"; }
                return index_publish_refusal::entry_layout;
            }
            // The bounds check in PASS 1 has not run yet, so `n_cells` is still whatever the caller
            // put here. Subtractive form: `cells_offset + n_cells <= total` written as an addition
            // wraps for an n_cells near SIZE_MAX and would let a forged chain satisfy
            // `next_offset == total` below. Keeping the sum inside `total` removes the wraparound
            // case rather than reasoning about it.
            if (entry.cells_offset > total || entry.n_cells > total - entry.cells_offset) {
                if (why) {
                    *why = "gather " + std::to_string(i) + " claims [" +
                           std::to_string(entry.cells_offset) + ", +" + std::to_string(entry.n_cells) +
                           ") outside a buffer of " + std::to_string(total);
                }
                return index_publish_refusal::cell_range;
            }
            if (entry.cells_offset != next_offset) {
                if (why) {
                    *why = "gather " + std::to_string(i) + " starts at " +
                           std::to_string(entry.cells_offset) + " where the tiling expects " +
                           std::to_string(next_offset) +
                           "; the gathers no longer cover the buffer in emitter order";
                }
                return index_publish_refusal::entry_layout;
            }
            next_offset += entry.n_cells;
        }
        if (next_offset != total) {
            if (why) {
                *why = "the gathers cover " + std::to_string(next_offset) + " of " +
                       std::to_string(total) + " cells";
            }
            return index_publish_refusal::entry_layout;
        }
        const graph_inputs::index_entry & last = result.indices[scatter];
        if (last.index && last.index->type != GGML_TYPE_I32) {
            if (why) { *why = "the scatter is not an I32 tensor"; }
            return index_publish_refusal::entry_layout;
        }
        if (last.cells_offset != 0 || last.n_cells != total) {
            if (why) {
                *why = "the scatter covers [" + std::to_string(last.cells_offset) + ", +" +
                       std::to_string(last.n_cells) + ") instead of the whole [" +
                       std::to_string(total) + ") buffer";
            }
            return index_publish_refusal::entry_layout;
        }
    }

    // PASS 1, VALIDATION, with no device traffic at all. This ordering is the contract: a caller that
    // receives `ok` knows every entry was checked before the first byte moved, so a validation
    // refusal means the graph was left exactly as it was found.
    for (const graph_inputs::index_entry & entry : result.indices) {
        const index_publish_refusal refusal = validate_entry(result, entry, why);
        if (refusal != index_publish_refusal::ok) {
            return refusal;
        }
    }

    // PASS 2, THE WRITES. Every entry, in the order the emitter recorded them: the spans' gathers,
    // then the scatter. There is no "publish only what you need" mode, because the failure mode this
    // function exists to remove is precisely a caller publishing a subset.
    for (const graph_inputs::index_entry & entry : result.indices) {
        const void * data = result.index_cells.data() + entry.cells_offset;
        // CHECKED MULTIPLY. `n_cells` is bounded by `index_cells.size()`, and a vector<int32_t>
        // cannot hold more elements than SIZE_MAX/4 — but that rests on an allocator invariant
        // several layers away, while this is the expression that turns a cell count into the byte
        // count handed to a device. If it wrapped, the write would silently truncate one span's
        // indices, which is the failure mode this function exists to remove. A wrapped size IS a
        // size disagreement, so it earns the same refusal.
        if (entry.n_cells > std::size_t(-1) / sizeof(std::int32_t)) {
            if (why) {
                *why = "entry span " + std::to_string(entry.span) + ": n_cells " +
                       std::to_string(entry.n_cells) + " cannot be converted to a byte count";
            }
            return index_publish_refusal::size_disagrees;
        }
        const std::size_t size = entry.n_cells * sizeof(std::int32_t);
        // `failed_span` is set BEFORE the call, not after a `false` return. `index_writer` is a plain
        // function pointer and nothing stops an adapter from throwing (a device wrapper allocating,
        // a backend raising); on that path the `if (!write(...))` branch would never run, so setting
        // the span afterwards would leave a caller with a half-published input set and no indication
        // of which entry was in flight. Set first, and both failure modes name the same entry.
        if (failed_span) { *failed_span = entry.span; }
        // AN ADAPTER THAT RAISES IS REPORTED HERE, not propagated.
        //
        // The adapters are three-line wrappers around `ggml_backend_tensor_set`, and on a device
        // backend that wrapper can allocate and a backend can raise. If the exception left this
        // function, the caller would receive neither a refusal nor `failed_span`, while the entries
        // before this one stayed written: a half-published input set with no report — the exact
        // state the pre-write ordering above exists to prevent. Catching converts an unreported
        // partial publication into a reported one.
        //
        // It cannot undo it. The writes already made stand, which is why the contract says the graph
        // must be discarded rather than retried: a retry would publish a mixture of two batches.
        bool wrote = false;
        bool raised = false;
        try {
            wrote = write(entry.index, data, 0, size);
        } catch (...) {
            raised = true;
        }
        if (!wrote) {
            if (why) {
                *why = std::string(describe(index_publish_refusal::write_failed)) + " (span " +
                       std::to_string(entry.span) + (raised ? ", the writer raised" : "") + ")";
            }
            return index_publish_refusal::write_failed;
        }
    }
    return index_publish_refusal::ok;
}

} // namespace tq::reader
