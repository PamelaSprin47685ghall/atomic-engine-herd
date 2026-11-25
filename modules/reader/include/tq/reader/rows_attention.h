// tq/reader/rows_attention.h — the guard for upstream's runtime row table.
//
// `build_attn_mha` receives `kv_rows`: an I32 INPUT tensor, `[n_kv, n_slices]`, whose contract
// ggml.h states in one line — "mask column i of slice i3 uses row kv_rows[i, i3] of k and v;
// negative rows are padding, their mask entries must be -inf".
//
// TWO CORRECTIONS TO WHAT THIS FILE USED TO SAY. Both were checked against the sources in this
// tree, and both were wrong in a direction that made the problem look worse than it is:
//
//   1. `ggml_get_rows` DOES bounds-check on the CPU backend. `ggml-cpu/ops.cpp` asserts
//      `GGML_ASSERT(i01 >= 0 && i01 < ne01)` in all four row-gather kernels (5312, 5353, 5394,
//      5435), and `GGML_ASSERT` is never compiled out. A negative or too-large row therefore
//      ABORTS; it does not read the neighbouring memory and hand back a plausible number. So the
//      hazard is "wrong input kills the process loudly", not "wrong input answers wrongly".
//   2. `GGML_OP_CLAMP` has a CPU implementation for F32, F16 and BF16 -- it is not missing. What
//      is missing is an I32 branch: the dispatch in `ggml_compute_forward_clamp` reaches
//      `GGML_ABORT("fatal error")`, so clamping an index tensor terminates the process.
//
// WHY THE GUARD STILL EXISTS, NOW THAT THE INDEX IS NOT A SILENT CORRUPTION. Two reasons, neither
// of which is "otherwise we read the wrong memory":
//
//   * A row list is a MIX of padding and real cells, and `-1` is legal padding. Handing it to
//     `ggml_get_rows` unchanged aborts on the first padded column -- an abort where the input was
//     valid, which is a defect of the consumer, not of the data.
//   * Upstream's own consumer does not abort and does not delete columns: it clamps
//     (`std::max<int32_t>(rows[i], 0)` in the flash-attention CPU kernel) and relies on the padding
//     columns' masks being `-inf`, whose softmax weight is exactly zero. This file reaches the same
//     VALUES by a different route -- dropping those columns -- which is value-preserving precisely
//     because the weight is zero, and which keeps the row list free of `-1` for consumers that
//     cannot clamp.
//
// Contract kept, not narrowed: ANY negative row is padding, exactly as ggml.h says. It is not
// restricted to `-1`, and a caller that emits `-7` as padding is not refused for being unusual.
// What is enforced is the price of dropping it: its mask column must be `-inf`. A negative row with
// a visible mask is refused by name, because dropping that column would delete a real term and
// renormalise the rest -- the one case where the compacted result would differ from upstream.
//
// WHAT THIS IS NOT: an emitter, and not a seam. It builds no graph, indexes nothing, and nothing in
// the production decode path calls it. The observation probes in `patches/herd_reader_attn.cpp`
// print the row table; they do not feed this guard, and no lane consumes its output today.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <ggml.h>

namespace tq::reader {

// ---------------------------------------------------------------------------------------------
// Compaction: the only place a negative row may be handled, because it is the only place values
// are visible.
// ---------------------------------------------------------------------------------------------

enum class rows_guard_refusal {
    ok,
    size_mismatch,                 // rows and mask disagree on length, or a null with length > 0
    negative_row_mask_not_neg_inf, // a negative row whose mask column is not -inf
    row_out_of_range,              // a non-negative row outside [0, n_cell)
};

const char * describe(rows_guard_refusal refusal);

struct compacted_rows {
    rows_guard_refusal refusal = rows_guard_refusal::ok;
    std::string why;
    std::vector<std::int32_t> rows;   // every non-negative row, original order
    std::vector<float>        mask;   // the matching columns, same order
    std::size_t dropped = 0;          // how many padding columns went away
};

// The two lengths are SEPARATE arguments on purpose. A single `n` made the most error-prone step
// the caller's responsibility: pass a length larger than either buffer and the guard itself reads
// out of bounds while appearing to validate. With both lengths stated, the guard can refuse a
// disagreement instead of trusting one of them, and the caller has to have both numbers in hand.
//
// WHAT THE TWO LENGTHS DO **NOT** BUY, stated because an earlier version of this comment implied
// otherwise. The guard compares `n_rows` against `n_mask`; it cannot compare either against the
// buffer they point at. A caller that passes 100 for both while holding two 10-element arrays gets
// `ok` and 90 elements of whatever follows in memory. Nor can it detect a pointer that is non-null
// and invalid. So the preconditions this function cannot check are, plainly:
//
//   * both pointers refer to at least `n_{rows,mask}` readable elements;
//   * the two lengths describe the same row list, in the same order.
//
// What it can check, it does: the two lengths agree, a null pointer with a non-zero length is
// refused (a length is a claim about a buffer that is not there), every negative row carries `-inf`
// and is dropped, and every non-negative row is inside `[0, n_cell)`. Those are total over any
// input that honours the two preconditions above -- which is the honest scope of the claim, and not
// the same thing as "total over its input".
//
// Empty is legal and means "no visible history": `n_rows == 0` with a null or non-null pointer is
// `ok` with empty vectors.
//
// `n_cell` is the cache depth; a non-negative row must satisfy `row < n_cell`.
compacted_rows compact_rows(const std::int32_t * rows, std::size_t n_rows,
                            const float * mask, std::size_t n_mask, std::int32_t n_cell);

// F16 mask, the type upstream actually uses for `kq_mask` (measured: `type 1`).
compacted_rows compact_rows_f16(const std::int32_t * rows, std::size_t n_rows,
                                const ggml_fp16_t * mask, std::size_t n_mask, std::int32_t n_cell);

// ---------------------------------------------------------------------------------------------
// THERE IS NO GRAPH HERE, ON PURPOSE, AND THIS IS THE REASON.
// ---------------------------------------------------------------------------------------------
//
// An earlier version of this file also emitted the attention graph, taking `rows` as an I32 graph
// input -- the shape upstream hands to `build_attn_mha`. It was removed for two reasons, and the
// first one is not "the index corrupts memory":
//
//   1. A graph cannot branch, so it cannot drop padding columns. The only in-graph repair is to
//      clamp, and an I32 clamp ABORTS (`GGML_OP_CLAMP` has no I32 branch). Aborting on a legal
//      `kv_rows` that contains padding is not a safety property, it is a crash on valid input.
//   2. The values arrive at COMPUTE time (measured: `input 1 data no` at the `build_attn_mha`
//      hook), so no build-time check can see them, and the choice of what to do with padding has
//      to be made before the graph is built.
//
// So the work moved to where the values are visible: `compact_rows` handles padding on the host,
// drops it only after proving the mask is `-inf`, and refuses everything else by name. It is total
// over the row list it is given -- subject to the two preconditions it cannot check (the pointers
// are readable to the stated lengths, and the lengths describe the same list). A consumer that
// wants this guard's output must have the row values on the host -- which is true after upstream's
// `set_input_attn_mask` fills them, and false at the hook where the graph is built.
//
// The consequence is stated plainly rather than hidden: this file validates and compacts a row
// list. It does not compute attention over one, and no production path calls it.
} // namespace tq::reader
