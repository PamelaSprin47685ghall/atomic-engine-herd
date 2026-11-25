// tq/reader/rows_validation.h — the host-side verification of upstream's runtime row table, for the
// one moment its values exist.
//
// THE PROBLEM, AND WHY VERIFY RATHER THAN COMPACT
// -----------------------------------------------
// `build_attn_mha` receives `kv_rows`: an I32 INPUT tensor. ggml.h states its contract in one line —
// "mask column i of slice i3 uses row kv_rows[i, i3] of k and v; negative rows are padding, their
// mask entries must be -inf".
//
// The consumer is `ggml_flash_attn_ext_rows`, and the CPU kernel resolves a row as
// `std::max<int32_t>(rows[i], 0)` before walking `k->data + row*nb[1]`:
//
//   * a NEGATIVE row reads ROW 0. That is harmless only because its mask entry is `-inf`, which
//     makes the softmax weight exactly zero. Leave that column visible and the answer is row 0's
//     value with a real weight — plausible, wrong, and silent;
//   * a POSITIVE row at or beyond `k->ne[1]` is an out-of-bounds read. No assert, no error, garbage
//     that looks like a number.
//
// `rows_attention.h`'s `compact_rows` takes the other route: it DELETES the padding columns and hands
// back a row list with no negatives. That is correct for a consumer that cannot clamp, and it is
// value-preserving only under the same `-inf` precondition. It cannot be used here, because this
// table is a GRAPH INPUT: its values arrive at compute time (measured `input 1 data no`), so a host
// cannot rewrite what the kernel will read. What the host CAN do is verify the invariant that makes
// the kernel's clamping harmless, and refuse when it does not hold.
//
// WHY EVERY TOKEN, EVERY SLICE
// ----------------------------
// A prefill mask is `[n_cell, n_tokens, 1, n_slices]`, and the padding price is per COLUMN OF THE
// MASK, not per row: `-inf` at token 0 does nothing for token 2. A check that looked at one mask
// column — which is what the previous attempt in this tree effectively did — would certify a table
// whose padding is visible to some tokens. So the scan is over (slice, token) for every negative
// column, and a violation names the column AND the token that can see it.
//
// WHAT THIS CANNOT CHECK, STATED BECAUSE THE LAST ATTEMPT DIED OF PRETENDING OTHERWISE
// -----------------------------------------------------------------------------------
// It cannot check the buffers' real extent. `n_columns`, `n_tokens_per_slice`, `n_slices` and the
// three byte strides are the caller's claims; a claim larger than the buffer is an out-of-bounds
// read inside this function. What it CAN check is that the claims are internally consistent — a
// stride smaller than a packed layout would make columns alias, and is refused — and that every
// value it does read satisfies the contract.
//
// It also cannot make the future safe. The values it reads are the ones present NOW; whoever fills
// `kv_rows` on the next decode writes new ones, unverified. That is why this is a check and not a
// guarantee: a host that verifies once and decodes twice has verified once.
//
// WHAT THIS IS NOT
// ----------------
// Not an emitter, not a seam, and not wired anywhere. It allocates nothing, mutates nothing, and
// touches no ggml type except the F16 width.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <ggml.h>

namespace tq::reader {

enum class rows_validation_refusal {
    ok,
    bad_shape,          // a null base with a non-zero count, a zero count, or a stride that would
                        // make columns alias
    row_out_of_range,   // a non-negative row at or beyond the cache depth
    padding_not_masked, // a negative row whose mask entry is not -inf for some (slice, token)
    buffer_too_small,   // the claimed shape reaches past the bytes the caller says it owns
    overflow,           // the byte arithmetic for an index would wrap
};

const char * describe(rows_validation_refusal refusal);

// The caller's claims about the two tensors, in the shape ggml hands them over.
//
// Byte strides rather than element counts, because that is what a tensor carries (`nb[1]`, `nb[3]`)
// and a conversion at the call site is a place to get the element width wrong.
struct runtime_rows_inputs {
    const std::int32_t * rows = nullptr;   // [n_columns, n_slices] I32

    // `kv_rows->ne[0]`, which the operator asserts equals `mask->ne[0]` and which the cache depth
    // must cover: a non-negative row is an index into k/v, so it must be < n_cell.
    std::size_t n_columns = 0;

    // `kv_rows->ne[1]`, which the operator asserts equals `q->ne[3]`.
    std::size_t n_slices = 0;

    // `kv_rows->nb[1]`: BYTES between slice s and s+1. Unused when n_slices == 1.
    std::size_t rows_slice_stride = 0;

    const void * mask = nullptr;           // [n_columns, n_tokens_per_slice, 1, n_slices]

    std::size_t n_tokens_per_slice = 0;    // `mask->ne[1]`
    std::size_t mask_token_stride = 0;     // `mask->nb[1]`: BYTES between token t and t+1
    std::size_t mask_slice_stride = 0;     // `mask->nb[3]`: BYTES between slice s and s+1

    // The operator asserts F16 on the CPU (`GGML_ASSERT(mask->type == GGML_TYPE_F16)`), so F16 is the
    // width that will actually be read. F32 is accepted here because the SAME validation is useful
    // before a cast, and because a host that stored an F32 mask should learn that from this check
    // rather than from an abort inside ggml.
    bool mask_is_f16 = false;

    // The cache depth: `k->ne[1]`. A non-negative row must be strictly below it.
    std::size_t n_cell = 0;

    // THE REAL EXTENTS OF THE TWO BUFFERS, IN BYTES, AND THEY ARE REQUIRED.
    //
    // Every other number above describes memory the CALLER owns: `n_columns`, `n_tokens_per_slice`,
    // `n_slices` and the three strides are a claim about layout, not about extent. Without the
    // extents, a claim larger than the buffer makes this function itself the out-of-bounds reader --
    // it would walk past the end while reporting on safety. The previous revision had no such fields
    // and instead said so in a comment, which is a paragraph, not a bound.
    //
    // So a caller passes the number of readable bytes behind each pointer (for a ggml tensor that is
    // `size`, not `nbytes(ne)` when the tensor is a view of a larger buffer -- the claim has to be
    // the memory that is really there). Zero is REFUSED rather than treated as "unbounded": an
    // unbounded reading is the failure this field exists to prevent.
    std::size_t rows_bytes = 0;
    std::size_t mask_bytes = 0;
};

// Where the first violation is, in the caller's own coordinates. `has_violation` is false for
// `bad_shape` and `overflow`, because those are decided before any value is read.
struct rows_validation_report {
    rows_validation_refusal refusal = rows_validation_refusal::ok;
    std::string why;

    // Work actually done, so a caller can tell a clean table from a table it never scanned.
    std::size_t columns_checked = 0;
    std::size_t slices_checked = 0;
    std::size_t tokens_checked = 0;    // columns_checked * slices_checked * tokens_per_slice
    std::size_t negative_rows = 0;     // padding columns seen (all of which were verified masked)
    std::size_t max_row = 0;           // largest non-negative row seen; 0 when there were none

    bool has_violation = false;
    std::size_t violation_slice = 0;
    std::size_t violation_column = 0;
    std::int32_t violation_row = 0;
    std::size_t violation_token = 0;   // set for padding_not_masked, which is per (slice, token)
    float violation_mask = 0.0f;
};

// Verifies that every non-negative row names a cell inside the cache and that every negative row is
// masked `-inf` for EVERY token of EVERY slice.
//
// What it can and cannot check, and the three lines below are the whole distinction.
//
//   CHECKED   every structural claim the caller makes -- zero or null fields, strides too small to
//             keep entries or slices distinct, the two buffers' DECLARED extents against the shape
//             those claims describe, overflow in any offset arithmetic, and then every value: each
//             non-negative row inside the cache, each negative row masked `-inf` for every token of
//             every slice.
//
//   NOT CHECKED   whether `rows_bytes` and `mask_bytes` are true. They are the caller's statement
//             about memory it owns; a caller that overstates them makes this function the
//             out-of-bounds reader it exists to prevent. The fields are REQUIRED and zero is refused
//             (`0` would read as "unbounded"), which makes the honest version of this limit
//             expressible -- but a lie about the extent is still a lie this code cannot detect.
//
//   NOT CHECKED   anything about later decodes. The values verified are the ones present NOW.
//
// The previous revision of this comment said the extents could not be checked at all, which stopped
// being true when `rows_bytes`/`mask_bytes` were added; leaving that sentence in place would tell a
// reviewer the check is weaker than it is, which is the same class of error as claiming it is
// stronger.
rows_validation_report validate_runtime_rows(const runtime_rows_inputs & in);

} // namespace tq::reader
