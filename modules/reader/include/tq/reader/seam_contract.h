// tq/reader/seam_contract.h — the checks a seam can make about a tensor's DESCRIPTION, with no
// memory touched.
//
// WHY THIS IS A SEPARATE, ggml-FREE FUNCTION
// ------------------------------------------
// The row-table seam in `patches/herd_reader_attn.cpp` used to interleave two kinds of check:
//
//   * facts about the description  — dtype, `nb[0]`, rank, each `ne`, whether the buffer is host-
//     resident, whether `data` lies inside the buffer, how many bytes remain from `data`
//   * one fact about the values    — every non-negative row inside the cache, every negative row
//     masked `-inf` (that is `validate_runtime_rows`, and it is a separate function already)
//
// The description checks were written inline, each refusing by calling `rows_validate_stop`, which
// ABORTS. A function that aborts cannot be tested: every case would be a process death, and a
// process death is observable only as an exit code — not as "the rank check fired before the stride
// check", and not as "this case was decided before any byte was read". So the checks lived in a
// shape where the only covered path was the accepting one, for four revisions in a row.
//
// This file fixes the shape rather than the tests. It reads NOTHING: every input is a number or a
// pointer value taken from a tensor, and the function compares numbers. That is what makes "the
// decision costs no memory access" a property of the code instead of a promise in a comment — and it
// is what makes the order of the checks observable, because the report names the FIRST refusal.
//
// THE TRUST BOUNDARY, STATED ONCE
// -------------------------------
// `buffer_size` and `buffer_base` are the caller's statement about the allocation. This function
// verifies that the description is internally consistent with them — `data` inside the allocation,
// and enough bytes remaining — but it cannot verify that the allocation is as large as claimed. What
// it does NOT do, and never implies: that the tensor's own logical extent is the remaining buffer
// size. One backend buffer routinely holds several tensors, so remaining bytes is an UPPER BOUND on
// what is readable from `data`, not the length of this tensor. Writing the bound as the length is a
// mistake this header's field name (`readable_bytes`) is chosen to make hard.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace tq::reader {

// A ggml tensor's description, with the ggml types themselves removed so this stays a pure function
// over numbers. `type` holds `ggml_type` as an int; the two values that matter are compared by name
// in the caller, which passes `type_is_f16` / `type_is_i32` so this file needs no ggml header.
struct tensor_face {
    std::int64_t ne[4] = {0, 0, 0, 0};
    std::size_t nb[4] = {0, 0, 0, 0};

    bool type_is_i32 = false;   // rows: the operator asserts I32
    bool type_is_f16 = false;   // mask: the CPU kernel asserts F16; F32 is accepted for a host check
    bool type_is_float = false; // mask: F16 or F32, i.e. a width the padding check can read

    const void * data = nullptr;          // the tensor's own data pointer, after placement
    const void * buffer_base = nullptr;   // the backend buffer's base
    std::size_t buffer_size = 0;          // the backend buffer's size
    bool buffer_present = false;
    bool buffer_is_host = false;
};

enum class seam_contract_refusal {
    ok,
    rows_not_i32,            // wrong dtype, or nb[0] is not one I32
    rows_not_host,           // no buffer, or the buffer is not host-resident
    rows_no_data,            // placed but unwritten: the original did not fill it
    rows_buffer_geometry,    // the buffer reports no base or no size
    rows_data_outside,       // data does not lie inside its own buffer
    mask_absent,             // the seam has a row table but no mask tensor
    mask_type,               // neither F16 nor F32
    mask_not_host,           // no buffer, not host-resident, or no data
    mask_buffer_geometry,    // the buffer reports no base or no size
    mask_data_outside,       // data does not lie inside its own buffer
    columns_differ,          // mask ne[0] != rows ne[0]
    dim_not_positive,        // a modelled dimension is <= 0 (zero underflows `n - 1`; negative is
                             // destroyed by a cast to size_t, so it is checked as int64 first)
    rank_unmodelled,         // rows ne[2]/ne[3] or mask ne[2] is not 1
    slices_differ,           // mask ne[3] != rows ne[1]
    mask_not_contiguous,     // mask nb[0] is wider than one element
    overflow,                // the extent arithmetic wraps
};

inline const char * describe(seam_contract_refusal refusal) {
    switch (refusal) {
        case seam_contract_refusal::ok:                    return "ok";
        case seam_contract_refusal::rows_not_i32:          return "rows is not a contiguous I32 row list";
        case seam_contract_refusal::rows_not_host:         return "rows is not host-resident";
        case seam_contract_refusal::rows_no_data:          return "rows has no data after the fill";
        case seam_contract_refusal::rows_buffer_geometry:  return "rows' buffer reports no base or size";
        case seam_contract_refusal::rows_data_outside:     return "rows' data lies outside its buffer";
        case seam_contract_refusal::mask_absent:           return "no mask tensor at all";
        case seam_contract_refusal::mask_type:             return "the mask is neither F16 nor F32";
        case seam_contract_refusal::mask_not_host:         return "the mask is not host-resident or has no data";
        case seam_contract_refusal::mask_buffer_geometry:  return "the mask's buffer reports no base or size";
        case seam_contract_refusal::mask_data_outside:     return "the mask's data lies outside its buffer";
        case seam_contract_refusal::columns_differ:        return "mask ne[0] differs from rows ne[0]";
        case seam_contract_refusal::dim_not_positive:      return "a modelled dimension is not positive";
        case seam_contract_refusal::rank_unmodelled:       return "an axis is not the 1 this validation models";
        case seam_contract_refusal::slices_differ:         return "mask ne[3] differs from rows ne[1]";
        case seam_contract_refusal::mask_not_contiguous:   return "the mask is not contiguous along its column axis";
        case seam_contract_refusal::overflow:              return "the extent arithmetic would wrap";
    }
    return "?";
}

struct seam_contract_report {
    seam_contract_refusal refusal = seam_contract_refusal::ok;
    std::string why;
    const char * field = "";   // which input the refusal is about, for a message with no prose search

    // Filled only on `ok`. The two extents are UPPER BOUNDS on readable bytes from each `data`
    // pointer (see the header note); they are not tensor lengths.
    std::size_t n_columns = 0;
    std::size_t n_slices = 0;
    std::size_t n_tokens = 0;
    std::size_t rows_readable_bytes = 0;
    std::size_t mask_readable_bytes = 0;
    bool mask_is_f16 = false;
};

namespace detail {

inline bool add_ok(std::size_t a, std::size_t b, std::size_t * out) {
    if (a > static_cast<std::size_t>(-1) - b) return false;
    *out = a + b;
    return true;
}

// Bytes readable from `f.data` to the end of `f`'s own allocation. Refuses rather than returning a
// number when the geometry is unusable; the caller turns that into its own named refusal.
inline bool readable_bytes(const tensor_face & f, std::size_t * out, seam_contract_refusal * why) {
    if (!f.buffer_present) {
        *why = seam_contract_refusal::rows_not_host;
        return false;
    }
    if (!f.buffer_is_host) {
        *why = seam_contract_refusal::rows_not_host;
        return false;
    }
    if (f.buffer_base == nullptr || f.buffer_size == 0) {
        *why = seam_contract_refusal::rows_buffer_geometry;
        return false;
    }
    if (f.data == nullptr) {
        *why = seam_contract_refusal::rows_no_data;
        return false;
    }
    // uintptr_t, not pointer comparison: relational comparison and subtraction are only defined for
    // pointers into the same array, and a tensor's `data` and its buffer's base are not provably
    // that. Integers have no such rule, so the compiler cannot assume the comparison away.
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(f.buffer_base);
    const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(f.data);
    if (p < base) {
        *why = seam_contract_refusal::rows_data_outside;
        return false;
    }
    const std::uintptr_t off = p - base;
    if (off >= f.buffer_size) {
        *why = seam_contract_refusal::rows_data_outside;
        return false;
    }
    *out = f.buffer_size - static_cast<std::size_t>(off);
    return true;
}

inline seam_contract_report refuse(seam_contract_refusal r, const char * field, std::string why) {
    seam_contract_report out;
    out.refusal = r;
    out.field = field;
    out.why = std::move(why);
    return out;
}

} // namespace detail

// Checks the description of the row table and the mask against each other and against their own
// allocations. Reads no tensor memory: every input is a number or a pointer value.
//
// The order is the order a caller wants to hear about problems: type and residency first (a
// description that cannot be read at all), then buffer geometry, then the modelled axes, then the
// two extents. A refusal is the FIRST problem, not a list, because the caller's next action depends
// on the first one.
inline seam_contract_report validate_seam_contract(const tensor_face & rows, const tensor_face & mask) {
    using detail::refuse;

    // ---- rows: dtype and residency -------------------------------------------------------------
    if (!rows.type_is_i32 || rows.nb[0] != sizeof(std::int32_t)) {
        return refuse(seam_contract_refusal::rows_not_i32, "rows",
                      "rows must be I32 with nb[0] == 4; a different width would reinterpret two "
                      "rows as one");
    }

    std::size_t rows_readable = 0;
    {
        seam_contract_refusal why = seam_contract_refusal::ok;
        if (!detail::readable_bytes(rows, &rows_readable, &why)) {
            // The helper names the ROWS variants; anything else it could return is a mask variant
            // it cannot produce, so this mapping is total.
            return refuse(why, "rows",
                          "rows cannot be read from the host: no buffer, a non-host buffer, an "
                          "unusable allocation, or data outside it");
        }
    }

    // ---- mask: dtype and residency -------------------------------------------------------------
    if (!mask.type_is_float) {
        return refuse(seam_contract_refusal::mask_type, "mask",
                      "the mask must be F16 (what the CPU kernel asserts) or F32 (what a host-side "
                      "check can read); any other width is not a bias");
    }
    std::size_t mask_readable = 0;
    {
        seam_contract_refusal why = seam_contract_refusal::ok;
        if (!detail::readable_bytes(mask, &mask_readable, &why)) {
            switch (why) {
                case seam_contract_refusal::rows_buffer_geometry:
                    return refuse(seam_contract_refusal::mask_buffer_geometry, "mask",
                                  "the mask's backend buffer reports no base or no size");
                case seam_contract_refusal::rows_data_outside:
                    return refuse(seam_contract_refusal::mask_data_outside, "mask",
                                  "the mask's data pointer does not lie inside its own backend "
                                  "buffer, so no extent derived from it can be trusted");
                default:
                    return refuse(seam_contract_refusal::mask_not_host, "mask",
                                  "the mask is not host-resident or has no data");
            }
        }
    }

    // ---- modelled dimensions, as SIGNED int64, BEFORE the cross-tensor comparison --------------
    //
    // Converting to `size_t` first and then testing `== 0` is the wrong order for a signed type: a
    // negative value becomes about 2^64, which passes every later comparison, and the failure then
    // surfaces as a gigantic count rather than an obviously broken description.
    //
    // And this runs BEFORE `columns_differ`, which the test caught: with `rows.ne[0] == 0` the
    // cross-tensor check fired first and reported "mask ne[0] is 6 but rows ne[0] is 0", sending a
    // reader to the mask -- which is correct. "One of the descriptions is not a shape at all" is
    // more basic than "the two shapes disagree", so the sign check comes first.
    if (rows.ne[0] <= 0 || rows.ne[1] <= 0 || mask.ne[0] <= 0 || mask.ne[1] <= 0) {
        return refuse(seam_contract_refusal::dim_not_positive, "ne",
                      "rows ne[0]/ne[1] and mask ne[0]/ne[1] must be positive: zero makes "
                      "`(n - 1)` underflow, and a negative value would become a huge count after "
                      "a cast");
    }
    const std::size_t n_columns = static_cast<std::size_t>(rows.ne[0]);
    const std::size_t n_slices = static_cast<std::size_t>(rows.ne[1]);
    const std::size_t n_tokens = static_cast<std::size_t>(mask.ne[1]);

    // ---- the shared column axis ----------------------------------------------------------------
    if (mask.ne[0] != rows.ne[0]) {
        return refuse(seam_contract_refusal::columns_differ, "mask.ne[0]",
                      "mask ne[0] is " + std::to_string(mask.ne[0]) + " but rows ne[0] is " +
                          std::to_string(rows.ne[0]) +
                          "; the operator asserts they are equal, so every column-to-row pairing "
                          "after a mismatch is arbitrary");
    }

    if (rows.ne[2] != 1 || rows.ne[3] != 1 || mask.ne[2] != 1) {
        return refuse(seam_contract_refusal::rank_unmodelled, "ne",
                      "rows must be [n_columns, n_slices] and the mask "
                      "[n_columns, n_tokens, 1, n_slices]; another axis would be folded into the "
                      "slice walk without anything noticing");
    }
    if (mask.ne[3] != rows.ne[1]) {
        return refuse(seam_contract_refusal::slices_differ, "mask.ne[3]",
                      "the mask carries " + std::to_string(mask.ne[3]) + " slice(s) but the row "
                      "table has " + std::to_string(rows.ne[1]) +
                          "; one mask slice is walked per table slice, so a shorter mask would be "
                          "read past its own last slice");
    }

    // ---- the mask's column stride --------------------------------------------------------------
    const std::size_t mask_el = mask.type_is_f16 ? sizeof(std::uint16_t) : sizeof(float);
    if (mask.nb[0] != mask_el) {
        return refuse(seam_contract_refusal::mask_not_contiguous, "mask.nb[0]",
                      "mask nb[0] is " + std::to_string(mask.nb[0]) + " but one element is " +
                          std::to_string(mask_el) +
                          "; a column would be read from the gap between entries and the extent "
                          "check would never see it");
    }

    // The stride lower bounds are NOT re-derived here. `validate_runtime_rows` compares
    // `rows_slice_stride` and `mask_token_stride` against `checked_offset` products and refuses an
    // aliasing stride with `bad_shape`; a second, bare `size_t` multiply here would wrap and could
    // certify exactly the overlap it exists to reject.

    seam_contract_report out;
    out.n_columns = n_columns;
    out.n_slices = n_slices;
    out.n_tokens = n_tokens;
    out.rows_readable_bytes = rows_readable;
    out.mask_readable_bytes = mask_readable;
    out.mask_is_f16 = mask.type_is_f16;
    return out;
}

} // namespace tq::reader
