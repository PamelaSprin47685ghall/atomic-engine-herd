#include <tq/reader/rows_validation.h>

#include <cmath>
#include <cstring>
#include <limits>

namespace tq::reader {

const char * describe(rows_validation_refusal refusal) {
    switch (refusal) {
        case rows_validation_refusal::ok:
            return "ok";
        case rows_validation_refusal::bad_shape:
            return "the caller's claims about the row table or mask are internally inconsistent";
        case rows_validation_refusal::row_out_of_range:
            return "a non-negative row is at or beyond the cache depth: the kernel would read past k";
        case rows_validation_refusal::padding_not_masked:
            return "a negative row's mask entry is not -inf, so the kernel's clamp to row 0 would "
                   "give that row a real weight";
        case rows_validation_refusal::buffer_too_small:
            return "the claimed table or mask reaches past the bytes the caller says it owns, so "
                   "reading it would be an out-of-bounds read inside this validator";
        case rows_validation_refusal::overflow:
            return "an index into the mask or the row table would wrap";
    }
    return "?";
}

namespace {

// True only for `-inf` itself. NaN and `+inf` are NOT padding markers: the kernel's clamp turns a
// negative row into row 0, so what decides whether that value survives is the weight, and a NaN
// entry makes every comparison in the softmax false while `+inf` makes the column dominant. An
// earlier draft accepted any non-finite value here, which would have certified both.
bool is_neg_inf(float v) {
    return std::isinf(v) && v < 0.0f;
}

std::size_t mask_element_size(bool is_f16) {
    return is_f16 ? sizeof(ggml_fp16_t) : sizeof(float);
}

// `base + n * stride`, refused rather than wrapped. Both factors are size_t, so the product can only
// be too large; the addition is checked the same way.
bool checked_offset(std::size_t n, std::size_t stride, std::size_t * out) {
    if (stride != 0 && n > std::numeric_limits<std::size_t>::max() / stride) return false;
    *out = n * stride;
    return true;
}

bool checked_add(std::size_t a, std::size_t b, std::size_t * out) {
    if (a > std::numeric_limits<std::size_t>::max() - b) return false;
    *out = a + b;
    return true;
}

rows_validation_report refuse(rows_validation_refusal r, std::string why) {
    rows_validation_report out;
    out.refusal = r;
    out.why = std::move(why);
    return out;
}

} // namespace

rows_validation_report validate_runtime_rows(const runtime_rows_inputs & in) {
    // ---------------------------------------------------------------------------------------------
    // STRUCTURE FIRST: every claim is checked before a stride is used, so a bad claim cannot make
    // this read outside the buffers it was given via the strides it was given.
    // ---------------------------------------------------------------------------------------------
    if (in.n_columns == 0 || in.n_slices == 0 || in.n_tokens_per_slice == 0) {
        return refuse(rows_validation_refusal::bad_shape,
                      "n_columns " + std::to_string(in.n_columns) + ", n_slices " +
                          std::to_string(in.n_slices) + ", n_tokens_per_slice " +
                          std::to_string(in.n_tokens_per_slice) +
                          ": a zero here means there is no table to verify, and \"verified nothing\" "
                          "must not read as \"verified\"");
    }
    if (in.rows == nullptr) {
        return refuse(rows_validation_refusal::bad_shape, "rows is null");
    }
    if (in.mask == nullptr) {
        return refuse(rows_validation_refusal::bad_shape, "mask is null");
    }
    if (in.n_cell == 0) {
        return refuse(rows_validation_refusal::bad_shape,
                      "n_cell is 0, so no non-negative row could name a cell; a cache of zero depth "
                      "is a caller error, not a table that passes");
    }
    // THE EXTENTS ARE REQUIRED, AND ZERO IS NOT "UNBOUNDED".
    if (in.rows_bytes == 0 || in.mask_bytes == 0) {
        return refuse(rows_validation_refusal::bad_shape,
                      "rows_bytes " + std::to_string(in.rows_bytes) + " and mask_bytes " +
                          std::to_string(in.mask_bytes) +
                          ": both are required. A zero here would mean \"unbounded\", and an "
                          "unbounded reading is exactly the out-of-bounds scan this field prevents");
    }

    // ---------------------------------------------------------------------------------------------
    // THE TWO ELEMENT WIDTHS ARE DIFFERENT, AND CONFLATING THEM WAS A REAL BUG.
    //
    // The row table is I32; the mask is F16 or F32. The previous revision computed ONE column offset
    // (`c * sizeof(std::int32_t)`) and reused it for the mask as well. For an F16 mask that is a
    // factor-of-two error: column 3 was read at byte 12 instead of byte 6, i.e. at column 6 of the
    // same token -- or, at the end of a token, at column 0 of the NEXT token, whose mask is `0.0f`.
    // The symptom was a clean F16 table being refused with "padding not masked", and the cause was
    // attributed to `ggml_fp32_to_fp16` losing infinities and the case deleted. That attribution was
    // wrong: the reciprocal test `tq_reader.rows_attention` round-trips `-INFINITY` through
    // `ggml_fp32_to_fp16`/`ggml_fp16_to_fp32` and asserts equality, and the converter is fine.
    // ---------------------------------------------------------------------------------------------
    const std::size_t el = mask_element_size(in.mask_is_f16);
    const std::size_t row_col = sizeof(std::int32_t);

    // THE FULL REACHABLE RANGE IS CHECKED BEFORE A SINGLE VALUE IS READ, and it is the range over
    // ALL slices and ALL tokens -- not the range actually visited. Those differ: padding columns are
    // the only mask entries this scan reads, so a table with no negatives would touch less memory
    // than it claims. Checking the whole claim is the stricter statement and it needs no assumption
    // about which columns are negative.
    //
    //   rows max = (n_slices-1)*rows_slice_stride + (n_columns-1)*row_col + row_col
    //   mask max = (n_slices-1)*mask_slice_stride + (n_tokens-1)*mask_token_stride
    //              + (n_columns-1)*el + el
    {
        std::size_t span = 0;
        std::size_t part = 0;
        if (!checked_offset(in.n_slices - 1, in.rows_slice_stride, &part) ||
            !checked_offset(in.n_columns - 1, row_col, &span) ||
            !checked_add(part, span, &span) || !checked_add(span, row_col, &span)) {
            return refuse(rows_validation_refusal::overflow, "the row table's reachable range would wrap");
        }
        if (span > in.rows_bytes) {
            return refuse(rows_validation_refusal::buffer_too_small,
                          "the row table reaches byte " + std::to_string(span) + " but rows_bytes is " +
                              std::to_string(in.rows_bytes) +
                              " (n_columns " + std::to_string(in.n_columns) + ", n_slices " +
                              std::to_string(in.n_slices) + ", rows_slice_stride " +
                              std::to_string(in.rows_slice_stride) + ")");
        }

        std::size_t mask_span = 0;
        std::size_t token_part = 0;
        std::size_t col_part = 0;
        if (!checked_offset(in.n_slices - 1, in.mask_slice_stride, &mask_span) ||
            !checked_offset(in.n_tokens_per_slice - 1, in.mask_token_stride, &token_part) ||
            !checked_offset(in.n_columns - 1, el, &col_part) ||
            !checked_add(mask_span, token_part, &mask_span) ||
            !checked_add(mask_span, col_part, &mask_span) ||
            !checked_add(mask_span, el, &mask_span)) {
            return refuse(rows_validation_refusal::overflow, "the mask's reachable range would wrap");
        }
        if (mask_span > in.mask_bytes) {
            return refuse(rows_validation_refusal::buffer_too_small,
                          "the mask reaches byte " + std::to_string(mask_span) + " but mask_bytes is " +
                              std::to_string(in.mask_bytes) + " (n_columns " +
                              std::to_string(in.n_columns) + " at " + std::to_string(el) +
                              " bytes, n_tokens " + std::to_string(in.n_tokens_per_slice) +
                              ", n_slices " + std::to_string(in.n_slices) +
                              ", strides " + std::to_string(in.mask_token_stride) + "/" +
                              std::to_string(in.mask_slice_stride) + ")");
        }
    }

    std::size_t row_bytes = 0;
    if (!checked_offset(in.n_columns, row_col, &row_bytes)) {
        return refuse(rows_validation_refusal::overflow, "column extent of rows would wrap");
    }
    if (in.n_slices > 1 && in.rows_slice_stride < row_bytes) {
        return refuse(rows_validation_refusal::bad_shape,
                      "rows_slice_stride is " + std::to_string(in.rows_slice_stride) +
                          " bytes but one slice of " + std::to_string(in.n_columns) +
                          " columns needs " + std::to_string(row_bytes) +
                          "; the slices would alias each other");
    }

    // `el` was computed ABOVE, before the reachable-range check, so that the two column widths are
    // visibly different at every point of use. It is deliberately not redeclared here: the previous
    // revision had one declaration and used it for both the I32 row table and the F16/F32 mask, which
    // is what put an F16 column at twice its correct byte offset.
    std::size_t token_span = 0;
    if (!checked_offset(in.n_columns, el, &token_span)) {
        return refuse(rows_validation_refusal::overflow, "token extent of the mask would wrap");
    }
    if (in.mask_token_stride < token_span) {
        return refuse(rows_validation_refusal::bad_shape,
                      "mask_token_stride is " + std::to_string(in.mask_token_stride) +
                          " bytes but " + std::to_string(in.n_columns) + " columns at " +
                          std::to_string(el) + " bytes each span " + std::to_string(token_span) +
                          "; consecutive tokens would alias, so a token's visibility could not be "
                          "told from its neighbour's");
    }
    if (in.n_slices > 1) {
        std::size_t slice_span = 0;
        if (!checked_offset(in.n_tokens_per_slice, in.mask_token_stride, &slice_span)) {
            return refuse(rows_validation_refusal::overflow, "slice extent of the mask would wrap");
        }
        if (in.mask_slice_stride < slice_span) {
            return refuse(rows_validation_refusal::bad_shape,
                          "mask_slice_stride is " + std::to_string(in.mask_slice_stride) +
                              " bytes but a slice of " + std::to_string(in.n_tokens_per_slice) +
                              " tokens spans " + std::to_string(slice_span) +
                              "; slices would alias");
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Then the values.
    // ---------------------------------------------------------------------------------------------
    rows_validation_report out;
    const auto * rows_base = reinterpret_cast<const unsigned char *>(in.rows);
    const auto * mask_base = reinterpret_cast<const unsigned char *>(in.mask);

    for (std::size_t s = 0; s < in.n_slices; ++s) {
        std::size_t slice_row_off = 0;
        if (!checked_offset(s, in.rows_slice_stride, &slice_row_off) ||
            !checked_add(slice_row_off, 0, &slice_row_off)) {
            return refuse(rows_validation_refusal::overflow, "row slice offset would wrap");
        }
        const auto * slice_rows = rows_base + slice_row_off;

        for (std::size_t c = 0; c < in.n_columns; ++c) {
            std::size_t col_off = 0;
            if (!checked_offset(c, sizeof(std::int32_t), &col_off)) {
                return refuse(rows_validation_refusal::overflow, "row column offset would wrap");
            }
            std::int32_t row = 0;
            std::memcpy(&row, slice_rows + col_off, sizeof row);
            ++out.columns_checked;

            if (row >= 0) {
                if (std::size_t(row) >= in.n_cell) {
                    out.refusal = rows_validation_refusal::row_out_of_range;
                    out.has_violation = true;
                    out.violation_slice = s;
                    out.violation_column = c;
                    out.violation_row = row;
                    out.why = "slice " + std::to_string(s) + " column " + std::to_string(c) +
                              " names cell " + std::to_string(row) + " but the cache holds " +
                              std::to_string(in.n_cell) +
                              "; the kernel walks k->data + row*nb[1] with no upper bound, so this "
                              "would read past the cache and return garbage";
                    return out;
                }
                if (std::size_t(row) > out.max_row) out.max_row = std::size_t(row);
                continue;
            }

            // A negative row: padding. Harmless only if EVERY token of THIS slice is invisible here.
            ++out.negative_rows;
            for (std::size_t t = 0; t < in.n_tokens_per_slice; ++t) {
                std::size_t t_off = 0;
                std::size_t s_off = 0;
                std::size_t mask_col_off = 0;
                std::size_t off = 0;
                // THE MASK COLUMN IS `c * el`, NOT `c * row_col`. `col_off` above is the ROW table's
                // column offset -- four bytes per I32 entry -- and reusing it here is what put an F16
                // mask at twice its correct byte offset, so a padding column was read from a
                // neighbouring token's column (whose bias is `0.0f`) and a clean table was refused
                // with "padding not masked". The two widths differ by design, so the two offsets are
                // computed separately and neither is derived from the other.
                if (!checked_offset(t, in.mask_token_stride, &t_off) ||
                    !checked_offset(s, in.mask_slice_stride, &s_off) ||
                    !checked_offset(c, el, &mask_col_off) ||
                    !checked_add(t_off, s_off, &off) ||
                    !checked_add(off, mask_col_off, &off)) {
                    return refuse(rows_validation_refusal::overflow,
                                  "mask offset would wrap at slice " + std::to_string(s) +
                                      " token " + std::to_string(t));
                }
                float bias = 0.0f;
                if (in.mask_is_f16) {
                    ggml_fp16_t h = 0;
                    std::memcpy(&h, mask_base + off, sizeof h);
                    // NB: widen AFTER copying the raw half, so a NaN payload is not silently
                    // normalised by the cast on some platforms.
                    bias = ggml_fp16_to_fp32(h);
                } else {
                    std::memcpy(&bias, mask_base + off, sizeof bias);
                }
                ++out.tokens_checked;
                if (!is_neg_inf(bias)) {
                    out.refusal = rows_validation_refusal::padding_not_masked;
                    out.has_violation = true;
                    out.violation_slice = s;
                    out.violation_column = c;
                    out.violation_row = row;
                    out.violation_token = t;
                    out.violation_mask = bias;
                    out.why = "slice " + std::to_string(s) + " column " + std::to_string(c) +
                              " is padding (row " + std::to_string(row) + ") but token " +
                              std::to_string(t) + " has mask " + std::to_string(bias) +
                              " there, not -inf. The kernel clamps a negative row to row 0, so that "
                              "token would mix row 0's value in with a real weight";
                    return out;
                }
            }
        }
        ++out.slices_checked;
    }

    out.refusal = rows_validation_refusal::ok;
    return out;
}

} // namespace tq::reader
