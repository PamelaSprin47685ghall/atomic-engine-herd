#include <tq/reader/rows_attention.h>

#include <cmath>
#include <cstring>
#include <limits>

namespace tq::reader {

const char * describe(rows_guard_refusal refusal) {
    switch (refusal) {
        case rows_guard_refusal::ok:                             return "ok";
        case rows_guard_refusal::size_mismatch:                  return "rows and mask lengths differ";
        case rows_guard_refusal::negative_row_mask_not_neg_inf:  return "a -1 row has a mask column that is not -inf";
        case rows_guard_refusal::row_out_of_range:               return "a row is outside [0, n_cell)";
    }
    return "?";
}

namespace {

// One implementation, two mask types. Shared because the rule is about ROWS, not about width.
//
// Both lengths are checked here and nowhere else, which is the point of taking them separately:
// the guard's whole job is to stop trusting the caller, and a single shared `n` would have made
// the guard's own loop bound the one number nobody verified.
template <typename MaskT>
compacted_rows compact_impl(const std::int32_t * rows, std::size_t n_rows, const MaskT * mask,
                            std::size_t n_mask, std::int32_t n_cell, float (*widen)(MaskT)) {
    compacted_rows out;
    if (n_rows != n_mask) {
        out.refusal = rows_guard_refusal::size_mismatch;
        out.why = "rows holds " + std::to_string(n_rows) + " entries, mask holds " +
                  std::to_string(n_mask) + "; one of them would be read past its end";
        return out;
    }
    // A null with length zero is the empty row list, which is legal: a first decode has no visible
    // history. A null with a positive length is a length claiming a buffer that is not there, and
    // reading from it would be the guard's own out-of-bounds access -- so length is what decides,
    // not the pointer, and the check is per buffer rather than joint.
    if ((rows == nullptr && n_rows != 0) || (mask == nullptr && n_mask != 0)) {
        out.refusal = rows_guard_refusal::size_mismatch;
        out.why = std::string("null pointer with non-zero length: ") +
                  (rows == nullptr ? "rows " : "mask ") + std::to_string(rows == nullptr ? n_rows : n_mask);
        return out;
    }
    out.rows.reserve(n_rows);
    out.mask.reserve(n_rows);
    for (std::size_t i = 0; i < n_rows; ++i) {
        const float bias = widen(mask[i]);
        // Any negative row is padding, exactly as ggml.h says ("negative rows are padding"). It is
        // not narrowed to -1: a caller whose allocator uses -7 is not doing anything the contract
        // forbids. What is required is that the column is invisible, because the drop below is only
        // value-preserving when its softmax weight is zero.
        if (rows[i] < 0) {
            if (!(std::isinf(bias) && bias < 0.0f)) {
                out.refusal = rows_guard_refusal::negative_row_mask_not_neg_inf;
                out.why = "row " + std::to_string(i) + " is " + std::to_string(rows[i]) +
                          " with mask " + std::to_string(bias) + "; padding must carry -inf";
                return out;
            }
            ++out.dropped;
            continue;
        }
        if (rows[i] >= n_cell) {
            out.refusal = rows_guard_refusal::row_out_of_range;
            out.why = "row " + std::to_string(i) + " is " + std::to_string(rows[i]) +
                      " but the cache holds " + std::to_string(n_cell);
            return out;
        }
        out.rows.push_back(rows[i]);
        out.mask.push_back(bias);
    }
    return out;
}

float widen_f32(float x) { return x; }
float widen_f16(ggml_fp16_t x) { return ggml_fp16_to_fp32(x); }

} // namespace

compacted_rows compact_rows(const std::int32_t * rows, std::size_t n_rows,
                            const float * mask, std::size_t n_mask, std::int32_t n_cell) {
    return compact_impl<float>(rows, n_rows, mask, n_mask, n_cell, widen_f32);
}

compacted_rows compact_rows_f16(const std::int32_t * rows, std::size_t n_rows,
                                const ggml_fp16_t * mask, std::size_t n_mask, std::int32_t n_cell) {
    return compact_impl<ggml_fp16_t>(rows, n_rows, mask, n_mask, n_cell, widen_f16);
}


} // namespace tq::reader
