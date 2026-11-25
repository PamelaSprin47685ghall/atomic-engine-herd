// The row-list guard: the only place a negative row may be seen, and therefore the only place it
// may be handled.
//
// What this file used to test -- attention over a runtime row tensor -- is gone, and the reason is
// recorded in `rows_attention.h`. The short version, because the first draft of this comment got
// both facts wrong: `ggml_get_rows` DOES bounds-check (`GGML_ASSERT(i01 >= 0 && i01 < ne01)` in all
// four CPU row-gather kernels), so a bad index aborts instead of reading the neighbouring memory;
// and `GGML_OP_CLAMP` does have a CPU implementation, just not for I32, which is why an index
// cannot be sanitised inside a graph. Neither fact makes the graph path safe -- a legal row list
// containing padding would abort on the first `-1` -- but both change WHY it is unsafe.
//
// What remains is total over its input and refuses by name: a negative row whose mask is not -inf,
// a row at or beyond the cache, a length disagreement, and a null pointer carrying a length. Each
// assertion below would fail if the corresponding rule were dropped, which is the property that
// makes the guard worth having. Nothing here is wired into a decode; no production path calls it.
#include <tq/reader/rows_attention.h>

#include <cmath>
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

constexpr std::int32_t kCell = 16;

// Padding is dropped, order is kept, and the mask travels with its row. The cell numbers are
// deliberately non-monotonic (13 before 2): an implementation that sorted would return (2, 7, 13)
// and "look correct" in a cache that happened to be contiguous.
void keeps_order_and_drops_padding() {
    const float inf = -INFINITY;
    const std::vector<std::int32_t> rows{13, -1, 2, -1, 7};
    const std::vector<float> mask{0.25f, inf, inf, inf, -0.75f};
    const auto out = tq::reader::compact_rows(rows.data(), rows.size(), mask.data(), mask.size(), kCell);
    check(out.refusal == tq::reader::rows_guard_refusal::ok,
          std::string("compaction ok, got ") + tq::reader::describe(out.refusal));
    check(out.rows == std::vector<std::int32_t>({13, 2, 7}),
          "row order preserved (13, 2, 7); sorting would read (2, 7, 13)");
    check(out.mask == std::vector<float>({0.25f, inf, -0.75f}), "mask follows its row");
    check(out.dropped == 2, "two padding columns dropped, got " + std::to_string(out.dropped));
}

// A -1 says "no such cell", which is only true when the column is invisible. Dropping a visible
// column silently removes a term from the softmax; so the pair is refused instead of repaired.
void refuses_negative_row_with_visible_mask() {
    const std::vector<std::int32_t> rows{3, -1};
    const std::vector<float> mask{0.0f, 0.0f};
    const auto out = tq::reader::compact_rows(rows.data(), rows.size(), mask.data(), mask.size(), kCell);
    check(out.refusal == tq::reader::rows_guard_refusal::negative_row_mask_not_neg_inf,
          std::string("a -1 with a visible mask is refused, got ") +
              tq::reader::describe(out.refusal));
    check(!out.why.empty(), "the refusal carries a reason");
}

// A positive row outside the cache is the other half of the same hazard, and the one an index
// would take from the far end of the buffer.
void refuses_out_of_range() {
    const std::vector<std::int32_t> rows{3, kCell};
    const std::vector<float> mask{0.0f, 0.0f};
    const auto out = tq::reader::compact_rows(rows.data(), rows.size(), mask.data(), mask.size(), kCell);
    check(out.refusal == tq::reader::rows_guard_refusal::row_out_of_range,
          std::string("row == n_cell is outside the cache, got ") + tq::reader::describe(out.refusal));
}

// The mask upstream hands to `build_attn_mha` is F16 (measured `type 1`), so the guard must read
// that width without changing the verdict.
void f16_mask_matches_f32() {
    const std::vector<float> wide{0.5f, -INFINITY, 0.0f};
    std::vector<ggml_fp16_t> half(wide.size());
    ggml_fp32_to_fp16_row(wide.data(), half.data(), std::int64_t(wide.size()));
    const std::vector<std::int32_t> rows{9, -1, 4};
    const auto out = tq::reader::compact_rows_f16(rows.data(), rows.size(), half.data(), half.size(), kCell);
    check(out.refusal == tq::reader::rows_guard_refusal::ok,
          std::string("f16 compaction ok, got ") + tq::reader::describe(out.refusal));
    check(out.rows == std::vector<std::int32_t>({9, 4}), "f16 drops -1");
    check(std::fabs(out.mask[0] - 0.5f) < 1e-6f && out.mask[1] == 0.0f, "f16 mask widened");
}

// The F16 path must reach the same verdict as F32 on the same violation; otherwise a caller could
// escape the guard by choosing the narrower mask.
void f16_refuses_visible_padding_too() {
    const std::vector<float> wide{0.0f, 0.0f};
    std::vector<ggml_fp16_t> half(wide.size());
    ggml_fp32_to_fp16_row(wide.data(), half.data(), std::int64_t(wide.size()));
    const std::vector<std::int32_t> rows{3, -1};
    const auto out = tq::reader::compact_rows_f16(rows.data(), rows.size(), half.data(), half.size(), kCell);
    check(out.refusal == tq::reader::rows_guard_refusal::negative_row_mask_not_neg_inf,
          std::string("f16 -1 with visible mask refused, got ") + tq::reader::describe(out.refusal));
}

// Null inputs are refused, not dereferenced.
// Null inputs, in the two spellings that differ: a null WITH a length (a claim about a buffer that
// is not there) and a null with length zero (the empty list, which is legal and is covered by
// `empty_row_list_is_legal_and_not_an_error`). The first must be refused; the second must not be,
// and a guard that conflated them would reject a first decode.
void refuses_null_inputs() {
    const std::vector<std::int32_t> rows{1};
    const std::vector<float> mask{0.0f};
    const auto a = tq::reader::compact_rows(nullptr, 1, mask.data(), mask.size(), kCell);
    check(a.refusal == tq::reader::rows_guard_refusal::size_mismatch, "null rows with a length refused");
    const auto b = tq::reader::compact_rows(rows.data(), rows.size(), nullptr, 1, kCell);
    check(b.refusal == tq::reader::rows_guard_refusal::size_mismatch, "null mask with a length refused");
}

// Two lengths that disagree mean one of them is wrong about a buffer. Before the signature carried
// both, the guard looped to the single `n` the caller passed -- so a caller who mixed up a tensor
// length and a cache count could make the GUARD itself read past the end of the shorter buffer
// while appearing to validate. The guard cannot catch that by inspecting pointers; it can only
// refuse the disagreement.
void refuses_disagreeing_lengths() {
    const std::vector<std::int32_t> rows{3, 4, 5};
    const std::vector<float> mask{0.0f, 0.0f};
    const auto a = tq::reader::compact_rows(rows.data(), rows.size(), mask.data(), mask.size(), kCell);
    check(a.refusal == tq::reader::rows_guard_refusal::size_mismatch,
          std::string("3 rows against 2 mask columns is refused, got ") +
              tq::reader::describe(a.refusal));
    check(!a.why.empty(), "the length refusal says which side was short");

    // The other direction, because a check written as `rows.size() <= mask.size()` would pass the
    // first case and silently read the mask's end here.
    const std::vector<float> long_mask{0.0f, 0.0f, 0.0f};
    const std::vector<std::int32_t> short_rows{3, 4};
    const auto b = tq::reader::compact_rows(short_rows.data(), short_rows.size(), long_mask.data(),
                                            long_mask.size(), kCell);
    check(b.refusal == tq::reader::rows_guard_refusal::size_mismatch,
          "2 rows against 3 mask columns is refused too");
}

// A first decode has nothing behind it. Empty is legal and must not be reported as an error, and
// must not be confused with the null-with-length case below: both look like "no data" if the guard
// only checks pointers.
void empty_row_list_is_legal_and_not_an_error() {
    const auto a = tq::reader::compact_rows(nullptr, 0, nullptr, 0, kCell);
    check(a.refusal == tq::reader::rows_guard_refusal::ok,
          std::string("0 rows, no pointers, is legal, got ") + tq::reader::describe(a.refusal));
    check(a.rows.empty() && a.mask.empty() && a.dropped == 0, "and produces nothing");

    // Non-null, zero length: the same verdict. A caller whose vectors are empty passes .data(),
    // which is not necessarily null, and both spellings must work.
    const std::vector<std::int32_t> none;
    const std::vector<float> none_mask;
    const auto b = tq::reader::compact_rows(none.data(), 0, none_mask.data(), 0, kCell);
    check(b.refusal == tq::reader::rows_guard_refusal::ok, "empty non-null is legal as well");
}

// A null WITH a length is a length claiming a buffer that is not there -- the opposite of empty,
// and the case that would be an out-of-bounds read inside the guard itself.
void refuses_null_with_length() {
    const std::vector<float> mask{0.0f};
    const auto a = tq::reader::compact_rows(nullptr, 1, mask.data(), mask.size(), kCell);
    check(a.refusal == tq::reader::rows_guard_refusal::size_mismatch,
          std::string("null rows with length 1 is refused, got ") + tq::reader::describe(a.refusal));
    const std::vector<std::int32_t> rows{3};
    const auto b = tq::reader::compact_rows(rows.data(), rows.size(), nullptr, 1, kCell);
    check(b.refusal == tq::reader::rows_guard_refusal::size_mismatch,
          "null mask with length 1 is refused");
    // And null with length 0 on one side, length 1 on the other, is the length disagreement -- not
    // a special null case, because neither side is allowed to be trusted.
    const auto c = tq::reader::compact_rows(nullptr, 0, mask.data(), 1, kCell);
    check(c.refusal == tq::reader::rows_guard_refusal::size_mismatch,
          "0 rows against 1 mask column is a disagreement, whatever the pointers are");
}

// The boundary is `n_cell - 1` in, `n_cell` out. Both sides are asserted because an off-by-one in
// either direction is a live read of a neighbouring cell on one side and a spurious refusal on the
// other.
void accepts_last_cell_and_refuses_one_past() {
    const std::vector<std::int32_t> rows{kCell - 1};
    const std::vector<float> mask{0.0f};
    const auto a = tq::reader::compact_rows(rows.data(), rows.size(), mask.data(), mask.size(), kCell);
    check(a.refusal == tq::reader::rows_guard_refusal::ok,
          std::string("n_cell - 1 is inside the cache, got ") + tq::reader::describe(a.refusal));
    check(a.rows == std::vector<std::int32_t>({kCell - 1}), "and survives unchanged");

    const std::vector<std::int32_t> past{kCell};
    const auto b = tq::reader::compact_rows(past.data(), past.size(), mask.data(), mask.size(), kCell);
    check(b.refusal == tq::reader::rows_guard_refusal::row_out_of_range,
          "n_cell is one past the end and is refused");
}

// Any negative row is padding, not just -1: ggml.h says "negative rows are padding", without
// naming a sentinel, and a caller whose allocator uses a different one is within the contract. The
// price of dropping it is that its column must be invisible.
void any_negative_row_is_padding() {
    const float inf = -INFINITY;
    const std::vector<std::int32_t> rows{-7, 4};
    const std::vector<float> mask{inf, 0.0f};
    const auto a = tq::reader::compact_rows(rows.data(), rows.size(), mask.data(), mask.size(), kCell);
    check(a.refusal == tq::reader::rows_guard_refusal::ok,
          std::string("-7 with -inf is padding, got ") + tq::reader::describe(a.refusal));
    check(a.rows == std::vector<std::int32_t>({4}) && a.dropped == 1, "and it is dropped");

    const std::vector<float> visible{inf, 0.0f};
    const std::vector<std::int32_t> bad{-7, 4};
    const std::vector<float> bad_mask{0.0f, 0.0f};
    const auto b = tq::reader::compact_rows(bad.data(), bad.size(), bad_mask.data(), bad_mask.size(), kCell);
    check(b.refusal == tq::reader::rows_guard_refusal::negative_row_mask_not_neg_inf,
          "-7 with a visible mask is a contract violation, whatever the sentinel is");
    (void) visible;
}

}  // namespace

int main() {
    keeps_order_and_drops_padding();
    refuses_negative_row_with_visible_mask();
    refuses_out_of_range();
    f16_mask_matches_f32();
    f16_refuses_visible_padding_too();
    refuses_null_inputs();
    refuses_disagreeing_lengths();
    empty_row_list_is_legal_and_not_an_error();
    refuses_null_with_length();
    accepts_last_cell_and_refuses_one_past();
    any_negative_row_is_padding();

    if (failures == 0) {
        std::fprintf(stderr, "rows guard: OK\n");
        return 0;
    }
    std::fprintf(stderr, "rows guard: %d failure(s)\n", failures);
    return 1;
}
