// tq_reader.seq_cell_projection — the transcription of `llama_kv_cells::get_seq_cells`, and the two
// things that make a row table safe to hand to a graph: the identity round-trip through the ledger,
// and the `-1` padding refused by `validate_runtime_rows` when a token can still see it.
//
// Every case here fails for a PLAUSIBLE implementation, not a broken one:
//   A1a  a `p_min` filter that is off by one, or a tail that is zero-filled instead of `-1`;
//   A1b  a sorted result. The multi-sequence case is deliberately non-monotonic ({5,1,3}), because
//        sorting it is the most natural "improvement" and it changes which cell a column names;
//   A1c  a silently truncated table instead of `overflow`;
//   A2   treating the physical row as the durable identity — the ledger's whole point is that it is
//        not, so the round-trip must go cell -> row -> cell;
//   A3   verifying only one mask column (the failure `rows_validation.h` was written to name).

#include <tq/reader/index_ledger.h>
#include <tq/reader/rows_validation.h>
#include <tq/reader/seq_cell_projection.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

#define CHECK_EQ(a, b)                                                    \
    do {                                                                  \
        const auto lhs_ = (a);                                            \
        const auto rhs_ = (b);                                            \
        if (!(lhs_ == rhs_)) {                                            \
            std::fprintf(stderr, "FAIL %s:%d  %s == %s  (%lld vs %lld)\n", \
                         __FILE__, __LINE__, #a, #b,                      \
                         static_cast<long long>(lhs_), static_cast<long long>(rhs_)); \
            ++failures;                                                   \
        }                                                                 \
    } while (0)

namespace {

constexpr std::int32_t kPad = -1;

// A1a: one sequence, bits 0..5 set, `p_min` drops the first two. Expect the survivors in cell order
// and EVERY remaining slot `-1`, because a zero-fill would make column 4 a real read of cell 0.
void case_single_sequence_with_window() {
    const std::int32_t   pos[]   = {0, 1, 2, 3, 4, 5, 6, 7};
    const std::uint64_t  bits[]  = {0x3Fu};   // cells 0..5
    const std::int32_t   seqs[]  = {0};

    tq::reader::seq_cell_projection_inputs in;
    in.pos = pos; in.seq_bits = bits; in.seq_ids = seqs;
    in.n_seq = 1; in.n_words = 1; in.n_used = 8; in.n_per_seq = 6; in.p_min = 2;

    const auto out = tq::reader::project_seq_cells(in);

    CHECK_EQ(out.rows.size(), std::size_t{6});
    CHECK_EQ(out.n_real, std::size_t{4});
    CHECK(!out.overflow);
    for (std::size_t i = 0; i < out.rows.size(); ++i) {
        const std::int32_t want = i < 4 ? static_cast<std::int32_t>(i + 2) : kPad;
        CHECK_EQ(out.rows[i], want);
    }
}

// A1b: two sequences in one slice. The order is UPSTREAM'S, and upstream is word-major: the member
// sequences' bitsets are OR-ed into one word per 64 cells, and that word's set bits are emitted
// ascending. So the table is {1, 3, 5} -- sorted by CELL, not grouped by which sequence owns it.
//
// This case used to assert {5, 1, 3}, described as "the slice's member order, then bits ascending
// within each". That sentence was the defect, not the specification: it was written from the
// projection's own loop (outer sequence, inner word) and then used as the evidence that the loop was
// right. `tq_reader.seq_cell_projection_ab` runs the real `llama_kv_cells::get_seq_cells` and
// disagreed on 7 of 8 entries; the numbers below are now taken from upstream, and they are also what
// the OR semantics force -- seq 0's cell 5 and seq 1's cells 1 and 3 land in ONE word.
//
// The shared-cell consequence is the part a fixture cannot show and the A/B test can: a cell owned by
// two members appears ONCE, because the OR erases the duplication. Sequence-major emitted it twice.
void case_two_sequences_keep_scan_order() {
    const std::int32_t   pos[]   = {0, 1, 2, 3, 4, 5, 6, 7};
    // seq 0 owns cell 5 only; seq 1 owns cells 1 and 3.
    const std::uint64_t  bits[]  = {1u << 5, (1u << 1) | (1u << 3)};
    const std::int32_t   seqs[]  = {0, 1};

    tq::reader::seq_cell_projection_inputs in;
    in.pos = pos; in.seq_bits = bits; in.seq_ids = seqs;
    in.n_seq = 2; in.n_words = 1; in.n_used = 8; in.n_per_seq = 6; in.p_min = 0;

    const auto out = tq::reader::project_seq_cells(in);

    CHECK_EQ(out.n_real, std::size_t{3});
    CHECK_EQ(out.rows[0], 1);
    CHECK_EQ(out.rows[1], 3);
    CHECK_EQ(out.rows[2], 5);
    CHECK_EQ(out.rows[3], kPad);
    CHECK_EQ(out.rows[4], kPad);
    CHECK_EQ(out.rows[5], kPad);
}

// A1c: more real cells than the table can hold. Upstream asserts; this reports it, so a wrong
// `n_per_seq` cannot pass as a short but valid-looking table.
void case_overflow_is_reported() {
    const std::int32_t   pos[]   = {0, 1, 2, 3};
    const std::uint64_t  bits[]  = {0xFu};
    const std::int32_t   seqs[]  = {0};

    tq::reader::seq_cell_projection_inputs in;
    in.pos = pos; in.seq_bits = bits; in.seq_ids = seqs;
    in.n_seq = 1; in.n_words = 1; in.n_used = 4; in.n_per_seq = 2; in.p_min = 0;

    const auto out = tq::reader::project_seq_cells(in);

    CHECK(out.overflow);
    CHECK_EQ(out.rows.size(), std::size_t{2});
}

// A1d: an empty slice is a table of padding, not an error and not a null read.
void case_empty_slice_is_all_padding() {
    tq::reader::seq_cell_projection_inputs in;
    in.n_per_seq = 3;

    const auto out = tq::reader::project_seq_cells(in);

    CHECK_EQ(out.rows.size(), std::size_t{3});
    CHECK_EQ(out.n_real, std::size_t{0});
    CHECK(!out.overflow);
    for (const auto r : out.rows) {
        CHECK_EQ(r, kPad);
    }
}

// A1e: a bit set past `n_used` must not be emitted. The last bitmap word is rounded up to 64 bits, so
// this is reachable with an over-wide `n_words`; reading `pos[]` there would be past the array.
void case_bit_past_n_used_is_ignored() {
    const std::int32_t   pos[]   = {0, 1};       // n_used == 2, but bit 40 is set below
    const std::uint64_t  bits[]  = {0x1u | (std::uint64_t{1} << 40)};
    const std::int32_t   seqs[]  = {0};

    tq::reader::seq_cell_projection_inputs in;
    in.pos = pos; in.seq_bits = bits; in.seq_ids = seqs;
    in.n_seq = 1; in.n_words = 1; in.n_used = 2; in.n_per_seq = 2; in.p_min = 0;

    const auto out = tq::reader::project_seq_cells(in);

    CHECK_EQ(out.n_real, std::size_t{1});
    CHECK_EQ(out.rows[0], 0);
    CHECK_EQ(out.rows[1], kPad);
    CHECK(!out.overflow);
}

// A2: the round trip. `project_seq_cells` gives cell indices; the ledger gives the durable identity.
// A physical row is an ADDRESS, so the only honest check is cell -> row -> cell on both sides.
void case_identity_round_trip_through_ledger() {
    const std::int32_t   pos[]   = {10, 11, 12, 13, 14, 15};
    const std::uint64_t  bits[]  = {0x2Bu};   // cells 0, 1, 3, 5
    const std::int32_t   seqs[]  = {0};

    tq::reader::seq_cell_projection_inputs in;
    in.pos = pos; in.seq_bits = bits; in.seq_ids = seqs;
    in.n_seq = 1; in.n_words = 1; in.n_used = 6; in.n_per_seq = 6; in.p_min = 0;

    const auto out = tq::reader::project_seq_cells(in);
    CHECK_EQ(out.n_real, std::size_t{4});

    tq::reader::ledger led;
    CHECK(led.set_n_stream(1) == tq::reader::resolve_refusal::ok);
    CHECK(led.set_capacity(64) == tq::reader::resolve_refusal::ok);
    CHECK(led.set_n_sequences(2) == tq::reader::resolve_refusal::ok);

    // Record the identity of every real row, keyed by the position the cache stores beside it.
    for (std::size_t i = 0; i < out.n_real; ++i) {
        const std::int32_t row = out.rows[i];
        CHECK(row >= 0);
        const tq::reader::cell_id id{seqs[0], pos[row]};
        CHECK(led.put(id, row) == tq::reader::resolve_refusal::ok);
    }

    // Now go the other way: the identity must resolve to the SAME row the projection emitted.
    for (std::size_t i = 0; i < out.n_real; ++i) {
        const std::int32_t row = out.rows[i];
        const tq::reader::cell_id id{seqs[0], pos[row]};
        CHECK_EQ(led.slot_of(id), row);
    }

    // A position outside the table is still absent, so the round trip did not just answer "yes".
    CHECK_EQ(led.slot_of(tq::reader::cell_id{seqs[0], pos[2]}), std::int32_t{-1});
}

// A3: the padding is safe only when its mask column is invisible to EVERY token. Two runs over the
// same row table: the first with `-inf` at the two padding columns (accepted), the second with a
// finite value at ONE of them (refused, and the refusal names the column).
void case_padding_must_be_masked() {
    const std::int32_t pos[] = {0, 1, 2, 3, 4, 5};
    const std::uint64_t bits[] = {0x2Bu};   // cells 0, 1, 3, 5
    const std::int32_t seqs[] = {0};

    tq::reader::seq_cell_projection_inputs in;
    in.pos = pos; in.seq_bits = bits; in.seq_ids = seqs;
    in.n_seq = 1; in.n_words = 1; in.n_used = 6; in.n_per_seq = 6; in.p_min = 0;

    const auto out = tq::reader::project_seq_cells(in);
    CHECK_EQ(out.n_real, std::size_t{4});

    const float ninf = -std::numeric_limits<float>::infinity();

    std::vector<float> good = {0.0f, 0.0f, ninf, 0.0f, ninf, ninf};   // padding at 2 and 4
    std::vector<float> bad  = {0.0f, 0.0f, ninf, 0.0f, 0.0f,  ninf};  // padding at 4 is visible

    const std::size_t n = out.rows.size();

    auto inputs_for = [&](const std::vector<float> & mask) {
        tq::reader::runtime_rows_inputs r;
        r.rows = out.rows.data();
        r.n_columns = n;
        r.n_slices = 1;
        r.rows_slice_stride = 0;
        r.mask = mask.data();
        r.n_tokens_per_slice = 1;
        r.mask_token_stride = n * sizeof(float);   // one token's mask is n columns wide
        r.mask_slice_stride = 0;
        r.mask_is_f16 = false;
        r.n_cell = 6;
        r.rows_bytes = n * sizeof(std::int32_t);
        r.mask_bytes = mask.size() * sizeof(float);
        return r;
    };

    const auto ok_report = tq::reader::validate_runtime_rows(inputs_for(good));
    CHECK(ok_report.refusal == tq::reader::rows_validation_refusal::ok);
    CHECK_EQ(ok_report.negative_rows, std::size_t{2});
    CHECK_EQ(ok_report.max_row, std::size_t{5});

    const auto bad_report = tq::reader::validate_runtime_rows(inputs_for(bad));
    CHECK(bad_report.refusal == tq::reader::rows_validation_refusal::padding_not_masked);
    CHECK(bad_report.has_violation);
    CHECK_EQ(bad_report.violation_column, std::size_t{4});
}

} // namespace

int main() {
    case_single_sequence_with_window();
    case_two_sequences_keep_scan_order();
    case_overflow_is_reported();
    case_empty_slice_is_all_padding();
    case_bit_past_n_used_is_ignored();
    case_identity_round_trip_through_ledger();
    case_padding_must_be_masked();

    if (failures != 0) {
        std::fprintf(stderr, "tq_reader.seq_cell_projection: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("tq_reader.seq_cell_projection: all cases passed\n");
    return 0;
}
