// The host-side verification of upstream's runtime row table: every non-negative row inside the
// cache, every negative row invisible to every token of every slice.
//
// WHAT EACH CASE IS FOR, AND WHY IT COULD NOT BE DROPPED
// -----------------------------------------------------
// A validation function is easy to write in a way that passes: scan the table, call `std::isinf`,
// report ok. Every case below exists because some plausible implementation of that shape gets it
// wrong, and each one names the mistake it is holding down:
//
//   * a NON-MONOTONIC table with INTERLEAVED padding, so "identity" and "ascending" cannot pass by
//     accident, and so the scan cannot be a prefix check;
//   * padding visible to ONE token out of three -- the case a single-mask-column check certifies.
//     This is the defect the previous attempt in this tree shipped;
//   * a row exactly AT the cache depth and one beyond it, because `row < n_cell` and `row <= n_cell`
//     differ only there;
//   * NaN and `+inf` at a padding column, because `!std::isfinite` accepts both and neither is `-inf`;
//   * a FINITE value placed between two token columns, i.e. in the stride gap: if the scan read the
//     mask contiguously instead of through `mask_token_stride`, that value would be seen and the
//     clean table would be refused -- so this case fails an implementation that ignores the stride;
//   * a second slice whose table is fine while the first is not, so a scan that stops after slice 0
//     is caught, and a scan that reuses slice 0's rows for slice 1 is caught by the mirror case;
//   * an F16 mask, because the SAME code path reads a different element width and `-inf` has to
//     survive that width;
//   * overlapping strides (token and slice), because a caller's arithmetic mistake must be a named
//     refusal and not an out-of-bounds read inside the validator;
//   * a zero-length claim, because "verified nothing" must not read as "verified".
//
// The reference is arithmetic, not a second implementation: each case states the coordinates it
// expects in the report, so a scan that finds the right VERDICT via the wrong COLUMN still fails.
#include <tq/reader/rows_validation.h>

#include <cmath>
#include <cstdio>
#include <functional>
#include <cstring>
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

constexpr std::size_t kColumns = 6;
constexpr std::size_t kTokens = 3;
constexpr std::size_t kCells = 8;   // cache depth: a row must be < this

const float kNegInf = -INFINITY;

// A mask laid out [columns, tokens, 1, slices] exactly as ggml would, with `visible(c, t, s)` deciding
// each entry. Stored through the same type the validator will read, so an F16 case exercises the
// width and not a conversion.
template <typename MaskT>
struct mask_buffer {
    std::vector<MaskT> data;
    std::size_t token_stride = 0;   // bytes
    std::size_t slice_stride = 0;   // bytes

    // `std::function`, not a function pointer: several cases capture the offending value (`[bad]`),
    // and a capture-less signature would have forced those cases to be written as statics with file
    // scope state -- which is exactly the kind of sharing that makes a test's cases interfere.
    mask_buffer(std::size_t n_tokens, std::size_t n_slices,
                const std::function<float(std::size_t, std::size_t, std::size_t)> & value) {
        token_stride = kColumns * sizeof(MaskT);
        slice_stride = n_tokens * token_stride;
        data.assign(kColumns * n_tokens * n_slices, MaskT{});
        for (std::size_t s = 0; s < n_slices; ++s)
            for (std::size_t t = 0; t < n_tokens; ++t)
                for (std::size_t c = 0; c < kColumns; ++c)
                    at(c, t, s) = from_float(value(c, t, s));
    }

    MaskT & at(std::size_t c, std::size_t t, std::size_t s) {
        const std::size_t byte_off = c * sizeof(MaskT) + t * token_stride + s * slice_stride;
        return *reinterpret_cast<MaskT *>(reinterpret_cast<unsigned char *>(data.data()) + byte_off);
    }

    static MaskT from_float(float v);
    const void * base() const { return data.data(); }
    // Readable bytes behind `base()`, so a case passes the buffer's own extent instead of restating it.
    std::size_t bytes() const { return data.size() * sizeof(MaskT); }
};

template <>
float mask_buffer<float>::from_float(float v) { return v; }
template <>
ggml_fp16_t mask_buffer<ggml_fp16_t>::from_float(float v) { return ggml_fp32_to_fp16(v); }

// Rows {7, -1, 0, -1, 5, 2}: two padding columns, neither at an end, order neither ascending nor the
// identity. A prefix-only or sorted-only scan cannot read this as clean by luck.
const std::int32_t kMixedRows[kColumns] = {7, -1, 0, -1, 5, 2};

bool is_padding(std::size_t c) { return kMixedRows[c] < 0; }

// Default mask: invisible exactly where the table is padded, zero bias elsewhere.
float masked_everywhere(std::size_t c, std::size_t, std::size_t) {
    return is_padding(c) ? kNegInf : 0.0f;
}

// The FIRST case. Non-monotonic, interleaved padding, every token invisible there.
template <typename MaskT>
void clean_table_is_accepted(const char * label) {
    mask_buffer<MaskT> mask(kTokens, 1, &masked_everywhere);
    tq::reader::runtime_rows_inputs in;
    in.rows = kMixedRows;
    in.n_columns = kColumns;
    in.n_slices = 1;
    in.rows_slice_stride = 0;   // unused for one slice
    in.mask = mask.base();
    in.n_tokens_per_slice = kTokens;
    in.mask_token_stride = mask.token_stride;
    in.mask_slice_stride = mask.slice_stride;
    in.mask_is_f16 = sizeof(MaskT) == sizeof(ggml_fp16_t);
    in.n_cell = kCells;
    in.rows_bytes = kColumns * sizeof(std::int32_t);
    in.mask_bytes = mask.bytes();

    const auto r = tq::reader::validate_runtime_rows(in);
    check(r.refusal == tq::reader::rows_validation_refusal::ok,
          std::string(label) + ": a clean table is accepted, got " +
              tq::reader::describe(r.refusal) + " -- " + r.why);
    check(!r.has_violation, std::string(label) + ": and reports no violation");
    check(r.columns_checked == kColumns, std::string(label) + ": checked " +
                                             std::to_string(r.columns_checked) + " of " +
                                             std::to_string(kColumns) + " columns");
    check(r.negative_rows == 2, std::string(label) + ": saw " +
                                    std::to_string(r.negative_rows) + " padding columns, expected 2");
    // Two padding columns times three tokens: the count is what proves the per-token scan happened
    // rather than a per-column one.
    check(r.tokens_checked == 2 * kTokens,
          std::string(label) + ": checked " + std::to_string(r.tokens_checked) +
              " mask entries for the padding columns, expected " + std::to_string(2 * kTokens));
    check(r.max_row == 7, std::string(label) + ": largest row is " + std::to_string(r.max_row));
}

// A padding column that only token 2 can see. This is THE case: a check that reads one mask column
// (or reads token 0 for every column) passes the clean table above and fails here.
void padding_visible_to_one_token_is_refused() {
    mask_buffer<float> mask(kTokens, 1, [](std::size_t c, std::size_t t, std::size_t) {
        if (!is_padding(c)) return 0.0f;
        return (c == 3 && t == 2) ? 0.5f : kNegInf;   // column 3, token 2 only
    });
    tq::reader::runtime_rows_inputs in;
    in.rows = kMixedRows;
    in.n_columns = kColumns;
    in.n_slices = 1;
    in.mask = mask.base();
    in.n_tokens_per_slice = kTokens;
    in.mask_token_stride = mask.token_stride;
    in.mask_slice_stride = mask.slice_stride;
    in.n_cell = kCells;
    in.rows_bytes = kColumns * sizeof(std::int32_t);
    in.mask_bytes = mask.bytes();

    const auto r = tq::reader::validate_runtime_rows(in);
    check(r.refusal == tq::reader::rows_validation_refusal::padding_not_masked,
          std::string("a padding column visible to token 2 is refused, got ") +
              tq::reader::describe(r.refusal));
    check(r.has_violation, "the refusal carries coordinates");
    check(r.violation_column == 3, "and names column 3, got " + std::to_string(r.violation_column));
    check(r.violation_token == 2, "and names token 2, got " + std::to_string(r.violation_token));
    check(r.violation_row == -1, "and reports the padding marker, got " +
                                     std::to_string(r.violation_row));
}

// `row == n_cell` and `row > n_cell`: an off-by-one in the bound lives exactly here.
void rows_at_and_past_the_depth_are_refused() {
    for (std::int32_t bad : {std::int32_t(kCells), std::int32_t(kCells + 40)}) {
        std::int32_t rows[kColumns] = {0, 1, 2, 3, 4, bad};
        mask_buffer<float> mask(kTokens, 1, [](std::size_t, std::size_t, std::size_t) { return 0.0f; });
        tq::reader::runtime_rows_inputs in;
        in.rows = rows;
        in.n_columns = kColumns;
        in.n_slices = 1;
        in.mask = mask.base();
        in.n_tokens_per_slice = kTokens;
        in.mask_token_stride = mask.token_stride;
        in.mask_slice_stride = mask.slice_stride;
        in.n_cell = kCells;
        in.rows_bytes = sizeof rows;
        in.mask_bytes = mask.bytes();

        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::row_out_of_range,
              "row " + std::to_string(bad) + " is out of range, got " +
                  tq::reader::describe(r.refusal));
        check(r.violation_column == kColumns - 1,
              "and the violation is the last column, got " + std::to_string(r.violation_column));
        check(r.violation_row == bad, "and reports the offending value");
    }
}

// Not-finite is not the test: NaN and +inf both fail "is -inf", and the kernel's clamp would make
// either one a real weight.
void nan_and_positive_infinity_are_not_padding() {
    for (float bad : {std::nanf(""), INFINITY}) {
        mask_buffer<float> mask(kTokens, 1, [bad](std::size_t c, std::size_t, std::size_t) {
            return is_padding(c) ? bad : 0.0f;
        });
        tq::reader::runtime_rows_inputs in;
        in.rows = kMixedRows;
        in.n_columns = kColumns;
        in.n_slices = 1;
        in.mask = mask.base();
        in.n_tokens_per_slice = kTokens;
        in.mask_token_stride = mask.token_stride;
        in.mask_slice_stride = mask.slice_stride;
        in.n_cell = kCells;
        in.rows_bytes = kColumns * sizeof(std::int32_t);
        in.mask_bytes = mask.bytes();

        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::padding_not_masked,
              std::string("mask value ") + std::to_string(bad) +
                  " at a padding column is refused, got " + tq::reader::describe(r.refusal));
    }
}

// A FINITE value sitting in the stride gap between token columns. It belongs to no mask entry; a scan
// that walked the mask contiguously would read it, see a finite value, and refuse a clean table.
// So this case is what makes `mask_token_stride` load-bearing rather than decorative.
void values_in_the_stride_gap_are_not_read() {
    // Deliberately over-allocate: 2x the packed token stride, with garbage in the gap.
    std::vector<float> raw(kColumns * kTokens * 2, 123.0f);
    const std::size_t token_stride = kColumns * sizeof(float) * 2;
    for (std::size_t t = 0; t < kTokens; ++t) {
        for (std::size_t c = 0; c < kColumns; ++c) {
            float v = is_padding(c) ? kNegInf : 0.0f;
            std::memcpy(reinterpret_cast<unsigned char *>(raw.data()) + token_stride * t +
                            c * sizeof(float),
                        &v, sizeof v);
        }
    }
    tq::reader::runtime_rows_inputs in;
    in.rows = kMixedRows;
    in.n_columns = kColumns;
    in.n_slices = 1;
    in.mask = raw.data();
    in.n_tokens_per_slice = kTokens;
    in.mask_token_stride = token_stride;
    in.mask_slice_stride = kTokens * token_stride;
    in.n_cell = kCells;
    in.rows_bytes = kColumns * sizeof(std::int32_t);
    in.mask_bytes = raw.size() * sizeof(float);

    const auto r = tq::reader::validate_runtime_rows(in);
    check(r.refusal == tq::reader::rows_validation_refusal::ok,
          std::string("a padded token stride is honoured, not walked through: got ") +
              tq::reader::describe(r.refusal) + " -- " + r.why);
}

// Two slices, each with its own row table and its own mask. Slice 1 is the broken one, so a scan that
// stops after slice 0 misses it -- and its rows differ from slice 0's, so a scan that reuses slice 0's
// rows reads the wrong numbers.
void second_slice_is_scanned_with_its_own_rows() {
    const std::int32_t slice0[kColumns] = {0, 1, 2, 3, 4, 5};          // all valid
    const std::int32_t slice1[kColumns] = {0, 1, 2, 3, 4, std::int32_t(kCells)};  // last one past the cache
    std::vector<std::int32_t> rows(std::size_t(kColumns) * 2);
    std::memcpy(rows.data(), slice0, sizeof slice0);
    std::memcpy(rows.data() + kColumns, slice1, sizeof slice1);

    mask_buffer<float> mask(kTokens, 2, [](std::size_t, std::size_t, std::size_t) { return 0.0f; });

    tq::reader::runtime_rows_inputs in;
    in.rows = rows.data();
    in.n_columns = kColumns;
    in.n_slices = 2;
    in.rows_slice_stride = kColumns * sizeof(std::int32_t);
    in.mask = mask.base();
    in.n_tokens_per_slice = kTokens;
    in.mask_token_stride = mask.token_stride;
    in.mask_slice_stride = mask.slice_stride;
    in.n_cell = kCells;
    in.rows_bytes = rows.size() * sizeof(std::int32_t);
    in.mask_bytes = mask.bytes();

    const auto r = tq::reader::validate_runtime_rows(in);
    check(r.refusal == tq::reader::rows_validation_refusal::row_out_of_range,
          std::string("the second slice's bad row is found, got ") +
              tq::reader::describe(r.refusal) + " -- " + r.why);
    check(r.violation_slice == 1, "and the violation names slice 1, got " +
                                      std::to_string(r.violation_slice));
    check(r.violation_column == kColumns - 1, "and its last column");

    // Mirror: a padding column in slice 1 visible only to a slice-1 token must also be caught, which a
    // scan that used slice 0's mask entries would miss entirely.
    mask_buffer<float> two(kTokens, 2, [](std::size_t c, std::size_t t, std::size_t s) {
        if (!is_padding(c)) return 0.0f;
        return (s == 1 && t == 1) ? 0.25f : kNegInf;
    });
    const std::int32_t padded[kColumns] = {7, -1, 0, -1, 5, 2};
    std::memcpy(rows.data(), padded, sizeof padded);
    std::memcpy(rows.data() + kColumns, padded, sizeof padded);
    in.mask = two.base();
    in.mask_token_stride = two.token_stride;
    in.mask_slice_stride = two.slice_stride;
    in.mask_bytes = two.bytes();   // a different buffer, so a stale extent would be a lie
    const auto r2 = tq::reader::validate_runtime_rows(in);
    check(r2.refusal == tq::reader::rows_validation_refusal::padding_not_masked,
          std::string("padding visible to a slice-1 token is refused, got ") +
              tq::reader::describe(r2.refusal));
    check(r2.violation_slice == 1 && r2.violation_token == 1,
          "and names slice 1 token 1, got slice " + std::to_string(r2.violation_slice) +
              " token " + std::to_string(r2.violation_token));
}

// Structural refusals: a caller's arithmetic mistake must be a named verdict, decided before any
// stride is dereferenced.
void inconsistent_claims_are_refused() {
    mask_buffer<float> mask(kTokens, 1, &masked_everywhere);

    auto base = [&]() {
        tq::reader::runtime_rows_inputs in;
        in.rows = kMixedRows;
        in.n_columns = kColumns;
        in.n_slices = 1;
        in.mask = mask.base();
        in.n_tokens_per_slice = kTokens;
        in.mask_token_stride = mask.token_stride;
        in.mask_slice_stride = mask.slice_stride;
        in.n_cell = kCells;
        in.rows_bytes = kColumns * sizeof(std::int32_t);
        in.mask_bytes = mask.bytes();
        return in;
    };

    {   auto in = base(); in.mask_token_stride = sizeof(float);   // tokens overlap
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::bad_shape,
              std::string("an overlapping token stride is bad_shape, got ") +
                  tq::reader::describe(r.refusal));
        check(!r.has_violation, "and it is decided before any value is read"); }

    {   auto in = base(); in.n_columns = 0;
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::bad_shape,
              "zero columns is bad_shape, not a vacuous ok"); }

    {   auto in = base(); in.mask = nullptr;
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::bad_shape, "a null mask is refused"); }

    {   auto in = base(); in.n_cell = 0;
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::bad_shape,
              "a zero-depth cache is refused rather than accepting every row"); }

    {   // Two slices whose row stride is smaller than one slice: the slices alias.
        //     A REAL TWO-SLICE BUFFER, because the extent check runs before this one -- and that
        //     ordering is right: "your claim does not fit your buffer" is a more basic defect than
        //     "your strides overlap". Passing a one-slice extent here would make the extent refusal
        //     fire and leave the stride check unexercised, which is exactly what the first run of
        //     this fix showed.
        auto in = base();
        in.n_slices = 2;
        in.rows_slice_stride = sizeof(std::int32_t);
        // A REAL TWO-SLICE ROW ARRAY, and this is the fix for a fixture that LIED about its extent.
        // The previous version kept `in.rows = kMixedRows` -- six I32 entries, 24 bytes -- while
        // claiming `rows_bytes = 48`. That claim is exactly the kind of overstatement the validator
        // cannot detect, so the case was passing on a buffer it did not own: had the scan reached
        // slice 1 it would have read 24 bytes past the array, in the test process, while the test
        // asserted a *stride* property.
        //
        // A fixture that misstates its own extent cannot be used to argue anything about extent
        // checks, and it makes the aliasing assertion accidental rather than established.
        const std::int32_t two_slice_rows[2 * kColumns] = {
            7, -1, 0, -1, 5, 2,
            3, 4, -1, 6, 1, 0,
        };
        in.rows = two_slice_rows;
        in.rows_bytes = sizeof two_slice_rows;
        // THE MASK NEEDS A TWO-SLICE BUFFER AS WELL, or the extent check (which runs first, and
        // should) refuses the claim before the stride check is reached. This case is about stride
        // aliasing, so everything else has to be consistent.
        mask_buffer<float> two_rows(kTokens, 2, &masked_everywhere);
        in.mask = two_rows.base();
        in.mask_token_stride = two_rows.token_stride;
        in.mask_slice_stride = two_rows.slice_stride;
        in.mask_bytes = two_rows.bytes();
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::bad_shape,
              std::string("an aliasing row slice stride is bad_shape, got ") +
                  tq::reader::describe(r.refusal) + " -- " + r.why); }

    {   // Two slices whose mask slice stride is smaller than one slice.
        auto in = base();
        in.n_slices = 2;
        in.rows_slice_stride = kColumns * sizeof(std::int32_t);
        // A REAL TWO-SLICE ROW ARRAY (same reason as the case above): claiming 48 bytes over a
        // 24-byte array would make the fixture the liar, and the aliasing verdict above would then
        // rest on a buffer the test does not own.
        const std::int32_t two_slice_rows[2 * kColumns] = {
            7, -1, 0, -1, 5, 2,
            3, 4, -1, 6, 1, 0,
        };
        in.rows = two_slice_rows;
        in.rows_bytes = sizeof two_slice_rows;
        mask_buffer<float> two(kTokens, 2, &masked_everywhere);
        in.mask = two.base();
        in.mask_token_stride = two.token_stride;
        in.mask_slice_stride = two.token_stride;   // one token row short of a slice
        in.mask_bytes = two.bytes();
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::bad_shape,
              std::string("an aliasing mask slice stride is bad_shape, got ") +
                  tq::reader::describe(r.refusal) + " -- " + r.why); }
}

// F16 MASKS ARE READ IN THEIR OWN WIDTH, and the case that used to say otherwise is back.
//
// The earlier revision deleted this case after it failed and blamed `ggml_fp32_to_fp16` for losing
// infinities. That attribution was wrong, and the refuting numbers were already in the tree:
// `tq_reader.rows_attention` round-trips `-INFINITY` through those two converters and asserts
// equality. The real cause was in the VALIDATOR -- one column offset (`c * sizeof(std::int32_t)`)
// reused for the mask as well, so an F16 column was read at byte 4c instead of 2c; at the end of a
// token that lands on the next token's column 0, whose bias is `0.0f`, and a clean table was refused
// with "padding not masked".
//
// So this case is load-bearing rather than decorative: with the old offset it fails, with the fixed
// one it passes, and it reads the LAST column through the narrow width, where a factor-of-two error
// first shows.
template <typename MaskT>
void narrow_mask_width_is_read_correctly(const char * label) {
    mask_buffer<MaskT> mask(kTokens, 1, &masked_everywhere);
    tq::reader::runtime_rows_inputs in;
    in.rows = kMixedRows;
    in.n_columns = kColumns;
    in.n_slices = 1;
    in.mask = mask.base();
    in.n_tokens_per_slice = kTokens;
    in.mask_token_stride = mask.token_stride;
    in.mask_slice_stride = mask.slice_stride;
    // THE WIDTH COMES FROM THE TYPE, NOT FROM THE CASE'S INTENTION. The first version wrote `true`
    // here and instantiated the template on `float` as well, so the "f32-width" case declared a
    // 2-byte mask while handing over a 4-byte buffer: the validator read every entry at half its
    // real offset, landed on a neighbouring column whose bias is `0.0f`, and refused a clean table.
    // The test was wrong, not the validator -- and the failure message said so precisely, naming
    // column 1 token 0.
    in.mask_is_f16 = (sizeof(MaskT) == sizeof(ggml_fp16_t));
    in.n_cell = kCells;
    in.rows_bytes = kColumns * sizeof(std::int32_t);
    in.mask_bytes = mask.bytes();

    const auto r = tq::reader::validate_runtime_rows(in);
    check(r.refusal == tq::reader::rows_validation_refusal::ok,
          std::string(label) + ": a clean table is accepted, got " +
              tq::reader::describe(r.refusal) + " -- " + r.why);
    check(r.negative_rows == 2, std::string(label) + ": two padding columns, got " +
                                   std::to_string(r.negative_rows));
    // TWO padding columns x THREE tokens: this count proves the scan is per (column, token) AND that
    // the narrow width does not make it read half as many entries.
    check(r.tokens_checked == 6, std::string(label) + ": six mask entries checked, got " +
                                    std::to_string(r.tokens_checked));

    // The violation side at the same width: token 2 alone sees padding column 3.
    mask_buffer<MaskT> bad(kTokens, 1, [](std::size_t c, std::size_t t, std::size_t) {
        if (!is_padding(c)) return 0.0f;
        return (c == 3 && t == 2) ? 0.25f : kNegInf;
    });
    in.mask = bad.base();
    in.mask_token_stride = bad.token_stride;
    in.mask_bytes = bad.bytes();
    const auto r2 = tq::reader::validate_runtime_rows(in);
    check(r2.refusal == tq::reader::rows_validation_refusal::padding_not_masked,
          std::string(label) + ": one visible padding entry is refused, got " +
              tq::reader::describe(r2.refusal));
    check(r2.violation_column == 3 && r2.violation_token == 2,
          std::string(label) + ": and names column 3 token 2, got " +
              std::to_string(r2.violation_column) + "/" + std::to_string(r2.violation_token));
}

// F16 WITH TWO SLICES, AND THE PADDING IN THE LAST COLUMN.
//
// Two things are only exercised here, and each is a place where a plausible implementation is wrong:
//
//   * THE LAST COLUMN. A factor-of-two offset error first shows at the end of a row: column
//     `kColumns-1` of an F16 mask lives at byte `2*(kColumns-1)`, and reading it at `4*(kColumns-1)`
//     lands outside the token row entirely -- on the next token's column 0, or past the buffer.
//     Earlier F16 cases used a single slice and padding in the middle, where an error can land on a
//     neighbouring column that happens to hold `-inf` too and the case still passes.
//   * THE SECOND SLICE. `slice_stride` is a third offset added to the same sum; with one slice it is
//     never used, so a validator that ignored it would pass every single-slice test.
//
// So the clean case puts padding at the last column of BOTH slices, and the violation case makes it
// visible to exactly one token of the SECOND slice -- the coordinates asserted are what a validator
// that mixed up slices or widths could not produce.
template <typename MaskT>
void f16_two_slices_last_column(const char * label) {
    // Last entry of each slice is padding: rows per slice = {3, ..., -1}.
    const std::int32_t rows[2 * kColumns] = {
        7, 2, 0, 3, 5, -1,
        3, 4, 2, 6, 1, -1,
    };
    const std::size_t last = kColumns - 1;

    {   // Clean: every padding column invisible to every token of its own slice.
        mask_buffer<MaskT> mask(kTokens, 2, [](std::size_t c, std::size_t, std::size_t) {
            return (c == last) ? kNegInf : 0.0f;
        });
        tq::reader::runtime_rows_inputs in;
        in.rows = rows;
        in.n_columns = kColumns;
        in.n_slices = 2;
        in.rows_slice_stride = kColumns * sizeof(std::int32_t);
        in.mask = mask.base();
        in.n_tokens_per_slice = kTokens;
        in.mask_token_stride = mask.token_stride;
        in.mask_slice_stride = mask.slice_stride;
        in.mask_is_f16 = (sizeof(MaskT) == sizeof(ggml_fp16_t));
        in.n_cell = kCells;
        in.rows_bytes = sizeof rows;
        in.mask_bytes = mask.bytes();

        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::ok,
              std::string(label) + " two slices: clean table accepted, got " +
                  tq::reader::describe(r.refusal) + " -- " + r.why);
        // One padding column per slice, so two; and two x three tokens checked. Both counts are
        // assertions, not decoration: a validator that read only slice 0 reports 1 and 3.
        check(r.negative_rows == 2, std::string(label) + " two slices: two padding columns, got " +
                                       std::to_string(r.negative_rows));
        check(r.tokens_checked == 6, std::string(label) + " two slices: six mask entries, got " +
                                        std::to_string(r.tokens_checked));
        check(r.slices_checked == 2, std::string(label) + " two slices: both slices scanned, got " +
                                        std::to_string(r.slices_checked));
    }

    {   // Violation: the LAST column of the SECOND slice is visible to token 2 only.
        mask_buffer<MaskT> mask(kTokens, 2, [](std::size_t c, std::size_t t, std::size_t s) {
            if (c != last) return 0.0f;
            return (s == 1 && t == 2) ? 0.25f : kNegInf;
        });
        tq::reader::runtime_rows_inputs in;
        in.rows = rows;
        in.n_columns = kColumns;
        in.n_slices = 2;
        in.rows_slice_stride = kColumns * sizeof(std::int32_t);
        in.mask = mask.base();
        in.n_tokens_per_slice = kTokens;
        in.mask_token_stride = mask.token_stride;
        in.mask_slice_stride = mask.slice_stride;
        in.mask_is_f16 = (sizeof(MaskT) == sizeof(ggml_fp16_t));
        in.n_cell = kCells;
        in.rows_bytes = sizeof rows;
        in.mask_bytes = mask.bytes();

        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::padding_not_masked,
              std::string(label) + " two slices: last-column padding visible to one token is " +
                  "refused, got " + tq::reader::describe(r.refusal) + " -- " + r.why);
        check(r.has_violation, std::string(label) + " two slices: the refusal carries coordinates");
        check(r.violation_slice == 1, std::string(label) + " two slices: names slice 1, got " +
                                         std::to_string(r.violation_slice));
        check(r.violation_column == last, std::string(label) + " two slices: names the LAST column (" +
                                              std::to_string(last) + "), got " +
                                              std::to_string(r.violation_column));
        check(r.violation_token == 2, std::string(label) + " two slices: names token 2, got " +
                                         std::to_string(r.violation_token));
    }
}

// A buffer one byte shorter than the claim must be REFUSED, not read. This is what makes
// `rows_bytes`/`mask_bytes` load-bearing instead of advisory: without them the validator would be the
// out-of-bounds reader it exists to prevent.
void truncated_buffers_are_refused() {
    mask_buffer<float> mask(kTokens, 1, &masked_everywhere);
    auto base = [&]() {
        tq::reader::runtime_rows_inputs in;
        in.rows = kMixedRows;
        in.n_columns = kColumns;
        in.n_slices = 1;
        in.mask = mask.base();
        in.n_tokens_per_slice = kTokens;
        in.mask_token_stride = mask.token_stride;
        in.mask_slice_stride = mask.slice_stride;
        in.n_cell = kCells;
        in.rows_bytes = kColumns * sizeof(std::int32_t);
        in.mask_bytes = mask.bytes();
        return in;
    };

    {   auto in = base(); in.mask_bytes = mask.bytes() - 1;
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::buffer_too_small,
              std::string("a mask one byte short is refused, got ") +
                  tq::reader::describe(r.refusal) + " -- " + r.why);
        check(!r.has_violation, "and it is decided before any value is read"); }

    {   auto in = base(); in.rows_bytes = kColumns * sizeof(std::int32_t) - 1;
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::buffer_too_small,
              std::string("a row table one byte short is refused, got ") +
                  tq::reader::describe(r.refusal)); }

    {   auto in = base(); in.mask_bytes = 0;
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::bad_shape,
              "mask_bytes 0 is bad_shape: zero would mean 'unbounded', the reading this field "
              "exists to prevent"); }

    {   auto in = base(); in.rows_bytes = 0;
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::bad_shape,
              "rows_bytes 0 is bad_shape"); }

    // The extent is checked over the WHOLE claimed range, not over the entries a clean table happens
    // to visit: padding lives at columns 1 and 3, yet losing the LAST token row must still be caught.
    {   auto in = base(); in.mask_bytes = mask.bytes() - (kTokens * sizeof(float));
        const auto r = tq::reader::validate_runtime_rows(in);
        check(r.refusal == tq::reader::rows_validation_refusal::buffer_too_small,
              std::string("a mask missing its last token row is refused even though no padding "
                          "column lives there: got ") + tq::reader::describe(r.refusal)); }
}

// An extent whose product wraps is refused, never accepted as a small number: the wrapped offset
// would read the WRONG entry while still landing inside the buffer, which no later check can see.
void overflowing_extents_are_refused() {
    mask_buffer<float> mask(kTokens, 1, &masked_everywhere);
    tq::reader::runtime_rows_inputs in;
    in.rows = kMixedRows;
    in.n_columns = kColumns;
    in.n_slices = 1;
    in.mask = mask.base();
    in.n_tokens_per_slice = kTokens;
    in.mask_token_stride = mask.token_stride;
    in.mask_slice_stride = mask.slice_stride;
    in.n_cell = kCells;
    in.rows_bytes = kColumns * sizeof(std::int32_t);
    in.mask_bytes = mask.bytes();

    {   auto big = in;
        big.n_tokens_per_slice = std::size_t(-1) / 2;
        big.mask_token_stride = kColumns * sizeof(float) * 2;
        const auto r = tq::reader::validate_runtime_rows(big);
        check(r.refusal == tq::reader::rows_validation_refusal::overflow ||
                  r.refusal == tq::reader::rows_validation_refusal::buffer_too_small,
              std::string("a wrapping token extent is refused, got ") +
                  tq::reader::describe(r.refusal)); }

    {   auto wide = in;
        wide.n_slices = std::size_t(-1) / 2;
        wide.rows_slice_stride = kColumns * sizeof(std::int32_t);
        wide.mask_slice_stride = mask.slice_stride;
        const auto r = tq::reader::validate_runtime_rows(wide);
        check(r.refusal == tq::reader::rows_validation_refusal::overflow ||
                  r.refusal == tq::reader::rows_validation_refusal::buffer_too_small,
              std::string("a wrapping slice extent is refused, got ") +
                  tq::reader::describe(r.refusal)); }
}

} // namespace

int main() {
    clean_table_is_accepted<float>("f32");
    padding_visible_to_one_token_is_refused();
    rows_at_and_past_the_depth_are_refused();
    nan_and_positive_infinity_are_not_padding();
    values_in_the_stride_gap_are_not_read();
    second_slice_is_scanned_with_its_own_rows();
    inconsistent_claims_are_refused();
    narrow_mask_width_is_read_correctly<float>("f32-width");
    narrow_mask_width_is_read_correctly<ggml_fp16_t>("f16");
    f16_two_slices_last_column<float>("f32-width");
    f16_two_slices_last_column<ggml_fp16_t>("f16");
    truncated_buffers_are_refused();
    overflowing_extents_are_refused();

    if (failures == 0) {
        std::fprintf(stderr, "rows validation: OK\n");
        return 0;
    }
    std::fprintf(stderr, "rows validation: %d failure(s)\n", failures);
    return 1;
}
