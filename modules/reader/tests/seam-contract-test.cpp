// The seam's DESCRIPTION contract, driven directly.
//
// Every case here constructs a `tensor_face` by hand — two integers arrays and three pointer values —
// and calls `validate_seam_contract`. No ggml context, no tensor, no allocation, no read. That is the
// point: the checks used to live inline in a seam where each refusal called `abort()`, so the only
// path ever executed was the accepting one, for four revisions.
//
// WHAT EACH CASE IS FOR, AND WHY IT COULD NOT BE OBSERVED BEFORE
// -------------------------------------------------------------
// A refusal in the old code was a process death. An exit code cannot tell you WHICH check fired, so
// "the rank check runs before the stride check" and "this case is decided before any byte is read"
// were both unobservable claims. Here the report carries the refusal code and the field name, so both
// become assertions.
//
// THE ONE CLAIM THIS FILE DOES *NOT* MAKE: that `buffer_size` is the tensor's length. It is the
// allocation's extent, and one allocation routinely holds several tensors, so the derived number is
// an upper bound on readable bytes. The last group asserts exactly that distinction rather than
// leaving a reader to infer the stronger, false claim from a green test.
#include <tq/reader/seam_contract.h>

#include <cstdio>
#include <cstring>
#include <string>

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

// A character buffer standing in for a backend allocation. Its address is the only thing that
// matters: the contract compares pointer VALUES and never dereferences them, so this can be any
// memory with a known extent.
struct arena {
    char bytes[4096] = {};

    void * base() { return bytes; }
    std::size_t size() const { return sizeof bytes; }
    void * at(std::size_t offset) { return bytes + offset; }
};

// A rows face that satisfies every check: I32, nb[0] == 4, six columns, one slice, data at the base
// of its own arena.
tq::reader::tensor_face good_rows(arena & a) {
    tq::reader::tensor_face f;
    f.ne[0] = std::int64_t(kColumns);
    f.ne[1] = 1;
    f.ne[2] = 1;
    f.ne[3] = 1;
    f.nb[0] = sizeof(std::int32_t);
    f.nb[1] = kColumns * sizeof(std::int32_t);
    f.type_is_i32 = true;
    f.data = a.base();
    f.buffer_base = a.base();
    f.buffer_size = a.size();
    f.buffer_present = true;
    f.buffer_is_host = true;
    return f;
}

// A mask face that satisfies every check: F16, nb[0] == 2, six columns by three tokens by one slice.
tq::reader::tensor_face good_mask(arena & a, std::size_t offset = 512) {
    tq::reader::tensor_face f;
    f.ne[0] = std::int64_t(kColumns);
    f.ne[1] = std::int64_t(kTokens);
    f.ne[2] = 1;
    f.ne[3] = 1;
    f.nb[0] = sizeof(std::uint16_t);
    f.nb[1] = kColumns * sizeof(std::uint16_t);
    f.nb[3] = kTokens * f.nb[1];
    f.type_is_f16 = true;
    f.type_is_float = true;
    f.data = a.at(offset);
    f.buffer_base = a.base();
    f.buffer_size = a.size();
    f.buffer_present = true;
    f.buffer_is_host = true;
    return f;
}

void refuses(tq::reader::tensor_face rows, tq::reader::tensor_face mask,
             tq::reader::seam_contract_refusal want, const char * field, const char * label) {
    const tq::reader::seam_contract_report r = tq::reader::validate_seam_contract(rows, mask);
    check(r.refusal == want,
          std::string(label) + ": expected " + tq::reader::describe(want) + ", got " +
              tq::reader::describe(r.refusal) + " -- " + r.why);
    check(std::strcmp(r.field, field) == 0,
          std::string(label) + ": the refusal names field \"" + field + "\", got \"" + r.field +
              "\" -- a message that does not say WHICH input is wrong sends its reader hunting");
    check(!r.why.empty(), std::string(label) + ": the refusal carries a reason");
}

} // namespace

int main() {
    arena ra;
    arena ma;

    // (0) THE CONTROL. Without it, a fixture broken in a way that makes everything refuse would
    //     satisfy every case below.
    {
        const tq::reader::seam_contract_report r =
            tq::reader::validate_seam_contract(good_rows(ra), good_mask(ma));
        check(r.refusal == tq::reader::seam_contract_refusal::ok,
              std::string("control: a well-formed description is accepted, got ") +
                  tq::reader::describe(r.refusal) + " -- " + r.why);
        check(r.n_columns == kColumns && r.n_tokens == kTokens && r.n_slices == 1,
              "control: the report carries the modelled shape");
        check(r.mask_is_f16, "control: and the mask width it read");
        // The extents are the remaining bytes from each data pointer, not the arena size: the mask
        // starts 512 bytes in, so its extent is smaller by exactly that much. Asserting the
        // difference is what shows the arithmetic uses the offset rather than dropping it.
        check(r.rows_readable_bytes == ra.size(),
              "control: rows at the base see the whole allocation, got " +
                  std::to_string(r.rows_readable_bytes));
        check(r.mask_readable_bytes == ma.size() - 512,
              "control: the mask sees the allocation minus its offset, got " +
                  std::to_string(r.mask_readable_bytes));
    }

    // (1) ROWS dtype and column width. Both are one check in the code, and both are real: an F32 row
    //     list would be read as pairs of I32, and a wider nb[0] would read the gap between entries.
    {
        auto rows = good_rows(ra);
        rows.type_is_i32 = false;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_not_i32, "rows",
                "rows with the wrong dtype");
    }
    {
        auto rows = good_rows(ra);
        rows.nb[0] = 8;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_not_i32, "rows",
                "rows with nb[0] == 8");
    }

    // (2) ROWS residency: no buffer, a non-host buffer, a buffer with no geometry, and data outside
    //     its own allocation. Each is a distinct way for "the values are in host memory" to be false.
    {
        auto rows = good_rows(ra);
        rows.buffer_present = false;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_not_host, "rows",
                "rows with no backend buffer");
    }
    {
        auto rows = good_rows(ra);
        rows.buffer_is_host = false;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_not_host, "rows",
                "rows in a device buffer");
    }
    {
        auto rows = good_rows(ra);
        rows.buffer_size = 0;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_buffer_geometry, "rows",
                "rows whose buffer reports no size");
    }
    {
        auto rows = good_rows(ra);
        rows.buffer_base = nullptr;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_buffer_geometry, "rows",
                "rows whose buffer reports no base");
    }
    {
        auto rows = good_rows(ra);
        rows.data = nullptr;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_no_data, "rows",
                "rows placed but not written");
    }
    {
        // data BEFORE the base: the case a pointer subtraction would wrap on.
        auto rows = good_rows(ra);
        rows.data = ra.base();
        rows.buffer_base = ra.at(64);
        rows.buffer_size = 64;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_data_outside, "rows",
                "rows whose data precedes its buffer");
    }
    {
        // data at or past base + size.
        auto rows = good_rows(ra);
        rows.data = ra.at(ra.size());
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rows_data_outside, "rows",
                "rows whose data lies one byte past its buffer");
    }

    // (3) MASK dtype, residency and geometry: the same four ways, reported under mask names so a
    //     reader is not told the row table is at fault when the mask is.
    {
        auto mask = good_mask(ma);
        mask.type_is_float = false;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::mask_type, "mask",
                "mask of a width that is not a bias");
    }
    {
        auto mask = good_mask(ma);
        mask.buffer_is_host = false;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::mask_not_host, "mask",
                "mask in a device buffer");
    }
    {
        auto mask = good_mask(ma);
        mask.data = nullptr;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::mask_not_host, "mask",
                "mask with no data");
    }
    {
        auto mask = good_mask(ma);
        mask.buffer_size = 0;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::mask_buffer_geometry, "mask",
                "mask whose buffer reports no size");
    }
    {
        auto mask = good_mask(ma);
        mask.data = ma.at(ma.size());
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::mask_data_outside, "mask",
                "mask whose data lies past its buffer");
    }

    // (4) The shared column axis. This is the one disagreement the operator itself asserts, and it is
    //     the one a mismatch makes every column-to-row pairing arbitrary.
    {
        auto mask = good_mask(ma);
        mask.ne[0] = std::int64_t(kColumns) - 1;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::columns_differ, "mask.ne[0]",
                "mask one column shorter than the table");
    }

    // (5) THE SIGNED-DIMENSION CASE, which is the whole reason the check runs on int64.
    //
    //     A zero dimension makes `(n - 1)` underflow. A NEGATIVE one is worse and quieter: converted
    //     to `size_t` it becomes about 2^64, which passes every later comparison, so the failure
    //     would arrive as a gigantic count instead of an obviously broken description. Both are
    //     asserted because a fix that only handled zero would leave the quieter one.
    {
        auto rows = good_rows(ra);
        rows.ne[0] = 0;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::dim_not_positive, "ne",
                "a zero column count");
    }
    {
        auto rows = good_rows(ra);
        rows.ne[0] = -1;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::dim_not_positive, "ne",
                "a negative column count");
    }
    {
        auto mask = good_mask(ma);
        mask.ne[1] = -3;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::dim_not_positive, "ne",
                "a negative token count");
    }
    {
        auto rows = good_rows(ra);
        rows.ne[1] = 0;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::dim_not_positive, "ne",
                "a zero slice count");
    }

    // (6) Rank. The validator models rows as [n_columns, n_slices] and the mask as
    //     [n_columns, n_tokens, 1, n_slices]; an extra axis would be folded into the slice walk with
    //     nothing to notice. Asserted on each of the three axes that must be 1.
    {
        auto rows = good_rows(ra);
        rows.ne[2] = 2;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rank_unmodelled, "ne",
                "rows with ne[2] == 2");
    }
    {
        auto rows = good_rows(ra);
        rows.ne[3] = 2;
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::rank_unmodelled, "ne",
                "rows with ne[3] == 2");
    }
    {
        auto mask = good_mask(ma);
        mask.ne[2] = 4;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::rank_unmodelled, "ne",
                "mask with ne[2] == 4");
    }

    // (7) Slice agreement. The validator walks one mask slice per table slice at `s * nb[3]`, so a
    //     mask with fewer slices is read past its own last one -- and a mask with MORE would leave
    //     slices of the table unverified while the counts still looked consistent.
    {
        auto rows = good_rows(ra);
        rows.ne[1] = 2;
        rows.nb[1] = kColumns * sizeof(std::int32_t);
        refuses(rows, good_mask(ma), tq::reader::seam_contract_refusal::slices_differ, "mask.ne[3]",
                "two table slices against one mask slice");
    }
    {
        auto mask = good_mask(ma);
        mask.ne[3] = 2;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::slices_differ, "mask.ne[3]",
                "one table slice against two mask slices");
    }

    // (8) The mask's column stride. `rows` had this check; the mask did not, and the gap is silent:
    //     a wider nb[0] makes every entry be read from the gap between elements while the extent
    //     check -- which counts remaining buffer, not stride conformance -- still passes.
    {
        auto mask = good_mask(ma);
        mask.nb[0] = 4;
        refuses(good_rows(ra), mask, tq::reader::seam_contract_refusal::mask_not_contiguous,
                "mask.nb[0]", "mask with nb[0] == 4 at F16 width");
    }

    // (9) THE ORDER OF THE CHECKS, asserted rather than described.
    //
    //     Several of the cases above would ALSO trip a later check if the earlier one were absent --
    //     a mask with ne[2] == 4 and a mask whose data is outside its buffer are both "bad mask", and
    //     which one you hear first is a design decision. Residency comes first because "these bytes
    //     are not readable at all" is more basic than "these bytes describe the wrong axes": a
    //     message about rank would send the reader to a shape that turns out not to be the problem.
    {
        auto mask = good_mask(ma);
        mask.buffer_is_host = false;  // also: ne[2] != 1 below, and nb[0] wrong
        mask.ne[2] = 4;
        mask.nb[0] = 4;
        const tq::reader::seam_contract_report r =
            tq::reader::validate_seam_contract(good_rows(ra), mask);
        check(r.refusal == tq::reader::seam_contract_refusal::mask_not_host,
              std::string("ordering: residency is reported before rank and stride, got ") +
                  tq::reader::describe(r.refusal));
    }
    //     And rows' dtype precedes the mask entirely: a caller who got the row table wrong should not
    //     be told about the mask, because fixing the mask would change nothing.
    {
        auto rows = good_rows(ra);
        rows.type_is_i32 = false;
        auto mask = good_mask(ma);
        mask.type_is_float = false;
        const tq::reader::seam_contract_report r = tq::reader::validate_seam_contract(rows, mask);
        check(r.refusal == tq::reader::seam_contract_refusal::rows_not_i32,
              std::string("ordering: the row table is reported before the mask, got ") +
                  tq::reader::describe(r.refusal));
    }

    // (10) THE BOUND, NOT A LENGTH — the distinction this file exists to keep honest.
    //
    //      A tensor 512 bytes into a 4096-byte allocation reports 3584 readable bytes, and that number
    //      says "reading up to 3584 bytes from this pointer cannot fault". It does NOT say the tensor
    //      is 3584 bytes long: the allocation may hold other tensors after this one. A green test that
    //      only checked "the extent is positive" would let a later reader write `mask_bytes =
    //      readable` into a validator and believe the tensor was exactly that long -- so the number is
    //      asserted to be the REMAINDER, which is only true of a bound.
    {
        auto mask = good_mask(ma, /*offset=*/1024);
        const tq::reader::seam_contract_report r =
            tq::reader::validate_seam_contract(good_rows(ra), mask);
        check(r.refusal == tq::reader::seam_contract_refusal::ok,
              std::string("bound: a deeper offset is still accepted, got ") +
                  tq::reader::describe(r.refusal));
        check(r.mask_readable_bytes == ma.size() - 1024,
              "bound: the extent is the allocation REMAINDER (" +
                  std::to_string(ma.size() - 1024) + "), not the tensor's length; got " +
                  std::to_string(r.mask_readable_bytes));
        check(r.mask_readable_bytes > kTokens * kColumns * sizeof(std::uint16_t),
              "bound: and it is larger than the tensor needs, which is why it cannot stand in for "
              "a length");
    }

    if (failures == 0) {
        std::fprintf(stderr, "seam contract: OK\n");
        return 0;
    }
    std::fprintf(stderr, "seam contract: %d failure(s)\n", failures);
    return 1;
}
