// Which result answers for which entry -- without any process-wide counter.
//
// The problem, stated as a value question rather than a provenance one:
//
//   result A: 5 cells total, entry {offset 0, n_cells 3}
//   result B: 9 cells total, entry from A passed to B.cells_of()
//
// A bounds check alone accepts that entry (0 <= 9, 3 <= 9) and returns B's first three cells:
// wrong numbers, no crash, nothing to notice. The previous fix gave every emission an id from a
// process-wide `std::atomic<std::uint64_t>` and compared it -- which was wrong twice over:
//
//   * the counter wraps. After 2^64 emissions `fetch_add(1)` returns 0, the value reserved for
//     "no emission", so a hand-made entry matches a live result. "It will not happen" is a claim
//     about run time, not a proof; the whiteboard's epoch was rejected on exactly that ground, and
//     so is this one. Making the counter refuse at exhaustion would trade a wrap for a permanent
//     emitter failure, which is worse.
//
//   * an id describes the EMISSION, and `emit_result` is copyable. Copy a result, edit the copy's
//     `index_cells`, hand an entry from the copy to the original: the ids match and the cells
//     differ. The id says "same emission" -- true, and useless. The caller needs "same cells".
//
// So identity is derived from the value: `cells_of` finds the entry among this result's own entries
// by comparing every field, and answers from the stored one. That test needs no state, has nothing
// to exhaust, and is sound for a reason that does not depend on where the entry came from: if the
// fields match, they name an offset and length this result holds.
//
// What is asserted here:
//
//   1. an entry resolves through its own result, with the cells the view named;
//   2. a foreign entry is refused even when its offsets are IN RANGE for the other result -- the
//      fixture proves that adversary explicitly, because an out-of-range foreign entry would pass
//      for the wrong reason;
//   3. two copies of a result are INDEPENDENT values: diverging one's `index_cells` changes no
//      entry's fields, so both keep answering and each answers from its own buffer. That is not a
//      failure to detect divergence -- it is what independent copies mean, and the earlier revision
//      of this file asserted the opposite in prose while its own assertions said this;
//   4. a copy that agrees answers identically (a value-equal entry names the same cells);
//   5. a default-constructed entry and a zero-cell entry resolve to nothing;
//   6. `cells_at(position)` agrees with `cells_of` over the whole set, and refuses out of range;
//   7. there is no counter to exhaust: many emissions in a row all answer correctly, and an entry
//      from an early one is still refused by a later one.
//
// (7) is the exhaustion test, and the point is that it can be a loop rather than an argument: with
// no counter there is no boundary at 2^64, so running a thousand emissions and re-checking an early
// entry establishes exactly what the old design could only assert about.
#include <tq/reader/attention.h>
#include <tq/reader/graph_emit.h>

#include <ggml.h>

#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int kQHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 8;
constexpr int kCells = 12;

// One allocating context for every emission in this file: the entries' identity question is about
// values, and separate contexts would make "foreign" trivially detectable by tensor pointer alone.
struct fixture {
    ggml_context * ctx = nullptr;

    fixture() {
        ctx = ggml_init({64u * 1024u * 1024u, nullptr, /*no_alloc=*/false});
    }
    ~fixture() {
        if (ctx) ggml_free(ctx);
    }
    fixture(const fixture &) = delete;
    fixture & operator=(const fixture &) = delete;

    // A cache of `nc` cells, zeroed so the arithmetic is not what this file measures.
    ggml_tensor * cache(int nc) {
        ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, nc);
        for (int i = 0; i < kHeadDim * kKvHeads * nc; ++i) {
            static_cast<float *>(t->data)[i] = 0.0f;
        }
        return t;
    }

    tq::reader::graph_inputs inputs(int nc) {
        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = cache(nc);
        in.v = cache(nc);
        in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        for (int i = 0; i < kHeadDim * kQHeads; ++i) {
            static_cast<float *>(in.q->data)[i] = 0.0f;
        }
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim;
        in.n_cell = std::size_t(nc);
        in.collect_indices = true;
        return in;
    }
};

tq::reader::query_view view_of(std::vector<std::vector<std::int32_t>> spans) {
    tq::reader::query_view view;
    std::int32_t pos = 1;
    for (auto & cells : spans) {
        tq::reader::span_keys span;
        span.effective_pos = pos++;
        span.cells = std::move(cells);
        view.spans.push_back(std::move(span));
    }
    return view;
}

} // namespace

int main() {
    fixture f;

    // ---- 1 and 2: own entries resolve; foreign entries are refused IN RANGE ----
    {
        tq::reader::graph_inputs in = f.inputs(kCells);
        auto A = tq::reader::emit_reader_attention(
            in, view_of({{0, 3, 7}, {9, 1}}), 0.70710678f, nullptr);
        check(A.refusal == tq::reader::emit_refusal::ok, "A emitted");
        check(A.indices.size() == 2, "A has one entry per span");

        auto B = tq::reader::emit_reader_attention(
            in, view_of({{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}), 0.70710678f, nullptr);
        check(B.refusal == tq::reader::emit_refusal::ok, "B emitted");

        for (std::size_t i = 0; i < A.indices.size(); ++i) {
            const std::int32_t * cells = A.cells_of(A.indices[i]);
            check(cells != nullptr, "A's own entry resolves");
            const std::vector<std::int32_t> expected = (i == 0 ? std::vector<std::int32_t>{0, 3, 7}
                                                              : std::vector<std::int32_t>{9, 1});
            if (cells) {
                bool same = true;
                for (std::size_t k = 0; k < expected.size(); ++k) {
                    same = same && cells[k] == expected[k];
                }
                check(same, "and the cells are the ones the view named, in order");
            }
        }

        // The adversary, proved before it is used: A's first entry has offset 0 and 3 cells, and
        // B holds 12, so B's bounds would accept it. What makes B refuse is that the FIELDS differ
        // (a different tensor pointer, a different length) -- not any notion of provenance.
        const auto & foreign = A.indices[0];
        check(foreign.cells_offset + foreign.n_cells <= B.index_cells.size(),
              "fixture: the foreign entry IS in range for B, so refusing it cannot be a bounds test");
        check(B.cells_of(foreign) == nullptr,
              "a foreign entry whose fields match none of B's is refused by the lookup, not "
              "answered from the wrong buffer");

        // AND THE OTHER HALF OF THE CONTRACT, which is what keeps the doc honest: an entry whose
        // fields DO equal one of B's is answered, from B's buffer, whatever object it was copied
        // from. A test that only asserted refusals would leave the claim "it is a value lookup"
        // unfalsifiable.
        tq::reader::graph_inputs::index_entry same_fields = B.indices[0];
        const std::int32_t * same = B.cells_of(same_fields);
        check(same == B.cells_at(0),
              "an entry equal by value to B's own resolves to B's own cells -- equal fields name "
              "the same place, so provenance is neither checked nor needed");

        // ---- 3: copies are independent, in both directions ----
        //
        // The lookup is by value, so an entry's fields are all it takes to answer. Diverging a copy
        // means editing `index_cells`, which changes no field -- so both copies match, and each
        // answers from ITS OWN buffer. That is the honest statement of the contract; the previous
        // revision of this file claimed a diverged copy "stops answering", which its own assertions
        // contradicted.
        tq::reader::emit_result copy = A;
        check(copy.cells_of(copy.indices[0]) != nullptr,
              "an unmodified copy answers for its own entries");
        copy.index_cells[0] = 12345;
        const std::int32_t * from_copy = copy.cells_of(copy.indices[0]);
        check(from_copy != nullptr && from_copy[0] == 12345,
              "a diverged copy answers with ITS buffer's cells: the entry's fields did not change, "
              "so the lookup still matches -- copies are independent values, not shared state");
        const std::int32_t * from_original = A.cells_of(copy.indices[0]);
        check(from_original != nullptr && from_original[0] == 0,
              "while the original answers from its own buffer -- one entry value, two results, two "
              "answers, and neither is a lie about the other");

        // A STORED entry altered to claim more cells than the buffer holds must not read past it.
        //
        // This replaces a case whose message claimed the lookup decides it "because the fields
        // differ", which was wrong: `edited.indices[0]` IS the stored entry, so mutating it makes
        // the fields equal by construction and the lookup matches itself. What actually stops the
        // read is the range check against `index_cells` -- and that is worth pinning, because it is
        // the one guard a caller can trigger by editing a result it owns.
        tq::reader::emit_result edited = A;
        edited.indices[0].n_cells = 99;
        check(edited.cells_of(edited.indices[0]) == nullptr,
              "a stored entry claiming 99 cells is refused by the range check, not read past the "
              "buffer: a caller may own the result, but it may not make the lookup overrun");

        // ---- 5: nothing, and zero cells ----
        check(A.cells_of(tq::reader::graph_inputs::index_entry{}) == nullptr,
              "a default-constructed entry resolves to nothing");
        tq::reader::graph_inputs::index_entry zero = A.indices[0];
        zero.n_cells = 0;
        check(A.cells_of(zero) == nullptr,
              "a zero-cell entry resolves to nothing, so no caller sees a null-plus-zero pointer");

        // ---- 6: cells_at agrees with cells_of ----
        for (std::size_t i = 0; i < A.indices.size(); ++i) {
            check(A.cells_at(i) == A.cells_of(A.indices[i]),
                  "cells_at(position) and cells_of(entry) agree for the same entry");
        }
        check(A.cells_at(A.indices.size()) == nullptr, "cells_at refuses a position past the end");

        // ---- 7: no counter to exhaust ----
        //
        // The entries from A are held across many further emissions and re-checked at the end. With
        // the old design this tested the increment; with none it tests that nothing accumulated.
        tq::reader::query_view small = view_of({{0, 1}});
        bool all_ok = true;
        for (int round = 0; round < 1000; ++round) {
            tq::reader::graph_inputs loop_in = f.inputs(kCells);
            const auto r = tq::reader::emit_reader_attention(loop_in, small, 0.70710678f, nullptr);
            all_ok = all_ok && r.refusal == tq::reader::emit_refusal::ok &&
                     r.cells_of(r.indices[0]) != nullptr;
        }
        check(all_ok, "a thousand emissions in a row all resolve their own entries");
        check(A.cells_of(A.indices[0]) != nullptr && A.cells_of(A.indices[0])[0] == 0,
              "and the first emission's entry still resolves, unchanged: nothing wraps, nothing is "
              "exhausted, because there is no counter");
        check(B.cells_of(A.indices[0]) == nullptr,
              "while it is still refused by the result that did not produce it");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "entry cells: a lookup by value in this result's buffer; copies are independent, "
                 "and no provenance is checked or needed\n");
    return 0;
}
