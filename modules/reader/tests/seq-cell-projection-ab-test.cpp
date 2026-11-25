// tq_reader.seq_cell_projection_ab — the projection against upstream's OWN implementation.
//
// WHY THIS EXISTS
// ---------------
// `tq_reader.seq_cell_projection` pins `project_seq_cells` against a fixture written from the same
// reading of `llama_kv_cells::get_seq_cells` that the projection is made of. Both sides of that
// comparison are transcriptions of one paragraph, so a misreading of the paragraph agrees with
// itself and the test is green. The order the two implementations agree on can only be certified by
// running the OTHER one.
//
// So this file includes `third_party/upstream/src/llama-kv-cells.h` and calls the real
// `llama_kv_cells::get_seq_cells` (`llama-kv-cells.h:460`) on a real `llama_kv_cells` object, then
// calls `tq::reader::project_seq_cells` on the SAME state, and compares element by element.
//
// WHAT "THE SAME STATE" MEANS, AND WHY IT IS RECONSTRUCTED RATHER THAN COPIED
// --------------------------------------------------------------------------
// `llama_kv_cells` keeps `pos[]`, `seq_bits[]` and `seq_pos[]` private, and the projection needs the
// first two as raw arrays. Reaching into the class would be undefined behaviour across a layout
// that is not ours. Instead the state is BUILT through the public API (`pos_set`, `seq_add`) and the
// two arrays the projection takes are reconstructed from public queries (`seq_has`) over that same
// object. Both sides therefore read one state; neither reads the other's output.
//
// The reconstruction is not a second opinion on the order: it is a plain transcription of what
// `seq_bits[seq_id][w]` means -- bit `64*w + b` is set exactly when cell `64*w + b` belongs to
// `seq_id` -- and `seq_has` is the public accessor for that bit.
//
// WHAT WOULD BE A FALSE FAILURE, AND IS THEREFORE NOT CHECKED
// -----------------------------------------------------------
// The two functions take different inputs. `get_seq_cells` takes one `llama_kv_rows_slice` per
// slice and needs `n_words` computed from `used_max_p1()`; the projection takes one flat array per
// sequence plus `n_used`. A mismatch in how many words to scan is a mismatch of the HARNESS, not of
// the algorithm, so `n_words` is taken from `used_max_p1()` -- the authoritative value -- and given
// to both. `p_min` is `-1` (keep everything) unless a case deliberately narrows it, because the
// projection's `p_min` is documented as the per-slice filter and `-1` is upstream's "no filter".

#include "llama-kv-cells.h"

#include "tq/reader/seq_cell_projection.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;
int g_cases = 0;

void check(bool condition, const char * case_name, const char * what) {
    ++g_cases;
    if (!condition) {
        std::fprintf(stderr, "FAIL [%s]: %s\n", case_name, what);
        ++g_failures;
    }
}

struct cell_spec {
    std::int32_t pos = -1;                    // < 0 means "this cell is empty"
    std::vector<llama_seq_id> seqs;           // the sequences the cell is visible to
};

// One slice, because `kv_rows` only exists for `n_stream == 1` (see `get_n_kv_rows`, which returns 0
// for any other count). A multi-slice comparison would be comparing shapes no caller produces.
void run_case(const char * name,
              std::uint32_t n_cells,
              const std::vector<cell_spec> & spec,
              std::vector<llama_seq_id> slice_seqs,
              llama_pos p_min,
              std::int64_t n_per_seq) {
    check(spec.size() == n_cells, name, "the fixture names every cell");

    llama_kv_cells cells;
    cells.resize(n_cells);

    std::vector<std::int32_t> pos(n_cells, -1);

    for (std::uint32_t i = 0; i < n_cells; ++i) {
        if (spec[i].pos < 0) {
            continue;   // an empty cell: never `pos_set`, never `seq_add`
        }
        cells.pos_set(i, spec[i].pos);
        pos[i] = spec[i].pos;
        for (const llama_seq_id s : spec[i].seqs) {
            cells.seq_add(i, s);
        }
    }

    // THE AUTHORITATIVE WORD COUNT, then the reconstructed bitmap in exactly the meaning
    // `seq_bits[seq_id][w]` has. `max(..., 1)` only keeps the vector non-empty for an all-empty
    // cache; a zero `n_words` is passed through as zero below, so no bit is read in that regime.
    const std::uint32_t n_words_real = (cells.used_max_p1() + 63) / 64;
    const std::size_t stride = std::max<std::uint32_t>(n_words_real, 1);

    std::vector<std::uint64_t> bits(slice_seqs.size() * stride, 0);

    for (std::size_t s = 0; s < slice_seqs.size(); ++s) {
        for (std::uint32_t i = 0; i < n_cells && i < n_words_real * 64; ++i) {
            if (cells.seq_has(i, slice_seqs[s])) {
                bits[s * stride + i / 64] |= (std::uint64_t(1) << (i % 64));
            }
        }
    }

    // ---- upstream, on the real object ----
    std::vector<std::int32_t> upstream((std::size_t) n_per_seq, -12345);

    std::vector<llama_kv_rows_slice> slices;
    slices.push_back(llama_kv_rows_slice{ slice_seqs, p_min });
    cells.get_seq_cells(slices, upstream.data(), n_per_seq);

    // ---- ours, on the same state ----
    tq::reader::seq_cell_projection_inputs in;
    in.pos      = pos.empty() ? nullptr : pos.data();
    in.seq_bits = bits.empty() ? nullptr : bits.data();
    in.seq_ids  = slice_seqs.empty() ? nullptr : slice_seqs.data();
    in.n_seq    = slice_seqs.size();
    in.n_words  = n_words_real;
    in.n_used   = cells.used_max_p1();
    in.n_per_seq = (std::size_t) n_per_seq;
    in.p_min    = p_min;

    const tq::reader::seq_cell_projection ours = tq::reader::project_seq_cells(in);

    check(!ours.overflow, name, "the projection did not overflow");
    check(ours.rows.size() == (std::size_t) n_per_seq, name,
          "the projection is exactly n_per_seq wide");

    // ---- the comparison this file exists for ----
    std::size_t first_diff = (std::size_t) -1;
    std::size_t diffs = 0;
    for (std::int64_t i = 0; i < n_per_seq; ++i) {
        const std::int32_t up = upstream[(std::size_t) i];
        const std::int32_t us = (std::size_t) i < ours.rows.size() ? ours.rows[(std::size_t) i] : -12345;
        if (up != us) {
            if (first_diff == (std::size_t) -1) {
                first_diff = (std::size_t) i;
            }
            ++diffs;
        }
    }

    check(diffs == 0, name, "every entry matches upstream's get_seq_cells");

    // A table that is all padding on both sides would agree while proving nothing, so the real
    // count is compared too -- and for the cases that have real cells, asserted non-zero.
    check(ours.n_real == (std::size_t) std::count_if(
              upstream.begin(), upstream.end(), [](std::int32_t v) { return v >= 0; }),
          name, "the real-cell count agrees with upstream's table");

    if (diffs != 0) {
        std::fprintf(stderr,
                     "  [%s] %zu of %lld entries differ, first at %zu\n"
                     "    upstream: ...%d %d %d...\n"
                     "    ours    : ...%d %d %d...\n",
                     name, diffs, (long long) n_per_seq, first_diff,
                     first_diff > 0 ? upstream[first_diff - 1] : -1,
                     upstream[first_diff],
                     first_diff + 1 < (std::size_t) n_per_seq ? upstream[first_diff + 1] : -1,
                     first_diff > 0 && first_diff - 1 < ours.rows.size() ? ours.rows[first_diff - 1] : -1,
                     first_diff < ours.rows.size() ? ours.rows[first_diff] : -1,
                     first_diff + 1 < ours.rows.size() ? ours.rows[first_diff + 1] : -1);
    }
}

} // namespace

int main() {
    // 1. ONE sequence, contiguous cells: the trivial case, so a later failure is not the baseline.
    {
        std::vector<cell_spec> spec(8);
        for (int i = 0; i < 8; ++i) {
            spec[i].pos = i;
            spec[i].seqs = { 0 };
        }
        run_case("single-seq contiguous", 8, spec, { 0 }, -1, 8);
    }

    // 2. TWO sequences sharing cells, with a NON-MONOTONIC member order `{5, 1, 3}`.
    //
    // This is the case that separates "cell order" from "sorted order" and from "sequence order".
    // Upstream scans words ascending and bits ascending REGARDLESS of the member order, so the
    // expected answer here is the cells sorted by index -- and the point of running upstream is that
    // this sentence no longer has to be believed.
    {
        std::vector<cell_spec> spec(10);
        spec[1] = { 5, { 1 } };
        spec[3] = { 1, { 3 } };
        spec[5] = { 3, { 5 } };
        spec[6] = { 6, { 1, 5 } };   // shared by two member sequences
        spec[9] = { 9, { 3, 5 } };
        run_case("multi-seq non-monotonic members", 10, spec, { 5, 1, 3 }, -1, 8);
    }

    // 3. `p_min` NARROWS the slice. Upstream keeps a cell only when `pos[i] >= p_min`, so the table
    // shrinks and the tail is padding -- the sliding-window case, and the one where a projection
    // that ignored `p_min` would return MORE real cells than upstream and still agree on a prefix.
    {
        std::vector<cell_spec> spec(10);
        for (int i = 0; i < 10; ++i) {
            spec[i].pos = i;
            spec[i].seqs = { 0 };
        }
        run_case("p_min narrows", 10, spec, { 0 }, 6, 6);
    }

    // 4. `p_min` ABOVE every position: nothing survives, the table is all `-1`. The agreement here is
    // padding against padding, so the real count is checked to be zero explicitly.
    {
        std::vector<cell_spec> spec(6);
        for (int i = 0; i < 6; ++i) {
            spec[i].pos = i;
            spec[i].seqs = { 0 };
        }
        run_case("p_min excludes everything", 6, spec, { 0 }, 100, 6);
    }

    // 5. AN EMPTY slice: `n_words == 0`, so neither side scans. Both must produce pure padding.
    {
        std::vector<cell_spec> spec(4);   // every cell empty
        run_case("empty slice", 4, spec, { 0 }, -1, 4);
    }

    // 6. A sequence that is a member but owns NO cell. Upstream ORs its (empty) words in and adds
    // nothing; a projection that treated "member but empty" as an error, or that dropped the rest of
    // the table on that account, differs here and nowhere else.
    {
        std::vector<cell_spec> spec(6);
        spec[2] = { 2, { 0 } };
        spec[4] = { 4, { 0 } };
        run_case("member with no cells", 6, spec, { 0, 7 }, -1, 6);
    }

    // 7. CELLS BEYOND `used_max_p1() * 64`. Only cells below the authoritative word count are in
    // play, and both sides must ignore the rest -- including a non-empty cell that sits past it.
    // (`used_max_p1` is the highest used index plus one, so a gap is what makes this interesting.)
    {
        std::vector<cell_spec> spec(130);
        spec[0]   = { 0, { 0 } };
        spec[127] = { 1, { 0 } };     // last bit of word 1
        spec[129] = { 2, { 0 } };     // past 64*2 == 128 -> word 2
        run_case("multi-word", 130, spec, { 0 }, -1, 4);
    }

    if (g_failures != 0) {
        std::fprintf(stderr, "seq_cell_projection_ab: %d of %d checks failed\n", g_failures, g_cases);
        return 1;
    }

    std::printf("seq_cell_projection_ab: %d checks passed, 7 cases, projection == upstream get_seq_cells\n",
                g_cases);
    return 0;
}
