// tq/reader/seq_cell_projection.h — the missing half of the kv_rows story: how a cache's
// occupancy bitmap becomes the row table the graph reads.
//
// WHY THIS EXISTS
// ---------------
// `build_attn_inp_attn_mask` (llama-graph.cpp:107) hands `build_attn_mha` an I32 tensor called
// `kv_rows`, and `llama_kv_cache::set_input_attn_mask` (llama-kv-cache.cpp:1907) is what fills it,
// by walking `llama_kv_cells`' own bitmap and position array. The reader lane could VERIFY that
// table (`rows_validation.h`) but could not PRODUCE one, so every test fed it a hand-built fixture
// and the one conversion upstream actually performs had no transcription to check against.
//
// `kv_rows` carries PHYSICAL cell indices only. There is no `(seq, pos)` in it, and upstream exports
// no `(seq, pos) -> cell` query — the identity lives in `llama_kv_cells::pos[]` and the occupancy
// lives in `llama_kv_cells::seq_bits[]`, both private to the cache. That is why the reader cannot
// reverse-engineer identity out of the row table, and why this file takes the bitmap as input.
//
// SAME ORDER AS UPSTREAM, ON PURPOSE
// ----------------------------------
// The output is exactly the order upstream's `get_seq_cells` produces: WORD-MAJOR. One 64-cell word
// at a time, ascending; within a word, the member sequences' bitsets are OR-ed together first, and
// the resulting word's set bits are emitted ascending — so entries come out in ascending CELL INDEX
// across the whole slice, not grouped by which sequence owns them, and a cell visible to two members
// appears ONCE (the OR erases the duplicate). The tail is filled with `-1`.
//
// The first transcription of this header said "sequences in the slice's own member order, then words
// ascending", and the code matched it: a sequence-major double loop. That reading is wrong twice —
// it duplicates shared cells and it groups the table by sequence — and it agrees with upstream only
// when `n_seq == 1`, which is why the hand-written fixture never caught it.
// `tq_reader.seq_cell_projection_ab` runs the real `llama_kv_cells::get_seq_cells` on the same state;
// it disagreed on 7 of 8 entries until the loop was rewritten word-major. That test, not this
// comment, is what pins the order.
//
// `p_min` is the sliding window's lower bound: a cell at a position BELOW it is not visible and is
// skipped, which is how upstream's per-slice `slices[s].p_min` filter is spelled here.
//
// THE PADDING IS NOT DECORATION
// -----------------------------
// The tail `-1` entries are what `ggml_flash_attn_ext_rows` clamps to row 0, so they are only safe
// when their mask column is `-inf`. This header produces the table; `validate_runtime_rows` is what
// refuses a table whose padding a token can still see. The two are meant to be used together, and
// the test files does exactly that.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tq::reader {

// The cache's state, in the coordinates `llama_kv_cells` keeps it.
//
// One slice, because `kv_rows` only exists when `n_stream == 1` (see `get_n_kv_rows`, which returns
// 0 for any other stream count). A multi-slice table would need one `p_min` per slice, and no such
// call site exists to test it against.
struct seq_cell_projection_inputs {
    const std::int32_t * pos = nullptr;        // `cells.pos[]`: cell -> position, length n_used
    const std::uint64_t * seq_bits = nullptr;  // `cells.seq_bits[]`: [n_seq][n_words] bitmaps
    const std::int32_t * seq_ids = nullptr;    // the slice's member sequence ids, in scan order
    std::size_t n_seq = 0;                     // number of member sequences in this slice
    std::size_t n_words = 0;                   // (used_max_p1 + 63) / 64, computed by the caller
    std::size_t n_used = 0;                    // `cells.used_max_p1()`
    std::size_t n_per_seq = 0;                  // `rows->ne[0]`: the PADDED width of one slice
    std::int32_t p_min = 0;                    // `slices[s].p_min`
};

struct seq_cell_projection {
    // Always exactly `n_per_seq` entries: `n_real` real cells, then `-1` to the end.
    std::vector<std::int32_t> rows;
    std::size_t n_real = 0;

    // A real cell arrived after the table was full. Upstream asserts (`GGML_ASSERT(dst < end)`);
    // refusing instead keeps a wrong-sized `n_per_seq` from being reported as a short table.
    bool overflow = false;
};

// Transcribes `llama_kv_cells::get_seq_cells` (llama-kv-cells.h:460) for one slice.
//
// A null `pos`, `seq_bits` or `seq_ids`, or a zero `n_seq`/`n_words`, yields a table of nothing but
// padding — which is what an empty slice really is — rather than a read through a null pointer.
seq_cell_projection project_seq_cells(const seq_cell_projection_inputs & in);

} // namespace tq::reader
