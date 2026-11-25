#include "tq/reader/seq_cell_projection.h"

namespace tq::reader {

seq_cell_projection project_seq_cells(const seq_cell_projection_inputs & in) {
    seq_cell_projection out;

    // An empty slice is a table of padding, not an error: `get_seq_cells` fills `[dst, end)` with -1
    // and a slice with no members simply never advances `dst`.
    out.rows.assign(in.n_per_seq, -1);

    if (in.pos == nullptr || in.seq_bits == nullptr || in.seq_ids == nullptr) {
        return out;
    }
    if (in.n_seq == 0 || in.n_words == 0 || in.n_per_seq == 0) {
        return out;
    }

    std::size_t dst = 0;

    // WORD-MAJOR, WITH THE SEQUENCES OR-ED WITHIN A WORD.
    //
    // This is not a style choice; it is the semantic. Upstream's `get_seq_cells` (llama-kv-cells.h:460)
    // builds one `word` per 64-cell group by OR-ing every member sequence's bitset into it, then
    // walks that word's set bits ascending. Two consequences follow, and the sequence-major loop that
    // used to be here got both wrong:
    //
    //   1. A cell that belongs to TWO member sequences appears ONCE, at its index. Sequence-major
    //      emits it twice, because each sequence's scan finds it.
    //   2. The result is ordered by CELL INDEX across the whole slice, not grouped by sequence
    //      membership.
    //
    // The two agree only when the slice has exactly one sequence, which is why the hand-written
    // fixture test could not see it: `tq_reader.seq_cell_projection_ab` runs the real upstream
    // function, and its two-sequence case failed with 7 of 8 entries differing -- ours started at
    // cell 5, upstream at cell 1.
    for (std::size_t w = 0; w < in.n_words; ++w) {
        std::uint64_t b = 0;
        for (std::size_t s = 0; s < in.n_seq; ++s) {
            b |= in.seq_bits[s * in.n_words + w];
        }

        while (b != 0) {
            const unsigned ctz = static_cast<unsigned>(__builtin_ctzll(b));
            b &= b - 1;

            const std::size_t cell = w * 64 + ctz;

            // A bitmap word is rounded up to 64 bits, so the tail of the last word can name cells
            // that do not exist. Upstream never sets those bits; refusing them here means a
            // caller's over-wide `n_words` cannot turn into a read past `pos[]`.
            if (cell >= in.n_used) {
                continue;
            }

            // The sliding window's lower bound: below it the cell is not visible.
            if (in.pos[cell] < in.p_min) {
                continue;
            }

            if (dst >= in.n_per_seq) {
                out.overflow = true;
                return out;
            }

            out.rows[dst++] = static_cast<std::int32_t>(cell);
        }
    }

    out.n_real = dst;
    return out;
}

} // namespace tq::reader
