// tq/reader/upstream_view.h — the view that makes upstream's mask sufficient.
//
// THE PROBLEM THIS SOLVES
// -----------------------
// The physical emitter fills every cache position with `-INFINITY` and writes the named cells over
// it, so a mask can only NARROW that set (`-inf + finite` is `-inf`). Reproducing `build_attn_mha`
// therefore needs a view that names EVERY position the mask could leave visible — and the mask's
// values do not exist when the graph is built (`data == nullptr` on a `ggml_set_input` tensor,
// measured). A caller cannot enumerate "the visible ones" at that moment.
//
// There is exactly one view whose coverage is complete regardless of the mask's contents: the one
// that names every cache position. Then `view ∩ mask` is `mask`, the fill's `-inf` is overwritten
// everywhere, and visibility is decided by the mask alone — which is what upstream computes. The
// assertion stops being a promise about unknown values and becomes a statement about shape.
//
// WHY THIS IS ONLY VALID WHEN `kv_rows` IS ABSENT
// -----------------------------------------------
// With `kv_rows` present, upstream does not read position i from column i: it reads row
// `kv_rows[i]`, with `-1` meaning padding (ggml.h). The identity view below names positions
// 0..n_cell-1 in position order, which is a DIFFERENT mapping. So the caller must establish that
// `kv_rows == nullptr` before using this — and that check cannot live here, because this function
// only knows a count.
#pragma once

#include <cstdint>

#include <tq/reader/attention.h>

namespace tq::reader {

// One span naming every cache position, in ascending order.
//
// `effective_pos = 0` IS ONLY SAFE FOR AN EMITTER THAT DOES NOT ROTATE, and that is exactly one
// emitter: `emit_physical_attention`. An earlier revision of this comment said "the emitter uses it
// to place the span", which reads as if the value did something. It does not, there:
//
//   * `emit_physical_attention` takes `q` ALREADY ROTATED and performs no rotation at all. It reads
//     `effective_pos` only to check that every span agrees (`mixed_effective_pos`), so with one span
//     the check is vacuous and the value never reaches arithmetic. This is the right shape for the
//     upstream decode path for a reason worth stating: K in the cache was rotated when it was
//     WRITTEN, and Q arrives rotated at its own position, so attention over the cache needs no
//     per-key rotation. `effective_pos = 0` is inert rather than convenient.
//
//   * `emit_reader_attention` DOES rotate: with a `rope` callback it computes
//     `rope(ctx, q, span.effective_pos, user)` per span. Passing this view to it would rotate the
//     query to position 0 for every key, which for a decode at position p is simply the wrong
//     question -- and it would produce plausible numbers, because a rotation by 0 is the identity.
//     So this view must not be handed to that emitter with a non-null `rope`, nor to the core in
//     `attention.h`. Nothing here can enforce that (the view carries no rope), which is why it is a
//     precondition written down rather than a check.
//
// The cells are ascending and contiguous on purpose. Attention is permutation invariant, so any
// order would compute the same numbers; ascending keeps the softmax's column axis identical to the
// cache axis, which is what makes a mask indexed by position meaningful at all.
inline query_view full_coverage_view(std::int32_t n_cell) {
    query_view view;
    if (n_cell <= 0) return view;
    span_keys span;
    span.effective_pos = 0;
    span.cells.reserve(static_cast<std::size_t>(n_cell));
    for (std::int32_t i = 0; i < n_cell; ++i) span.cells.push_back(i);
    view.spans.push_back(std::move(span));
    return view;
}

} // namespace tq::reader
