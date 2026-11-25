// tq/reader/graph_emit.h — the reader's attention as graph nodes.
//
// attention.h owns the arithmetic: what the numbers must be, checked against a
// double-precision reference and against ggml's own kernels. This file owns the other
// half of the same feature: the nodes that make a graph produce those numbers, without
// one line changed in upstream and without a second copy of upstream's conventions.
//
// The seam in patches/herd_reader_attn.cpp ends with a contract for "whoever implements
// the emission": gather K rows by index, rotate Q per span, accumulate ONE softmax over
// the union of spans, reduce V, and return the shape build_attn returns today. This is
// that contract, written where it can be tested.
//
// Two deliberate non-goals:
//
//   * No rope convention. The rotation is a capability the caller injects (see
//     span_rope_fn) rather than a formula re-derived here. Upstream's rotation carries
//     yarn factors, an attention scale, and a dozen parameters; a second derivation would
//     be a second thing to keep true, and comparing it with itself would prove nothing.
//     In the product the injected rotation is upstream's ggml_rope_ext; in a test it can
//     be any function whose effect is known by hand.
//   * No ggml_context discovery. The context and the KV tensors arrive as arguments. A
//     module that reached into llm_graph_context's private layout to find ctx0 would make
//     every upstream refactor a silent ABI break. The wiring that can legitimately supply
//     the context is the caller's business; naming the argument is this module's.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <tq/reader/attention.h>

struct ggml_context;
struct ggml_tensor;

namespace tq::reader {

// Everything the emission needs from the graph it is being built into.
//
// Shapes are llama's, spelled out because a wrong one here is a silent wrong answer:
//
//   k   [head_dim, n_head_kv, n_cell]   key `cell`, kv head `h`, dim `d` at
//                                       d + head_dim*(h + n_head_kv*cell)
//   v   [head_dim, n_head_kv, n_cell]   the SAME indexing as k: d + head_dim*(h +
//                                       n_head_kv*cell). This is where the value axis
//                                       first went wrong, so it is worth stating precisely
//                                       what upstream does -- and it is not one thing.
//
//                                       Upstream has TWO V cache layouts and detects which
//                                       it has from the strides, never from the shape:
//
//                                         llama_kv_cache::get_v, llama-kv-cache.cpp:1430
//                                           if (!v_trans)   // v->nb[1] <= v->nb[2]
//                                             ne = (head_dim, n_head_kv, n_kv)
//                                             nb = (row(head_dim), row(head_dim*n_head_kv), ...)
//                                           else            // v->nb[1] >  v->nb[2]
//                                             ne = (n_kv, n_head_kv, head_dim)
//                                             nb = (row(n_kv*head_dim), row(n_kv), ...)
//
//                                         build_attn then re-derives the same bit
//                                         (`v_trans = v->nb[1] > v->nb[2]`, llama-graph.cpp:3282)
//                                         and transposes V for the multiply when it is set.
//
//                                       THIS MODULE READS BOTH. Each branch is converted to
//                                       the one form ggml_get_rows walks -- [head_dim,
//                                       n_cell], cells on ne1 -- and the difference between
//                                       them is a single transpose: the default layout already
//                                       has that form, the cell-major one is transposed into
//                                       it. Both are measured against the same reference
//                                       (tests/graph-emit-cache-layout-test.cpp, 4.5e-08 for
//                                       each). The branch is chosen by stride, never by shape,
//                                       because with n_cell == head_dim the two views have
//                                       identical ne and different bytes.
//
//                                       The tempting-but-false alternative is [n_cell,
//                                       head_dim, n_head_kv]: index cell + n_cell*(d +
//                                       head_dim*h). Under that declaration the per-head slice
//                                       reads another cell's values -- finite, plausible, and
//                                       wrong by ~0.7 on the same fixtures that pass when
//                                       only K is gathered. That was a real defect in this
//                                       module and the reason this paragraph exists.
//
//                                       The key axis is the one the reduction runs over,
//                                       which is why the final mul_mat needs v transposed;
//                                       that transpose is a step in this module, not a
//                                       property the input layout can be chosen to avoid.
//   q   [head_dim, 1, n_head]           one query row, n_head query heads
struct graph_inputs {
    ggml_context * ctx = nullptr;
    ggml_tensor * k    = nullptr;
    ggml_tensor * v    = nullptr;
    ggml_tensor * q    = nullptr;

    // UPSTREAM'S MASK, when the caller has one, instead of a visibility derived from the cell list.
    //
    // `build_attn_mha` receives `kq_mask` and hands it to `ggml_soft_max_ext`, so it is a per-column
    // BIAS over the cache positions, not a boolean: a finite value is a sliding-window or ALiBi term,
    // and `-inf` means invisible. Measured on the tiny model it is F16 (`type 1`), `ne [2048 1 1 1]`,
    // with `data` null at graph-build time (a `ggml_set_input` tensor). Nothing here dereferences
    // `data`; ggml reads the values at compute time.
    //
    // The physical emitter can consume it: it scatters the span scores into the full cache length and
    // runs ONE softmax over exactly `n_cell` columns — the same column space the mask is indexed by.
    //
    // BUT IT CANNOT WIDEN, and this is the part a caller must read before assuming equivalence with
    // upstream. The emitter fills every cache position with `-INFINITY` and then writes the named
    // cells over it, so a position no span names keeps its `-inf`, and `-inf + finite` is still
    // `-inf`. A mask can therefore only NARROW the visible set; it cannot open a cache position the
    // view left unnamed. Consequence: to reproduce upstream's visibility, the view must already name
    // every cell that is visible, and the mask then contributes the finite biases (sliding window,
    // ALiBi) on top. Supplying upstream's `kq_mask` over a view that names fewer cells than the mask
    // opens silently yields a DIFFERENT visibility than upstream — narrower, never wider. That is
    // asserted by `tq_reader.physical_mask`'s `a_bias_cannot_open_an_unnamed_position` case rather
    // than left to be discovered.
    //
    // The compact emitter cannot consume it at all: it softmaxes over the named cells only, so a
    // mask indexed by cache position has no axis to attach to. It refuses rather than reshapes.
    //
    // Null means "no mask": visibility then comes from the cell list, as before.
    ggml_tensor * mask = nullptr;

    // WHICH AUTHORITY DECIDES VISIBILITY, stated by the caller instead of assumed.
    //
    // The physical emitter fills every cache position with `-INFINITY` and then writes the named
    // cells over it, so a mask can only NARROW that set: `-inf + finite` is `-inf`. Reproducing
    // upstream therefore needs something this emitter cannot check -- the mask's VALUES, which do
    // not exist until `set_input_attn_mask` fills them at compute time (`data == nullptr` at build
    // time, measured). With the numbers unavailable, "the view names every cell the mask leaves
    // visible" is a claim only the caller can make.
    //
    // So the caller states an INTENT, and the emitter refuses to guess it. `unstated` is the
    // default and is REFUSED whenever a mask is supplied, because proceeding quietly would run with
    // one of two different semantics and report the result either way.
    //
    //   intersect -- visibility is the INTERSECTION of the view's cells and the mask's visible
    //                positions. This is what the emitter actually computes, and it is a legitimate
    //                request in its own right: a caller that wants exactly "these cells, further
    //                filtered by this bias" is not approximating anything. Narrowing here is the
    //                stated intent, not an accident.
    //
    // An earlier revision had a single value `view_names_all`, which asserted "the view names every
    // cell the mask leaves visible" -- i.e. that the intersection equals the mask's visible set, so
    // no narrowing happens. That claim is about VALUES the emitter cannot read, so the enum could
    // only ever record a promise; worse, the tests supplied an all-zero mask (every position
    // visible) over a view naming three cells and set it anyway, which made the stored claim false
    // while the run passed. A field whose only possible content is an unverified assertion is not a
    // contract, it is a comment with a type. `view_covers_finite_mask` below is the honest form of
    // that idea: a host-side CHECK a caller may run for itself, at the moment the values exist,
    // rather than a boolean the emitter is asked to take on faith.
    enum class mask_coverage {
        unstated,   // refuse if a mask is supplied: the intent would be a guess
        intersect,  // visibility is view ∩ mask; narrowing is intended
    };
    mask_coverage coverage = mask_coverage::unstated;

    // THE SPAN-INDEX HAND-OFF, for a context that does not allocate its own tensors.
    //
    // The emitter creates each span's index tensor ([n_cells] I32) because it is the node the
    // gather consumes, and in an allocating context it writes the cells into it immediately.
    // Under `no_alloc` -- the regime a product's graph runs in, where ggml allocates nothing
    // until a scheduler places the graph -- `tensor->data` is null, so there is nowhere to
    // write, and the previous version refused the whole emission with a generic
    // `graph_error: cannot allocate the index tensor`.
    //
    // That refusal was accurate and useless: it named a failure of ggml that was in fact the
    // caller's missing half of a contract. Upstream's answer is the graph-input pattern -- a
    // tensor created during the build, filled at `set_inputs()` time, after
    // `ggml_backend_sched_alloc_graph()` has given it memory (llama-graph.cpp:1590 is upstream's
    // own set_inputs). This sink is the caller's end of that pattern.
    //
    // THE ENTRIES COME BACK IN THE RESULT -- not through a callback, and not as pointers into the
    // view. Both of those were tried, and both were wrong for reasons worth keeping:
    //
    //   * a callback runs INSIDE the emission, so a caller that records into its own state has
    //     already changed that state if the callback throws. The batch version reduced the window
    //     to one call and then claimed "nothing was published" -- false in exactly the case that
    //     matters, and unfixable from this side, because the emitter cannot see the caller's state.
    //
    //   * a `const std::int32_t * cells` pointing into the view dangles the moment a caller passes
    //     a temporary, which is the natural way to write `emit_reader_attention(in, the_view(), s)`
    //     -- the temporary dies at the semicolon, and every entry handed back after that points at
    //     freed memory. A dangling pointer in a result is worse than no pointer: it reads as a
    //     convenience and is a use-after-free waiting for the first caller that trusts it.
    //
    // So the entries are values, the cells they name are OWNED by the result, and `cells_offset`
    // indexes into `emit_result::index_cells`. Offsets rather than pointers for a second reason:
    // they survive a copy or a move of the result, where an interior pointer would not.
    //
    // Each entry:
    //   `span`         position in `view.spans`, so the caller can match it to the view it passed.
    //                  It is the position, so `cells_at(span)` is the direct way to the cells;
    //   `index`        the tensor the emitter created, owned by `graph_inputs::ctx` -- it stays
    //                  valid exactly as long as that context does, and no longer. The result does
    //                  not own it and cannot: a ggml tensor belongs to the arena that made it;
    //   `cells_offset` where this span's cells begin in `emit_result::index_cells`;
    //   `n_cells`      how many, which always equals `index->ne[0]` and is never zero (an empty
    //                  span is refused with `empty_view`, so there is no such thing as a valid
    //                  entry that names no cells).
    //
    // There is no `owner`, and its absence is the point: see the note on emit_result::cells_of.
    struct index_entry {
        std::size_t   span = 0;
        ggml_tensor * index = nullptr;
        std::size_t   cells_offset = 0;
        std::size_t   n_cells = 0;
    };

    // WHETHER THE CALLER WANTS THEM. A flag, not a callback, for the reason above.
    //
    // `collect_indices == false` skips the recording entirely -- no vector, no reserve -- so an
    // emitter that supplies its own indices and wants nothing back pays nothing.
    //
    // A caller whose context allocates nothing and who DOES NOT ask gets
    // `index_data_unavailable`, because the gather would otherwise run on uninitialised indices --
    // the one outcome that produces plausible, wrong numbers.
    bool collect_indices = false;

    std::size_t n_head    = 0;   // query heads; from q->ne[2]
    std::size_t n_head_kv = 0;   // kv heads;    from k->ne[1]
    std::size_t head_dim  = 0;   // from q->ne[0]
    std::size_t n_cell    = 0;   // key positions in k/v; from k->ne[2]
};

// Rotates `q` for the span's position, returning a tensor of the same shape.
//
// Injected rather than implemented (see the header note). `user` is whatever the caller
// bound at wiring time -- upstream's rope parameters, or a test's own rotation.
using span_rope_fn = ggml_tensor * (*)(ggml_context * ctx, ggml_tensor * q,
                                       std::int32_t effective_pos, void * user);

// Why an emission declined. Named, because the seam's contract requires a refusal to be
// distinguishable from a success: an emitter that silently falls back to ordinary
// attention produces normal-looking numbers for a feature that did not run.
enum class emit_refusal {
    ok,
    no_context,          // ctx, k, v or q is null
    shape_mismatch,      // k/v/q disagree with n_head/n_head_kv/head_dim/n_cell
    multi_stream_view,   // k or v has more than one stream on ne3; this emits a single stream
    quantized_cache_unsupported, // k or v is not F32; the layouts here are measured on floats only
    bad_head_layout,     // n_head_kv does not divide n_head
    empty_view,          // no spans, or no keys in the view
    cell_out_of_range,   // an index outside [0, n_cell)
    duplicate_cell,      // the same cell twice: the softmax would weight it twice
    no_softmax_input,    // the union of spans was empty after gathering
    index_data_unavailable, // no_alloc and no index_sink: the indices have nowhere to be written
    // `graph_inputs::mask` cannot be honoured here, and ignoring it is the one outcome that must
    // not happen: the caller supplied a visibility rule and a bias, and this emitter's softmax runs
    // over the NAMED CELLS ONLY, so a mask indexed by cache position has no axis to attach to.
    // Refused by name rather than dropped, because a dropped mask is a run that reports the
    // visibility of a different configuration.
    mask_not_supported,
    graph_error,         // ggml returned null (allocation or shape assertion)
};

struct emit_result {
    ggml_tensor * out = nullptr;      // [head_dim, 1, n_head] when ok
    emit_refusal  refusal = emit_refusal::ok;
    std::size_t   n_visible = 0;      // keys the softmax saw: the union's size
    std::string   why;                // the refusal in words, for a caller's message

    // THE EMISSION'S IDENTITY -- GONE, and its absence is the design.
    //
    // It used to hold a per-emission id from a process-wide `std::atomic<std::uint64_t>` that
    // `cells_of` compared against. Two defects, both wrong for reasons that outlive the code:
    //
    //   1. IT WRAPS. `fetch_add(1)` returns to 0 after 2^64 emissions, and 0 was the value reserved
    //      for "no emission" -- so after the wrap a hand-made entry matched a live result. ABA in
    //      the one field whose entire job is telling results apart. "2^64 will not happen" is a
    //      claim about run time, not a proof; the whiteboard's epoch was rejected on exactly that
    //      ground, and accepting it here would have been an inconsistency rather than a decision.
    //      Making the counter refuse at exhaustion would have turned a wrap into a permanent
    //      emitter failure, which is a worse defect than the one it fixes.
    //
    //   2. IT DESCRIBED STATE, NOT VALUE. `emit_result` is copyable, and the copies are independent
    //      from that moment: copy a result, edit the copy's `index_cells`, hand an entry from the
    //      copy to the original -- the ids match, and the cells differ. The id answered "same
    //      emission", which was true and useless; the caller needed "same cells", which was false.
    //
    // What remains is a LOOKUP BY VALUE, and the distinction matters because a previous revision of
    // this comment claimed more than the code does:
    //
    //   TRUE   `cells_of` returns the cells this result holds for an entry whose fields match one of
    //          its own entries. Equal fields name the same offset and length IN THIS BUFFER, so the
    //          answer is correct for any entry that matches, whichever object it was copied from.
    //
    //   FALSE  "a copy that diverges stops matching". A copy of an `emit_result` is an independent
    //          value, and diverging it means editing its `index_cells` -- which changes no entry's
    //          fields. Both copies therefore keep matching, and each answers from ITS OWN buffer.
    //          That is not a failure to detect divergence; it is what independent copies mean.
    //
    //   FALSE  "cells_of tells you whether an entry came from here". It cannot, and no field kept in
    //          this struct could make it: an entry is data, so a foreign entry with equal fields is
    //          indistinguishable by construction -- and if the fields are equal it names the same
    //          offset and length, so answering is correct regardless of origin.
    //
    // So there is no cross-result identity guarantee to state, and none is claimed. A caller that
    // wants cells should use `cells_at(position)`: a position cannot name an entry belonging to
    // another result, and it cannot be confused with a copy's edit. `cells_of` exists for the case
    // where a caller holds an entry and has lost its position, and it is documented as a value
    // lookup rather than a check.

    // Every span's index tensor, in `view.spans` order, when the caller asked for them.
    //
    // Delivered as data, after the emitter returns: a caller that must roll back has not started
    // when it receives these, which is the only ordering under which its own commit can be atomic.
    // See `graph_inputs::collect_indices` for why the callback was removed.
    //
    // Empty when `collect_indices` was false, and always empty on a refusal -- a refusal hands back
    // nothing the caller might mistake for a usable set.
    std::vector<graph_inputs::index_entry> indices;

    // The cells those entries name, owned here so that no entry can dangle.
    //
    // A `const std::int32_t *` into the caller's view would be valid exactly as long as that view
    // lives, and `emit_reader_attention(in, the_view(), s)` -- the natural way to call it -- gives
    // it until the semicolon. The entries are read afterwards, so the pointer would be a
    // use-after-free on the first line of caller code that touched it.
    //
    // One contiguous buffer, indexed by `index_entry::cells_offset`: the whole set costs a single
    // allocation however many spans there are. Offsets also survive a copy or move of the result,
    // which an interior pointer would not -- and so does the id, so a copied result still answers
    // for its own entries.
    std::vector<std::int32_t> index_cells;

    // THE CELLS OF AN ENTRY, FOUND BY VALUE. Not an identity check, and not presented as one.
    //
    // Returns this result's cells for an entry whose fields equal one of `indices`, else nullptr.
    // Equal fields mean the entry names an offset and length this buffer holds, so the answer is
    // correct for any such entry whichever object it was copied from. What it cannot tell you is
    // where the entry came from -- an entry is plain data, so a foreign entry with equal fields is
    // indistinguishable, and answering it is still correct because the fields name the same place.
    // Two copies of a result are independent values: diverging one changes no entry, so both keep
    // answering, each from its own buffer. See the note on emit_result for why no stronger claim
    // is made.
    //
    // Refusals:
    //
    //   * not found among `indices` -- the fields match no stored entry. A default-constructed
    //     entry cannot match either: no entry here has a null `index`, because the emitter only
    //     records tensors it just created.
    //   * `entry.n_cells == 0` -- there is no such valid entry. An empty span is refused with
    //     `empty_view`, so a zero here means the fields were never filled in. It also avoids
    //     `index_cells.data() + 0` on an empty buffer, a pointer no caller can use and none can
    //     distinguish from a successful lookup.
    //   * the STORED entry's range is checked before use, so even a match cannot read past
    //     `index_cells`. Unreachable while the emitter fills both containers consistently; it stays
    //     because the cost is two comparisons and the alternative is a heap overread if that ever
    //     stops being true.
    //
    // `cells_at(position)` is the same answer without the scan, and it is the accessor to prefer:
    // a position cannot name another result's entry, and it cannot be a copy's edit.
    const std::int32_t * cells_of(const graph_inputs::index_entry & entry) const {
        if (entry.n_cells == 0) {
            return nullptr;
        }
        for (const graph_inputs::index_entry & mine : indices) {
            if (mine.span != entry.span || mine.index != entry.index ||
                mine.cells_offset != entry.cells_offset || mine.n_cells != entry.n_cells) {
                continue;
            }
            // Found: `mine` is the stored one, so its range is the one to trust.
            if (mine.cells_offset > index_cells.size() ||
                mine.n_cells > index_cells.size() - mine.cells_offset) {
                return nullptr;
            }
            return index_cells.data() + mine.cells_offset;
        }
        return nullptr;
    }

    // The cells of the `position`-th entry, bounds-checked. The preferred accessor: a position
    // cannot name an entry that belongs to another result, so the mismatch class does not arise.
    const std::int32_t * cells_at(std::size_t position) const {
        if (position >= indices.size()) {
            return nullptr;
        }
        return cells_of(indices[position]);
    }
};

const char * describe(emit_refusal refusal);

// Emits the reader's attention for one query row over one view.
//
// The returned tensor is the same shape build_attn returns today -- [head_dim, 1,
// n_head] -- so a caller that swaps the two cannot tell them apart by shape, which is
// what "the caller is unaware of anything new" means in the seam contract.
//
// Ordering matters and is not arbitrary: the softmax spans the union of the spans in
// view order, and the value reduction reads cells in that same order. One order for
// both is what makes a key from a later span compete with one from an earlier span in a
// single normalization -- the property that separates this from ordinary attention.
emit_result emit_reader_attention(const graph_inputs & inputs,
                                  const query_view & view,
                                  float kq_scale,
                                  const span_rope_fn & rope = nullptr,
                                  void * rope_user = nullptr);

} // namespace tq::reader
