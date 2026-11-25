// tq/reader/graph_emit_physical.h — the reader's attention in the form the fork builds: scatter the
// span scores into the PHYSICAL cache length with -INFINITY everywhere else, ONE global softmax over
// that length, then the native value reduction.
//
// Why a second emitter rather than a switch on the first: `graph_emit.h` emits the COMPACT form
// (gather the named cells, softmax over exactly those). The fork does neither, and its own comment
// says why — "Preserve the native physical reduction order and padding. A tiny change in a compact
// sum can cross a later quantization boundary." So the compact emitter is not an equal substitute,
// and this file is the difference, as nodes.
//
// Reproduced from `build_attn_reader` (llama-graph.cpp), in its order:
//   k_flat  = reshape_2d(cont(k), head_dim*n_head_kv, n_cell)
//   per span: get_rows(k_flat, cells) -> reshape_3d(hd, kvh, S) -> permute(0,2,1,3)
//             mul_mat(keys, q)                                   // [S, 1, n_head]
//   rows    = reshape_3d(concat(span scores, dim 0), 1, n_visible, n_head)
//   physical= fill(new_tensor_3d(F32, 1, n_cell, n_head), -INFINITY)
//   physical= set_rows(physical, rows, cells_index)               // scatter into cache positions
//   weights = soft_max_ext(reshape_3d(physical, n_cell, 1, n_head), nullptr, kq_scale, 0.0f)
//   out     = mul_mat(permute(v,0,2,1,3)[, cont(transpose(...))], weights)   // [hd, 1, n_head]
//
// Refused rather than approximated, each by name:
//   * a per-span RoPE rebuild when a span's effective position differs from the first one's — the
//     fork re-creates the query from the pre-rope source with the rope parameters, which is a
//     different input surface. Rotating once and using it everywhere would be a wrong answer.
//   * Hadamard rotations, soft-cap, attention sinks, LoRA output scaling: distinct node sequences.
//
// TWO REGIMES, TWO ANSWERS.
//
//   no_alloc    -> EMITS. Every node is one TENSOR object holding no data, and ggml's own allocator
//                  makes that a constant, so the arena cost is `objects * D` with D measured from the
//                  allocator rather than assumed. See "THE ARENA BOUND" below.
//   allocating  -> REFUSES (arena_admission_unproven). An allocating graph also owns data regions —
//                  `cont` copies the cache, the concat chain reallocates the union once per span,
//                  `fill`/`soft_max` are full n_cell*n_head arrays — and the cost of those is not
//                  derivable from this file. Three revisions estimated it; all three were low, the
//                  last by a term large enough to abort ggml at a 16 KiB arena after its own guard
//                  passed. So that regime is refused by name, before any allocation, instead of
//                  approximated.
//
// THE ARENA BOUND, derived from the pinned ggml rather than tuned:
//
//   ggml_new_tensor_impl:  obj_alloc_size = (view_src == NULL && !ctx->no_alloc) ? data_size : 0;
//                          ggml_new_object(ctx, TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
//   ggml_new_object:       size_needed = GGML_PAD(size, GGML_MEM_ALIGN);
//                          offs = cur_end + GGML_OBJECT_SIZE;  size = size_needed;
//   ggml_used_mem:         last->offs + last->size
//
// Under no_alloc every object is therefore `GGML_OBJECT_SIZE + GGML_PAD(GGML_TENSOR_SIZE, ALIGN)`
// bytes — the same for a 1-D index tensor, a 3-D view, a `cont` and a `mul_mat` result, because the
// size depends only on the header struct and on zero data. That is algebra, not a sample: no tensor
// shape, type or rank can enter it. The only two other object kinds in ggml are the GRAPH and
// WORK_BUFFER objects (`ggml_new_graph*`, `ggml_new_buffer`), and no op this file calls creates
// either — the one WORK_BUFFER call site is inside the CPU backend's own compute plan, on the
// backend's context, not the caller's.
//
// D is MEASURED at runtime (`ggml_tensor_overhead()`-style delta in a scratch context) instead of
// hardcoded, because GGML_MEM_ALIGN is not public header surface; a computation, not a guess.
//
// WHAT THE BOUND DOES NOT COVER, and the reason `bytes_required` is in the result: the caller's own
// scaffolding. A caller that then calls `ggml_new_graph` on the SAME context allocates a GRAPH object
// whose size dwarfs this whole emission (a default graph is GGML_DEFAULT_GRAPH_SIZE nodes). Measured:
// with the fixture arena set to 8, 12 or 16 KiB the test process dies inside its own `ggml_new_graph`,
// not in the emitter. The emitter cannot know a caller's graph size, so it reports its exact cost and
// the caller adds its own — `free_at_entry >= bytes_required + your_own_scaffolding`.
//
// The N+1 index handoff is implemented, not advertised: `indices` carries one entry per span (the
// gathers) plus one for the scatter, which is the same shape the fork uses, where every
// `reader.indices` is a graph input. Under no_alloc nothing has storage at build time, so the write
// happens after the scheduler places the graph — and it is `publish_index_inputs` that performs it,
// covering all N+1 tensors in one call. Do not hand-roll that loop: a caller that writes the scatter
// and forgets a gather gets no error, only wrong indices.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <ggml.h>

#include <tq/reader/attention.h>
#include <tq/reader/graph_emit.h>

namespace tq::reader {

enum class physical_emit_refusal {
    ok,
    no_context,
    bad_shape,
    bad_head_layout,
    unsupported_cache_type,
    multi_stream,
    // A mask was supplied that this emission cannot honour: wrong type, or an axis that does not
    // correspond to the cache columns the softmax runs over. Refused rather than broadcast, because
    // a mask silently reinterpreted along the wrong axis produces a plausible number.
    bad_mask,
    empty_view,
    empty_span,
    cell_out_of_range,
    duplicate_cell,
    mixed_effective_pos,
    v_layout_cell_major,
    index_data_unavailable,
    no_alloc_context,
    // An ALLOCATING context: this emitter can bound the objects it creates, but not the data regions
    // an allocating graph also owns, and an estimate that is low ends the process inside
    // ggml_new_object. Refused before the first allocation; a no_alloc context is admitted.
    arena_admission_unproven,
    no_room,
    graph_error,
};

const char * describe(physical_emit_refusal refusal);

struct physical_emit_result {
    physical_emit_refusal refusal = physical_emit_refusal::ok;
    std::string           why;

    // [head_dim, 1, n_head], indexed d + head_dim*head — the layout the domain core returns, so a
    // caller compares values rather than axis conventions.
    //
    // THIS emitter's linear order is measured, not assumed: `graph-emit-physical-test` compares these
    // values against an independent hand-written double reference that indexes `out[d + head_dim*h]`,
    // with non-trivial data and hidden cells; a different order could not agree.
    //
    // Upstream `build_attn_mha` should return the same order for a single token, and that half is
    // REPRODUCED, not observed: `upstream-attention-contrast-test` replays upstream's tail (mul_mat,
    // soft_max_ext with a -INFINITY mask, mul_mat, permute `(0,2,1,3)`, cont_2d) over a fixture whose
    // k/v it converts by hand, and agrees with this emitter value by value (0.0 for one span,
    // 2.98e-08 for two). Nothing in this repository has ever read a tensor that llama.cpp actually
    // returned, so the agreement covers "upstream's operators in upstream's order", not the live
    // call. What the live call was observed to return is one shape on one model: rank 1
    // `[32 1 1 1]`, F32, 3-layer tiny transformer, single token. That is not a universal layout.
    //
    // The interesting part is WHY the permute does not matter: it is equivalent to the identity when
    // `n_tokens == 1`, because `ggml_cont_2d` copies in view order and the middle dim is 1, so both
    // collapse to `d + head_dim*head`. A mutation to identity permute produces bit-identical output.
    // The layout therefore rests on `cont_2d`'s view order and on `n_tokens == 1`, and the permute
    // only starts to mean something once a second token arrives -- which is exactly the case this
    // lane has not covered yet.
    //
    // Still absent: a REPLACEMENT. That test recomputes and compares a tensor; it installs no seam,
    // touches no llama.cpp code, and says nothing about whether the hook site is usable.
    ggml_tensor * out = nullptr;

    // Post-softmax weights over the full cache, [n_cell, 1, n_head]. Exposed because it is the
    // strongest single comparison: hidden cells must be exactly zero.
    ggml_tensor * weights = nullptr;

    // One entry per span (its gather) plus one for the scatter, in emitter order: entry i's `span`
    // is i for a gather and the span count for the scatter. `index_cells` is the concatenation those
    // gathers tile and the scatter covers whole.
    //
    // Both are the CALLER'S to hold between emit and publish, and `publish_index_inputs` re-checks
    // their SHAPE — not their origin, and not the values in `index_cells`. The previous revision's
    // field comment here said "one entry, so `span` is 0", which was already false when it was
    // written: the emitter has produced N+1 entries since the per-span gathers were added.
    std::vector<graph_inputs::index_entry> indices;
    std::vector<std::int32_t>              index_cells;

    std::size_t n_visible = 0;
    std::size_t n_cell = 0;
    std::size_t n_head = 0;
    std::size_t head_dim = 0;

    // The arena this emission consumed, in bytes, as bounded before the first ggml call. A caller
    // must keep its OWN scaffolding — `ggml_new_graph` on the same context allocates a GRAPH object
    // far larger than this — inside `free_at_entry - bytes_required`. Reported instead of documented
    // because a number a caller cannot read is a number a caller cannot budget with.
    std::size_t bytes_required = 0;

    // The per-object cost the bound used, and the object count it charged. Together they let a caller
    // re-derive `bytes_required` against a different ggml build instead of trusting this one.
    std::size_t object_cost = 0;
    std::size_t objects_charged = 0;

    // NOTHING HERE RECORDS WHERE THE RESULT CAME FROM, and that absence is the correction.
    //
    // The previous revision carried `index_entry_count`, written by the emitter and compared at
    // publish time, claiming a hand-assembled result would therefore be refused. It would not: the
    // field is public and mutable, so the same caller who erases a gather can write a matching
    // count, and the check would then be measuring the caller's intent rather than the emitter's
    // output. A guard an adversary passes by assignment is not a guard.
    //
    // What `publish_index_inputs` actually provides is STRUCTURAL validation: the entries must tile
    // `index_cells` in emitter order and end with a scatter covering all of it; every tensor must be
    // a placed I32 graph input whose `ne[0]` equals its entry's `n_cells`; and nothing is written
    // until all of that holds. That catches the mistakes a programmer makes — a dropped entry, a
    // re-pointed offset, a publish before allocation — which is what an API of this shape can
    // honestly catch.
    //
    // It cannot prove provenance, and it cannot detect an edited cell value, because `index_cells`
    // is the caller's buffer. Provenance would need an identity the caller cannot write — a private
    // token minted by the emitter and exposed read-only — which is a change to this type's
    // interface, not another field on it.
};

// THE MISSING HALF OF THE HAND-OFF, and the reason it is one function instead of a documented loop.
//
// The fork's contract has two parts, and this emitter until now had only the first:
//
//   build   `reader.indices = ggml_new_tensor_1d(...); ggml_set_input(reader.indices);`
//           -- plus one such tensor per span. Done here: N+1 tensors, each marked input.
//   set_inputs  `ggml_backend_tensor_set(reader.indices, reader.values.data(), 0, n*4);`
//           -- after `ggml_backend_sched_alloc_graph()` gave every input its memory.
//           NOT done here, and that was the gap.
//
// A caller left to write those N+1 tensors by hand has a failure mode with no symptom: write the
// scatter and forget the gathers, and ggml reads uninitialized memory as row indices. No error, no
// warning -- a gather from the wrong cells, or a read past the cache. The same shape of mistake as
// "only two of the three inputs were set". So the write is one call that covers every entry or
// refuses before touching the device.
//
// VALIDATION PRECEDES EVERY WRITE. All entries are checked first -- tensor present, marked input,
// storage placed, `ne[0]` equal to `n_cells`, and the cell range inside `index_cells` -- and only a
// fully valid set is written. That ordering is what makes the refusals below meaningful: a caller
// that gets `ok` knows the whole set was consistent, not that the first three happened to be.
//
// It is NOT a transaction, and does not pretend to be: a device write cannot be rolled back. If the
// writer returns false partway through, the entries already written stay written, `failed_span` says
// which one failed, and the graph must be discarded rather than run -- a partially published input
// set is exactly the silent-wrong-answer case this function exists to prevent, so the caller is told
// to abandon it instead of being offered a retry that would publish a mixture of two batches.
enum class index_publish_refusal {
    ok,
    no_entries,          // nothing to publish: the result was a refusal, or collect_indices was false
    not_an_input,        // an entry's tensor does not carry GGML_TENSOR_FLAG_INPUT
    no_storage,          // the tensor has no data: publish ran before the scheduler allocated
    size_disagrees,      // entry.n_cells != index->ne[0]: the write would be truncated or overrun
    cell_range,          // cells_offset/n_cells fall outside index_cells
    entry_layout,        // the entries are not the layout the emitter produces: the gathers do not
                         // tile [0, total) contiguously in order, or the scatter is not the whole
                         // buffer, or an entry is not an I32 tensor
    write_failed,        // the writer returned false on `failed_span`
};

const char * describe(index_publish_refusal refusal);

// The write itself, in the shape `ggml_backend_tensor_set` has (tensor, data, offset, size) with a
// bool for "the write reached the device". A caller wraps ggml_backend_tensor_set and returns true,
// or returns false when its own error path fired; the adapter is three lines and is deliberately not
// hidden behind a virtual interface, because there is exactly one implementation in production.
//
// TWO REQUIREMENTS ON THE ADAPTER, both of which are silent corruption if broken:
//
//   1. GO THROUGH THE BACKEND. `entry.index->data` being non-null means the scheduler PLACED the
//      tensor, not that the host may dereference it: on a GPU backend that pointer is a device
//      address, and a `memcpy` into it either faults or writes somewhere unrelated. Use
//      `ggml_backend_tensor_set`.
//   2. DO NOT THROW. A function pointer that throws escapes this function, and although `failed_span`
//      names the entry that was in flight, the entries before it stay written and `ok` was never
//      returned — the graph must be discarded either way. The failure is reported, not neutralised.
using index_writer = bool (*)(ggml_tensor * tensor, const void * data,
                              std::size_t offset, std::size_t size);

// Publishes every entry of `result`, in span order and then the scatter's. Returns `ok` only if the
// whole set validated and every write succeeded.
//
// `failed_span` is set on `write_failed` (and points at the scatter's entry for the last one, whose
// `span` equals the number of spans). It is left untouched on a validation refusal, because nothing
// was written in that case.
index_publish_refusal publish_index_inputs(const physical_emit_result & result,
                                           index_writer write,
                                           std::string * why,
                                           std::size_t * failed_span = nullptr);

// WHETHER A VIEW NAMES EVERY CACHE POSITION THE MASK LEAVES VISIBLE.
//
// This is the check the emitter cannot do for itself. It fills the whole cache with `-INFINITY` and
// writes the named cells over it, so a mask can only NARROW the visible set; reproducing upstream's
// visibility therefore requires the view to already name every finite-mask position. The mask's
// values do not exist at graph-build time (measured: `data == nullptr` on a `ggml_set_input`
// tensor), which is why the caller has to assert it -- but there IS a moment when they are readable
// and identifiable: after upstream's `set_input_attn_mask` fills the host buffer, and the
// `HERD_READER_ROWS` probe in `patches/herd_reader_attn.cpp` reads them there. This function is for
// that moment, and it is plain arithmetic over caller-owned memory: no ggml context, no tensor.
//
// `-inf` at a position the view does not name is FINE and is the normal case: it says "invisible",
// which is exactly what the emitter's fill already says. What must not happen is a FINITE value
// outside the view, because that is upstream asking for a cell to contribute while this emission
// has already erased it. `+inf` and `NaN` are counted as violations too: neither is a bias the
// softmax can use, and treating them as "finite enough" would hide a contract break.
//
// WHAT THIS FUNCTION ASSUMES, and cannot check:
//   * `mask` points at `n_cell` readable floats. One length is taken, not two, because the mask is
//     a flat per-cache-position bias and the cache depth is the only extent that means anything;
//     a caller holding a wider buffer passes the cache depth and gets the first `n_cell` entries,
//     which is the intended reading. A shorter buffer is an out-of-bounds read and nothing here can
//     see it -- the same unverifiable precondition `compact_rows` states.
//   * the mask is F32. Upstream's `kq_mask` is F16 (measured `type 1`), so a caller holding the raw
//     tensor must widen first; a `ggml_fp16_t*` cast into this signature would read garbage, and
//     the report would be a confident answer about the wrong bytes. There is deliberately no
//     overload: widening needs a destination buffer, which is the caller's to own.
//   * the VIEW and the MASK describe the same decode. Nothing here compares epochs or positions.
//
// It reports a QUESTION, not a verdict, and `applicable` says whether the question applied at all.
// `complete == true` on a null mask previously read as "the view covers the mask", which is true and
// meaningless: there is no mask. A caller that branches on `complete` alone would then believe
// coverage was verified when nothing was examined, so the two facts are separated.
//
// THE EMPTY DOMAIN is the third case, and it is the one the two fields had disagreed about:
// `n_cell == 0` with a mask is APPLICABLE (a mask was given, so the question was asked) and
// COMPLETE (there is no position, so no finite bias can be missing from a view). Reporting it as
// incomplete would send a caller to widen a view that has nothing in it. All three cases are
// distinguished by (applicable, complete): (false, false) no mask, (true, true) empty domain,
// (true, false) a finite value outside the view.
struct mask_coverage_report {
    bool applicable = false;           // false: no mask was given, so nothing was examined
    bool complete = false;             // (only meaningful when applicable) no finite value outside the view
    std::size_t first_uncovered = 0;   // cache position of the first violating entry
    std::size_t finite_positions = 0;  // how many positions the mask leaves visible
    std::size_t named = 0;             // how many of those the view also names
};

mask_coverage_report view_covers_finite_mask(const query_view & view, const float * mask,
                                             std::size_t n_cell);

// The width of one mask element, named here so this header stays ggml-free. Upstream's `kq_mask` is
// F16; a caller holding the raw tensor says so, and the widening happens inside, where the
// destination belongs to the function that called the predicate.
enum class mask_width { f16, f32 };

// THE PER-ROW FORM, for a caller that holds a mask with one row per token.
//
// `row` points at the START of one row: `n_cell` elements of the given width, contiguous. This is
// the same question as `view_covers_finite_mask` and returns the same report; the difference is
// only that the caller hands over bytes it has not widened and does not have to duplicate the
// widening and the per-element kind switch at each call site.
//
// A NULL `row` means no mask was given, and the report says so with `applicable == false`: the
// absence of an answer stays distinguishable from a pass, exactly as in the scalar form.
//
// `n_cell == 0` with a row is the empty domain: applicable (a mask WAS given) and complete (no
// position can be missing from a view). Handled explicitly rather than by falling through to a
// zero-length buffer, whose `data()` may be null and would be read as "no mask" -- which is the
// third case, not this one.
mask_coverage_report view_covers_finite_mask_row(const query_view & view, const void * row,
                                                 mask_width width, std::size_t n_cell);

// The width in bytes of one mask element. Exposed because a caller that sizes a widened buffer, or
// asks for a row of a multi-row mask, should ask rather than repeat the `sizeof` switch -- a second
// copy of it is the one that drifts.
std::size_t mask_element_bytes(mask_width width);

// THE ROW START, for a caller holding a mask with one row per token.
//
// Prefill's `kq_mask` is `[n_cell, n_tokens, 1, 1]` and contiguous (measured), so token `t` begins at
// `base + t * n_cell * sizeof(element)`. That product -- the element width, the row index, and the
// ORDER of their multiplication -- used to be the only untested arithmetic between the product and
// the tested predicate. It lives here for the same reason the widening does: a test can then drive
// the arithmetic production runs instead of a copy of it.
//
// A null `base` yields null, so the result can be handed straight back to
// `view_covers_finite_mask_row` and the three cases still read as they do there: null means no mask.
// `n_cell == 0` makes every `t` the base pointer, which is the empty domain, not an absence.
//
// NOT CHECKED, deliberately: the offset is not range-limited. The caller owns the allocation and
// knows how many rows it holds; a `t` past them is an out-of-bounds read that no arithmetic here can
// see. The function also assumes a CONTIGUOUS mask -- for a strided tensor `nb[1]` is the stride, and
// the caller must use it instead of this function.
const void * finite_mask_row(const void * base, std::size_t t, std::size_t n_cell,
                             mask_width width);

// Emits the physical-form attention for ONE query row.
//
// Checked, not assumed: `in.q` is [head_dim, 1, n_head] and already carries this token's rotation;
// every span's `effective_pos` is equal; `in.n_cell` is the cache extent the scatter fills.
// `in.collect_indices` decides whether the cell list comes back (required under no_alloc).
physical_emit_result emit_physical_attention(const graph_inputs & in, const query_view & view,
                                             float kq_scale);

} // namespace tq::reader
