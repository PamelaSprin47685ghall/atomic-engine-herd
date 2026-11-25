#include "tq/reader/graph_emit.h"

#include <ggml.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <vector>

namespace tq::reader {
namespace {

emit_result refuse(emit_refusal refusal, std::string why) {
    emit_result result;
    result.refusal = refusal;
    result.why = std::move(why);
    return result;
}

// The union, in view order, with every cell checked once. Duplicates are refused rather
// than deduplicated: the same cell counted twice changes the answer (measured in
// attention.h's tests), and a caller that produced one has a bug this function cannot fix
// without guessing which occurrence was meant.
bool flatten(const query_view & view, std::size_t n_cell, std::vector<std::int32_t> & cells,
             std::string & why) {
    cells.clear();
    for (const span_keys & span : view.spans) {
        for (std::int32_t cell : span.cells) {
            if (cell < 0 || std::size_t(cell) >= n_cell) {
                why = "cell " + std::to_string(cell) + " outside [0, " +
                      std::to_string(n_cell) + ")";
                return false;
            }
            cells.push_back(cell);
        }
    }
    std::vector<std::int32_t> sorted = cells;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        why = "a cell appears twice in the view; the softmax would weight it twice";
        return false;
    }
    return true;
}

} // namespace

const char * describe(emit_refusal refusal) {
    switch (refusal) {
        case emit_refusal::ok:                return "ok";
        case emit_refusal::no_context:         return "no graph context or tensor";
        case emit_refusal::shape_mismatch:    return "tensor shapes disagree with the declared layout";
        case emit_refusal::multi_stream_view: return "k or v carries more than one stream on ne3";
        case emit_refusal::quantized_cache_unsupported: return "the cache is not F32; only float caches are measured";
        case emit_refusal::bad_head_layout:    return "n_head_kv does not divide n_head";
        case emit_refusal::empty_view:         return "the view has no keys";
        case emit_refusal::cell_out_of_range:  return "a cell index is outside the cache";
        case emit_refusal::duplicate_cell:     return "a cell appears twice in the view";
        case emit_refusal::no_softmax_input:   return "the union of spans is empty";
        case emit_refusal::mask_not_supported:
            return "a mask was supplied, and this emitter softmaxes over the named cells only, so "
                   "a cache-positioned mask has no axis here -- use the physical emitter";
        case emit_refusal::index_data_unavailable:
            // This text used to name an "index sink", an API that no longer exists: the entries
            // are returned in `emit_result::indices` behind `graph_inputs::collect_indices`, and
            // a caller reading the old message would go looking for a callback to install. A
            // refusal that names a removed API is worse than no refusal, because it reads as
            // guidance.
            return "the context does not allocate tensors (no_alloc), so each span's index tensor "
                   "has no storage at graph-build time; the entries come back in "
                   "emit_result::indices, but only if graph_inputs::collect_indices was set, and "
                   "the caller must then write them after the graph is allocated. A build that "
                   "cannot do that -- llama.cpp's own graph, where allocation happens after "
                   "build_attn returns -- needs an emission that does not gather at all";
        case emit_refusal::graph_error:        return "ggml refused a node (shape or allocation)";
    }
    return "unknown refusal";
}

emit_result emit_reader_attention(const graph_inputs & in, const query_view & view,
                                  float kq_scale, const span_rope_fn & rope,
                                  void * rope_user) {
    if (!in.ctx || !in.k || !in.v || !in.q) {
        return refuse(emit_refusal::no_context, "a reader emission needs ctx, k, v and q");
    }

    // A supplied mask is REFUSED, first, before any node is built. This emitter softmaxes over the
    // cells the view names, gathered into a dense axis of size "cells in the view"; a mask is
    // indexed by CACHE POSITION, and this emitter's axes do not correspond to those positions at
    // all. There is no reshaping that makes them agree, so the only honest answers are "here is a
    // mask-aware emission" (the physical one) or "no". Ignoring the pointer -- which is what
    // happened before this check, since nothing in this file read `in.mask` -- would run with the
    // caller's visibility rule discarded and report the result as if it had been honoured.
    //
    // Checked before the layout, because the reason does not depend on the layout: a caller has to
    // be sent to the other emitter, not told about n_head_kv.
    if (in.mask != nullptr) {
        return refuse(emit_refusal::mask_not_supported,
                      "graph_inputs::mask is set, but this emission's softmax runs over the "
                      "named cells, not the cache positions the mask indexes; "
                      "emit_physical_attention consumes a mask");
    }

    // The declared layout is checked against the tensors rather than trusted: a caller that
    // passed k as [head_dim, n_cell, n_head_kv] would otherwise gather along the wrong axis
    // and produce plausible floats out of the wrong place.
    if (in.n_head == 0 || in.n_head_kv == 0 || in.head_dim == 0 || in.n_cell == 0) {
        return refuse(emit_refusal::shape_mismatch, "a layout dimension is zero");
    }
    if (in.n_head % in.n_head_kv != 0) {
        return refuse(emit_refusal::bad_head_layout,
                      "n_head_kv " + std::to_string(in.n_head_kv) + " does not divide n_head " +
                      std::to_string(in.n_head));
    }
    // NARROWING THE ARITHMETIC TO ONE TYPE, and doing it here rather than at each multiply.
    //
    // ggml_mul_mat asserts a->type == b->type (ggml.c:2631), so every operand of this emission
    // must reach the same type before it meets another. A real model's cache is not F32 -- the
    // tiny transformer in this tree builds it as F16 (measured: cache type 1, and the probe run
    // reports exactly that) -- so "F32 only" refused the configuration that actually exists.
    //
    // The conversion is a deliberate widening, not a silent one: F16 -> F32 is exact (every
    // half is a float), so the arithmetic below runs on values that are precisely the ones the
    // cache holds, and the result differs from a native F32 cache only by the cache's own
    // quantisation -- which is what a run with an F16 cache should produce.
    //
    // Wider types are NOT accepted. A quantized cache (Q8_0 and friends) would need a
    // dequantising pass through ggml_cpy's type table, whose behaviour per quantization type
    // this module has not measured; refusing it by name keeps that a stated boundary rather
    // than a guess. The refusal text names the type it saw.
    const bool k_float16 = in.k->type == GGML_TYPE_F16;
    const bool v_float16 = in.v->type == GGML_TYPE_F16;
    if (!((in.k->type == GGML_TYPE_F32 || k_float16) &&
          (in.v->type == GGML_TYPE_F32 || v_float16))) {
        return refuse(emit_refusal::quantized_cache_unsupported,
                      std::string("k is ") + ggml_type_name(in.k->type) + " and v is " +
                      ggml_type_name(in.v->type) +
                      "; this emission widens F32 and F16 only, because widening a quantized "
                      "cache needs a dequantising pass whose per-type behaviour is unmeasured");
    }
    if (in.q->type != GGML_TYPE_F32 && in.q->type != GGML_TYPE_F16) {
        return refuse(emit_refusal::quantized_cache_unsupported,
                      std::string("q is ") + ggml_type_name(in.q->type) +
                      "; this emission widens F32 and F16 only");
    }
    // One widening, reused by every span: both the rotation and the multiply consume this
    // tensor, and a second cast per span would double the graph for no gain.
    ggml_tensor * q = in.q->type == GGML_TYPE_F32 ? in.q
                                                  : ggml_cast(in.ctx, in.q, GGML_TYPE_F32);

    // ne3 of the query is the stream axis here as well. A 4-D q with several streams would
    // multiply silently: mul_mat takes result ne3 from the second operand, so the output
    // would grow a stream axis the caller never asked about and the first stream's numbers
    // would be indistinguishable from the rest.
    if (in.q->ne[3] > 1) {
        return refuse(emit_refusal::multi_stream_view,
                      "q has " + std::to_string(in.q->ne[3]) +
                      " streams on ne3; this emission serves a single stream");
    }
    if (std::size_t(in.q->ne[0]) != in.head_dim || std::size_t(in.q->ne[1]) != 1 ||
        std::size_t(in.q->ne[2]) != in.n_head) {
        return refuse(emit_refusal::shape_mismatch,
                      "q is [" + std::to_string(in.q->ne[0]) + ", " + std::to_string(in.q->ne[1]) +
                      ", " + std::to_string(in.q->ne[2]) + "], expected [head_dim=" +
                      std::to_string(in.head_dim) + ", 1, n_head=" + std::to_string(in.n_head) + "]");
    }
    if (std::size_t(in.k->ne[0]) != in.head_dim || std::size_t(in.k->ne[1]) != in.n_head_kv ||
        std::size_t(in.k->ne[2]) != in.n_cell) {
        return refuse(emit_refusal::shape_mismatch, "k is not [head_dim, n_head_kv, n_cell] as declared");
    }
    // The two V caches upstream has, told apart by STRIDE and never by shape.
    //
    // llama_kv_cache::get_v (llama-kv-cache.cpp:1432 and :1451) returns one of:
    //
    //   !v_trans  ne = (head_dim, n_head_kv, n_cell)   dim-major within a cell
    //   v_trans   ne = (n_cell, n_head_kv, head_dim)   cell-major: element (c,h,d) sits at
    //                                                  c + d*nb[2] + h*nb[1], i.e. cells are
    //                                                  contiguous and dims step by kv_size
    //
    // and build_attn re-derives the bit as `v->nb[1] > v->nb[2]` (:3282). Both are read
    // exactly; the branches differ only in which axis carries the cell. Not refusing the
    // second one is the correction this module needed: the previous revision tested the
    // stride and declined, which was honest but left half of upstream's configurations
    // unserved -- and in a quantized run v_trans is common, because a cell-major V cache
    // is what lets a quantized row span all of a cell's dims.
    const bool v_cell_major = in.v->nb[1] > in.v->nb[2];
    if (v_cell_major) {
        // Declared as (n_cell, n_head_kv, head_dim) over the same buffer.
        if (std::size_t(in.v->ne[0]) != in.n_cell || std::size_t(in.v->ne[1]) != in.n_head_kv ||
            std::size_t(in.v->ne[2]) != in.head_dim) {
            return refuse(emit_refusal::shape_mismatch,
                          "v is cell-major (nb[1] > nb[2]) but not (n_cell, n_head_kv, head_dim)");
        }
    } else if (std::size_t(in.v->ne[0]) != in.head_dim ||
               std::size_t(in.v->ne[1]) != in.n_head_kv ||
               std::size_t(in.v->ne[2]) != in.n_cell) {
        return refuse(emit_refusal::shape_mismatch,
                      "v is dim-major but not (head_dim, n_head_kv, n_cell)");
    }
    if (in.k->nb[1] > in.k->nb[2]) {
        return refuse(emit_refusal::shape_mismatch,
                      "k has cell-major rows (nb[1] > nb[2]); the declared [head_dim, n_head_kv, "
                      "n_cell] indexing assumes dim-major");
    }

    // One stream only. Upstream's cache is 4-D when n_stream > 1 (a non-unified cache with
    // several sequences), and the stream axis steps by nb[3]. Reading stream 0 while a caller
    // meant stream s would answer with another sequence's tokens -- finite, plausible, wrong.
    // Named refusal, because the emitter cannot know which stream a caller wants and guessing
    // is the one thing worse than declining.
    if (in.k->ne[3] > 1 || in.v->ne[3] > 1) {
        return refuse(emit_refusal::multi_stream_view,
                      "k or v has n_stream = " +
                      std::to_string(std::max(in.k->ne[3], in.v->ne[3])) +
                      " on ne3; this emission serves a single stream and will not pick one");
    }

    // Only float caches are verified, and this refuses the rest BY NAME.
    //
    // The previous revision tested `ne0 % blck == 0` and called that "a quantized row is not a
    // whole block". Two things were wrong with it. The predicate looked at ne0 of the CACHE,
    // while the row the gather walks is ne1 -- so for a cell-major V cache it tested the cell
    // count, not the dim count. And the condition was not the obstacle: ggml_cont of a
    // transposed quantized tensor needs a dequantizing pass this module does not do, so a
    // quantized cache whose shape passed the old check could still fail in the graph builder.
    //
    // A predicate that is right about the wrong axis is worse than no predicate, because it
    // reads as coverage. The honest statement is the one below, and it has moved up with the
    // rest of the type gate: F32 and F16 are widened to one arithmetic type, everything else is
    // declined by name.

    if (view.spans.empty()) {
        return refuse(emit_refusal::empty_view, "the view has no spans");
    }
    std::vector<std::int32_t> cells;
    std::string why;
    if (!flatten(view, in.n_cell, cells, why)) {
        const bool out_of_range = why.rfind("cell ", 0) == 0;
        return refuse(out_of_range ? emit_refusal::cell_out_of_range : emit_refusal::duplicate_cell,
                      std::move(why));
    }
    if (cells.empty()) {
        return refuse(emit_refusal::empty_view, "every span in the view is empty");
    }

    // Key gather, per kv head: k is [head_dim, n_head_kv, n_cell], and mul_mat needs the
    // keys on ne1 with head_dim on ne0 ([head_dim, S, n_head_kv]) -- not
    // [head_dim, n_head_kv, S], which would put the head axis where the key axis belongs
    // and make the output's ne0 the head count.
    std::vector<ggml_tensor *> k_head_rows;
    k_head_rows.reserve(in.n_head_kv);
    for (std::size_t h = 0; h < in.n_head_kv; ++h) {
        // ne0 = head_dim (element stride), ne1 = the cell axis, whose stride in k's
        // [head_dim, n_head_kv, n_cell] layout is nb[2]; head h starts nb[1] bytes in. The
        // first version passed nb[1] as the cell stride, which silently read another kv
        // head's row for every cell after the first -- the numbers stayed finite and wrong.
        ggml_tensor * head = ggml_view_3d(in.ctx, in.k, std::int64_t(in.head_dim),
                                          std::int64_t(in.n_cell), 1,
                                          in.k->nb[2], in.k->nb[2], h * in.k->nb[1]);
        ggml_tensor * rows = ggml_cont(in.ctx, head);   // [head_dim, n_cell], contiguous
        // Widen AFTER the copy, not before. A cast of the strided view would go through cpy's
        // strided path, while a cast of this contiguous tensor is a straight elementwise
        // conversion -- and the copy is needed anyway for get_rows.
        k_head_rows.push_back(k_float16 ? ggml_cast(in.ctx, rows, GGML_TYPE_F32) : rows);
    }

    // Value gather, in the form ggml_get_rows actually selects along.
    //
    // ggml_get_rows(a, b) asserts a->ne[2] == b->ne[1] and a->ne[3] == b->ne[2] (ggml.c:3978);
    // b is 1-D, so the two assertions reduce to "a is a matrix" -- and it copies
    // `a->data + i*nb[1]` for each index i, i.e. it selects ROWS ALONG ne1 and emits
    // [a->ne0, b->ne0]. The previous revision built the per-head tensor as [n_cell, head_dim]
    // with cells on ne0, on the reading that the key axis "should be ne0". That is ne0 for the
    // MULTIPLY, not for the GATHER: with cells on ne0 the gather selects head dimensions, so
    // each block came back [head_dim, S] holding the wrong axis, and the shapes that followed
    // were consistent-looking garbage.
    //
    // So the gather form is [head_dim, n_cell] for BOTH cache layouts -- dims on ne0 (the
    // element-contiguous axis), cells on ne1 (the axis get_rows walks) -- and the transpose
    // into the multiply's layout happens after the gather, on the small [head_dim, S] block.
    //
    //   !v_trans  the natural view is already [head_dim, n_cell]: no transpose
    //   v_trans   the natural view is [n_cell, head_dim] (cells contiguous, dims step by
    //             kv_size; llama-kv-cache.cpp:1451): transpose, then cont
    //
    // Both end contiguous, so a quantized cache only has to satisfy whole-block rows.
    std::vector<ggml_tensor *> v_head_dc;
    v_head_dc.reserve(in.n_head_kv);
    for (std::size_t h = 0; h < in.n_head_kv; ++h) {
        ggml_tensor * dense = nullptr;
        if (v_cell_major) {
            // ne0 = n_cell (c stride = element), ne1 = head_dim (d stride = v->nb[2]).
            ggml_tensor * head = ggml_view_3d(in.ctx, in.v, std::int64_t(in.n_cell),
                                              std::int64_t(in.head_dim), 1,
                                              in.v->nb[2], in.v->nb[2], h * in.v->nb[1]);
            dense = ggml_cont(in.ctx, ggml_transpose(in.ctx, head));   // [head_dim, n_cell]
        } else {
            ggml_tensor * head = ggml_view_3d(in.ctx, in.v, std::int64_t(in.head_dim),
                                              std::int64_t(in.n_cell), 1,
                                              in.v->nb[2], in.v->nb[2], h * in.v->nb[1]);
            dense = ggml_cont(in.ctx, head);   // [head_dim, n_cell], already the form
        }
        // Same rule as the keys: the transpose and the copy come first, then the widening, so
        // the cast is always of a contiguous tensor. Note the cell-major branch transposes an
        // F16 view -- that is a permute of halves, which needs no conversion.
        v_head_dc.push_back(v_float16 ? ggml_cast(in.ctx, dense, GGML_TYPE_F32) : dense);
    }

    ggml_tensor * scores = nullptr;   // [S, 1, n_head], the union in view order
    ggml_tensor * values = nullptr;   // [S, head_dim, n_head_kv], the same order

    // The span's position in the view, which the sink receives. Counted here rather than derived
    // from the loop variable because the loop is a range-for over a vector of spans: an index the
    // caller can match against `view.spans` is the only form a caller can act on, and a counter is
    // the only thing that gives it.
    std::size_t span_number = 0;

    // THE SPANS' INDEX TENSORS, recorded only if the caller asked.
    //
    // `want_indices` comes from `graph_inputs::collect_indices`. When it is false this vector is
    // never touched, so an emitter that supplies its own indices and wants nothing back pays
    // nothing -- not an empty vector, not a reserve.
    //
    // The cells are COPIED into `cells_flat`, not pointed at. `view` is a const reference, so a
    // caller may pass a temporary; the entries outlive the call, and a pointer into the view would
    // be dangling by the time anyone read it. One buffer, appended in span order, so an entry
    // needs only an offset -- which also survives the move into `emit_result`.
    // NO COUNTER. The entries carry no identity token, and `cells_of` decides membership by
    // comparing values -- see the note on `emit_result`. A process-wide counter would wrap (its 0
    // reserved for "no emission" becomes reachable) and would describe the emission rather than the
    // cells, which two independent copies can share while holding different data.
    std::vector<graph_inputs::index_entry> pending;
    std::vector<std::int32_t> cells_flat;
    const bool want_indices = in.collect_indices;
    if (want_indices) {
        pending.reserve(view.spans.size());
        std::size_t total_cells = 0;
        for (const span_keys & span : view.spans) {
            total_cells += span.cells.size();
        }
        cells_flat.reserve(total_cells);
    }

    for (const span_keys & span : view.spans) {
        if (span.cells.empty()) {
            // Not a no-op: skipping it silently would let a caller ship a view whose span
            // collapsed to nothing and read the numbers as if it had keys.
            return refuse(emit_refusal::empty_view, "a span in the view has no keys");
        }

        const std::int64_t n_span = std::int64_t(span.cells.size());
        ggml_tensor * idx = ggml_new_tensor_1d(in.ctx, GGML_TYPE_I32, n_span);
        if (!idx) {
            // Only the NODE can fail here. The previous version tested `!idx || !idx->data` and
            // reported both as "cannot allocate the index tensor", which named ggml for a
            // failure that was the caller's missing half of the graph-input contract.
            return refuse(emit_refusal::graph_error, "ggml refused the index tensor node");
        }
        if (idx->data) {
            // An allocating context: the cells go in now.
            std::copy(span.cells.begin(), span.cells.end(), static_cast<std::int32_t *>(idx->data));
        } else if (!want_indices) {
            // No storage and no one to fill it. Refused by name, because building the gather on
            // uninitialised indices yields finite, plausible numbers from arbitrary cells --
            // the failure mode this module exists to prevent.
            return refuse(emit_refusal::index_data_unavailable,
                          "span " + std::to_string(span_number) + " has " +
                              std::to_string(span.cells.size()) +
                              " cells but the context allocates nothing and the caller did not " +
                              "ask for the entries; set graph_inputs::collect_indices and write " +
                              "each tensor after the graph has been allocated");
        }
        if (want_indices) {
            graph_inputs::index_entry entry;
            entry.span = span_number;
            entry.index = idx;
            entry.cells_offset = cells_flat.size();
            entry.n_cells = span.cells.size();
            cells_flat.insert(cells_flat.end(), span.cells.begin(), span.cells.end());
            pending.push_back(entry);
        }
        ++span_number;

        // Keys for this span: per kv head [head_dim, S_span], heads stacked on axis 2 so the
        // result is [head_dim, S_span, n_head_kv] -- the shape whose ne2 is the kv-head axis
        // the query heads are grouped by, and whose ne1 is the key axis the softmax and the
        // value reduction both run along.
        ggml_tensor * k_span = nullptr;
        for (std::size_t h = 0; h < in.n_head_kv; ++h) {
            ggml_tensor * rows = ggml_get_rows(in.ctx, k_head_rows[h], idx);   // [head_dim, S]
            k_span = k_span ? ggml_concat(in.ctx, k_span, rows, 2) : rows;
        }
        if (!k_span) {
            return refuse(emit_refusal::graph_error, "no kv head produced keys");
        }

        ggml_tensor * q_span = q;
        if (rope) {
            q_span = rope(in.ctx, q, span.effective_pos, rope_user);
            if (!q_span) {
                return refuse(emit_refusal::graph_error,
                              "the injected rotation returned null for position " +
                              std::to_string(span.effective_pos));
            }
        }

        // [S_span, 1, n_head]: ne2 broadcast over the head group, exactly as upstream's
        // mul_mat does for a GQA model.
        ggml_tensor * kq = ggml_scale(in.ctx, ggml_mul_mat(in.ctx, k_span, q_span), kq_scale);

        // Values for this span: per kv head, [S_span, head_dim], heads stacked on axis 2 so
        // the result is [S_span, head_dim, n_head_kv]. Head blocks are contiguous within the
        // head (element i + S*d), so concatenation on axis 2 is a plain join.
        ggml_tensor * v_span = nullptr;
        for (std::size_t h = 0; h < in.n_head_kv; ++h) {
            // get_rows walks ne1, so this is [head_dim, S] -- dims first, keys second.
            ggml_tensor * rows = ggml_get_rows(in.ctx, v_head_dc[h], idx);        // [head_dim, S]
            // The multiply wants keys on ne0 (mul_mat(v, w) reduces v's ne0 against w's ne0),
            // so the block is transposed and made contiguous: [S, head_dim]. This is the one
            // transpose the layout forces, and it happens after the gather -- which is what
            // makes it cheap (S x head_dim) and, on a quantized cache, possible at all: the
            // gathered rows are contiguous floats, while transposing the cache itself would
            // need a dequantizing pass.
            ggml_tensor * block = ggml_cont(in.ctx, ggml_transpose(in.ctx, rows));  // [S, head_dim]
            // Heads join on axis 2, the kv-head axis. Dim 0 is the key axis: concatenating
            // heads there (the previous revision) produced [S*n_head_kv, head_dim], a tensor
            // whose ne0 no longer equals the weights' ne0 -- which is the GGML_ASSERT that
            // made this fail. With one kv head the two readings coincide, so only a
            // multi-head fixture can tell them apart, and the test uses two.
            v_span = v_span ? ggml_concat(in.ctx, v_span, block, 2) : block;
        }
        if (!v_span) {
            return refuse(emit_refusal::graph_error, "no kv head produced values");
        }

        // Concatenation along the key axis is what makes the normalization shared: the
        // softmax below sees every span's logits at once, so a key from a later span competes
        // with one from an earlier span in one distribution.
        scores = scores ? ggml_concat(in.ctx, scores, kq, 0) : kq;
        values = values ? ggml_concat(in.ctx, values, v_span, 0) : v_span;
    }

    if (!scores || !values) {
        return refuse(emit_refusal::no_softmax_input, "no span produced logits");
    }

    ggml_tensor * weights = ggml_soft_max(in.ctx, scores);      // [S, 1, n_head]
    ggml_tensor * out = ggml_mul_mat(in.ctx, values, weights);  // [head_dim, 1, n_head]
    if (!weights || !out) {
        return refuse(emit_refusal::graph_error, "ggml refused the softmax or the reduction");
    }

    // NO CALLBACK, NO "PUBLICATION", AND THEREFORE NOTHING TO CLAIM.
    //
    // The entries travel back inside the result. That is the whole of the transaction design on
    // this side: the emitter finishes, and only then does the caller receive anything it could act
    // on. A caller that records into a registry, a map, or a queue does it after this function has
    // returned, inside whatever scope it chooses -- so a failure there cannot leave the emitter's
    // work and the caller's state disagreeing, because the emitter's work is already finished and
    // handed over as a value.
    //
    // What the previous two designs claimed, and why the claim was false: both called into the
    // caller mid-emission and then said "nothing was published". A callback that appends to the
    // caller's registry and THEN throws has published; the emitter cannot see it, and no wording
    // in a refusal message undoes it. Removing the callback removes the claim, and the claim was
    // the problem.
    emit_result result;
    result.out = out;
    result.n_visible = cells.size();
    if (want_indices) {
        result.indices = std::move(pending);
        result.index_cells = std::move(cells_flat);
    }
    return result;
}

} // namespace tq::reader
