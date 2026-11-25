// tq/reader/views.h — the reader lane's domain core, with no llama and no ggml.
//
// RERoT asks a query to attend over a hand-picked set of stored keys, with one
// softmax over the whole set and a rope position that differs per key. The fork
// implemented that inside llama-graph.cpp, where it could not be tested without a
// model, and inside llama-kv-cache.cpp, where it was welded to the cache's private
// cell layout. The parts that hold the *rules* -- what a valid view is, and which
// physical cells a view resolves to -- are pure functions of (views, cells). They
// are ported here as such.
//
// Ported from:
//   llama_kv_cache::prepare_attention_views   (validation, src/llama-kv-cache.cpp:2649)
//   llama_kv_cache::resolve_attention_view    (resolution, :2740)
//   llama_kv_cache::validate_attention_batch  (batch agreement, :2709)
//
// Deliberately not ported: the graph construction that consumes the result. That
// needs fields in upstream's llm_graph_input_attn_kv and a build function that
// emits per-span gather/rope nodes; see the module README for why that boundary is
// not crossable by a function-entry seam.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace tq::reader {

using seq_id = std::int32_t;
using pos_t  = std::int32_t;

// One key the query may see: where it lives (key_seq, key_pos) and what rope
// position the reader pretends it has (virtual_pos).
struct entry {
    seq_id key_seq      = 0;
    pos_t  key_pos      = 0;
    pos_t  virtual_pos  = 0;
};

// A query and the exact set of keys it sees.
struct view {
    seq_id query_seq         = 0;
    pos_t  query_pos         = 0;
    pos_t  query_virtual_pos = 0;
    std::vector<entry> entries;
};

// Why a view set was rejected. Each value names the condition a caller violated,
// so a driver can report something a reader can act on instead of "invalid".
enum class refusal {
    none,
    no_views,
    too_many_views,      // more views than the sequence limit
    query_seq_invalid,
    query_pos_invalid,
    virtual_pos_invalid,
    no_entries,
    too_many_entries,    // more entries than the cache holds
    duplicate_query_seq, // two views for one sequence
    duplicate_query_pos, // the same (seq, pos) twice
    query_already_stored,// a view for a position that is already in the cache
    entry_seq_invalid,
    entry_pos_invalid,
    entry_after_query,   // key strictly after the query in the same sequence
    virtual_not_ascending,
    virtual_beyond_query,
    effective_pos_invalid, // query_virtual + key_pos - virtual_pos out of range
    own_entry_missing,   // the query's own (seq,pos) is not among its entries
    own_entry_duplicated,
    own_entry_position,  // the own entry's virtual_pos != query_virtual_pos
    key_missing,         // a non-own entry names a cell that is not in the cache
    key_cell_negative,
    key_duplicated,      // the same physical cell appears twice in one view
    key_is_pending,      // names a key that is itself a query in this same batch
    shifted,             // the cache has been shifted; indices are stale
    multi_stream,        // more than one stream; the lane supports one
    // Batch agreement has its own reasons. Reusing a validation reason here would
    // tell the caller to fix the wrong thing: "duplicate query position" when the
    // real problem is that the batch has one token fewer than the view set.
    batch_count_mismatch,// tokens and views disagree in count
    batch_has_embd,      // an embedding batch: tokens have no ids to sample
    token_without_view,  // a batch token that no view describes
    token_multi_seq,     // a token that claims more than one sequence
};

const char * name(refusal why);

// Physical KV cell index for a stored (seq, pos). -1 means "stored, but no usable
// cell" -- the same convention the cache uses, and rejected rather than skipped.
using cell_map = std::map<std::pair<seq_id, pos_t>, std::int32_t>;

// One rope group: every key in `indices` is read with the query rotated to
// `effective_pos`. Consecutive entries that share an effective position collapse
// into one span, because one rotation serves all of them.
struct span {
    pos_t              effective_pos = 0;
    std::vector<std::int32_t> indices;
};

struct resolved {
    std::vector<span>         spans;
    std::vector<std::int32_t> indices; // every cell, in view order
};

struct resolution {
    refusal           why = refusal::none;
    std::vector<resolved> views; // parallel to the validated input
};

// Which rules apply. These are not tunables: they mirror the cache's own limits,
// and the caller passes them because this core does not know a cache.
//
// Two of them are easy to confuse and must not be merged. The fork bounds a view
// count and a sequence id by two different numbers (src/llama-kv-cache.cpp:2652,
// :2660):
//
//   count > LLAMA_MAX_SEQ (256)         -> the whole set is refused
//   query_seq >= seq_to_stream.size()   -> that view is refused
//
// `seq_to_stream.size()` is 256 only in the unpipelined case; it shrinks to the
// configured stream count otherwise (:324). An earlier version of this struct had
// a single `max_views = 512` for both roles, which accepted twice the permitted
// number of views and accepted sequence ids a cache with fewer streams does not
// have -- both silently, since a too-large id simply finds no cells.
struct limits {
    std::size_t max_views   = 256;     // LLAMA_MAX_SEQ: the view-count bound
    std::size_t n_seqs      = 256;     // seq_to_stream.size(): valid seq ids are [0, n_seqs)
    std::size_t cache_size  = 1u << 20;
    bool        shifted     = false;
    int         n_streams   = 1;
};

// Validate a view set against the cache's cell map. On success every view is
// accepted; on failure the whole set is rejected and `why` names the first
// violated rule. All-or-nothing is deliberate: a partially accepted set would
// silently run some queries with reader semantics and others with stock ones,
// producing numbers that look fine and mean two different things.
refusal validate(const std::vector<view> & views, const cell_map & cells, const limits & lim);

// Resolution: the spans a graph builder needs. A SEPARATE PHASE from validate(),
// mirroring the fork (src/llama-attention-view.h: "Logical identities are resolved
// after KV slot preparation. No physical index survives a decode call."):
//
//   validate()  before the batch's keys are written; the query's own position must
//               be ABSENT from `cells`.
//   resolve()   after that write; every entry, own included, must be PRESENT, and
//               resolves to its real cell index.
//
// The caller must have validated first -- the fork structures it as set-then-resolve
// (llama_context::set_attention_views, then llm_build_context::build_attn) and this
// pair mirrors that. Calling resolve() with the pre-write map reports key_missing,
// which is the accurate diagnosis rather than a substitute index.
//
// `cells` must follow the cache's own convention for ambiguous identities: when two
// cells claim one (seq, pos), the value is negative (llama-kv-cache.cpp
// attention_cells). Such a view is refused, never resolved to one of them.
resolution resolve(const std::vector<view> & views, const cell_map & cells, const limits & lim);

// Batch agreement, in the same two steps the fork performs
// (llama_kv_cache::validate_attention_batch, src/llama-kv-cache.cpp:2709):
//
//   step 1  every token corresponds to exactly one view, each token claims exactly
//           ONE sequence (`n_seq_id == 1`), and no sequence appears twice;
//   step 2  the whole view set passes validate() again.
//
// Step 2 is not redundant. The fork re-runs prepare_attention_views on the views it
// extracted from its own stored set, so a view whose entries or rotations went stale
// between set_attention_views() and this call is caught here rather than during
// graph construction. A port that checked only the batch would accept a batch that
// agrees with a view set that no longer validates.
//
// `batch_seqs[i]`, `batch_pos[i]` and `batch_n_seq_ids[i]` describe token i.
// `cells` and `lim` are for step 2, and must describe the pre-write cache, exactly
// as validate() expects.
//
// Two checks in the fork's step 1 cannot be expressed here, and are the caller's:
// the null-array tests (`!batch.pos`, `!batch.n_seq_id`, `!batch.seq_id`,
// `!batch.token`) and `!batch.seq_id[i]`. A std::vector passed by reference is
// either sized or empty, and an empty one is rejected by the count check; a null
// inner pointer has no equivalent. A caller that reads these arrays from a
// llama_batch must perform those four tests itself before calling. This is a real
// gap in the port's coverage, stated rather than papered over.
refusal validate_batch(const std::vector<view> & views,
                      const std::vector<seq_id> & batch_seqs,
                      const std::vector<pos_t> & batch_pos,
                      const std::vector<std::int32_t> & batch_n_seq_ids,
                      bool has_embd,
                      const cell_map & cells,
                      const limits & lim);

} // namespace tq::reader
