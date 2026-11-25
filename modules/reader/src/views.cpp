// The reader lane's rules, ported from the fork's KV cache and expressed as pure
// functions. Every check below exists because the fork performed it; the comments
// say what a violation would mean rather than restating the condition.
#include <tq/reader/views.h>

#include <algorithm>
#include <limits>

namespace tq::reader {
namespace {

using namespace std;

bool in_range(seq_id s, size_t seq_limit) {
    return s >= 0 && size_t(s) < seq_limit;
}

// Effective rope position for one entry: where the key pretends to be, relative
// to where the query pretends to be.
bool effective_of(const view & v, const entry & e, pos_t & out) {
    const int64_t effective = int64_t(v.query_virtual_pos) + int64_t(e.key_pos) - int64_t(e.virtual_pos);
    if (effective < 0 || effective > numeric_limits<pos_t>::max()) {
        return false;
    }
    out = pos_t(effective);
    return true;
}

} // namespace

const char * name(refusal why) {
    switch (why) {
        case refusal::none:                 return "ok";
        case refusal::no_views:             return "no views";
        case refusal::too_many_views:       return "more views than the sequence limit";
        case refusal::query_seq_invalid:    return "query sequence out of range";
        case refusal::query_pos_invalid:    return "query position negative";
        case refusal::virtual_pos_invalid:  return "query virtual position negative";
        case refusal::no_entries:           return "view has no entries";
        case refusal::too_many_entries:     return "more entries than the cache holds";
        case refusal::duplicate_query_seq:  return "two views for one sequence";
        case refusal::duplicate_query_pos:  return "the same query position twice";
        case refusal::query_already_stored: return "query position already in the cache";
        case refusal::entry_seq_invalid:    return "entry sequence out of range";
        case refusal::entry_pos_invalid:    return "entry position negative";
        case refusal::entry_after_query:    return "entry lies after the query in its sequence";
        case refusal::virtual_not_ascending:return "entry virtual positions are not ascending";
        case refusal::virtual_beyond_query: return "entry virtual position exceeds the query's";
        case refusal::effective_pos_invalid:return "effective position out of range";
        case refusal::own_entry_missing:    return "the query's own position is not among its entries";
        case refusal::own_entry_duplicated: return "the query's own position appears twice";
        case refusal::own_entry_position:   return "the own entry's virtual position differs from the query's";
        case refusal::key_missing:          return "entry names a position the cache does not hold";
        case refusal::key_cell_negative:    return "entry resolves to an unusable cell";
        case refusal::key_duplicated:       return "the same physical cell appears twice in one view";
        case refusal::key_is_pending:       return "entry names a key that is a query in this same batch";
        case refusal::shifted:              return "cache was shifted; stored indices are stale";
        case refusal::multi_stream:         return "more than one KV stream; the lane supports one";
        case refusal::batch_count_mismatch: return "batch has a different number of tokens than the view set";
        case refusal::batch_has_embd:       return "an embedding batch has no token ids for the reader";
        case refusal::token_without_view:   return "a batch token has no view";
        case refusal::token_multi_seq:      return "a batch token that claims more than one sequence";
    }
    return "unknown";
}

refusal validate(const std::vector<view> & views, const cell_map & cells, const limits & lim) {
    // The lane is defined over a single stream: with two streams a (seq, pos) can
    // resolve to two cells, and picking one would silently read the wrong keys.
    if (lim.n_streams != 1) {
        return refusal::multi_stream;
    }
    // A shift rewrites positions in place, so any cell index captured before it is
    // stale. Refusing is the only option: recomputing would mean re-deriving every
    // view, which is the caller's job, not a validator's.
    if (lim.shifted) {
        return refusal::shifted;
    }
    if (views.empty()) {
        return refusal::no_views;
    }
    if (views.size() > lim.max_views) {
        return refusal::too_many_views;
    }

    std::set<std::pair<seq_id, pos_t>> pending;
    std::set<seq_id>                   readers;
    for (const view & v : views) {
        if (!in_range(v.query_seq, lim.n_seqs)) {
            return refusal::query_seq_invalid;
        }
        if (v.query_pos < 0) {
            return refusal::query_pos_invalid;
        }
        if (v.query_virtual_pos < 0) {
            return refusal::virtual_pos_invalid;
        }
        if (v.entries.empty()) {
            return refusal::no_entries;
        }
        if (v.entries.size() > lim.cache_size) {
            return refusal::too_many_entries;
        }
        // One view per sequence: the batch carries one token per view, so two views
        // for one sequence would need two tokens at the same sequence and the
        // batch's own seq-id check would reject it later with a worse message.
        if (!readers.insert(v.query_seq).second) {
            return refusal::duplicate_query_seq;
        }
        if (!pending.emplace(v.query_seq, v.query_pos).second) {
            return refusal::duplicate_query_pos;
        }
        // The query's own position must not be in the cache: its keys are being
        // written by this very batch, so they are not addressable yet.
        if (cells.count({v.query_seq, v.query_pos}) != 0) {
            return refusal::query_already_stored;
        }
    }

    for (const view & v : views) {
        std::set<std::int32_t> physical;
        bool     own = false;
        pos_t    previous_virtual = -1;
        for (const entry & e : v.entries) {
            if (!in_range(e.key_seq, lim.n_seqs)) {
                return refusal::entry_seq_invalid;
            }
            if (e.key_pos < 0) {
                return refusal::entry_pos_invalid;
            }
            // A reader may look back, never forward: a key after the query in the
            // same sequence has not been computed with respect to this query.
            if (e.key_seq == v.query_seq && e.key_pos > v.query_pos) {
                return refusal::entry_after_query;
            }
            // Ascending virtual positions keep the effective positions monotone, so
            // a span's keys are contiguous in the order the softmax sees them.
            if (e.virtual_pos <= previous_virtual) {
                return refusal::virtual_not_ascending;
            }
            if (e.virtual_pos > v.query_virtual_pos) {
                return refusal::virtual_beyond_query;
            }
            pos_t effective = 0;
            if (!effective_of(v, e, effective)) {
                return refusal::effective_pos_invalid;
            }
            previous_virtual = e.virtual_pos;

            const bool is_own = e.key_seq == v.query_seq && e.key_pos == v.query_pos;
            if (is_own) {
                if (own) {
                    return refusal::own_entry_duplicated;
                }
                // The query's own key must be read with the query's own rotation,
                // or the token would attend to itself at a different rope position
                // than the one its K was written with.
                if (e.virtual_pos != v.query_virtual_pos) {
                    return refusal::own_entry_position;
                }
                own = true;
                continue;
            }
            // A key that is itself a query in this batch is being written now, so
            // its cell is not available to read.
            if (pending.count({e.key_seq, e.key_pos}) != 0) {
                return refusal::key_is_pending;
            }
            const auto found = cells.find({e.key_seq, e.key_pos});
            if (found == cells.end()) {
                return refusal::key_missing;
            }
            if (found->second < 0) {
                return refusal::key_cell_negative;
            }
            // Two entries resolving to one cell would attend to the same keys twice,
            // doubling their weight -- a silent numeric change, not a duplicate.
            if (!physical.insert(found->second).second) {
                return refusal::key_duplicated;
            }
        }
        // Without its own entry the query does not see the token it is decoding,
        // which is a different model, not a smaller one.
        if (!own) {
            return refusal::own_entry_missing;
        }
    }
    return refusal::none;
}

resolution resolve(const std::vector<view> & views, const cell_map & cells, const limits & lim) {
    // Phase B, and it is a different phase from validate() -- not a stricter one.
    //
    // The fork says so in its own words (src/llama-attention-view.h): "Logical
    // identities are resolved after KV slot preparation. No physical index survives
    // a decode call." validate() runs at set time, before the batch's keys are
    // written, which is why it requires the query's own position to be ABSENT from
    // the cache. resolve() runs during graph building, after that write, which is
    // why every entry -- the query's own included -- must now be PRESENT.
    //
    // So this function does not call validate(): validating here would reject every
    // legitimate resolution, because the query has just been written. The earlier
    // version of this file did call it, then patched over the contradiction by
    // resolving the own entry to -1. That marker does not exist in the fork, and a
    // -1 handed to ggml_get_rows is a wild row index, not a placeholder. The caller
    // is expected to have validated; the fork's own structure is set-then-resolve.
    resolution out;
    if (lim.n_streams != 1) {
        out.why = refusal::multi_stream;
        return out;
    }
    if (lim.shifted) {
        out.why = refusal::shifted;
        return out;
    }
    if (views.empty()) {
        out.why = refusal::no_views;
        return out;
    }

    out.views.reserve(views.size());
    for (const view & v : views) {
        resolved r;
        std::set<std::int32_t> physical;
        for (const entry & e : v.entries) {
            pos_t effective = 0;
            // Validation proved this succeeds; a failure here would mean the two
            // passes disagree, which is a programming error worth surfacing.
            if (!effective_of(v, e, effective)) {
                out.why = refusal::effective_pos_invalid;
                out.views.clear();
                return out;
            }
            // Every entry is looked up, the query's own included: by this phase the
            // cache holds it. A missing own entry means the caller resolved before
            // the write, and the honest answer is key_missing -- not a marker the
            // graph would then use as a row index.
            const auto found = cells.find({e.key_seq, e.key_pos});
            if (found == cells.end()) {
                out.why = refusal::key_missing;
                out.views.clear();
                return out;
            }
            const std::int32_t cell = found->second;
            // Negative is the cache's ambiguous-identity marker: two cells claiming
            // one logical (seq, pos). Reading either would be a guess between them.
            if (cell < 0) {
                out.why = refusal::key_cell_negative;
                out.views.clear();
                return out;
            }
            if (!physical.insert(cell).second) {
                out.why = refusal::key_duplicated;
                out.views.clear();
                return out;
            }
            // Consecutive entries sharing an effective position collapse into one
            // span: one rotation serves all of them.
            if (r.spans.empty() || r.spans.back().effective_pos != effective) {
                r.spans.push_back(span{effective, {}});
            }
            r.spans.back().indices.push_back(cell);
            r.indices.push_back(cell);
        }
        out.views.push_back(std::move(r));
    }
    return out;
}

refusal validate_batch(const std::vector<view> & views,
                       const std::vector<seq_id> & batch_seqs,
                       const std::vector<pos_t> & batch_pos,
                       const std::vector<std::int32_t> & batch_n_seq_ids,
                       bool has_embd,
                       const cell_map & cells,
                       const limits & lim) {
    // An empty view set imposes no constraints, and the fork returns true before
    // looking at the batch at all (llama-kv-cache.cpp:2711).
    if (views.empty()) {
        return refusal::none;
    }

    // ---- step 1: the batch agrees with the view set ----
    if (batch_seqs.size() != views.size() || batch_pos.size() != views.size() ||
        batch_n_seq_ids.size() != views.size()) {
        return refusal::batch_count_mismatch;
    }
    // An embedding row has no token id, so there is nothing for a view to describe.
    if (has_embd) {
        return refusal::batch_has_embd;
    }
    std::set<seq_id> readers;
    for (size_t i = 0; i < views.size(); ++i) {
        // The fork's `batch.n_seq_id[i] != 1`. A token shared by two sequences has
        // no single "this query's view": the pair (seq, pos) that identifies a view
        // would be ambiguous, and the fork rejects it rather than picking one.
        if (batch_n_seq_ids[i] != 1) {
            return refusal::token_multi_seq;
        }
        // A token contributing to two sequences would make the same pair ambiguous
        // from the other direction, so the sequence may appear only once.
        if (!readers.insert(batch_seqs[i]).second) {
            return refusal::duplicate_query_seq;
        }
        const auto found = std::find_if(views.begin(), views.end(), [&](const view & v) {
            return v.query_seq == batch_seqs[i] && v.query_pos == batch_pos[i];
        });
        if (found == views.end()) {
            return refusal::token_without_view;
        }
    }

    // ---- step 2: the view set itself still validates ----
    // The fork calls prepare_attention_views() here (llama-kv-cache.cpp:2739) and
    // returns its verdict, so a batch that agrees with a stale view set is refused.
    // The reason comes from validate() unchanged: the caller needs to know which
    // rule broke, not that "the batch was bad".
    return validate(views, cells, lim);
}

} // namespace tq::reader
