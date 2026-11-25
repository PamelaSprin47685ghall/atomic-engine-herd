// tq/reader/view_bridge.h — the resolved ledger view, in the form the graph consumes.
//
// THE SEAM THIS CLOSES
// --------------------
// Two halves of the reader lane already exist and have never met:
//
//   * `ledger::resolve()` answers with `resolved_view`, whose spans carry `effective_pos` and a
//     vector of physical slots. That is the side table §7.3 of HANDOFF.md lists as missing, and it
//     is not missing: it is in `index_ledger.h`, tested by `tq_reader.index_ledger`.
//   * `emit_reader_attention` / `emit_physical_attention` / `station::open` all take `query_view`,
//     whose spans carry `effective_pos` and a vector of physical cells.
//
// The two structs have the same shape and different names, and no code crossed between them. So the
// index the ledger computes could not be handed to any emitter, and the emitters could only be fed
// by hand-built fixtures — which is exactly what every existing reader test does.
//
// WHY THIS IS A FUNCTION AND NOT A TYPEDEF
// ----------------------------------------
// The shapes agree today; the meanings agree only because `resolve()` groups the way the graph
// needs. `resolve()` builds one span per run of consecutive entries with equal `effective_pos` and
// drops a span whose indices are empty (the query's own cell, not yet written, arrives as the
// graph's native K). `query_view` is consumed the same way: one rotation per span, cells gathered
// in the order given.
//
// Copying rather than aliasing is deliberate. `resolved_span::indices` is const-once-issued and
// borrowed by the caller for one decode; the emitter's `query_view` may be stored in a `station`
// (`graph_station.h` keeps a COPY "no reference to the caller's temporary"), and a station outlives
// the resolution it was opened from. A view that shared the resolution's storage would dangle the
// moment the resolution went out of scope, and nothing in the station would say so.
//
// ORDER IS PRESERVED, and it is not cosmetic. Within a span, `indices` follows the request's
// `virtual_pos` order (the ledger refuses an unordered batch); the emitter gathers in the order it
// receives, so the weights come back in that order too. Attention is permutation invariant, so a
// reordering would still be *correct* and would stop being *comparable* — every bitwise check
// against upstream would start disagreeing in the last digits for no reason a reader could see.
//
// WHAT THIS DOES NOT DO
// ---------------------
// It does not compute the index, validate it, or cache it. The ledger owns those, and duplicating
// even the cheap checks here would create a second place to keep right. It also does not decide
// visibility: `resolve()` was given the visible entries, and a cell absent from them is absent from
// the answer. A hidden cell is expressed by not naming it, not by a sentinel.
#pragma once

#include <utility>
#include <vector>

#include <tq/reader/attention.h>
#include <tq/reader/index_ledger.h>

namespace tq::reader {

// One resolved view, as the graph reads it.
//
// Total: every `resolved_view` the ledger can issue converts, including the empty one (a first
// decode with no past has nothing to attend to but itself — zero spans is a legal view, not an
// error, and the emitter's own shape checks are what decide whether it can run).
inline query_view to_query_view(const resolved_view & resolved) {
    query_view view;
    view.spans.reserve(resolved.spans.size());
    for (const resolved_span & span : resolved.spans) {
        span_keys keys;
        keys.effective_pos = span.effective_pos;
        keys.cells = span.indices;
        view.spans.push_back(std::move(keys));
    }
    return view;
}

// A whole resolution, in view order.
//
// `resolution::views` is empty for any refusal, and this preserves that: a refused batch converts
// to no views rather than to one empty view. A caller must still check `refusal == ok` before using
// these — an empty result is what both a refusal and a zero-view request look like, and the
// resolution carries the difference. Returning views without that check is the silent path this
// file will not take by itself: `to_query_views` is a conversion, not an endorsement.
inline std::vector<query_view> to_query_views(const resolution & answer) {
    std::vector<query_view> views;
    views.reserve(answer.views.size());
    for (const resolved_view & resolved : answer.views) {
        views.push_back(to_query_view(resolved));
    }
    return views;
}

} // namespace tq::reader
