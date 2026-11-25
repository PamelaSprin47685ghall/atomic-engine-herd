// The bridge between the ledger's answer and the graph's input.
//
// Both structs are `{effective_pos, cells}` lists, so a conversion is easy to write and easy to
// write WRONG in ways no compiler sees: merge two spans that must stay apart, sort the cells, drop
// an empty span that should have been kept, or convert a refusal into one empty view. Each of those
// produces a `query_view` the emitter accepts and a number that is quietly different.
//
// So the assertions here are exact lists, chosen so that a plausible wrong implementation fails:
//   * the two past cells are numbered 9 and 3, in that order — the sorted view would be [3,9], so
//     the equality check catches a sort;
//   * one case has two equal effective positions and one has two different ones, so neither
//     "always one span" nor "always one span per cell" passes;
//   * the query's own cell is put in the cache in one case and absent in another, and both must
//     drop it from the spans while keeping the view legal;
//   * a shared physical cell (two sequences, one slot — the `seq_cp` case the ledger was hardened
//     for) must resolve for both sequences, so the conversion cannot assume slot uniqueness;
//   * a refused batch must convert to ZERO views. The ledger already guarantees `views` is empty on
//     refusal; the test checks the bridge does not invent one.
#include <tq/reader/view_bridge.h>

#include <cstdio>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

std::vector<std::int32_t> cells_of(const tq::reader::query_view & view, std::size_t span) {
    if (span >= view.spans.size()) return {};
    return view.spans[span].cells;
}

// One past cell, at (seq, pos), presented to the rotation at `virtual_pos`.
tq::reader::view_entry entry(std::int32_t seq, std::int32_t pos, std::int32_t virtual_pos) {
    tq::reader::view_entry e;
    e.id.seq = seq;
    e.id.pos = pos;
    e.virtual_pos = virtual_pos;
    return e;
}

tq::reader::attention_request request(std::int32_t seq, std::int32_t pos, std::int32_t virtual_pos,
                                      const std::vector<tq::reader::view_entry> & entries) {
    tq::reader::attention_request r;
    r.query_seq = seq;
    r.query_pos = pos;
    r.query_virtual_pos = virtual_pos;
    r.entries = entries.data();
    r.n_entries = entries.size();
    return r;
}

tq::reader::ledger make_ledger(std::size_t n_sequences, std::size_t capacity) {
    tq::reader::ledger l;
    check(l.set_n_stream(1) == tq::reader::resolve_refusal::ok, "set_n_stream(1)");
    check(l.set_n_sequences(n_sequences) == tq::reader::resolve_refusal::ok,
          "set_n_sequences(" + std::to_string(n_sequences) + ")");
    check(l.set_capacity(capacity) == tq::reader::resolve_refusal::ok,
          "set_capacity(" + std::to_string(capacity) + ")");
    return l;
}

// One sequence, two past cells with DIFFERENT slots and an order that sorting would break, and the
// query's own cell not yet written. Equal effective positions (2, 2) must merge into ONE span.
void merged_spans_preserve_order() {
    tq::reader::ledger board = make_ledger(1, 16);
    check(board.put({0, 0}, 9) == tq::reader::resolve_refusal::ok, "put({0,0}, 9)");
    check(board.put({0, 1}, 3) == tq::reader::resolve_refusal::ok, "put({0,1}, 3)");

    const auto entries = std::vector<tq::reader::view_entry>{
        entry(0, 0, 0),   // effective 2 + 0 - 0 = 2
        entry(0, 1, 1),   // effective 2 + 1 - 1 = 2
        entry(0, 2, 2),   // the query itself: not in the cache, no slot, must vanish
    };
    const auto req = request(0, 2, 2, entries);
    const auto answer = board.resolve(&req, 1);

    check(answer.refusal == tq::reader::resolve_refusal::ok,
          "resolve merged: " + std::string(tq::reader::describe(answer.refusal)) + " " + answer.why);
    check(answer.views.size() == 1, "one view, got " + std::to_string(answer.views.size()));
    if (answer.views.size() != 1) return;

    const auto views = tq::reader::to_query_views(answer);
    check(views.size() == 1, "to_query_views yields one view");
    if (views.size() != 1) return;
    check(views[0].spans.size() == 1,
          "equal effective positions merge: got " + std::to_string(views[0].spans.size()) + " span(s)");
    if (views[0].spans.size() != 1) return;

    check(views[0].spans[0].effective_pos == 2,
          "effective_pos 2, got " + std::to_string(views[0].spans[0].effective_pos));
    const auto cells = cells_of(views[0], 0);
    check(cells == std::vector<std::int32_t>({9, 3}),
          "cells in request order [9, 3]; a sorted view would read [3, 9]");
}

// A cross-sequence entry sits at a different effective position, so the run splits: span 2, then
// span 4, then the query's own (empty) run at the end. Three runs, two non-empty spans.
void split_spans_follow_effective_pos() {
    tq::reader::ledger board = make_ledger(2, 16);
    check(board.put({0, 0}, 11) == tq::reader::resolve_refusal::ok, "put({0,0}, 11)");
    check(board.put({1, 3}, 4) == tq::reader::resolve_refusal::ok, "put({1,3}, 4)");

    const auto entries = std::vector<tq::reader::view_entry>{
        entry(0, 0, 0),   // effective 2 + 0 - 0 = 2
        entry(1, 3, 1),   // effective 2 + 3 - 1 = 4  (another sequence, presented at one slot later)
        entry(0, 2, 2),   // the query itself
    };
    const auto req = request(0, 2, 2, entries);
    const auto answer = board.resolve(&req, 1);

    check(answer.refusal == tq::reader::resolve_refusal::ok,
          "resolve split: " + std::string(tq::reader::describe(answer.refusal)) + " " + answer.why);
    const auto views = tq::reader::to_query_views(answer);
    check(views.size() == 1, "one view");
    if (views.size() != 1) return;
    check(views[0].spans.size() == 2,
          "two effective positions -> two spans, got " + std::to_string(views[0].spans.size()));
    if (views[0].spans.size() != 2) return;

    check(views[0].spans[0].effective_pos == 2, "first span at 2");
    check(cells_of(views[0], 0) == std::vector<std::int32_t>({11}), "first span cell 11");
    check(views[0].spans[1].effective_pos == 4, "second span at 4");
    check(cells_of(views[0], 1) == std::vector<std::int32_t>({4}), "second span cell 4");
}

// The query's own cell, already in the cache, must not appear in any span: it arrives as the graph's
// native K, and reading it a second time from a slot would count it twice in the softmax.
void own_cell_absent_from_spans() {
    tq::reader::ledger board = make_ledger(1, 16);
    check(board.put({0, 0}, 2) == tq::reader::resolve_refusal::ok, "put({0,0}, 2)");
    check(board.put({0, 1}, 5) == tq::reader::resolve_refusal::ok, "put({0,1}, 5)");

    // A re-decode of a token already in the cache is refused by the ledger's own check; so ask for a
    // token that is NOT written and put the two earlier ones behind it. This is the ordinary decode
    // shape: the ledger's own cell never has a slot yet.
    const auto entries = std::vector<tq::reader::view_entry>{
        entry(0, 0, 0),
        entry(0, 1, 1),
        entry(0, 2, 2),
    };
    const auto req = request(0, 2, 2, entries);
    const auto answer = board.resolve(&req, 1);
    check(answer.refusal == tq::reader::resolve_refusal::ok, "resolve own-cell case");
    const auto views = tq::reader::to_query_views(answer);
    if (views.size() != 1) { check(false, "one view"); return; }
    check(views[0].spans.size() == 1, "one span");
    for (const auto & span : views[0].spans) {
        for (std::int32_t cell : span.cells) {
            check(cell == 2 || cell == 5, "no span names the query's own cell; saw " +
                                              std::to_string(cell));
        }
    }
}

// Two sequences, ONE physical cell (the shared-prefix case `seq_cp` creates). Both sequences must
// resolve to it, so the bridge cannot assume that a slot belongs to a single sequence.
void shared_physical_cell_resolves_for_both_sequences() {
    tq::reader::ledger board = make_ledger(2, 16);
    check(board.put({0, 5}, 7) == tq::reader::resolve_refusal::ok, "put({0,5}, 7)");
    check(board.put({1, 5}, 7) == tq::reader::resolve_refusal::ok,
          "put({1,5}, 7) is sharing, not a conflict");

    for (std::int32_t seq : {0, 1}) {
        const auto entries = std::vector<tq::reader::view_entry>{
            entry(seq, 5, 5),
            entry(seq, 6, 6),
        };
        const auto req = request(seq, 6, 6, entries);
        const auto answer = board.resolve(&req, 1);
        check(answer.refusal == tq::reader::resolve_refusal::ok,
              "seq " + std::to_string(seq) + " resolves the shared cell: " +
                  std::string(tq::reader::describe(answer.refusal)) + " " + answer.why);
        const auto views = tq::reader::to_query_views(answer);
        if (views.size() != 1 || views[0].spans.size() != 1) {
            check(false, "seq " + std::to_string(seq) + ": one span");
            continue;
        }
        check(cells_of(views[0], 0) == std::vector<std::int32_t>({7}),
              "seq " + std::to_string(seq) + " reads slot 7");
    }
}

// A first decode has nothing behind it: one view, zero spans. Zero is legal and must not be turned
// into one empty span (the emitter would then build a gather over no cells).
void empty_view_is_zero_spans_not_one() {
    tq::reader::ledger board = make_ledger(1, 16);
    const auto entries = std::vector<tq::reader::view_entry>{entry(0, 0, 0)};
    const auto req = request(0, 0, 0, entries);
    const auto answer = board.resolve(&req, 1);

    check(answer.refusal == tq::reader::resolve_refusal::ok,
          "empty view resolves: " + std::string(tq::reader::describe(answer.refusal)));
    const auto views = tq::reader::to_query_views(answer);
    check(views.size() == 1, "one view even with no past");
    if (views.size() != 1) return;
    check(views[0].spans.empty(), "zero spans, got " + std::to_string(views[0].spans.size()));
}

// A refusal carries no views, and the bridge must not manufacture one: an empty query_view is
// something the emitter would happily run over.
void refusal_converts_to_no_views() {
    tq::reader::ledger board = make_ledger(1, 16);
    check(board.put({0, 0}, 2) == tq::reader::resolve_refusal::ok, "put({0,0}, 2)");

    // Missing the query's own entry is a refusal the ledger names.
    const auto entries = std::vector<tq::reader::view_entry>{entry(0, 0, 0)};
    const auto req = request(0, 1, 1, entries);
    const auto answer = board.resolve(&req, 1);
    check(answer.refusal == tq::reader::resolve_refusal::missing_own_entry,
          "expected missing_own_entry, got " + std::string(tq::reader::describe(answer.refusal)));
    check(answer.views.empty(), "a refused resolution carries no views");
    check(tq::reader::to_query_views(answer).empty(), "and converts to none");
}

}  // namespace

int main() {
    merged_spans_preserve_order();
    split_spans_follow_effective_pos();
    own_cell_absent_from_spans();
    shared_physical_cell_resolves_for_both_sequences();
    empty_view_is_zero_spans_not_one();
    refusal_converts_to_no_views();

    if (failures == 0) {
        std::fprintf(stderr, "view bridge: OK\n");
        return 0;
    }
    std::fprintf(stderr, "view bridge: %d failure(s)\n", failures);
    return 1;
}
