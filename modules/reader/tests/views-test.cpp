// The reader lane's rules, tested without a model, a cache, or a GPU.
//
// Every case names a rule the fork enforced inside llama-kv-cache.cpp. A test that
// only checked "a valid view resolves" would pass with the validation deleted, so
// each rejection below is pinned by the reason it must produce.
//
// One rule shapes every fixture: entry virtual positions must ascend. That is not
// decoration -- the spans are built by walking the entries in order, so entries
// out of virtual order would produce spans whose rotations are not monotone, and
// the graph would read keys at positions the query never faced. The first draft of
// these tests violated it and the validator rejected the fixture, which is the
// validator doing its job.
#include <tq/reader/views.h>

#include <algorithm>
#include <cstdio>

using namespace tq::reader;

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void check_refusal(refusal got, refusal want, const char * what) {
    if (got != want) {
        std::fprintf(stderr, "FAIL: %s (got %s, want %s)\n", what, name(got), name(want));
        ++failures;
    }
}

// Two sequences of three tokens each, all stored: cells 0..2 for sequence 0,
// 3..5 for sequence 1. The queries below are at position 3, which is not in the
// cache -- the reader decodes a token that has not been written yet.
cell_map stored_cells() {
    return {
        {{0, 0}, 0}, {{0, 1}, 1}, {{0, 2}, 2},
        {{1, 0}, 3}, {{1, 1}, 4}, {{1, 2}, 5},
    };
}

// The same cache AFTER this batch's keys are written. The fork states the reason
// this second map exists (src/llama-attention-view.h): "Logical identities are
// resolved after KV slot preparation. No physical index survives a decode call."
// resolve() is defined on this phase, so the query's own position now has a cell --
// cell 6 for sequence 0, cell 7 for sequence 1, because allocation is the cache's
// business and the test only needs the indices to be real.
cell_map written_cells() {
    cell_map cells = stored_cells();
    cells[{0, 3}] = 6;
    cells[{1, 3}] = 7;
    return cells;
}

limits default_limits() {
    limits lim;
    lim.max_views  = 512;
    lim.cache_size = 1024;
    return lim;
}

// A query at (seq, 3) that sees its own stored keys and its own pending position,
// with the identity virtual map. Ascending by construction.
view identity_view(seq_id seq, pos_t query_pos) {
    view v;
    v.query_seq = seq;
    v.query_pos = query_pos;
    v.query_virtual_pos = query_pos;
    for (pos_t p = 0; p < query_pos; ++p) {
        v.entries.push_back({seq, p, p});
    }
    v.entries.push_back({seq, query_pos, query_pos});
    return v;
}

// Effective position is query_virtual + key_pos - virtual_pos. Entries sharing an
// effective position share one rotation, so they collapse into one span.
void spans_group_by_effective_position() {
    view v;
    v.query_seq = 0;
    v.query_pos = 3;
    v.query_virtual_pos = 3;
    v.entries = {
        {0, 0, 0},  // effective 3 + 0 - 0 = 3
        {0, 1, 1},  // effective 3          -> same span
        {0, 2, 2},  // effective 3          -> same span
        {0, 3, 3},  // own entry, effective 3
    };
    const auto r = resolve({v}, written_cells(), default_limits());
    check_refusal(r.why, refusal::none, "a valid identity view resolves");
    check(r.views.size() == 1, "one view in, one resolution out");
    if (r.views.size() == 1) {
        check(r.views[0].spans.size() == 1, "equal effective positions collapse into one span");
        check(r.views[0].indices.size() == 4, "all four entries are present, in view order");
        const auto & idx = r.views[0].indices;
        // The own entry resolves to its real cell, because by this phase the cache
        // has written it. A -1 here would be handed to ggml_get_rows as a row index:
        // the fork never produces such a marker, and neither may this port.
        check(idx.size() == 4 && idx[3] == 6, "the own entry resolves to its written cell");
        check(idx[0] == 0 && idx[2] == 2, "stored entries resolve to their cells");
        check(std::find(idx.begin(), idx.end(), -1) == idx.end(), "no placeholder index survives");
        check(r.views[0].spans[0].indices.size() == 4, "the own entry joins the query's span");
    }
}

// Keys from another sequence join the same softmax. Their virtual positions must
// still ascend, so a reader that wants a foreign key late in the span gives it a
// late virtual position -- not the identity one.
void foreign_keys_join_the_same_span() {
    view v;
    v.query_seq = 0;
    v.query_pos = 3;
    v.query_virtual_pos = 3;
    v.entries = {
        {0, 0, 0},  // effective 3
        {0, 1, 1},  // effective 3
        {1, 0, 2},  // foreign, effective 3 + 0 - 2 = 1
        {0, 3, 3},  // own,     effective 3
    };
    const auto r = resolve({v}, written_cells(), default_limits());
    check_refusal(r.why, refusal::none, "a view mixing two sequences resolves");
    if (r.views.size() == 1) {
        check(r.views[0].spans.size() == 3, "the foreign key opens its own span, and the own entry reopens the query's");
        check(r.views[0].spans[0].effective_pos == 3, "the first span keeps the query rotation");
        check(r.views[0].spans[1].effective_pos == 1, "the foreign span is rotated to its own position");
        check(r.views[0].spans[2].effective_pos == 3, "the own entry is rotated back to the query's position");
        check(r.views[0].indices.size() == 4, "every entry is present");
        check(r.views[0].indices[2] == 3, "the foreign key resolves to sequence 1's first cell");
    }
}

void different_virtual_positions_split_spans() {
    view v;
    v.query_seq = 0;
    v.query_pos = 3;
    v.query_virtual_pos = 3;
    v.entries = {
        {0, 0, 0},  // effective 3 + 0 - 0 = 3
        {0, 2, 1},  // effective 3 + 2 - 1 = 4 -> new span
        {0, 1, 2},  // effective 3 + 1 - 2 = 2 -> new span
        {0, 3, 3},  // own,     effective 3 -> new span
    };
    const auto r = resolve({v}, written_cells(), default_limits());
    check_refusal(r.why, refusal::none, "a view with shifted rotations resolves");
    if (r.views.size() == 1) {
        check(r.views[0].spans.size() == 4, "four distinct effective positions make four spans");
        check(r.views[0].spans[0].effective_pos == 3, "first span keeps the query's rotation");
        check(r.views[0].spans[1].effective_pos == 4, "second span is rotated one later");
        check(r.views[0].spans[2].effective_pos == 2, "third span is rotated earlier still");
        check(r.views[0].spans[3].effective_pos == 3, "order is preserved, not sorted");
    }
}

void every_rejection_is_named() {
    const auto cells = stored_cells();
    const auto lim = default_limits();
    const view good = identity_view(0, 3);

    check_refusal(validate({}, cells, lim), refusal::no_views, "an empty view set");

    {
        limits l = lim;
        l.n_streams = 2;
        check_refusal(validate({good}, cells, l), refusal::multi_stream, "two streams");
        l.n_streams = 1;
        l.shifted = true;
        check_refusal(validate({good}, cells, l), refusal::shifted, "a shifted cache");
    }

    {
        view v = good;
        v.query_pos = 2;
        v.query_virtual_pos = 2;
        v.entries = {{0, 0, 0}, {0, 1, 1}, {0, 2, 2}};
        check_refusal(validate({v}, cells, lim), refusal::query_already_stored,
                      "a query position that is already stored");
    }

    {
        const view a = identity_view(0, 3);
        const view b = identity_view(0, 3);
        check_refusal(validate({a, b}, cells, lim), refusal::duplicate_query_seq,
                      "two views for one sequence");
    }

    // A key the cache does not hold: the reader would attend over a set it cannot
    // resolve, so the whole batch is refused rather than shrinking the span.
    {
        view v = identity_view(0, 3);
        v.entries.insert(v.entries.begin() + 2, {1, 9, 2});
        check_refusal(validate({v}, cells, lim), refusal::key_missing,
                      "an entry the cache does not hold");
    }

    // The same physical cell twice: those keys would be weighted twice, which is a
    // silent numeric change rather than a duplicate to be tolerated.
    {
        view v = identity_view(0, 3);
        v.entries = {{0, 0, 0}, {0, 0, 1}, {0, 3, 3}};
        check_refusal(validate({v}, cells, lim), refusal::key_duplicated,
                      "the same cell twice in one view");
    }

    // Ascending is a rule, not a nicety: out-of-order rotations would produce spans
    // the query never faced.
    {
        view v = identity_view(0, 3);
        v.entries = {{0, 1, 2}, {0, 0, 0}, {0, 3, 3}};
        check_refusal(validate({v}, cells, lim), refusal::virtual_not_ascending,
                      "entry virtual positions out of order");
    }

    // Own entry rotated differently from the query: the token would attend to
    // itself at a rope position its K was never written with.
    {
        view v = identity_view(0, 3);
        v.entries = {{0, 0, 0}, {0, 1, 1}, {0, 3, 2}};
        check_refusal(validate({v}, cells, lim), refusal::own_entry_position,
                      "own entry rotated differently from the query");
    }

    // A rotation beyond the query's own: the key would be placed in the query's
    // future, which no stored key can occupy.
    {
        view v = identity_view(0, 3);
        v.entries = {{0, 0, 4}, {0, 3, 3}};
        check_refusal(validate({v}, cells, lim), refusal::virtual_beyond_query,
                      "entry rotated past the query");
    }

    // No own entry: the query does not see the token it is decoding.
    {
        view v = identity_view(0, 3);
        v.entries.pop_back();
        check_refusal(validate({v}, cells, lim), refusal::own_entry_missing, "no own entry");
    }

    // A key after the query in the same sequence has not been computed with respect
    // to this query.
    {
        view v = identity_view(0, 3);
        v.entries.insert(v.entries.begin(), {0, 4, 1});
        check_refusal(validate({v}, cells, lim), refusal::entry_after_query, "a key after the query");
    }

    // A key that is itself a query in this batch is being written now.
    {
        const view a = identity_view(0, 3);
        view b = identity_view(1, 3);
        b.entries.insert(b.entries.begin(), {0, 3, 3});
        check_refusal(validate({a, b}, cells, lim), refusal::key_is_pending,
                      "an entry that names another query in the batch");
    }
}

void batch_agreement() {
    const std::vector<view> views = {identity_view(0, 3), identity_view(1, 3)};

    check_refusal(validate_batch(views, {0, 1}, {3, 3}, {1, 1}, false, stored_cells(), default_limits()),
                  refusal::none, "a batch that matches the views");
    check_refusal(validate_batch(views, {0}, {3}, {1}, false, stored_cells(), default_limits()),
                  refusal::batch_count_mismatch, "one token fewer than views");
    check_refusal(validate_batch(views, {0, 1}, {3, 3}, {1}, false, stored_cells(), default_limits()),
                  refusal::batch_count_mismatch, "n_seq_id shorter than the batch");
    check_refusal(validate_batch(views, {0, 1}, {3, 3}, {1, 1}, true, stored_cells(), default_limits()),
                  refusal::batch_has_embd, "an embedding batch");
    check_refusal(validate_batch(views, {0, 0}, {3, 3}, {1, 1}, false, stored_cells(), default_limits()),
                  refusal::duplicate_query_seq, "two tokens in one sequence");
    check_refusal(validate_batch(views, {0, 1}, {3, 4}, {1, 1}, false, stored_cells(), default_limits()),
                  refusal::token_without_view, "a token at a position no view describes");
    check_refusal(validate_batch({}, {}, {}, {}, false, stored_cells(), default_limits()), refusal::none,
                  "no views means no constraints");

    // The fork's `batch.n_seq_id[i] != 1`: a token shared by two sequences makes the
    // (seq, pos) pair that identifies a view ambiguous, so it is refused rather than
    // resolved to one of them.
    check_refusal(validate_batch(views, {0, 1}, {3, 3}, {2, 1}, false, stored_cells(), default_limits()),
                  refusal::token_multi_seq, "a token claiming two sequences");
    check_refusal(validate_batch(views, {0, 1}, {3, 3}, {1, 0}, false, stored_cells(), default_limits()),
                  refusal::token_multi_seq, "a token claiming no sequence");

    // Step 2: the batch agrees, but the view set no longer validates. The fork
    // re-runs prepare_attention_views here, so this batch must be refused -- and the
    // reason must name the view problem, not the batch.
    const std::vector<view> stale = {identity_view(0, 3), view{1, 3, 3, {{1, 0, 0}}}};
    check_refusal(validate_batch(stale, {0, 1}, {3, 3}, {1, 1}, false, stored_cells(), default_limits()),
                  refusal::own_entry_missing,
                  "a batch that agrees with an invalid view set is refused, naming the view");
    // A batch that agrees with a view whose entry is missing from the cache: valid
    // view shape, absent key -- refused for that reason and no other. The key is
    // removed from the map rather than never added, which is what a mid-run
    // eviction looks like.
    cell_map sparse = stored_cells();
    sparse.erase({0, 2});
    const std::vector<view> missing_key = {view{0, 3, 3, {{0, 0, 0}, {0, 2, 2}, {0, 3, 3}}}};
    check_refusal(validate_batch(missing_key, {0}, {3}, {1}, false, sparse, default_limits()),
                  refusal::key_missing, "step 2 reports the missing key, not a batch fault");
}

// A rejected set must produce no resolutions at all: a partially filled result
// would be consumed as if it were complete, and the run would mix reader and stock
// semantics in one batch.
void rejection_is_all_or_nothing() {
    view bad = identity_view(0, 3);
    bad.entries.pop_back();
    // validate() is the phase that judges the set's shape; resolve() runs after the
    // write and would report key_missing for this fixture, naming the wrong problem.
    check_refusal(validate({identity_view(1, 3), bad}, stored_cells(), default_limits()),
                  refusal::own_entry_missing, "the set is rejected");

    // The same fixture handed to resolve() before the write: it must refuse rather
    // than fill the missing cell with a stand-in.
    const auto early = resolve({identity_view(1, 3), bad}, written_cells(), default_limits());
    check_refusal(early.why, refusal::none, "a complete set resolves in the written phase");
    check(early.views.size() == 2, "both views resolve once their keys exist");

    const auto incomplete = resolve({view{0, 3, 3, {{0, 0, 0}, {0, 3, 3}}}}, stored_cells(), default_limits());
    check_refusal(incomplete.why, refusal::key_missing,
                  "resolving before the write refuses: the own key is not in the cache yet");
    check(incomplete.views.empty(), "a failed resolution yields no partial spans");

    const auto good = resolve({identity_view(0, 3)}, written_cells(), default_limits());
    const auto again = resolve({identity_view(0, 3)}, written_cells(), default_limits());
    check(good.why == again.why && good.views.size() == again.views.size() &&
              good.views[0].indices == again.views[0].indices,
          "resolution is deterministic");
}

// The two phases, stated as one test because the port's worst defect was collapsing
// them: the earlier version called validate() from resolve() and then resolved the
// query's own entry to -1 to escape the contradiction it had created. The fork does
// Bounds come before phases: this test is about the two independent limits, and
// the two-phase test follows it. (The two-phase rationale lives in views.h and the
// README, where the resolve() contract is defined.)
// Two bounds, two roles. The fork refuses a set of more than LLAMA_MAX_SEQ views
// AND separately refuses a sequence id the cache does not have
// (src/llama-kv-cache.cpp:2652, :2660). Collapsing them -- as an earlier version of
// this port did, with a single `max_views = 512` -- accepts twice the permitted view
// count and accepts ids from sequences that do not exist. Both mistakes are silent:
// an out-of-range id simply finds no cells, so the view resolves to nothing instead
// of being refused, and the caller sees an empty span list rather than an error.
void seq_bounds_are_separate_from_view_count() {
    limits two_sequences = default_limits();
    two_sequences.n_seqs = 2;

    // Sequence 2 is out of range for a two-sequence cache, while the view count is
    // still far below its own bound.
    check_refusal(validate({identity_view(2, 3)}, stored_cells(), two_sequences),
                  refusal::query_seq_invalid, "a query sequence the cache does not have");

    view foreign = identity_view(0, 3);
    foreign.entries.insert(foreign.entries.begin() + 2, {2, 1, 1});
    check_refusal(validate({foreign}, stored_cells(), two_sequences),
                  refusal::entry_seq_invalid, "an entry sequence the cache does not have");

    // The two defaults are the fork's own numbers, and they stay distinguishable.
    check(limits{}.max_views == 256, "the view-count bound is LLAMA_MAX_SEQ (256)");
    check(limits{}.n_seqs == 256, "the default sequence bound is the unpipelined cache's");

    // The count bound is independent: one view allowed, two supplied.
    limits one_view = default_limits();
    one_view.max_views = 1;
    check_refusal(validate({identity_view(0, 3), identity_view(1, 3)}, stored_cells(), one_view),
                  refusal::too_many_views, "more views than the view-count bound");
}

void resolution_is_a_second_phase() {
    const view v = identity_view(0, 3);

    // Phase A: before the batch is written.
    check_refusal(validate({v}, stored_cells(), default_limits()), refusal::none,
                  "phase A accepts the view while the query is still unstored");

    // Phase B on the wrong map: the fork's answer is a refusal, not a marker.
    const auto early = resolve({v}, stored_cells(), default_limits());
    check_refusal(early.why, refusal::key_missing,
                  "phase B on the pre-write map refuses instead of inventing an index");

    // Phase B on the right map: every index is a real cache row, own entry included.
    const auto late = resolve({v}, written_cells(), default_limits());
    check_refusal(late.why, refusal::none, "phase B on the post-write map resolves");
    if (late.views.size() == 1 && late.views[0].indices.size() == 4) {
        check(late.views[0].indices[3] == 6, "the own entry is the cell the cache wrote");
        check(late.views[0].spans.size() == 1, "the own entry shares the query's rotation");
    }

    // Ambiguous identity: the cache marks a (seq, pos) claimed by two cells as -1
    // (llama-kv-cache.cpp attention_cells). Resolving it would pick one at random.
    cell_map ambiguous = written_cells();
    ambiguous[{0, 1}] = -1;
    check_refusal(resolve({v}, ambiguous, default_limits()).why, refusal::key_cell_negative,
                  "an ambiguous cache identity is refused, not guessed");
}

} // namespace

int main() {
    spans_group_by_effective_position();
    foreign_keys_join_the_same_span();
    different_virtual_positions_split_spans();
    every_rejection_is_named();
    batch_agreement();
    rejection_is_all_or_nothing();
    resolution_is_a_second_phase();
    seq_bounds_are_separate_from_view_count();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "reader view contract holds\n");
    return 0;
}
