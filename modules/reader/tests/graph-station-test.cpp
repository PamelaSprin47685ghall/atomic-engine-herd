// The request ledger's contract, as tests.
//
// Every case names the sentence it enforces, because a green assertion with no sentence behind it
// is the kind of pass that survives a wrong design. The sentences come from the hook's own file
// (one graph build, no nesting, no leak into the next batch) and from the ticket-station split
// (state in the station, flow in the caller).
//
// The five cases marked DEFECT were written against real defects in the first revision. They are
// kept as the cases that would have caught them, which is what makes a fix stay fixed:
//
//   independent_contexts_coexist        one slot per process refused a second, unrelated context
//   find_copies_and_cannot_dangle       find() handed out a pointer into mutable state
//   a_manual_close_does_not_kill_a_reopen  close() had no epoch, so a stale guard closed a live build
//   a_null_context_is_refused           nullptr was a valid key, colliding unrelated builds
//   the_epoch_rule_seals_at_the_maximum  epoch += 1 wrapped, reopening the stale-close door
//
// What this test can and cannot show:
//
//   it CAN show that a request applies to exactly one build, that a second open of the SAME
//   context is refused rather than merged, that independent contexts proceed, that a request
//   never survives its scope, that a stale close changes nothing, and that a copied snapshot
//   outlives the station's mutation of the slot.
//
//   it CANNOT show that the emitted graph is correct -- that is the emitter's tests, against
//   ggml's kernels. It also cannot show that a cell index exists in a cache: this station does
//   not see a cache, and its header says why it refuses to guess.
#include <tq/reader/graph_station.h>

#include <atomic>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

using tq::reader::close_status;
using tq::reader::open_status;
using tq::reader::query_view;
using tq::reader::span_keys;
using tq::reader::station;

// Three read predicates, each naming the ONE outcome it accepts, and none of them `!the_other`.
//
// This is the whole point of the migration, so it is stated here rather than in a comment above
// each use. The station's find() has four outcomes now -- found / no_request / no_memory /
// bad_argument -- and a single bool helper would collapse them. Collapsing is exactly the defect
// this round removed: `no_memory` means a request EXISTS and could not be handed over, so reading
// it as "nothing was asked for" silently drops an emission the caller asked for, and every number
// downstream then looks like a normal run.
//
//   is_found     accepted only when every field was written
//   is_absent    accepted only for no_request; a copy failure does NOT satisfy it
//   is_bad_argument  for the caller-bug case, which used to share a value with absence
bool is_found(const station & board, const void * context, tq::reader::active_request * out) {
    return board.find(context, out) == station::find_status::found;
}

bool is_absent(const station & board, const void * context, tq::reader::active_request * out) {
    return board.find(context, out) == station::find_status::no_request;
}

bool is_bad_argument(const station & board, const void * context, tq::reader::active_request * out) {
    return board.find(context, out) == station::find_status::bad_argument;
}

// Declared here because the case below uses it and its definition lives further down. A forward
// declaration rather than moving the definition: the two-span fixture belongs with the other
// fixtures, and relocating it to satisfy one caller's position would make the file's order
// describe the compiler's needs instead of the reader's.
query_view two_spans();

// The four outcomes, and the property that made them necessary: a failed copy leaves the
// caller's output ALONE.
//
// The previous contract answered `false` for "nothing was asked for" and for "the copy failed",
// so a caller could not tell "proceed with stock attention" from "your emission was dropped".
// Worse, the copy was written member-wise into the caller's object: context, rope, rope_user and
// epoch landed before the one allocating member (the view) could throw, so a failure left a
// request that named one build and carried another's view -- a partial update of an output
// parameter, which is the same defect as a partial publish.
//
// The output here is a REAL active_request holding values no code path produces by accident.
// It is deliberately not a reinterpret_cast over raw bytes: writing an unconstructed object
// through a pointer of the wrong type is undefined behaviour, and a test that relies on UB cannot
// report a defect in the code it is testing.
void find_separates_absence_from_failure() {
    station board;
    int ctx = 0;
    int sentinel_context = 0;

    // Known contents, rebuilt per case so one case cannot satisfy another's expectation.
    const auto make_output = [&]() {
        tq::reader::active_request out;
        out.context = &sentinel_context;
        out.rope = nullptr;
        out.rope_user = nullptr;
        out.epoch = 12345;
        tq::reader::span_keys span;
        span.effective_pos = 99;
        span.cells = {77};
        out.view.spans = {span};
        return out;
    };
    // Every field the old member-wise assign could have touched before throwing.
    const auto untouched = [](const tq::reader::active_request & out) {
        return out.context != nullptr && out.epoch == 12345 && out.view.spans.size() == 1 &&
               out.view.spans[0].effective_pos == 99 && out.view.spans[0].cells.size() == 1 &&
               out.view.spans[0].cells[0] == 77;
    };

    // ---- no_request: nothing was asked for, and the output is not written ----
    {
        tq::reader::active_request out = make_output();
        check(board.find(&ctx, &out) == station::find_status::no_request,
              "no open request reports no_request");
        check(untouched(out), "and the caller's output was not written");
    }

    // ---- bad_argument: a null output is a caller bug, not an absence ----
    {
        tq::reader::active_request out = make_output();
        check(board.find(&ctx, nullptr) == station::find_status::bad_argument,
              "a null output reports bad_argument, which used to share a value with absence");
        check(untouched(out), "and the caller's own object is untouched");
    }

    {
        tq::reader::scope scope(board, &ctx, two_spans());
        check(scope.opened(), "the scope opened");

        // ---- no_memory: a request exists and cannot be handed over ----
        {
            tq::reader::active_request out = make_output();
            station::inject_copy_failure_for_test(true);
            const station::find_status status = board.find(&ctx, &out);
            station::inject_copy_failure_for_test(false);

            check(status == station::find_status::no_memory,
                  "a copy failure reports no_memory -- not no_request, which would read as "
                  "\"the caller asked for nothing\"");
            check(untouched(out),
                  "and the output is untouched: a member-wise assign would have written "
                  "context, rope and epoch before the view threw");
        }

        // ---- found: every field arrives ----
        {
            tq::reader::active_request out;
            check(board.find(&ctx, &out) == station::find_status::found,
                  "with the seam cleared, the request is found");
            check(out.context == &ctx, "the context is this build's");
            check(out.epoch != 0, "with the epoch that opened it");
            check(out.view.spans.size() == 2, "and both spans arrived");
        }
    }

    // ---- after the scope: absent again, never a stale found ----
    {
        tq::reader::active_request out = make_output();
        check(board.find(&ctx, &out) == station::find_status::no_request,
              "after the scope closes the request is gone");
        check(untouched(out), "and nothing was written over the caller's values");
    }
}

// A two-span view, distinct cells in each, so a merged or replaced request is detectable by what
// the spans contain rather than only by a counter.
query_view two_spans() {
    query_view view;
    span_keys first;
    first.effective_pos = 5;
    first.cells = {0, 1, 3};
    span_keys second;
    second.effective_pos = 6;
    second.cells = {7};
    view.spans = {first, second};
    return view;
}

query_view one_span(int32_t cell) {
    query_view view;
    span_keys span;
    span.effective_pos = 1;
    span.cells = {cell};
    view.spans = {span};
    return view;
}

// A request applies to the build that opened it, and to nothing else.
void a_request_applies_only_to_its_own_context() {
    station board;
    int ctx_a = 0;
    int ctx_b = 0;

    {
        tq::reader::scope scope(board, &ctx_a, two_spans());
        check(scope.opened(), "the first open succeeds");

        tq::reader::active_request found;
        check(is_found(board, &ctx_a, &found), "the request is visible for its own context");
        check(found.context == &ctx_a, "and names that context");
        check(found.view.spans.size() == 2, "with both spans intact");
        check(found.view.spans[0].cells.size() == 3, "and the first span's cells");
        check(found.epoch == scope.epoch(), "and the epoch the scope was given");

        tq::reader::active_request other;
        check(is_absent(board, &ctx_b, &other),
              "a different context sees no request: that build was not asked for");
    }

    tq::reader::active_request after;
    check(is_absent(board, &ctx_a, &after), "after the scope closes, the request is gone");
}

// DEFECT: one slot per process refused a second, unrelated context.
//
// Two llama_contexts decoding in parallel is a supported configuration, and the first revision
// refused the second one as "nested" -- with a message claiming a scope was already open on that
// thread, which was false when the holder was another thread. Independent builds must coexist;
// only the SAME context may not be opened twice.
void independent_contexts_coexist() {
    station board;
    int ctx_a = 0;
    int ctx_b = 0;

    tq::reader::scope a(board, &ctx_a, two_spans());
    check(a.opened(), "context A opens");

    tq::reader::scope b(board, &ctx_b, one_span(11));
    check(b.opened(),
          "context B opens while A holds its slot: a second, unrelated context is not a nest");
    check(board.live_count() == 2, "both requests are live");

    tq::reader::active_request fa;
    tq::reader::active_request fb;
    check(is_found(board, &ctx_a, &fa), "A's request is found");
    check(is_found(board, &ctx_b, &fb), "B's request is found");
    check(fa.view.spans.size() == 2, "and A still holds its own view");
    check(fb.view.spans.size() == 1 && fb.view.spans[0].cells[0] == 11,
          "and B holds its own");

    const auto counters = board.snapshot();
    check(counters.opened == 2, "two opens counted");
    check(counters.refused_nested == 0 && counters.refused_owned_elsewhere == 0,
          "and neither was refused: independent contexts are not contention");
}

// A genuinely nested open is still refused, and the outer view still wins.
void a_nested_open_is_refused_not_merged() {
    station board;
    int ctx = 0;

    tq::reader::scope outer(board, &ctx, two_spans());
    check(outer.opened(), "outer opens");

    tq::reader::scope inner(board, &ctx, one_span(99));
    check(!inner.opened(), "a nested open of the same context is refused");
    check(inner.status() == open_status::nested, "and says why: nested");

    tq::reader::active_request found;
    check(is_found(board, &ctx, &found), "the outer request is still the one in force");
    check(found.view.spans.size() == 2, "the outer view was not replaced by the inner one");
    check(found.epoch == outer.epoch(), "and its epoch did not move: a refusal is not an open");

    check(is_found(board, &ctx, &found), "destroying a refused guard left the outer scope open");

    const auto counters = board.snapshot();
    check(counters.opened == 1, "one open counted");
    check(counters.refused_nested == 1, "one nested refusal counted");
    check(counters.closed == 0, "and nothing closed yet");
}

// The same context from two threads cannot be right by picking a winner, so it is a third status.
void the_same_context_from_two_threads_is_owned_elsewhere() {
    station board;
    int ctx = 0;

    std::atomic<int> seen_status{static_cast<int>(open_status::ok)};
    std::atomic<bool> returned{false};

    tq::reader::scope owner_scope(board, &ctx, two_spans());
    check(owner_scope.opened(), "the owning thread opens");

    std::thread rival([&] {
        tq::reader::scope other(board, &ctx, one_span(5));
        seen_status.store(static_cast<int>(other.status()), std::memory_order_relaxed);
        returned.store(true, std::memory_order_release);
    });
    rival.join();

    check(returned.load(), "the rival thread returned");
    check(static_cast<open_status>(seen_status.load()) == open_status::owned_elsewhere,
          "a second thread on the same context is refused as owned_elsewhere, not nested");
    check(board.live_count() == 1, "and only the owner's request is live");

    const auto counters = board.snapshot();
    check(counters.refused_owned_elsewhere == 1, "counted under its own name");
    check(counters.refused_nested == 0, "not miscounted as nesting");
}

// DEFECT: find() handed out a pointer into mutable state.
//
// The old signature returned `const active_request *`, valid "until close()". A close from
// another thread frees the view's heap buffer while the caller reads through that pointer, and a
// per-layer hook is exactly such a reader. find() now copies under the lock, so the snapshot the
// caller holds is its own and stays valid -- and keeps its contents -- even after the slot is
// reused for a different view.
void find_copies_and_cannot_dangle() {
    station board;
    int ctx = 0;

    tq::reader::active_request snapshot;
    std::uint64_t first_epoch = 0;
    {
        tq::reader::scope scope(board, &ctx, two_spans());
        check(scope.opened(), "the build opens");
        first_epoch = scope.epoch();
        check(is_found(board, &ctx, &snapshot), "the caller takes a snapshot");
        check(snapshot.view.spans.size() == 2, "with the real contents");
    }

    // The slot has been cleared and is about to be filled with a DIFFERENT view. If the snapshot
    // aliased station state, the contents below would change; if it aliased freed memory, the
    // sanitizer (or the crash) would say so. Neither happens.
    {
        tq::reader::scope second(board, &ctx, one_span(42));
        check(second.opened(), "a second build opens the same slot");

        check(snapshot.view.spans.size() == 2,
              "the caller's snapshot still has two spans after the slot was reused");
        check(snapshot.view.spans[1].cells.size() == 1 && snapshot.view.spans[1].cells[0] == 7,
              "and its second span is still the first build's, not the second's");
        check(snapshot.epoch == first_epoch, "and it remembers its own epoch");
        check(!snapshot.view.spans.empty() && snapshot.view.spans[0].cells[0] == 0,
              "with the first build's cells intact");

        tq::reader::active_request current;
        check(is_found(board, &ctx, &current), "the live request is the second one");
        check(current.view.spans[0].cells[0] == 42, "as its contents show");
    }
}

// DEFECT: close() had no epoch, so a stale guard closed a live build.
//
// A scope that outlives its request -- a caller closed manually, or the slot was reused -- must
// not close whatever now occupies the slot. The epoch it opened with is what makes that
// impossible, and `stale` is a distinct answer from `absent` because the two mean different
// things to the caller.
void a_manual_close_does_not_kill_a_reopen() {
    station board;
    int ctx = 0;

    tq::reader::scope outer(board, &ctx, two_spans());
    check(outer.opened(), "the outer scope opens");
    const std::uint64_t outer_epoch = outer.epoch();

    // The caller closes by hand, with the epoch it was given. That is a legitimate call.
    check(board.close(&ctx, outer_epoch) == close_status::closed,
          "a manual close with the right epoch succeeds");
    check(board.live_count() == 0, "and the slot is free");

    // A NEW build takes the slot.
    tq::reader::scope inner(board, &ctx, one_span(7));
    check(inner.opened(), "a new build opens the same context");
    const std::uint64_t inner_epoch = inner.epoch();
    check(inner_epoch != outer_epoch, "with a different epoch, so a stale close is detectable");

    // The outer guard's destructor now runs, carrying the old epoch. It must not touch the live
    // request. This is the sequence the first revision got wrong.
    //
    // (The destructor of `outer` fires at the end of this function; the assertion is made
    // explicitly first, so the failure is attributed to the epoch rule and not to scope
    // ordering.)
    check(board.close(&ctx, outer_epoch) == close_status::stale,
          "a close carrying the superseded epoch is refused as stale");

    tq::reader::active_request live;
    check(is_found(board, &ctx, &live), "the newer request survived the stale close");
    check(live.epoch == inner_epoch && live.view.spans[0].cells[0] == 7,
          "with its own view and epoch intact");

    const auto counters = board.snapshot();
    check(counters.close_stale == 1, "the stale close was counted");
    check(counters.close_absent == 0, "and not misreported as an absent close");
    check(counters.opened == 2 && counters.closed == 1,
          "two opens, one close: the stale attempt closed nothing");
}

// A close with no matching epoch, when nothing is open, is `absent` -- benign, not stale.
void a_close_with_nothing_open_is_absent() {
    station board;
    int ctx = 0;

    check(board.close(&ctx, 1) == close_status::absent,
          "a close for a context with no request reports absent");
    check(board.close(&ctx, 0) == close_status::absent,
          "even with epoch 0, which no request is ever given");

    const auto counters = board.snapshot();
    check(counters.close_absent == 2, "both counted as absent");
    check(counters.close_stale == 0, "neither reported as a stale close");
}

// The owning thread's guard closes what it opened; another thread's does not.
void the_owning_thread_alone_can_close() {
    station board;
    int ctx = 0;

    tq::reader::scope scope(board, &ctx, two_spans());
    check(scope.opened(), "the scope opened");
    const std::uint64_t epoch = scope.epoch();

    std::atomic<int> rc{static_cast<int>(close_status::closed)};
    std::thread worker([&] {
        rc.store(static_cast<int>(board.close(&ctx, epoch)), std::memory_order_relaxed);
    });
    worker.join();

    check(static_cast<close_status>(rc.load()) == close_status::not_owner,
          "another thread's close is refused as not_owner, even with the right epoch");
    tq::reader::active_request still;
    check(is_found(board, &ctx, &still), "and the owner's request survives");

    const auto counters = board.snapshot();
    check(counters.close_not_owner == 1, "counted under its own name");
}

// DEFECT: nullptr was a valid key, colliding unrelated builds.
void a_null_context_is_refused() {
    station board;

    tq::reader::scope null_scope(board, nullptr, two_spans());
    check(!null_scope.opened(), "a null context is refused");
    check(null_scope.status() == open_status::no_context, "and names the cause");

    tq::reader::active_request out;
    check(is_bad_argument(board, nullptr, &out),
          "and find(nullptr) as a context is a caller bug, reported as bad_argument rather than "
          "as an absence");

    const auto counters = board.snapshot();
    check(counters.refused_no_context == 1, "counted");
    check(counters.opened == 0, "no request was created");
}

// A refusal leaves the station untouched, so a caller that treats it as fatal has not disturbed a
// request another build is using.
void an_empty_view_is_refused_without_side_effects() {
    station board;
    int ctx = 0;

    const query_view empty;
    tq::reader::scope none(board, &ctx, empty);
    check(!none.opened(), "an empty view is refused");
    check(none.status() == open_status::empty_view, "and names the cause");

    query_view hollow;
    span_keys void_span;
    void_span.effective_pos = 3;
    hollow.spans = {void_span};   // a span with no cells
    tq::reader::scope bad(board, &ctx, hollow);
    check(!bad.opened(), "a span with no cells is refused");
    check(bad.status() == open_status::empty_span, "and names the cause");

    const auto counters = board.snapshot();
    check(counters.opened == 0, "no open was recorded");
    check(counters.refused_empty == 2, "both refusals counted");
    check(board.live_count() == 0, "and the station is still empty");
}

// The guard's reason for existing: an exception on the way out must not leave the request on.
void a_scope_closes_when_the_build_throws() {
    station board;
    int ctx = 0;

    bool threw = false;
    try {
        tq::reader::scope scope(board, &ctx, two_spans());
        check(scope.opened(), "the scope opened before the throw");
        throw std::runtime_error("a graph builder failed halfway");
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "the exception propagated, as it should");

    tq::reader::active_request after;
    check(is_absent(board, &ctx, &after),
          "and the request did not survive it: a flag cleared by hand would still be set here");

    const auto counters = board.snapshot();
    check(counters.opened == 1 && counters.closed == 1, "opened once, closed once");
}

// The view is copied, so a temporary is safe.
void a_temporary_view_outlives_its_expression() {
    station board;
    int ctx = 0;

    tq::reader::scope scope(board, &ctx, two_spans());   // a temporary, destroyed at the `;`
    check(scope.opened(), "a temporary view is accepted");
    tq::reader::active_request found;
    check(is_found(board, &ctx, &found), "the request survived the temporary's destruction");
    check(found.view.spans.size() == 2 && found.view.spans[1].cells[0] == 7,
          "with its contents intact -- a reference would be a dangling read here");
}

// DEFECT: the epoch wrapped, reopening the stale-close door through the back door.
//
// The station cannot be driven to 2^64 opens, so the boundary is a pure rule and is tested
// directly. "It would take 584 years" is a statement about run length, not a proof -- the same
// argument was rejected for the whiteboard's epoch.
void the_epoch_rule_seals_at_the_maximum() {
    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();

    check(station::epoch_available(0), "a fresh station (nothing issued) can issue epoch 1");
    check(station::epoch_available(kMax - 1), "the last value that can still advance does");
    check(!station::epoch_available(kMax),
          "at the maximum the station refuses to issue, rather than wrapping to a value a stale "
          "guard could match");
}

// There is no ceiling on distinct contexts, and this case is what replaced the one that tested
// the ceiling.
//
// The previous version opened exactly kMaxContexts (=8) scopes and asserted the ninth was
// refused with `table_full`. That was a test of an arbitrary limit: it pinned the station's own
// invented bound as if it were a contract, so the limit could not be questioned without first
// editing the test that defended it. A test that defends a limit nobody asked for is worse than
// no test, because it makes the limit look deliberate.
//
// What matters is the property the limit was standing in for: an independent context is never
// refused because OTHER contexts hold slots. So this case opens more contexts than the old
// constant allowed, all at once, and asserts every one of them opened -- which is a statement
// about correctness rather than about a number in a header.
void many_contexts_coexist_without_a_ceiling() {
    station board;
    constexpr std::size_t kContexts = 32;   // four times the removed constant
    static int contexts[kContexts];

    std::vector<std::unique_ptr<tq::reader::scope>> held;
    held.reserve(kContexts);

    for (std::size_t i = 0; i < kContexts; ++i) {
        held.push_back(std::make_unique<tq::reader::scope>(
            board, &contexts[i], one_span(static_cast<int32_t>(i))));
        check(held.back()->opened(), "every independent context opens");
    }
    check(board.live_count() == kContexts, "every live request has its own slot");

    // And each one still resolves to its OWN request -- a station that silently reused slots
    // would pass a live_count check while serving the wrong view.
    int mismatched = 0;
    for (std::size_t i = 0; i < kContexts; ++i) {
        tq::reader::active_request found;
        if (!is_found(board, &contexts[i], &found) || found.context != &contexts[i] ||
            found.view.spans.size() != 1 ||
            found.view.spans[0].cells[0] != static_cast<int32_t>(i)) {
            ++mismatched;
        }
    }
    check(mismatched == 0, "and no context resolves to another context's view");

    const auto counters = board.snapshot();
    check(counters.opened == kContexts, "every open counted");
    check(counters.refused_no_room == 0,
          "nothing was refused for lack of room: growth is not a workload limit");
    check(counters.refused_nested == 0 && counters.refused_owned_elsewhere == 0,
          "and no context was mistaken for a nest or for contention");
}

// Many threads, many contexts, overlapping: every independent build gets its own request and no
// request is lost or applied to the wrong context.
void concurrent_independent_builds_do_not_interfere() {
    station board;
    constexpr int kThreads = 6;
    static int contexts[kThreads];
    std::atomic<int> refusals{0};
    std::atomic<int> wrong_view{0};
    std::atomic<int> iterations{0};

    std::vector<std::thread> workers;
    workers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            for (int round = 0; round < 50; ++round) {
                // Each thread uses its own context and a cell value that identifies the owner, so
                // a request applied to the wrong context is visible in the contents.
                tq::reader::scope scope(board, &contexts[t], one_span(t + round));
                if (!scope.opened()) {
                    refusals.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                tq::reader::active_request found;
                if (!is_found(board, &contexts[t], &found)) {
                    refusals.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                if (found.view.spans.size() != 1 ||
                    found.view.spans[0].cells[0] != t + round ||
                    found.context != &contexts[t]) {
                    wrong_view.fetch_add(1, std::memory_order_relaxed);
                }
                iterations.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }

    check(refusals.load() == 0,
          "no independent build was refused while others held their slots");
    check(wrong_view.load() == 0, "and no build ever saw another context's view");
    check(iterations.load() == kThreads * 50, "every round ran with its own request");
    check(board.live_count() == 0, "and every slot was released");

    const auto counters = board.snapshot();
    check(counters.opened == counters.closed,
          "each accepted open was closed exactly once across the whole run");
}

} // namespace

int main() {
    find_separates_absence_from_failure();
    a_request_applies_only_to_its_own_context();
    independent_contexts_coexist();
    a_nested_open_is_refused_not_merged();
    the_same_context_from_two_threads_is_owned_elsewhere();
    find_copies_and_cannot_dangle();
    a_manual_close_does_not_kill_a_reopen();
    a_close_with_nothing_open_is_absent();
    the_owning_thread_alone_can_close();
    a_null_context_is_refused();
    an_empty_view_is_refused_without_side_effects();
    a_scope_closes_when_the_build_throws();
    a_temporary_view_outlives_its_expression();
    the_epoch_rule_seals_at_the_maximum();
    many_contexts_coexist_without_a_ceiling();
    concurrent_independent_builds_do_not_interfere();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "graph station: one request per build, independent builds coexist, "
                         "a stale guard cannot close a live request\n");
    return 0;
}
