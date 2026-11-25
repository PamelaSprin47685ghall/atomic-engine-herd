// The whiteboard's contract, as tests.
//
// Each case names the sentence it enforces, because a passing assertion with no sentence
// behind it is the kind of green that survives a wrong design:
//
//   commit publishes every submission        AXIOM 3 (all)
//   a refused commit changes nothing         AXIOM 1, 3 (none)
//   discard changes nothing                  "fail-stop", not "fail-partially"
//   stale epoch is refused, not merged       AXIOM 1 (peers' LAST committed token)
//   no token is not token zero               AXIOM 1 (last COMMITTED, which may not exist)
//   destroying an open step does not publish  AXIOM 3 (no exception for early return)
//   a snapshot is one point in time             AXIOM 3 on the observation side
//   concurrent steps never publish a mix     AXIOM 3 under real threads
#include <tq/rerot/whiteboard.h>

#include <atomic>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

// Atomic, because the concurrency case below reports failures from four writer threads.
// A plain int there was a data race in the test itself -- and a test that races on its own
// verdict can lose a failure that happened, which is the one bug a test cannot afford.
std::atomic<int> failures{0};

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

using tq::rerot::station;
using tq::rerot::step;
using tq::rerot::step_status;

// A step's submissions land together -- all of them, because that is the unit.
//
// This case used to publish a strict subset and assert that the untouched stage stayed
// untouched. That assertion was the defect: LEMMA 2 says one body step advances EACH bound
// stage by one token, so a step that publishes two of three has published part of an update,
// which is what AXIOM 3 forbids -- and every counter would have called it a clean commit,
// because each of the two writes was legal. The case now pins the rule in both directions:
// a complete set publishes, an incomplete one does not.
void commit_publishes_every_bound_stage() {
    {
        station board(3);
        step partial = board.open();
        check(partial.submit(0, 41) == step_status::ok, "partial: stage 0");
        check(partial.submit(2, 43) == step_status::ok, "partial: stage 2");
        check(partial.commit() == step_status::missing_stage,
              "a step missing a bound stage is refused, not published in part");
        check(board.epoch() == 0, "the refused commit moved nothing");
        check(!board.peek(0).valid && !board.peek(2).valid,
              "neither submitted token landed: the unit is the whole set");
        check(board.refusals() == 1, "the refusal is counted");
        partial.discard();
    }
    {
        station board(3);
        step complete = board.open();
        check(complete.submit(0, 41) == step_status::ok, "complete: stage 0");
        check(complete.submit(1, 42) == step_status::ok, "complete: stage 1");
        check(complete.submit(2, 43) == step_status::ok, "complete: stage 2");
        check(complete.commit() == step_status::ok, "commit: every bound stage is present");
        check(board.peek(0).token == 41, "stage 0 holds its token");
        check(board.peek(1).token == 42, "stage 1 holds its token");
        check(board.peek(2).token == 43, "stage 2 holds its token");
        check(board.epoch() == 1, "epoch advanced exactly once");
        check(board.commits() == 1, "one commit counted");
    }
}

// A refusal is not a partial publish.
void a_refused_commit_writes_nothing() {
    station board(2);
    {
        step first = board.open();
        check(first.submit(0, 10) == step_status::ok, "setup: stage 0");
        check(first.submit(1, 10) == step_status::ok, "setup: stage 1");
        check(first.commit() == step_status::ok, "setup: first commit");
    }
    step moving = board.open();            // opens at epoch 1
    step rival = board.open();             // opens at epoch 1 too
    check(rival.submit(0, 99) == step_status::ok, "rival submits stage 0");
    check(rival.submit(1, 99) == step_status::ok, "rival submits stage 1");
    check(rival.commit() == step_status::ok, "rival commits first");

    check(moving.submit(0, 77) == step_status::ok, "late submit stage 0");
    check(moving.submit(1, 78) == step_status::ok, "late submit stage 1");
    const std::uint64_t before = board.epoch();
    check(moving.commit() == step_status::stale_epoch, "the late step is refused as stale");

    check(board.epoch() == before, "a refused commit does not move the epoch");
    check(board.peek(0).token == 99, "stage 0 still holds the rival's token");
    check(board.peek(1).token == 99, "stage 1 still holds the rival's token, not a mix");
    check(board.commits() == 2, "the refusal is not counted as a commit");
    check(board.refusals() == 1, "the refusal is counted as a refusal");
}

// Discard is invisible to the board.
void discard_changes_nothing() {
    station board(2);
    {
        step s = board.open();
        check(s.submit(0, 5) == step_status::ok, "discard case: submit");
        s.discard();
        check(!s.open(), "discard closes the step");
        check(s.commit() == step_status::already_closed, "a discarded step cannot commit");
    }
    check(board.epoch() == 0, "discard does not move the epoch");
    check(!board.peek(0).valid, "discard publishes nothing");
    check(board.commits() == 0, "discard is not a commit");
    check(board.discards() == 1, "discard is counted");
}

// Destruction is a discard, and a discard publishes nothing.
void destroying_an_open_step_does_not_publish() {
    station board(2);
    {
        step s = board.open();
        check(s.submit(0, 7) == step_status::ok, "dying step: submit");
    }   // early return / throw / forgotten: the same path
    check(!board.peek(0).valid, "an abandoned step published nothing");
    check(board.epoch() == 0, "an abandoned step did not move the epoch");
    check(board.discards() == 1, "the abandon is counted, so the run does not look clean");
}

// Absent is not zero.
void no_token_is_distinct_from_token_zero() {
    // One stage, so the commit carries the whole bound set and the case is about the token
    // value rather than about completeness (a 2-stage board would now need both submitted).
    station board(1);
    check(!board.peek(0).valid, "a fresh board has no token for stage 0");
    {
        step s = board.open();
        check(s.submit(0, 0) == step_status::ok, "submit the real token 0");
        check(s.commit() == step_status::ok, "commit it");
    }
    check(board.peek(0).valid, "after commit the token exists");
    check(board.peek(0).token == 0, "and it really is token 0");

    station untouched(1);
    check(!untouched.peek(0).valid,
          "an uncommitted board still reports no token -- the value and its absence differ");
}

// The read AXIOM 1 is about: peers, excluding self, in stage order.
void peers_exclude_self() {
    station board(3);
    step s = board.open();
    check(s.submit(0, 100) == step_status::ok, "peers: stage 0");
    check(s.submit(1, 101) == step_status::ok, "peers: stage 1");
    check(s.submit(2, 102) == step_status::ok, "peers: stage 2");
    check(s.commit() == step_status::ok, "peers: commit");

    const std::vector<tq::rerot::token_view> peers = s.peers_of(1);
    check(peers.size() == 2, "stage 1 has two peers");
    check(peers[0].valid && peers[0].token == 100, "peer 0 is stage 0's token");
    check(peers[1].valid && peers[1].token == 102, "peer 1 is stage 2's token, not stage 1's");
    check(board.peers_of(9).empty(), "a stage that does not exist has no peers");
}

// Submission hygiene: one token per stage, and a refusal taints the whole step.
//
// This case used to assert the opposite, and the assertion was the defect: "the step still
// commits what it legitimately holds" after a submission was refused. AXIOM 3 forbids
// publishing part of an update, and the stage that did not land is exactly as much of the
// update as the stages that did -- the caller built one token per bound stage (LEMMA 2) and
// is about to publish a set that is missing one. The assertion is inverted rather than
// deleted, so the case still has teeth in both directions, and a third block keeps the
// no-refusal path honest.
void submissions_are_checked() {
    {
        station board(2);
        step s = board.open();
        check(s.submit(0, 1) == step_status::ok, "submit once");
        // The out-of-range refusal comes FIRST, so it is the one that taints. An earlier
        // version submitted the duplicate first and the out-of-range one only as "the next
        // submit into a tainted step", which never exercised the out-of-range branch's own
        // poisoning: removing that branch's `poison()` left this case passing. The order is
        // the coverage.
        check(s.submit(5, 3) == step_status::stage_out_of_range, "an unknown stage is refused");
        check(s.submit(0, 2) == step_status::refused_submission,
              "a tainted step refuses further submissions instead of quietly accepting them");
        check(s.submit(0, 4) == step_status::refused_submission,
              "and keeps refusing: the duplicate it would have made is never compared");
        const std::uint64_t before = board.epoch();
        check(s.commit() == step_status::refused_submission,
              "a tainted step cannot publish the part that landed");
        check(board.epoch() == before, "the refused commit moves nothing");
        check(!board.peek(0).valid, "so stage 0 -- whose own submission was legal -- did not land");
        check(board.refusals() == 4,
              "all four refusals are counted: the out-of-range submit, two submits into the "
              "tainted step, and the commit");
        s.discard();
        check(board.discards() == 1, "the tainted step is discarded, and that is counted too");
    }
    {
        // The other direction, so neither statement above is vacuously true: an untainted
        // step with two legal submissions publishes both.
        station board(2);
        step s = board.open();
        check(s.submit(0, 11) == step_status::ok, "clean: stage 0");
        check(s.submit(1, 12) == step_status::ok, "clean: stage 1");
        check(s.commit() == step_status::ok, "clean: commits");
        check(board.peek(0).token == 11 && board.peek(1).token == 12,
              "clean: both tokens are on the board");
        check(board.refusals() == 0, "clean: nothing was refused");
    }
    {
        station other(2);
        step empty = other.open();
        check(empty.commit() == step_status::nothing_to_publish, "an empty step publishes nothing");
        check(other.epoch() == 0, "and moves nothing");
        check(empty.commit() == step_status::nothing_to_publish, "and stays open to be filled or dropped");
        empty.discard();
    }
}

// The epoch guard's boundary, as a pure function.
//
// The station cannot reach it: sealing takes 2^64 commits. The earlier reasoning -- "wraparound
// needs 584 years, so there is no ABA" -- was not a proof; it was a statement about run length,
// and it left the guard's only failing case untested. Testing the rule directly costs three
// assertions and covers the case the station can never drive.
void the_epoch_rule_seals_instead_of_wrapping() {
    using tq::rerot::check_epoch;
    using tq::rerot::epoch_verdict;
    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();

    check(check_epoch(0, 0) == epoch_verdict::commit, "the epoch a step opened at commits");
    check(check_epoch(1, 0) == epoch_verdict::stale, "a moved epoch is stale");
    check(check_epoch(0, 1) == epoch_verdict::stale, "a board behind the step is stale too");
    // The boundary itself: the board cannot advance past the maximum, so a step that opened
    // *at* the maximum must not commit either -- wrapping to 0 would equal the epoch of every
    // step opened on a fresh board, and the stale test would start comparing unrelated runs.
    check(check_epoch(kMax, kMax) == epoch_verdict::sealed,
          "a board at the maximum refuses even a step that opened there");
    check(check_epoch(kMax, kMax - 1) == epoch_verdict::sealed, "and refuses a step from before it");
    check(check_epoch(kMax - 1, kMax - 1) == epoch_verdict::commit,
          "the last value that can still advance does advance");
}

// The taint survives a move, and a moved step cannot publish.
//
// `station::open()` returns by value, so every ordinary step is a moved one. The taint used to
// be encoded as a sentinel inside the submission vector -- which also meant appending it could
// throw, leaving a refusal counted and the step untainted. It is a bool now, carried explicitly
// by the move constructor and assignment; this case is why that line exists and what fails if
// it is dropped.
void a_taint_survives_a_move() {
    station board(2);
    step tainted = board.open();
    check(tainted.submit(9, 1) == step_status::stage_out_of_range, "taint: an out-of-range submit");

    step moved = std::move(tainted);
    check(moved.submit(0, 1) == step_status::refused_submission,
          "the moved step is still tainted and refuses submissions");
    check(moved.commit() == step_status::refused_submission,
          "the moved step cannot publish what it holds");

    step assigned_moved;
    assigned_moved = std::move(moved);
    check(assigned_moved.commit() == step_status::refused_submission,
          "a taint survives move assignment as well");
    check(board.epoch() == 0, "no tainted step moved the board");
    assigned_moved.discard();
}

// A partial binding is a valid episode, and the board must say which set it is judging.
//
// This case exists because the previous implementation -- and its test -- treated "every stage
// on the board" as "every stage this episode must advance". With four logical stages and only
// two bound, no step could commit at all, and the refusal reported the caller as incomplete.
// The distinction is the contract's, not a convenience: LEMMA 2 says each BOUND stage, and
// LEMMA 3 puts the logical-stage-to-sequence mapping in the caller, so the subset is a
// property of the episode and the board is told it at construction.
//
// Both directions are pinned here, because either alone is satisfiable by a wrong rule: a
// complete bound set commits, a complete-but-for-a-bound-stage set does not.
void a_partial_binding_is_a_valid_episode() {
    station board(4, {0, 2});
    check(board.n_stages() == 4, "partial: four logical stages are declared");
    check(board.n_bound() == 2, "partial: two are bound in this episode");
    check(board.is_bound(2), "partial: a listed stage is bound");
    check(!board.is_bound(1), "partial: an unlisted stage is not");

    {
        step s = board.open();
        check(s.submit(0, 100) == step_status::ok, "partial: bound stage 0 accepted");
        check(s.submit(2, 102) == step_status::ok, "partial: bound stage 2 accepted");
        check(s.commit() == step_status::ok, "partial: exactly the bound set commits");
        check(board.epoch() == 1, "partial: the epoch advanced");
        check(board.peek(2).token == 102, "partial: the committed token is on the board");
        check(!board.peek(1).valid, "partial: an unbound stage holds nothing");
    }
    {
        // An unbound stage cannot publish here, and accepting the token would only move the
        // drop one call later -- which is the partial publish this board refuses.
        step s = board.open();
        check(s.submit(1, 7) == step_status::stage_not_bound,
              "partial: submitting an unbound stage is refused by name");
        check(s.commit() == step_status::refused_submission,
              "partial: the refusal taints the step, so the rest of it cannot publish");
        check(board.epoch() == 1, "partial: nothing moved");
        s.discard();
    }
    {
        // The other direction: a prefix of the BOUND set is still incomplete.
        step s = board.open();
        check(s.submit(0, 5) == step_status::ok, "partial: one bound stage of two");
        check(s.commit() == step_status::missing_stage,
              "partial: a missing bound stage refuses the commit");
        check(board.epoch() == 1, "partial: and writes nothing");
        s.discard();
    }
    {
        // Reading peers from an unbound stage is a question about a stage that is not an
        // insertion point this episode. Empty is the read set; every-bound-stage would read as
        // "my peers have not committed yet".
        step s = board.open();
        check(s.peers_of(1).empty(), "partial: an unbound stage has no peers");
        const std::vector<tq::rerot::token_view> peers = s.peers_of(0);
        check(peers.size() == 1, "partial: a bound stage has one active peer (stage 2)");
        check(peers[0].token == 102, "partial: and it is that peer's token");
        check(board.peers_of(1).empty(), "partial: the station agrees about an unbound stage");
    }
    {
        // The single-argument form still binds everything, and does it with one board rather
        // than building an empty-mask board and replacing it.
        station all(3);
        check(all.n_bound() == 3, "all-bound: station(n) binds every stage");
        step s = all.open();
        check(s.submit(0, 1) == step_status::ok && s.submit(1, 2) == step_status::ok &&
              s.submit(2, 3) == step_status::ok,
              "all-bound: every stage accepted");
        check(s.commit() == step_status::ok, "all-bound: commits");
        check(all.n_bound() == 3 && all.is_bound(2), "all-bound: the mask is the identity");
    }
    {
        // A binding the board cannot honour is a caller bug with no safe reading.
        bool threw = false;
        try {
            station bad(2, {5});
        } catch (const std::out_of_range &) {
            threw = true;
        }
        check(threw, "binding: a stage outside the board is refused");

        threw = false;
        try {
            station twice(2, {1, 1});
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        check(threw, "binding: the same stage bound twice is refused, not deduplicated");
    }
}

// A step outliving its station stays safe, and still means what it said.
void a_step_outlives_its_station() {
    step survivor;
    std::uint64_t opened_at = 0;
    {
        station board(2);
        survivor = board.open();
        opened_at = survivor.epoch();
        check(survivor.submit(0, 3) == step_status::ok, "survivor: submit stage 0");
        check(survivor.submit(1, 4) == step_status::ok, "survivor: submit stage 1");
    }   // the station is gone; the board it named lives on through the step
    check(survivor.epoch() == opened_at, "the step still reports the epoch it opened at");
    check(survivor.peek(0).valid == false, "and can still read the board");
    check(survivor.commit() == step_status::ok, "and can still publish, owning the only handle");
}

// The real thing: steps from many threads, one invariant, checked by a reader that must never
// see a torn commit.
//
// Both stages are always written with the SAME value in one step, so any observation of two
// different valid tokens is a published mix -- exactly what AXIOM 3 forbids. Retrying on
// stale_epoch is the caller's policy here, not the station's: the station's job is to make
// the retry necessary rather than to hide that it happened.
void concurrent_steps_never_publish_a_mix() {
    station board(2);
    constexpr int kThreads = 4;
    constexpr int kCommitsPerThread = 200;

    std::atomic<bool> running{true};
    std::atomic<int> torn{0};
    std::atomic<long> observations{0};
    std::atomic<int> stale_seen{0};

    std::thread reader([&] {
        while (running.load(std::memory_order_relaxed)) {
            // One call, one point in time. The first version used two `peek` calls and
            // reported 1 torn commit out of ~4000 observations -- all of them real as an
            // observation, none of them published by the board: a commit between the two
            // reads yields stage 0 from epoch N and stage 1 from epoch N+1. The publish was
            // atomic; the comparison was not. `snapshot` is the API that closes that gap.
            const auto all = board.snapshot();
            if (all.size() == 2 && all[0].valid && all[1].valid && all[0].token != all[1].token) {
                torn.fetch_add(1, std::memory_order_relaxed);
            }
            observations.fetch_add(1, std::memory_order_relaxed);
        }
    });

    std::vector<std::thread> writers;
    writers.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        writers.emplace_back([&, t] {
            int done = 0;
            std::int32_t value = static_cast<std::int32_t>(t) * 1000;
            while (done < kCommitsPerThread) {
                step s = board.open();
                // AXIOM 1: the token is computed from the peers' last committed values, read
                // through the step so that the commit validates them.
                const auto peers = s.peers_of(0);
                std::int32_t next = value + done;
                for (const auto & peer : peers) {
                    if (peer.valid) next += peer.token % 7;   // a real dependency on the read set
                }
                if (s.submit(0, next) != step_status::ok || s.submit(1, next) != step_status::ok) {
                    std::fprintf(stderr, "FAIL: concurrent submit refused\n");
                    ++failures;
                    return;
                }
                const step_status status = s.commit();
                if (status == step_status::ok) {
                    ++done;
                } else if (status == step_status::stale_epoch) {
                    stale_seen.fetch_add(1, std::memory_order_relaxed);
                } else {
                    std::fprintf(stderr, "FAIL: concurrent commit refused: %s\n",
                                 tq::rerot::describe(status));
                    ++failures;
                    return;
                }
            }
        });
    }
    for (auto & writer : writers) writer.join();
    running.store(false);
    reader.join();

    check(torn.load() == 0, "no reader ever saw stage 0 and stage 1 disagree (no torn commit)");
    check(observations.load() > 0, "the reader actually observed the board");
    check(board.commits() == std::uint64_t(kThreads) * kCommitsPerThread,
          "every intended commit landed exactly once");
    check(board.epoch() == board.commits(), "the epoch counts commits and nothing else");
    check(board.peek(0).token == board.peek(1).token, "the last commit's pair is still consistent");
    check(board.refusals() == std::uint64_t(stale_seen.load()),
          "every refusal was a stale epoch, and every refusal was counted");
    std::fprintf(stderr, "concurrent: %ld observations, %d stale retries, %llu commits\n",
                 observations.load(), stale_seen.load(),
                 static_cast<unsigned long long>(board.commits()));
}

} // namespace

int main() {
    commit_publishes_every_bound_stage();
    a_refused_commit_writes_nothing();
    discard_changes_nothing();
    destroying_an_open_step_does_not_publish();
    no_token_is_distinct_from_token_zero();
    peers_exclude_self();
    submissions_are_checked();
    a_partial_binding_is_a_valid_episode();
    the_epoch_rule_seals_instead_of_wrapping();
    a_taint_survives_a_move();
    a_step_outlives_its_station();
    concurrent_steps_never_publish_a_mix();

    const int failure_count = failures.load();
    if (failure_count) {
        // `failures` is variadic-unsafe: an atomic passed to fprintf would need a copy, and
        // the copy constructor is deleted. Load once, print the int.
        std::fprintf(stderr, "%d failure(s)\n", failure_count);
        return 1;
    }
    std::fprintf(stderr, "whiteboard contract holds\n");
    return 0;
}
