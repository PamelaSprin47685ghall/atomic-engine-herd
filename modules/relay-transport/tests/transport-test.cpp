// The transport's rules, exercised with real threads on host memory.
//
// Two kinds of assertion here, and the difference matters:
//
//   protocol   what the plane promises: a message is either complete or not
//              readable, an epoch cannot be published twice, an ack cannot precede a
//              commit. Each is checked against a deliberately hostile caller.
//   data       no reader ever sees a byte from a different epoch. A handshake can be
//              perfectly ordered and still carry torn bytes if the release/acquire
//              pairing is wrong, and only a payload check can catch that.
//
// The hostile cases are the point. A transport tested only on its happy path proves
// that threads can be created; what protects a five-rank pipeline is that the plane
// refuses misuse loudly.
#include <tq/relay/control_block.h>
#include <tq/relay/loopback.h>

#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

using namespace tq::relay;

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Every rank writes a value that is a function of (epoch, rank, index). Two epochs
// can therefore never agree, and any mixture of two epochs in the output shows up as
// a wrong word -- which is what a torn message is.
std::uint32_t pattern(std::uint64_t epoch, std::size_t rank, std::size_t index) {
    return std::uint32_t((epoch * 1'000'003u) + (rank * 7919u) + index);
}

// ---- protocol, one thread, hostile callers ---------------------------------

void publish_rules() {
    control plane{};
    initialise_plane(plane);

    check(publish_input(plane, 0, 0) == false, "epoch 0 is never published");
    check(publish_input(plane, control_ranks, 1) == false, "a rank outside the topology is refused");
    check(publish_input(plane, 2, 5) == true, "a first publish is accepted");
    check(publish_input(plane, 2, 5) == false, "a double publish is refused");
    check(publish_input(plane, 2, 4) == false, "an epoch that goes backwards is refused");
    check(publish_input(plane, 2, 6) == true, "a later epoch is accepted");

    // Partial readiness must be visible as partial: the reducer's decision depends on
    // the mask, never on "no error yet".
    initialise_plane(plane);
    check(publish_input(plane, 1, 9) == true, "rank 1 publishes");
    check(publish_input(plane, 3, 9) == true, "rank 3 publishes");
    const readiness partial = observe(plane, 9);
    check(partial.mask == ((1u << 1) | (1u << 3)), "the mask names exactly the publishers");
    check(!partial.complete, "two of five is not complete");

    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        if (rank != 1 && rank != 3) {
            check(publish_input(plane, rank, 9) == true, "the remaining ranks publish");
        }
    }
    const readiness full = observe(plane, 9);
    check(full.complete, "five of five is complete");
    check(!full.failed, "no failure was reported");
}

void commit_and_ack_rules() {
    control plane{};
    initialise_plane(plane);

    // Consumers cannot ack what was never committed, and the plane says so rather
    // than accepting an ack that would let the station reuse a live bank.
    //
    // On its own plane, because a refused ack is not a no-op: the device marks the
    // rank with a generation mismatch when its ack finds the wrong generation
    // (relay.hip:296-298), and a poisoned rank then fails every commit of this
    // channel. Checking it here on the shared plane would poison the fixture.
    {
        control fresh{};
        initialise_plane(fresh);
        check(ack_output(fresh, 0, 1) == false, "an ack before any commit is refused");
        check(!fully_acked(fresh, 1), "nothing is fully acked before a commit");
        check(fresh.error[0].load(std::memory_order_relaxed) == std::uint64_t(error_generation),
              "and the refusal is recorded as a generation mismatch, as the device does");
    }

    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        check(publish_input(plane, rank, 1), "rank publishes epoch 1");
    }
    check(commit_output(plane, 1) == true, "the commit is accepted");
    check(commit_output(plane, 1) == false, "a second commit of the same epoch is refused");
    check(commit_output(plane, 0) == false, "epoch 0 is never committed");
    check(readable_epoch(plane) == 1, "the readable epoch is the committed one");
    check(plane.generation.load(std::memory_order_relaxed) == 1, "the committed epoch is stored, not counted");

    check(ack_output(plane, 0, 1) == true, "a consumer acks a committed epoch");
    check(ack_output(plane, 0, 1) == false, "a duplicate ack is refused");
    check(!fully_acked(plane, 1), "one ack of five is not complete");

    // An ack for a generation that is not the committed one poisons that rank, and
    // deliberately so -- the device records error 2 at this point (relay.hip:296-298)
    // rather than dropping the acknowledgement. It is checked on its own plane because
    // of that consequence: once rank 1 carries an error, it can no longer acknowledge
    // anything, which is the next test's subject.
    {
        control stray{};
        initialise_plane(stray);
        for (std::size_t rank = 0; rank < control_ranks; ++rank) {
            check(publish_input(stray, rank, 1), "rank publishes on the stray plane");
        }
        check(commit_output(stray, 1), "the stray plane commits epoch 1");
        check(ack_output(stray, 1, 2) == false, "an ack for an uncommitted epoch is refused");
        check(stray.error[1].load(std::memory_order_relaxed) == std::uint64_t(error_generation),
              "and the rank is marked with the device's generation-mismatch code");
        check(!ack_output(stray, 1, 1), "the poisoned rank cannot then acknowledge epoch 1");
        check(!fully_acked(stray, 1), "so the epoch is never fully acknowledged");
    }

    for (std::size_t rank = 1; rank < control_ranks; ++rank) {
        check(ack_output(plane, rank, 1), "every consumer acks");
    }
    check(fully_acked(plane, 1), "five acks of five is complete");

    // A commit must advance: reusing an epoch number after a reset is the caller's
    // business, but within one plane's life the word only increases.
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        check(publish_input(plane, rank, 2), "rank publishes epoch 2");
    }
    check(commit_output(plane, 2) == true, "the next commit is accepted");
    check(plane.generation.load(std::memory_order_relaxed) == 2, "the next epoch replaces it");
}

void failure_is_visible_to_the_reducer() {
    control plane{};
    initialise_plane(plane);
    check(publish_input(plane, 0, 3), "rank 0 publishes");
    publish_error(plane, 4, -42);
    const readiness seen = observe(plane, 3);
    check(seen.failed, "a producer failure is visible while the epoch is incomplete");
    check(seen.error == -42, "the failure code is carried through");
    check(!seen.complete, "a failed epoch is not complete");

    // The commit gate. The device reaches `store(&generation, epoch)` only for a
    // channel whose ready_seen covers all ranks (relay.hip:503, :2694-2696), so a
    // commit with a rank outstanding is unreachable there. Here it must be refused
    // explicitly, because a partial commit publishes a full, valid-looking message
    // assembled from four ranks and stale bytes.
    initialise_plane(plane);
    for (std::size_t rank = 0; rank < control_ranks - 1; ++rank) {
        check(publish_input(plane, rank, 4), "four of five ranks publish");
    }
    check(observe(plane, 4).complete == false, "four of five is not complete");
    check(!commit_output(plane, 4), "a commit with one rank outstanding is refused");
    check(readable_epoch(plane) == 0, "nothing became readable");
    check(publish_input(plane, control_ranks - 1, 4), "the fifth rank publishes");
    check(commit_output(plane, 4), "the commit is accepted once every rank published");

    // A rank error blocks the commit even when every rank published: the device
    // fails the channel instead of publishing a message built around a failure.
    initialise_plane(plane);
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        check(publish_input(plane, rank, 5), "every rank publishes epoch 5");
    }
    publish_error(plane, 2, error_timeout);
    check(!commit_output(plane, 5), "a commit is refused while a rank reports an error");

    // An errored rank must not acknowledge. The device guards its ack store with
    // `!load(&c->error[rank])` and says why: "An abort is not a normal consumption
    // ack". Without the guard a rank that aborted could claim it consumed the
    // message, the reducer would conclude the epoch, clear the error words, and
    // release a channel that rank never read.
    //
    // These two lines do NOT test the error guard, and the earlier comment here claimed
    // they did. Epoch 5 was never committed (the commit above was refused), so
    // ack_output refuses on `readable_epoch(plane) != epoch` alone -- remove the guard and
    // both assertions still pass, because no ack is written either way. What they cover is
    // the generation path, which is worth having but is a different rule.
    check(!ack_output(plane, 2, 5), "an errored rank cannot acknowledge an uncommitted epoch");
    check(plane.ack[2].load(std::memory_order_relaxed) == 0, "and no ack word was written");

    // The guard itself, on the only state that isolates it: an epoch that WAS committed,
    // followed by a failure of one rank while it consumes.
    //
    // The device reaches this state: a reducer commits once every rank has published and
    // none has reported an error, and a rank's wait can still expire afterwards
    // (relay.hip:243 abort, :254 generation, :259 timeout -- each sets error[rank] at the
    // point where the rank cannot complete its part). It must then not acknowledge, since
    // "an abort is not a normal consumption ack".
    //
    // Here generation == epoch and the ack word is still 0, so the generation test and the
    // duplicate test both pass and only the guard can refuse. That is what makes this the
    // assertion that covers it: with the guard removed, the call below returns true and
    // writes ack[3] = 8.
    {
        control committed_then_failed{};
        initialise_plane(committed_then_failed);
        for (std::size_t rank = 0; rank < control_ranks; ++rank) {
            check(publish_input(committed_then_failed, rank, 8),
                  "every rank publishes epoch 8");
        }
        check(commit_output(committed_then_failed, 8),
              "epoch 8 commits while every rank is healthy");
        check(readable_epoch(committed_then_failed) == 8, "and it is readable");

        publish_error(committed_then_failed, 3, error_timeout);
        check(!ack_output(committed_then_failed, 3, 8),
              "a rank that failed after the commit cannot acknowledge it");
        check(committed_then_failed.ack[3].load(std::memory_order_relaxed) == 0,
              "and no ack word was written for it");

        // A guard that refused every acknowledgement would satisfy the two lines above;
        // these are what show it refuses the failed rank and nobody else.
        check(ack_output(committed_then_failed, 0, 8), "a healthy rank still acknowledges");
        check(!fully_acked(committed_then_failed, 8),
              "the epoch is never fully acknowledged while one rank has failed");
        check(!conclude_epoch(committed_then_failed, 8),
              "so it cannot be concluded, and its error words cannot be cleared");
        check(observe(committed_then_failed, 8).failed, "and the failure stays visible");
    }

    // Error words carry codes, not epochs, so they are the one part of the plane
    // that equality cannot make inert. The device clears them only when no native
    // work or borrow remains (relay.hip:1851-1855).
    check(!conclude_epoch(plane, 5), "an epoch nobody acknowledged is never concluded");
    check(observe(plane, 5).failed, "the error survives a refused conclusion");

    // A failed epoch never commits, so it never earns acknowledgements, so
    // conclude_epoch can never be satisfied for it. That is the protocol, not a gap:
    // recovery is a new channel. This is the only path that clears error words, and
    // it is legal here because the test drives the plane by hand -- nothing is in
    // flight.
    initialise_plane(plane);
    check(!observe(plane, 5).failed, "a re-created channel carries no error words");
    check(readable_epoch(plane) == 0, "and no committed epoch");

    // The success path does conclude, and only after every acknowledgement, so the
    // clear cannot be reached early. This is the assertion that would have caught
    // the discarded conclude: with the call in the wrong place the plane never
    // reaches this state at all.
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        check(publish_input(plane, rank, 6), "every rank publishes a clean epoch");
    }
    check(commit_output(plane, 6), "the clean epoch commits");
    check(!conclude_epoch(plane, 6), "concluding before the acknowledgements is refused");
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        check(ack_output(plane, rank, 6), "every rank acknowledges the clean epoch");
    }
    check(conclude_epoch(plane, 6), "a fully acknowledged epoch is concluded");

    // A word ahead of the epoch is a protocol violation, not a slow rank.
    control ahead{};
    initialise_plane(ahead);
    publish_input(ahead, 0, 7);
    const readiness future = observe(ahead, 6);
    check(future.future, "a future ready word is reported");
    check(!future.complete, "and it does not read as complete");

    // The plane has no slot for a CPU-owned error, and must not grow one: the
    // device's control block is generation + ready[5] + ack[5] + error[5], sixteen
    // words exactly (relay.hip:98-105, static_assert at :118). A reducer failure is
    // therefore expressed structurally -- the commit never happens, so no rank's
    // wait can succeed -- and that is what this asserts, using the protocol rather
    // than a private word.
    initialise_plane(plane);
    check(!commit_output(plane, 0), "no commit is possible for epoch zero");
    check(readable_epoch(plane) == 0, "nothing is readable when the reducer never commits");
    check(!ack_output(plane, 0, 1), "a rank cannot acknowledge a message that was never committed");
}

// ---- data, many threads ----------------------------------------------------

// One full epoch with real threads, checking the bytes each consumer sees.
void one_epoch_runs_end_to_end() {
    control plane{};
    initialise_plane(plane);
    payload store = make_payload(16);
    loopback_channel channel(plane, store);

    const std::size_t width = store.words.size() / control_ranks;
    const epoch_report report = channel.run_epoch(
        1,
        [width](std::uint64_t epoch, std::size_t rank, std::uint32_t * words, std::size_t count) {
            for (std::size_t i = 0; i < count && i < width; ++i) {
                words[i] = pattern(epoch, rank, i);
            }
        },
        [](std::uint64_t epoch, std::size_t rank, const std::uint32_t * output, std::size_t count) {
            for (std::size_t i = 0; i < count; ++i) {
                if (output[i] != pattern(epoch, rank, i)) {
                    check(false, "a consumer read a word from another epoch");
                    return;
                }
            }
        });

    check(report.ok, "the epoch completed");
    check(report.ready == (1u << control_ranks) - 1u, "every rank published");
    check(report.acked == (1u << control_ranks) - 1u, "every consumer acked");
    check(report.error == 0, "no error was reported");
    check(readable_epoch(plane) == 1, "the plane committed epoch 1");
}

// The case the two-bank rotation exists for: many epochs back to back, with the
// plane holding a committed message while the next one is prepared. A single torn
// read anywhere in this loop is the defect the release/acquire pairing prevents.
void epochs_do_not_tear() {
    control plane{};
    initialise_plane(plane);
    payload store = make_payload(32);
    loopback_channel channel(plane, store);

    const std::size_t width = store.words.size() / control_ranks;
    std::vector<std::uint32_t> seen_by_consumer(control_ranks, 0);
    std::size_t completed = 0;

    for (std::uint64_t epoch = 1; epoch <= 200; ++epoch) {
        // No reset between epochs, and that is the point. Every test in the plane is
        // an equality against the epoch (ready[r] == e, ack[r] == e), so a value from
        // an older epoch already reads as "not this one". The device never clears
        // between epochs either; a port that needed a clear would have changed the
        // protocol rather than implemented it.
        const epoch_report report = channel.run_epoch(
            epoch,
            [width](std::uint64_t e, std::size_t rank, std::uint32_t * words, std::size_t count) {
                for (std::size_t i = 0; i < count && i < width; ++i) {
                    words[i] = pattern(e, rank, i);
                }
            },
            [&seen_by_consumer, width](std::uint64_t e, std::size_t rank,
                                       const std::uint32_t * output, std::size_t count) {
                for (std::size_t i = 0; i < count; ++i) {
                    if (output[i] != pattern(e, rank, i)) {
                        check(false, "a consumer read a word from another epoch");
                        return;
                    }
                }
                ++seen_by_consumer[rank];
            });
        if (report.ok) {
            ++completed;
        } else {
            std::fprintf(stderr, "epoch %llu failed: ready=%x acked=%x error=%d\n",
                         static_cast<unsigned long long>(epoch), report.ready, report.acked,
                         report.error);
            break;
        }
    }

    check(completed == 200, "two hundred epochs completed with no tear");
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        check(seen_by_consumer[rank] == 200, "each consumer read every epoch exactly once");
    }
}

// A failing producer must end the epoch with an error instead of leaving the
// reducer waiting for a rank that will never publish: a hang is the one outcome a
// transport must never produce, because nothing downstream can distinguish it from
// slow work.
void a_failing_rank_does_not_hang_the_epoch() {
    control plane{};
    initialise_plane(plane);
    payload store = make_payload(8);
    loopback_channel channel(plane, store);

    std::size_t consumed = 0;
    const epoch_report report = channel.run_epoch(
        1,
        [](std::uint64_t, std::size_t rank, std::uint32_t * words, std::size_t count) {
            if (rank == 3) {
                throw std::runtime_error("rank 3 cannot deliver");
            }
            for (std::size_t i = 0; i < count; ++i) { words[i] = 1; }
        },
        [&consumed](std::uint64_t, std::size_t, const std::uint32_t *, std::size_t) { ++consumed; });

    check(!report.ok, "a failed epoch is not reported as ok");
    check(report.error != 0, "the failure carries a code");
    check(consumed == 0, "no consumer read an incomplete message");
}

// Reusing a plane that still holds an unread message must be refused, not written
// over: that is the difference between a protocol and a buffer.
void a_plane_holding_a_message_is_not_reused() {
    control plane{};
    initialise_plane(plane);
    payload store = make_payload(8);
    loopback_channel channel(plane, store);

    const auto writer = [](std::uint64_t, std::size_t, std::uint32_t * words, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) { words[i] = 1; }
    };
    const auto reader = [](std::uint64_t, std::size_t, const std::uint32_t *, std::size_t) {};

    check(channel.run_epoch(1, writer, reader).ok, "the first epoch completes");
    // The plane still reads as committed epoch 1; writing epoch 1 again is refused by
    // the plane's own monotonicity, not by the channel's bookkeeping.
    const epoch_report again = channel.run_epoch(1, writer, reader);
    check(!again.ok, "an epoch that does not advance is refused");
    check(again.error == -2, "the refusal names the reason");
}

} // namespace

int main() {
    publish_rules();
    commit_and_ack_rules();
    failure_is_visible_to_the_reducer();

    one_epoch_runs_end_to_end();
    epochs_do_not_tear();
    a_failing_rank_does_not_hang_the_epoch();
    a_plane_holding_a_message_is_not_reused();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "relay transport contract holds\n");
    return 0;
}
