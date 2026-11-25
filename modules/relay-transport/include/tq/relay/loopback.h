// tq/relay/loopback.h — the transport's waiting half, on host memory.
//
// Method (AGENTS.md): a station owns state and decides synchronously; a ticket
// bears every wait. The split here is literal:
//
//   control.h  the station's shared state: words and the handshake rules.
//                    No thread, no sleep, no clock. Pure functions over a plane.
//   loopback_channel  the ticket. It owns the five producer threads, the reducer
//                    and the consumer acks, and it is the only thing that waits.
//
// What this is NOT: a GPU backend. There is no device, no staging pool, no HIP
// queue, and nothing here claims to move bytes to a GPU. It is the protocol the
// GPU transport also speaks, executed on host memory so that its rules can be
// tested. Building the device side needs the five-GPU rig; see
// modules/tp5-compose/CMakeLists.txt for what is missing and why it is not stubbed.
#pragma once

#include <tq/relay/control_block.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace tq::relay {

// One bank's host storage. The plane carries the handshake; this carries the
// bytes, so a test can check that a reader never sees a torn message.
struct payload {
    // Stands in for a rank's gathered contribution. Sized by the caller; a test
    // fills it with a value derived from (epoch, rank) so any mix of two epochs
    // becomes visible as a wrong byte.
    std::vector<std::uint32_t> words;
    // The reduced output, written by the reducer and read by the consumers.
    std::vector<std::uint32_t> output;
};

// Sizes the payloads for `words` per rank and one output holding the five rank
// slices back to back.
//
// The output is exactly `width * ranks`, not `words * ranks`: the reducer copies
// each rank's slice into the output at the same offset, so the two must be laid out
// the same way. An output larger than the slices it holds is not a bigger buffer --
// it is a buffer whose second half a consumer would read as garbage, which is
// precisely the failure this test caught on its first run.
inline payload make_payload(std::size_t words) {
    const std::size_t width = words / control_ranks;
    payload p;
    p.words.resize(width * control_ranks);
    p.output.resize(width * control_ranks);
    return p;
}

struct epoch_report {
    bool     ok           = false;
    std::uint64_t epoch   = 0;
    std::uint32_t ready   = 0;   // ranks that published
    std::uint32_t acked   = 0;   // ranks that finished reading
    std::int32_t error    = 0;
    std::uint64_t observed_before_commit = 0; // what a reader saw early; must be < epoch
    bool     torn         = false;            // a consumer read a message that was not complete
};

// Runs one epoch end to end with real threads: five producers publish, the reducer
// waits for all five and writes the output, five consumers read and ack.
//
// `producer` and `consumer` are the work each thread does. They are callbacks
// rather than fixed code so a test can inject a stall, a failure, or a torn-write
// pattern without a second transport.
//
// `reuse_bank` decides whether this epoch is allowed to observe the plane already
// holding a previous epoch: with two banks in flight, an epoch may start while the
// previous one is still being read, and that is the case the handshake must survive.
class loopback_channel {
  public:
    using producer_fn = std::function<void(std::uint64_t epoch, std::size_t rank,
                                           std::uint32_t * words, std::size_t count)>;
    using consumer_fn = std::function<void(std::uint64_t epoch, std::size_t rank,
                                           const std::uint32_t * output, std::size_t count)>;

    loopback_channel(control & plane, payload & store) : plane_(plane), store_(store) {}

    // One epoch. Blocks until every producer published, the reducer consumed them,
    // and every consumer acked. Never returns a partial success: a failed rank or a
    // reduce error is reported in the result and the plane is left as it was, so a
    // caller cannot mistake a broken epoch for a slow one.
    epoch_report run_epoch(std::uint64_t epoch, const producer_fn & producer,
                           const consumer_fn & consumer);

    // The data invariants a consumer checked, for tests that want the detail rather
    // than a single bool: `torn` is set when any consumer read an incomplete
    // message, and `observed_before_commit` is the newest generation a consumer saw
    // before the plane said it was committed.
    const epoch_report & last() const { return last_; }

  private:
    control & plane_;
    payload &       store_;
    epoch_report    last_{};
};

} // namespace tq::relay
