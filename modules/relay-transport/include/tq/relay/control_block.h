// tq/relay/control_block.h — the TP5 relay control plane, field for field.
//
// "Field for field", not "byte for byte": the layout matches the device's (same fields,
// same order, same 16 x 64-byte size) but the types do not -- see the note further down
// on std::atomic<uint64_t> versus the device's uint64_t. A header line that claimed the
// stronger thing would be the same over-claim the README had to retract.
//
// This is not a model of the GPU plane. It is the same sixteen words, in the same
// order, with the same comparison rules, so that a device backend and this host
// implementation can be described by one layout instead of two that drift.
//
// The authority is the fork's own definition, with its own assertion:
//
//   ggml/src/ggml-cuda/relay.hip:98-105
//     struct alignas(64) word { uint64_t value; };
//     struct control { word generation; word ready[ranks]; word ack[ranks]; word error[ranks]; };
//   ggml/src/ggml-cuda/relay.hip:118
//     static_assert(sizeof(control) == ggml_relay_plan::control_bytes, ...);
//   ggml/include/ggml-relay-plan.h:36
//     constexpr std::size_t control_bytes = 16 * 64;
//   ggml/include/ggml-relay.h:10
//     #define GGML_RELAY_RANKS 5
//
// A previous version of this file was NOT that layout: it added a `committed_epoch`
// word, packed each rank's error next to its ready word, and grew a `cpu_error`
// slot, reaching the same 1024 bytes with a different meaning at almost every
// offset. Two blocks of equal size are not the same block: a GPU backend writing
// `error[3]` would have been writing into this file's `ready[3]` cache line. The
// claims in the README ("the same plane", "the same handshake") were therefore
// false for checkable reasons, not just imprecise ones. The layout below removes
// the discrepancy, and the static assertions are the check that it stays removed.
//
// Three rules the GPU code relies on, reproduced here rather than paraphrased:
//
//  1. `generation` holds the committed EPOCH, not a counter. The host stores
//     p.status.epoch into it (relay.hip:2781); a consumer waits for
//     `generation == epoch`, and a generation GREATER than the expected epoch is an
//     error, not a newer message (relay.hip:248-255).
//  2. `ready[r] == epoch` and `ack[r] == epoch` are equality tests. That is what
//     makes a control block reusable without clearing it: a value from an older
//     epoch simply is not equal. There is no reset in the protocol, and adding one
//     would create a window in which a late acknowledgement could be cleared.
//  3. `error[r]` is a separate word with a small code domain, written by the rank
//     that failed: 1 abort claimed, 2 generation mismatch, 3 wait expired
//     (relay.hip:243, 254, 259). A non-zero error on any rank fails the epoch.
//
// Ordering. The GPU side uses __threadfence_system() before publishing and
// `__ATOMIC_RELEASE` stores / `__ATOMIC_ACQUIRE` loads (relay.hip:132-136, 209,
// 231, 288). Here that is `memory_order_release` on the publish, commit and
// acknowledge stores and `memory_order_acquire` on every read that follows one.
// The one deviation is the device's relaxed poll of a word in a spin loop
// (relay.hip:127), which is safe there because the value is only a hint to keep
// spinning; this host version uses acquire on every observation, which is
// stronger and cannot be wrong.
//
// What this file does not do: no waiting, no threads, no clock, no allocation.
// Every function is a synchronous operation on a plane the caller owns.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tq::relay {

// GGML_RELAY_RANKS. Five because TP5 is five devices; a sixth needs a second plane,
// which is why this is a compile-time constant and not a parameter.
inline constexpr std::size_t control_ranks = 5;
inline constexpr std::size_t cache_line    = 64;

// One control word. The GPU definition is `alignas(64) struct { uint64_t value; }`:
// one machine word of state per cache line, so five ranks' publishers never share a
// line.
//
// std::atomic<uint64_t> is NOT the same object as the device's uint64_t. Same size,
// same alignment, and the same lock-free instruction on x86-64, but a different
// language-level type. A shared mapping can be read by the device as plain
// uint64_t and here as std::atomic<uint64_t> only under two conditions: both sides
// use nothing but atomic operations, and the mapping is initialised by one side
// before the other is allowed to touch it. That is a working arrangement between
// two implementations, not a portable guarantee -- and it is why this file claims to
// share the LAYOUT with the device, never the object. The distinction is not
// pedantry: it is the difference between a statement a compiler can enforce and one
// that only a review can.
struct alignas(cache_line) word {
    std::atomic<std::uint64_t> value{0};

    std::uint64_t load(std::memory_order order) const noexcept { return value.load(order); }
    void          store(std::uint64_t v, std::memory_order order) noexcept { value.store(v, order); }
};

static_assert(sizeof(word) == cache_line, "a control word owns exactly one cache line");
static_assert(sizeof(std::atomic<std::uint64_t>) == 8, "a control word's payload is one 64-bit word");
static_assert(std::atomic<std::uint64_t>::is_always_lock_free, "the plane must not take a lock");

// The plane. Field order and count are the fork's; do not reorder, and do not add.
struct control {
    word generation;             // committed epoch
    word ready[control_ranks];   // per rank: epoch whose payload it finished writing
    word ack[control_ranks];     // per rank: epoch it finished consuming
    word error[control_ranks];   // per rank: 0 ok, 1 abort, 2 generation mismatch, 3 wait expired
};

// 16 words, 1024 bytes: the value ggml_relay_plan::control_bytes asserts against on
// the device side. Checked here too, so a change to this struct fails this build
// rather than a GPU run three months later.
static_assert(sizeof(control) == 16 * cache_line, "the control plane is sixteen cache lines");
static_assert(alignof(control) == cache_line, "the plane must start on a cache line");
static_assert(control_ranks == 5, "GGML_RELAY_RANKS is five; the layout depends on it");

// Failure codes, named. The numbers are the device's, not invented here.
inline constexpr std::int32_t error_none       = 0;
inline constexpr std::int32_t error_aborted    = 1;
inline constexpr std::int32_t error_generation = 2;
inline constexpr std::int32_t error_timeout    = 3;

// Producer side: this rank's payload is written; publish the epoch it wrote for.
//
// Returns false when the epoch does not advance this rank's own word. That is a
// caller bug, not a race: epochs come from the station, and a rank that reuses one
// has lost its place. Refusing is what keeps a stale payload from being read as new.
inline bool publish_input(control & plane, std::size_t rank, std::uint64_t epoch) noexcept {
    if (rank >= control_ranks || epoch == 0) {
        return false;
    }
    const std::uint64_t seen = plane.ready[rank].load(std::memory_order_relaxed);
    if (seen >= epoch) {
        return false;
    }
    // Release: the payload stores the caller made before this call are visible to a
    // reader that acquires this word.
    plane.ready[rank].store(epoch, std::memory_order_release);
    return true;
}

// Producer side: this rank cannot deliver this epoch. The code is the device's
// domain (error_aborted / error_generation / error_timeout), so a reducer reading
// both implementations sees one vocabulary.
inline void publish_error(control & plane, std::size_t rank, std::int32_t code) noexcept {
    if (rank < control_ranks && code != error_none) {
        plane.error[rank].store(std::uint64_t(code), std::memory_order_release);
    }
}

// Reducer side: which ranks published `epoch`, and whether any of them failed.
struct readiness {
    std::uint32_t mask     = 0;      // bit i: rank i published this epoch
    std::int32_t  error    = 0;      // first non-zero rank error
    bool          complete = false;  // every rank published
    bool          failed   = false;  // some rank reported an error
    // A ready or ack word ahead of the epoch means the plane is inconsistent: some
    // rank is working on a later epoch than the reducer is. The device treats that
    // as a protocol violation and fails the channel (relay.hip:916-919, EPROTO
    // "observed a future ready or last-reader epoch"). Reporting it here lets the
    // commit refuse for the same reason instead of reading it as "not ready yet",
    // which would wait forever on a plane that can never catch up.
    bool          future   = false;
};

inline readiness observe(control & plane, std::uint64_t epoch) noexcept {
    readiness out;
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        // Acquire pairs with publish_input's release, and with the device's
        // __threadfence_system-then-release store (relay.hip:209).
        const std::uint64_t ready = plane.ready[rank].load(std::memory_order_acquire);
        if (ready == epoch) {
            out.mask |= 1u << rank;
        } else if (ready > epoch) {
            out.future = true;
        }
        const std::uint64_t ack = plane.ack[rank].load(std::memory_order_acquire);
        if (ack > epoch) {
            out.future = true;
        }
        const auto code = std::int32_t(plane.error[rank].load(std::memory_order_acquire));
        if (code != error_none && out.error == 0) {
            out.error = code;
        }
    }
    out.complete = out.mask == ((1u << control_ranks) - 1u);
    out.failed   = out.error != error_none;
    return out;
}

// Reducer side: the inputs of `epoch` are consumed and the output is written.
//
// Three conditions, and the first one is the one this file was missing:
//
//  1. every rank published this epoch. The device cannot commit otherwise: a
//     channel becomes an offer candidate only when `ready_seen == all_ranks`
//     (relay.hip:503) and the publish loop only examines candidates
//     (relay.hip:2694-2696), so `store(&generation, epoch)` at relay.hip:2781 is
//     unreachable with a rank outstanding. Without this check a reducer bug
//     publishes a partial epoch and every consumer reads a full, valid-looking
//     message built from four ranks and stale bytes -- the failure this whole
//     plane exists to make impossible.
//  2. no rank, and no reduction, reported an error.
//  3. no word is ahead of the epoch (see readiness::future).
//
// Additionally the commit must advance. Going backwards would un-commit a message
// a consumer may already have read; repeating the current epoch would let a waiter
// believe a second message arrived.
//
// What this cannot do, stated because the plane genuinely lacks it: the device's
// abort/success linearisation is a CAS on a *claim word outside this struct*
// (claim_host_success, relay.hip:2771-2779). The sixteen-word plane has no room for
// it, so a host reducer has no equivalent protection against an abort that claims
// the channel between this observation and this store. A caller that needs that
// guarantee must serialize with the producer side; the plane alone cannot provide
// it, and pretending otherwise would be the same mistake as the layout that used to
// be "the same" by size.
inline bool commit_output(control & plane, std::uint64_t epoch) noexcept {
    if (epoch == 0) {
        return false;
    }
    const readiness seen = observe(plane, epoch);
    if (!seen.complete || seen.failed || seen.future) {
        return false;
    }
    const std::uint64_t committed = plane.generation.load(std::memory_order_relaxed);
    if (epoch <= committed) {
        return false;
    }
    // Release: every payload store the reducer made before this call, and every
    // producer's payload store ordered by its own release, are visible to a
    // consumer that acquires this word.
    plane.generation.store(epoch, std::memory_order_release);
    return true;
}

// Clears the per-rank error words for a new epoch.
//
// The error array is the one part of the plane that cannot be epoch-scoped by
// equality: its values are codes, not epochs, so `error[2] == 3` is
// indistinguishable between epoch 7 and epoch 70. The device clears these words at
// exactly one point, when nothing is in flight, and says why in its own comment:
//
//   relay.hip:1851-1855
//     // No native work or CPU borrow remains. GPU error words are only reset
//     // here, never while their single-writer GPU could still access them.
//     for (auto & error : c.host_control->error) { store(&error, 0); }
//
// This mirrors that, with the same precondition made explicit: call only when the
// previous epoch is fully acknowledged, so no producer is still writing an error
// word. `conclude_epoch` below is the checked form -- and now the only public one.
//
// An earlier version of this file asserted that the protocol needs no reset at
// all. That was true of ready and ack (equality makes old values inert) and false
// of error, where a single timeout would have poisoned every later epoch of the
// channel.
namespace detail {

// Deliberately unreachable from outside this header. A public function whose
// correctness rests on a documented precondition gets called without the check;
// that is what happened -- the test suite called the public clear directly to
// unblock a channel whose rank had reported an error, erasing a failure code before
// anything had read it. Removing the name is the fix; the comment was not.
inline void clear_error_words(control & plane) noexcept {
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        plane.error[rank].store(error_none, std::memory_order_release);
    }
}

} // namespace detail

// Consumer side: the epoch currently readable, 0 when none is.
inline std::uint64_t readable_epoch(const control & plane) noexcept {
    return plane.generation.load(std::memory_order_acquire);
}

// Consumer side: this rank finished reading `epoch`.
//
// Refused unless `generation == epoch` right now, which is the device's own test
// before it acknowledges (relay.hip:291-293). Equality rather than "at least":
// acknowledging an older epoch after a newer commit would claim to have read the
// message that replaced it.
inline bool ack_output(control & plane, std::size_t rank, std::uint64_t epoch) noexcept {
    if (rank >= control_ranks || epoch == 0) {
        return false;
    }

    // An errored rank does not acknowledge, and this is the device's rule rather than
    // a caution: relay.hip:295 guards its ack store with
    // `!load(abort_word) && !load(&c->error[rank])` and comments "An abort is not a
    // normal consumption ack". The earlier version here omitted the guard, so a rank
    // that had aborted could still claim it consumed the message -- and the reducer
    // would then `conclude` the epoch, clear the error words, and release a channel
    // whose payload that rank never read. The failure would surface later, on a
    // different channel, as wrong numbers.
    if (plane.error[rank].load(std::memory_order_acquire) != error_none) {
        return false;
    }

    if (readable_epoch(plane) != epoch) {
        // The device records a generation mismatch when it happens at ack time
        // (relay.hip:296-298) rather than dropping the acknowledgement silently: a
        // rank that could not acknowledge leaves the epoch permanently unconcluded, and
        // a channel that fails without saying why is the defect this plane exists to
        // prevent. The code is the device's own (error_generation = 2).
        plane.error[rank].store(error_generation, std::memory_order_release);
        return false;
    }
    if (plane.ack[rank].load(std::memory_order_relaxed) >= epoch) {
        // Stricter than the device, which simply re-stores the same value (an
        // idempotent write). Refusing makes a caller that lost track of its state hear
        // about it; it cannot turn a correct run into a failed one, because storing the
        // same epoch twice changes nothing either way.
        return false;
    }
    plane.ack[rank].store(epoch, std::memory_order_release);
    return true;
}

// Reducer side: has every rank finished reading `epoch`?
inline bool fully_acked(const control & plane, std::uint64_t epoch) noexcept {
    if (epoch == 0) {
        return false;
    }
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        if (plane.ack[rank].load(std::memory_order_acquire) != epoch) {
            return false;
        }
    }
    return true;
}

// Ends an epoch: clears the error words, but only once every rank has acknowledged
// it. Before that point a producer may still be writing an error word for this
// epoch, and erasing it would hide a failure that has not been reported yet -- the
// same hazard the device names when it says its error words are reset only when no
// native work or borrow remains (relay.hip:1851-1855).
//
// Returns false when the epoch is still in flight, so a caller cannot clear by
// accident; the check is what makes the clear safe to reach at all.
//
// A failed epoch can never be concluded, and that is a property of the protocol
// rather than a limitation of this function: an epoch with an error word set never
// commits, and an epoch that never commits produces no acknowledgements, so
// fully_acked() is false forever. Recovery is therefore a new channel
// (initialise_plane), performed while nothing is in flight -- which mirrors the
// device, where a failed channel is torn down rather than repaired. This function
// exists for the success path; a caller that reaches for it to "unblock" a failed
// epoch has misread the protocol, and gets false rather than a silent repair.
inline bool conclude_epoch(control & plane, std::uint64_t epoch) noexcept {
    if (!fully_acked(plane, epoch)) {
        return false;
    }
    detail::clear_error_words(plane);
    return true;
}

// Clears the plane for a channel that is being brought up.
//
// For the first state of a fresh channel only. Ready and ack need no clearing
// between epochs -- every test on them is an equality against the epoch, so an old
// value is inert -- but error words do, because they carry codes and not epochs;
// detail::clear_error_words()/conclude_epoch() are that step, and the device performs it at the
// same point with the same precondition (relay.hip:1851-1855). Calling this while
// anything is in flight would erase acknowledgements and error reports that are
// still being written, which is why it is named as initialisation and not as reset.
inline void initialise_plane(control & plane) noexcept {
    plane.generation.store(0, std::memory_order_relaxed);
    for (std::size_t rank = 0; rank < control_ranks; ++rank) {
        plane.ready[rank].store(0, std::memory_order_relaxed);
        plane.ack[rank].store(0, std::memory_order_relaxed);
        plane.error[rank].store(0, std::memory_order_relaxed);
    }
}

} // namespace tq::relay
