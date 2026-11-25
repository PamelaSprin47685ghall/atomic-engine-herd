// herd/prefab_station.hpp — the prefab lane's state, and its rulings.
//
// Method (AGENTS.md): the station holds state and decides synchronously; it never
// waits. Here that is literal — this class has no thread, no lock (a context is
// decoded by one thread at a time, and every method is entered from inside that
// thread), no allocation in the ruling path, and no call into ggml or llama. It is
// therefore testable with fabricated numbers, which is the point.
//
// What the station owns:
//
//   * the envelope ladder — the set of fixed widths the lane may pad to;
//   * the eligibility ruling — whether a given ubatch may be padded at all, and
//     if not, which precondition failed (a named reason, never a silent "no");
//   * the counters — the only evidence anyone has that the lane did anything.
//
// What it deliberately does not own: the padded ubatch itself. That is a value the
// seam builds from a plan and throws away, because it lives only as long as one
// call into upstream.
#pragma once

#include <cstdint>

namespace herd::prefab {

// Pad targets. Upstream reuses a compute graph when the ubatch topology matches
// the previous one, and reallocates when the size changes; making the size stable
// is the entire mechanism. Ascending, last is the maximum.
struct config {
    bool     enabled = false;
    uint32_t envelopes[4] = {0, 0, 0, 0};
    int      n_envelopes = 0;
};

// Minimal description of the ubatch a caller is asking about. Copied out of the
// real one by the seam, so this header needs no llama type.
struct shape {
    uint32_t n_tokens = 0;
    uint32_t n_seqs = 0;
    uint32_t n_seq_tokens = 0;
    uint32_t n_pos = 0;
    bool     equal_seqs = false;
    bool     has_type = false;   // mixed token/embedding ubatch
    bool     has_embd = false;   // embedding input
};

enum class verdict {
    accept,
    disabled,        // the lane is off
    max_exceeded,    // larger than the widest envelope
    not_tokenwise,   // n_seq_tokens != 1: rows are not independently addressable
    unaligned_seqs,  // n_seqs != n_tokens: padding would change the seq/token ratio
    multi_pos,       // n_pos > 1: M-RoPE positions per row, padding would need them
    mixed_input,     // embeddings or a mixed ubatch: pad rows have no embedding
    already_wide,    // n_tokens already equals the chosen envelope
};

struct ruling {
    verdict  why = verdict::disabled;
    uint32_t n_env = 0;   // set iff why == accept
};

// Named reason, for a log line that tells a reader what to change.
const char * name(verdict why);

// Whether the cache can pay for a padded ubatch, and what it costs.
//
// This is the second half of the lane's decision, and the half the first attempt at
// padding ignored. Upstream does not allocate KV cells per sequence: `find_slot`
// (llama-kv-cache.cpp:935) accumulates cell indices until it has `n_tokens` of them
// and returns an empty slot_info otherwise, at which point `prepare` (:785) rejects
// the ubatch and the whole decode fails. So an envelope of width N costs N free
// cells for the duration of that ubatch, not one.
//
// Where those cells come from is the fork's `scratch_cell`, and it is why the fork
// kept a counter named `n_fallback_no_scratch`: without a free run of cells, a
// padded ubatch cannot be placed at all, and the run must be returned afterwards or
// the cache fills up with rows nobody reads.
//
// Modelled here as arithmetic, because that is what it is: how many cells the
// envelope needs, how many are free right now, and whether the scratch rows will be
// returned. A lane that cannot answer this before it pads is a lane that aborts
// inside upstream instead of falling back.
struct capacity {
    uint32_t n_tokens_real  = 0;  // rows the caller actually submitted
    uint32_t n_envelope     = 0;  // width being considered
    uint32_t n_cells_free   = 0;  // free cells in the target stream right now
    bool     scratch_owned  = false; // pad rows land on a sequence this lane can clear
    bool     scratch_clearable = false; // and clearing it is a call upstream exposes
};

enum class placement {
    fits,              // the envelope's cells are available and will be returned
    no_free_cells,     // fewer free cells than the envelope needs
    scratch_unowned,   // pad rows would occupy cells no sequence owns
    scratch_unclearable,// the cells would never be returned
    not_padded,        // nothing to do: the ubatch already has envelope width
};

// The ruling. Pure.
struct placement_plan {
    placement why              = placement::not_padded;
    uint32_t  cells_required   = 0;
    uint32_t  cells_reusable   = 0;  // returned after the ubatch, i.e. the real cost
    uint32_t  cells_permanent  = 0;  // unrecoverable; non-zero means refuse
};

placement_plan place(const capacity & need);

const char * name(placement why);

struct counters {
    std::uint64_t accepted = 0;
    std::uint64_t pad_rows = 0;
    std::uint64_t declined[8] = {};
};

class station {
public:
    void configure(const config & cfg);

    // The ruling. Pure: reads the ladder, writes nothing. A caller may ask twice
    // and get the same answer, which is what lets the seam check before it copies
    // any array.
    ruling consider(const shape & s) const;

    // Commit side of the accounting. Called only after the seam actually ran a
    // padded ubatch, so the counters cannot claim work that did not happen.
    void note_ran(uint32_t n_env, uint32_t n_tokens);
    void note_declined(verdict why);

    counters snapshot() const { return counters_; }
    const config & settings() const { return config_; }

private:
    config   config_{};
    counters counters_{};
};

// ---------------------------------------------------------------------------
// Scratch cell pool: the state half of padding.
// ---------------------------------------------------------------------------
//
// `place()` above answers "may this ubatch be padded". That is arithmetic and it
// has no memory. What a padded ubatch actually needs is a borrow: N cells taken
// from the cache, held for the duration of one ubatch, and given back afterwards
// so the next ubatch can use them. The fork called these scratch cells and kept a
// counter for the case where none were available.
//
// This class is the ACCOUNTING half of that borrow, and nothing more. It holds no KV
// cells: it never calls llama_memory_seq_rm, never touches a cache, and cannot make a
// cell exist. What it does is refuse to record more concurrent borrowing than the
// caller declared as available, so that a padded ubatch finds out it has no cells from
// this object rather than from a failed decode somewhere else.
//
// The distinction matters and an earlier comment in this file elided it by calling
// this "that borrow, as real state". It is real state -- of a ledger -- and calling a
// ledger a pool is how a project ends up reporting a reservation it never made. The
// cell supplier is a separate, unimplemented piece: it must own a sequence, write the
// pad rows into it, and call llama_memory_seq_rm to return them. Nothing in this
// repository does that today.
//
// Why it exists even so: the failure a future implementation must avoid is silent
// over-borrowing -- padding every ubatch until the cache is full, with the failure
// landing in an unrelated decode. A ledger that refuses to record more than it was
// told is available turns that into a named refusal at the point of borrowing.
//
// Deliberately not thread-safe, like the rest of the station: a scheduler belongs to
// the context that owns it, and the caller serialises.
class scratch_pool {
public:
    using id = std::uint32_t;
    static constexpr id no_reservation = 0;

    // A borrow, as a value a caller can hold.
    //
    // Two fields, and the second one is the whole point. `slot_id` is the compact tag
    // that makes the table readable and the wrap test meaningful; `epoch` is monotonic
    // across the pool's lifetime and is what release() actually matches on.
    //
    // Why a bare id was not enough: an id is unique only among LIVE reservations, so
    // once the 32-bit counter wraps, an id that was released long ago is handed out
    // again for a different borrow. A stale holder -- a caller that kept an old id, or
    // that released it and still has it in a log, a retry path, or a stale plan -- would
    // then release somebody else's live reservation and credit cells it never held. That
    // is the classic ABA, and it is silent: the ledger looks consistent afterwards and
    // the only symptom is cells that go missing later, in an unrelated ubatch.
    //
    // With the epoch, a stale handle cannot be expressed: releasing it matches no slot,
    // so it is refused and the live reservation is untouched. The type is the guard --
    // there is no bare-id overload to call by accident.
    // Third field, and it is not decoration. `slot_id` and `epoch` are both per-pool
    // counters, so two freshly constructed pools agree on both: the first borrow from
    // each is {1, 1}. The documented deployment is two windows and two stations that
    // share nothing, which means a handle carried across from the other pool would
    // release a live reservation here and credit cells this pool never lent. `owner`
    // is a process-unique token minted at construction, so a foreign handle matches no
    // slot and is refused.
    struct reservation {
        id            slot_id = no_reservation;
        std::uint64_t epoch   = 0;
        std::uint64_t owner   = 0;

        bool valid() const { return slot_id != no_reservation; }
    };

    // Identity, for tests and diagnostics only -- never for release authority.
    //
    // A handle has THREE parts, and equality must cover all three. The earlier version
    // compared only slot_id and epoch, which was correct when those were the only fields;
    // after `owner` was added it became the one comparison that answers "same borrow?"
    // with the wrong answer for handles from two different pools -- and those are exactly
    // the two handles the owner field exists to tell apart. Both pools start their
    // counters at 1, so `{1,1,poolA} == {1,1,poolB}` returned true.
    //
    // Release authority does not depend on this operator (release() compares owner and
    // epoch itself, so a foreign handle cannot free anything). What breaks without the
    // fix is reasoning: a caller or a test that asks "is this the handle I hold?" gets
    // \`true\` for a stranger's handle, and the next step after that answer is to trust it.
    //
    // The bare-id comparisons below still ignore the epoch and the owner on purpose: they
    // answer "which slot does this name", which is what a wrap test asks. They can never
    // gate a release, and the type system enforces that by giving release() no id overload.
    friend bool operator==(const reservation & a, const reservation & b) {
        return a.slot_id == b.slot_id && a.epoch == b.epoch && a.owner == b.owner;
    }
    friend bool operator!=(const reservation & a, const reservation & b) { return !(a == b); }

    // Slot identity against a bare id: `h == 5` and `h != no_reservation` read as
    // questions about the tag. Provided because a test that checks "the counter honoured
    // 0xFFFFFFFE" is asking about the tag, not about validity -- but note what they do
    // NOT do: they ignore the epoch, so they must never gate a release.
    friend bool operator==(const reservation & a, id b) { return a.slot_id == b; }
    friend bool operator!=(const reservation & a, id b) { return a.slot_id != b; }
    friend bool operator==(id a, const reservation & b) { return b == a; }
    friend bool operator!=(id a, const reservation & b) { return b != a; }

    explicit scratch_pool(std::uint32_t capacity);

    // A pool is a station: an identity plus a ledger, and both must live in exactly one
    // place. The default special members were a live hole rather than a style question.
    //
    // Copying duplicates `owner_`, so the original and the copy answer to the same token
    // -- and both start their slots and epochs at 1. A handle taken from the copy then
    // releases the original's live reservation: same owner, same slot_id, same epoch, and
    // the original really did lend those cells, so the release looks legitimate while it
    // credits cells out of a different ledger. That is precisely the cross-station
    // confusion `owner` was introduced to prevent, reached by a route that never crosses
    // a station boundary.
    //
    // Moving would be worse in a quieter way: the moved-from object keeps its owner and
    // its counters, so handles issued before the move stay "valid" against an object that
    // no longer holds their slots -- a release that silently does nothing, or (with a
    // reuse of the same address) does something to a different pool.
    //
    // Deleting both makes the illegal state unrepresentable, which is the rule this
    // project follows elsewhere: a guard at the use site has to name the right object, and
    // the one that names the wrong object reads as protection.
    scratch_pool(const scratch_pool &) = delete;
    scratch_pool & operator=(const scratch_pool &) = delete;
    scratch_pool(scratch_pool &&) = delete;
    scratch_pool & operator=(scratch_pool &&) = delete;

    // Records a borrow of `cells` cells for one ubatch. Returns an invalid reservation
    // when the request is zero, would exceed what the caller declared available, or
    // cannot be given a free slot -- refusal, never partial, because a caller holding a
    // plan it cannot fill pads into a failed decode.
    //
    // This does not allocate. `capacity` is a number the caller supplies; the cells
    // themselves come from a cache this class never sees.
    reservation reserve(std::uint32_t cells);

    // Gives a reservation back. False for an invalid, unknown, already-released, or
    // stale handle -- a stale one being the ABA case above. A double release would
    // credit cells the ledger never got back, and the next over-borrow would be
    // invisible.
    bool release(const reservation & held);

    // Releases everything. For teardown, and for a test that wants a clean slate.
    // Returns how many reservations were live.
    std::uint32_t release_all();

    std::uint32_t capacity() const { return capacity_; }
    std::uint32_t borrowed() const { return borrowed_; }
    std::uint32_t high_water() const { return high_water_; }
    std::uint32_t live() const { return live_; }

    // Refusals, split by cause. One counter for two causes is how a caller ends up
    // tuning the wrong number: "no cells" means the declared capacity is too small,
    // "table full" means too many reservations are outstanding at once, and they have
    // different fixes. `refusals()` is their sum for a caller that only wants a total.
    std::uint32_t refusals_cells() const { return refusals_cells_; }
    std::uint32_t refusals_table() const { return refusals_table_; }
    // Handles refused because they named a slot whose epoch had moved on: the ABA case.
    // Counted apart from the other two because the fix is different again -- it is not a
    // capacity or a table limit, it is a caller holding something that expired.
    std::uint32_t refusals_stale() const { return refusals_stale_; }
    // Handles refused because they belong to a different pool. A separate cause again:
    // the fix is "do not pass handles between stations", not a number.
    std::uint32_t refusals_foreign() const { return refusals_foreign_; }
    // Reservations refused because the epoch space is spent. This one is unreachable in
    // any real run, and it exists so that "unreachable" is enforced by the code rather
    // than asserted in a comment: once every distinct epoch has been issued, issuing
    // another would repeat one, and a repeated (slot_id, epoch) is precisely the ABA
    // this class exists to prevent.
    std::uint32_t refusals_exhausted() const { return refusals_exhausted_; }
    std::uint32_t refusals() const {
        return refusals_cells_ + refusals_table_ + refusals_stale_ + refusals_foreign_ +
               refusals_exhausted_;
    }

#if defined(HERD_SCRATCH_POOL_INSPECT)
    // Test-only: place the id counter so wraparound is exercised in a test rather than
    // after four billion reservations. Compiled into both the test and the
    // implementation by defining HERD_SCRATCH_POOL_INSPECT on that one target, so the
    // class layout stays identical across translation units.
    void force_next_id_for_test(id next) { next_id_ = next; }
    id next_id_for_test() const { return next_id_; }
    // The epoch is what makes a recycled id safe, so a test must be able to force the id
    // back to a value already used while the epoch keeps going: that is precisely the
    // state in which a stale handle could be mistaken for a live one.
    std::uint64_t next_epoch_for_test() const { return next_epoch_; }
    // The epoch a live slot was issued under, or 0 when the slot is idle. Lets a test
    // assert the pairing rather than a symptom.
    std::uint64_t entry_epoch_for_test(std::size_t index) const {
        return index < max_entries && entries_[index].slot_id != no_reservation
                   ? entries_[index].epoch
                   : 0;
    }
    // The token this pool was minted with. Identity is what makes a foreign handle
    // detectable, so a test of the exhaustion boundary has to be able to read it.
    std::uint64_t owner_for_test() const { return owner_; }
    // The process-wide identity counter. Declared here and defined in the implementation,
    // because the counter and the rule that moves it live there; these three are the only
    // way a test can reach the boundary without aborting (constructing an exhausted pool
    // is a process stop by design).
    static bool          pool_token_available_for_test();
    static void          force_pool_token_for_test(std::uint64_t next);
    static std::uint64_t peek_pool_token_for_test();
    // The mint itself, so the boundary can be exercised as behaviour rather than as a
    // predicate over the counter. A test that only asks "is the counter spent?" cannot
    // tell a refusal that happens before the store from one that happens after it, and
    // that ordering is the whole property: the post-store version reissues token 1.
    static bool          mint_token_for_test(std::uint64_t * out);
#endif

private:
    struct entry {
        id            slot_id = no_reservation;
        std::uint64_t epoch   = 0;
        std::uint32_t cells   = 0;
    };

    static constexpr int max_entries = 16;
    // True when no live reservation holds this slot id. A slot id is never recycled
    // while it is outstanding; the epoch is what makes recycling safe after release.
    bool id_in_use(id candidate) const;

    std::uint32_t capacity_;
    std::uint32_t borrowed_        = 0;
    std::uint32_t high_water_      = 0;
    std::uint32_t refusals_cells_  = 0;
    std::uint32_t refusals_table_  = 0;
    std::uint32_t refusals_stale_  = 0;
    std::uint32_t refusals_foreign_= 0;
    std::uint32_t refusals_exhausted_ = 0;
    std::uint32_t live_            = 0;
    id            next_id_         = 1;
    // Monotonic for the pool's lifetime: never decremented, never reset by
    // release_all(), because a handle that survived a teardown must stay detectable as
    // stale.
    //
    // 0 is the exhausted sentinel, and it is not "no epoch": an issued epoch is never 0
    // because the counter starts at 1 and only ever moves forward. Reaching the maximum
    // therefore stops issuing rather than repeating -- the previous version saturated at
    // the maximum and handed the same epoch out forever, which combined with a wrapped
    // slot id to reproduce exactly the stale-handle ABA this class is built to prevent.
    // A refusal at 2^64 borrows is not a limitation anyone meets; repeating an epoch is
    // a wrong answer, and wrong answers do not get to be "unlikely enough".
    std::uint64_t next_epoch_      = 1;
    // This pool's identity in the reservation handles it issues. Minted from a process-wide
    // counter at construction; see the note on `reservation::owner`.
    //
    // Never 0 in a constructed pool: 0 is the counter's "spent" state, not a token, and
    // minting refuses at that point rather than wrapping. See mint_owner() for why the
    // decision has to be made before the counter moves.
    std::uint64_t owner_           = 0;
    entry         entries_[max_entries]{};
};

// One station per process, constructed on first use so no static-initialization
// order can expose it half-built to a seam that runs from a constructor.
station & shared();

} // namespace herd::prefab
