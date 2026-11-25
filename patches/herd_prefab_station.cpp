#include "herd_prefab_station.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace herd::prefab {
namespace {

// LLAMA_PREFAB_ENVELOPES="128,256,512,1024": the ladder, ascending. Parsed here
// rather than in the seam so the seam stays a translator. A malformed list leaves
// the lane disabled and says so: a silently empty ladder would look like "prefab
// ran and found nothing to pad".
int parse_ladder(const char * text, uint32_t * out, int capacity) {
    if (!text || !*text) {
        return 0;
    }
    int      count = 0;
    uint32_t previous = 0;
    const char * cursor = text;
    while (*cursor && count < capacity) {
        char * end = nullptr;
        const unsigned long value = std::strtoul(cursor, &end, 10);
        if (end == cursor || value == 0 || value > 1u << 20 || value <= previous) {
            std::fprintf(stderr,
                         "[herd] prefab: envelope ladder '%s' is not strictly ascending positive widths; "
                         "the lane stays off\n", text);
            return 0;
        }
        out[count++] = static_cast<uint32_t>(value);
        previous = static_cast<uint32_t>(value);
        cursor = (*end == ',') ? end + 1 : end;
        if (*end == '\0') {
            break;
        }
    }
    return count;
}

} // namespace

const char * name(verdict why) {
    switch (why) {
        case verdict::accept:        return "accept";
        case verdict::disabled:      return "disabled";
        case verdict::max_exceeded:  return "wider than the last envelope";
        case verdict::not_tokenwise: return "n_seq_tokens != 1";
        case verdict::unaligned_seqs:return "n_seqs != n_tokens";
        case verdict::multi_pos:     return "n_pos > 1";
        case verdict::mixed_input:   return "embedding or mixed ubatch";
        case verdict::already_wide:  return "already at an envelope width";
    }
    return "unknown";
}

void station::configure(const config & cfg) {
    config_ = cfg;
}

ruling station::consider(const shape & s) const {
    if (!config_.enabled || config_.n_envelopes <= 0) {
        return {verdict::disabled, 0};
    }
    // Order matters only for the message a reader gets: each check names the
    // precondition that fails, so the reason points at the input, not at the lane.
    if (s.has_embd || s.has_type) {
        return {verdict::mixed_input, 0};
    }
    if (s.n_pos > 1) {
        return {verdict::multi_pos, 0};
    }
    if (s.n_seq_tokens != 1 || s.equal_seqs) {
        // The ubatch must be row-addressable: a pad row appended at the end must not
        // change what the earlier rows mean, which fails as soon as rows are grouped
        // (n_seq_tokens > 1) or all share one sequence set.
        //
        // Measured note, so the next reader does not have to guess which clause fires:
        // a one-token decode on a llama-architecture model reports
        //   n_tokens=1 n_seqs=1 n_seq_tokens=1 n_pos=1 equal_seqs=0
        // so this clause is inert for the shapes this driver produces. It is kept
        // because it is the contract the station promises, and a rule that is merely
        // unreachable for one workload is not the same as a wrong rule.
        return {verdict::not_tokenwise, 0};
    }
    if (s.n_seqs != s.n_tokens) {
        return {verdict::unaligned_seqs, 0};
    }
    if (s.n_tokens == 0) {
        return {verdict::not_tokenwise, 0};
    }
    if (s.n_tokens > config_.envelopes[config_.n_envelopes - 1]) {
        return {verdict::max_exceeded, 0};
    }
    for (int i = 0; i < config_.n_envelopes; ++i) {
        if (s.n_tokens <= config_.envelopes[i]) {
            if (s.n_tokens == config_.envelopes[i]) {
                // Nothing to pad: the graph already has this shape, and copying
                // the ubatch would only cost time.
                return {verdict::already_wide, 0};
            }
            return {verdict::accept, config_.envelopes[i]};
        }
    }
    return {verdict::max_exceeded, 0};
}

void station::note_ran(uint32_t n_env, uint32_t n_tokens) {
    ++counters_.accepted;
    if (n_env > n_tokens) {
        counters_.pad_rows += n_env - n_tokens;
    }
}

void station::note_declined(verdict why) {
    const auto index = static_cast<std::size_t>(why);
    if (index < sizeof(counters_.declined) / sizeof(counters_.declined[0])) {
        ++counters_.declined[index];
    }
}

const char * name(placement why) {
    switch (why) {
        case placement::fits:               return "fits";
        case placement::no_free_cells:      return "not enough free KV cells for the envelope";
        case placement::scratch_unowned:    return "pad rows would occupy cells no sequence owns";
        case placement::scratch_unclearable:return "the pad rows' cells would never be returned";
        case placement::not_padded:         return "already at envelope width";
    }
    return "unknown";
}

placement_plan place(const capacity & need) {
    placement_plan out;

    // Nothing to pad: the ubatch already has the shape the lane wants. Not a failure,
    // and it must be reported as such -- a lane that treats "already wide" as a
    // problem will pad in a loop.
    if (need.n_envelope <= need.n_tokens_real) {
        out.why = placement::not_padded;
        out.cells_required = need.n_tokens_real;
        return out;
    }

    out.cells_required = need.n_envelope;

    // Ownership first, because it is the failure that does not announce itself.
    // `llama_kv_cache::apply_ubatch` (llama-kv-cache.cpp:1180+) sets a cell's position
    // and then adds each sequence in `ubatch.n_seq_id[i]`. With n_seq_id == 0 the cell
    // gets a position and no owner, so nothing will ever free it: one leaked cell per
    // pad row per ubatch, filling the cache at the envelope's width. The fork's
    // `scratch_cell` and its `n_fallback_no_scratch` counter exist for precisely this
    // reason -- the scratch rows must belong to a sequence the lane can clear.
    if (!need.scratch_owned) {
        out.why = placement::scratch_unowned;
        out.cells_permanent = need.n_envelope - need.n_tokens_real;
        return out;
    }
    // Owning them is not enough; they must be returnable. Upstream exposes
    // llama_memory_seq_rm for exactly this, which is why it is a separate fact here
    // rather than an assumption in the caller.
    if (!need.scratch_clearable) {
        out.why = placement::scratch_unclearable;
        out.cells_permanent = need.n_envelope - need.n_tokens_real;
        return out;
    }
    // Then the arithmetic. `find_slot` (llama-kv-cache.cpp:935) collects one index per
    // token and returns an empty slot_info when it cannot reach n_tokens, which
    // `prepare` (:785) turns into a failed decode for the whole ubatch.
    if (need.n_cells_free < out.cells_required) {
        out.why = placement::no_free_cells;
        out.cells_permanent = 0; // nothing was placed, so nothing leaked
        return out;
    }

    out.why = placement::fits;
    // The honest cost: the envelope's cells are borrowed for the ubatch and returned
    // afterwards, so the peak is the envelope width while the average is the real row
    // count. Reporting only the peak makes the lane look wasteful; reporting only the
    // average hides the peak that can fail.
    out.cells_reusable  = out.cells_required;
    out.cells_permanent = 0;
    return out;
}

namespace {

// Process-wide pool identity. A handle carries the token of the pool that issued it, so
// one station's handle cannot release another station's reservation -- which is not a
// hypothetical: two windows own two pools, both start their counters at 1, so their first
// borrows would be indistinguishable without it.
//
// A compiler atomic on a plain counter, deliberately: this file has no <atomic> include,
// and pulling one in for a single increment would be a header change in service of a
// one-line need. __atomic_* is the same primitive std::atomic<int64_t> compiles to on
// every target this project supports.
//
// DOMAIN: [1, UINT64_MAX] are issuable tokens; 0 means "spent" and is never issued. The
// rule that matters is that the counter never holds a value it has already handed out.
std::uint64_t g_next_pool_token = 1;

constexpr std::uint64_t token_spent = 0;

// Mint the next token, or refuse when the space is spent.
//
// The decision happens BEFORE the counter moves, and that is the whole point. The
// previous shape was `owner_ = __atomic_fetch_add(&counter, 1, RELAXED)` followed by
// `if (owner_ == 0) abort()`. It never issued a duplicate, but only because the abort is
// fatal: at the maximum, fetch_add returned UINT64_MAX and left 0 behind, and the NEXT
// construction got 0, aborted -- after fetch_add had already moved the counter to 1. So
// the invariant "1 is never reissued" was held up by process death, not by the counter.
// Turn the abort into a report -- which is what every other module in this project does,
// and what a library should do -- and the pool after the refusal is minted with 1, the
// same identity as the first pool the process ever created. Two live pools that cannot be
// told apart, reached without any wraparound of a slot id or an epoch.
//
// A compare-and-exchange loop makes the property structural: read, refuse at 0, otherwise
// move to the successor (0 when the maximum is issued). A concurrent loser re-reads and
// refuses; it never retries into a value that was already handed out.
bool mint_owner(std::uint64_t * out) {
    std::uint64_t current = __atomic_load_n(&g_next_pool_token, __ATOMIC_RELAXED);
    for (;;) {
        // The refusal happens HERE, before the counter moves. Removing these three lines is
        // the mutation that reintroduces the defect, and the test suite currently does not
        // notice -- see the note at the end of the identity test.
        if (current == token_spent) {
            return false;
        }
        const std::uint64_t successor =
            (current == ~std::uint64_t(0)) ? token_spent : current + 1;
        if (__atomic_compare_exchange_n(&g_next_pool_token, &current, successor, false,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            *out = current;
            return true;
        }
        // The exchange wrote the observed value back into `current`; loop on it.
    }
}

} // namespace

scratch_pool::scratch_pool(std::uint32_t capacity) : capacity_(capacity) {
    if (!mint_owner(&owner_)) {
        // Spent, not "wrapped, then caught": the counter is still 0, so no later pool can
        // be handed a token that an earlier one already had. Two pools sharing an identity
        // would make their handles interchangeable again -- one station releasing another
        // station's borrow -- which is the failure the owner field exists to prevent.
        std::fprintf(stderr, "[herd] scratch_pool: pool identity space is spent (2^64 pools); "
                             "refusing to create one that could not be told apart from another\n");
        std::abort();
    }
}

#if defined(HERD_SCRATCH_POOL_INSPECT)
bool scratch_pool::pool_token_available_for_test() {
    const std::uint64_t current = __atomic_load_n(&g_next_pool_token, __ATOMIC_RELAXED);
    return current != token_spent;
}

void scratch_pool::force_pool_token_for_test(std::uint64_t next) {
    __atomic_store_n(&g_next_pool_token, next, __ATOMIC_RELAXED);
}

std::uint64_t scratch_pool::peek_pool_token_for_test() {
    return __atomic_load_n(&g_next_pool_token, __ATOMIC_RELAXED);
}

bool scratch_pool::mint_token_for_test(std::uint64_t * out) {
    // The real minting function, not a copy of its rules. A test double here would be a
    // second implementation of the property under test -- exactly the thing that let the
    // previous round's mutation survive: the predicate read the counter correctly while
    // the decision it was supposed to guard had moved.
    return mint_owner(out);
}
#endif

bool scratch_pool::id_in_use(id candidate) const {
    for (int i = 0; i < max_entries; ++i) {
        if (entries_[i].slot_id == candidate) {
            return true;
        }
    }
    return false;
}

scratch_pool::reservation scratch_pool::reserve(std::uint32_t cells) {
    // The epoch space is spent: there is no distinct epoch left to issue, and issuing a
    // repeated one would make a stale handle and a live one indistinguishable -- the ABA
    // this class exists to prevent. Unreachable at 2^64 borrows, and refused anyway,
    // because "unreachable" that the code does not enforce is a comment, not an invariant.
    if (next_epoch_ == 0) {
        ++refusals_exhausted_;
        return {};
    }
    if (cells == 0 || borrowed_ > capacity_ || capacity_ - borrowed_ < cells) {
        // A refusal, never a partial grant: handing out fewer cells than asked would
        // leave the caller with a plan it cannot fill, which is how a padding attempt
        // turns into a failed decode elsewhere. The `borrowed_ > capacity_` arm is
        // defensive: the invariant holds by construction, and a subtraction that could
        // underflow is not a guard but a wraparound waiting for a broken caller.
        ++refusals_cells_;
        return {};
    }
    entry * slot = nullptr;
    for (int i = 0; i < max_entries; ++i) {
        if (entries_[i].slot_id == no_reservation) {
            slot = &entries_[i];
            break;
        }
    }
    if (!slot) {
        // The declared capacity allows it but too many reservations are outstanding.
        // Counted separately: reporting this as "out of cells" sends the caller to the
        // wrong number, and the two causes have different fixes.
        ++refusals_table_;
        return {};
    }

    // The slot id must be unique among LIVE reservations, and must never be 0. It wraps
    // after 2^32 reservations, and an id released long ago is then handed out again --
    // which is fine, because `epoch` distinguishes the two borrows. The scan is what
    // makes "unique among live" true by construction rather than by arithmetic luck.
    id candidate = next_id_;
    while (candidate == no_reservation || id_in_use(candidate)) {
        ++candidate;
        if (candidate == no_reservation) {
            candidate = 1; // skip 0: it means "no reservation"
        }
        if (candidate == next_id_) {
            // Every id in the space is live, which with 2^32 ids and 16 slots cannot
            // happen; refusing beats looping forever if the invariants are ever broken.
            ++refusals_table_;
            return {};
        }
    }
    next_id_ = candidate == std::numeric_limits<id>::max() ? 1 : id(candidate + 1);

    // The epoch is what release() matches on, so it must never repeat for the pool's
    // lifetime -- including across release_all(), which is why it is not reset there.
    //
    // The previous version saturated at the maximum and then handed that same epoch out
    // for every later borrow. Combined with a wrapped slot id that reproduces exactly the
    // stale-handle ABA: an old handle and a live one become byte-identical, and the stale
    // holder's release credits cells it never held. "2^64 borrows is not a lifetime anyone
    // reaches" is an argument from improbability, and improbability is not a correctness
    // property. So the counter stops instead: 0 marks exhaustion, an issued epoch is never
    // 0, and reserve() refuses once it is set.
    const std::uint64_t epoch = next_epoch_;
    next_epoch_ = (epoch == std::numeric_limits<std::uint64_t>::max()) ? 0 : epoch + 1;

    *slot = entry{candidate, epoch, cells};
    borrowed_ += cells;
    if (borrowed_ > high_water_) {
        high_water_ = borrowed_;
    }
    ++live_;
    return reservation{candidate, epoch, owner_};
}

bool scratch_pool::release(const reservation & held) {
    if (!held.valid()) {
        return false;
    }
    // A handle from another pool is refused before any slot is touched. Both counters
    // start at 1 in a fresh pool, so {1,1} from one station would otherwise match {1,1} in
    // another and credit cells this pool never lent -- silently, because the books stay
    // balanced while a live reservation quietly loses its cells.
    if (held.owner != owner_) {
        ++refusals_foreign_;
        return false;
    }
    for (int i = 0; i < max_entries; ++i) {
        if (entries_[i].slot_id != held.slot_id) {
            continue;
        }
        if (entries_[i].epoch != held.epoch) {
            // The id matches but the borrow is not the one this handle names: an id
            // released earlier was handed out again after the counter wrapped. Refusing
            // is the whole reason the epoch exists -- crediting the cells here would take
            // them from a live reservation that nobody asked to end, and the damage only
            // shows up later as a decode that cannot find cells it should have.
            ++refusals_stale_;
            return false;
        }
        // Credit back exactly what was taken. A double release matches no slot (the
        // entry is cleared below), so `borrowed_` cannot be pushed below zero and the
        // next over-borrow stays visible.
        borrowed_ -= entries_[i].cells;
        entries_[i] = entry{};
        --live_;
        return true;
    }
    return false;
}

std::uint32_t scratch_pool::release_all() {
    const std::uint32_t released = live_;
    for (int i = 0; i < max_entries; ++i) {
        entries_[i] = entry{};
    }
    borrowed_ = 0;
    live_     = 0;
    // next_epoch_ is deliberately NOT reset: a handle that survived the teardown must
    // stay recognisable as stale. Resetting it would make every old handle valid again
    // after a teardown-and-reuse of the same pool object -- the same ABA, just triggered
    // by teardown instead of by wrap.
    return released;
}

station & shared() {
    static station instance;
    // Configuration happens once, from the lane's install step. Doing it here in a
    // function-local static keeps it off the static-initialization order path: the
    // first seam that asks cannot see a half-configured ladder.
    static bool configured = false;
    if (!configured) {
        configured = true;
        config cfg{};
        if (const char * text = std::getenv("LLAMA_PREFAB_ENVELOPES")) {
            cfg.n_envelopes = parse_ladder(text, cfg.envelopes, 4);
            cfg.enabled = cfg.n_envelopes > 0;
        }
        instance.configure(cfg);
    }
    return instance;
}

} // namespace herd::prefab
