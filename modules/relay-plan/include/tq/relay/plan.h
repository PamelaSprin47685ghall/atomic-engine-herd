// tq/relay/plan.h — the relay lane's policy, with no backend and no ggml.
//
// The fork's ggml-relay-plan.h held the decisions the relay provider and the
// compose runtime both need: plan identity, the plan cache, the exact channel
// pool layout, and the switch matrix. All of that is pure state and arithmetic —
// it names slots and offsets and never allocates, never touches a device, never
// waits. It is therefore a station in this project's sense, and it can be tested
// without a GPU, a model, or a patched binary.
//
// Three things were changed while porting, each for a defect that the original
// shape invited. They are called out where they appear:
//
//   1. acquire() returns a handle (index + generation), not a pointer into the
//      cache. A pointer outlives a flush() and then points at a recycled slot;
//      a handle carries the generation, so a stale one is refused instead of
//      silently naming someone else's plan.
//   2. every mutation validates the generation, so "act on the plan I looked up
//      earlier" cannot become "act on whatever occupies that slot now".
//   3. channel_slots::retire() requires proof that no reader is still holding a
//      channel. It used to clear on the caller's word alone, which is a
//      policy that only works while every caller is correct.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace tq::relay {

// One provider owns at most eight prepared plans and eight captured frames.
inline constexpr std::size_t plan_capacity      = 8;
inline constexpr std::size_t frame_capacity_max = 8;
// The relay channel table is fixed at 256 slots; the working set uses <= 181.
inline constexpr std::size_t channel_capacity   = 256;
// control is 16 cache-line words (generation, 5 ready, 5 ack, 5 error).
inline constexpr std::size_t control_bytes      = 16 * 64;
inline constexpr std::size_t pool_alignment     = 64;
inline constexpr std::size_t pool_budget_default = 68ull << 20;
// 256 * (1024 + 65536 * 4 * 6) bytes, the pre-plan-union fixed pool.
inline constexpr std::size_t pool_bytes_legacy   = 402915328;

static_assert((pool_alignment & (pool_alignment - 1)) == 0, "pool alignment must be a power of two");
static_assert(control_bytes % pool_alignment == 0, "control bytes must preserve pool alignment");

// Plan identity. Two epochs with the same key must be interchangeable for every
// captured graph. `channels` is part of the identity: a 180- and a 181-channel
// plan never share a slot. `binding` is the relay-internal second half — a
// digest of the terminal bindings the plan was prepared with — because a graph
// rebuild can keep the coarse fields identical while the storage moves, and
// "same key, different binding" must be a miss rather than a wrong reuse.
struct plan_key {
    std::uint64_t model       = 0;
    std::uint32_t window      = 0;
    std::uint32_t n_tokens    = 0;
    std::uint32_t n_outputs   = 0;
    std::int32_t  rs_class    = 0;
    std::uint32_t n_kv_bucket = 0;
    std::uint64_t alloc_gen   = 0;
    std::uint32_t channels    = 0;
    std::uint64_t binding     = 0;

    friend bool operator==(const plan_key & a, const plan_key & b) noexcept {
        return a.model == b.model && a.window == b.window && a.n_tokens == b.n_tokens &&
               a.n_outputs == b.n_outputs && a.rs_class == b.rs_class &&
               a.n_kv_bucket == b.n_kv_bucket && a.alloc_gen == b.alloc_gen &&
               a.channels == b.channels && a.binding == b.binding;
    }
    friend bool operator!=(const plan_key & a, const plan_key & b) noexcept { return !(a == b); }
};

struct plan_stats {
    std::uint64_t hits             = 0;
    std::uint64_t misses           = 0;
    std::uint64_t inserts          = 0;
    std::uint64_t evictions        = 0;
    std::uint64_t pinned_refusals  = 0;
    std::uint64_t flushes          = 0;
    std::uint64_t stale_generation = 0;
};

struct plan_slot {
    bool          valid      = false;
    plan_key      key        = {};
    std::uint64_t generation = 0; // unique per insert; never reused
    std::uint64_t last_used  = 0;
    std::uint32_t pins       = 0;
    std::uint32_t index      = 0;
};

// A borrowed reference to one plan. The generation is what makes it safe to hold
// across other cache activity: at() re-checks it, so a handle whose slot has been
// evicted or flushed reports "gone" instead of describing the new occupant.
struct plan_handle {
    std::uint32_t index      = 0;
    std::uint64_t generation = 0;

    bool valid() const noexcept { return generation != 0; }
};

// Fixed-capacity plan cache. Hit touches the slot and returns it without
// pinning; pin() protects a plan while a graph is executing, release() ends
// that protection. A miss inserts into an empty slot or evicts the least
// recently used unpinned slot. Every slot pinned fails closed (returns an
// invalid handle) instead of silently dropping a plan.
//
// Threading: none of the classes in this header lock. A station is owned by one
// window and entered from the thread that is submitting that window's work --
// the same "one thread per scheduler" premise upstream's schedulers already
// make. Two windows have two stations and share nothing, which is why the
// independence matters more than a lock here: a lock would make concurrent use
// safe but would not make the recording order meaningful.
//
// A caller that wants to use one station from two threads must serialize the
// calls itself. That requirement is stated rather than silently unmet: there is
// no atomic, no lock, and no claim anywhere that these types are thread-safe.
class plan_cache {
public:
    static constexpr std::size_t capacity = plan_capacity;

    plan_handle find(const plan_key & key) const noexcept;

    // On a miss, `evicted` receives the generation displaced from the victim
    // slot (0 for an empty victim). The caller must retire that generation's
    // frames and channel slots before using the returned slot. An invalid
    // handle means the cache is full of pinned plans.
    plan_handle acquire(const plan_key & key, std::uint64_t epoch,
                        std::uint64_t * evicted = nullptr) noexcept;

    // The slot a handle names, or null when that generation is gone. This is the
    // replacement for the pointer the fork returned: it cannot dangle.
    const plan_slot * at(plan_handle handle) const noexcept;

    bool pin(plan_handle handle) noexcept;
    bool release(plan_handle handle) noexcept;

    // Refused while pinned, and a refusal never half-acts.
    bool flush(plan_handle handle) noexcept;
    // All-or-nothing: if any slot is pinned, nothing is cleared.
    bool flush_all() noexcept;

    std::size_t live() const noexcept;
    std::size_t pinned() const noexcept;
    bool empty() const noexcept { return live() == 0; }

    const plan_stats & stats() const noexcept { return stats_; }
    void reset_stats() noexcept { stats_ = {}; }

private:
    plan_slot * mutable_find(const plan_key & key) noexcept;
    plan_slot * slot_of(plan_handle handle) noexcept;
    int         pick_victim() const noexcept;

    std::array<plan_slot, capacity> slots_{};
    plan_stats                      stats_{};
    std::uint64_t                   next_generation_ = 0;
};

// ---------------------------------------------------------------------------
// Frame metadata LRU. The provider owns the native graph; this mirrors per-frame
// identity so a frame is a fast-path candidate only when uid, plan generation
// and signature all match.
// ---------------------------------------------------------------------------
class frame_lru {
public:
    struct entry {
        bool          valid      = false;
        std::uint64_t uid        = 0;
        std::uint64_t generation = 0;
        std::uint64_t signature  = 0;
        std::uint64_t last_used  = 0;
    };

    explicit frame_lru(std::size_t capacity = 2) noexcept;

    // A frame is reusable only when every identity field matches. Checking uid
    // alone would reuse a frame built for another plan.
    const entry * find(std::uint64_t uid, std::uint64_t generation,
                       std::uint64_t signature) const noexcept;

    void record(std::uint64_t uid, std::uint64_t generation, std::uint64_t signature,
                std::uint64_t epoch) noexcept;
    // Drops every frame belonging to a plan generation. Returns how many.
    std::size_t retire_generation(std::uint64_t generation) noexcept;
    void clear() noexcept;
    std::size_t live() const noexcept;
    std::size_t capacity() const noexcept { return capacity_; }

private:
    std::array<entry, frame_capacity_max> entries_{};
    std::size_t capacity_;
};

// ---------------------------------------------------------------------------
// Channel pool: exact layout arithmetic. Pure functions, so the budget question
// is answered the same way in a test and in a provider.
// ---------------------------------------------------------------------------
struct pool_layout {
    std::array<std::size_t, channel_capacity> offset{};
    std::array<std::size_t, channel_capacity> capacity{};
    std::size_t bytes = 0;
    std::size_t count = 0;

    void clear() noexcept;
};

// Bytes for one channel slot: control block plus payload, rounded up so the next
// control block stays aligned. False on overflow or a zero element count.
bool pool_slot_bytes(std::size_t elements, std::size_t stride, std::size_t & bytes) noexcept;

// Byte need for a plan over `ranks` + output. False on overflow or bad counts.
bool pool_need(const std::size_t * elements, std::size_t count, std::size_t ranks,
               std::size_t & need) noexcept;

// Exact layout when it fits `budget`; `out` is cleared on failure so a caller
// cannot mistake a partial layout for a usable one.
bool compute_pool_layout(const std::size_t * elements, std::size_t count, std::size_t ranks,
                         std::size_t budget, pool_layout & out) noexcept;

bool layout_covers(const pool_layout & layout, const std::size_t * elements,
                   std::size_t count) noexcept;

enum class pool_growth {
    keep,        // the current layout covers the plan
    grow,        // the plan fits the budget and needs a new layout
    fail_budget, // the plan cannot fit the budget at all
};

pool_growth pool_growth_decision(const pool_layout & layout, const std::size_t * elements,
                                 std::size_t count, std::size_t ranks, std::size_t budget,
                                 std::size_t & need) noexcept;

// ---------------------------------------------------------------------------
// Switch matrix. Every switch defaults to the legacy behaviour; a value that is
// neither 0 nor 1 is rejected with a message naming the variable, because a
// silently ignored switch is a configuration the operator believes they set.
// ---------------------------------------------------------------------------
struct config {
    bool        plan_union    = false;
    std::size_t frames        = 2;
    bool        pool_exact    = false;
    bool        slot_channels = false;
    bool        verify_uid    = false;
    bool        plan_stats    = false;
    bool        graph_uid     = false;
    bool        plan_flush    = false;
};

bool parse_switch(const char * value, bool & out) noexcept;

// Returns false and fills `error` when a value is unusable. `error` is always
// terminated; pass nullptr/0 to ignore it.
bool parse_config(const char * plan_union, const char * frames, const char * pool_exact,
                  const char * slot_channels, const char * verify_uid,
                  const char * plan_stats, const char * graph_uid,
                  const char * plan_flush, config & out,
                  char * error, std::size_t error_size) noexcept;

// True when `slot_channels` was not set and inherited `plan_union`. The wiring
// uses this to report which lane is actually active without re-deriving it.
bool slot_channels_inherited(const char * slot_channels) noexcept;

// ---------------------------------------------------------------------------
// Per-window channel ownership. Eight plan slots, each holding up to 256 channel
// states. A slot is bound to one plan generation; retirement clears it only when
// no reader still owns a channel. Two windows own independent tables, so the
// same generation number in both never shares state.
// ---------------------------------------------------------------------------
template <typename ChannelState>
class channel_slots {
public:
    static constexpr std::size_t slot_capacity = plan_capacity;

    struct slot {
        std::uint64_t generation = 0; // 0 = empty
        std::uint64_t last_used  = 0;
        std::array<std::unique_ptr<ChannelState>, channel_capacity> states;
    };

    slot * find(std::uint64_t generation) noexcept {
        if (generation == 0) {
            return nullptr;
        }
        for (auto & candidate : slots_) {
            if (candidate.generation == generation) {
                return &candidate;
            }
        }
        return nullptr;
    }

    const slot * find(std::uint64_t generation) const noexcept {
        return const_cast<channel_slots *>(this)->find(generation);
    }

    // Bind a generation to a slot, reusing an existing binding. An empty slot is
    // preferred; otherwise the least recently used slot for which `idle(slot)`
    // is true is cleared and rebound. Null when every slot still holds a reader.
    template <typename IdleFn>
    slot * acquire(std::uint64_t generation, std::uint64_t epoch, IdleFn && idle) {
        if (generation == 0) {
            return nullptr;
        }
        if (slot * existing = find(generation)) {
            existing->last_used = epoch;
            return existing;
        }
        for (auto & candidate : slots_) {
            if (candidate.generation == 0) {
                candidate.generation = generation;
                candidate.last_used  = epoch;
                return &candidate;
            }
        }
        slot * victim = nullptr;
        for (auto & candidate : slots_) {
            if (!idle(candidate)) {
                continue;
            }
            if (victim == nullptr || candidate.last_used < victim->last_used) {
                victim = &candidate;
            }
        }
        if (victim == nullptr) {
            return nullptr;
        }
        victim->states = {};
        victim->generation = generation;
        victim->last_used  = epoch;
        return victim;
    }

    // Drop one generation's slot, but only when `idle` agrees that no reader
    // still holds a channel of it. The retired state is destroyed here, so
    // trusting the caller's word would be a use-after-free with a comment
    // attached; the predicate makes the trust unnecessary.
    template <typename IdleFn>
    bool retire(std::uint64_t generation, IdleFn && idle) noexcept {
        slot * existing = find(generation);
        if (existing == nullptr || !idle(*existing)) {
            return false;
        }
        *existing = {};
        return true;
    }

    std::size_t live() const noexcept {
        std::size_t count = 0;
        for (const auto & candidate : slots_) {
            count += candidate.generation != 0 ? 1 : 0;
        }
        return count;
    }

    std::size_t live_channels() const noexcept {
        std::size_t count = 0;
        for (const auto & candidate : slots_) {
            if (candidate.generation == 0) {
                continue;
            }
            for (const auto & owner : candidate.states) {
                count += owner ? 1 : 0;
            }
        }
        return count;
    }

    bool empty() const noexcept { return live() == 0; }

    void clear() noexcept {
        for (auto & candidate : slots_) {
            candidate = {};
        }
    }

    slot & at(std::size_t index) noexcept { return slots_[index]; }
    const slot & at(std::size_t index) const noexcept { return slots_[index]; }

private:
    std::array<slot, slot_capacity> slots_{};
};

} // namespace tq::relay
