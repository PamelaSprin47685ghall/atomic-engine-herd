// The relay lane's policy, tested without a GPU, a model, or upstream.
//
// Each case names the decision it pins. The interesting ones are the refusals:
// a plan that is pinned must not be evicted, a handle whose generation is gone
// must not describe the slot's new occupant, and a configuration that asks for
// something the lane cannot do must come back with the variable's name.
#include <tq/relay/plan.h>

#include <cstdio>
#include <cstring>
#include <string>
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

plan_key key_of(std::uint32_t n_tokens, std::uint32_t channels = 8, std::uint64_t binding = 0) {
    plan_key key;
    key.model    = 1;
    key.window   = 2;
    key.n_tokens = n_tokens;
    key.channels = channels;
    key.binding  = binding;
    return key;
}

// ---------------------------------------------------------------------------
// identity
// ---------------------------------------------------------------------------

// The coarse fields alone are not identity: a 180- and a 181-channel plan must
// never share a slot, because their channel tables have different shapes.
void channels_are_part_of_identity() {
    plan_cache cache;
    const auto a = cache.acquire(key_of(16, 180), 1);
    check(a.valid(), "first plan is admitted");
    const auto b = cache.acquire(key_of(16, 181), 2);
    check(b.valid() && b.generation != a.generation, "a different channel count is a different plan");
    check(cache.live() == 2, "both plans live");
}

// A graph rebuild can keep every coarse field identical while the terminal
// bindings move. Reusing the old plan then would run a graph over stale storage.
void binding_digest_separates_same_shape() {
    plan_cache cache;
    const auto a = cache.acquire(key_of(16, 8, 0xA), 1);
    const auto b = cache.acquire(key_of(16, 8, 0xB), 2);
    check(a.generation != b.generation, "same shape with another binding is a miss");
    check(cache.stats().misses == 2, "the second lookup counted as a miss");
}

// ---------------------------------------------------------------------------
// eviction and pinning
// ---------------------------------------------------------------------------

void lru_evicts_the_oldest_unpinned() {
    plan_cache cache;
    std::vector<plan_handle> handles;
    for (std::uint32_t i = 0; i < plan_cache::capacity; ++i) {
        handles.push_back(cache.acquire(key_of(1 + i), i));
        check(handles.back().valid(), "each plan up to capacity is admitted");
    }
    // Touch the oldest so the second becomes the least recently used.
    check(cache.find(key_of(1)).valid(), "the first plan is still findable");
    cache.acquire(key_of(1), 100); // a hit updates last_used

    std::uint64_t evicted = 0;
    const auto fresh = cache.acquire(key_of(99), 200, &evicted);
    check(fresh.valid(), "one more plan is admitted by eviction");
    check(evicted == handles[1].generation, "the evicted generation is the least recently used");
    check(cache.at(handles[1]) == nullptr, "the evicted handle no longer resolves");
    check(cache.at(handles[0]) != nullptr, "the touched plan survived");
}

// A plan being executed must not be evicted under it. When every slot is pinned
// the cache refuses a new plan instead of destroying one.
void pinned_plans_are_never_evicted() {
    plan_cache cache;
    std::vector<plan_handle> handles;
    for (std::uint32_t i = 0; i < plan_cache::capacity; ++i) {
        handles.push_back(cache.acquire(key_of(1 + i), i));
        check(cache.pin(handles.back()), "pin is accepted for a live plan");
    }
    check(cache.pinned() == plan_cache::capacity, "every slot is pinned");

    const auto refused = cache.acquire(key_of(99), 500);
    check(!refused.valid(), "a full cache of pinned plans refuses, it does not evict");
    check(cache.stats().pinned_refusals == 1, "the refusal is counted");
    for (const auto & handle : handles) {
        check(cache.at(handle) != nullptr, "no pinned plan was displaced");
    }

    check(cache.release(handles[3]), "release is accepted");
    check(!cache.release(handles[3]), "releasing twice is refused rather than going negative");
    check(cache.acquire(key_of(99), 600).valid(), "the freed slot takes the new plan");
    check(cache.at(handles[3]) == nullptr, "the old plan is gone from that slot");
}

// The handle fix: a pointer into the cache would still look usable after a
// flush, and the slot's new occupant would be silently addressed through it.
void stale_handles_are_refused_not_recycled() {
    plan_cache cache;
    const auto first = cache.acquire(key_of(4), 1);
    check(cache.flush(first), "an unpinned plan flushes");
    check(cache.at(first) == nullptr, "a flushed generation does not resolve");
    check(!cache.pin(first), "pinning a flushed generation is refused");

    const auto second = cache.acquire(key_of(4), 2);
    check(second.index == first.index, "the slot is reused, which is the hazard");
    check(second.generation != first.generation, "but the generation differs");
    check(cache.at(first) == nullptr, "the old handle still does not name the new plan");
    check(cache.at(second) != nullptr, "the new handle does");
}

// Every pin must be undone or the plan is stuck; the count must be exact.
void pins_are_counted() {
    plan_cache cache;
    const auto handle = cache.acquire(key_of(2), 1);
    check(cache.pin(handle) && cache.pin(handle), "a plan can be pinned twice (two graphs)");
    check(!cache.flush(handle), "a pinned plan refuses to flush");
    check(cache.release(handle), "one release");
    check(!cache.flush(handle), "still pinned by the second");
    check(cache.release(handle), "second release");
    check(cache.flush(handle), "now it flushes");
}

// flush_all is all-or-nothing: a partial clear would leave the caller unable to
// tell which plans survived.
void flush_all_is_atomic() {
    plan_cache cache;
    const auto a = cache.acquire(key_of(1), 1);
    const auto b = cache.acquire(key_of(2), 2);
    check(cache.pin(b), "pin the second");
    check(!cache.flush_all(), "a pinned plan blocks the whole flush");
    check(cache.at(a) != nullptr && cache.at(b) != nullptr, "nothing was cleared");
    check(cache.release(b), "release");
    check(cache.flush_all(), "the flush succeeds once nothing is pinned");
    check(cache.empty(), "the cache is empty");
    check(cache.stats().flushes == 2, "both slots are counted as flushed");
}

// ---------------------------------------------------------------------------
// frames
// ---------------------------------------------------------------------------

// A frame is reusable only when uid, plan generation and signature all match.
// Matching on uid alone would hand back a frame built for another plan.
void frames_need_all_three_fields() {
    frame_lru frames(4);
    frames.record(/*uid*/ 7, /*generation*/ 11, /*signature*/ 0xABC, /*epoch*/ 1);
    check(frames.find(7, 11, 0xABC) != nullptr, "an exact match is found");
    check(frames.find(7, 11, 0xABD) == nullptr, "another signature is not the same frame");
    check(frames.find(7, 12, 0xABC) == nullptr, "another plan generation is not the same frame");
    check(frames.find(8, 11, 0xABC) == nullptr, "another uid is not the same frame");
}

void frame_recording_is_idempotent_and_retirable() {
    frame_lru frames(4);
    frames.record(1, 11, 0xA, 1);
    frames.record(1, 11, 0xA, 2);
    check(frames.live() == 1, "re-recording a frame does not consume a second entry");

    frames.record(2, 11, 0xB, 3);
    check(frames.retire_generation(11) == 2, "retiring a generation drops all of its frames");
    check(frames.live() == 0, "nothing survives");
    frames.record(3, 12, 0xC, 4);
    check(frames.retire_generation(11) == 0, "retiring again removes nothing");
    check(frames.live() == 1, "another generation is untouched");
}

// A full table replaces the least recently used frame; the same rule as the plan
// cache, so one mental model covers both.
void frames_evict_the_oldest() {
    frame_lru frames(2);
    frames.record(1, 11, 0xA, 10);
    frames.record(2, 11, 0xB, 20);
    frames.record(2, 11, 0xB, 30); // refresh
    frames.record(3, 11, 0xC, 40);
    check(frames.find(1, 11, 0xA) == nullptr, "the least recently used frame was evicted");
    check(frames.find(2, 11, 0xB) != nullptr, "the refreshed frame survived");
    check(frames.find(3, 11, 0xC) != nullptr, "the new frame is present");
}

// ---------------------------------------------------------------------------
// pool arithmetic
// ---------------------------------------------------------------------------

// The layout must be exact: aligned control blocks, no double counting, and a
// byte total that matches the sum of the slots a test computes by hand.
void pool_layout_is_exact_and_aligned() {
    const std::size_t elements[] = {64, 128, 1};
    const std::size_t ranks = 5;
    const std::size_t stride = sizeof(float) * (ranks + 1);

    pool_layout layout;
    check(compute_pool_layout(elements, 3, ranks, pool_budget_default, layout),
          "a small plan fits the default budget");
    check(layout.count == 3, "all three channels are laid out");

    std::size_t expected = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        std::size_t slot = 0;
        check(pool_slot_bytes(elements[i], stride, slot), "each slot has a size");
        check(slot % pool_alignment == 0, "each slot is a whole number of alignments");
        check(layout.offset[i] == expected, "offsets follow each other without gaps");
        check(layout.capacity[i] == elements[i], "capacity is the element count, not the padded size");
        expected += slot;
    }
    check(layout.bytes == expected, "the total is the sum of the slots");
}

// A plan that cannot fit must say so rather than return a layout that overflows.
void pool_refuses_what_it_cannot_hold() {
    const std::size_t huge[] = {1ull << 40};
    pool_layout layout;
    check(!compute_pool_layout(huge, 1, 5, pool_budget_default, layout),
          "a plan beyond the budget is refused");
    check(layout.count == 0 && layout.bytes == 0, "a refusal leaves no partial layout");

    std::size_t need = 0;
    check(!pool_need(nullptr, 1, 5, need), "a null element list is refused");
    check(!pool_need(huge, 1, 0, need), "zero ranks is refused");
    check(!pool_need(huge, channel_capacity + 1, 5, need), "more channels than the table holds is refused");
    check(!pool_slot_bytes(0, sizeof(float), need), "zero elements is not a slot");
}

// The growth ruling: keep what covers the plan, grow when it fits the budget,
// refuse when it does not. A wrong answer here either reallocates every decode
// or runs out of memory mid-graph.
void pool_growth_rulings() {
    const std::size_t small[] = {64};
    const std::size_t bigger[] = {1024};
    const std::size_t huge[] = {1ull << 40};

    pool_layout layout;
    check(compute_pool_layout(small, 1, 5, pool_budget_default, layout), "layout the small plan");

    std::size_t need = 0;
    check(pool_growth_decision(layout, small, 1, 5, pool_budget_default, need) == pool_growth::keep,
          "a covering layout is kept");
    check(need == layout.bytes, "keep reports the existing size");

    check(pool_growth_decision(layout, bigger, 1, 5, pool_budget_default, need) == pool_growth::grow,
          "a larger plan that fits needs a new layout");
    check(need > layout.bytes, "grow reports the larger size");

    check(pool_growth_decision(layout, huge, 1, 5, pool_budget_default, need) == pool_growth::fail_budget,
          "a plan beyond the budget fails closed");
    check(pool_growth_decision(layout, huge, 1, 5, /*budget*/ 1024, need) == pool_growth::fail_budget,
          "and so does one beyond a small budget");
}

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

void config_defaults_match_the_legacy_lane() {
    config cfg;
    char error[128];
    check(parse_config(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       cfg, error, sizeof(error)), "all-unset is valid");
    check(!cfg.plan_union && !cfg.slot_channels, "the union lanes are off by default");
    check(cfg.frames == 2, "two frames is the default");
    check(!cfg.pool_exact && !cfg.graph_uid && !cfg.plan_flush, "the remaining switches are off");
    check(error[0] == '\0', "no message when nothing was rejected");
}

// The inheritance rule: setting only GGML_TP5_PLAN_UNION must also select the
// slotized channel lane, which is what the fork's default did.
void slot_channels_follow_the_union_switch() {
    config cfg;
    char error[128];
    check(parse_config("1", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       cfg, error, sizeof(error)), "union=1 parses");
    check(cfg.plan_union && cfg.slot_channels, "the channel lane follows the union lane");
    check(slot_channels_inherited(nullptr) && slot_channels_inherited(""), "unset is inheritable");

    check(parse_config("1", nullptr, nullptr, "0", nullptr, nullptr, nullptr, nullptr,
                       cfg, error, sizeof(error)), "an explicit channel switch parses");
    check(cfg.plan_union && !cfg.slot_channels, "an explicit 0 overrides the inheritance");
}

// A rejected value must name the variable. A silently ignored switch is a
// configuration the operator believes they set.
void bad_values_are_rejected_by_name() {
    config cfg;
    char error[128];
    struct probe {
        const char * plan_union; const char * frames; const char * pool_exact;
        const char * slot_channels; const char * verify_uid; const char * plan_stats;
        const char * graph_uid; const char * plan_flush; const char * expected;
    };
    const probe probes[] = {
        {"2", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "GGML_TP5_PLAN_UNION"},
        {nullptr, "9", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "GGML_TP5_RELAY_FRAMES"},
        {nullptr, "1", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "GGML_TP5_RELAY_FRAMES"},
        {nullptr, "x", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "GGML_TP5_RELAY_FRAMES"},
        {nullptr, nullptr, "yes", nullptr, nullptr, nullptr, nullptr, nullptr, "GGML_TP5_POOL_EXACT"},
        {nullptr, nullptr, nullptr, "2", nullptr, nullptr, nullptr, nullptr, "GGML_TP5_RELAY_SLOT_CHANNELS"},
        {nullptr, nullptr, nullptr, nullptr, "-1", nullptr, nullptr, nullptr, "GGML_TP5_RELAY_VERIFY_UID"},
        {nullptr, nullptr, nullptr, nullptr, nullptr, "true", nullptr, nullptr, "GGML_TP5_RELAY_PLAN_STATS"},
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "on", nullptr, "GGML_TP5_GRAPH_UID"},
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "yes", "GGML_TP5_RELAY_PLAN_FLUSH"},
    };
    for (const auto & p : probes) {
        cfg = config{};
        error[0] = '\0';
        const bool ok = parse_config(p.plan_union, p.frames, p.pool_exact, p.slot_channels,
                                     p.verify_uid, p.plan_stats, p.graph_uid, p.plan_flush,
                                     cfg, error, sizeof(error));
        check(!ok, "the invalid value is rejected");
        check(std::strstr(error, p.expected) != nullptr, "the message names the variable");
    }
}

void frame_bounds_are_enforced() {
    config cfg;
    char error[128];
    check(parse_config(nullptr, "8", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       cfg, error, sizeof(error)) && cfg.frames == 8, "8 frames is the maximum");
    check(parse_config(nullptr, "2", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       cfg, error, sizeof(error)) && cfg.frames == 2, "2 frames is the minimum");
    check(parse_config(nullptr, "0", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       cfg, error, sizeof(error)) && cfg.frames == 2, "0 means the default");
}

// ---------------------------------------------------------------------------
// per-window channel ownership
// ---------------------------------------------------------------------------

struct channel_state {
    std::uint64_t token = 0;
};

using slots = channel_slots<channel_state>;

// Retirement destroys the channel states, so it must require proof that no
// reader still holds one. This is the contract the fork left to the caller's
// good behaviour.
void channel_retire_requires_idleness() {
    slots table;
    auto * bound = table.acquire(11, /*epoch*/ 1, [](const slots::slot &) { return true; });
    check(bound != nullptr, "a generation binds to an empty slot");
    bound->states[0] = std::make_unique<channel_state>(channel_state{42});

    // A live reader: retirement is refused and nothing is destroyed.
    check(!table.retire(11, [](const slots::slot & s) { return s.states[0] == nullptr; }),
          "retirement is refused while a reader holds a channel");
    check(table.find(11) != nullptr && table.live_channels() == 1, "the slot and its channel survive");

    bound->states[0].reset();
    check(table.retire(11, [](const slots::slot & s) { return s.states[0] == nullptr; }),
          "retirement proceeds once the reader is gone");
    check(table.find(11) == nullptr, "the generation is gone");
    check(table.empty(), "the table is empty");
}

// Two windows must not share state even when they use the same generation
// number, and a full table must refuse to bind rather than drop a live reader.
void channel_slots_prefer_empty_then_idle() {
    slots table;
    const auto idle_always = [](const slots::slot &) { return true; };
    const auto idle_never  = [](const slots::slot &) { return false; };

    for (std::uint64_t generation = 1; generation <= slots::slot_capacity; ++generation) {
        check(table.acquire(generation, generation, idle_always) != nullptr,
              "each generation up to capacity binds");
    }
    check(table.live() == slots::slot_capacity, "the table is full");
    check(table.acquire(999, 999, idle_never) == nullptr, "no idle slot means no binding");

    // Touch generation 1 so the LRU victim becomes generation 2, then check both
    // the reuse rule and that eviction picks by last use rather than by index.
    auto * existing = table.acquire(1, 100, idle_never);
    check(existing != nullptr && existing->last_used == 100, "an existing binding is reused and touched");

    check(table.acquire(999, 999, idle_always) != nullptr, "an idle slot is reused");
    check(table.find(1) != nullptr, "the touched generation survived the eviction");
    check(table.find(2) == nullptr, "the least recently used generation was the victim");
    check(table.acquire(0, 0, idle_always) == nullptr, "generation 0 never binds");
}

// The arithmetic helpers must refuse everything outside their domain, including
// the values that only appear when a caller is already confused. Each assertion
// here is a read that would otherwise run past the end of a 256-entry array, or
// an overflow that turns a stride into zero.
void pool_bounds_are_total() {
    const std::size_t huge = static_cast<std::size_t>(-1);
    const std::size_t one[1] = {64};

    // A hand-written layout is legal input (the fields are public), so the
    // coverage test has to distrust its count as much as the elements it is asked
    // about.
    pool_layout layout;
    layout.count = channel_capacity + 1;
    check(!layout_covers(layout, one, 1), "a layout claiming more channels than the array is refused");

    layout.count = 1;
    layout.capacity[0] = 64;
    check(layout_covers(layout, one, 1), "a covering layout is accepted");
    check(!layout_covers(layout, one, 2), "count beyond the layout is refused, not read");
    check(!layout_covers(layout, one, channel_capacity + 1), "count beyond the array is refused");
    check(!layout_covers(layout, one, 0), "an empty request is refused");
    check(!layout_covers(layout, nullptr, 1), "a null element list is refused");

    // Overflow on every axis: too many channels, too many ranks, one enormous
    // element count. None of them may yield a number a caller could allocate.
    std::size_t need = 12345;
    check(!pool_need(one, channel_capacity + 1, 5, need) && need == 0,
          "more channels than the pool has is refused and need is cleared");
    check(!pool_need(one, 1, huge, need) && need == 0,
          "a rank count whose stride would overflow is refused, not wrapped to zero");
    check(!pool_need(one, 1, 0, need) && need == 0, "zero ranks is refused");
    const std::size_t enormous[1] = {huge};
    check(!pool_need(enormous, 1, 5, need) && need == 0, "an element count that overflows is refused");
    check(pool_need(one, 1, 5, need) && need > control_bytes, "a sane request is still answered");
}

// Two windows own two stations. Sharing nothing is the property that makes the
// missing locks acceptable, so it is worth a test of its own: the same generation
// number in two tables must not be the same state.
void two_windows_share_nothing() {
    using slots = channel_slots<int>;
    slots first;
    slots second;
    const auto idle_always = [](const slots::slot &) { return true; };

    auto * a = first.acquire(7, 7, idle_always);
    auto * b = second.acquire(7, 7, idle_always);
    check(a != nullptr && b != nullptr, "the same generation binds in both windows");
    a->states[0] = std::unique_ptr<int>(new int(1));
    b->states[0] = std::unique_ptr<int>(new int(2));

    check(first.retire(7, idle_always), "the first window retires its slot");
    check(second.find(7) != nullptr, "the second window keeps its own slot");
    check(second.find(7)->states[0] && *second.find(7)->states[0] == 2,
          "the second window's channel state is untouched");

    plan_cache one;
    plan_cache two;
    const auto h1 = one.acquire(key_of(8), 1);
    const auto h2 = two.acquire(key_of(8), 1);
    check(one.live() == 1 && two.live() == 1, "each cache holds its own plan");
    check(one.flush(h1), "the first cache flushes its plan");
    check(two.at(h2) != nullptr, "the second cache is unaffected by the first");
}

} // namespace

int main() {
    channels_are_part_of_identity();
    binding_digest_separates_same_shape();
    lru_evicts_the_oldest_unpinned();
    pinned_plans_are_never_evicted();
    stale_handles_are_refused_not_recycled();
    pins_are_counted();
    flush_all_is_atomic();
    frames_need_all_three_fields();
    frame_recording_is_idempotent_and_retirable();
    frames_evict_the_oldest();
    pool_layout_is_exact_and_aligned();
    pool_refuses_what_it_cannot_hold();
    pool_growth_rulings();
    config_defaults_match_the_legacy_lane();
    slot_channels_follow_the_union_switch();
    bad_values_are_rejected_by_name();
    frame_bounds_are_enforced();
    channel_retire_requires_idleness();
    channel_slots_prefer_empty_then_idle();
    pool_bounds_are_total();
    two_windows_share_nothing();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "relay plan contract holds\n");
    return 0;
}
