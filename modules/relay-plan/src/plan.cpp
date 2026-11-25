#include <tq/relay/plan.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace tq::relay {

// ---------------------------------------------------------------------------
// plan_cache
// ---------------------------------------------------------------------------

plan_handle plan_cache::find(const plan_key & key) const noexcept {
    for (const auto & slot : slots_) {
        if (slot.valid && slot.key == key) {
            return {slot.index, slot.generation};
        }
    }
    return {};
}

plan_slot * plan_cache::mutable_find(const plan_key & key) noexcept {
    for (auto & slot : slots_) {
        if (slot.valid && slot.key == key) {
            return &slot;
        }
    }
    return nullptr;
}

plan_handle plan_cache::acquire(const plan_key & key, std::uint64_t epoch,
                                std::uint64_t * evicted) noexcept {
    if (evicted != nullptr) {
        *evicted = 0;
    }
    if (plan_slot * hit = mutable_find(key)) {
        ++stats_.hits;
        hit->last_used = epoch;
        return {hit->index, hit->generation};
    }

    ++stats_.misses;
    const int index = pick_victim();
    if (index < 0) {
        // Every slot is pinned. Refusing is the only answer that does not destroy
        // a plan somebody is executing; the caller sees an invalid handle.
        ++stats_.pinned_refusals;
        return {};
    }

    plan_slot & slot = slots_[static_cast<std::size_t>(index)];
    if (slot.valid) {
        ++stats_.evictions;
        ++stats_.stale_generation;
        if (evicted != nullptr) {
            *evicted = slot.generation;
        }
    }
    slot.valid      = true;
    slot.key        = key;
    slot.generation = ++next_generation_;
    slot.last_used  = epoch;
    slot.pins       = 0;
    slot.index      = static_cast<std::uint32_t>(index);
    ++stats_.inserts;
    return {slot.index, slot.generation};
}

plan_slot * plan_cache::slot_of(plan_handle handle) noexcept {
    if (!handle.valid() || handle.index >= capacity) {
        return nullptr;
    }
    plan_slot & slot = slots_[handle.index];
    // The generation check is the whole point of handing out handles: a stamp
    // from a previous occupant must not address the new one.
    return (slot.valid && slot.generation == handle.generation) ? &slot : nullptr;
}

const plan_slot * plan_cache::at(plan_handle handle) const noexcept {
    return const_cast<plan_cache *>(this)->slot_of(handle);
}

bool plan_cache::pin(plan_handle handle) noexcept {
    plan_slot * slot = slot_of(handle);
    if (slot == nullptr) {
        return false;
    }
    if (slot->pins == std::numeric_limits<std::uint32_t>::max()) {
        // Saturation instead of wrap: a wrapped count would let an eviction
        // through while a plan is still executing.
        return false;
    }
    ++slot->pins;
    return true;
}

bool plan_cache::release(plan_handle handle) noexcept {
    plan_slot * slot = slot_of(handle);
    if (slot == nullptr || slot->pins == 0) {
        return false;
    }
    --slot->pins;
    return true;
}

bool plan_cache::flush(plan_handle handle) noexcept {
    plan_slot * slot = slot_of(handle);
    if (slot == nullptr) {
        return false;
    }
    if (slot->pins != 0) {
        ++stats_.pinned_refusals;
        return false;
    }
    *slot = {};
    ++stats_.flushes;
    return true;
}

bool plan_cache::flush_all() noexcept {
    for (const auto & slot : slots_) {
        if (slot.valid && slot.pins != 0) {
            ++stats_.pinned_refusals;
            return false;
        }
    }
    for (auto & slot : slots_) {
        if (slot.valid) {
            ++stats_.flushes;
        }
        slot = {};
    }
    return true;
}

std::size_t plan_cache::live() const noexcept {
    std::size_t count = 0;
    for (const auto & slot : slots_) {
        count += slot.valid ? 1 : 0;
    }
    return count;
}

std::size_t plan_cache::pinned() const noexcept {
    std::size_t count = 0;
    for (const auto & slot : slots_) {
        count += (slot.valid && slot.pins != 0) ? 1 : 0;
    }
    return count;
}

// Empty slot first; otherwise the oldest unpinned slot. Ties break on the lowest
// index, so eviction order is reproducible and a test can assert it.
int plan_cache::pick_victim() const noexcept {
    int best = -1;
    for (std::size_t i = 0; i < capacity; ++i) {
        const plan_slot & slot = slots_[i];
        if (!slot.valid) {
            return static_cast<int>(i);
        }
        if (slot.pins != 0) {
            continue;
        }
        if (best < 0 || slot.last_used < slots_[static_cast<std::size_t>(best)].last_used) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// frame_lru
// ---------------------------------------------------------------------------

frame_lru::frame_lru(std::size_t capacity) noexcept
    : capacity_(capacity == 0 ? 1 : (capacity > frame_capacity_max ? frame_capacity_max : capacity)) {}

const frame_lru::entry * frame_lru::find(std::uint64_t uid, std::uint64_t generation,
                                         std::uint64_t signature) const noexcept {
    for (const auto & entry : entries_) {
        if (entry.valid && entry.uid == uid && entry.generation == generation &&
            entry.signature == signature) {
            return &entry;
        }
    }
    return nullptr;
}

void frame_lru::record(std::uint64_t uid, std::uint64_t generation, std::uint64_t signature,
                       std::uint64_t epoch) noexcept {
    // Re-recording the same frame refreshes it rather than consuming a second
    // entry: two entries for one frame would make eviction order meaningless.
    for (auto & entry : entries_) {
        if (entry.valid && entry.uid == uid && entry.generation == generation &&
            entry.signature == signature) {
            entry.last_used = epoch;
            return;
        }
    }
    // Bounded by capacity_, not by the array: entries past capacity_ are outside
    // this table's policy, and writing one would mean a frame that no scan can
    // retire.
    for (std::size_t i = 0; i < capacity_; ++i) {
        if (!entries_[i].valid) {
            entries_[i] = {true, uid, generation, signature, epoch};
            return;
        }
    }
    // Full: replace the entry with the oldest stamp.
    //
    // Deliberately NOT the same rule as plan_cache::acquire, which refreshes the
    // stamp of a hit: `find()` here is a pure lookup and cannot refresh, because
    // the caller's epoch is what orders the stamps and find does not receive it.
    // The consequence is exact and worth knowing: a frame that is found on every
    // ubatch but never re-recorded keeps its old stamp and becomes the first
    // victim -- the hottest entry is evicted. The reuse path must therefore call
    // record() again on a hit (it is idempotent by construction: re-recording a
    // matching frame only refreshes it). Making find() refresh would need it to
    // take the epoch, and that is a signature change no caller needs yet.
    std::size_t victim = 0;
    for (std::size_t i = 1; i < capacity_; ++i) {
        if (entries_[i].last_used < entries_[victim].last_used) {
            victim = i;
        }
    }
    entries_[victim] = {true, uid, generation, signature, epoch};
}

std::size_t frame_lru::retire_generation(std::uint64_t generation) noexcept {
    std::size_t removed = 0;
    for (auto & entry : entries_) {
        if (entry.valid && entry.generation == generation) {
            entry = {};
            ++removed;
        }
    }
    return removed;
}

void frame_lru::clear() noexcept {
    for (auto & entry : entries_) {
        entry = {};
    }
}

std::size_t frame_lru::live() const noexcept {
    std::size_t count = 0;
    for (const auto & entry : entries_) {
        count += entry.valid ? 1 : 0;
    }
    return count;
}

// ---------------------------------------------------------------------------
// pool arithmetic
// ---------------------------------------------------------------------------

void pool_layout::clear() noexcept {
    offset.fill(0);
    capacity.fill(0);
    bytes = 0;
    count = 0;
}

bool pool_slot_bytes(std::size_t elements, std::size_t stride, std::size_t & bytes) noexcept {
    bytes = 0;
    if (stride == 0 || elements == 0) {
        return false;
    }
    if (elements > (std::numeric_limits<std::size_t>::max() - control_bytes) / stride) {
        return false;
    }
    const std::size_t raw = control_bytes + elements * stride;
    if (raw > std::numeric_limits<std::size_t>::max() - (pool_alignment - 1)) {
        return false;
    }
    bytes = (raw + pool_alignment - 1) & ~(pool_alignment - 1);
    return true;
}

bool pool_need(const std::size_t * elements, std::size_t count, std::size_t ranks,
               std::size_t & need) noexcept {
    need = 0;
    // The guard must be written so that it cannot overflow while guarding an
    // overflow: `ranks + 1 > MAX/4` wraps to 0 when ranks is SIZE_MAX and then
    // computes a zero stride, so it failed in exactly the case it exists for.
    // Comparing `ranks` against the same bound directly has no such hole.
    if (elements == nullptr || count == 0 || count > channel_capacity || ranks == 0 ||
        ranks >= std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        return false;
    }
    const std::size_t stride = sizeof(float) * (ranks + 1);
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t slot = 0;
        if (!pool_slot_bytes(elements[i], stride, slot)) {
            return false;
        }
        if (slot > std::numeric_limits<std::size_t>::max() - need) {
            return false;
        }
        need += slot;
    }
    return true;
}

bool compute_pool_layout(const std::size_t * elements, std::size_t count, std::size_t ranks,
                         std::size_t budget, pool_layout & out) noexcept {
    out.clear();
    std::size_t need = 0;
    if (!pool_need(elements, count, ranks, need) || need > budget) {
        return false;
    }
    const std::size_t stride = sizeof(float) * (ranks + 1);
    std::size_t       offset = 0;
    for (std::size_t i = 0; i < count; ++i) {
        std::size_t slot = 0;
        if (!pool_slot_bytes(elements[i], stride, slot)) {
            out.clear();
            return false;
        }
        out.offset[i]   = offset;
        out.capacity[i] = elements[i];
        offset += slot;
    }
    out.bytes = offset;
    out.count = count;
    return true;
}

bool layout_covers(const pool_layout & layout, const std::size_t * elements,
                   std::size_t count) noexcept {
    // The bound on count is part of this function's job, not the caller's. Its
    // two siblings reject count > channel_capacity before touching an array, and
    // a layout whose count was written by hand (the struct's fields are public)
    // would otherwise let the loop below index past capacity[] -- a read past the
    // end of a 256-entry array, in the one function whose whole purpose is
    // arithmetic that cannot run off the end.
    if (elements == nullptr || count == 0 || count > channel_capacity ||
        layout.count < count || layout.count > channel_capacity) {
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (elements[i] > layout.capacity[i]) {
            return false;
        }
    }
    return true;
}

pool_growth pool_growth_decision(const pool_layout & layout, const std::size_t * elements,
                                 std::size_t count, std::size_t ranks, std::size_t budget,
                                 std::size_t & need) noexcept {
    if (layout_covers(layout, elements, count)) {
        need = layout.bytes;
        return pool_growth::keep;
    }
    if (!pool_need(elements, count, ranks, need)) {
        return pool_growth::fail_budget;
    }
    return need > budget ? pool_growth::fail_budget : pool_growth::grow;
}

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

bool parse_switch(const char * value, bool & out) noexcept {
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
        out = false;
        return true;
    }
    if (std::strcmp(value, "1") == 0) {
        out = true;
        return true;
    }
    return false;
}

bool slot_channels_inherited(const char * slot_channels) noexcept {
    return slot_channels == nullptr || slot_channels[0] == '\0';
}

bool parse_config(const char * plan_union, const char * frames, const char * pool_exact,
                  const char * slot_channels, const char * verify_uid,
                  const char * plan_stats, const char * graph_uid,
                  const char * plan_flush, config & out,
                  char * error, std::size_t error_size) noexcept {
    if (error != nullptr && error_size != 0) {
        error[0] = '\0';
    }
    const auto reject = [&](const char * message) {
        if (error != nullptr && error_size != 0) {
            std::snprintf(error, error_size, "%s", message);
        }
        return false;
    };

    if (!parse_switch(plan_union, out.plan_union)) {
        return reject("GGML_TP5_PLAN_UNION must be 0 or 1");
    }
    if (frames == nullptr || frames[0] == '\0' || std::strcmp(frames, "0") == 0) {
        out.frames = 2;
    } else {
        char * end = nullptr;
        const long value = std::strtol(frames, &end, 10);
        if (end == frames || *end != '\0' || value < 2 ||
            value > static_cast<long>(frame_capacity_max)) {
            return reject("GGML_TP5_RELAY_FRAMES must be 0 or 2..8");
        }
        out.frames = static_cast<std::size_t>(value);
    }
    if (!parse_switch(pool_exact, out.pool_exact)) {
        return reject("GGML_TP5_POOL_EXACT must be 0 or 1");
    }
    if (slot_channels_inherited(slot_channels)) {
        // Unset means "follow the lane switch", which is what the fork did and
        // what an operator who set only GGML_TP5_PLAN_UNION expects.
        out.slot_channels = out.plan_union;
    } else if (!parse_switch(slot_channels, out.slot_channels)) {
        return reject("GGML_TP5_RELAY_SLOT_CHANNELS must be 0 or 1");
    }
    if (!parse_switch(verify_uid, out.verify_uid)) {
        return reject("GGML_TP5_RELAY_VERIFY_UID must be 0 or 1");
    }
    if (!parse_switch(plan_stats, out.plan_stats)) {
        return reject("GGML_TP5_RELAY_PLAN_STATS must be 0 or 1");
    }
    if (graph_uid == nullptr || graph_uid[0] == '\0') {
        out.graph_uid = false; // OFF does not force uid 0: sched/meta still assign uids
    } else if (!parse_switch(graph_uid, out.graph_uid)) {
        return reject("GGML_TP5_GRAPH_UID must be 0 or 1");
    }
    if (!parse_switch(plan_flush, out.plan_flush)) {
        return reject("GGML_TP5_RELAY_PLAN_FLUSH must be 0 or 1");
    }
    return true;
}

} // namespace tq::relay
