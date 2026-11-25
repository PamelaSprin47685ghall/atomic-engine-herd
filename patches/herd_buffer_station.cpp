#include "herd_buffer_station.hpp"

namespace herd::buffers {

slot * station::find_slot(void * handle) {
    for (int i = 0; i < count_; ++i) {
        if (entries_[i].handle == handle) {
            return &entries_[i];
        }
    }
    return nullptr;
}

const slot * station::find_slot(void * handle) const {
    return const_cast<station *>(this)->find_slot(handle);
}

outcome station::bind(void * handle, void * buddy, int n_bufts) {
    if (!handle) {
        return outcome::refused_null;
    }
    // The token is never its own owner. `handle` is the allocator upstream holds and
    // passes back to us; `buddy` is the second allocator that does the work. Handing
    // the handle as its own buddy would make "the allocator to free" mean the token,
    // and freeing that leaves upstream holding a dangling pointer. Refused at the
    // entry point rather than checked at every use: a guard at the use site has to
    // name the right object, and this file already had one that named the wrong one.
    if (buddy == handle) {
        buddy = nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (slot * existing = find_slot(handle)) {
        // Same address, still tracked. Whether it is the same allocator cannot be
        // known from an address, so the entry keeps its word: a retired handle must
        // be re-bound by its creator, and a live one keeps its owner. Silently
        // re-binding would be the address-reuse bug.
        if (buddy && existing->owner == nullptr) {
            existing->owner    = buddy;
            existing->live     = true;
            existing->released = false;
        }
        existing->n_bufts = n_bufts;
        return outcome::rebound;
    }
    if (count_ >= capacity) {
        return outcome::refused_full;
    }
    entries_[count_++] = slot{handle, buddy, n_bufts, buddy != nullptr, false};
    return outcome::bound;
}

void * station::owner_of(void * handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const slot * found = find_slot(handle);
    return (found && found->live) ? found->owner : nullptr;
}

bool station::adopt(void * handle, void * buddy) {
    if (!handle || !buddy) {
        return false;
    }
    // Same invariant as bind(), and here it is the whole point of the call: adopt()
    // exists to install a *second*, distinct allocator. A handle adopting itself is
    // not a pairing, and accepting it would make release() hand back the token for
    // the caller to free.
    if (buddy == handle) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    slot * found = find_slot(handle);
    if (!found) {
        return false;
    }
    // Only a handle with no owner may adopt one. A second adopt would mean two
    // allocators for one scheduler: the first one's buffers would then never be
    // freed, which is the leak this station exists to prevent.
    if (found->owner != nullptr) {
        return false;
    }
    found->owner    = buddy;
    found->live     = true;
    found->released = false;
    return true;
}

void * station::release(void * handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    slot * found = find_slot(handle);
    if (!found || found->owner == nullptr) {
        return nullptr;
    }
    void * owner = found->owner;
    found->owner    = nullptr;
    found->live     = false;
    found->released = true;
    return owner;
}

void * station::retire(void * handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < count_; ++i) {
        if (entries_[i].handle == handle) {
            void * owner = entries_[i].owner;
            entries_[i] = entries_[--count_];
            entries_[count_] = slot{};
            return owner;
        }
    }
    return nullptr;
}

bool station::was_released(void * handle) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const slot * found = find_slot(handle);
    return found != nullptr && found->released;
}

void station::link_scheduler(void * sched, void * handle) {
    if (!sched || !handle) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < sched_capacity; ++i) {
        if (links_[i].sched == sched) {
            links_[i].handle = handle;
            return;
        }
    }
    for (int i = 0; i < sched_capacity; ++i) {
        if (links_[i].sched == nullptr) {
            links_[i] = link{sched, handle};
            return;
        }
    }
    // Deliberately silent at this point: the caller reports the consequence once
    // (a release that answers false), and a library logging from inside an upstream
    // call would be a defect of its own.
}

void * station::release_scheduler(void * sched) {
    if (!sched) {
        return nullptr;
    }
    // One lock for the whole decision. The pairing lookup and the release must be
    // one step: see the header for the window that opens when they are two.
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < sched_capacity; ++i) {
        if (links_[i].sched != sched) {
            continue;
        }
        void * handle = links_[i].handle;
        slot * found = find_slot(handle);
        if (!found || found->owner == nullptr) {
            // Nothing to hand back. The pairing is left in place: a scheduler whose
            // allocator was already released (or never populated) may still adopt
            // one on its next reserve, and dropping the link here would make that
            // adoption the only way to learn the pairing again.
            return nullptr;
        }
        void * owner = found->owner;
        found->owner    = nullptr;
        found->live     = false;
        found->released = true;
        // The pairing goes with the allocator: a scheduler that will re-reserve has
        // no allocator to name until it does, and a stale link would answer for the
        // freed one.
        links_[i] = link{};
        return owner;
    }
    return nullptr;
}

void * station::handle_of_scheduler(void * sched) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < sched_capacity; ++i) {
        if (links_[i].sched == sched) {
            return links_[i].handle;
        }
    }
    return nullptr;
}

int station::unlink_scheduler(void * handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    int removed = 0;
    for (int i = 0; i < sched_capacity; ++i) {
        if (links_[i].handle == handle) {
            links_[i] = link{};
            ++removed;
        }
    }
    return removed;
}

int station::tracked() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

int station::live_owners() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int live = 0;
    for (int i = 0; i < count_; ++i) {
        live += entries_[i].live ? 1 : 0;
    }
    return live;
}

station & shared() {
    // Immortal, for the same reason the phase station is: a scheduler can be
    // destroyed during another translation unit's static destruction, and the
    // retire path runs then. A function-local static would already be gone.
    static station * instance = new station();
    return *instance;
}

} // namespace herd::buffers
