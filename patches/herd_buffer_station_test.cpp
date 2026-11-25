// The allocator station's bookkeeping, tested with fabricated pointers.
//
// Each case names the failure it prevents. The dangerous ones are address reuse
// (a new scheduler inheriting a freed allocator) and double ownership (two
// allocators for one scheduler, so one is never freed).
#include "herd_buffer_station.hpp"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using herd::buffers::outcome;
using herd::buffers::station;

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void * h(std::uintptr_t v) { return reinterpret_cast<void *>(v); }

// A handle with no owner must report none: that is the signal to allocate, and a
// non-null answer here would make the caller reuse a handle that never allocates.
void unknown_and_ownerless_handles_report_none() {
    station s;
    check(s.owner_of(h(0x1000)) == nullptr, "unknown handle has no owner");
    check(s.bind(h(0x1000), nullptr, 1) == outcome::bound, "a handle binds without an owner");
    check(s.owner_of(h(0x1000)) == nullptr, "bound-but-ownerless reports none");
    check(s.tracked() == 1, "it is tracked");
    check(s.live_owners() == 0, "no live allocator yet");
}

// Binding with an allocator already in hand (the common path) must expose it.
void bind_with_owner_is_immediately_usable() {
    station s;
    check(s.bind(h(0x2000), h(0x3000), 2) == outcome::bound, "binds with an owner");
    check(s.owner_of(h(0x2000)) == h(0x3000), "the owner is returned");
    check(s.live_owners() == 1, "one live allocator");
    check(s.bind(h(0x2000), nullptr, 2) == outcome::rebound, "re-binding reports rebound");
    check(s.owner_of(h(0x2000)) == h(0x3000), "re-binding keeps the owner");
}

// adopt() is the post-release path. It must refuse a second owner: two allocators
// for one scheduler means the first one's memory is never returned.
void adopt_refuses_a_second_owner() {
    station s;
    check(s.bind(h(0x4000), nullptr, 1) == outcome::bound, "ownerless handle binds");
    check(s.adopt(h(0x4000), h(0x5000)), "the first adopt succeeds");
    check(!s.adopt(h(0x4000), h(0x6000)), "the second adopt is refused");
    check(s.owner_of(h(0x4000)) == h(0x5000), "the first owner survives the refusal");
    check(!s.adopt(h(0x9999), h(0x7000)), "adopting an unknown handle is refused");
    check(!s.adopt(h(0x4000), nullptr), "adopting a null owner is refused");
}

// The release contract: the owner is handed back exactly once, and afterwards the
// handle reports none so the next use allocates a fresh one.
void release_hands_back_the_owner_once() {
    station s;
    check(s.bind(h(0x8000), h(0x8100), 1) == outcome::bound, "binds with an owner");
    check(s.release(h(0x8000)) == h(0x8100), "release returns the owner to free");
    check(s.owner_of(h(0x8000)) == nullptr, "a released handle reports no owner");

    // A second release must not hand back the same pointer: the caller has already
    // freed it, and freeing twice is a heap corruption that no test would catch
    // afterwards.
    check(s.release(h(0x8000)) == nullptr, "a second release returns nothing");
    check(s.release(h(0xDEAD)) == nullptr, "releasing an unknown handle returns nothing");

    check(s.adopt(h(0x8000), h(0x8200)), "a released handle adopts a fresh owner");
    check(s.owner_of(h(0x8000)) == h(0x8200), "the fresh owner is returned");
    check(s.tracked() == 1, "re-adopting does not duplicate the handle");
}

// The failure the station exists to prevent: a scheduler destroyed, its allocator
// freed, then a new scheduler at the same address. The new one must start empty.
void address_reuse_does_not_inherit_an_allocator() {
    station s;
    check(s.bind(h(0x9000), h(0x9100), 1) == outcome::bound, "first scheduler binds");
    check(s.retire(h(0x9000)) == h(0x9100), "retire returns the allocator to free");
    check(s.tracked() == 0, "the entry is gone");
    check(s.owner_of(h(0x9000)) == nullptr, "the retired handle answers nothing");

    // Same address, new scheduler.
    check(s.bind(h(0x9000), nullptr, 1) == outcome::bound, "the reused address binds fresh");
    check(s.owner_of(h(0x9000)) == nullptr, "it inherits no allocator");
    check(s.live_owners() == 0, "no live allocator after reuse");
}

// Capacity is a boundary, not a policy: the station must not evict a scheduler that
// still holds memory, because that entry's allocator would then never be freed.
void capacity_refuses_rather_than_evicts() {
    station s;
    for (int i = 0; i < station::capacity; ++i) {
        check(s.bind(h(0x10000 + std::uintptr_t(i) * 8), h(0x20000 + std::uintptr_t(i) * 8), 1) ==
                  outcome::bound,
              "every slot up to capacity binds");
    }
    check(s.tracked() == station::capacity, "the station is full");
    check(s.bind(h(0x99000), h(0x9A000), 1) == outcome::refused_full, "the next one is refused");
    check(s.owner_of(h(0x10000)) == h(0x20000), "the oldest entry still answers");
    check(s.tracked() == station::capacity, "a refusal changes nothing");
}

// A null handle is not a scheduler. Treating it as one would put an entry in the
// table that no retire can ever remove.
void null_handles_are_refused() {
    station s;
    check(s.bind(nullptr, h(0x100), 1) == outcome::refused_null, "null handle is refused");
    check(s.owner_of(nullptr) == nullptr, "null has no owner");
    check(s.release(nullptr) == nullptr, "null releases nothing");
    check(s.retire(nullptr) == nullptr, "null retires nothing");
    check(s.tracked() == 0, "nothing was recorded");
}

// Each handle's owner must be handed back exactly once, no matter how many threads
// race to release, retire or re-adopt it. Two returns of one pointer would be a
// double free that only shows up as heap corruption much later; zero returns would
// be a leak. Both are counted here rather than argued about.
//
// A handle is touched by one thread at a time by construction upstream (a scheduler
// belongs to the context that owns it), so the churn below is deliberately per
// thread; what it proves is that the station's own bookkeeping stays exact under
// contention, which is what makes the fail-loud paths in the hooks reachable only
// for genuine misuse rather than for scheduling.
void concurrent_release_and_retire_hand_the_owner_back_once() {
    constexpr int workers = 8;
    constexpr int rounds  = 500;
    station s;

    std::atomic<int> owner_returns{0};
    std::atomic<int> double_returns{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (int w = 0; w < workers; ++w) {
        threads.emplace_back([&, w] {
            const auto base = std::uintptr_t(0x100000) + std::uintptr_t(w) * 0x1000;
            for (int round = 0; round < rounds; ++round) {
                void * handle = h(base + std::uintptr_t(round % 8) * 8);
                void * owner  = h(base + 0x400 + std::uintptr_t(round % 8) * 8);
                // A fresh pairing each round: bind, learn the scheduler, release it.
                const auto bound = s.bind(handle, owner, 1);
                if (bound == outcome::refused_full) {
                    continue; // capacity is another test's business
                }
                s.link_scheduler(h(base + 0x800), handle);

                const int handed = (s.release_scheduler(h(base + 0x800)) == owner) ? 1 : 0;
                owner_returns += handed;
                // A second release of the same scheduler must return nothing.
                if (s.release_scheduler(h(base + 0x800)) != nullptr) {
                    ++double_returns;
                }
                // A repeated direct release of the handle must also return nothing.
                if (s.release(handle) != nullptr) {
                    ++double_returns;
                }
                // Retire finds no owner (it was released), and clears the entry.
                static_cast<void>(s.retire(handle));
            }
        });
    }
    for (auto & thread : threads) {
        thread.join();
    }

    check(double_returns.load() == 0, "no owner was handed back twice under contention");
    check(owner_returns.load() == workers * rounds,
          "every round handed back exactly one owner");
    check(s.tracked() == 0, "every entry was retired");
}

// A token is never its own owner.
//
// This is the case a dead guard used to stand in for. `release_sched` compared the
// owner against the *scheduler* -- two unrelated object types that can never be
// equal -- intending to protect the token from being freed as if it were the second
// allocator. The guard could not fire; worse, the state it was meant to exclude was
// also reachable, because adopt() accepted (h, h). Both halves are asserted here:
// the state is refused, and therefore the caller's unconditional free is correct.
void a_token_is_never_its_own_owner() {
    station s;
    const void * handle = h(0xA000);
    const void * sched  = h(0xB000);

    // bind() with the handle as its own buddy records the handle but no owner.
    check(s.bind(const_cast<void *>(handle), const_cast<void *>(handle), 1) == outcome::bound,
          "a self-paired handle still binds");
    check(s.owner_of(const_cast<void *>(handle)) == nullptr,
          "a handle passed as its own buddy is not recorded as an owner");
    check(s.live_owners() == 0, "nothing is live");

    // adopt() is where the pairing would normally be created, so it is where the
    // self-pairing must be refused.
    check(!s.adopt(const_cast<void *>(handle), const_cast<void *>(handle)),
          "a handle cannot adopt itself");
    check(s.owner_of(const_cast<void *>(handle)) == nullptr, "and it stays ownerless");

    // Consequence, stated as a test: a scheduler over an ownerless handle releases
    // nothing, so release_sched() never has a token to free.
    s.link_scheduler(const_cast<void *>(sched), const_cast<void *>(handle));
    check(s.release_scheduler(const_cast<void *>(sched)) == nullptr,
          "an ownerless handle hands nothing back for freeing");
}

} // namespace

int main() {
    unknown_and_ownerless_handles_report_none();
    bind_with_owner_is_immediately_usable();
    adopt_refuses_a_second_owner();
    release_hands_back_the_owner_once();
    address_reuse_does_not_inherit_an_allocator();
    capacity_refuses_rather_than_evicts();
    null_handles_are_refused();
    concurrent_release_and_retire_hand_the_owner_back_once();
    a_token_is_never_its_own_owner();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "buffer station contract holds\n");
    return 0;
}
