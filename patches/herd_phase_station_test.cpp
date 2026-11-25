// Station-level tests for herd::phase. No llama, no ggml, no Dobby, no model:
// every handle here is a fabricated pointer that is never dereferenced, so the
// only thing under test is the bookkeeping the seam depends on.
//
// Each case names the failure it would catch. A test that only asserts "no
// crash" would pass with an empty implementation.
#include "herd_phase_station.hpp"

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

using herd::phase::scheduler_station;

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

const llama_context * ctx_at(std::uintptr_t slot) {
    return reinterpret_cast<const llama_context *>(slot);
}

ggml_backend_sched_t sched_at(std::uintptr_t slot) {
    return reinterpret_cast<ggml_backend_sched_t>(slot);
}

// Reads the scheduler the station would hand to the accounting code.
ggml_backend_sched_t scheduler_seen_by(scheduler_station & s, const llama_context * ctx) {
    ggml_backend_sched_t seen = nullptr;
    const bool known = s.with_scheduler(ctx, [&](ggml_backend_sched_t value) { seen = value; });
    return known ? seen : nullptr;
}

// A context that never decoded has no scheduler. Reporting "known" here would
// make the driver print a byte count for a context that has never run.
void unknown_before_any_decode() {
    scheduler_station s;
    check(!s.with_scheduler(ctx_at(0x1000), [](ggml_backend_sched_t) {
        check(false, "callback must not run for an unknown context");
    }), "unknown context is reported unknown");
    check(s.tracked() == 0, "nothing is tracked before the first decode");
}

// The seam records the scheduler upstream chose. A later query must see it.
void records_then_updates() {
    scheduler_station s;
    const llama_context * ctx = ctx_at(0x2000);
    check(s.observe(ctx, sched_at(0x3000)) == scheduler_station::observation::recorded,
          "first sighting is recorded");
    check(scheduler_seen_by(s, ctx) == sched_at(0x3000), "the recorded scheduler is returned");
    check(s.tracked() == 1, "one context is tracked");

    check(s.observe(ctx, sched_at(0x4000)) == scheduler_station::observation::updated,
          "a second sighting updates rather than duplicates");
    check(scheduler_seen_by(s, ctx) == sched_at(0x4000), "the update is the value returned");
    check(s.tracked() == 1, "still one context after an update");
}

// The reason the free seam exists: a context's memory can be reused, and a binding
// that survived would answer for the wrong context. This is the regression test
// for the use-after-free the earlier review found.
void address_reuse_does_not_inherit() {
    scheduler_station s;
    const llama_context * reused = ctx_at(0x5000);
    check(s.observe(reused, sched_at(0x6000)) == scheduler_station::observation::recorded,
          "first context binds");
    check(s.forget(reused), "forget finds the entry");
    check(!s.with_scheduler(reused, [](ggml_backend_sched_t) {
        check(false, "a forgotten context must not answer");
    }), "forgotten context is unknown");

    // Same address, new context: it must be bound to its own scheduler, never to
    // the previous occupant's.
    check(s.observe(reused, sched_at(0x7000)) == scheduler_station::observation::recorded,
          "the reused address is recorded fresh, not updated");
    check(scheduler_seen_by(s, reused) == sched_at(0x7000),
          "the reused address answers for the new context");
    check(s.tracked() == 1, "no duplicate entry for one address");
}

void forget_of_unknown_is_refused() {
    scheduler_station s;
    check(!s.forget(ctx_at(0x8000)), "forgetting an unknown context reports false");
    check(s.tracked() == 0, "a refused forget changes nothing");
    check(!s.forget(nullptr), "a null context is refused");
}

// A null scheduler means "no evidence", not "zero bytes". Storing it would let the
// driver print a measurement it never took.
void null_scheduler_is_not_a_binding() {
    scheduler_station s;
    check(s.observe(ctx_at(0x9000), nullptr) == scheduler_station::observation::refused_null,
          "a null scheduler is refused");
    check(s.observe(nullptr, sched_at(0xA000)) == scheduler_station::observation::refused_null,
          "a null context is refused");
    check(s.tracked() == 0, "nothing was bound");
}

// Capacity is a hard boundary. When it is reached, existing bindings stay valid and
// the new one is reported: the alternative is evicting an entry whose context is
// still alive, which would make a live context's accounting unknown without notice.
void capacity_is_refused_not_evicted() {
    scheduler_station s;
    for (int i = 0; i < scheduler_station::capacity; ++i) {
        check(s.observe(ctx_at(0x10000 + std::uintptr_t(i) * 0x10),
                        sched_at(0x20000 + std::uintptr_t(i) * 0x10)) ==
                  scheduler_station::observation::recorded,
              "every slot up to capacity is recorded");
    }
    check(s.tracked() == scheduler_station::capacity, "the table is full");

    const llama_context * overflow = ctx_at(0x99000);
    check(s.observe(overflow, sched_at(0x9A000)) == scheduler_station::observation::refused_full,
          "the entry past capacity is refused");
    check(s.tracked() == scheduler_station::capacity, "a refusal does not change the table");
    check(scheduler_seen_by(s, ctx_at(0x10000)) == sched_at(0x20000),
          "the oldest binding still answers after a refusal");
}

// Two decoders on two threads must not corrupt the table, and both must be able to
// read while the other writes. A torn table here would surface as a wrong byte
// count on a live run, which is worse than a crash.
// A scheduler can also die on its own, mid-context: upstream's decode() rebuilds it
// in place on some paths. Every binding that pointed at it must go, and the query
// must then answer "unknown" rather than hand out a freed pointer. This is the
// regression test for that window -- it is reachable without any llama_free, so the
// context-keyed hook alone cannot close it.
void scheduler_death_drops_its_bindings() {
    scheduler_station s;
    const llama_context * first  = ctx_at(0xD000);
    const llama_context * second = ctx_at(0xE000);
    const llama_context * other  = ctx_at(0xF000);

    check(s.observe(first, sched_at(0x10000)) == scheduler_station::observation::recorded,
          "first context binds to the doomed scheduler");
    check(s.observe(second, sched_at(0x10000)) == scheduler_station::observation::recorded,
          "second context binds to the same scheduler");
    check(s.observe(other, sched_at(0x11000)) == scheduler_station::observation::recorded,
          "an unrelated context binds elsewhere");

    check(s.forget_scheduler(sched_at(0x10000)) == 2,
          "both bindings of the destroyed scheduler are dropped");
    check(!s.with_scheduler(first, [](ggml_backend_sched_t) {
        check(false, "a destroyed scheduler must never be handed out");
    }), "the query reports unknown instead of freed memory");
    check(scheduler_seen_by(s, other) == sched_at(0x11000),
          "a binding to another scheduler is untouched");
    check(s.forget_scheduler(sched_at(0x10000)) == 0, "forgetting a dead scheduler twice is harmless");
    check(s.forget_scheduler(nullptr) == 0, "a null scheduler removes nothing");
}

// Re-decoding after a rebuild must re-bind, not resurrect the old entry: the new
// scheduler is a different object and the query has to answer with it.
void rebind_after_scheduler_rebuild() {
    scheduler_station s;
    const llama_context * ctx = ctx_at(0x12000);
    check(s.observe(ctx, sched_at(0x13000)) == scheduler_station::observation::recorded, "binds before rebuild");
    check(s.forget_scheduler(sched_at(0x13000)) == 1, "the rebuilt-away scheduler drops its binding");
    check(!s.with_scheduler(ctx, [](ggml_backend_sched_t) {
        check(false, "the window after a rebuild must be unknown");
    }), "unknown during the rebuild window");
    check(s.observe(ctx, sched_at(0x14000)) == scheduler_station::observation::recorded,
          "the next decode records the new scheduler as a fresh binding");
    check(scheduler_seen_by(s, ctx) == sched_at(0x14000), "the query answers with the new scheduler");
    check(s.tracked() == 1, "rebinding does not duplicate the context");
}

void concurrent_observation_is_consistent() {
    scheduler_station s;
    constexpr int threads = 4;
    constexpr int rounds  = 2000;
    std::atomic<int> observed{0};

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            const llama_context * ctx = ctx_at(0xB000 + std::uintptr_t(t) * 0x100);
            const ggml_backend_sched_t sched = sched_at(0xC000 + std::uintptr_t(t) * 0x100);
            for (int i = 0; i < rounds; ++i) {
                if (s.observe(ctx, sched) == scheduler_station::observation::recorded) {
                    ++observed;
                }
                ggml_backend_sched_t seen = nullptr;
                s.with_scheduler(ctx, [&](ggml_backend_sched_t value) { seen = value; });
                if (seen == sched) {
                    ++observed;
                }
                s.forget(ctx);
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }

    check(s.tracked() == 0, "every worker's binding is gone after the final forget");
    check(observed.load() == threads * rounds * 2,
          "each round saw its own binding: no torn read, no lost write");
}

} // namespace

int main() {
    unknown_before_any_decode();
    records_then_updates();
    address_reuse_does_not_inherit();
    forget_of_unknown_is_refused();
    null_scheduler_is_not_a_binding();
    capacity_is_refused_not_evicted();
    scheduler_death_drops_its_bindings();
    rebind_after_scheduler_rebuild();
    concurrent_observation_is_consistent();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "phase station contract holds\n");
    return 0;
}
