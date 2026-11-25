// herd/phase_station.hpp — the station that owns "which scheduler runs this
// context".
//
// Written to the ticket/station method in AGENTS.md, with the one adaptation it
// authorises for a multithreaded host: a plain lock around the state change, held
// for a few integer operations and never across a call into upstream.
//
// What makes this a station and not a state bag:
//
//   * It owns the invariant that matters — one scheduler per context, and an
//     entry that cannot outlive its context. Every method either preserves that
//     invariant or reports that it could not.
//   * It refuses to store a binding it could not later answer for: a null
//     scheduler is not recorded, and a full table reports refused_full instead of
//     overwriting somebody's entry.
//   * It never waits. There is no blocking call, no thread, no ticket, and no
//     table of in-flight work here; the waiting (a decode) belongs to the caller,
//     which is why the class is pure state plus a query.
//
// No gemm/llama header is included. Both handles stay opaque, so the accounting
// arithmetic that needs ggml lives in the wiring and this unit is testable with
// fabricated pointers.
#pragma once

#include <cstdint>
#include <mutex>
#include <utility>

struct llama_context;

// Identical to ggml-backend.h's typedef; repeated here so this header can be
// included without ggml.
typedef struct ggml_backend_sched * ggml_backend_sched_t;

namespace herd::phase {

class scheduler_station {
public:
    // Outcome of reporting a decode. The caller decides what to print; the
    // station does not log, because a station that writes to stderr from inside a
    // hot upstream call is a defect, not a diagnostic.
    enum class observation {
        recorded,      // first binding for this context
        updated,       // same context, same or a new scheduler
        refused_null,  // nothing to bind; the caller passed no scheduler
        refused_full,  // capacity reached; the existing bindings were kept
    };

    static constexpr int capacity = 64;

    // Event entry: a decode for `ctx` is executing on `sched`. Called from the
    // seam, which is already inside the call, so this is a synchronous prefix and
    // never a dispatch point.
    observation observe(const llama_context * ctx, ggml_backend_sched_t sched);

    // Event entry: `ctx` is being destroyed. Returns false when it was never
    // bound, which is normal for a context that never decoded.
    bool forget(const llama_context * ctx);

    // Event entry: a scheduler is being destroyed. Returns how many bindings were
    // removed.
    //
    // Needed because a scheduler does not only die with its context: upstream
    // rebuilds one inside llama_context::decode() on some paths
    // (`sched.reset(); sched_reserve();` at the compact-inkling and stream-wide
    // prompt branches). A binding keyed on the context alone would keep a pointer
    // to the destroyed object until the next decode happened to record a new one,
    // and the query in between would read freed memory. Dropping by identity makes
    // that window report "unknown", which is honest, instead of a number read from
    // a dead object.
    int forget_scheduler(ggml_backend_sched_t sched);

    // The query the accounting needs. Runs `fn` with the scheduler while the
    // binding is pinned by the lock: releasing the lock between "found" and "read
    // the scheduler's buffers" is the use-after-free that a concurrent llama_free
    // would win. Returns false when this context has no binding, which must be
    // reported as unknown rather than as zero bytes.
    //
    // Lock rule, and it is the whole safety argument for holding a lock while
    // calling out: this mutex is always the innermost one. `fn` runs under it, so
    // `fn` must not take any other lock and must not call back into this station
    // (std::mutex is not recursive). Today the only caller is the byte count,
    // which reads four lock-free accessors on the scheduler; upstream's
    // ggml_backend_sched_get_* take no lock and cannot reenter. A future caller
    // that needs to allocate, log to a sink, or decode has to be moved outside:
    // copy the handle out, then act.
    template <class Fn>
    bool with_scheduler(const llama_context * ctx, Fn && fn) const {
        if (!ctx) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        for (int i = 0; i < count_; ++i) {
            if (entries_[i].ctx == ctx) {
                std::forward<Fn>(fn)(entries_[i].sched);
                return true;
            }
        }
        return false;
    }

    // Diagnostics, not state exposed for a caller to re-derive a decision.
    int tracked() const;

    scheduler_station() = default;
    scheduler_station(const scheduler_station &) = delete;
    scheduler_station & operator=(const scheduler_station &) = delete;

private:
    struct entry {
        const llama_context * ctx;
        ggml_backend_sched_t  sched;
    };

    mutable std::mutex mutex_;
    entry              entries_[capacity]{};
    int                count_ = 0;
};

// The process-wide station. A function-local static, so its construction is
// ordered by first use rather than by translation-unit order: no seam can run
// before it exists, and no static initializer elsewhere can observe it half-made.
scheduler_station & station();

} // namespace herd::phase
