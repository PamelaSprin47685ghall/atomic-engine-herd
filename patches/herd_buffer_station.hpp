// herd/buffer_station.hpp — which allocator owns a scheduler's compute buffers.
//
// The problem this exists for: `--phase-memory` frees the compute buffers between
// prompt processing and generation to cap peak memory. The fork could do it because
// it owned `ggml_backend_sched::galloc` and could replace the field. This project
// does not own that struct, and an earlier conclusion stopped there ("private
// field, no setter, therefore impossible"). That conclusion was wrong, and the
// reason it was wrong is worth writing down: the field is only unreachable if you
// try to *write* it. Read the other direction instead -- upstream never touches
// the allocator's guts, it only calls six public entry points on it
// (ggml_gallocr_{new_n,free,reserve_n,reserve_n_size,alloc_graph,get_buffer_size},
// verified by grepping every `galloc` use in ggml-backend.cpp). Redirect those six
// and the allocator upstream holds becomes a handle that never allocates, while a
// second allocator, owned here, does the real work -- and can be freed at a phase
// boundary, which is exactly what "release" means.
//
// A second correction, because the first draft of this file repeated it: after a
// release, the next decode does NOT fail on a multi-backend setup. The claim was
// made from ggml_gallocr_alloc_graph() alone, which returns false on a shape
// mismatch when n_buffers > 1. Its caller, ggml_backend_sched_alloc_splits()
// (ggml-backend.cpp), treats a false return as "reserve and try once more":
//
//     if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
//         ... synchronise every backend ...
//         if (!ggml_gallocr_reserve_n(sched->galloc, ...)) return false;
//         if (!ggml_gallocr_alloc_graph(sched->galloc, ...)) return false;
//     }
//
// Both of those calls go through the redirection, so the retry reserves the fresh
// allocator and then allocates on it. The re-reservation the feature needs is
// therefore already reachable through a public path that upstream takes by itself;
// it needs no extra hook and no write to `sched_need_reserve`. What the release
// must guarantee is only that the retry finds a usable fresh allocator, which is
// what this station tracks.
//
// This header is the station: pure bookkeeping over opaque handles plus the
// scheduler -> allocator pairing it answers for. No ggml type, no allocation
// policy, no waiting. One mutex, held for a handful of pointer comparisons.
//
// One invariant carries most of the safety, and the station enforces it at the two
// entry points rather than checking it at each use:
//
//   a handle is never its own owner.
//
// `handle` is the allocator upstream holds (a token we created and returned); `owner`
// is the second allocator this project allocates with. If the two could coincide,
// "the allocator to free" would mean the token, and freeing it would leave upstream
// holding a dangling allocator -- corruption far from here. bind() drops a
// self-pairing and adopt() refuses it, so every pointer this station hands back for
// freeing is a distinct allocation. That is why the callers free unconditionally:
// there is no case to test for, and a test that names the wrong object reads as
// protection while providing none.
#pragma once

#include <cstdint>
#include <mutex>

// Declared at global scope: upstream's own type. Declaring it inside this
// namespace would create a distinct, unrelated herd::buffers::llama_context.
struct llama_context;

namespace herd::buffers {

// One scheduler-facing allocator. `handle` is the allocator upstream holds and
// passes back to us (opaque here); `owner` is the one that allocates for real.
struct slot {
    void * handle  = nullptr;
    void * owner   = nullptr;
    int    n_bufts = 0;
    bool   live    = false;  // false once released: the next use must allocate anew
    // Set by release(), cleared when a fresh owner is adopted. It is the answer to
    // "was this released, or merely never populated?" -- a distinction the retry
    // path does not need, but a diagnostic and a test do.
    bool   released = false;
};

enum class outcome {
    bound,        // first time this handle was seen
    rebound,      // same handle, still live
    refused_full, // capacity reached; nothing was recorded
    refused_null, // a null handle
};

class station {
public:
    static constexpr int capacity      = 32;
    static constexpr int sched_capacity = 32;

    // ---- allocator handles -------------------------------------------------

    // First sighting of an allocator upstream created. `buddy` is the allocator
    // this project will allocate with; it may be null until the first work call,
    // because a scheduler is often created before it is ever reserved.
    outcome bind(void * handle, void * buddy, int n_bufts);

    // The allocator to use right now. Null means "create one": either the handle
    // is unknown, or it was released. Never returns the handle itself, because the
    // handle does not allocate.
    //
    // The pointer is borrowed, not owned, and the borrow has a lifetime the station
    // cannot see: it ends when someone releases this handle on another thread. That
    // cannot happen for the same handle without two threads driving one scheduler,
    // which upstream forbids (a scheduler belongs to the context that owns it). Two
    // contexts means two handles, and the station's lock is what makes their
    // bookkeeping independent. A caller that wants a longer guarantee than "one
    // thread per scheduler" must add reference counting here; nothing in this
    // experiment does, and the release path is not wired into any product.
    void * owner_of(void * handle) const;

    // Records a freshly created buddy. Refused when the handle already has one:
    // two allocators for one scheduler means the first one's memory is never freed.
    bool adopt(void * handle, void * buddy);

    // Drops ownership: the caller receives the real allocator to free, and the
    // next use allocates a new one. Returns null when there was nothing to release,
    // so a second release cannot free the same pointer twice.
    void * release(void * handle);

    // The handle is going away. Returns the real allocator to free, or null. The
    // entry is removed either way: a later allocator that reuses the same address
    // must not inherit the pairing.
    void * retire(void * handle);

    // True between release() and the next adopt(): the caller's reservation is
    // stale and one full reserve is expected before the next successful allocate.
    bool was_released(void * handle) const;

    // ---- scheduler -> allocator pairing ------------------------------------

    // Records which allocator a scheduler holds. Learned from the correlation
    // available at reserve time (the scheduler whose call we are inside) rather
    // than from any struct field.
    void link_scheduler(void * sched, void * handle);

    // The allocator behind a scheduler, or null.
    void * handle_of_scheduler(void * sched) const;

    // Releases the allocator behind `sched` and drops the pairing, in ONE lock.
    //
    // Two separate calls are not equivalent, and the difference is a
    // use-after-free: `handle_of_scheduler(sched)` then `release(handle)` leaves a
    // window in which the scheduler can be destroyed (herd_free → retire, which
    // frees the entry) and its address reused by a *new* allocator. The second call
    // would then find a different allocator under the same address and free a live
    // one -- later, in the new scheduler, as corruption rather than as a crash.
    //
    // Returns the allocator the caller must free, or null when the scheduler is
    // unknown or was already released. On success the pairing is removed, so a
    // second release of the same scheduler reports null instead of freeing twice.
    void * release_scheduler(void * sched);

    // Removes pairings that named this allocator, so a scheduler address reused
    // after teardown cannot answer with somebody else's allocator.
    int unlink_scheduler(void * handle);

    // ---- diagnostics -------------------------------------------------------

    int tracked() const;
    int live_owners() const;

private:
    slot * find_slot(void * handle);
    const slot * find_slot(void * handle) const;

    struct link {
        void * sched  = nullptr;
        void * handle = nullptr;
    };

    mutable std::mutex mutex_;
    slot               entries_[capacity]{};
    int                count_ = 0;
    link               links_[sched_capacity]{};
};

// Process-wide station, constructed on first use and never destroyed: the release
// path can be reached from another translation unit's teardown, and a destroyed
// station would be a use-after-death at exit. See the note in the .cpp.
station & shared();

// Releases the compute buffers behind a scheduler, freeing the allocator that
// actually holds them. Returns false when the scheduler is unknown (for example one
// created before these hooks were installed) or was already released: false means
// nothing was freed, so a caller must report "not done", never "done".
//
// This is the raw step. It leaves the context holding graph-input tensors whose
// data pointers address the memory that was just freed, so a decode after it must
// not run until the scheduler is re-armed -- see release_and_arm().
bool release_sched(void * sched);

// Release, then arm the scheduler so the next decode rebuilds its graph before
// writing any input.
//
// Why arming is not optional: llama_context::decode calls sched_reserve() before
// its ubatch loop, and sched_reserve() returns immediately unless the scheduler's
// private `sched_need_reserve` flag is set. Nothing else in the decode path
// rebinds graph-input tensors, and those tensors' data pointers were fixed by the
// *previous* alloc_graph -- they point into the memory that was just freed. A
// decode in that state writes tokens into released memory: measured with ASan as
// heap-use-after-free in llm_graph_input_embd::set_input.
//
// `embeddings_currently` is the context's current embeddings flag. Upstream exposes
// no getter for it, and the flag is armed through llama_set_embeddings(), so a
// caller that does not know the value must not call this: guessing would silently
// change how the context runs. The driver that created the context always knows.
//
// PARITY AND ITS LIMITS, against the fork's llama_context::release_compute()
// (src/llama-context.cpp:2646-2688). The fork refuses in five situations and
// synchronizes before mutating; ours does what upstream lets it do, and says what it
// cannot:
//
//   fork: !sched                             ours: false, no binding was recorded
//   fork: synchronize() before any mutation  ours: llama_synchronize(ctx), now inside
//                                            release_and_arm -- the same guarantee, and
//                                            the guard this experiment was missing
//   fork: sched_has_unproven -> refuse       no upstream query exists; the synchronize
//                                            above establishes the same precondition and
//                                            is the stronger form (it waits rather than
//                                            refusing). Upstream has no `unproven`.
//   fork: model.hparams.no_alloc -> refuse   NOT CHECKABLE: `no_alloc` lives on
//                                            llama_model_params (include/llama.h:374) and
//                                            no public accessor reaches it from a context.
//                                            A caller that loaded with no_alloc must not
//                                            request a release: there are no buffers.
//   fork: opt_ctx / phase arena -> refuse    NO ANALOGUE: both are fork-only. If a driver
//                                            adds one, release must be disabled for it;
//                                            there is no upstream state to detect, so this
//                                            cannot be enforced from inside.
//
// So "release works" holds for the measured configuration (a plain decoder context,
// CPU, one buffer type). The three marked items are the conditions under which it
// would be false.
//
// Implemented in the reverse direction from the rest of this header: it takes a
// llama_context, so this declaration is the only place in the file that names one.
bool release_and_arm(llama_context * ctx, bool embeddings_currently);

} // namespace herd::buffers
