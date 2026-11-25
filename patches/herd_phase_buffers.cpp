// herd/phase_buffers.cpp — compute-buffer release by allocator redirection.
//
// STATUS: an experiment with its own executable and its own measurement; linked
// into no product (CMake: herd_buffer_hooks, and the probe target). It is enabled
// in a product only after the probe shows what it claims to show.
//
// Upstream never reads the allocator's fields; it calls six public entry points on
// it. Verified by enumerating every `galloc` use in ggml-backend.cpp:
//
//   2558  ggml_gallocr_new_n            (created by ggml_backend_sched_new)
//   2130  ggml_gallocr_reserve_n        ┐
//   2131  ggml_gallocr_alloc_graph      ├ reserve_n
//   2152  ggml_gallocr_alloc_graph      ┐
//   2181  ggml_gallocr_reserve_n        ├ alloc_graph
//   2185  ggml_gallocr_alloc_graph      ┘
//   2621  ggml_gallocr_reserve_n_size   (reserve_size)
//   2632  ggml_gallocr_reserve_n        (reserve)
//   2746  ggml_gallocr_get_buffer_size  (byte reporting)
//   2576  ggml_gallocr_free             (sched free)
//
// That set is closed, so redirecting it makes the allocator upstream holds an inert
// token while a second allocator, owned here, does the work -- and freeing that one
// returns the memory, which is what `--phase-memory` always meant.
//
// Two facts that decide the design, both read rather than assumed:
//
//  1. ggml_backend_sched_alloc_graph() never reserves. It splits the graph and
//     calls ggml_backend_sched_alloc_splits().
//  2. ggml_backend_sched_alloc_splits() treats a failed gallocr_alloc_graph as
//     "reserve, then try once more":
//         if (backend_ids_changed || !ggml_gallocr_alloc_graph(...)) {
//             ... synchronise every backend ...
//             if (!ggml_gallocr_reserve_n(...)) return false;
//             if (!ggml_gallocr_alloc_graph(...)) return false;
//         }
//     Both calls go through this redirection, so the fresh allocator is reserved on
//     demand and then allocated -- on one buffer type and on many. The earlier
//     reading of this file ("multi-device fails outright") stopped at
//     gallocr_alloc_graph and never read its caller; it was wrong, and the retry is
//     why no extra hook and no write to sched_need_reserve is needed.
//
// What is left is the narrow part: create a fresh owner when the old one was freed,
// and keep the pairing straight.
//
// ============================================================================
// WHY THE UNARMED PATH IS UNSAFE, MEASURED RATHER THAN ARGUED
//
// ASan on herd-release-probe reported, on the first decode after a release:
//
//   heap-use-after-free, WRITE of size 4
//     #1 llm_graph_input_embd::set_input(llama_ubatch const*)   <- writes into it
//     #2 llm_graph_result::set_inputs(llama_ubatch const*)
//     #3 llama_context::process_ubatch(...)
//     #4 llama_context::decode(...)
//   0 bytes inside of a 78624-byte region
//   freed by: ggml_backend_buffer_free
//   allocated by: ggml_aligned_malloc
//
// The freed region is the compute buffer. It is freed correctly (the release did
// what it says) and then written into correctly-shaped by a caller that has every
// right to expect it to exist: `set_inputs()` copies the ubatch's tokens and
// embeddings into graph input tensors, and those tensors' `data` pointers were
// bound to the compute buffer by the *previous* reservation. Nothing re-binds them
// before the write, because `ggml_backend_sched_alloc_splits()` -- the retry that
// makes re-reservation look automatic -- runs only after the inputs are set.
//
// So the retry path exists, and on the unarmed path it is still too late: the
// re-reservation happens after the first use of the freed memory. Fixing the ordering
// needs the graph's input tensors re-bound to new buffers before `set_inputs`, which is
// inside upstream's `process_ubatch` -- no function-entry seam reaches it directly.
//
// It does not have to. `release_and_arm` does not insert code into process_ubatch; it
// makes upstream take the OTHER branch at the top of the same function. The reuse
// branch (`!graph_reuse_disable && gf_res_prev_active == res && res->can_reuse(gparams)`)
// is what skips alloc_graph; the rebuild branch calls `ggml_backend_sched_alloc_graph`
// *before* `set_inputs` (llama-context.cpp:2038-2068). llama_set_embeddings() on a real
// change sets sched_need_reserve, decode() calls sched_reserve(), sched_reserve() clears
// gf_res_prev_active -- and the rebuild branch is taken. Measured, not reasoned:
// HERD_RELEASE_EXPERIMENT=1 is ASan-clean and bit-identical; HERD_RELEASE_NO_ARM=1
// reproduces the report above.
//
// The earlier conclusion here ("cannot be made safe from outside upstream") was wrong,
// and it was wrong in the specific way this file warns about elsewhere: it named the
// ordering as unreachable without checking whether another ordering was reachable.
//
// Consequence, now that the armed path is measured: the MECHANISM works on one CPU
// buffer type (herd::phase::release_experiment_verified() is true) and it is still NOT
// wired into a product, so the product's capability query stays false
// (release_available()). Those are two different answers to two different questions,
// and the split exists because one flag answering both reported a feature the same run
// refused. A scheduler with several buffer types re-reserves per type inside
// ggml_backend_sched_alloc_splits; the same retry should cover them, but "should" is not
// a measurement and this host has no second backend. The product therefore refuses
// --phase-memory by name and asks for --no-phase-memory, which is a decision, not an
// impossibility.
// ============================================================================
#include "herd_patch.hpp"
#include "herd_buffer_station.hpp"
#include "herd_phase.hpp"

#include <ggml-backend.h>
#include <llama.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

// Upstream defines this inside ggml-backend.cpp (16). Our copies of per-scheduler
// buffer-type lists need the same bound; taking the number here states a fact about
// the pin instead of importing a macro from a private header.
constexpr int kMaxBackends = 16;

using new_n_fn          = ggml_gallocr_t (*)(ggml_backend_buffer_type_t *, int);
using free_fn           = void (*)(ggml_gallocr_t);
using reserve_n_fn      = bool (*)(ggml_gallocr_t, struct ggml_cgraph *, const int *, const int *);
using reserve_n_size_fn = void (*)(ggml_gallocr_t, struct ggml_cgraph *, const int *, const int *, size_t *);
using alloc_graph_fn    = bool (*)(ggml_gallocr_t, struct ggml_cgraph *);
using get_bufsize_fn    = size_t (*)(ggml_gallocr_t, int);
using sched_reserve_fn  = bool (*)(ggml_backend_sched_t, struct ggml_cgraph *);

new_n_fn          g_orig_new_n          = nullptr;
free_fn           g_orig_free           = nullptr;
reserve_n_fn      g_orig_reserve_n      = nullptr;
reserve_n_size_fn g_orig_reserve_n_size = nullptr;
alloc_graph_fn    g_orig_alloc_graph    = nullptr;
get_bufsize_fn    g_orig_get_bufsize    = nullptr;
sched_reserve_fn  g_orig_sched_reserve  = nullptr;

// -------- buffer types per handle, so a fresh allocator can be rebuilt --------
// Keyed by handle. A pointer to the caller's array would be wrong: it belongs to
// the scheduler. The lock here is only for this table; it is never held while the
// station is called, so there is one lock order -- this one, then the station's.
std::mutex g_bufts_mutex;
struct buft_list {
    ggml_backend_buffer_type_t types[kMaxBackends];
    int n = 0;
};
buft_list g_bufts[herd::buffers::station::capacity];
void *    g_handle_key[herd::buffers::station::capacity];

bool copy_bufts(void * handle, ggml_backend_buffer_type_t * out, int * n_out) {
    std::lock_guard<std::mutex> lock(g_bufts_mutex);
    for (int i = 0; i < herd::buffers::station::capacity; ++i) {
        if (g_handle_key[i] == handle && g_bufts[i].n > 0) {
            std::memcpy(out, g_bufts[i].types, std::size_t(g_bufts[i].n) * sizeof(out[0]));
            *n_out = g_bufts[i].n;
            return true;
        }
    }
    return false;
}

void remember_bufts(void * handle, ggml_backend_buffer_type_t * bufts, int n) {
    if (!handle || n <= 0 || n > kMaxBackends) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_bufts_mutex);
    for (int i = 0; i < herd::buffers::station::capacity; ++i) {
        if (g_handle_key[i] == handle) {
            return; // same handle: the types cannot have changed
        }
    }
    for (int i = 0; i < herd::buffers::station::capacity; ++i) {
        if (g_handle_key[i] == nullptr) {
            g_handle_key[i] = handle;
            g_bufts[i].n = n;
            std::memcpy(g_bufts[i].types, bufts, std::size_t(n) * sizeof(bufts[0]));
            return;
        }
    }
    // Full table: a handle whose types are unknown cannot be rebuilt after a
    // release, and release_sched() answers false for it rather than half-doing it.
}

void forget_bufts(void * handle) {
    std::lock_guard<std::mutex> lock(g_bufts_mutex);
    for (int i = 0; i < herd::buffers::station::capacity; ++i) {
        if (g_handle_key[i] == handle) {
            g_handle_key[i] = nullptr;
            g_bufts[i] = buft_list{};
            return;
        }
    }
}

// The allocator that must do the work for this handle, creating one when the handle
// has none (first use, or the first use after a release).
ggml_gallocr_t owner_for(ggml_gallocr_t handle) {
    if (!handle) {
        return nullptr;
    }
    if (void * owner = herd::buffers::shared().owner_of(handle)) {
        return static_cast<ggml_gallocr_t>(owner);
    }
    ggml_backend_buffer_type_t types[kMaxBackends];
    int n = 0;
    if (!copy_bufts(handle, types, &n)) {
        // Not created through this module (a scheduler built before the hooks were
        // installed). Its own allocator is the only one that knows its buffer types,
        // so use it and record nothing: a release on it will answer false instead of
        // guessing.
        return handle;
    }
    ggml_gallocr_t fresh = g_orig_new_n(types, n);
    if (!fresh) {
        // Out of memory, or a buffer type the allocator cannot take. Falling back to
        // the token would hand upstream an allocator that does not allocate, so this
        // is a failure the caller must see, not a value to return.
        std::fprintf(stderr, "[herd] buffers: could not create an allocator for %p\n",
                     static_cast<void *>(handle));
        std::abort();
    }
    if (!herd::buffers::shared().adopt(handle, fresh)) {
        // Lost a race, or the entry vanished. Two threads can reach here together
        // only for the same handle, which means the same scheduler -- something
        // upstream's scheduler does not support either. But aborting would punish a
        // benign ordering, and the fresh allocator would leak on the way out, so the
        // loser frees what it made and re-reads: either the winner's allocator is now
        // installed, or the handle is gone and the caller gets the token's own
        // (non-allocating) allocator, which is the documented fallback.
        g_orig_free(fresh);
        if (void * winner = herd::buffers::shared().owner_of(handle)) {
            return static_cast<ggml_gallocr_t>(winner);
        }
        return handle;
    }
    return fresh;
}

// -------- hooks: redirection, no policy ------------------------------------

ggml_gallocr_t herd_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs) {
    ggml_gallocr_t handle = g_orig_new_n(bufts, n_bufs);
    remember_bufts(handle, bufts, n_bufs);
    const auto outcome = herd::buffers::shared().bind(handle, nullptr, n_bufs);
    if (outcome == herd::buffers::outcome::refused_full) {
        // The handle still works through upstream's own allocator -- owner_for()
        // falls back to it -- so this is not fatal, only un-releasable.
        std::fprintf(stderr, "[herd] buffers: station full; %p will not be releasable\n",
                     static_cast<void *>(handle));
    }
    return handle;
}

// Correlation, not layout: during ggml_backend_sched_reserve the scheduler calls
// reserve_n on its own allocator, so the scheduler being executed is the one whose
// allocator this is. That is how a pairing is learned without reading the struct.
thread_local ggml_backend_sched_t t_sched = nullptr;

bool herd_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    t_sched = sched;
    const bool ok = g_orig_sched_reserve(sched, graph);
    t_sched = nullptr;
    return ok;
}

bool herd_reserve_n(ggml_gallocr_t handle, struct ggml_cgraph * graph,
                    const int * node_ids, const int * leaf_ids) {
    if (t_sched) {
        herd::buffers::shared().link_scheduler(t_sched, handle);
    }
    return g_orig_reserve_n(owner_for(handle), graph, node_ids, leaf_ids);
}

void herd_reserve_n_size(ggml_gallocr_t handle, struct ggml_cgraph * graph,
                         const int * node_ids, const int * leaf_ids, size_t * sizes) {
    g_orig_reserve_n_size(owner_for(handle), graph, node_ids, leaf_ids, sizes);
}

bool herd_alloc_graph(ggml_gallocr_t handle, struct ggml_cgraph * graph) {
    return g_orig_alloc_graph(owner_for(handle), graph);
}

std::size_t herd_get_bufsize(ggml_gallocr_t handle, int index) {
    // A query must not allocate: owner_of(), not owner_for(). After a release the
    // answer is zero because nothing is held, and that zero is the truth.
    if (void * owner = herd::buffers::shared().owner_of(handle)) {
        return g_orig_get_bufsize(static_cast<ggml_gallocr_t>(owner), index);
    }
    return 0;
}

void herd_free(ggml_gallocr_t handle) {
    if (!handle) {
        return;
    }
    // Retire first: the owner is freed below, and a later allocator at the same
    // address must not inherit the pairing.
    herd::buffers::shared().unlink_scheduler(handle);
    void * owner = herd::buffers::shared().retire(handle);
    forget_bufts(handle);
    // Unconditional: the station never records a handle as its own owner (see
    // bind/adopt), so an owner that came back is always a distinct allocator. The
    // `owner != handle` test that used to stand here could never fire, and code that
    // looks like a check but cannot fire is worse than none -- it invites the next
    // reader to assume a guarantee nobody provides.
    if (owner) {
        g_orig_free(static_cast<ggml_gallocr_t>(owner));
    }
    // Upstream's token has no buffers of its own, but its bookkeeping is real.
    g_orig_free(handle);
}

} // namespace

HERD_SEAM(buffers_new_n, "ggml_gallocr_new_n", &ggml_gallocr_new_n, &herd_new_n,
          &g_orig_new_n, nullptr)
HERD_SEAM(buffers_reserve_n, "ggml_gallocr_reserve_n", &ggml_gallocr_reserve_n, &herd_reserve_n,
          &g_orig_reserve_n, nullptr)
HERD_SEAM(buffers_reserve_n_size, "ggml_gallocr_reserve_n_size", &ggml_gallocr_reserve_n_size,
          &herd_reserve_n_size, &g_orig_reserve_n_size, nullptr)
HERD_SEAM(buffers_alloc_graph, "ggml_gallocr_alloc_graph", &ggml_gallocr_alloc_graph,
          &herd_alloc_graph, &g_orig_alloc_graph, nullptr)
HERD_SEAM(buffers_get_bufsize, "ggml_gallocr_get_buffer_size", &ggml_gallocr_get_buffer_size,
          &herd_get_bufsize, &g_orig_get_bufsize, nullptr)
HERD_SEAM(buffers_free, "ggml_gallocr_free", &ggml_gallocr_free, &herd_free, &g_orig_free, nullptr)
HERD_SEAM(buffers_sched_reserve, "ggml_backend_sched_reserve", &ggml_backend_sched_reserve,
          &herd_sched_reserve, &g_orig_sched_reserve, nullptr)

namespace herd::buffers {

bool release_and_arm(llama_context * ctx, bool embeddings_currently) {
    if (!ctx) {
        return false;
    }
    // The context's scheduler is known to the phase station, which learned it from
    // the compute hooks. Asking there keeps this file from needing llama-context.h
    // and from reading any upstream struct.
    void * sched = herd::phase::scheduler_of(ctx);
    if (!sched) {
        return false; // never decoded: no scheduler was ever recorded, nothing to free
    }

    // Wait for the device BEFORE freeing anything, and this is the one guard the
    // experiment was missing.
    //
    // The fork's release_compute() begins with `synchronize()` (src/llama-context.cpp
    // :2651) and refuses outright when
    // `ggml_backend_sched_has_unproven(sched)` reports an async submission still in
    // flight (:2647-2650, "refusing to release compute buffers: an async submission is
    // still unproven"). Upstream has no `unproven` query -- `ggml_backend_sched_synchronize`
    // is the public operation that establishes the same precondition -- so this call is
    // the parity mechanism: after it returns, no queued work can still be writing into
    // the buffers that `release_sched` is about to free.
    //
    // Without it the experiment was correct only because every caller happened to
    // synchronize first (the probe does, after each decode) and because the CPU backend
    // finishes inline. On an async backend, freeing the allocator under a live submission
    // is a use-after-free the host cannot see: the device writes into memory the host has
    // already returned. That is the defect this line fixes, and it is the difference
    // between "the probe passes" and "the operation is safe".
    llama_synchronize(ctx);

    if (!release_sched(sched)) {
        return false;
    }

    // Arm the scheduler's reservation through the one public lever that sets the
    // flag: llama_set_embeddings() marks sched_need_reserve whenever the value
    // actually changes (llama-context.cpp:1583-1590). Toggling twice restores
    // cparams.embeddings exactly and leaves the flag set -- one intended side
    // effect, stated here rather than hidden in a helper name.
    //
    // What the arming buys, in upstream's own lines:
    //   decode()        :2506  calls sched_reserve() before the ubatch loop;
    //   sched_reserve() :935   returns early unless sched_need_reserve is set;
    //   sched_reserve() :962   clears gf_res_prev_active and resets every graph
    //                          result when it does run;
    //   process_ubatch():2024  takes the rebuild branch because gf_res_prev_active
    //                          is now null -- and that branch calls
    //                          ggml_backend_sched_alloc_graph BEFORE set_inputs.
    // That last ordering is exactly what the ASan report said was missing: the
    // input tensors are bound to the fresh allocator before anything writes to them.
    llama_set_embeddings(ctx, !embeddings_currently);
    llama_set_embeddings(ctx, embeddings_currently);
    return true;
}

bool release_sched(void * sched) {
    // One station call, one lock. The pairing lookup and the release used to be two
    // calls, which left a window: the scheduler could be destroyed and its address
    // reused by a new allocator between them, and the second call would then free a
    // live one. See release_scheduler() in the header.
    void * owner = shared().release_scheduler(sched);
    if (!owner) {
        return false; // unknown scheduler, or already released: nothing was freed
    }
    // No comparison here, and that is the fix rather than an omission.
    //
    // The previous version read `if (owner != sched)`, intending "do not free the
    // token". `sched` is a ggml_backend_sched_t and `owner` is a ggml_gallocr_t:
    // two unrelated objects that can never be equal, so the guard always passed and
    // protected nothing. It was not caught because the state it was meant to exclude
    // -- an allocator owning itself -- was also never produced.
    //
    // Both facts are now enforced where they can be: the station refuses to install
    // a handle as its own owner (bind/adopt), so whatever comes back from
    // release_scheduler() is a distinct allocator and is freed unconditionally. A
    // guard that names the wrong object is worse than no guard: it reads as
    // protection.
    g_orig_free(static_cast<ggml_gallocr_t>(owner));
    return true;
}

} // namespace herd::buffers
