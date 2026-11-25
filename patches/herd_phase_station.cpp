#include "herd_phase_station.hpp"

// The release rule's contract lives in herd_phase.hpp, next to release_compute(); its
// definition lives here, with the other pure logic. Both headers are free of ggml and
// llama, so this TU still links without either -- which is what lets the rule's test run
// on a machine that has no GPU to refuse.
#include "herd_phase.hpp"

namespace herd::phase {

release_verdict decide_release(const release_facts & facts) {
    // The rule lives in this translation unit, not beside the call that applies it.
    // This file has no ggml, no llama and no Dobby, so the rule can be exercised on any
    // machine -- including the configurations a given machine cannot produce, which is the
    // whole point: the two-buffer-type case only exists on multi-device hardware, and a
    // rule tested only there is a rule tested nowhere.
    if (!facts.scheduler_known) {
        return {false, "no decode has run on this context, so its scheduler has never been seen"};
    }
    if (facts.n_backends <= 0) {
        return {false, "the scheduler reports no backend; there is no allocator whose release "
                       "could be observed"};
    }
    // Incomplete facts are refused before they are judged.
    //
    // A null slot is the one input that can make the other fields look BETTER than they are:
    // skip it and `all_backends_cpu` stays true, `n_buffer_types` stays at the count of the
    // slots that answered. The gather loop did exactly that with `continue`, so a scheduler
    // it could not fully read was reported as a clean single-type CPU scheduler. This check
    // makes "I could not see it" a refusal instead of an absence of evidence in favour of
    // the run.
    if (facts.n_null_backends > 0) {
        return {false, "the scheduler reported a null backend; its configuration cannot be read, "
                       "so no release can be justified from it"};
    }
    if (facts.n_null_buffer_types > 0) {
        return {false, "a backend reported a null buffer type; the allocator that would be freed "
                       "cannot be identified"};
    }
    // Exactly one backend, conservatively.
    //
    // The mechanism is per buffer type, and two CPU backends sharing one type really does go
    // through the same retry path as one -- that argument is sound, and it is still not a
    // measurement. Nothing on this host can produce a two-backend scheduler, so allowing that
    // row would mean shipping an unexercised allowance inside the one function whose whole job
    // is to keep unexercised configurations out of a memory-freeing path. A refusal here costs
    // an operator one flag; the alternative costs a use-after-free nobody has looked for.
    if (facts.n_backends != 1) {
        return {false, "the scheduler has more than one backend; the release is measured on "
                       "exactly one, and a multi-backend configuration has never been through "
                       "this path"};
    }
    // Fail closed on what has not been measured. "one backend" is not "CPU": a single CUDA
    // backend satisfies the old test, and the release synchronizes and then frees memory a
    // device may still be writing into. That ordering is observed only where completion is
    // synchronous with the host.
    if (!facts.all_backends_cpu) {
        return {false, "the scheduler has a non-CPU backend; the release is measured only on CPU, "
                       "where synchronize-then-free is observed. On a GPU the same sequence has "
                       "never been measured"};
    }
    // Kept although the one-backend rule above makes it unreachable from a real scheduler
    // (one backend reports one buffer type). It stays because it is the mechanism's actual
    // limit: if the one-backend refusal is ever relaxed -- after somebody measures it -- this
    // is the condition that must still hold, and dropping it now would quietly widen the
    // contract at the moment the relaxation lands.
    if (facts.n_buffer_types != 1) {
        return {false, "the scheduler holds more than one buffer type; the retry in "
                       "ggml_backend_sched_alloc_splits would have to re-reserve every type at "
                       "once, which is not measured"};
    }
    return {true, nullptr};
}

scheduler_station::observation scheduler_station::observe(const llama_context * ctx,
                                                          ggml_backend_sched_t sched) {
    // A null scheduler is not a binding. Storing it would make with_scheduler()
    // report "known" for a context whose bytes cannot be counted, which is the
    // difference between an honest "unknown" and a fabricated zero.
    if (!ctx || !sched) {
        return observation::refused_null;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < count_; ++i) {
        if (entries_[i].ctx == ctx) {
            entries_[i].sched = sched;
            return observation::updated;
        }
    }
    if (count_ >= capacity) {
        // Keep what is already known. Overwriting an entry would silently make
        // another context's accounting unknown, and evicting by any policy would
        // be a guess about which context still matters.
        return observation::refused_full;
    }
    entries_[count_++] = entry{ctx, sched};
    return observation::recorded;
}

bool scheduler_station::forget(const llama_context * ctx) {
    if (!ctx) {
        return false;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < count_; ++i) {
        if (entries_[i].ctx == ctx) {
            // Swap-remove: order carries no meaning, so the last entry refills the
            // hole and the array stays dense.
            entries_[i] = entries_[--count_];
            entries_[count_].ctx = nullptr;
            entries_[count_].sched = nullptr;
            return true;
        }
    }
    return false;
}

int scheduler_station::tracked() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
}

int scheduler_station::forget_scheduler(ggml_backend_sched_t sched) {
    if (!sched) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    int removed = 0;
    for (int i = 0; i < count_;) {
        if (entries_[i].sched == sched) {
            // Swap-remove, as in forget(): a destroyed scheduler must leave no
            // pointer behind for ANY context, so the loop continues at the same
            // index with the element that was just moved into the hole.
            entries_[i] = entries_[--count_];
            entries_[count_] = entry{};
            ++removed;
        } else {
            ++i;
        }
    }
    return removed;
}

scheduler_station & station() {
    // Immortal on purpose. The binding is released from the llama_free seam, and
    // a context owned by another translation unit's static object is destroyed at
    // exit, in an order this file cannot see. A function-local static here would
    // be destroyed at some point during that same shutdown, and a forget() after
    // it is a use-after-destroy of the mutex -- a crash during exit that depends
    // on link order, so it appears and disappears with unrelated edits.
    //
    // Leaking one mutex and 64 pointers costs nothing and removes the ordering
    // question entirely: every user of the station runs before process teardown.
    static scheduler_station * instance = new scheduler_station();
    return *instance;
}

} // namespace herd::phase
