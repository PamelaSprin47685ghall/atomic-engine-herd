#include "tq/reader/graph_station.h"

#include <limits>
#include <new>
#include <atomic>

namespace tq::reader {

namespace {
// The test seam's flag. Namespace scope, atomically read, false by default: a product never sets
// it, so the shipping path has one relaxed atomic load and no branch that changes behaviour.
std::atomic<bool> t_copy_failure{false};
} // namespace

const char * describe(open_status status) {
    switch (status) {
        case open_status::ok:              return "ok";
        case open_status::no_context:      return "no context: a graph build always has one, and a null key would "
                                                  "make two unrelated builds collide";
        case open_status::nested:          return "this context is already open on this thread; the open request "
                                                  "keeps the slot, because replacing it would apply the inner view "
                                                  "to the outer graph";
        case open_status::owned_elsewhere: return "this context is open on another thread; one context cannot build "
                                                  "two graphs at once";
        case open_status::empty_view:      return "the view has no spans, so there is nothing to emit";
        case open_status::empty_span:      return "a span in the view has no cells; its absence would make the "
                                                  "union smaller than the caller believes";
        case open_status::no_room:          return "the slot table could not grow: allocation failed, so nothing "
                                                  "was recorded";
        case open_status::epoch_exhausted: return "the station's epoch counter is exhausted; it refuses to issue a "
                                                  "value a stale guard could match";
    }
    return "unknown status";
}

const char * describe(close_status status) {
    switch (status) {
        case close_status::closed:    return "closed";
        case close_status::absent:    return "nothing to close: no request is open for this context";
        case close_status::stale:     return "the slot holds a newer request; this guard outlived its own";
        case close_status::not_owner: return "the request belongs to another thread";
    }
    return "unknown status";
}

bool station::epoch_available(std::uint64_t last_issued) {
    // The station issues 1, 2, ..., max. `last_issued` never wraps: at the maximum there is no
    // further value to issue, and issuing a wrapped one would let a stale guard's epoch match a
    // live request's -- which is the same defect the epoch exists to prevent, reached the long
    // way round. "It would take 2^64 opens" is a statement about run length, not a proof.
    return last_issued != std::numeric_limits<std::uint64_t>::max();
}

station::slot * station::find_slot(const void * context) {
    for (slot & candidate : slots_) {
        if (candidate.live && candidate.context == context) {
            return &candidate;
        }
    }
    return nullptr;
}

const station::slot * station::find_slot(const void * context) const {
    for (const slot & candidate : slots_) {
        if (candidate.live && candidate.context == context) {
            return &candidate;
        }
    }
    return nullptr;
}

open_status station::open(const void * context, query_view view, span_rope_fn rope,
                          void * rope_user, std::uint64_t * issued_epoch) {
    std::lock_guard<std::mutex> guard(mutex_);

    // A null context is refused rather than used as a key. Two unrelated builds without a context
    // would then match each other through find(), which is a far worse failure than a refusal
    // that names the cause: one build would apply another build's view.
    if (!context) {
        ++counters_.refused_no_context;
        return open_status::no_context;
    }

    // The view is validated before anything is recorded, so a refusal leaves the station exactly
    // as it was. A caller that treats the refusal as fatal cannot have disturbed a request another
    // build is using.
    if (view.spans.empty()) {
        ++counters_.refused_empty;
        return open_status::empty_view;
    }
    for (const span_keys & span : view.spans) {
        if (span.cells.empty()) {
            ++counters_.refused_empty;
            return open_status::empty_span;
        }
    }

    // This context already has a request. Two different causes, two different names: the same
    // thread trying to open twice is nesting (a caller bug, and the outer view must win), while
    // another thread holding this context means two builds of one graph are interleaved -- which
    // cannot be made correct by choosing a winner, so it is reported separately.
    if (const slot * existing = find_slot(context)) {
        if (existing->owner == std::this_thread::get_id()) {
            ++counters_.refused_nested;
            return open_status::nested;
        }
        ++counters_.refused_owned_elsewhere;
        return open_status::owned_elsewhere;
    }

    // A refused open does not spend an epoch: the counter tracks requests that exist, and a
    // refusal that consumed one would move it for no one.
    if (!epoch_available(next_epoch_)) {
        ++counters_.refused_epoch;
        return open_status::epoch_exhausted;
    }

    slot * free_slot = nullptr;
    for (slot & candidate : slots_) {
        if (!candidate.live) {
            free_slot = &candidate;
            break;
        }
    }

    const std::uint64_t epoch = next_epoch_ + 1;

    if (!free_slot) {
        // No ceiling on distinct contexts: grow by one. The number of live builds is the host's
        // workload, and a station that cannot see it has no business bounding it.
        //
        // Growth is attempted BEFORE anything is recorded, so a failed allocation leaves the
        // station exactly as it was -- the same rule as every other refusal here. `no_room` is
        // the only remaining way to be turned away, and it names its cause rather than telling
        // an operator to raise a constant.
        //
        // Reallocation invalidates slot pointers, which is why the pointer above is taken after
        // this point and not before: on the growth path there is no stale pointer to use.
        try {
            slots_.push_back(slot{});
            free_slot = &slots_.back();
        } catch (const std::bad_alloc &) {
            ++counters_.refused_no_room;
            return open_status::no_room;
        }
    }
    next_epoch_ = epoch;

    // Fields first, `live` last, and the epoch published before the slot can be seen.
    //
    // Order matters because of what a reader may observe: `find_slot` matches on `live`, so a
    // slot is invisible to every other operation until the last assignment. That is what makes a
    // half-written slot unreachable rather than merely unlikely -- and it is why `next_epoch_`
    // is written before `live = true`: if the slot became visible first, a find() could see a
    // request whose epoch had not been published, and close() would then compare against the
    // previous value.
    //
    // None of these assignments can throw (a moved-from query_view is an empty vector, and the
    // rest are scalars), which is what lets `live = true` be the single commit point.
    free_slot->context = context;
    free_slot->request.context = context;
    free_slot->request.view = std::move(view);
    free_slot->request.rope = rope;
    free_slot->request.rope_user = rope_user;
    free_slot->request.epoch = epoch;
    free_slot->owner = std::this_thread::get_id();
    free_slot->live = true;

    ++counters_.opened;
    if (issued_epoch) {
        *issued_epoch = epoch;
    }
    return open_status::ok;
}

close_status station::close(const void * context, std::uint64_t epoch) {
    std::lock_guard<std::mutex> guard(mutex_);

    slot * existing = find_slot(context);
    if (!existing) {
        // Benign: a guard for a request a caller already closed, or a double close. Counted so a
        // run can tell "the caller closed twice" from "the caller's guard was stale", but it is
        // not an error and it changes nothing.
        ++counters_.close_absent;
        return close_status::absent;
    }

    // The epoch test comes BEFORE the thread test, because it is the one that protects a live
    // build. A guard whose request was closed and whose context was re-opened must not close the
    // newer request -- that is a stale guard killing a build in progress, and the counter records
    // that it happened rather than reporting it as a quiet mismatch.
    if (existing->request.epoch != epoch) {
        ++counters_.close_stale;
        return close_status::stale;
    }
    if (existing->owner != std::this_thread::get_id()) {
        ++counters_.close_not_owner;
        return close_status::not_owner;
    }

    existing->live = false;
    existing->context = nullptr;
    existing->owner = std::thread::id{};
    existing->request = active_request{};
    ++counters_.closed;
    return close_status::closed;
}

station::find_status station::find(const void * context, active_request * out) const {
    if (!out) {
        // A null out is a caller bug. It used to answer `false`, the same value as "nothing was
        // asked for", which turned a programming error into an ordinary absence.
        return find_status::bad_argument;
    }
    if (!context) {
        // A null CONTEXT is the same class of mistake, and reporting it as `no_request` was
        // asymmetric with open(): open(nullptr) is refused as no_context, so a station can never
        // hold a request under that key -- which means "there is no request for nullptr" is true
        // but useless. The caller wanted a build it does not have, and answering absence invites
        // it to continue with stock attention, which is the silent degradation this round exists
        // to remove. One meaning for an invalid argument, not two.
        return find_status::bad_argument;
    }
    std::lock_guard<std::mutex> guard(mutex_);

    const slot * existing = find_slot(context);
    if (!existing) {
        return find_status::no_request;
    }
    // A request belongs to the build that opened it, on the thread that opened it. Returning
    // another thread's request would let a worker apply this view to a context it does not own,
    // with the view's contents mutated underneath it.
    if (existing->owner != std::this_thread::get_id()) {
        return find_status::no_request;
    }

    // Assembled HERE, not in `out`.
    //
    // The previous version did `*out = existing->request`, which is member-wise: `context`,
    // `rope`, `rope_user` and `epoch` are assigned before `view`, and `view` is the only member
    // that allocates. So a failed copy left the caller holding a request that named one build and
    // carried another's view -- a partial update of an output parameter, which is the same defect
    // as a partial publish.
    //
    // Every field below is a scalar or a pointer copy; the one allocating step is the view, and
    // nothing is written to `out` until it has succeeded.
    active_request assembled;
    assembled.context = existing->request.context;
    assembled.rope = existing->request.rope;
    assembled.rope_user = existing->request.rope_user;
    assembled.epoch = existing->request.epoch;

    try {
        if (t_copy_failure.load(std::memory_order_acquire)) {
            // The test seam substitutes the allocation outcome only; the catch below and the
            // move after it are the shipping code.
            throw std::bad_alloc();
        }
        assembled.view = existing->request.view;
    } catch (const std::bad_alloc &) {
        // Reported, not swallowed. A caller that asked for an emission can now tell "nobody asked"
        // from "your request exists and I could not hand it to you", and act on the second.
        return find_status::no_memory;
    }

    // No allocation: pointers, scalars, and a vector move whose allocator is stateless.
    *out = std::move(assembled);
    return find_status::found;
}

void station::inject_copy_failure_for_test(bool on) {
    t_copy_failure.store(on, std::memory_order_release);
}

station::counters station::snapshot() const {
    std::lock_guard<std::mutex> guard(mutex_);
    return counters_;
}

std::size_t station::live_count() const {
    std::lock_guard<std::mutex> guard(mutex_);
    std::size_t live = 0;
    for (const slot & candidate : slots_) {
        if (candidate.live) {
            ++live;
        }
    }
    return live;
}

} // namespace tq::reader
