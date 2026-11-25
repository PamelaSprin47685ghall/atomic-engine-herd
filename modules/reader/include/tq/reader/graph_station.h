// tq/reader/graph_station.h — the request ledger for reader graph builds.
//
// This is the ticket-station method applied to the one hard part of the reader lane: a request
// must apply to EXACTLY ONE graph build, a build that was not asked for must not be touched, and
// two builds that are genuinely independent must not block each other.
//
//   * the STATION holds the state and has no asynchronous behaviour. Every function below either
//     records a fact or answers one. Nothing waits, nothing queues, nothing retries.
//   * the TICKET is upstream's graph build; its local state lives in a stack frame this module
//     cannot see. What this station owns is only "was a reader emission asked for by this build,
//     and with what view".
//   * the CAPABILITY is injected, never reached for. The station does not call the emitter; it
//     answers what the caller should do. That is why this header has no ggml in it and the
//     station is testable without a model, a context, or a device.
//
// Why a guard rather than a flag: a thread_local "reader is on" boolean has to be cleared by
// every exit path, and the paths that matter -- early return, a throw from ggml, a refusal
// halfway through -- are exactly the ones a caller forgets. A scope object closes on the way out
// because the language guarantees it, and a refusal becomes a *recorded* fact rather than a
// stale request that silently applies to the next build.
//
// ---------------------------------------------------------------------------------------------
// The first revision of this station had five defects. They are listed here because each one is
// a shape that recurs, and the next person changing this file should see what was got wrong:
//
//   1. ONE SLOT PER PROCESS. A second open was refused as "nested" even when it came from another
//      thread for a DIFFERENT context -- two llama_contexts decoding in parallel is a supported
//      configuration, and it was refused with a message that named the wrong cause ("a scope is
//      already open on this thread", which was a lie when the holder was another thread).
//      Now: one slot per context, so independent builds coexist; nesting is detected per context.
//
//   2. find() RETURNED A POINTER INTO MUTABLE STATE. `const active_request *` stayed valid "until
//      close()", but close() from another thread frees the view's heap buffer while the caller
//      reads through that pointer. A per-layer hook reading the view is exactly the reader that
//      would race. Now: find() COPIES into caller storage under the lock, and no pointer to the
//      station's internals ever escapes.
//
//   3. close() HAD NO EPOCH. `close(context)` from a scope whose context had since been re-opened
//      would close the NEWER request -- a stale guard killing a live build, and the reason a
//      manual close followed by a destructor was not merely a counter bump. Now: close takes the
//      epoch the scope opened with, and a stale close is refused by name.
//
//   4. A NULL CONTEXT WAS A VALID KEY. `open(nullptr, ...)` and `find(nullptr)` would match each
//      other, so two unrelated builds without a context would collide. Now: refused as no_context.
//
//   5. THE EPOCH COULD WRAP. `epoch += 1` as uint64 wraps after 2^64 opens, and once it wraps a
//      stale guard's epoch matches a live request's -- reopening defect 3 through the back door.
//      "584 years" is a statement about run length, not a proof; the same argument was rejected
//      for the whiteboard. Now: at the maximum the station refuses to open (epoch_exhausted),
//      which is checkable as a pure rule.
// ---------------------------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <tq/reader/attention.h>
// span_rope_fn is the module's own declaration of the rotation capability the emitter injects.
// graph_emit.h forward-declares ggml_context/ggml_tensor, so including it costs this station no
// ggml dependency -- which is what keeps the station testable without a model.
#include <tq/reader/graph_emit.h>

namespace tq::reader {

// There is no limit on distinct contexts, and the earlier constant kMaxContexts = 8 was removed
// deliberately rather than raised.
//
// The bound was justified as "allocation-free on the hot path", and that justification does not
// survive its own premise: the number of simultaneous graph builds is a property of the host's
// workload, not of this station. A server decoding N sequences in parallel has N live contexts,
// and at N == 9 the ninth was refused with a message telling the operator to raise a constant in
// a header -- an arbitrary limit on someone else's configuration, invented by a component that
// cannot see it.
//
// What the station actually needs is one slot per context, and the number of slots is bounded by
// the number of contexts, which is data. So the slots are a vector: it grows to the size the
// workload has, and it stops growing as soon as the workload stops growing. Steady-state
// builds reuse freed slots, so the amortized allocation is zero -- the property the constant was
// trying to buy, obtained without the ceiling.
//
// Allocation CAN fail. That is the one case left, and it is reported as `no_room` rather than
// allowed to throw out of a function whose callers are not written for exceptions.

// Why an open() was refused. Every cause is named, because the one that matters for a
// silent-fallback bug is the difference between "another build already owns THIS context" (a
// caller error, and the inner one must not replace the outer) and "another context already holds
// a slot" (never a reason to refuse -- that was defect 1).
enum class open_status {
    ok,
    no_context,       // a null context: a graph build always has one, and null would alias
    nested,           // this context is already open on this thread; the previous one keeps it
    owned_elsewhere,  // this context is open on ANOTHER thread: one context cannot build twice
    empty_view,       // the view has no spans, so there is nothing to emit
    empty_span,       // some span has no cells: the union would be smaller than the caller thinks
    no_room,          // the slot table could not grow: allocation failed, nothing was recorded
    epoch_exhausted,  // the station-wide counter is at its maximum and cannot issue a fresh epoch
};

const char * describe(open_status status);

// Why a close() did not take effect. `absent` is benign (a double close, or a guard for a request
// that a caller already closed); `stale` and `not_owner` are not, and they are different facts:
//   absent     nothing to close -- the slot for this context is empty
//   stale      the slot holds a request with a DIFFERENT epoch: a newer build owns it now, and
//              the caller's guard outlived its own request
//   not_owner  the slot holds this epoch, but a different thread opened it
enum class close_status { closed, absent, stale, not_owner };

const char * describe(close_status status);

// The fact a scope publishes while it is open.
struct active_request {
    const void *   context = nullptr;   // the graph context this build belongs to
    query_view     view;                // a COPY: no reference to the caller's temporary
    span_rope_fn   rope = nullptr;      // injected rotation, or null for none
    void *         rope_user = nullptr;
    std::uint64_t  epoch = 0;           // identifies this open, for stale-close detection
};

// One station per process. Every operation takes one mutex; nothing here blocks, polls, or calls
// out, so a lock held across a few field writes and one vector copy is the whole contention story.
class station {
public:
    station() = default;
    station(const station &) = delete;
    station & operator=(const station &) = delete;

    // Opens a request for `context`. On ok, `issued_epoch` (may be null) receives the epoch the
    // request was given; a scope needs it to close only its own request.
    open_status open(const void * context, query_view view, span_rope_fn rope = nullptr,
                     void * rope_user = nullptr, std::uint64_t * issued_epoch = nullptr);

    // Closes the request for `context` IF it is the one at `epoch`. Anything else is reported and
    // changes nothing -- this is what stops a stale guard from killing a live build.
    close_status close(const void * context, std::uint64_t epoch);

    // Copies the request that applies to `context` right now into `out`.
    //
    // The copy is made under the lock, so `out` never aliases station state and cannot be
    // invalidated by a concurrent close -- that was defect 2. The cost is one small vector copy
    // per call, which the hook can pay once per build and reuse across layers.
    //
    // The result is NAMED, and it has to be. The previous signature returned bool, and it answered
    // false in two situations that a caller must not treat alike:
    //
    //   * nothing was asked for -- this thread holds no request for this context. Proceed with
    //     stock attention: no reader emission was wanted.
    //   * something WAS asked for and the copy could not be made (allocation failure). Proceeding
    //     here silently drops an emission the caller explicitly requested, which is the exact
    //     degradation this project keeps removing: every number downstream looks like a normal
    //     run, and the only difference is that a feature did not happen.
    //
    // A caller that asked for a reader emission must treat `no_memory` as a failure of its own
    // request. A caller that did not ask cannot see the difference, which is correct -- it says
    // nothing about a request it never made.
    //
    // `out` is either fully written or untouched. The copy is assembled locally and moved into
    // `out` only after the one allocating step succeeds, because the previous version assigned
    // member-wise straight into `out`: `context`, `rope` and `epoch` landed before the view copy
    // could throw, leaving the caller with a request that named one build and carried another's
    // view. A partial update of an output parameter is the same defect as a partial publish.
    enum class find_status {
        found,        // `out` holds the request; every field was written
        no_request,   // this thread holds no live request for this context
        no_memory,    // a request exists, but copying it failed; `out` was not touched
        bad_argument, // a null pointer argument (the output or the context): a caller bug,
                      // not a state of the station
    };

    find_status find(const void * context, active_request * out) const;

    // Test-only: make the next find() take its allocation-failure path.
    //
    // The branch above is otherwise unreachable in a test, and a branch nobody runs is a branch
    // nobody has checked -- including the property this round is about, that `out` is untouched.
    // The seam substitutes only the ALLOCATION OUTCOME; the catch, the local assembly and the
    // move are the shipping code. What it does not prove: that a real operator-new failure
    // arrives at that instruction. That is the allocator's contract, not this station's.
    static void inject_copy_failure_for_test(bool on);

    // The epoch rule, as a pure function, so the boundary is testable without 2^64 opens.
    static bool epoch_available(std::uint64_t next_epoch);

    // Evidence. A run that reports reader numbers without these cannot say whether a request was
    // ever refused, and a refusal is the difference between "the feature ran" and "the feature
    // was asked for and the caller was told why not".
    struct counters {
        std::uint64_t opened = 0;
        std::uint64_t closed = 0;
        std::uint64_t close_absent = 0;
        std::uint64_t close_stale = 0;
        std::uint64_t close_not_owner = 0;
        std::uint64_t refused_no_context = 0;
        std::uint64_t refused_nested = 0;
        std::uint64_t refused_owned_elsewhere = 0;
        std::uint64_t refused_empty = 0;
        std::uint64_t refused_no_room = 0;
        std::uint64_t refused_epoch = 0;
    };
    counters snapshot() const;

    // Live requests, for diagnostics: how many slots are in use.
    std::size_t live_count() const;

private:
    struct slot {
        const void *  context = nullptr;
        active_request request{};
        std::thread::id owner{};
        bool          live = false;
    };

    // Linear over the live set. A scan beats hashing for the sizes this station actually sees
    // (one entry per decoding context), and it keeps the state a single flat container: no
    // second structure to keep in step with the slots.
    //
    // The vector may reallocate when it grows, which invalidates pointers into it. Every use of
    // a slot pointer in the implementation is bounded by one critical section with no growth in
    // it -- growth happens only in open(), before the pointer is taken -- so the invalidation
    // cannot be observed.
    slot * find_slot(const void * context);
    const slot * find_slot(const void * context) const;

    mutable std::mutex mutex_;
    std::vector<slot>  slots_;
    std::uint64_t      next_epoch_ = 0;   // 0 means "no epoch issued yet"
    counters           counters_{};
};

// The scope object the hook and its callers use.
//
// Non-copyable and non-movable: a moved guard would close a scope the moved-from object still
// names, and the whole point is that exactly one object is responsible for closing. The scope
// remembers the epoch it opened with, which is what makes a stale close impossible (defect 3).
class scope {
public:
    scope(station & board, const void * context, query_view view, span_rope_fn rope = nullptr,
          void * rope_user = nullptr)
        : board_(board), context_(context) {
        status_ = board_.open(context_, std::move(view), rope, rope_user, &epoch_);
        live_ = status_ == open_status::ok;
    }

    ~scope() {
        if (live_) {
            board_.close(context_, epoch_);
        }
    }

    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;

    // False when the request was refused. A caller that asked for a reader emission must treat
    // that as a failure of its own request, not as permission to continue with stock attention.
    bool opened() const { return live_; }
    open_status status() const { return status_; }
    std::uint64_t epoch() const { return epoch_; }

private:
    station &     board_;
    const void *  context_;
    open_status   status_ = open_status::no_context;
    std::uint64_t epoch_ = 0;
    bool          live_ = false;
};

} // namespace tq::reader
