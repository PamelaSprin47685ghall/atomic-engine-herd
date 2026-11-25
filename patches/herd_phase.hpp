// herd/phase.hpp — the two things batch-compose needs at a phase boundary.
//
// The fork put three entry points in llama.h (llama_set_decode_phase,
// llama_memory_release_compute, llama_memory_compute_bytes) and backed them with
// members of llama_context. Upstream has none of them, and no member of an
// opaque struct can be added from outside, so this is the honest split:
//
//   compute_bytes  — implementable. The number is a pure function of the
//                    scheduler's buffers, and upstream exposes every accessor
//                    needed to compute it. Implemented here, for real.
//   release        — not implementable. Freeing the graph allocator's buffers
//                    requires handing the scheduler a replacement allocator:
//                    `sched->galloc` is a private field and upstream offers no
//                    way to assign it. Reported as unavailable with the reason,
//                    never as a silent no-op.
//   decode phase   — not implementable, and not a loss: upstream has no consumer
//                    for a phase label, so storing one would change nothing.
//                    A stored-but-unread field is the exact kind of "it ran"
//                    claim this project refuses.
//
// The scheduler for a context is learned rather than assumed: the fork read
// ctx->sched because it owned the struct. Here the two ggml scheduler entry
// points are hooked and record which context was executing, using a claim the
// caller opens around its own decode. Association by call site beats guessing by
// layout.
//
// The binding itself lives in herd_phase_station.hpp — a station that owns the
// one invariant that matters (at most one scheduler per context, and no binding
// that outlives its context) plus the capacity limit, and that never waits. This
// header is the wiring-facing surface only.
#pragma once

#include <cstdint>

struct llama_context;

namespace herd::phase {

// Installs the scheduler-capture seams. Aborts if a seam cannot be installed:
// a probe that cannot see the scheduler must not pretend compute accounting
// works. Idempotent.
void install();

// Tells the capture seams which context is executing. The caller opens this
// around its own decode call; it is not inferred, because inference would need
// the struct layout upstream owns. Nesting is tolerated and ignored beyond the
// outermost scope, so an eval callback that decodes cannot mis-attribute a
// scheduler to the wrong context.
class scope_guard {
public:
    explicit scope_guard(const llama_context * ctx);
    ~scope_guard();
    scope_guard(const scope_guard &) = delete;
    scope_guard & operator=(const scope_guard &) = delete;

private:
    const llama_context * ctx_;
};

// Bytes currently owned by the context's graph allocator, counted once per
// distinct buffer type — the same rule the fork used, computed through public
// accessors only. Returns false when this context's scheduler has not been seen
// yet (no decode has run), which is different from "zero bytes": a caller that
// printed 0 there would report a measurement it never took.
bool compute_bytes(const llama_context * ctx, std::uint64_t * out);

// The scheduler this context was seen decoding on, or null. Exposed for the
// release experiment: the allocator station is keyed by scheduler, and the mapping
// from a context to its scheduler is the only thing this module knows that the
// caller cannot. Returning null means "no decode seen yet", which is also why the
// probe reports an unmeasured state instead of inventing one.
void * scheduler_of(const llama_context * ctx);

// Drops the context's compute buffers and re-arms the scheduler's reservation.
//
// `embeddings_enabled` is the value this context was created with (llama_context_params).
// It cannot be read back -- upstream has no getter for that field -- and it is not optional:
// the re-arm is a double llama_set_embeddings() toggle, so a wrong value would leave the
// context with embeddings INVERTED, which changes what every later decode returns. The
// caller created the context, so the caller owns the fact.
//
// True only when buffers were actually returned. False means nothing happened, and
// release_refusal_reason() says why in the caller's terms.
bool release_compute(const llama_context * ctx, bool embeddings_enabled);

// Why the most recent release_compute() on THIS THREAD returned false. null when the last
// one succeeded. Thread-local: a refusal belongs to the call that hit it, not to the
// process, and a shared string would report one thread's GPU refusal to another thread's
// CPU context.
const char * release_refusal_reason();

// What a release request is decided on.
//
// Separated from the gathering so the decision is testable without a device of each
// kind: a machine with one GPU cannot produce a two-buffer-type scheduler, which is
// precisely the configuration the rule exists to refuse. A rule that can only be
// exercised on the hardware it is meant to protect is a rule nobody checks.
//
// The facts are what the decision depends on, and nothing else. In particular the
// number of BACKENDS is here only to catch "no backend at all"; it is not a safety
// condition, and the version this replaces confused the two:
//
//   it refused `n_backends > 1` while claiming "release is measured only for one
//   buffer type". Those are different predicates. One CUDA backend passes the old
//   test and was never measured; two CPU backends sharing one buffer type failed the
//   old test even though the retry path treats them identically to one.
//
// The mechanism is per buffer type: ggml_backend_sched_alloc_splits re-reserves after
// a failed alloc_graph, and with a single type that retry re-creates the one
// allocator this station freed. With several types the same retry has to satisfy all
// of them at once, and a partial re-reservation is a decode failure.
struct release_facts {
    bool scheduler_known   = false;  // a scheduler was recorded for this context
    int  n_backends        = 0;
    int  n_buffer_types    = 0;      // distinct buffer types across those backends
    bool all_backends_cpu  = true;
    // A slot the scheduler reported as null. Counted rather than skipped: the gather loop
    // used to `continue` past one, which left `all_backends_cpu` true and `n_buffer_types`
    // at whatever the other slots said -- so an incomplete picture of a scheduler read as a
    // complete one, and the release was allowed on facts that were not all observed. A
    // safety rule must not be satisfiable by the part of the input it failed to read.
    int  n_null_backends   = 0;
    int  n_null_buffer_types = 0;
};

struct release_verdict {
    bool         allowed = false;
    const char * reason  = nullptr;   // null only when allowed
};

// Pure. No device query, no allocation, no state.
release_verdict decide_release(const release_facts & facts);

// Can this build honour --phase-memory? True: the seam is linked, the mechanism is
// measured, and the per-run limits are now REFUSALS rather than an absence of code --
// release_compute() declines a multi-backend scheduler by name (see the reason), so a
// GPU run reports `release_compute_success:false` with a cause instead of quietly
// returning memory that was never freed.
bool release_available();
const char * release_unavailable_reason();

// Has the release MECHANISM been measured safe? True, on CPU with one buffer type. A
// separate question from the one above, and deliberately a separate entry point: when
// one flag answered both, the JSON reported the feature available while the same run
// refused it at the argument. Report this one as evidence, never as availability.
bool release_experiment_verified();
const char * release_experiment_note();

} // namespace herd::phase
