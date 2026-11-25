// herd/prefab_split.hpp — the reservation the padding lane acts on.
//
// One header for two translation units (the seam and its probe) because a class
// declared twice with different layouts is not a style problem: the linker keeps one
// set of member offsets and the other translation unit writes through them. The
// earlier draft did exactly that.
#pragma once

#include <cstdint>

#include <llama.h>

namespace herd::prefab {

// A reservation for pad rows: which sequence owns them, and how wide the envelope is.
//
// A scope, not an environment variable, for the reason the NKV lane had to learn: a
// process-wide switch cannot say which context, cannot be seen at the call site, and
// cannot report whether it took effect.
//
// NOT A PRODUCTION CAPABILITY YET, and the reasons are properties of the mechanism, not
// of the implementation's maturity. A driver must satisfy all three before this lane can
// be turned on outside an experiment:
//
//   1. ONE CONTEXT PER RESERVATION, ON ONE THREAD. The seam cannot see the decoding
//      context (llama_batch_allocr carries none); it learns it from the decode entry hook,
//      which is per-thread. A reservation held across two contexts on one thread is
//      detected and refused, but a reservation opened on thread A while the decode runs on
//      thread B is simply invisible: the lane stays off, which is safe. What is NOT
//      detected is a caller that opens a reservation on the thread that decodes and then
//      keeps it open while a *different* thread decodes the same context -- upstream does
//      not support that either, and nothing here can check it.
//   2. SINGLE KV STREAM. With n_stream > 1, llama-kv-cache.cpp:1013-1027 asserts
//      `n_tokens % n_seqs_unq == 0` and that the first row of every sequence group carries
//      that group's seq id; pad rows carry the scratch id, so the padded ubatch aborts.
//      `split_mode` is a MODEL parameter (include/llama.h) and no seam can see it, so a
//      driver must gate on LLAMA_SPLIT_MODE_NONE itself.
//   3. FREE CELLS FOR THE ENVELOPE. find_slot will evict a live SWA-masked cell to place a
//      pad row (llama-kv-cache.cpp:1082-1096), destroying a real sequence's history that a
//      later decode still attends to. The seam cannot see the cache; a driver must compare
//      pad_rows() against the free cells it measured before trusting a run.
//
// None of the three is a missing feature to be added later inside this class -- they are
// boundaries of what an out-of-tree seam can observe. Until a driver answers all three,
// the experiment stays opt-in (HERD_PREFAB_SPLIT) and no product links it.
class scratch_scope {
public:
    // `memory` is the context's memory module (llama_get_memory). Passing it is what
    // makes the reservation recoverable: see the destructor.
    scratch_scope(llama_memory_t memory, llama_seq_id scratch, uint32_t envelope);
    ~scratch_scope();

    // Clears the scratch sequence. Called by the destructor, and callable directly so a
    // caller can settle a decode before it frees the context.
    //
    // Why this lives here and not in a comment saying "the caller must": the pad rows are
    // cells in a real cache, and a decode that fails -- a full cache, an abort, an
    // exception out of the graph builder -- returns without anyone remembering to release
    // them. One leaked cell per pad row per ubatch, and the symptom is a decode failing
    // much later with nothing pointing at this code. The destructor is the only place that
    // runs on every path, including a stack unwind.
    //
    // Idempotent: a second call finds the sequence empty and reports success.
    bool release();

    // True once release() has run (and succeeded). A failed release leaves this false,
    // because the cells are still there and saying otherwise is the quiet leak.
    bool released() const { return released_; }

    // True once release() has been ATTEMPTED, whatever its outcome.
    //
    // The distinction from released() is what the seam reads. After an attempt this
    // reservation is over: a successful one has given its cells back and any further pad
    // row would be written into a sequence nothing will ever clear (the destructor's
    // release() short-circuits on `released_`), and a failed one has already been told it
    // could not clear, so writing more rows into that sequence only makes the shortage
    // worse. Either way the lane must stop, and "stop" has to be a state rather than a
    // caller's memory of having called release().
    bool finished() const { return finished_; }

    // True when this reservation may be used: the sequence was empty when the scope
    // opened.
    //
    // False means the reservation was refused because the sequence already held cells.
    // It is not merely "not yet usable" -- nothing can make an occupied sequence clean
    // except destroying whoever put it there, and that is exactly what a reservation
    // must never do implicitly.
    //
    // A refused scope does not install itself as the seam's scope, so no ubatch is
    // padded and release() touches nothing. The lane is simply absent for this run,
    // which is the only outcome that is safe without knowing what occupies the
    // sequence.
    bool usable() const { return opened_ && !occupied_at_open_; }
    bool occupied_at_open() const { return occupied_at_open_; }

    // The memory module this reservation was opened for.
    //
    // Exposed because of a limit that cannot be fixed inside this class, and stating it
    // is the honest alternative to a guard that cannot exist. The seam that pads ubatches
    // receives a `llama_batch_allocr`, which carries no context and no memory handle
    // (llama-batch.h:167 -- n_pos_per_embd and allow_mixed, nothing else), so it cannot
    // ask which context is decoding. Therefore:
    //
    //   one reservation belongs to ONE context, and must be destroyed before another
    //   context decodes on the same thread.
    //
    // If a caller keeps a reservation open across two contexts, the second context's
    // ubatches are padded into the scratch sequence of ITS OWN cache -- and the
    // emptiness check at open time was made against the first context's cache, so it says
    // nothing about the second. That is the same data-destruction failure the constructor
    // now refuses within one context, reached by a route the constructor cannot see.
    //
    // `memory()` exists so a caller can check the obvious half of this (am I still
    // releasing the context I opened for?) instead of trusting a comment.
    llama_memory_t memory() const { return memory_; }

    scratch_scope(const scratch_scope &) = delete;
    scratch_scope & operator=(const scratch_scope &) = delete;

    // Padding as the SEAM saw it: ubatches widened, and rows added.
    //
    // These are ATTEMPTS, and the distinction is not pedantry. A ubatch is widened
    // inside llama_batch_allocr::split_simple, which runs while the batch is being
    // split -- before the cache has had any say. If the decode is then refused (a full
    // cache, a bad shape), the widening has already happened and is already counted.
    // So a run that hits the cache limit reports one more padded ubatch than it
    // executed: true about what was submitted, false about what ran, and wrong by
    // exactly one envelope's worth of rows.
    int           padded_ubatches() const { return padded_ubatches_; }
    std::uint64_t pad_rows()        const { return pad_rows_; }

    // Padding as the CALLER observed it: rows that reached a decode which returned 0.
    //
    // Only the caller can know this -- the seam returns before the cache gives its
    // verdict -- so it is recorded by note_settled() rather than inferred here. A probe
    // that compares the two counters is measuring the seam's honesty: attempts minus
    // settlements must equal exactly the rows submitted for the ubatch that failed.
    int           settled_ubatches()  const { return settled_ubatches_; }
    std::uint64_t settled_pad_rows()  const { return settled_pad_rows_; }
    void          note_settled(std::uint32_t n_env, std::uint32_t n_real);

    int          declined()       const { return declined_; }
    const char * last_decline()   const { return last_decline_; }

    // THE CALLER'S OBLIGATION, and it is not optional.
    //
    // Opening a scope is a request to pad this context's ubatches. Three counters say
    // whether the request was met, and a driver that reports a run without reading them is
    // reporting a configuration it did not verify:
    //
    //   padded_ubatches()  ubatches actually widened -- the only positive evidence;
    //   declined()         ubatches turned away for their SHAPE (packed, embeddings,
    //                      M-RoPE, or an allocation that could not be built);
    //   binding_refusals() ubatches refused because another context was decoding.
    //
    // `declined() > 0` with `padded_ubatches() == 0` means the run was not padded at all:
    // the destructor prints one line in that case, and a caller that captures stderr still
    // has to check the counters. Declining is the safe outcome (the ubatch is split exactly
    // as upstream split it, and the output is correct), which is precisely why nothing
    // downstream will notice that the requested optimisation did not happen.
    //
    // A single ubatch's allocation failure is not a failed run -- that was this file's own
    // earlier defect, where a std::bad_alloc escaping the seam terminated the process. It is
    // a failed REQUEST when nothing at all was padded, and only the caller can decide
    // whether that matters for its purpose.
    //
    // `request_unmet()` is that ruling, as a query. The destructor reports the same
    // condition on stderr; exposing it means a test can assert the judgement without
    // capturing stderr, so the report cannot rot untested.
    bool request_unmet() const {
        return usable() && padded_ubatches_ == 0 && (declined_ + binding_refusals_) > 0;
    }

    // Ubatch split attempts refused because the decode in progress belonged to a
    // DIFFERENT context than this reservation was opened for.
    //
    // This is the cross-context risk, closed. `llama_batch_allocr` carries no context
    // and no memory handle (llama-batch.h:167), so the seam cannot ask "whose ubatch is
    // this?" -- but the decode entry point can, and does: both `llama_context::decode`
    // overloads are hooked to publish the decoding context's memory module for the
    // duration of the call, and the seam pads only when that matches `memory()`.
    //
    // Counted separately from `declined()` because the two mean different things to a
    // caller: a shape decline says "this ubatch is not paddable", a binding refusal says
    // "this ubatch is not yours". The second one is a caller error that would otherwise
    // destroy data in the other context's cache, so a run that reports it should be read
    // as a bug in the driver, not as a lane that found nothing to do.
    int binding_refusals() const { return binding_refusals_; }

    // Read by the seam, which is not a member: the reservation and the width are
    // facts it acts on, not state it owns.
    uint32_t     envelope() const { return envelope_; }
    llama_seq_id sequence() const { return scratch_; }

    void note_padded(uint32_t n_env, uint32_t n_real);
    void note_declined(const char * why);

private:
    llama_memory_t memory_ = nullptr;
    llama_seq_id  scratch_  = -1;
    uint32_t      envelope_ = 0;
    bool          opened_   = false;
    bool          released_ = false;
    bool          finished_ = false;
    // Set when the sequence already held cells as the scope opened. Read by the seam
    // through usable(); also read by release(), which must not clear cells it did not
    // create.
    bool          occupied_at_open_ = false;
    int           padded_ubatches_ = 0;
    std::uint64_t pad_rows_ = 0;
    int           settled_ubatches_ = 0;
    std::uint64_t settled_pad_rows_ = 0;
    int           declined_ = 0;
    int           binding_refusals_ = 0;
    const char *  last_decline_ = "";

    // The seam's entry point for the binding refusal; not public, because no caller has
    // a use for "record that someone else decoded" other than the seam itself.
    friend void prefab_note_binding_refusal(scratch_scope &);
};

} // namespace herd::prefab
