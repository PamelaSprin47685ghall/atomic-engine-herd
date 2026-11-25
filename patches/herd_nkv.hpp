// herd/nkv.hpp — decoupled KV capacity, requested per context and nothing else.
//
// The mechanism (patches/herd_nkv.cpp): hook llama_model::create_memory, hand it a
// copied cparams with a larger n_ctx_seq. The context keeps its own copy of cparams,
// and llama_n_ctx()/n_ctx_seq(), decode bounds and get_rope_factors all read that one,
// so the cell pool grows while the logical window does not move. Measured on the tiny
// transformer: kv_bytes 98304 -> 196608 with n_ctx and n_ctx_seq unchanged in both
// runs, and logits byte-identical.
//
// What changed and why: the first version read a process-wide environment variable.
// That is wrong in three ways, and each has a concrete failure attached:
//
//   1. It cannot say WHICH context. A server creates several, and an environment
//      variable applies to all of them, including ones that must stay ordinary.
//   2. It is invisible at the call site. A reader of `llama_init_from_model(model,
//      params)` cannot tell whether a hook will rewrite the capacity underneath.
//   3. It cannot report success. The caller had to re-derive whether the request took
//      effect, and every way of guessing looks the same as the request being ignored.
//
// So the request is a scope: open it immediately around one init call, on one thread,
// and ask it afterwards what happened. Nothing global, nothing inherited by a later
// context, and no way to run decoupled "by accident".
#pragma once

#include <cstdint>

namespace herd::nkv {

// Why a request was not applied. Named, because the driver's message has to tell the
// operator which one happened: "the cache declined" covers a split shape, a recurrent
// model, an out-of-range number and a request below the window, and those have four
// different fixes.
//
// The list is the fork's own gate (llama-context.cpp:955-980), plus the two cases this
// port discovers for itself:
//
//   below_window  requested <= n_ctx_seq: nothing to widen, not an error (the window
//                 already provides that many cells)
//   split_shape   !kv_unified && n_seq_max > 1: n_ctx_seq is the PER-SEQUENCE window
//                 there, so widening it widens every sequence's window
//   recurrent     the model's memory keeps one ring of states per sequence, sized by
//                 n_ctx_seq; that ring IS the sequence window, so a wider one is not
//                 decoupled capacity -- it is a bigger ring nothing may address, which
//                 is how a mamba model once reported kv_decoupled:true (measured)
//   exceeds_range  requested > INT32_MAX - 255, the fork's own bound (:958)
//   exhausted     the epoch space of the reservation ledger (not used here)
enum class refusal {
    none,
    below_window,
    split_shape,
    recurrent,
    exceeds_range,
    // The model's memory is not a plain KV cell pool: hybrid, or the DEEPSEEK4 family.
    // Distinct from `recurrent` because the message differs -- a recurrent model has a
    // ring of states, a hybrid model has a cell pool that is only one of the things sized
    // by the context length, and DEEPSEEK4 has its own memory shape. Three different
    // explanations of the same "no", and a caller deciding what to do next needs the right
    // one.
    unsupported_shape,
    // The capacity passed every gate and the memory module still could not be built:
    // create_memory returned null, or it threw. Its own reason because the caller's
    // next move differs from every other case here -- the request was legal and the
    // system could not deliver it, so the answer is fewer cells, a smaller model, or
    // more memory, not a different flag.
    allocation_failed,
};

const char * name(refusal why);

// Opens a request for the context that this thread is about to create.
//
// `physical_cells` is the total KV capacity to hand the memory module. It must be
// larger than the context's own window to mean anything; a smaller or equal value is
// recorded as "not applied" rather than clamped, because a caller asking for less
// capacity has made a different mistake and should hear about it.
//
// Nesting is refused: two live requests on one thread would make "which one applied"
// ambiguous, and there is no use for it -- the scope exists for the span of a single
// init call.
class scope {
public:
    explicit scope(std::uint32_t physical_cells);
    ~scope();

    // Ends the request and captures what the hook did. Must be called after the init
    // call; the destructor calls it too, so an early throw still closes the scope.
    // Reading the accessors before close() would report the empty initial state, which is
    // why it is explicit rather than destructor-only: the caller needs the answer while it
    // can still act on it.
    void close();

    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;

    // True when the hook consumed this request, which requires that it ran at all.
    bool consumed() const { return consumed_; }

    // The capacity the memory module was actually given, AFTER the hook's padding.
    // A request of 300 becomes 512 (the fork's own GGML_PAD(…, 256), :974), and a caller
    // that reported or admitted on 300 would be naming a number no buffer has.
    std::uint32_t applied_cells() const { return applied_cells_; }

    // True when the request took effect, i.e. the cache was sized from the request
    // rather than from the context's own window.
    bool applied() const { return applied_; }

    // Why the request was not applied, when it was not. `none` means it was applied.
    //
    // Nothing here is a silent fallback: the driver must turn every value except `none`
    // into a refusal of its own, because a run that asked for capacity it did not get is
    // a different run than the one the operator described.
    refusal reason() const { return reason_; }

    // How many create_memory calls this request was seen by.
    //
    // Must be exactly 1. The request names ONE context's capacity, and upstream calls
    // create_memory once per context (llama-context.cpp:602, the only call site, and the
    // only definition in that TU -- checked with nm, not assumed). So 2 means the caller
    // created two contexts while this scope was open, and the second one must not be
    // widened: it would be acting on a request nobody made for it.
    //
    // The hook refuses to widen it and says so on stderr; this counter is what lets a
    // driver turn that into an error of its own, and what lets a test assert the isolation
    // instead of trusting the report.
    int consumptions() const { return consumptions_; }

private:
    std::uint32_t requested_ = 0;
    bool          consumed_  = false;
    bool          applied_   = false;
    std::uint32_t applied_cells_ = 0;
    refusal       reason_    = refusal::none;
    int           consumptions_ = 0;
    bool          opened_    = false;

    friend struct scope_access;
};

// The process-wide observations the probe prints. Counters, not configuration: nothing
// here can change what a context does.
//
// `logical_window_seen` / `physical_cells_used` describe the most recent create_memory
// call; `calls` counts them; `shape_refusals` counts requests declined because the
// cache cannot express them (see below); `applied_requests` counts the ones honoured.
std::uint32_t logical_window_seen();
std::uint32_t physical_cells_used();
int           calls();
int           shape_refusals();
int           applied_requests();

} // namespace herd::nkv
