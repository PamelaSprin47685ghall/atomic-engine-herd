// herd/nkv_decision.hpp — the decoupled-capacity ruling, separated from the hook.
//
// Why this exists. The three failure paths that matter for honesty — the memory module
// returning null, throwing, or never being asked — all live behind a call to upstream
// (`g_orig`) that a unit test cannot replace. The decision that precedes that call is
// pure, but it was written inline between the hook's globals, so the only way to test
// "a failed build must not be reported as applied" was to make the allocator fail on a
// real model. That is not testable, and the consequence was measured: an earlier version
// set `applied = true` *before* the call, and nothing caught it.
//
// So the ruling is a value in, a value out, and the recording of the outcome is a second
// value in, a mutation out. The hook becomes a translator: read the facts, apply the
// ruling, call upstream, record what happened. Behaviour is unchanged by construction —
// every branch below is the branch that was already there, in the same order, and the
// regression run compares the probe's output before and after.
//
// Nothing here names a llama type. The facts are numbers and booleans the hook already
// has (`cparams.n_ctx_seq`, `cparams.n_seq_max`, the arch classification), so the ruling
// can be tested without a model, a cache, or a GGUF.
#pragma once

#include "herd_nkv.hpp"

#include <cstdint>

namespace herd::nkv {

// What the cache asked for, and what the call site knows about this context.
struct call_facts {
    std::uint32_t requested     = 0;   // the scope's request, before padding
    std::uint32_t window        = 0;   // cparams.n_ctx_seq, the per-sequence window
    std::uint32_t n_seq_max     = 1;
    bool          kv_unified    = true;
    // From patches/herd_model_arch.cpp: a plain KV cell pool is the only shape whose
    // capacity can be decoupled from its window.
    bool          memory_supports_decoupled = true;
    // The arch classification, when it said no — to name the reason correctly.
    bool          recurrent_shape = false;
    // A previous create_memory call already consumed this request (one scope, two
    // contexts: a caller bug, and the second context must not be widened).
    bool          already_consumed = false;
};

// What to do. One of four, and the difference between `refuse` and `refuse_duplicate`
// is which counter moves — the duplicate is not a ruling about capacity at all, it is a
// ruling about the caller.
struct ruling {
    enum class action {
        widen,            // pass the original a copy with the padded capacity
        refuse,           // pass the original the caller's own cparams, and say why
        refuse_duplicate, // a second context under one request: no widening, counted
    };

    action        act   = action::refuse;
    refusal       why   = refusal::none;
    std::uint32_t cells = 0;   // widen: the padded target. We keep the window separately.
    std::uint32_t window = 0;  // every non-widen outcome reports this, never the request
};

// The decision. Pure: no globals, no logging, no call into upstream.
ruling decide(const call_facts & facts);

// What happened when the ruling was executed. The hook cannot know this before it calls.
enum class outcome {
    built,         // upstream returned a memory module
    returned_null, // upstream returned nullptr
    threw,         // upstream raised (its constructors do, for configurations they refuse)
    not_called,    // no upstream call was made: a refusal or a duplicate
};

// The per-request record, mutated in place. Kept as a struct rather than as accessors on
// the scope so the same recording runs in the unit test and in the hook — a second copy
// of these rules is how the pre-call `applied = true` survived a review.
struct request_state {
    bool          consumed     = false;
    bool          applied      = false;
    std::uint32_t applied_cells = 0;
    refusal       reason       = refusal::none;
    int           consumptions = 0;
};

// Applies the ruling and its outcome to the record.
//
// The invariant this function exists to keep: `applied` is true only for
// `widen` + `built`. Every other combination leaves it false, whatever the outcome — a
// request that was refused, or that upstream failed to deliver, must never read as a
// capacity the process holds.
void record(request_state & state, const ruling & decision, outcome result);

} // namespace herd::nkv
