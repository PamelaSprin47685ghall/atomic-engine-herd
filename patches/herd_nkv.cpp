// herd/nkv.cpp — decoupled KV capacity by handing create_memory a different cparams.
//
// The mechanism, unchanged from the experiment that established it: hook
// llama_model::create_memory, copy the cparams it was given, raise n_ctx_seq, pass the
// copy. Upstream consumes that number twice, at two moments, and only the second one
// sees the copy:
//
//   the context keeps its own cparams  -> llama_n_ctx()/n_ctx_seq(), decode bounds,
//                                         get_rope_factors: the WINDOW
//   create_memory(model, params, cp)   -> cache/tensor sizing only: the CELLS
//
// The mask follows cells in use, not capacity (get_n_kv() = min(cells.size(),
// PAD(used_max_p1, 256))), so spare cells cost KV memory and nothing else. That is why
// the fork's n_ctx_kv can be reproduced without adding a field.
//
// MEASURED (tiny transformer, CPU, one variable at a time):
//
//   control        n_ctx=256 n_ctx_seq=256 kv_bytes=98304   2nd sequence rc=1 (no cells)
//   physical=512   n_ctx=256 n_ctx_seq=256 kv_bytes=196608  2nd sequence rc=0
//   logits byte-identical (raw dump + FNV digest), a decode past the window refused in
//   both. kv_bytes comes from llama_memory_i::memory_breakdown(), i.e. from the module
//   that allocated the buffers, not from this layer echoing its own request.
//
// SAFETY, verified rather than assumed: create_memory passes cparams.n_ctx_seq BY VALUE
// into the cache constructor (uint32_t kv_size), and llama_kv_cache holds no reference
// to cparams (the type is forward-declared in llama-kv-cache.h). Passing a stack copy is
// therefore sound; the one arrangement that would make this a use-after-return is not
// the one upstream uses. A pin bump that changes either fact fails to compile here,
// because this file includes the private headers.
//
// WHAT THIS FILE IS NOT: a default. The hook is inert unless a herd::nkv::scope is open
// on the calling thread, and it is linked into products only so that a driver can open
// that scope deliberately (batch-compose does, for --total-kv). Nothing here reads the
// environment, so no configuration can leak from one context to the next.
#include "herd_patch.hpp"
#include "herd_nkv.hpp"
// The model's memory shape, from upstream's own arch declaration rather than from a
// membership predicate that merely happens to list it. See herd_model_arch.hpp.
#include "herd_model_arch.hpp"
// The ruling and its recording. The hook translates: facts in, ruling out, call, record.
#include "herd_nkv_decision.hpp"

#include <llama.h>

// The two types the hooked signature names. Upstream's public header does not declare
// them (they live in src/), so this includes the private headers it needs — read, never
// edited. Same trade as the prefab seam with llama-batch.h: a pin bump that moves a
// field fails this build, which is the wanted direction.
#include "llama-cparams.h"
#include "llama-memory.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

// llama_model::create_memory(llama_memory_params const&, llama_cparams const&) const
// Read from the built archive with nm, not written from the signature: it is a const
// member of a class declared in a private header, and a pin bump that changes the
// mangling must fail loudly here instead of hooking the wrong symbol.
constexpr const char * kSymbol =
    "_ZNK11llama_model13create_memoryERK19llama_memory_paramsRK13llama_cparams";

using create_memory_fn = llama_memory_i * (*)(const llama_model *,
                                              const llama_memory_params &,
                                              const llama_cparams &);

create_memory_fn g_orig = nullptr;

// The request in flight on this thread. Thread-local because create_memory runs inside
// the caller's init call: two threads creating contexts at once must not see each
// other's request, and a lock would serialise context creation for no benefit.
struct pending_request {
    bool          open    = false;
    std::uint32_t request = 0;
    // The record is herd::nkv::request_state -- the same struct the unit test mutates, so
    // there is one definition of what "applied / refused / counted" means rather than two
    // that drift. The fields it used to duplicate are now inside it.
    herd::nkv::request_state state;
};

thread_local pending_request t_pending;

// Process-wide observations. Counters only; nothing here can change what a context does.
std::atomic<std::uint32_t> g_logical_seen{0};
std::atomic<std::uint32_t> g_physical_used{0};
std::atomic<int>           g_calls{0};
std::atomic<int>           g_shape_refusals{0};
std::atomic<int>           g_applied{0};

llama_memory_i * herd_create_memory(const llama_model * model,
                                    const llama_memory_params & params,
                                    const llama_cparams & cparams) {
    ++g_calls;
    g_logical_seen = cparams.n_ctx_seq;

    if (!t_pending.open) {
        // Inert, and that is the common case: no request, upstream's own arguments.
        g_physical_used = cparams.n_ctx_seq;
        return g_orig(model, params, cparams);
    }

    // A request is single-use, and this is where that becomes structure rather than a
    // sentence in the header.
    //
    // Upstream calls create_memory once per context, from one call site, with one
    // definition in its TU (verified with nm as well as grep), so a second call while this
    // scope is open means the caller created a SECOND context under one request. The
    // capacity in the request names one context; applying it to the second would widen a
    // context nobody asked to widen, which is the per-context isolation this class exists
    // for, broken.
    //
    // Two failures are possible here and both are wrong: widening the second context
    // (acting on a request not made for it), and passing it through in silence (a
    // configuration that did not apply, with nothing said). So: refuse to widen, record
    // the count, and say it on stderr *unconditionally* -- this path is reachable only by
    // a caller bug, so it cannot pollute a correct run's output.
    pending_request &         slot  = t_pending;
    herd::nkv::request_state & state = slot.state;

    // Facts for the ruling, and nothing more. Every `if` that used to follow now lives in
    // herd_nkv_decision.cpp: this function observes, asks, calls, and records.
    herd::nkv::call_facts facts;
    facts.requested        = slot.request;
    facts.window           = cparams.n_ctx_seq;
    facts.n_seq_max        = cparams.n_seq_max;
    facts.kv_unified       = cparams.kv_unified;
    facts.already_consumed = state.consumed;

    const herd::arch::shape memory_shape = herd::arch::shape_of(model);
    facts.recurrent_shape           = memory_shape == herd::arch::shape::recurrent;
    facts.memory_supports_decoupled = herd::arch::supports_decoupled_capacity(memory_shape);

    const herd::nkv::ruling decision = herd::nkv::decide(facts);

    // A second context under one request. Not a ruling about capacity, so `record` only
    // counts it and leaves the first call's answer standing -- that is the context the
    // request named.
    if (decision.act == herd::nkv::ruling::action::refuse_duplicate) {
        herd::nkv::record(state, decision, herd::nkv::outcome::not_called);
        g_physical_used = cparams.n_ctx_seq;
        std::fprintf(stderr,
                     "[herd] nkv: a scoped request was consumed by a second "
                     "create_memory call; the second context is NOT widened. One scope "
                     "belongs to one context -- open a scope per init call.\n");
        return g_orig(model, params, cparams);
    }

    // Every refusal, with the window as the capacity it reports -- never the request,
    // which names cells the process does not hold.
    if (decision.act == herd::nkv::ruling::action::refuse) {
        if (decision.why == herd::nkv::refusal::split_shape) {
            ++g_shape_refusals;
        }
        herd::nkv::record(state, decision, herd::nkv::outcome::not_called);
        g_physical_used = state.applied_cells;
        return g_orig(model, params, cparams);
    }

    // The gates are not restated here. They live in herd_nkv_decision.cpp -- one copy,
    // called from one place -- because a second copy is how `applied = true` survived a
    // review once already: the duplicate read as "the hook's version" and nobody compared
    // the two. The order below is the ruler's: arch, range, below-window, split.

    // Order: arch, range, below-window, split -- all inside decide().

    // The copy. Everything but the capacity is upstream's own value, including
    // n_seq_max and kv_unified: this changes one number -- the ruling's padded target.
    //
    // decide() pads to 256 exactly as the fork did (:974). Upstream pads n_ctx the same
    // way, and a cache size that is not a multiple of the pad is the case the fork's own
    // comment says only holds on the CPU path.
    llama_cparams widened = cparams;
    widened.n_ctx_seq = decision.cells;

    // The call comes first, the claim second -- because the claim is about what the call
    // did, not about what was attempted.
    //
    // The earlier version set applied / applied_cells / ++g_applied *before* invoking the
    // original. Two paths then produced a false claim:
    //
    //   * create_memory returns null -- the memory module could not be built. Before this
    //     fix the scope reported applied == true and a capacity the process does not hold;
    //   * create_memory throws -- upstream's own constructors raise std::runtime_error on
    //     configurations they refuse (the fork's n_ctx_kv checks at llama-context.cpp:956
    //     and :965-967 do exactly that). The exception travels out through
    //     llama_init_from_model, the scope's destructor copies applied == true on the way
    //     out, and the caller reads "applied" for a context that was never created.
    //
    // A driver that checks the scope after a failed init would then be told the capacity
    // was honoured. That is the one thing this class must never say wrongly: every other
    // answer it gives is "refused, and here is why".
    llama_memory_i * created = nullptr;
    try {
        created = g_orig(model, params, widened);
    } catch (...) {
        herd::nkv::record(state, decision, herd::nkv::outcome::threw);
        g_physical_used = state.applied_cells;
        throw;
    }
    if (!created) {
        herd::nkv::record(state, decision, herd::nkv::outcome::returned_null);
        g_physical_used = state.applied_cells;
        return nullptr;
    }

    herd::nkv::record(state, decision, herd::nkv::outcome::built);
    ++g_applied;
    g_physical_used = state.applied_cells;
    return created;
}

// Exit report, silent unless tracing: a product's stderr is part of its output contract.
struct nkv_report {
    ~nkv_report() {
        const int calls = g_calls.load();
        if (std::getenv("HERD_SEAM_TRACE") == nullptr || calls == 0) {
            return;
        }
        std::fprintf(stderr,
                     "[herd] nkv: create_memory called %d time(s), %d applied, %d refused by "
                     "shape; last logical window %u, last cells handed over %u\n",
                     calls, g_applied.load(), g_shape_refusals.load(),
                     g_logical_seen.load(), g_physical_used.load());
    }
} g_report;

} // namespace

namespace herd::nkv {

scope::scope(std::uint32_t physical_cells)
    : requested_(physical_cells) {
    if (t_pending.open) {
        // Two live requests on one thread would make "which one applied" ambiguous, and
        // there is no use for nesting: the scope exists for one init call.
        std::fprintf(stderr,
                     "[herd] nkv: a scoped request is already open on this thread; nesting "
                     "is refused because it makes the applied capacity ambiguous\n");
        std::abort();
    }
    t_pending = pending_request{};
    t_pending.open    = true;
    t_pending.request = physical_cells;
    opened_ = true;
}

scope::~scope() {
    close();
}

void scope::close() {
    if (!opened_) {
        return;
    }
    consumed_      = t_pending.state.consumed;
    applied_       = t_pending.state.applied;
    applied_cells_ = t_pending.state.applied_cells;
    reason_        = t_pending.state.reason;
    consumptions_  = t_pending.state.consumptions;
    t_pending      = pending_request{};
    opened_        = false;
}

// `name(refusal)` lives in herd_nkv_decision.cpp: it is a pure mapping and the ruling's
// test needs it without pulling llama into that target. One definition, one owner.

std::uint32_t logical_window_seen() { return g_logical_seen.load(); }
std::uint32_t physical_cells_used() { return g_physical_used.load(); }
int           calls() { return g_calls.load(); }
int           shape_refusals() { return g_shape_refusals.load(); }
int           applied_requests() { return g_applied.load(); }

} // namespace herd::nkv

#if !defined(HERD_NO_AUTOLOAD)
// Registered unconditionally: the hook is inert without a scope, and a registrar that
// only fires under an environment variable would make the product's behaviour depend on
// configuration again -- which is the thing this design removes.
__attribute__((constructor(101))) static void herd_nkv_autoload() {
    if (std::getenv("HERD_NO_AUTOLOAD") != nullptr) {
        return;
    }
    void * target = herd::resolve(kSymbol);
    if (!target) {
        // Not installed, and NOT fatal -- which is a change from the earlier version.
        //
        // This registrar is linked into every product (herd_seams is attached to all of
        // them), and only one of them uses the capability: batch-compose's --total-kv.
        // Aborting here made a pin bump that renamed the symbol break `llama-corpus
        // --help`, i.e. a startup failure in four binaries for a feature one of them
        // calls. The wrong blast radius, and it is avoidable without any silence:
        //
        // batch-compose already asks the scope what happened and refuses with a named
        // reason when the hook did not run ("the nkv hook never ran during context
        // creation; this build cannot honour --total-kv"). So the failure is reported at
        // the point of use, with the numbers attached, and only for the caller who asked.
        //
        // The trace line keeps the diagnosis visible for anyone who wants it. Under
        // HERD_SEAM_TRACE because a product's stderr is part of its output contract.
        if (std::getenv("HERD_SEAM_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "[herd] nkv: llama_model::create_memory was not found in this image; "
                         "the decoupled-KV hook is not installed, so --total-kv above the "
                         "context's window will be refused when it is asked for\n");
        }
        return;
    }
    static_cast<void>(herd::add("nkv.create_memory", target,
                                herd::raw(&herd_create_memory),
                                reinterpret_cast<void **>(&g_orig), nullptr));
}
#endif
