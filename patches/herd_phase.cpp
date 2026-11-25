// herd/phase.cpp — wiring for the phase station.
//
// Layering, per the ticket/station method:
//
//   station  (herd_phase_station.cpp)  owns the context -> scheduler binding
//   seam     (below)                   translates one upstream call into a
//                                      station call and nothing else
//   ticket   (the caller)              owns the waiting: llama_decode is the only
//                                      blocking step, and it lives in the driver
//
// The seam bodies here contain no policy: they take the scheduler upstream chose,
// hand it to the station, and defer. No seam waits, allocates, or decides.
#include "herd_phase.hpp"
#include "herd_buffer_station.hpp"

#include "herd_patch.hpp"
#include "herd_phase_station.hpp"

#include <ggml-backend.h>
#include <llama.h>

#include <cstdio>
#include <cstdlib>
#include <atomic>

namespace {

// The ticket announces itself: a guard is opened around the caller's decode, and
// the innermost open claim is the context being executed. A stack rather than a
// single value, because an eval callback can decode a second context inside the
// first; the innermost claim is the one whose scheduler is about to run, and
// attributing it to the outer context would bind the wrong pair.
constexpr int kMaxClaims = 8;
thread_local const llama_context * t_claims[kMaxClaims] = {};
thread_local int                   t_depth = 0;

const llama_context * current_claim() {
    // t_depth above the stack size means a claim was opened deeper than the stack
    // records; that call is unattributed (accounting stays unknown) rather than
    // attributed to an unrelated context.
    return (t_depth > 0 && t_depth <= kMaxClaims) ? t_claims[t_depth - 1] : nullptr;
}

ggml_status (*g_orig_compute)(ggml_backend_sched_t, struct ggml_cgraph *)      = nullptr;
ggml_status (*g_orig_compute_async)(ggml_backend_sched_t, struct ggml_cgraph *) = nullptr;
void        (*g_orig_free)(llama_context *)                                    = nullptr;
void        (*g_orig_sched_free)(ggml_backend_sched_t)                         = nullptr;

void observe(ggml_backend_sched_t sched) {
    const llama_context * claim = current_claim();
    if (!claim) {
        return;
    }
    const auto outcome = herd::phase::station().observe(claim, sched);
    if (outcome == herd::phase::scheduler_station::observation::refused_full) {
        // Reported once: a full table means the station stopped learning, which
        // silently turns accounting into "unknown" for every later context.
        //
        // Two decoders can reach this line at the same time, so the once-only
        // decision is an atomic exchange rather than a plain bool. A data race
        // here would be benign-looking UB whose worst case is a duplicated line,
        // which is exactly the kind of defect that is never worth keeping.
        static std::atomic<bool> warned{false};
        bool expected = false;
        if (warned.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr,
                         "[herd] phase: context table full (%d); further contexts are not tracked\n",
                         herd::phase::scheduler_station::capacity);
        }
    }
}

ggml_status herd_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    observe(sched);
    return g_orig_compute(sched, graph);
}

ggml_status herd_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    observe(sched);
    return g_orig_compute_async(sched, graph);
}

// Both compute entry points are hooked: upstream's synchronous one calls the
// async one, so a single decode reports the same pair twice and the second write
// is an idempotent update. Hooking only one would tie the seam to whichever entry
// point today's callers happen to use.
void herd_free(llama_context * ctx) {
    // The binding is dropped before upstream frees the scheduler. The other order
    // leaves an entry pointing at freed memory for the window between the two, and
    // a context allocated at the same address afterwards would inherit it: a
    // use-after-free that only surfaces when the allocator reuses the address.
    //
    // Order alone is not enough to make this safe, and the lock is what closes the
    // rest of the window: forget() takes the station's mutex, so a concurrent
    // compute_bytes() either finished reading the scheduler under that lock or has
    // not started, and after forget() returns the entry is gone. The scheduler
    // itself is freed after both.
    herd::phase::station().forget(ctx);
    g_orig_free(ctx);
}

// The other way a scheduler dies, and the one that made this hook necessary:
// llama_context::decode() rebuilds its scheduler in place on some paths
// (`sched.reset(); sched_reserve();` in the compact-inkling and stream-wide prompt
// branches), so a context outlives its scheduler and a context-keyed binding would
// dangle until the next decode re-recorded one. Dropping by scheduler identity
// makes the query in that window answer "unknown" instead of reading a freed
// object.
//
// No log line here on purpose: a scheduler rebuild is an upstream decision, and
// every later compute_bytes() already reports the consequence as "unknown" rather
// than a number. The seam stays a translator.
void herd_sched_free(ggml_backend_sched_t sched) {
    herd::phase::station().forget_scheduler(sched);
    g_orig_sched_free(sched);
}

// Counted once per distinct buffer type: several devices can share one allocator,
// and summing per backend would report the same bytes twice.
std::uint64_t bytes_of(ggml_backend_sched_t sched) {
    std::uint64_t total = 0;
    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n_backends; ++i) {
        const ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        const ggml_backend_buffer_type_t buft = ggml_backend_sched_get_buffer_type(sched, backend);
        bool seen = false;
        for (int j = 0; j < i && !seen; ++j) {
            const ggml_backend_t previous = ggml_backend_sched_get_backend(sched, j);
            seen = ggml_backend_sched_get_buffer_type(sched, previous) == buft;
        }
        if (!seen) {
            total += ggml_backend_sched_get_buffer_size(sched, backend);
        }
    }
    return total;
}

} // namespace

HERD_SEAM(ggml_sched_compute, "ggml_backend_sched_graph_compute",
          &ggml_backend_sched_graph_compute, &herd_compute, &g_orig_compute, nullptr)
HERD_SEAM(ggml_sched_compute_async, "ggml_backend_sched_graph_compute_async",
          &ggml_backend_sched_graph_compute_async, &herd_compute_async, &g_orig_compute_async, nullptr)
// The lifecycle end of the same fact: a binding outlives neither the context nor
// its scheduler, so the entry has to die with them.
HERD_SEAM(llama_free, "llama_free", &llama_free, &herd_free, &g_orig_free, nullptr)
// And the scheduler's own death, which also happens mid-context: decode() rebuilds
// it in place on some paths. Without this hook a context-keyed binding would point
// at a destroyed scheduler until the next decode re-recorded one.
HERD_SEAM(ggml_sched_free, "ggml_backend_sched_free",
          &ggml_backend_sched_free, &herd_sched_free, &g_orig_sched_free, nullptr)

namespace herd::phase {

// Why the most recent release on this thread was refused, or null. Thread-local so one
// thread's multi-device refusal cannot be reported as another thread's CPU answer.
thread_local const char * t_refusal = nullptr;

void install() {
    install_all();
}

scope_guard::scope_guard(const llama_context * ctx) : ctx_(ctx) {
    if (t_depth < kMaxClaims) {
        t_claims[t_depth] = ctx_;
    }
    ++t_depth;
}

scope_guard::~scope_guard() {
    --t_depth;
    if (t_depth >= 0 && t_depth < kMaxClaims) {
        t_claims[t_depth] = nullptr;
    }
}

bool compute_bytes(const llama_context * ctx, std::uint64_t * out) {
    if (!out) {
        return false;
    }
    // The station holds its lock for the whole callback, so the scheduler cannot
    // be freed between being found and being read.
    return station().with_scheduler(ctx, [out](ggml_backend_sched_t sched) {
        *out = bytes_of(sched);
    });
}

void * scheduler_of(const llama_context * ctx) {
    void * sched = nullptr;
    station().with_scheduler(ctx, [&sched](ggml_backend_sched_t value) {
        sched = static_cast<void *>(value);
    });
    return sched;
}

// The release path is present only when the build that contains it also contains the
// allocator hooks it needs.
//
// Six allocator entry points are hooked, and two of them (ggml_gallocr_get_buffer_size,
// ggml_gallocr_free) are short enough that a hook cannot land without a padded prologue --
// measured: without it the installer aborts with "DobbyHook failed". That prologue comes
// from compiling ggml-alloc.c with -fno-inline, which costs inlining in a hot translation
// unit for every build, used or not. So the whole path is one option (HERD_BUILD_BUFFER_HOOKS),
// and this file follows it instead of guessing: HERD_RELEASE_AVAILABLE is defined by the
// same block that adds the TU and the prologue.
//
// The consequence is deliberately boring: with the option off, the product compiles, links,
// refuses --phase-memory by name, and contains no reference to release_and_arm. With it on,
// the hooks are installed or the installer aborts -- a loud failure, never a half-patched
// binary that reports itself as memory-bounded.
#ifdef HERD_RELEASE_AVAILABLE

bool release_compute(const llama_context * ctx, bool embeddings_enabled) {
    t_refusal = nullptr;
    if (!ctx) {
        t_refusal = "no context";
        return false;
    }
    // The scheduler is learned from the capture seams, never from the struct: the context
    // may be one that never decoded, in which case there is no allocator to free and
    // saying so is different from "freed nothing".
    void * sched = scheduler_of(ctx);
    if (!sched) {
        t_refusal = "no decode has run on this context, so its scheduler has never been seen";
        return false;
    }
    // Gather what the decision needs, then let the pure rule decide.
    //
    // This used to be `if (get_n_backends(sched) > 1)`, under a comment promising that
    // the gate was "one buffer type". The two are different predicates and the code was
    // the wrong one:
    //
    //   * a single CUDA backend satisfies n_backends == 1 and was never measured;
    //   * two CPU backends sharing one buffer type were refused although the retry path
    //     in ggml_backend_sched_alloc_splits cannot tell them from one.
    //
    // So the facts are now reported honestly -- how many backends, how many DISTINCT
    // buffer types, and whether every one of them is a CPU device -- and decide_release()
    // applies the rule the mechanism actually rests on. CPU-only is a fail-closed choice,
    // not an inference from the backend count: the release is measured on CPU, and on a
    // GPU the synchronize-then-free ordering has never been observed on a real device.
    {
        auto * s = static_cast<ggml_backend_sched_t>(sched);
        release_facts facts;
        facts.scheduler_known = true;
        facts.n_backends = ggml_backend_sched_get_n_backends(s);
        for (int i = 0; i < facts.n_backends; ++i) {
            const ggml_backend_t backend = ggml_backend_sched_get_backend(s, i);
            if (!backend) {
                // Counted, not skipped. Skipping left the remaining facts looking clean --
                // all_backends_cpu true, n_buffer_types one -- so a scheduler this loop could
                // not fully read was handed to decide_release() as a verified single-type CPU
                // scheduler. The rule now refuses on the missing observation.
                ++facts.n_null_backends;
                continue;
            }
            // Typed, not a name string: "CPU" is a display label, and a BLAS or wrapper
            // backend may carry any name at all. The device type is the fact.
            const ggml_backend_dev_t dev = ggml_backend_get_device(backend);
            if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                facts.all_backends_cpu = false;
            }
            // Distinct buffer types, counted the way bytes_of() counts them: several
            // devices can share one allocator, and a per-backend sum would count it twice.
            const ggml_backend_buffer_type_t buft = ggml_backend_sched_get_buffer_type(s, backend);
            if (!buft) {
                // Same reasoning one level down: a backend whose buffer type cannot be
                // identified is not an allocator this station can name, free, or re-reserve.
                ++facts.n_null_buffer_types;
                continue;
            }
            bool seen = false;
            for (int j = 0; j < i && !seen; ++j) {
                const ggml_backend_t previous = ggml_backend_sched_get_backend(s, j);
                seen = previous && ggml_backend_sched_get_buffer_type(s, previous) == buft;
            }
            if (!seen) {
                ++facts.n_buffer_types;
            }
        }
        const release_verdict verdict = decide_release(facts);
        if (!verdict.allowed) {
            t_refusal = verdict.reason;
            return false;
        }
    }
    // release_and_arm synchronizes first, frees the allocator, then re-arms the reservation
    // through llama_set_embeddings. The order and the reason are in
    // patches/herd_phase_buffers.cpp; what matters here is that this call is the whole
    // mechanism, and that the const cast is honest: the context's scheduler state is being
    // deliberately mutated, and the API says so rather than pretending to be const.
    if (!herd::buffers::release_and_arm(const_cast<llama_context *>(ctx), embeddings_enabled)) {
        t_refusal = "the allocator freed nothing (no buffers were owned) or the re-arm failed";
        return false;
    }
    return true;
}

#else

bool release_compute(const llama_context *, bool) {
    // No release path in this image, and the reason names the option that would add it.
    // Returning false is not a fallback: the caller reports the refusal and the run
    // continues with its buffers intact, which is the correct behaviour for a build that
    // was asked to be small.
    t_refusal = "this build contains no release path (HERD_BUILD_BUFFER_HOOKS=OFF); the "
                "allocator hooks need a padded prologue in ggml-alloc.c, which that option adds";
    return false;
}

#endif

const char * release_refusal_reason() {
    return t_refusal;
}

// Has the mechanism itself been measured safe? That is release_experiment_verified(),
// and it is a different fact with a different owner. Conflating the two is exactly what
// went wrong here: this query answered true while the product still refused the flag at
// the argument, so a driver reading the JSON availability field reported a capability
// that the same run rejected. A capability and the evidence for the mechanism behind it
// are not the same claim, and a caller cannot tell them apart if one flag carries both.
//
bool release_available() {
#ifdef HERD_RELEASE_AVAILABLE
    // A fact read from the registry, not a constant, and the difference is the bug that
    // shipped: this used to return true while the TU implementing `release_and_arm` was
    // excluded from the link, so the product accepted --phase-memory and then failed to
    // build. A capability claim nothing can check is a wish.
    //
    // The names are the minimum the release actually touches:
    //
    //   ggml_gallocr_new_n               the only way to learn the allocator's handle
    //   ggml_gallocr_free                the free itself
    //   ggml_gallocr_alloc_graph         re-binding tensors to the fresh allocator
    //   ggml_backend_sched_reserve       the reservation that must run after re-arming
    //   ggml_backend_sched_graph_compute how the phase station learns which scheduler to free
    //
    // Checked one by one rather than "any seam is installed": a single failed symbol is the
    // half-installed state that makes the release answer true and then refuse every run.
    static const char * const required[] = {
        "ggml_gallocr_new_n",
        "ggml_gallocr_free",
        "ggml_gallocr_alloc_graph",
        "ggml_backend_sched_reserve",
        "ggml_backend_sched_graph_compute",
    };
    for (const char * name : required) {
        const herd::Seam * seam = herd::find(name);
        if (!seam || !seam->installed.load(std::memory_order_acquire)) {
            return false;
        }
    }
    return true;
#else
    // No release path in this image. The mechanism is measured (release_experiment_verified),
    // but a binary without the hooks cannot honour the request, and a capability answer that
    // described the mechanism instead of the binary is exactly the confusion this split was
    // made to end.
    return false;
#endif
}

const char * release_unavailable_reason() {
#ifdef HERD_RELEASE_AVAILABLE
    // Reached when release_compute() refuses at RUN time: the build is capable and this
    // configuration is not.
    return "this build can release compute memory; a run is refused when its scheduler has "
           "more than one backend, where the re-reservation was not measured";
#else
    return "this build contains no release path: --phase-memory is refused at the argument. "
           "The release mechanism itself is measured on CPU with one buffer type (see "
           "release_experiment_verified); building with HERD_BUILD_BUFFER_HOOKS=ON adds the "
           "hooks it needs";
#endif
}

// The measurement, kept as its own fact so it cannot be mistaken for availability.
//
// Measured on this pin, CPU, one buffer type:
//   armed   : ASan clean, logits bit-identical to the no-release control, two contexts
//             independent, one thread releasing while another decodes
//   no-arm  : ASan heap-use-after-free in llm_graph_input_embd::set_input <-
//             llm_graph_result::set_inputs, reachable with HERD_RELEASE_NO_ARM=1
//
// The second line is why the first is evidence rather than an absence of symptoms.
//
// The mechanism is `release_and_arm`: llama_set_embeddings() on a real change sets
// sched_need_reserve, decode() calls sched_reserve() before the ubatch loop, and
// sched_reserve() clears gf_res_prev_active -- so process_ubatch takes its rebuild
// branch, which allocates the graph *before* set_inputs rebinds the tensors.
//
// Outside this claim, because unmeasured or unwired: a multi-buffer-type
// (multi-device) scheduler, any GPU backend, and steps 1-2 above.
bool release_experiment_verified() {
    return true;
}

const char * release_experiment_note() {
    return "release mechanism measured on CPU, one buffer type: ASan clean, logits "
           "bit-identical, unarmed path reproduces the use-after-free; not measured on a "
           "multi-device scheduler and not wired into a product";
}

} // namespace herd::phase

// ---------------------------------------------------------------------------
// No autoload here, on purpose.
//
// This module's seams are section constants, so the installer ingests them with
// no registration step. An earlier version installed from a constructor(101)
// here, and that made the ordering a race: another module's registrar also runs
// at 101, and if the installer won, the late seam was added to a table that had
// already been walked. Measured, not reasoned: the prefab lane printed
// "FATAL seam 'prefab.process_ubatch': registered after seams were installed".
//
// The rule this leaves behind is structural rather than documentary: exactly one
// constructor installs (patches/herd_patch.cpp, priority 102, the highest number
// among constructors that matter), and every module only registers or declares.
// A module that needs an address the linker cannot give it registers at 101 and
// relies on 102 running after it, which add() enforces by aborting otherwise.
//
// herd::phase::install() stays for callers that want the seams up early and for
// the tests; it is idempotent through install_all().
// ---------------------------------------------------------------------------
