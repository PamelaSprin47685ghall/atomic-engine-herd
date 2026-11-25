// herd/reader_attn.cpp — an observation point inside attention, and nothing more.
//
// What this file is for. The reader lane (RERoT) needs a query to attend over a
// hand-picked set of stored keys, with the query's rope rotation differing per key.
// That decision lives inside llm_graph_context::build_attn. The rules are already
// ported and tested (modules/reader: two phases, twenty-seven named refusals, spans
// with a per-span effective position). What was unknown was whether the point where
// those spans must be turned into graph nodes can be reached from outside upstream.
// It can: the KV overload is an ordinary const member function, resolvable from
// .symtab by its mangled name. See kSymbol below.
//
// Why there is no request API any more. The previous revision let a caller open a
// "reader request" that this seam would honour or abort on. That was wrong, and the
// reason was measured: upstream has EIGHT build_attn overloads
// (llama-graph.cpp:3442, 3574, 3692, 3800, 3984, 4005, 4087, 4589) plus a shared
// build_attn_mha helper (:3270). Only one of them is hooked here — the KV one — so a
// request opened for a model that routes through any other overload would never
// reach this function, and the ordinary attention would run with a request
// outstanding. Nothing would report it: the numbers would look normal, which is the
// worst possible failure for a feature named "reader attention".
//
// A runtime check cannot close that hole, because the seam is not on the path at
// all. So the state is designed out instead: there is no way to open a request,
// therefore there is no way to be silently served stock attention. The contract for
// the future implementation is written below, where the emitter will live, and the
// preconditions it must satisfy are named.
//
// What remains is honest and useful:
//
//   * the interception point is proven reachable (installed, with an origin pointer);
//   * tracing prints the tensors this layer is handed — the shapes an implementation
//     needs and nobody has written down;
//   * with tracing off, the function is a pass-through with one counter increment.
//
// Contract for whoever implements the emission, from modules/reader's resolved
// output and this function's arguments:
//
//   in   spans[{ effective_pos, indices[] }]  per query token, indices into the KV
//        cache flattened over heads; effective_pos is where that span's keys sit in
//        rope terms relative to the query.
//   step for each span: gather K rows by index, apply a rope rotation of
//        effective_pos to Q, accumulate one softmax over the union of spans.
//   out  the same tensor shape build_attn returns today (ne[0] = n_embd_head_v,
//        ne[1] = n_tokens, ...), because the caller is unaware of anything new.
//
// WHAT THESE ARGUMENTS DO NOT CONTAIN (established by reading the KV overload's own
// signature comment, llama-graph.h:1389-1402, after the emission was queued as the next
// step; the earlier note above was written from the emitter's inputs, not from this
// function's):
//
//   q_cur is [n_embd_head_q, n_head_q, n_tokens]   -- the emitter wants [head_dim, 1, n_head]
//   k_cur is [n_embd_head_k, n_head_k, n_tokens]   -- these are THIS batch's new keys
//   v_cur is [n_embd_head_v, n_head_v, n_tokens]   -- not the cache
//
// So the tensors the emitter gathers from -- the K/V cache over all cells -- are ABSENT from
// this argument list. They are produced inside the original body by
// mctx->get_k(ctx0, il) / get_v(ctx0, il) (llama-graph.cpp:3727, 3847), where mctx is a
// llama_kv_cache_context held by the opaque `inp` (llm_graph_input_attn_kv::mctx,
// llama-graph.h:1500-1505). Three consequences, each with a known mechanism rather than an
// open question:
//
//   1. reaching the cache means reading `inp`'s private field and CALLING get_k/get_v on it.
//      The precedent exists and is already in this tree: herd_graph_access.cpp reads a private
//      member with a compile-time pointer-to-member, so the offset cannot drift. Extending it
//      from a field read to these two calls is the same technique, but the call allocates ggml
//      nodes, so it must happen at the point in the build where the original would have made
//      it -- which means this hook must eventually REPLACE the body, not wrap it.
//   2. q's layout differs from the emitter's by a permute: [hd, n_head, n_tokens] against
//      [hd, 1, n_head]. For n_tokens == 1 that is still a permute (ne1 and ne2 swap), so a
//      silent "it is close enough" reading would gather the wrong axis. ggml_permute is the
//      public API for the fix.
//   3. a multi-token batch needs one emission per query row (the emitter's view is per query
//      token), or an emitter that takes n_tokens. This is not a wiring detail: it decides
//      whether the first working configuration is decode-only.
//
// None of this makes the emission impossible; all of it makes the "call the emitter with the
// arguments we already have" version wrong. Recorded here because that version was the obvious
// next step and it would have produced a plausible graph from the wrong tensors.
//
// Preconditions the emitter must satisfy, each learned the hard way above:
//   P1 every one of the eight overloads must either implement the emission or refuse
//      it. Hooking one and calling it done is the defect this file exists to avoid.
//   P2 a request must be scoped to exactly one graph build, with no nesting and no
//      way to leak into the next batch (the previous design used thread_local state
//      whose lifetime the caller had to manage correctly; scope it in a guard).
//   P3 the request must be validated before the graph is built, not inside this
//      function: modules/reader's validate() requires the query's own position to be
//      absent from the cache, and by the time this runs, it has been written.
//
// The parameters are opaque on purpose. Dereferencing the graph input would mean
// including upstream's private headers and depending on their layout; this seam
// needs only ggml_tensor, which is public API.
#include "herd_patch.hpp"
#include "herd_graph_access.hpp"
#include "herd_graph_cache.hpp"
// The emitter is NOT included here any more, and the switch that made it necessary is gone.
//
// The block that used to be at this spot explained why the reader's graph header was included: a
// replacement path called `emit_reader_attention` from inside a live graph build, and later a
// publisher called `publish_index_inputs` from a graph input. BOTH paths are deleted. The
// replacement path was unsound (see the block below the arguments check); the publisher could not be
// made to work, because the scheduler places THE GRAPH rather than the context, so row tensors with
// no consuming node are never placed and the publish refuses on the first layer. The include, the
// CMake link to `tq_reader_graph`, and the include directory went with them.
//
// The consequence is worth stating plainly: a product no longer carries the reader emitter at all.
// That is a real capability loss, not a tidy-up -- and it is the honest state, because the only code
// that used it could not compute a correct answer.


#include <ggml.h>
#include <ggml-backend.h>

// The real `llm_attn_mask`, not a guessed prefix. Declaring the type here costs one include and
// removes the whole class of "the layout happens to match" reasoning from this probe; the compiler
// then checks the signature against the header the mangled name came from.
#include <memory>
#include <string>
#include <vector>

#include "llama-graph.h"

#ifdef HERD_READER_PHYSICAL
// The physical emitter, linked here only when CMake proved it exists (see the `if(TARGET tq_reader)`
// block appended after `add_subdirectory(modules)`). A `__has_include` guard would be the wrong test:
// it can be true while nothing is linked, which compiles and then fails at link time on a switch
// nobody expected to matter.
#include <tq/reader/graph_emit_physical.h>
#include <tq/reader/graph_emit_prefill.h>
#include "herd_reader_prefill_publisher.hpp"
#endif

// The reader emitter is NOT included here any more. It was included for a replacement experiment
// that has been deleted (see the note at the probe block), and with the call gone the include, the
// `herd_seams` link to `tq_reader_graph`, and the include directory went with it. A patch layer that
// carries a link it never uses is a dependency nobody can justify when it breaks a configuration.
#include <utility>

// The runtime row-table validator, when the CMake target it lives in is linked.
//
// A COMPILE DEFINITION FROM CMAKE, NOT `__has_include`. `__has_include` asks whether a HEADER is
// reachable, which is a different question from whether the LIBRARY is: the include directory is
// added by the same CMake block that links `tq_reader_rows`, so in practice the two agree -- but a
// build that happens to see the header on some other include path (a system-wide install, a
// stale `-I`) would compile the validating code and fail at link, or worse, compile a stub that
// claims the switch cannot be honoured when it could. `HERD_ROWS_VALIDATOR` is set by CMake exactly
// when the target is linked, so the two can no longer disagree.
#if defined(HERD_ROWS_VALIDATOR)
#  include <tq/reader/rows_validation.h>
// The description contract, from the same include directory. It is ggml-free and header-only, so it
// costs nothing when the target is linked -- and it is INSIDE this guard because that directory is
// added by the same CMake block that links `tq_reader_rows`.
#  include <tq/reader/seam_contract.h>
#endif

#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace {

// How many times the seam body ran. Without this, "the numbers did not change" has
// two explanations -- the hook was transparent, or the hook was never called -- and
// the second is indistinguishable from the output alone. Reported at exit under
// HERD_SEAM_TRACE, so "it is on the path" is measured rather than assumed. It also
// records the honest zero: on a model with no attention layers (mamba), the count
// stays 0 and no transparency claim can be made from that run.
std::atomic<long> g_calls{0};

// The upstream signature, spelled with opaque handles.
using build_attn_fn = ggml_tensor * (*)(const void *, const void *,
                                        ggml_tensor *, ggml_tensor *, ggml_tensor *,
                                        ggml_tensor *, ggml_tensor *, ggml_tensor *,
                                        ggml_tensor *, ggml_tensor *, ggml_tensor *,
                                        float, int);

build_attn_fn g_orig = nullptr;

// ---------------------------------------------------------------------------
// THE SECOND SEAM, AND THE ONE THE READER ACTUALLY NEEDS.
//
// `build_attn` is the wrong boundary to replace, and reading its body in this pin says why.
// Between its entry and the attention math it does two things a replacement would have to
// reproduce exactly:
//
//     ggml_build_forward_expand(gf, q_cur / v_cur / k_cur);          // llama-graph.cpp:3707
//     ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
//     ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
//
// and between the math and its return it applies four more:
//
//     cur = build_attn_scale_kv_output(ctx0, mctx_cur, cur, il);
//     if (inp->self_v_rot[il]) cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot[il]);
//     if (wo) cur = build_lora_mm(wo, cur, wo_s);
//     if (wo_b) cur = ggml_add(ctx0, cur, wo_b);
//
// A replacement at `build_attn` that returned its own tensor would therefore SILENTLY STOP
// WRITING THE KV CACHE -- the model would decode one token correctly and then attend to a
// cache that never advanced. That is the worst failure shape available here: the first token
// looks right, and there is no error to find. The mask is also private at that level
// (`inp->self_attn_mask.raw`), so a replacement would have to reconstruct it through a
// private field, and `gf` is a member as well, so the expanded nodes could not be re-emitted.
//
// `build_attn_mha` has none of those problems. It receives q, k, v, kq_b, the mask, kv_rows,
// sinks and v_mla as ARGUMENTS, is called from seven sites in this file (every attention
// variant, including MLA), and its return is the attention result before any of the
// post-processing above -- but the SHAPE of that return is not one shape: an earlier note here
// said `[n_embd_head_q, n_head_q, n_tokens]`, and measurement refused it. On the tiny model the
// observed return is rank 1, `ne [32 1 1 1]`, F32, i.e. `n_embd` values for one token. The non-flash
// branch visibly flattens with `ggml_cont_2d` before returning; which branch this model takes, and
// therefore exactly where the 1-D shape is produced, is not established -- the probe prints the
// result, not the path. The values are ordered `d + head_dim*head`, which is exactly the emitter's
// `[head_dim, 1, n_head]` order, so a replacement matches by reshaping, not by reordering. Everything
// that must not be duplicated stays where upstream put it.
//
// This probe exists to measure that seam before anything is replaced at it: the shapes it
// prints are the emitter's contract, and `n_stream`/`v_trans` are the two properties of the
// incoming tensors the emitter currently assumes rather than checks.
using build_attn_mha_fn = ggml_tensor * (*)(const void *,
                                            ggml_tensor *, ggml_tensor *, ggml_tensor *,
                                            ggml_tensor *, ggml_tensor *, ggml_tensor *,
                                            ggml_tensor *, ggml_tensor *,
                                            std::int64_t, float, int);

build_attn_mha_fn g_orig_mha = nullptr;
std::atomic<long> g_mha_calls{0};

// The phase seam (`graph_reserve`) is defined further down, next to the hook that raises it. These
// two declarations exist because the adapt log ABOVE asks the question -- and asking it is the whole
// point: the tag has to come from the call, not from a shape. `herd_phase_known()` distinguishes
// "not a reserve graph" from "nobody is watching", which are different facts and both appear in a log.
bool herd_phase_is_reserve();
bool herd_phase_known();

// A SEPARATE THROTTLE FOR MULTI-TOKEN CALLS, because `g_mha_calls` cannot serve both questions.
//
// Every measurement this probe produced before this existed was a SINGLE-TOKEN decode, and the reason
// was the gate itself: `n <= 2` over a process-global counter. llama builds a reserve graph and then
// the running graph, so the first two calls are the reserve ones -- a multi-token call, which is
// exactly the question the prefill lane turns on, was filtered out before it could print.
//
// THE BUDGET IS 40, NOT 6, AND THE REASON IS THE SAME DEFECT SEEN FROM THE OTHER SIDE.
// `graph_reserve` (`llama-context.cpp:3407`) calls `model.build_graph`, so RESERVE GRAPHS REACH THIS
// PROBE TOO and cannot be told apart from running graphs by shape: a reserve ubatch is built for the
// worst case and can be identical to a real one. With a budget of 6, the six printed calls were all
// reserve graphs and the log looked like a prefill measurement. 40 lets the sequence be seen -- in
// one run, calls #1-#2 were single-token, #16-#18/#22-#24/#28-#30/#34-#36 had `k ne[2] = 2048`
// (n_ctx, the reserve worst case), and only #25-#27 had `k ne[2] = 256` (the tiny model's real n_kv).
// That distinction is an INFERENCE from the sequence, not a labelled fact: this probe has no phase
// tag, and adding one needs a marker at both `graph_reserve` and `decode`.
std::atomic<long> g_multi_probe{0};

bool mha_probe_enabled() {
    // `HERD_READER_ADMIT` turns on the same line set, because the fact it exists to establish --
    // how much room the metadata arena has left when attention is built -- is measured in exactly
    // this block. A separate switch with a separate print would be a second copy of the same
    // arithmetic, and the copy that nobody re-ran is the one that lies.
    static const bool enabled = std::getenv("HERD_READER_MHA") != nullptr ||
                                std::getenv("HERD_READER_ADMIT") != nullptr;
    return enabled;
}

// The graph-input diagnostic, on its own switch. It lives here rather than as a bare `getenv` at
// its use site because the INSTALL gate needs the same answer: a request that installs no seam is
// a request that cannot be honoured, and it must be visible as such instead of producing an empty
// log that later gets read as a measurement.
bool input_probe_enabled() {
    static const bool enabled = std::getenv("HERD_READER_INPUT") != nullptr;
    // NO EXIT-TIME TALLY, and the reason is worth a line. The counters live in the anonymous
    // namespace below (they are file-local to this TU by construction), and a tally registered here
    // would need them declared first -- an `extern` in the enclosing scope names a DIFFERENT entity
    // with external linkage, so the tally would read its own zero-initialised twin and print
    // "0 calls" after a run that called thousands. A number that is silently wrong is worse than no
    // number, so the per-call lines stay throttled and every reader is told they are throttled: the
    // printed count is an UPPER BOUND on what the log shows, never a call count. What the log does
    // establish is which `ubatch->n_tokens` values reached `set_input`, and that is the fact a
    // design decision needs.
    return enabled;
}

// An observer: prints, then calls the original with the original arguments. It replaces
// nothing, so it cannot change a model's numbers, and that is the point -- a probe that
// altered the result would not be able to tell a shape mismatch from its own effect.
// ---------------------------------------------------------------------------
// THE REPLACEMENT, AND WHY IT REFUSES TODAY.
//
// Replacing this call means reproducing everything the original does with these
// arguments. The list is short and every item is a fact about the argument set,
// not a wish:
//
//   kq_b      per-head bias added to the scores
//   sinks     extra attention sinks carrying their own logits
//   v_mla     MLA value-projection input
//   kq_mask   per-column bias. Measured on the tiny model: F16 (`type 1`), and it
//             is a BIAS, not a boolean -- a finite value is a sliding-window or
//             ALiBi term, and treating it as "visible" loses it
//   kv_rows   I32 INPUT tensor: mask column i reads row kv_rows[i] of k/v, -1 is
//             padding and requires that column's mask to be -inf
//   arena     `emit_physical_attention` admits a no_alloc context only when it can
//             prove an object bound; an allocating context is refused on principle
//
// The emitter in `modules/reader` satisfies none of the first five: it builds its
// OWN mask from a cell list, takes cell indices as HOST constants (so it cannot
// consume `kv_rows`, whose values appear at compute time), and has no notation for
// kq_b, sinks or v_mla. Its tests prove its arithmetic, not its fitness for THESE
// arguments. Wiring it in today would silently drop at least one of them, and a
// dropped per-head bias or sink produces a plausible number -- the failure class
// this file exists to refuse.
//
// So the switch exists and it REFUSES. It refuses instead of falling back, because
// a switch that quietly runs the original path while reporting success is the same
// defect seen from the other side: the operator asked for the reader lane, and a
// run that looks like it delivered one while computing the stock answer is worse
// than a loud error. Refusing is also the only outcome that can be verified today,
// which is why it is the one that ships.
// ---------------------------------------------------------------------------

// REFUSED AT STARTUP, not at the first layer.
//
// Three ways this used to be decided, and why none of them survives:
//
//   * inside the seam, on the first call -- then a run that never builds attention (model fails to
//     load, no decode, exits early) never enters the refusing code and exits 0, a verdict this lane
//     did not reach;
//   * an `atexit` handler patching that up -- which converts a legitimately failing exit and even
//     `--help` into SIGABRT, and still cannot see `_Exit` or a signal;
//   * inspecting the arguments to list the reasons -- a `std::vector` and a dozen enum values in
//     service of a decision that is always the same, because NOTHING in this tree can replace this
//     call today. The list described work, and the work is already recorded in the block above.
//
// What is left is the honest shape: the request is checked once, at static-init time, before any
// model is loaded and before `main` runs. There is no supported scenario to preserve, so there is
// nothing to decide at runtime -- the answer is "this build cannot do that", said once, loudly, with
// the reasons in the comment above.
void refuse_if_replace_requested() {
    // A RETIRED SWITCH MUST NOT BE A NO-OP, and this one earned the rule twice.
    //
    // `HERD_READER_ATTN_TRY` used to replace the layer's attention inside the probe block below.
    // That block is gated on `HERD_READER_MHA` and throttled to the first two calls, so setting TRY
    // alone installed no seam and the run exited 0 having done nothing -- a request dropped in
    // silence. Setting both replaced two layers and left the rest stock, producing a mixture of two
    // implementations reported as one result.
    //
    // The block is deleted, and the variable is REFUSED HERE, at static-init time, before a model
    // loads and before `main` runs. Silence would read as "the replacement ran and agreed with the
    // stock answer", which is the single most misleading thing this file could print.
    if (std::getenv("HERD_READER_ATTN_TRY") != nullptr) {
        std::fprintf(stderr,
                     "[herd] reader.try: HERD_READER_ATTN_TRY is retired -- this build performs no "
                     "attention replacement. The experiment it named was throttled to the first two "
                     "hook calls, so a run could mix replaced and stock layers and report the result "
                     "as one; it moved a value it then read back; and its full-coverage emission "
                     "produced no gather, so the index hand-off it registered never ran. See the "
                     "note at the probe block in patches/herd_reader_attn.cpp. Nothing was computed "
                     "for this request, and the run stops here rather than reporting a stock answer "
                     "as the reader lane's.\n");
        std::fflush(stderr);
        std::abort();
    }

    if (std::getenv("HERD_READER_REPLACE") == nullptr) return;
    std::fprintf(stderr,
                 "[herd] reader.replace: REFUSED -- HERD_READER_REPLACE asks this build to replace "
                 "the layer's attention, and the emitter cannot express these arguments:\n"
                 "[herd] reader.replace:   kq_b (per-head score bias), sinks, v_mla have no notation "
                 "in it; kq_mask is a bias (F16, finite values), not a visibility flag; kv_rows is a "
                 "runtime row indirection whose values do not exist when the graph is built.\n"
                 "[herd] reader.replace: running the stock path instead would report a result this "
                 "lane did not compute, so the run stops here -- before any model is loaded.\n");
    std::fflush(stderr);
    std::abort();
}

namespace {
std::atomic<long> g_input_calls{0};
std::atomic<long> g_inputs_registered{0};

// There is NO table of graphs here, and the reason is worth keeping.
//
// The previous revision kept a 64-entry array of `llm_graph_result *` to register once per graph.
// Every part of that was wrong, and each failure was silent:
//
//   * It stored addresses of objects it did not own. llama frees each `llm_graph_result` when its
//     decode finishes, so the array filled up with pointers to dead objects -- and the allocator
//     reuses those addresses, so a LATER graph could be mistaken for one already registered and get
//     no input at all. The symptom would be "set_input stopped being called", which reads exactly
//     like a property of upstream.
//   * It answered a question that cannot be answered from an address: two graphs at different
//     times can share an address, and one graph at one time has one address. Identity is not
//     derivable from a value that is only unique while it is alive.
//   * It was a process-global mutable structure written from the decode path, which is the shape
//     this project's patch rules refuse.
//
// What replaces it is the honest version: while `HERD_READER_INPUT` is set, EVERY call registers an
// input, and only PRINTING is throttled. Registration is unconditional because a graph that gets no
// input cannot be observed, and observing graphs is the entire purpose; the cost of an extra input
// on a graph that already has one is a stored `unique_ptr`, not a wrong answer. `can_reuse` is
// false on the probe input, so a graph carrying it is never reused -- which is also why registering
// on each build is safe here: there is no second build of a reused graph to duplicate.
} // namespace

ggml_tensor * herd_build_attn_mha(const void * self,
                                  ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                                  ggml_tensor * kq_b, ggml_tensor * kq_mask,
                                  ggml_tensor * kv_rows, ggml_tensor * sinks,
                                  ggml_tensor * v_mla,
                                  std::int64_t n_kv_max, float kq_scale, int il) {
    const long n = ++g_mha_calls;

    // No replacement branch: the request was refused at static-init time (see
    // `refuse_if_replace_requested`), so reaching this line means no replacement was asked for.
    // Nothing is decided here on purpose -- a decision here could only be reached by a run that
    // never built attention at all.
    // One layer's report is enough to establish the contract; printing every layer of every
    // decode would bury the first useful line in a hundred identical ones.
    if (mha_probe_enabled() && n <= 4) {
        const bool v_trans = v && v->nb[1] > v->nb[2];
        const auto n_stream = kv_rows ? kv_rows->ne[1] : (k ? k->ne[3] : -1);
        std::fprintf(stderr,
                     "[herd] reader.mha: layer %d q ne [%lld %lld %lld] | k ne [%lld %lld %lld %lld]"
                     " nb[1] %zu nb[2] %zu type %d | v ne [%lld %lld %lld] type %d"
                     " | v_trans %d | n_stream %lld | mask %s | kv_rows %s | scale %g | n_kv_max %lld\n",
                     il,
                     (long long) q->ne[0], (long long) q->ne[1], (long long) q->ne[2],
                     (long long) k->ne[0], (long long) k->ne[1], (long long) k->ne[2],
                     (long long) k->ne[3], k->nb[1], k->nb[2], (int) k->type,
                     (long long) v->ne[0], (long long) v->ne[1], (long long) v->ne[2], (int) v->type,
                     v_trans ? 1 : 0, (long long) n_stream,
                     kq_mask ? "present" : "NULL", kv_rows ? "present" : "NULL",
                     (double) kq_scale, (long long) n_kv_max);
    }
    // THE ADAPTER CONTRACT, measured instead of assumed.
    //
    // Two facts decide whether `emit_physical_attention` can run at this call site at all, and
    // neither had ever been observed:
    //
    //   1. Whether the graph context is no_alloc. The emitter proves an OBJECT bound and refuses an
    //      allocating context on principle, so an allocating ctx0 makes the whole path impossible
    //      regardless of shapes. `herd::graph::context_of` reaches the private member through an
    //      accessor the compiler generates from upstream's own header, so this is a fact about the
    //      running object rather than a reading of a comment.
    //   2. Whether q's axes are [head_dim, n_head, n_tokens] or [head_dim, 1, n_head]. The emitter
    //      takes the second. If upstream passes the first, a replacement needs a PERMUTE, not a
    //      reshape: reshaping would keep the data contiguous and silently transpose every head,
    //      which produces plausible numbers rather than an error. Printing both sizes is what makes
    //      that difference visible instead of inferred.
    //
    // `q_perm_needed` compares the two readings directly: it is true when the axis that carries
    // n_head is ne[1] rather than ne[2], which is exactly when `ggml_permute(0,2,1,3)` is required.
    // THE THROTTLE MUST NOT HIDE PREFILL, AND `q->ne[2]` IS WHAT TELLS THEM APART.
    //
    // `q` arrives as `[head_dim, n_head, n_tokens]` (measured: `ne [8 4 1]` for one token), so the
    // TOKEN axis is ne[2]. Testing ne[1] would be wrong in the most misleading way possible: ne[1] is
    // n_head, which is > 1 even for a single decode, so every decode would be counted as a prefill.
    const bool multi_token = q != nullptr && q->ne[2] > 1;
    if (mha_probe_enabled() &&
        (multi_token ? g_multi_probe.fetch_add(1) < 40 : n <= 2)) {
        ggml_context * ctx0 = herd::graph::context_of(self);
        // NO DERIVED AXIS VERDICT IS PRINTED, AND TWO EARLIER ONES WERE DELETED RATHER THAN FIXED.
        //
        //   `perm_needed`     (q->ne[1] > 1)          claimed a consumer requirement nobody measured.
        //   `n_head_on_ne1`   (ne[1] != 1 && ne[2] == 1)  claimed to locate `n_head`, which the axes
        //                     cannot decide: for GQA/MQA both of those dims are > 1, and a tensor of
        //                     shape [8 4 1] is equally "head on ne[1], token on ne[2]" and "token on
        //                     ne[1], head on ne[2]" -- the shape does not say which. A field that
        //                     looks like it answers the question while being undecidable is worse than
        //                     no field, because it will be quoted.
        //
        // What remains is what the probe can actually see: `ne`, `nb`, the type, the INPUT flag, and
        // whether data exists. A reader who needs to know which axis carries heads gets that from
        // upstream's own signature and from `n_embd`/`head_dim` arithmetic, not from this line.
        //
        // Contiguity as the emitter's admission sees it: nb[0] must be one element and each
        // following stride must cover the dimension before it.
        auto contiguous_3d = [](const ggml_tensor * t) {
            if (t == nullptr) return false;
            const std::size_t el = ggml_element_size(t);
            return t->nb[0] == el && t->nb[1] == std::size_t(t->ne[0]) * el &&
                   t->nb[2] == std::size_t(t->ne[1]) * t->nb[1];
        };
        std::fprintf(stderr,
                     "[herd] reader.adapt: layer %d phase %s | ctx0 %s no_alloc %d used_mem %zu"
                     " free_meta %zu size %zu | "
                     "k [%lld %lld %lld] nb1 %zu nb2 %zu | v [%lld %lld %lld] nb1 %zu nb2 %zu | "
                     "q [%lld %lld %lld] nb1 %zu nb2 %zu | contiguous k %d v %d q %d | "
                     "mask ne [%lld %lld %lld %lld] type %d contig %d | n_cell %lld | multi %d\n",
                     il,
                     // THE PHASE IS READ FROM THE SEAM, NOT INFERRED FROM SHAPES.
                     //
                     // `graph_reserve` (llama-context.cpp:3407) also calls `model.build_graph`, and its
                     // ubatch comes from `ubatch_reserve(n_tokens/n_seqs, n_seqs)` -- a WORST-CASE
                     // batch, so `q`, `k` and `mask` can have exactly the shapes a real prefill has.
                     // Measured earlier in this tree: one run produced `n_cell` values of both 2048
                     // (= n_ctx) and 256 (= the model's real n_kv), and the "prefill" reading of those
                     // rows turned out to be reserve graphs. Nothing in the tensor distinguishes them;
                     // the call does. So the tag comes from `herd_phase_is_reserve()`, which is set by
                     // the `graph_reserve` seam.
                     //
                    // WHAT THE TAG NAMES, AND WHAT IT DOES NOT. An earlier revision printed
                    // `reserve` / `running`, and that vocabulary claimed more than the signal holds.
                    //
                    // The depth is a thread-local counter raised only inside the ONE `graph_reserve`
                    // this tree hooks. Depth 0 therefore means "not inside that call's dynamic
                    // extent on this thread" -- not "this is a real inference". At least three cases
                    // land in depth 0 without being a running graph:
                    //
                    //   * a reserve reached by a path that does not go through this symbol (a
                    //     different TU's inline copy, another reserve entry point);
                    //   * a reserve built on another thread, whose depth counter is its own zero;
                    //   * a graph built for sizing by something that is not `graph_reserve` at all.
                    //
                    // So the tag says `inside_reserve_scope` / `outside_reserve_scope`, which is the
                    // fact: membership in the hooked call's dynamic extent. Reading `outside` as
                    // "running inference" is the same mistake as reading an un-driven counter as a
                    // negative measurement, and it is the reason the words changed rather than the
                    // mechanism.
                    //
                    // What an `inside_reserve_scope` tag DOES settle: `graph_reserve` never computes.
                    // It builds the graph only so `ggml_backend_sched_reserve` can size the compute
                    // buffers (`split_only` only splits it), and the returned graph is discarded. So
                    // a multi-token call inside that scope is not evidence of real prefill, whatever
                    // its shapes say.
                    //
                    // THREE STATES, NOT TWO. `herd_phase_is_reserve()` reads a depth that only the
                    // `HERD_READER_PHASE` seam raises; with the seam absent it is always 0, so a
                    // two-state tag would print `outside_reserve_scope` for a graph nobody is
                    // watching. `herd_phase_known()` separates "outside the hooked scope" from
                    // "nobody is watching", which are different facts.
                    !herd_phase_known() ? "unknown(no phase seam)"
                                        : (herd_phase_is_reserve() ? "inside_reserve_scope"
                                                                   : "outside_reserve_scope"),
                     ctx0 ? "present" : "NULL",
                     ctx0 ? (ggml_get_no_alloc(ctx0) ? 1 : 0) : -1,
                     ctx0 ? ggml_used_mem(ctx0) : std::size_t(0),
                     // THE METADATA ARENA'S REMAINING BYTES, in the unit `arena_budget` reasons in.
                     // Tensors and their offsets live in this pool, and `ggml_new_object` ABORTS on
                     // overflow (ggml.c:1758) instead of returning something the caller could report.
                     // So this number is the only form in which the admission question is answerable:
                     // arithmetic on bytes, not a guess about an allocator's mood.
                     ctx0 ? ggml_get_mem_size(ctx0) - ggml_used_mem(ctx0) : std::size_t(0),
                     ctx0 ? ggml_get_mem_size(ctx0) : std::size_t(0),
                     k ? (long long) k->ne[0] : -1, k ? (long long) k->ne[1] : -1,
                     k ? (long long) k->ne[2] : -1, k ? k->nb[1] : std::size_t(0),
                     k ? k->nb[2] : std::size_t(0),
                     v ? (long long) v->ne[0] : -1, v ? (long long) v->ne[1] : -1,
                     v ? (long long) v->ne[2] : -1, v ? v->nb[1] : std::size_t(0),
                     v ? v->nb[2] : std::size_t(0),
                     q ? (long long) q->ne[0] : -1, q ? (long long) q->ne[1] : -1,
                     q ? (long long) q->ne[2] : -1, q ? q->nb[1] : std::size_t(0),
                     q ? q->nb[2] : std::size_t(0),
                     contiguous_3d(k) ? 1 : 0, contiguous_3d(v) ? 1 : 0, contiguous_3d(q) ? 1 : 0,
                     kq_mask ? (long long) kq_mask->ne[0] : -1,
                     kq_mask ? (long long) kq_mask->ne[1] : -1,
                     kq_mask ? (long long) kq_mask->ne[2] : -1,
                     kq_mask ? (long long) kq_mask->ne[3] : -1,
                     kq_mask ? (int) kq_mask->type : -1,
                     kq_mask ? (ggml_is_contiguous(kq_mask) ? 1 : 0) : -1,
                     k ? (long long) k->ne[2] : -1,
                     multi_token ? 1 : 0);
        std::fprintf(stderr, "[herd] reader.adapt: call #%ld layer %d\n", n, il);

    // THE ENVELOPE IS AN OBSERVATION, NOT A PROMISE.
    //
        // A replacement is only sound when every clause holds, and each has been the reason a
        // previous attempt was wrong: an argument with no notation in the emitter, a mask that is a
        // bias rather than a flag, a runtime row indirection, more than one token or stream, or a
        // context whose allocation model forbids the emitter's object bound.
        //
        // Nothing consumes this verdict any more -- the replacement that did is deleted. It is kept
        // because it is the only part of that experiment which produced a durable fact: on the tiny
        // model the decode call satisfies all eight clauses and prefill does not. A future attempt
        // starts from that measurement instead of re-deriving it.
        //
        // Printed as a verdict on ONE line, because a list of satisfied conditions is unreadable and
        // a list of violated ones is what the next step needs. `n_cell` is the cache depth; it must
        // equal `k->ne[2]`, and the emitter's grid takes k as [head_dim, n_head_kv, n_cell].
        const bool no_rows       = kv_rows == nullptr;
        const bool no_head_bias  = kq_b == nullptr;
        const bool no_sinks      = sinks == nullptr;
        const bool no_mla        = v_mla == nullptr;
        const bool mask_cols_ok  = kq_mask != nullptr && k != nullptr && kq_mask->ne[0] == k->ne[2];
        const bool mask_ok       = kq_mask != nullptr &&
                                   (kq_mask->type == GGML_TYPE_F16 || kq_mask->type == GGML_TYPE_F32) &&
                                   kq_mask->ne[2] == 1 && kq_mask->ne[3] == 1 && mask_cols_ok;
        const bool single_token  = q && q->ne[2] == 1 && k && k->ne[3] == 1 && v && v->ne[3] == 1;
        const bool shapes_ok     = k && v && q && k->ne[0] == q->ne[0] &&
                                   v->ne[1] == k->ne[1] && v->ne[2] == k->ne[2];
        const bool ctx_ok        = ctx0 != nullptr && ggml_get_no_alloc(ctx0);
        std::fprintf(stderr,
                     "[herd] reader.adapt: layer %d fields -- rows %d head_bias %d sinks %d mla %d "
                     "mask %d single_token %d shapes %d no_alloc %d\n",
                     il, no_rows ? 0 : 1, no_head_bias ? 0 : 1, no_sinks ? 0 : 1, no_mla ? 0 : 1,
                     mask_ok ? 1 : 0, single_token ? 1 : 0, shapes_ok ? 1 : 0, ctx_ok ? 1 : 0);
        std::fflush(stderr);

        // THE REPLACEMENT WAS HERE AND IS GONE. It is not commented out; it is deleted, because a
        // disabled copy of a wrong experiment is a thing the next reader will re-enable.
        //
        // What it did, and why each part was unsafe:
        //
        //   * It lived INSIDE this probe block, which is gated on `HERD_READER_MHA` and throttled to
        //     the first two hook calls. So `HERD_READER_ATTN_TRY` alone did nothing at all -- the
        //     seam was never installed and the run exited 0 -- and even with both switches set, only
        //     two layers were replaced while the rest computed stock. A run whose output is a MIXTURE
        //     of two implementations, reported as one, is the failure class this file exists to
        //     refuse, and it was the shape of the experiment rather than a bug in it.
        //   * It called `add_input(std::make_unique<publish_once>(std::move(em)))` and then read
        //     `em.indices`/`em.out` for its log line and its return value: a use after move. That it
        //     happened to print and return something is not evidence the values were valid.
        //   * It returned `ggml_reshape_1d(em.out, ...)` only for the layers inside the throttle, so
        //     the graph mixed a replaced attention with a stock one and no comparison could attribute
        //     the result to either.
        //   * `emit` produced ZERO index entries for a full-coverage view, so the publisher it
        //     registered never ran, and the one mechanism it existed to demonstrate was never
        //     exercised -- while the log said "REPLACED".
        //
        // The envelope print above stays: it is an OBSERVATION about the call site, and it is the
        // only part of this experiment that produced a fact worth keeping (decode satisfies all eight
        // clauses; prefill does not).
        //
        // A real replacement needs, in this order: a whole-graph decision (not a per-layer throttle),
        // a constructor that does not consume the value it returns, and a gather that actually exists
        // so the index hand-off is exercised. None of the three is small, and none is served by
        // keeping a switch that can half-do the job. `HERD_READER_ATTN_TRY` is therefore refused at
        // startup -- see `refuse_if_replace_requested`.
        //
        // (The block that used to follow is deleted; see the note above.)
    }

    // THE GRAPH-OWNED INPUT: the registration path, exercised rather than assumed.
    //
    // The open question was "who publishes the gather indices after placement". `publish_index_inputs`
    // is proven correct (`tq_reader.index_lifecycle`), but its caller has to reach the window between
    // placement and compute. Upstream has exactly one such place: every `llm_graph_input_i` in
    // `llm_graph_result::inputs` gets `set_input(ubatch)` called from `set_inputs`, after the
    // scheduler allocates. The registration is `res->add_input(...)`, PUBLIC at llama-graph.h:1028.
    //
    // This input does NOT publish indices yet -- there are no indices at this seam, because no
    // reader emission happens here. What it proves is the three things no static reading could:
    //
    //   1. `res` is reachable from this hook. `herd::graph::result_of(self)` uses the same
    //      pointer-to-member capture that already reads `ctx0`, so it is comptime-checked against
    //      upstream's header rather than offset-guessed.
    //   2. `add_input` ACCEPTS an input at this moment. If `set_inputs` had already run for this
    //      graph, the input would be stored and never called -- a silent no-op, which is why the
    //      counter is printed at exit rather than trusted.
    //   3. WHEN `set_input` runs, and how often. That is the window a real publisher needs.
    //
    // The cost is a behaviour change worth naming: `can_reuse` defaults to false, so adding any
    // input here means the graph is NEVER reused while this switch is on. Upstream's own reuse is
    // what makes decode fast, so this is a diagnostic and not something to leave enabled.
    struct input_probe final : llm_graph_input_i {
        int layer = 0;
        explicit input_probe(int il) : layer(il) {}
        void set_input(const llama_ubatch * ubatch) override {
            const long n = g_input_calls.fetch_add(1) + 1;
            if (n <= 4) {
                std::fprintf(stderr,
                             "[herd] reader.input: set_input ran (call %ld) for layer %d, "
                             "ubatch %s n_tokens %u\n",
                             n, layer, ubatch ? "present" : "NULL", ubatch ? ubatch->n_tokens : 0);
                std::fflush(stderr);
            }
        }
        // `can_reuse` left at the base default (false). Not an oversight: this input holds no
        // tensors of its own, so there is nothing to compare, and returning true would let
        // upstream reuse a graph while this object's identity is irrelevant -- which is exactly
        // the assumption a real publisher must NOT inherit. Documented here so the reuse cost is
        // a decision rather than a side effect nobody noticed.
    };

    // THE PUBLICATION BRIDGE: a graph-owned input that writes a row table into placed tensors.
    //
    // The window is the one this file has been circling. `publish_index_inputs` is the correct
    // writer (`tq_reader.index_lifecycle` proves it against a real backend), but it needs a caller
    // that runs AFTER the scheduler places the tensors and BEFORE the graph computes. Upstream has
    // exactly one such place per graph: every `llm_graph_input_i` in `llm_graph_result::inputs` gets
    // `set_input(ubatch)` there, and `add_input` is public (llama-graph.h:1028). Measured: `n_tokens`
    // 13 (prefill) and 1 (decode) both reach it.
    //
    // So the OWNER is upstream. This object is a `unique_ptr` inside `llm_graph_result`, freed with
    // the graph; the tensors it borrows belong to the same context that owns it, so "the graph died
    // but the publisher still writes" is not representable. That is the lifetime guarantee the
    // deleted `index_holder` could not give -- it held raw tensor pointers with no graph token, so a
    // caller could free the context first and it would write into freed memory, undetectably.
    //
    // WHAT IT PUBLISHES TODAY: an eight-cell diagnostic row table, not the cache's real one. The
    // point is the MECHANISM -- entries prepared at build time, values written after placement,
    // read back to prove where they landed -- and eight cells buy that check for 64 bytes per tensor
    // instead of eight kilobytes per layer. A real publisher substitutes the emission's own
    // `indices` here and changes nothing else.
    //
    // MEASURED, AND IT FAILS: `HERD_READER_PUBLISH=1` aborts on layer 0 with
    //
    //   REFUSED: an entry's tensor has no data: publish ran before
    //   ggml_backend_sched_alloc_graph() placed the graph ... (entry span 0: no storage yet)
    //
    // The window is right -- `llama-context.cpp` calls `ggml_backend_sched_alloc_graph` at 2054 and
    // `res->set_inputs(&ubatch)` at 2068 -- so the cause is one level down and worth stating: the
    // scheduler allocates THE GRAPH, not the context. A tensor no node consumes is not in the graph,
    // so it is never placed, and `set_input` then runs with `buffer == nullptr` no matter how late it
    // is called. Upstream's own inputs are fine because ops consume them (`kq_mask` feeds
    // `soft_max_ext`, `kv_rows` feeds the flash-attention gather); my two tensors were orphans.
    //
    // THE PUBLISHER IS GONE, AND THIS IS WHY. It created two I32 row tensors in the graph context,
    // marked them inputs, registered a `llm_graph_input_i` that called `publish_index_inputs` on
    // them -- and aborted on layer 0 of the very first run:
    //
    //   REFUSED: an entry's tensor has no data: publish ran before
    //   ggml_backend_sched_alloc_graph() placed the graph ... (entry span 0: no storage yet)
    //
    // The window was correct (`llama-context.cpp` allocates at 2054 and calls `set_inputs` at 2068),
    // so the failure is one level down: the scheduler places THE GRAPH, not the context. A tensor
    // that no node consumes is not in the graph and is never placed, so `set_input` sees
    // `buffer == nullptr` however late it runs. Upstream's inputs escape this because ops consume
    // them -- `kq_mask` feeds `soft_max_ext`, `kv_rows` feeds the flash-attention gather.
    //
    // That makes "register a graph input" insufficient as a publication mechanism, and the fix is
    // not a tweak: the row tensors must be consumed BY A NODE whose result is part of the graph,
    // i.e. they have to move inside the attention graph rather than sit beside it. Until that is
    // built, a switch that always aborts is worse than no switch: it looks like a feature that is
    // merely misconfigured. So the block, its variable and its product-library dependency were
    // removed, and the finding was kept here instead -- this comment is the deliverable.

    // Asked for the graph-owned input.
    //
    // REGISTERED ON EVERY CALL, with only the printing throttled. An earlier revision gated the
    // registration itself (`n <= 2` over a process-global counter), and since llama builds a graph
    // to reserve the arena before it builds the one that runs, the first calls were reserve graphs:
    // the running graph registered nothing, `set_input` was never entered, and the empty log was
    // read as "upstream does not traverse inputs added here". The gate was measuring the counter,
    // not upstream. A second revision keyed registration on the `llm_graph_result` pointer through a
    // 64-entry table, which was worse -- dead addresses, reused addresses, and a global mutable
    // structure on the decode path (see the block above).
    //
    // So: while the switch is set, every call adds an input, and `n <= 8` only decides how many
    // times that fact is printed. Registering twice on one graph cannot produce a wrong number --
    // the input carries no tensors and only counts calls -- while a graph without one is invisible.
    if (input_probe_enabled()) {
        void * res_raw = herd::graph::result_of(self);
        if (!res_raw) {
            std::fprintf(stderr,
                         "[herd] reader.input: HERD_READER_INPUT is set but llm_graph_context::res "
                         "was not reachable at layer %d\n", il);
            std::abort();
        }
        auto * res = static_cast<llm_graph_result *>(res_raw);
        llm_graph_input_i * stored = res->add_input(std::make_unique<input_probe>(il));
        if (!stored) {
            std::fprintf(stderr,
                         "[herd] reader.input: add_input refused at layer %d; the graph will not "
                         "call set_input, so nothing was registered\n", il);
            std::abort();
        }
        g_inputs_registered.fetch_add(1);
        // The announcement is throttled; the registration is not. Without this line, "no
        // `reader.input` output" meant any of: seam not installed, `res` unreachable, `add_input`
        // refused, registered but `set_input` never called. Two earlier revisions drew conclusions
        // from that ambiguity, so the four cases are now separated by what is printed here.
        if (n <= 8) {
            std::fprintf(stderr,
                         "[herd] reader.input: registered for layer %d (call %ld) -> stored %p; "
                         "set_input has been called %ld time(s) so far\n",
                         il, n, static_cast<void *>(stored), g_input_calls.load());
            std::fflush(stderr);
        }
    }

    // THE INDEX SOURCE: upstream's own row table, and it is a ROW LIST, not a position map.
    //
    // `HANDOFF.md` §7.3 said upstream has no way to name physical cells and that a side table had
    // to be built. Half of that is wrong: upstream names them, in the `kv_rows` argument. The other
    // half of the correction is the one that matters, because the first version of this comment
    // called `kv_rows` "exactly that map" and it is not. ggml.h states the contract outright
    // (`ggml_flash_attn_ext_rows`):
    //
    //     kv_rows: [n_kv, ne3] I32, mask column i of slice i3 uses row kv_rows[i, i3] of k and v
    //     negative rows are padding, their mask entries must be -inf
    //
    // So `kv_rows` is an INDIRECTION from a MASK COLUMN to a K/V ROW. It carries no sequence and no
    // position; there is no `(seq, pos)` pair anywhere in it. The row order comes from
    // `get_seq_cells`, documented in llama-kv-cells.h as "the lists of the cells that contain any of
    // the sequences of each slice, in CELL order ... padded with -1". Two facts follow, and neither
    // is optional for a replacement:
    //
    //   * CELL order, not position order. A consumer that assumes column i is position i is wrong
    //     the moment the cache is fragmented or shared.
    //   * -1 is legal and means "padding": ggml requires that column's mask entry to be -inf. A
    //     replacement that treats -1 as an index reads row -1 of k.
    //
    //   llama-graph.cpp: build_attn_inp_attn_mask()
    //       attn_mask.rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv_rows, n_slices);
    //       ggml_set_input(attn_mask.rows);            // <- an INPUT, filled at compute time
    //       ggml_set_name(attn_mask.rows, "attn_inp_kv_rows");
    //
    //   llama-kv-cache.cpp: llama_kv_cache::set_input_attn_mask()
    //       const auto & cells = v_cells[0];
    //       int32_t * data = (int32_t *) attn_mask.rows->data;
    //       cells.get_seq_cells(get_kv_rows_slices(*ubatch, n_slices, n_swa, swa_type), data, n_rows);
    //
    // Three consequences decide the design:
    //
    //   1. `kv_rows` is an INPUT tensor, so its VALUES do not exist when this hook runs. The graph is
    //      being BUILT here; the fill happens later in `llm_graph_input_attn_kv::set_input`. A
    //      replacement must therefore CONSUME the tensor in its own graph (gather by it), not read
    //      its contents from here. Measured on the tiny model: `input 1 data no buffer no` for both
    //      `kq_mask` and `kv_rows`, every call. Reading `->data` here would read nothing on the
    //      first decode and stale bytes afterwards -- a silent wrong answer, the failure class this
    //      file refuses.
    //   2. `set_input_attn_mask` asserts `n_stream == 1` before filling rows. That part is hard. But
    //      WHETHER rows are used at all is a runtime probe, not a property of the build: the default
    //      is `cparams.fused_kv_rows = cparams.flash_attn` inside `if (auto_fkvr)`, then
    //      `resolve(llm_fused_op_flash_attn_kv_rows_probe, ...)` may rewrite it, and
    //      `LLAMA_KV_ROWS_DISABLE` forces it off. An earlier version of this comment claimed "on a
    //      non-flash build kv_rows is nullptr for EVERY call"; the probe output refuted it on this
    //      very model -- 3 decode calls with `kv_rows NULL`, then the batch call WITH
    //      `kv_rows ne [2048 1 1 1] type 26 input 1 data no`. Both regimes occur in one process.
    //   3. The two regimes are different ggml operators, not one operator with an optional argument:
    //      rows non-NULL goes through `ggml_flash_attn_ext_rows`, NULL through `ggml_flash_attn_ext`.
    //      A replacement has to pick per call, which is why the probe prints both tensors.
    //
    // `data` is printed separately from the INPUT flag because "declared an input" and "value
    // present" are different facts, and the second one is what a reader would otherwise assume.
    if (mha_probe_enabled() && n <= 4) {
        auto describe = [](const char * name, const ggml_tensor * t) {
            if (!t) {
                std::fprintf(stderr, " | %s NULL", name);
                return;
            }
            std::fprintf(stderr,
                         " | %s ne [%lld %lld %lld %lld] type %d input %d data %s buffer %s",
                         name, (long long) t->ne[0], (long long) t->ne[1],
                         (long long) t->ne[2], (long long) t->ne[3], (int) t->type,
                         (t->flags & GGML_TENSOR_FLAG_INPUT) ? 1 : 0,
                         t->data ? "yes" : "no", t->buffer ? "yes" : "no");
        };
        std::fprintf(stderr, "[herd] reader.mha: layer %d contract", il);
        describe("kq_mask", kq_mask);
        describe("kv_rows", kv_rows);
        std::fprintf(stderr, "\n");
    }

#ifdef HERD_READER_PHYSICAL
    // ===========================================================================================
    // K2: REPLACE THE ATTENTION ENTIRELY, OR DO NOTHING THIS GRAPH.
    // ===========================================================================================
    //
    // WHY THIS SITE, after two earlier attempts died here. `build_attn` (the KV overload) is the
    // only place a replacement can live: it also performs `cpy_k`/`cpy_v` (llama-graph.cpp:3707) and
    // four post-processing nodes, so replacing above it would silently stop the cache from advancing
    // -- correct first token, wrong everything after, with no error anywhere. `build_attn_mha` is
    // INSIDE that body, after the write-back, and receives k/v as the CACHE (`ne[2] == n_ctx`, measured
    // 2048 with a 2-token batch), not as this batch's new keys. That measurement is why no
    // `seq_cell_projection` call, no `mctx`, and no move to the whole-layer seam is needed: a single
    // identity span naming every cell is the whole view.
    //
    // WHY AN IDENTITY VIEW REPRODUCES UPSTREAM EXACTLY, not approximately. The emitter fills all
    // `n_cell` positions with `-INFINITY` and then writes the named cells over it, so visibility is
    // `view ∩ mask`, and `view` naming every cell makes that intersection equal the mask itself --
    // which is what upstream's `soft_max_ext(mask)` sees. The check `view_covers_finite_mask` is still
    // performed in the publisher, where the mask's VALUES finally exist, because "the view covers the
    // mask" is a claim that becomes checkable at exactly one moment and is not checkable here.
    //
    // WHAT THIS DOES NOT BUY: per-key rope. The emitter refuses `mixed_effective_pos` by name, and `q`
    // arrives already rotated, so the first admitted configuration is the identity/whole-shift view.
    // The acceptance criterion is therefore "bit-identical logits against K2 off", not a RERoT gain.
    // That is a smaller claim than the fork's and an honest one; it is what a seam can be proven to
    // do before anything is built on top of it.
    {
        static const bool k2_enabled = std::getenv("HERD_READER_GRAPH_K2") != nullptr;
        // K2c: the prefill half of the same seam. It is a SEPARATE switch because the two cover
        // disjoint token counts -- the emitter refuses `n_tokens == 1` by name, and K2 admits only
        // that case -- so turning one on never silently enables the other, and a run cannot report
        // a mixture of the two implementations as a single one.
        static const bool k2c_enabled = std::getenv("HERD_READER_GRAPH_K2C") != nullptr;
        if (k2c_enabled) {
            if (std::getenv("HERD_READER_GRAPH_K1") != nullptr) {
                std::fprintf(stderr,
                             "[herd] reader.prefill: HERD_READER_GRAPH_K1 and HERD_READER_GRAPH_K2C "
                             "are mutually exclusive (K1 observes the whole graph, K2C replaces part "
                             "of it); refusing rather than reporting one as the other\n");
                std::abort();
            }

            static std::atomic<long> k2c_replaced{0};
            static std::atomic<long> k2c_skipped{0};
            // Two counters, not one with two thresholds. A shared counter is drained by the T==1
            // reserve builds that arrive first, so the multi-token refusals -- the only ones worth
            // reading -- are never printed, and the silence reads as "the clause passed".
            static std::atomic<long> k2c_skip_multi{0};
            static std::atomic<long> k2c_skip_single{0};

            ggml_context * ctx0 = herd::graph::context_of(self);
            void * res_raw = herd::graph::result_of(self);

            const bool multi_token = q != nullptr && q->ne[2] > 1;
            const bool no_extras   = kq_b == nullptr && sinks == nullptr && v_mla == nullptr;
            // `ne[1] == n_tokens` and CONTIGUITY are K2c's extra clauses: the prefill emitter needs
            // exactly `[n_cell, n_tokens, 1, 1]`, contiguous, and a broadcast row would be read as if
            // every token had its own row. Upstream's `build_inp_attn_mask` satisfies both, but stating
            // the clause is what keeps a future upstream from passing a view silently.
            const bool mask_ok     = kq_mask != nullptr && k != nullptr && q != nullptr &&
                                     kq_mask->ne[0] == k->ne[2] &&
                                     kq_mask->ne[1] == q->ne[2] &&
                                     ggml_is_contiguous(kq_mask) &&
                                     (kq_mask->type == GGML_TYPE_F16 ||
                                      kq_mask->type == GGML_TYPE_F32);
            const bool kv_ok       = k != nullptr && v != nullptr &&
                                     k->ne[3] == 1 && v->ne[3] == 1 &&
                                     v->ne[1] == k->ne[1] && v->ne[2] == k->ne[2] &&
                                     // The emitter refuses a cell-major `v` by name.
                                     v->nb[1] <= v->nb[2];
            const bool heads_ok    = q != nullptr && k != nullptr && k->ne[1] > 0 &&
                                     q->ne[1] % k->ne[1] == 0 && q->ne[0] == k->ne[0];
            const bool ctx_ok      = ctx0 != nullptr && ggml_get_no_alloc(ctx0);
            const bool want = multi_token && no_extras && mask_ok && kv_ok && heads_ok &&
                              ctx_ok && res_raw != nullptr;

            if (!want) {
                // The cap is per-kind, not global: four T==1 reserve builds must not spend the
                // budget of the multi-token builds whose refusal is the one worth reading.
                const bool show = multi_token ? (k2c_skip_multi.fetch_add(1) < 8)
                                              : (k2c_skip_single.fetch_add(1) < 4);
                if (show) {
                    // `ne3` is printed because q->ne[2] > 1 and a stacked batch on k/v are the two
                    // shapes that decide this clause; without it a refusal on the stacked axis is
                    // indistinguishable from a refusal on the token axis.
                    std::fprintf(stderr,
                                 "[herd] reader.prefill: layer %d falls back (multi_token %d extras %d "
                                 "mask %d kv %d heads %d no_alloc %d res %d) q ne %lld %lld %lld %lld, "
                                 "k ne %lld %lld %lld %lld, v ne %lld %lld %lld %lld\n",
                                 il, multi_token ? 1 : 0, no_extras ? 1 : 0, mask_ok ? 1 : 0,
                                 kv_ok ? 1 : 0, heads_ok ? 1 : 0, ctx_ok ? 1 : 0,
                                 res_raw ? 1 : 0,
                                 q ? (long long) q->ne[0] : -1, q ? (long long) q->ne[1] : -1,
                                 q ? (long long) q->ne[2] : -1, q ? (long long) q->ne[3] : -1,
                                 k ? (long long) k->ne[0] : -1, k ? (long long) k->ne[1] : -1,
                                 k ? (long long) k->ne[2] : -1, k ? (long long) k->ne[3] : -1,
                                 v ? (long long) v->ne[0] : -1, v ? (long long) v->ne[1] : -1,
                                 v ? (long long) v->ne[2] : -1, v ? (long long) v->ne[3] : -1);
                    std::fflush(stderr);
                }
            } else {
                const std::size_t n_cell   = (std::size_t) k->ne[2];
                const std::size_t n_tokens = (std::size_t) q->ne[2];
                const std::size_t head_dim = (std::size_t) q->ne[0];
                const std::size_t n_head   = (std::size_t) q->ne[1];
                const std::size_t n_head_kv = (std::size_t) k->ne[1];

                tq::reader::query_view view;
                {
                    tq::reader::span_keys span;
                    span.effective_pos = 0;
                    span.cells.resize(n_cell);
                    for (std::size_t i = 0; i < n_cell; ++i) {
                        span.cells[i] = (std::int32_t) i;
                    }
                    view.spans.push_back(std::move(span));
                }

                // ARENA ADMISSION IS DECIDED BEFORE A NODE EXISTS. A capacity limit is not a
                // disagreement between the clauses above and the emitter -- it is the graph arena
                // being too small for `O(n_tokens * n_cell)` full-cache attention -- so it falls back
                // to upstream and is counted, while a structural refusal aborts.
                bool overflow = false;
                const std::size_t objects =
                    tq::reader::prefill_object_count(n_tokens, view.spans.size(), true, &overflow);
                const std::size_t free_bytes =
                    overflow ? 0 : ggml_get_mem_size(ctx0) - ggml_used_mem(ctx0);

                // The charge per object is `ggml_tensor_overhead()`, the figure ggml itself
                // reserves -- not a guessed 4096. A guess ten times too high vetoes a graph that
                // would have fit, and a veto is silent: it looks exactly like "the seam refused".
                const std::size_t per_object = (std::size_t) ggml_tensor_overhead();
                const bool arena_ok =
                    !overflow && free_bytes != 0 &&
                    objects <= free_bytes / (per_object == 0 ? 1 : per_object);

                if (!arena_ok) {
                    // A skip on a MULTI-TOKEN build is never swallowed by decode noise: the first
                    // 8 are printed unconditionally. The earlier cap of 4 was spent on the T==1
                    // reserve builds, which is what let a wrong verdict look measured.
                    const bool show = multi_token ? (k2c_skipped.fetch_add(1) < 8)
                                                  : (k2c_skipped.fetch_add(1) < 4);
                    if (show) {
                        std::fprintf(stderr,
                                     "[herd] reader.prefill: layer %d falls back (multi_token %d, "
                                     "arena %zu B free, %zu objects x %zu B = %zu B, %zu tokens x "
                                     "%zu cells)\n",
                                     il, multi_token ? 1 : 0, free_bytes, objects, per_object,
                                     objects * per_object, n_tokens, n_cell);
                        std::fflush(stderr);
                    }
                } else {
                    tq::reader::prefill_inputs gi;
                    gi.ctx = ctx0;
                    gi.k = k;
                    gi.v = v;
                    // Upstream `q` is `[head_dim, n_head, n_tokens]`; the emitter takes
                    // `[head_dim, n_tokens, n_head]`. A PERMUTE, never a reshape.
                    gi.q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                    gi.mask = kq_mask;
                    gi.view = std::move(view);
                    gi.coverage = tq::reader::graph_inputs::mask_coverage::intersect;
                    gi.n_head = n_head;
                    gi.n_head_kv = n_head_kv;
                    gi.head_dim = head_dim;
                    gi.n_cell = n_cell;

                    tq::reader::prefill_result em = tq::reader::emit_prefill_attention(gi, kq_scale);
                    if (em.refusal != tq::reader::prefill_refusal::ok) {
                        // Structural refusal only: the arena clause above already removed the one
                        // refusal that is a capacity statement. A graph that ran half replaced would
                        // report one implementation's output as the other's.
                        std::fprintf(stderr,
                                     "[herd] reader.prefill: layer %d REFUSED: %s (%s); a partially "
                                     "replaced graph would report one implementation as the other\n",
                                     il, tq::reader::describe(em.refusal),
                                     em.why.empty() ? "no detail" : em.why.c_str());
                        std::abort();
                    }

                    // The publisher takes the result BY VALUE, so what the log and the return value
                    // need is captured here, before the move. There is no attach(): the constructor
                    // validates the emission and aborts, which is the same policy as the REFUSED
                    // branch above and needs no second test.
                    ggml_tensor * const out_tensor = em.out;
                    const std::size_t em_bytes = em.bytes_required;

                    auto stats = std::make_shared<herd::reader::prefill_publisher_stats>();
                    auto publisher =
                        std::make_unique<herd::reader::prefill_publisher>(std::move(em), stats);

                    // THE MASK COVERAGE CHECK FOR PREFILL, WHICH HAS NOWHERE ELSE TO LIVE.
                    // `prefill_publisher` writes index tensors only, so without this object the
                    // finite-bias contract is asserted by comment alone: a mask narrower than the
                    // declared view would publish silently and the graph would compute a narrower
                    // attention than upstream's, with numbers that look correct. Unlike K2's
                    // single-token case, `kq_mask` here is `[n_cell, n_tokens, 1, 1]`, so each token
                    // has its OWN row and the check is per row -- row `t` starts at `t * n_cell`.
                    struct prefill_mask_input final : llm_graph_input_i {
                        tq::reader::query_view view;
                        ggml_tensor *          mask = nullptr;
                        std::size_t            n_cell = 0;
                        std::size_t            n_tokens = 0;
                        int                    layer = 0;
                        // No `widened` buffer here: the F16-to-F32 widening happens inside
                        // `view_covers_finite_mask_row`, so this object holds no scratch bytes that
                        // a caller could forget to resize, and there is nothing to keep in step
                        // with the tested widening.

                        void set_input(const llama_ubatch * ubatch) override {
                            (void) ubatch;
                            if (mask == nullptr || mask->data == nullptr) {
                                return;
                            }
                            const tq::reader::mask_width width =
                                mask->type == GGML_TYPE_F16 ? tq::reader::mask_width::f16
                                                            : tq::reader::mask_width::f32;
                            for (std::size_t t = 0; t < n_tokens; ++t) {
                                // The row START comes from the tested function. The element width and
                                // the `t * n_cell` product are exactly what the row form's test
                                // drives, so this loop adds no second copy of them: the arithmetic
                                // that used to sit here as `base + t * n_cell * elem` was the last
                                // step between this call site and the tested predicate.
                                const auto cov = tq::reader::view_covers_finite_mask_row(
                                    view,
                                    tq::reader::finite_mask_row(mask->data, t, n_cell, width),
                                    width, n_cell);
                                if (!(cov.applicable && cov.complete)) {
                                    std::fprintf(stderr,
                                                 "[herd] reader.prefill: layer %d token %zu has a "
                                                 "finite bias at cache position %zu that the view "
                                                 "does not name (%zu of %zu finite positions "
                                                 "named); the emission is NARROWER than upstream, so "
                                                 "the graph must not run\n",
                                                 layer, t, cov.first_uncovered, cov.named,
                                                 cov.finite_positions);
                                    std::abort();
                                }
                            }
                            if (layer == 0) {
                                std::fprintf(stderr,
                                             "[herd] reader.prefill: layer 0 mask coverage ok "
                                             "(%zu tokens x %zu cells)\n", n_tokens, n_cell);
                                std::fflush(stderr);
                            }
                        }
                    };

                    auto mask_input = std::make_unique<prefill_mask_input>();
                    mask_input->mask = kq_mask;
                    mask_input->n_cell = n_cell;
                    mask_input->n_tokens = n_tokens;
                    mask_input->layer = il;
                    // An identity view OVER EVERY CELL, rebuilt here rather than moved out of `view`:
                    // `view` has already been moved into the emission, and naming all cells cannot
                    // narrow the mask, so this is the same view the emitter computed its own
                    // intersection against.
                    {
                        tq::reader::span_keys span;
                        span.effective_pos = 0;
                        span.cells.resize(n_cell);
                        for (std::size_t i = 0; i < n_cell; ++i) {
                            span.cells[i] = (std::int32_t) i;
                        }
                        mask_input->view.spans.push_back(std::move(span));
                    }

                    auto * res = static_cast<llm_graph_result *>(res_raw);
                    if (res->add_input(std::move(mask_input)) == nullptr) {
                        std::fprintf(stderr,
                                     "[herd] reader.prefill: layer %d add_input refused for the "
                                     "mask check; the finite-bias contract would go unverified\n",
                                     il);
                        std::abort();
                    }
                    if (res->add_input(std::move(publisher)) == nullptr) {
                        std::fprintf(stderr,
                                     "[herd] reader.prefill: layer %d add_input refused; without it "
                                     "the index tensors are placed but never filled\n", il);
                        std::abort();
                    }

                    const long replaced = k2c_replaced.fetch_add(1) + 1;
                    if (replaced <= 8) {
                        std::fprintf(stderr,
                                     "[herd] reader.prefill: layer %d REPLACED (n_tokens %zu, "
                                     "n_cell %zu, heads %zu/%zu, head_dim %zu, arena %zu B in %zu "
                                     "objects)\n",
                                     il, n_tokens, n_cell, n_head, n_head_kv, head_dim,
                                     em_bytes, objects);
                        std::fflush(stderr);
                    }

                    // THE RETURN SHAPE IS UPSTREAM'S RANK-2 ORDER, RESTORED. `em.out` is
                    // `[head_dim, n_tokens, n_head]` in linear order `d + head_dim*(t + T*h)`, while
                    // upstream returns `[head_dim*n_head, n_tokens]` in order `d + head_dim*h +
                    // head_dim*n_head*t`. For `T == 1` those coincide, which is why K2's `reshape_1d`
                    // is right there and would be WRONG here: it silently transposes every token.
                    return ggml_cont_2d(ctx0,
                                        ggml_permute(ctx0, out_tensor, 0, 2, 1, 3),
                                        (std::int64_t) (head_dim * n_head),
                                        (std::int64_t) n_tokens);
                }
            }
        } else if (k2_enabled) {
            // K1 intercepts `build_graph`; K2 replaces the attention inside it. Both on would report a
            // replaced graph through a seam that only claims to have observed one, so the pair is
            // refused at the first call instead of producing a number nobody can attribute.
            if (std::getenv("HERD_READER_GRAPH_K1") != nullptr) {
                std::fprintf(stderr,
                             "[herd] reader.k2: HERD_READER_GRAPH_K1 and HERD_READER_GRAPH_K2 are "
                             "mutually exclusive (K1 observes the whole graph, K2 replaces part of "
                             "it); refusing rather than reporting one as the other\n");
                std::abort();
            }

            static std::atomic<long> k2_replaced{0};
            static std::atomic<long> k2_skipped{0};

            ggml_context * ctx0 = herd::graph::context_of(self);
            void * res_raw = herd::graph::result_of(self);

            // THE ADMISSION CLAUSES. Each one has been the specific reason an earlier attempt was
            // wrong, so each is a clause and not a comment.
            const bool single_token = q != nullptr && q->ne[2] == 1;
            const bool no_extras    = kq_b == nullptr && sinks == nullptr && v_mla == nullptr;
            const bool mask_ok      = kq_mask != nullptr && k != nullptr &&
                                      kq_mask->ne[0] == k->ne[2] &&
                                      (kq_mask->type == GGML_TYPE_F16 ||
                                       kq_mask->type == GGML_TYPE_F32);
            const bool kv_ok        = k != nullptr && v != nullptr &&
                                      k->ne[3] == 1 && v->ne[3] == 1 &&
                                      v->ne[1] == k->ne[1] && v->ne[2] == k->ne[2];
            const bool heads_ok     = q != nullptr && k != nullptr && k->ne[1] > 0 &&
                                      q->ne[1] % k->ne[1] == 0 && q->ne[0] == k->ne[0];
            const bool ctx_ok       = ctx0 != nullptr && ggml_get_no_alloc(ctx0);
            const bool want = single_token && no_extras && mask_ok && kv_ok && heads_ok &&
                              ctx_ok && res_raw != nullptr;

            if (!want) {
                // The reason is printed for the FIRST few skips only: a reserve graph skips every
                // layer, and a hundred identical lines hide the one that matters.
                if (k2_skipped.fetch_add(1) < 4) {
                    std::fprintf(stderr,
                                 "[herd] reader.k2: layer %d falls back (single_token %d extras %d "
                                 "mask %d kv %d heads %d no_alloc %d res %d)\n",
                                 il, single_token ? 1 : 0, no_extras ? 1 : 0, mask_ok ? 1 : 0,
                                 kv_ok ? 1 : 0, heads_ok ? 1 : 0, ctx_ok ? 1 : 0,
                                 res_raw ? 1 : 0);
                    std::fflush(stderr);
                }
            } else {
                const std::size_t n_cell   = (std::size_t) k->ne[2];
                const std::size_t head_dim = (std::size_t) q->ne[0];
                const std::size_t n_head   = (std::size_t) q->ne[1];

                tq::reader::graph_inputs gi;
                gi.ctx = ctx0;
                gi.k = k;
                gi.v = v;
                // `q` is [head_dim, n_head, n_tokens] at this seam (measured nb[1] == head_dim * 4, so
                // the head axis is ne[1]); the emitter takes [head_dim, 1, n_head]. A PERMUTE, not a
                // reshape: reshape keeps the data contiguous and silently transposes every head, which
                // produces plausible numbers instead of an error. With n_tokens == 1 the permute is
                // value-identical to the identity, which is exactly why the emitter's linear order
                // (`d + head_dim*head`) can be compared against upstream's at all.
                gi.q = ggml_permute(ctx0, q, 0, 2, 1, 3);
                gi.mask = kq_mask;
                gi.coverage = tq::reader::graph_inputs::mask_coverage::intersect;
                gi.collect_indices = true;
                gi.n_head = n_head;
                gi.n_head_kv = (std::size_t) k->ne[1];
                gi.head_dim = head_dim;
                gi.n_cell = n_cell;

                // EVERY CELL, IN CELL ORDER. This is the identity view, and it is the only view this
                // seam can justify: `kv_rows` is an INPUT tensor whose values do not exist yet, and its
                // order is `get_seq_cells`'s cell order -- which for one stream over a live cache is
                // the cells themselves, names omitted. Naming all of them cannot narrow the mask, so
                // the intersection the emitter computes is the mask.
                tq::reader::query_view view;
                {
                    tq::reader::span_keys span;
                    span.effective_pos = 0;
                    span.cells.resize(n_cell);
                    for (std::size_t i = 0; i < n_cell; ++i) {
                        span.cells[i] = (std::int32_t) i;
                    }
                    view.spans.push_back(std::move(span));
                }

                tq::reader::physical_emit_result em =
                    tq::reader::emit_physical_attention(gi, view, kq_scale);
                if (em.refusal != tq::reader::physical_emit_refusal::ok) {
                    // No fallback here on purpose. The clauses above admitted this call, so a refusal
                    // is a disagreement between the admission and the emitter -- and a graph that ran
                    // half replaced would report one implementation's output as another's.
                    std::fprintf(stderr,
                                 "[herd] reader.k2: layer %d REFUSED: %s (%s); a partially replaced "
                                 "graph would report one implementation as the other\n",
                                 il, tq::reader::describe(em.refusal),
                                 em.why.empty() ? "no detail" : em.why.c_str());
                    std::abort();
                }

                ggml_tensor * const out_tensor = em.out;

                // THE PUBLISHER. `publish_index_inputs` is proven correct on its own; what it needs is
                // a caller between placement and compute, and upstream has exactly one: every
                // `llm_graph_input_i` in the result gets `set_input(ubatch)` there
                // (llama-context.cpp: alloc 2054, set_inputs 2068). The tensors survive the move into
                // this object because they belong to the SAME graph context that owns it -- and unlike
                // the deleted `index_holder`, "the graph died but the publisher still writes" is not
                // representable.
                //
                // `can_reuse` is deliberately NOT overridden: the base default is false, and this
                // object holds a per-batch emission, so reusing its graph would publish one batch's
                // indices into another's placement.
                struct k2_input final : llm_graph_input_i {
                    tq::reader::physical_emit_result em;
                    tq::reader::query_view            view;
                    ggml_tensor *                     mask = nullptr;
                    std::size_t                       n_cell = 0;
                    int                               layer = 0;
                    // No `widened` buffer: the widening is the shared helper's, not this object's.

                    void set_input(const llama_ubatch * ubatch) override {
                        (void) ubatch;

                        // THE COVERAGE CHECK RUNS HERE BECAUSE NOWHERE ELSE HAS THE VALUES.
                        // `kq_mask` is a `ggml_set_input` tensor: measured `data == nullptr` at build
                        // time, and filled by `set_input_attn_mask` before this call. F16 is widened by
                        // hand -- `view_covers_finite_mask` takes F32 and has no overload on purpose,
                        // because widening needs a destination the caller must own.
                        if (mask != nullptr && mask->data != nullptr) {
                            // ONE ROW, so the row start IS `mask->data`. The width switch and the
                            // widening live in `view_covers_finite_mask_row` -- the same code the
                            // prefill path runs and the mask tests drive -- so the F16 branch here
                            // cannot be a second, untested copy of that arithmetic.
                            const auto cov = tq::reader::view_covers_finite_mask_row(
                                view, mask->data,
                                mask->type == GGML_TYPE_F16 ? tq::reader::mask_width::f16
                                                            : tq::reader::mask_width::f32,
                                n_cell);
                            if (!(cov.applicable && cov.complete)) {
                                std::fprintf(stderr,
                                             "[herd] reader.k2: layer %d mask has a finite bias at "
                                             "cache position %zu that the view does not name "
                                             "(%zu of %zu finite positions named); the emission is "
                                             "NARROWER than upstream, so the graph is discarded\n",
                                             layer, cov.first_uncovered, cov.named,
                                             cov.finite_positions);
                                std::abort();
                            }
                            if (layer == 0) {
                                std::fprintf(stderr,
                                             "[herd] reader.k2: layer 0 mask coverage ok (%zu finite "
                                             "of %zu named)\n", cov.finite_positions, cov.named);
                            }
                        }

                        // THROUGH THE BACKEND, ALWAYS. `t->data` non-null means the tensor was PLACED,
                        // not that the host may dereference it: on a device backend that is an address
                        // in device memory.
                        auto writer = [](ggml_tensor * t, const void * data, std::size_t offset,
                                         std::size_t size) -> bool {
                            if (t == nullptr || t->data == nullptr) {
                                return false;
                            }
                            ggml_backend_tensor_set(t, data, offset, size);
                            return true;
                        };

                        std::string why;
                        std::size_t failed_span = 0;
                        const auto r = tq::reader::publish_index_inputs(em, writer, &why,
                                                                       &failed_span);
                        if (r != tq::reader::index_publish_refusal::ok) {
                            // A partially published input set is the silent-wrong-answer case: ggml
                            // would read uninitialized memory as row indices. The graph is discarded
                            // rather than retried with a mixture of two batches.
                            std::fprintf(stderr,
                                         "[herd] reader.k2: layer %d publish REFUSED: %s (%s) at "
                                         "span %zu; the graph must not run\n",
                                         layer, tq::reader::describe(r),
                                         why.empty() ? "no detail" : why.c_str(), failed_span);
                            std::abort();
                        }
                    }
                };

                std::unique_ptr<llm_graph_input_i> publisher = std::make_unique<k2_input>();
                auto * k2 = static_cast<k2_input *>(publisher.get());
                k2->em = std::move(em);
                k2->view = std::move(view);
                k2->mask = kq_mask;
                k2->n_cell = n_cell;
                k2->layer = il;

                auto * res = static_cast<llm_graph_result *>(res_raw);
                if (res->add_input(std::move(publisher)) == nullptr) {
                    std::fprintf(stderr,
                                 "[herd] reader.k2: layer %d add_input refused; without it the index "
                                 "tensors are placed but never filled\n", il);
                    std::abort();
                }

                const long replaced = k2_replaced.fetch_add(1) + 1;
                if (replaced <= 8) {
                    std::fprintf(stderr,
                                 "[herd] reader.k2: layer %d REPLACED (n_cell %zu, heads %zu/%zu, "
                                 "head_dim %zu, arena %zu B in %zu objects)\n",
                                 il, n_cell, n_head, gi.n_head_kv, head_dim,
                                 em.bytes_required, em.objects_charged);
                    std::fflush(stderr);
                }
                if (std::getenv("HERD_READER_K2_FINAL") != nullptr) {
                    std::atexit([]() {
                        std::fprintf(stderr, "[herd] reader.k2: total replaced %ld, skipped %ld\n",
                                     k2_replaced.load(), k2_skipped.load());
                    });
                }

                // THE RETURN SHAPE IS UPSTREAM'S, AND IT WAS MEASURED RATHER THAN ASSUMED. On the tiny
                // model the live call returns rank 1 `[head_dim*n_head]` F32, in the order
                // `d + head_dim*head` -- the same order this emitter's `[head_dim, 1, n_head]` holds
                // (`upstream-attention-contrast-test` agrees to 2.98e-08 on two spans).
                return ggml_reshape_1d(ctx0, out_tensor,
                                       (std::int64_t) (head_dim * n_head));
            }
        }
    }
#endif // HERD_READER_PHYSICAL

    ggml_tensor * out = g_orig_mha(self, q, k, v, kq_b, kq_mask, kv_rows, sinks, v_mla,
                                   n_kv_max, kq_scale, il);
    // The RETURN is the contract a replacement has to match, and it was the one thing this probe
    // did not print -- the comment above asserted a shape nobody had measured. Upstream returns a
    // different rank per branch: flash attention returns the permuted 3-D result, the non-flash
    // branch ends with `ggml_cont_2d`, so the same numbers arrive flattened. Printing the rank is
    // what makes that difference visible instead of assumed.
    if (mha_probe_enabled() && n <= 4) {
        std::fprintf(stderr,
                     "[herd] reader.mha: layer %d -> return rank %d ne [%lld %lld %lld %lld] type %d\n",
                     il, ggml_n_dims(out),
                     (long long) out->ne[0], (long long) out->ne[1],
                     (long long) out->ne[2], (long long) out->ne[3], (int) out->type);
    }
    return out;
}

bool trace_enabled() {
    static const bool enabled = std::getenv("HERD_READER_TRACE") != nullptr;
    return enabled;
}

// The graph-context probe, opt-in and separate from tracing on purpose: tracing only reads
// the arguments it is given, while the probe MAKES ggml allocate inside the live context.
// One is an observation; the other changes what it observes by a tensor header. They must
// not share a switch, or a diagnostic run would differ from a traced one for a reason
// nobody intended.
bool probe_enabled() {
    static const bool enabled = std::getenv("HERD_READER_PROBE") != nullptr;
    return enabled;
}

// The mutating probe, behind a second switch.
//
// HERD_READER_PROBE makes the hook read the graph context; HERD_READER_PROBE_DEEP additionally
// makes it create a tensor inside that context. Two switches for two different costs: the first
// only observes, the second changes the graph's arena by one header -- permanently, since a
// ggml arena has no per-tensor free. A diagnostic that mutates what it measures must be
// something the operator opted into by name, not something a general "trace" flag turned on.
bool deep_probe_enabled() {
    static const bool enabled = std::getenv("HERD_READER_PROBE_DEEP") != nullptr;
    return enabled;
}

// The cache probe, behind a third switch, because its cost is a third thing.
//
// HERD_READER_PROBE reads the arena's counters (free). HERD_READER_PROBE_DEEP spends one tensor
// header once per process. This one calls mctx->get_k/get_v, which allocate VIEW NODES IN THE
// LIVE GRAPH on EVERY call -- so with one call per layer, a 32-layer model gets 64 extra nodes
// per graph build that the original is about to allocate again. That is the same perturbation
// class as the deep probe, one order of magnitude larger, which is why it does not share a
// switch with either.
//
// It exists to answer one question that nothing else in this tree has answered by measurement:
// can the reader lane obtain the K/V cache tensors from outside upstream, through public API,
// while the graph is being built? The alternative was to conclude, as an earlier note in this
// file did, that the cache was unreachable and that the hook would have to replace upstream's
// body. That conclusion was wrong -- build_attn's parameter is already the concrete input type
// and its cache context is a public member -- and a claim like that deserves a number rather
// than a paragraph.
bool cache_probe_enabled() {
    static const bool enabled = std::getenv("HERD_READER_CACHE") != nullptr;
    return enabled;
}

// The switch that ASKS, not one that acts -- and the distinction is the whole of this round.
//
// HERD_READER_TRACE observes arguments. HERD_READER_PROBE reads the arena. HERD_READER_CACHE
// reads the cache. Those three read facts and cannot change a number. This one names a request
// that this build cannot honour, so reading it is immediately followed by a refusal: the process
// stops before a model loads, with a status that says so.
//
// It used to hand the layer's attention to the emitter. It does not any more, because the
// replacement was unsound in three ways that are properties of upstream's attention -- see
// herd_try_emit's block in this file. Keeping the name would have been the cheapest way to leave
// the impression that a reader path exists; `emit_now_requested` says what it actually is.
// Whether the operator asked for the layer's attention to be REPLACED.
//
// Named for the request rather than the capability, because those are no longer the same thing:
// the request exists and this build refuses it (see the gate in herd_reader_attn_autoload). The
// name keeps the refusal site readable -- `if (emit_now_requested()) { refuse }` says what is
// being refused and why it is not `if (emit_now_enabled())`.
//
// A second reader would be a second place to keep true, so the autoload gate and the hook's own
// fatal check both call this.
bool emit_now_requested() {
    static const bool enabled = std::getenv("HERD_READER_EMIT_NOW") != nullptr;
    return enabled;
}

// WHERE THE EMISSION WOULD GO, and why it is not here.
//
// The hook now reads this layer's K/V cache (HERD_READER_CACHE) and reports its shapes, which is
// the fact the emitter needs and could not previously obtain. The next step -- calling
// emit_reader_attention() on those tensors -- is deliberately absent, and the reasons are
// recorded rather than implied:
//
//   1. no request source. A reader emission is supposed to be conditional on a view a caller
//      asked for; the station that carries that request exists (tq::reader::station) and no
//      product opens a scope on it. Without a request, an emission here is unconditional and the
//      caller has no way to not get it.
//   2. arena cost, unbounded. The emitter widens the cache (ggml_cast per head) and gathers per
//      span; on a 2048-cell cache that is hundreds of KB per layer per build, which a ggml arena
//      never reclaims. There is no headroom check on that path -- unlike the cache probe, which
//      grew one after ggml_new_tensor_impl's assert was found to end the process.
//   3. unverified in place. The emitter's correctness is established in modules/reader against
//      ggml's kernels, on fixtures whose layouts are asserted. Running it on this hook's tensors
//      is a different claim (real strides, real types, real arena) and has not been measured.
//
// So the emission stays out of the product binary until all three are answered. What is verified
// -- the emitter itself, including the F16 cache path this model actually uses -- lives in
// modules/reader with its own tests.

void describe(const char * label, const ggml_tensor * t) {
    if (!t) {
        std::fprintf(stderr, "[herd] reader: %-8s (null)\n", label);
        return;
    }
    std::fprintf(stderr, "[herd] reader: %-8s ne = [%lld, %lld, %lld, %lld] type %d\n",
                 label, (long long) t->ne[0], (long long) t->ne[1],
                 (long long) t->ne[2], (long long) t->ne[3], (int) t->type);
}

// The replacement path: emit, project, return -- or decline and defer.
//
// Gated by HERD_READER_EMIT_NOW, a fourth switch, because this is the first one that CHANGES the
// model's numbers. The earlier switches read; this one decides.
//
// What it reproduces from llama-graph.cpp:3579-3685, and what it refuses:
//
//   reproduced  the cache read (mctx->get_k/get_v), the query as it arrives, the kq scale, the
//               output projection (ggml_mul_mat(wo, cur)), and the bias add.
//   refused     anything the preconditions in herd_graph_cache can detect -- Hadamard rotations,
//               dequantisation scales, an empty cache -- plus, from the arguments: kq_b, sinks,
//               v_mla, wo_s, and a multi-token query.
//
// THE REPLACEMENT ASSEMBLY IS DELETED, and the reasoning that condemned it is here instead.
//
// It had no caller (`-Wunused-function` said so), it was marked [[maybe_unused]] to keep the
// compiler quiet, and its author kept it "so the next person does not write it again". That is a
// poor trade: dead code that looks live is a liability in every later edit -- a rename, a
// signature change, or a search for "who calls the emitter" all have to reason about a function
// that never runs -- while a paragraph costs nothing to read and nothing to maintain.
//
// The four refutations it carried, kept because each one is a design constraint the next attempt
// must satisfy rather than rediscover:
//
//   1. `get_n_kv()` is the cache's EXTENT, not the set of cells this sequence may attend to.
//      Holes from eviction and other sequences' cells are in that range. Gathering 0..n_kv-1
//      therefore reads keys the query must not see, and produces finite, plausible, wrong
//      numbers -- the failure mode this whole lane exists to remove.
//   2. The fork does not gather-and-attend at all. It scatters per-span scores into a
//      full-cache-length tensor filled with -INFINITY (`ggml_set_rows`) and normalises once, which
//      is what preserves the native reduction order -- its own comment says a compact sum can
//      cross a later quantization boundary. A compact union sum is therefore not an equal
//      substitute, by the fork authors' design.
//   3. `wo_s` (LoRA output scaling), `kq_b`, `sinks` and `v_mla` are part of what the replaced call
//      does. A replacement that ignores them changes the answer wherever they are non-null.
//   4. An arena admission test does not exist. `ggml_new_object` aborts rather than returning null
//      (ggml.c:1832), so an emission that does not fit kills the process mid-decode; arena_budget
//      is advisory and estimate_drift is post-hoc.
//
// The refusal below names these. Building them is the work; a fall-through was measured and
// rejected (every layer declined, stock numbers, exit 0).

// The replacement. It observes and defers; it never decides.
ggml_tensor * herd_build_attn(const void * self, const void * inp,
                              ggml_tensor * wo, ggml_tensor * wo_b, ggml_tensor * wo_s,
                              ggml_tensor * q_cur, ggml_tensor * k_cur, ggml_tensor * v_cur,
                              ggml_tensor * kq_b, ggml_tensor * sinks, ggml_tensor * v_mla,
                              float kq_scale, int il) {
    ++g_calls;

    if (trace_enabled()) {
        std::fprintf(stderr, "[herd] reader: layer %d, scale %g\n", il, (double) kq_scale);
        describe("q_cur", q_cur);
        describe("k_cur", k_cur);
        describe("v_cur", v_cur);
        describe("kq_b", kq_b);
    }

    // The graph context behind this call, and proof that it is one.
    //
    // `self` is the llm_graph_context subobject pointer -- a member function always
    // receives the pointer adjusted for its own class, whichever derived graph context the
    // call came through. That is what lets one accessor serve every build_attn overload,
    // and it is why the probe can run from inside a single hook.
    //
    // The probe is opt-in and never runs in a product: ggml_new_tensor_1d writes into the
    // context's arena, and a diagnostic must not change what it measures unless the
    // operator asked for the measurement.
    if (probe_enabled()) {
        ggml_context * ctx = herd::graph::context_of(self);
        if (!ctx) {
            std::fprintf(stderr,
                         "[herd] reader: the graph context is null at layer %d; a reader "
                         "emission could not create a node here\n",
                         il);
        } else {
            // The allocation-free check first: it reports the arena's size and usage on every
            // layer, which is what makes the pointer's identity visible (a wrong offset would
            // not produce a used/size pair that grows by graph construction). It does NOT
            // validate the pointer -- reading through a wrong pointer faults, and no check in
            // this file can change that; the header says so.
            std::size_t used = 0;
            std::size_t size = 0;
            if (!herd::graph::arena_usage(ctx, &used, &size)) {
                std::fprintf(stderr,
                             "[herd] reader: layer %d: the arena reports used %zu > size %zu, "
                             "so this graph context is not trustworthy; a reader emission must "
                             "refuse rather than build on it\n",
                             il, used, size);
            } else if (deep_probe_enabled()) {
                // The mutating check, one shot for the whole process: it appends a tensor
                // header to the live arena, so it is behind a SECOND switch and cannot run
                // once per layer.
                std::size_t after = 0;
                std::size_t after_size = 0;
                switch (herd::graph::probe_one_tensor(ctx, &after, &after_size)) {
                    case herd::graph::probe_result::created:
                        break;   // probe_one_tensor printed the measurement
                    case herd::graph::probe_result::exhausted:
                        std::fprintf(stderr,
                                     "[herd] reader: layer %d: no headroom for the deep probe; "
                                     "the arena is full, which is not a verdict on the context\n",
                                     il);
                        break;
                    case herd::graph::probe_result::already_done:
                        break;   // the one allowed probe was spent on an earlier layer
                    case herd::graph::probe_result::null_context:
                        break;   // unreachable: ctx is non-null here
                }
            } else {
                std::fprintf(stderr,
                             "[herd] reader: layer %d: arena %zu of %zu bytes used "
                             "(allocation-free check; HERD_READER_PROBE_DEEP spends the one "
                             "mutating probe)\n",
                             il, used, size);
            }
        }
    }

    // The cache read, behind its own switch. This is the capability the reader lane's emission
    // needs and did not have: the K/V tensors for this layer, obtained through public API while
    // the graph is live.
    //
    // The numbers printed are the contract check the emitter will need. Its declared input is
    // K [head_dim, n_head_kv, n_cell] and V either that or the cell-major permutation,
    // distinguished by nb[1] > nb[2] -- so printing ne and nb together is what lets a reader
    // tell "the access worked" from "the access returned something shaped plausibly".
    if (cache_probe_enabled()) {
        herd::graph::cache_view cache;
        // Called ONCE. The failure branch of the first version called cache_tensors() again to
        // build its message, which on a success-then-print path would have allocated a second
        // pair of view nodes -- a diagnostic whose cost depended on which branch printed.
        const herd::graph::cache_status cache_state =
            herd::graph::cache_tensors(inp, herd::graph::context_of(self), il, &cache);
        if (cache_state == herd::graph::cache_status::ok) {
            std::fprintf(stderr,
                         "[herd] reader: layer %d cache k ne [%lld %lld %lld %lld] nb [%zu %zu]"
                         " type %d | v ne [%lld %lld %lld %lld] nb [%zu %zu] type %d | n_kv %u"
                         " | v_cell_major %d\n",
                         il,
                         (long long) cache.ne_k[0], (long long) cache.ne_k[1],
                         (long long) cache.ne_k[2], (long long) cache.ne_k[3],
                         cache.nb_k[1], cache.nb_k[2], cache.type_k,
                         (long long) cache.ne_v[0], (long long) cache.ne_v[1],
                         (long long) cache.ne_v[2], (long long) cache.ne_v[3],
                         cache.nb_v[1], cache.nb_v[2], cache.type_v, cache.n_kv,
                         cache.nb_v[1] > cache.nb_v[2] ? 1 : 0);
        } else {
            // Named, never silent. A reader emission that cannot see the cache must refuse; a
            // hook that printed nothing here would make the refusal indistinguishable from a
            // build that was never asked to emit.
            std::fprintf(stderr, "[herd] reader: layer %d no cache view: %s\n", il,
                         herd::graph::describe(cache_state));
        }

        // The emission, on the tensors just read -- the real F16 cache of a real model.
        //
        // It is NOT attempted here. The authoritative list of what is missing is "WHAT IS GENUINELY
        // MISSING" below, in build order; an earlier revision of this comment summarised a
        // DIFFERENT list (a request source, arena headroom, unverified strides) while the block
        // below named four other items, so the same function carried two incompatible answers to
        // "why not". A summary that disagrees with the thing it summarises is worse than no
        // summary: the reader who trusts the nearer comment builds the wrong piece first.
        //
        // What this branch establishes is the prerequisite only: the tensors are obtainable from
        // outside upstream, during a live graph build, and their shapes can be compared against
        // the emitter's declared contract instead of assumed.
    }

    // NO REPLACEMENT, AND NO FALL-THROUGH EITHER.
    //
    // The reasons below were REWRITTEN after reading this fork's actual reader implementation
    // (atomic-llama-cpp-turboquant/src/llama-graph.cpp:3614 `build_attn_reader`). The previous
    // list named two causes that were wrong, and a refusal that names a wrong cause is itself a
    // defect -- it sends the next reader to build the wrong thing, which is what happened here.
    //
    // WHAT THE FORK ACTUALLY DOES, and it is not what this lane has been building:
    //
    //   * it does NOT gather a cell set and attend over it alone. It computes partial scores per
    //     span, then SCATTERS them into a full-cache-length tensor filled with -INFINITY via
    //     `ggml_set_rows(physical, rows, reader.indices)`, and normalises ONCE over that full
    //     length. Hidden cells carry -inf, so they get zero weight. Visibility is expressed by
    //     the -inf fill, not by a "visible cell list".
    //   * the comment there is explicit about why: "Preserve the native physical reduction order
    //     and padding. A tiny change in a compact sum can cross a later quantization boundary."
    //     A compact union-sum -- what this lane's emitter computes -- is therefore NOT numerically
    //     equal to the fork's path, by the fork authors' own design.
    //   * it writes no K/V: `build_attn` never did. (The previous refusal claimed it did; the read
    //     shows `build_attn_mha(q, k, v, ...)` consuming cache views and the cache written earlier,
    //     in process_ubatch.) That item is withdrawn rather than reworded.
    //
    // WHAT IS GENUINELY MISSING, in the order it must be built:
    //
    //   1. A READER INDEX SOURCE. The fork resolves each (sequence, position) to spans of cells
    //      through `mctx_cur->attention_view(seq, pos)` -- a method of its own KV cache, absent
    //      from upstream. Reaching it needs a per-sequence view ledger of our own, populated by
    //      the caller (the RERoT whiteboard/station), plus the layer's cells for each span.
    //   2. THE FULL-LENGTH SCATTER. Without the -inf fill and the single global softmax, the
    //      numbers differ from the fork's; with it, the arena cost is the full cache per layer.
    //   3. LoRA OUTPUT SCALING (`wo_s`) and sinks, which the fork's reader path threads through.
    //   4. ARENA ADMISSION: `ggml_new_object` aborts rather than returning null (ggml.c:1832), so
    //      an emission that does not fit ends the process mid-decode. `arena_budget` is advisory
    //      and `estimate_drift` is post-hoc; neither admits.
    //
    // None of that is a warning; each is a thing to build. And the alternative -- print, then call
    // the original -- was measured: every layer declined, the run produced stock numbers with exit
    // 0, and nothing but a warning count said the feature had not run.
    //
    // This check therefore only fires when the operator installed the hook AND asked for the
    // replacement, so it cannot affect a program that never opted in.
    if (emit_now_requested()) {
        std::fprintf(stderr,
                     "[herd] reader: layer %d was asked to REPLACE attention "
                     "(HERD_READER_EMIT_NOW) and this build cannot. Missing, in build order:\n"
                     "  (1) a reader index source -- the fork resolves (sequence, position) to "
                     "span cells via its own KV-cache method, which upstream does not have;\n"
                     "  (2) the full-cache-length scatter with a -INFINITY fill and one global "
                     "softmax, which is what makes the fork's numbers equal to the native "
                     "reduction order (a compact union sum does not);\n"
                     "  (3) LoRA output scaling and attention sinks;\n"
                     "  (4) an arena admission test -- ggml aborts on an overfull arena.\n"
                     "Refusing instead of returning stock attention under a reader label.\n",
                     il);
        std::fflush(stderr);
        std::_Exit(2);
    }

    // The original, with the original arguments. This is the whole behaviour.
    return g_orig(self, inp, wo, wo_b, wo_s, q_cur, k_cur, v_cur, kq_b, sinks, v_mla,
                  kq_scale, il);
}

// Mangled name of llm_graph_context::build_attn(const llm_graph_input_attn_kv *,
// ggml_tensor* x9, float, int), read from the built archive with nm rather than
// written from the signature: eight overloads share the name, and a guess that
// happens to assemble is still a guess. A pin bump that renames it fails here.
constexpr const char * kSymbol =
    "_ZNK17llm_graph_context10build_attnEP23llm_graph_input_attn_kvP11ggml_tensorS3_S3_S3_S3_"
    "S3_S3_S3_S3_fi";

// llm_graph_context::build_attn_mha(ggml_tensor* x8, int64_t, float, int), from the same
// archive. Eight tensor arguments then `lfi` -- the int64_t n_kv_max is `l`, not `x`.
constexpr const char * kSymbolMha =
    "_ZNK17llm_graph_context14build_attn_mhaEP11ggml_tensorS1_S1_S1_S1_S1_S1_S1_lfi";

// llm_kv_cache_context::set_input_attn_mask(const llm_attn_mask&, const llama_ubatch*, bool), from
// the same archive. This is the context's override, which is what
// `llm_graph_input_attn_kv::set_input` actually calls through `mctx`; hooking the base
// (`_ZNK14llama_kv_cache19...`) would only fire if the override forwarded, and that is a fact
// nobody had established.
constexpr const char * kSymbolMask =
    "_ZNK22llama_kv_cache_context19set_input_attn_maskERK13llm_attn_maskPK12llama_ubatchb";

// ---------------------------------------------------------------------------
// THE PHASE SEAM: telling a RESERVE graph from a RUNNING graph.
//
// Nothing in the tensors distinguishes them. `graph_reserve` (llama-context.cpp:3407) also calls
// `model.build_graph`, and its ubatch is built by `ubatch_reserve(n_tokens/n_seqs, n_seqs)` -- a
// WORST-CASE batch, so `q`, `k` and `mask` can have exactly the shapes a real prefill has. Measured
// in this tree: one run produced `n_cell` values of both 2048 (= n_ctx) and 256 (= the model's real
// n_kv), and `mask->ne[1]` values of both 9328 and 44. Guessing which was which is how §7.24's
// "prefill measurement" turned out to be a reserve graph.
//
// What IS decisive is the CALL, not the arguments: a graph built inside `llama_context::graph_reserve`
// is a reserve graph, and the symbol is not inlined (verified: `_ZN13llama_context13graph_reserve...`
// is a global `T` in the object file). So this seam marks the phase with a thread-local DEPTH, and
// any consumer asks `herd_phase_is_reserve()` instead of inferring from shapes.
//
// Why a depth and not a bool: `graph_reserve` can recurse through the pipeline-parallel retry at
// :1026, and the prefill-window path allocates its own reserve ubatch. A counter makes nesting
// correct by construction, and it is restored by RAII so an early return cannot leave it stuck --
// a stuck flag would silently mislabel every later graph, which is worse than not having the seam.
//
// SCOPE, stated because it bounds the claim: this marks the CURRENT THREAD. llama builds a graph on
// the thread that decodes, and reserve happens on that same thread in the paths this tree exercises;
// a build on another thread would read depth 0 and be called "running". That is the correct default
// (a build that is not inside a reserve call is not a reserve graph), and it needs no global table --
// only one integer per thread, owned by the scope that sets it.
constexpr const char * kSymbolReserve =
    "_ZN13llama_context13graph_reserveEjjjPK22llama_memory_context_ibPmb";

thread_local int g_reserve_depth = 0;

bool herd_phase_is_reserve() { return g_reserve_depth > 0; }

// `ggml_cgraph * llama_context::graph_reserve(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs,
// const llama_memory_context_i *, bool split_only, size_t * sizes, bool reserve_embeddings)`
using graph_reserve_fn = void * (*)(void *, unsigned, unsigned, unsigned, const void *, bool,
                                    std::size_t *, bool);

graph_reserve_fn g_orig_reserve = nullptr;

// Whether anyone is watching. `g_orig_reserve` is written by `install_all()`, which runs from main, so
// before that it is null for every seam; during decode it is non-null exactly when the phase hook is
// installed. Consumers above use this to avoid reporting `running` for a graph nobody is watching --
// an un-driven signal printing a confident verdict is the defect this seam was corrected for once.
//
// DEFINED AFTER `g_orig_reserve`, not before it. The first placement put this function above the
// declaration and the build said so in one line; an ordering that reads correctly and compiles
// incorrectly is worth a sentence here, because the two neighbours look interchangeable.
bool herd_phase_known() { return g_orig_reserve != nullptr; }

bool phase_probe_enabled() { return std::getenv("HERD_READER_PHASE") != nullptr; }

// ---------------------------------------------------------------------------
// THE K1 SEAM: identity takeover at `llama_model::build_graph`.
//
// Why here and not at `build_attn`. `build_attn` returns a TENSOR, so a wrapper around it cannot
// decide for the whole graph: layer 0 is returned before layer 1 is even built, and a decision taken
// at layer 0 cannot know whether the remaining layers qualify. "All layers replaced, or none" is
// not expressible there -- it is expressible only where the WHOLE graph comes back, which is
// `ggml_cgraph * llama_model::build_graph(const llm_graph_params&) const` (llama-model.h:1082).
// modules/reader's README calls this step K1 and puts it before any attention is replaced; the
// reason is stated there: changing the math before the identity path is proven means the baseline
// was never established.
//
// The body is TRANSPARENT BY CONSTRUCTION: it counts, it prints, and it returns `g_orig_graph`'s
// pointer unchanged. No node is added, removed or inspected, so the graph the caller receives is
// the same object the original built.
//
// THE COUNT IS ALSO THE ASSERTION ABOUT REACHABILITY. `HERD_READER_ATTN` hooks exactly one of
// upstream's eight `build_attn` overloads -- the KV one (see the file header). A model that routes
// through any of the other seven would leave the reader lane unreachable while the run looked
// normal, and nothing in the tensors would say so. With both switches on, this seam reports the
// per-graph delta of `build_attn` calls: it must be the layer count on every graph that builds
// attention, and a delta of zero is the silent-routing case, printed as such instead of inferred.
constexpr const char * kSymbolGraph = "_ZNK11llama_model11build_graphERK16llm_graph_params";

// `ggml_cgraph * llama_model::build_graph(const llm_graph_params&) const`, spelled with opaque
// handles: the graph pointer is carried through this seam and never dereferenced in it.
using build_graph_fn = void * (*)(const void *, const void *);

build_graph_fn g_orig_graph = nullptr;
std::atomic<long> g_graph_calls{0};

bool graph_k1_enabled() { return std::getenv("HERD_READER_GRAPH_K1") != nullptr; }

void * herd_build_graph(const void * self, const void * params) {
    const long before = g_calls.load(std::memory_order_relaxed);
    void * const graph = g_orig_graph(self, params);
    const long after = g_calls.load(std::memory_order_relaxed);
    const long n = g_graph_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    std::fprintf(stderr, "[herd] reader.k1: build_graph #%ld phase=%s attn_calls=%ld%s\n", n,
                 !herd_phase_known()
                     ? "unknown(no phase seam)"
                     : (herd_phase_is_reserve() ? "inside_reserve_scope" : "outside_reserve_scope"),
                 after - before,
                 (after == before && g_orig != nullptr)
                     ? " [zero: the KV build_attn seam is not on this graph's path]"
                     : "");
    std::fflush(stderr);
    return graph;
}

struct reserve_depth_guard {
    reserve_depth_guard() { ++g_reserve_depth; }
    ~reserve_depth_guard() { --g_reserve_depth; }
    reserve_depth_guard(const reserve_depth_guard &) = delete;
    reserve_depth_guard & operator=(const reserve_depth_guard &) = delete;
};

void * herd_graph_reserve(void * self, unsigned n_tokens, unsigned n_seqs, unsigned n_outputs,
                          const void * mctx, bool split_only, std::size_t * sizes,
                          bool reserve_embeddings) {
    reserve_depth_guard guard;   // RAII: an early return from the original cannot leave it raised

    // The counts on both sides of the call are the evidence, and they are the reason this prints
    // instead of only marking: `mha_after > mha_before` PROVES the reserve graph reached
    // `build_attn_mha`, which is the fact that had to be inferred before. `depth` is printed so a
    // nested reserve is visible rather than collapsing into one line.
    const long before = g_mha_calls.load();
    void * gf = g_orig_reserve(self, n_tokens, n_seqs, n_outputs, mctx, split_only, sizes,
                               reserve_embeddings);
    const long after = g_mha_calls.load();
    if (phase_probe_enabled()) {
        // THREE OUTCOMES, NOT TWO, because the count only means something when its source is
        // installed.
        //
        // `g_mha_calls` is incremented inside `herd_build_attn_mha`, which is only reached when the
        // MHA seam is installed. With `HERD_READER_PHASE=1` alone the counter is never driven, so
        // `after == before == 0` for every call -- and the first version printed
        // "no attention built inside this reserve" for all eleven reserves of a run in which every
        // one of them DID build attention (the same run with `HERD_READER_MHA=1` prints
        // `calls N -> N+3`). A diagnostic that reports a confident negative from an un-driven
        // counter is worse than one that stays silent: it is a measurement-shaped lie.
        //
        // `g_orig_mha != nullptr` is the right test for "is the counter driven", and at THIS point in
        // time it is valid: `install_all()` has run (this is a decode-time call), so a captured
        // original means the trampoline exists. (The constructor must not test it -- see the note at
        // the seam installation.)
        const bool counted = g_orig_mha != nullptr;
        const char * verdict =
            !counted ? "attention count UNAVAILABLE (HERD_READER_MHA not installed); this reserve's "
                       "attention build cannot be observed"
                     : (after > before ? "the reserve graph DID build attention"
                                       : "no attention built inside this reserve");
        std::fprintf(stderr,
                     "[herd] phase: reserve %s (depth %d) n_tokens %u n_seqs %u n_outputs %u "
                     "split_only %d | build_attn_mha calls %ld -> %ld (%s)\n",
                     gf ? "returned a graph" : "FAILED", g_reserve_depth, n_tokens, n_seqs,
                     n_outputs, split_only ? 1 : 0, before, after, verdict);
        std::fflush(stderr);
    }
    return gf;
}

// ---------------------------------------------------------------------------
// THE ROWS SEAM: the one moment a row list has values.
//
// `build_attn_mha` receives `kv_rows` as a `ggml_set_input` tensor, so its `data` is null at graph
// BUILD time (measured: `input 1 data no`, every call). An earlier note over-read that into "the
// values can never be read", which is wrong, and wrong in the direction that matters: it declared
// a capability impossible. They are written a few microseconds later by this very function, which
// first asserts `ggml_backend_buffer_is_host(attn_mask.rows->buffer)` and then fills the buffer
// from `v_cells[0].get_seq_cells(...)`. Host-resident, filled, readable -- just not where the graph
// was being built.
//
// So the observation point is AFTER the original returns: the first instant the fill has happened.
// This hook only READS. It writes no row, replaces no tensor, and does not touch llama.cpp's code.
// An observer that changed what it observed would not be an observer, and a probe that could
// perturb the mask would turn every later numeric comparison into a comparison against itself.
//
// What it is for: `kv_rows` is the one upstream artefact whose CONTENT has never been seen in this
// project. ggml.h promises "negative rows are padding, their mask entries must be -inf", and
// `get_seq_cells` promises "in CELL order ... padded with -1". Both are promises in a comment. This
// prints the actual values, so the next decision rests on a measurement instead of a contract.
// ---------------------------------------------------------------------------

// There is no mirrored struct here, on purpose.
//
// The first version of this probe declared its own `llm_attn_mask_view { raw, rows, n_kv }` and
// took a pointer to it, on the theory that a matching prefix is enough. It is not: the object that
// arrives is an `llm_attn_mask` living in this program, and reading it through an unrelated type is
// undefined behaviour regardless of how many bytes agree -- an optimiser is entitled to assume the
// two are distinct, and the failure it produces is a wrong number, not a crash. `llama-graph.h`
// declares the type and other patch TUs in this tree already include upstream sources, so the fix
// is the type itself. Its layout is then checked by the compiler rather than by a comment.
//
// The signature also matches, rather than approximating: the mangled name this is installed under
// says `llm_attn_mask`, so taking anything else would be a lie told to the linker.

using set_input_attn_mask_fn =
    void (*)(const void * self, const llm_attn_mask * mask, const void * ubatch, bool causal_attn);

set_input_attn_mask_fn g_orig_mask = nullptr;
std::atomic<long> g_mask_calls{0};

bool rows_probe_enabled() { return std::getenv("HERD_READER_ROWS") != nullptr; }

// ---------------------------------------------------------------------------------------------
// THE VALIDATION SEAM, ON ITS OWN SWITCH.
//
// `HERD_READER_ROWS` only OBSERVES, and it is throttled to the first few calls -- deliberate, so a
// normal run pays nothing. Verification is the opposite trade: it must run on EVERY fill, because a
// row table that was fine on the first decode says nothing about the next one. So it lives behind
// `HERD_READER_VALIDATE_ROWS`, and when that switch is set every call is checked.
//
// WHY THE CHECK BELONGS HERE AND NOWHERE EARLIER. `kv_rows` is a `ggml_set_input` tensor: at graph
// build time `data` is null (measured: `input 1 data no`), so a check there reads nothing. This
// function runs after the original has filled both tensors, which is the first instant the values
// exist -- and the last instant before they are used.
//
// WHAT IT VERIFIES, AND WHAT HAPPENS WHEN IT FAILS. `tq::reader::validate_runtime_rows` checks that
// every non-negative row names a cell inside the cache and that every negative row is masked `-inf`
// for EVERY token of EVERY slice. A violation is not printed and skipped: the run STOPS. That is the
// whole difference between observing and verifying -- the kernel resolves a negative row as
// `max(row, 0)` and walks `k->data + row*nb[1]` with no upper bound, so a table it should not read is
// a wrong answer that looks right, and a log line nobody greps.
//
// UNSUPPORTED IS NAMED, NOT SKIPPED. A device-resident buffer, a non-I32 row tensor, an absent mask
// tensor, or a build without the validator each produce their own message and their own stop. The
// switch was asked for; "nothing happened" is not a possible outcome.
// ---------------------------------------------------------------------------------------------
// The `__has_include` block used to be repeated here, and it is gone. It sat inside the anonymous
// namespace -- the same place the original include went and the reason the first link failed -- so
// its `#include` was a no-op (the header was already included at the top of this file) and its
// `#define` was a redefinition of a macro that already had the same value. Harmless, and therefore
// worse than useless: a reader could not tell whether this second block was load-bearing, and the
// next person to add a conditional include would have copied the pattern into a namespace again.

bool rows_validate_enabled() { return std::getenv("HERD_READER_VALIDATE_ROWS") != nullptr; }

[[noreturn]] void rows_validate_stop(const char * what, const char * detail) {
    std::fprintf(stderr,
                 "[herd] reader.validate: %s -- %s. HERD_READER_VALIDATE_ROWS asks for the runtime row "
                 "table to be verified before use, and it cannot be certified: the kernel would read "
                 "k/v rows this check could not clear, and a wrong attention result cannot be told "
                 "from a right one afterwards. Stopping instead of continuing unverified.\n",
                 what, detail);
    std::fflush(stderr);
    std::abort();
}

void herd_validate_runtime_rows(const llm_attn_mask * mask) {
    if (!rows_validate_enabled()) return;

    // THE MISSING VALIDATOR IS REPORTED FIRST, BEFORE ANY EARLY RETURN.
    //
    // The previous revision had this refusal in an `#else` at the END of the function, after two early
    // returns: `mask == nullptr` and `mask->rows == nullptr` (the implicit-cells regime, which prints
    // "nothing to verify" and returns). In a build without `tq_reader_rows`, a run that hit either
    // point therefore exited 0 having printed a sentence about a legitimate regime -- and the actual
    // reason nothing was verified, that the build cannot verify anything, was never said. Moving the
    // refusal to the top is what makes "asked for, cannot be honoured" an outcome that no later branch
    // can pre-empt.
#if !defined(HERD_ROWS_VALIDATOR)
    rows_validate_stop("this build has no row-table validator",
                       "the tq_reader_rows target was not linked (HERD_BUILD_MODULES=OFF?), so the "
                       "switch cannot be honoured");
#endif

    // A NULL MASK IS NOT A REGIME, IT IS A DROPPED REQUEST.
    //
    // The switch was set and there is no mask to examine. Upstream's `set_input_attn_mask` always
    // receives a reference, so this is a caller or layout error rather than a normal state -- and the
    // old `|| mask == nullptr` folded it into the same silent `return` as "switch off", which is the
    // one reading a reader must never make: no output at all is indistinguishable from a clean run.
    if (mask == nullptr) {
        rows_validate_stop("no mask was supplied",
                           "HERD_READER_VALIDATE_ROWS asks for the row table and its padding to be "
                           "verified, and this call carries neither; nothing was examined, so the run "
                           "must not look verified");
    }

    static std::atomic<long> calls{0};
    const long call = calls.fetch_add(1) + 1;

    // `raw` IS CHECKED BEFORE ANYTHING READS IT. The previous revision dereferenced
    // `mask->raw->ne[...]` inside the `rows == nullptr` message -- a branch that exists precisely to
    // report an unusual state, taken before any null check on `raw`. A layout whose `rows` is null
    // and whose `raw` is null would have faulted inside a diagnostic, which is the worst place for a
    // crash: the message that was supposed to explain the situation is the thing that dies.
    if (mask->raw == nullptr) {
        rows_validate_stop("no mask tensor at all",
                           "`llm_attn_mask::raw` is null, so neither the row table nor the padding "
                           "entries can be located; there is nothing to observe or verify");
    }

    if (mask->rows == nullptr) {
        // The other regime: cells are implicit (0..n_kv-1), so there is no table to verify. This is
        // a supported case, not a failure, and it is said once so "no validation output" cannot be
        // read as "validation passed".
        if (call == 1) {
            std::fprintf(stderr,
                         "[herd] reader.validate: rows NULL -- cells are implicit, nothing to verify "
                         "(mask ne [%lld %lld %lld %lld])\n",
                         (long long) mask->raw->ne[0], (long long) mask->raw->ne[1],
                         (long long) mask->raw->ne[2], (long long) mask->raw->ne[3]);
            std::fflush(stderr);
        }
        return;
    }

    const ggml_tensor * r = mask->rows;
    const ggml_tensor * m = mask->raw;   // non-null: checked above, before anything read it

#if defined(HERD_ROWS_VALIDATOR)

    // ONE IMPLEMENTATION OF THE DESCRIPTION RULES, AND IT IS THE TESTED ONE.
    //
    // Every check from here to the validator used to live inline in this function, and each refusal
    // called `rows_validate_stop`, which ABORTS. A function that aborts cannot be driven case by
    // case: a process death reports an exit code, not which check fired -- so "residency is reported
    // before rank" and "this case is decided before any byte is read" were unobservable claims, and
    // the only path with any coverage was the accepting one. `validate_seam_contract` is the same
    // rules over numbers, ggml-free, with a report instead of an abort; `tq_reader.seam_contract`
    // drives every refusal in it by name.
    //
    // A duplicated copy would be worse than either version alone: the two would drift, and the
    // untested one would be the one the product ran. So the inline chain is GONE, not kept beside it.
    //
    // The equivalents, one for one, so nothing was relaxed in the move:
    //   I32 + nb[0]        -> rows_not_i32
    //   rows buffer/host   -> rows_not_host        (also covers "never placed")
    //   rows data null     -> rows_no_data
    //   mask type          -> mask_type
    //   mask host/data     -> mask_not_host
    //   mask ne[0] != rows -> columns_differ
    //   ne <= 0 (int64)    -> dim_not_positive     (checked BEFORE the cross-tensor comparison)
    //   rank / slices      -> rank_unmodelled, slices_differ
    //   mask nb[0]         -> mask_not_contiguous
    //   buffer geometry    -> rows_buffer_geometry / mask_buffer_geometry
    //   data outside       -> rows_data_outside / mask_data_outside
    // Everything the old chain checked is checked; the order differs in one place and is better for
    // it -- "one of these descriptions is not a shape at all" now precedes "the two shapes disagree",
    // which the seam test caught as a real misdirection.
    auto face_of = [](const ggml_tensor * t) {
        ::tq::reader::tensor_face f;
        for (int i = 0; i < 4; ++i) {
            f.ne[i] = t->ne[i];
            f.nb[i] = t->nb[i];
        }
        f.data = t->data;
        if (t->buffer != nullptr) {
            f.buffer_present = true;
            f.buffer_is_host = ggml_backend_buffer_is_host(t->buffer) != 0;
            // BASE AND SIZE ONLY FOR A HOST BUFFER. `ggml_backend_buffer_get_base` on a device buffer
            // hands back a device address; nothing here dereferences it (the contract refuses a
            // non-host buffer before it compares anything), but reading a device address into a field
            // that exists to be compared against host pointers invites a later check to do exactly
            // that. Skipping it keeps "this face carries host geometry" true by construction.
            if (f.buffer_is_host) {
                f.buffer_base = ggml_backend_buffer_get_base(t->buffer);
                f.buffer_size = ggml_backend_buffer_get_size(t->buffer);
            }
        }
        return f;
    };

    ::tq::reader::tensor_face rows_face = face_of(r);
    rows_face.type_is_i32 = (r->type == GGML_TYPE_I32);
    ::tq::reader::tensor_face mask_face = face_of(m);
    mask_face.type_is_f16 = (m->type == GGML_TYPE_F16);
    mask_face.type_is_float = (m->type == GGML_TYPE_F16 || m->type == GGML_TYPE_F32);

    const ::tq::reader::seam_contract_report contract =
        ::tq::reader::validate_seam_contract(rows_face, mask_face);
    if (contract.refusal != ::tq::reader::seam_contract_refusal::ok) {
        // The field name goes into the message, so a reader is not sent hunting: the helper already
        // says WHICH input is wrong, and that is information the old prose lost.
        std::fprintf(stderr, "[herd] reader.validate: call %ld contract refused at %s\n", call,
                     contract.field);
        std::fflush(stderr);
        rows_validate_stop(::tq::reader::describe(contract.refusal), contract.why.c_str());
    }
    const std::size_t n_columns = contract.n_columns;
    const std::size_t n_slices = contract.n_slices;
    const std::size_t n_tokens = contract.n_tokens;
    const std::size_t rows_readable = contract.rows_readable_bytes;
    const std::size_t mask_readable = contract.mask_readable_bytes;

    // `n_kv` IS UPSTREAM'S OWN CACHE DEPTH AND BELONGS TO THIS FUNCTION, NOT TO THE TENSOR FACES.
    // `attn_mask.n_kv = mctx->get_n_kv()` is the number a non-negative row is bounded by; the faces
    // carry shapes, and a shape cannot say how many k/v rows exist. Checked after the contract so a
    // broken description is reported as a description, not as a depth problem.
    if (mask->n_kv == 0) {
        rows_validate_stop("cache depth is zero", "no non-negative row could name a cell");
    }

    ::tq::reader::runtime_rows_inputs in;
    in.rows = static_cast<const std::int32_t *>(r->data);
    in.n_columns = n_columns;
    in.n_slices = n_slices;
    in.rows_slice_stride = r->nb[1];
    // The REAL extent, from ggml. A buffer may hold several tensors, so this is an upper bound on
    // what is readable from here rather than the tensor's own size -- which is the right kind of
    // bound for "reading this claim cannot fault", and it is not something this code computed.
    in.rows_bytes = rows_readable;
    in.mask = m->data;
    in.n_tokens_per_slice = n_tokens;
    in.mask_token_stride = m->nb[1];
    in.mask_slice_stride = m->nb[3];
    in.mask_is_f16 = m->type == GGML_TYPE_F16;
    in.mask_bytes = mask_readable;
    // The cache depth as UPSTREAM defines it (`attn_mask.n_kv = mctx->get_n_kv()`), not something
    // inferred from a shape: the non-negative rows index exactly this many k/v rows.
    in.n_cell = mask->n_kv;

    const ::tq::reader::rows_validation_report rep = ::tq::reader::validate_runtime_rows(in);
    if (rep.refusal != ::tq::reader::rows_validation_refusal::ok) {
        std::fprintf(stderr,
                     "[herd] reader.validate: call %ld REFUSED (%s) -- %s\n",
                     call, ::tq::reader::describe(rep.refusal), rep.why.c_str());
        if (rep.has_violation) {
            std::fprintf(stderr,
                         "[herd] reader.validate: at slice %zu column %zu row %d token %zu mask %g\n",
                         rep.violation_slice, rep.violation_column, int(rep.violation_row),
                         rep.violation_token, double(rep.violation_mask));
        }
        std::fflush(stderr);
        rows_validate_stop("the runtime row table is not consumable",
                           "see the refusal above for the column and the token that can see it");
    }
    if (call == 1) {
        std::fprintf(stderr,
                     "[herd] reader.validate: call 1 verified -- %zu column(s) x %zu slice(s) x %zu "
                     "token(s), %zu padding column(s), max row %zu, cache depth %u, mask %s\n",
                     rep.columns_checked, rep.slices_checked, rep.tokens_checked, rep.negative_rows,
                     rep.max_row, (unsigned) mask->n_kv, in.mask_is_f16 ? "F16" : "F32");
        std::fflush(stderr);
    }
#endif
}

void herd_set_input_attn_mask(const void * self, const llm_attn_mask * mask,
                              const void * ubatch, bool causal_attn) {
    g_orig_mask(self, mask, ubatch, causal_attn);

    // VALIDATION RUNS BEFORE THE OBSERVATION, AND INDEPENDENTLY OF ITS THROTTLE.
    //
    // The observation below stops after a few calls on purpose: it is a print, and a long run would
    // flood stderr. Verification cannot share that bargain -- a table that was clean on the first
    // fill says nothing about the next one -- so it is called every time and gated on its own
    // variable. It goes first because a run that is going to stop should stop before producing the
    // log this file's other probes exist to explain.
    herd_validate_runtime_rows(mask);

    if (!rows_probe_enabled() || mask == nullptr) return;

    const long call = g_mask_calls.fetch_add(1) + 1;
    if (call > 4) return;  // the first few calls already say everything a decision needs
    if (mask->raw == nullptr) {
        std::fprintf(stderr, "[herd] reader.rows: call %ld no mask at all\n", call);
        return;
    }
    if (mask->rows == nullptr) {
        // The other regime: cells are implicit (0..n_kv-1) and the mask alone carries visibility.
        // Printed rather than skipped, so "no rows in this run" is a fact in the log, not an absence.
        std::fprintf(stderr,
                     "[herd] reader.rows: call %ld rows NULL, mask ne [%lld %lld %lld %lld] type %d "
                     "-> cells are implicit\n",
                     call, (long long) mask->raw->ne[0], (long long) mask->raw->ne[1],
                     (long long) mask->raw->ne[2], (long long) mask->raw->ne[3],
                     (int) mask->raw->type);
        return;
    }

    const ggml_tensor * r = mask->rows;
    const std::int64_t n = r->ne[0];

    // Everything below is a precondition of touching `r->data`, and each one is checked because
    // the alternative is reading memory that is not the row list. A probe that segfaults or prints
    // plausible nonsense from the wrong bytes is worse than no probe: the numbers would be believed.
    //
    //   * I32 -- the field is typed, and a different width would reinterpret two rows as one.
    //   * nb[0] == sizeof(int32_t) -- a non-contiguous view (a slice of a wider tensor) would make
    //     `v[i]` read padding between elements while still looking like a valid row list.
    //   * host buffer -- `set_input_attn_mask` asserts this itself, but only on the path it takes;
    //     a device-resident buffer would require the backend API, and dereferencing its pointer from
    //     the CPU is a device address, which is exactly the mistake HANDOFF §4.2 warns about.
    //   * ne[1] <= 1 -- one slice is what this print describes; more than one is reported and
    //     skipped rather than read as a flat list.
    if (r->type != GGML_TYPE_I32 || r->nb[0] != (std::size_t) sizeof(std::int32_t)) {
        std::fprintf(stderr,
                     "[herd] reader.rows: call %ld rows type %d nb0 %zu -- not a contiguous I32 row "
                     "list, not read\n",
                     call, (int) r->type, (std::size_t) r->nb[0]);
        return;
    }
    if (r->buffer == nullptr || !ggml_backend_buffer_is_host(r->buffer)) {
        std::fprintf(stderr,
                     "[herd] reader.rows: call %ld rows type I32 but the buffer is %s -- not read "
                     "from the CPU\n",
                     call, r->buffer ? "not host-resident" : "absent");
        return;
    }
    if (r->ne[1] > 1) {
        std::fprintf(stderr,
                     "[herd] reader.rows: call %ld rows ne [%lld %lld] -- %lld slices, only slice 0 "
                     "would be meaningful; not flattened\n",
                     call, (long long) r->ne[0], (long long) r->ne[1], (long long) r->ne[1]);
        return;
    }

    const std::int32_t * v = static_cast<const std::int32_t *>(r->data);
    if (v == nullptr) {
        // Say exactly this and no more. It is NOT evidence that the values are unreachable; it
        // means this call did not have them, and the previous claim that they never can be read is
        // what this print exists to test.
        std::fprintf(stderr,
                     "[herd] reader.rows: call %ld rows ne [%lld %lld] but data is NULL here\n",
                     call, (long long) r->ne[0], (long long) r->ne[1]);
        return;
    }

    long negatives = 0;
    long ascending = 1;
    long out_of_range = 0;
    long identity = 0;   // v[i] == i, counted over EVERY row, not over the printed prefix
    std::int32_t prev = -1;
    for (std::int64_t i = 0; i < n; ++i) {
        if (v[i] < 0) {
            ++negatives;
            continue;
        }
        if (v[i] == std::int32_t(i)) ++identity;
        if (prev >= 0 && v[i] <= prev) ascending = 0;
        if (std::uint32_t(v[i]) >= mask->n_kv) ++out_of_range;
        prev = v[i];
    }

    // `identity` is reported as a count over all rows rather than inferred from the twelve shown.
    // The earlier version printed twelve ascending values and the notes then read that as "cell ->
    // row is the identity map"; a list can be strictly ascending over its whole length and still
    // skip a cell (0,1,2,...,40,41), so the prefix never supported that claim. This count does, and
    // it is data rather than an argument.
    const std::int64_t effective = n - negatives;
    std::fprintf(stderr,
                 "[herd] reader.rows: call %ld rows ne [%lld %lld] n_kv %u | -1 columns %ld | "
                 "strictly ascending %s | >= n_kv %ld | v[i]==i %ld of %lld | first:",
                 call, (long long) r->ne[0], (long long) r->ne[1], mask->n_kv, negatives,
                 ascending ? "yes" : "NO", out_of_range, identity, (long long) effective);
    for (std::int64_t i = 0; i < n && i < 12; ++i) {
        std::fprintf(stderr, " %d", v[i]);
    }
    std::fprintf(stderr, "\n");
    std::fflush(stderr);
}

// Exit report. Silent unless tracing: a product's stderr is part of its output
// contract, and this is a diagnostic.
struct seam_report {
    ~seam_report() {
        if (trace_enabled()) {
            std::fprintf(stderr, "[herd] reader: seam observed %ld build_attn call(s)\n",
                         g_calls.load());
        }
    }
} g_seam_report;

} // namespace

// ---------------------------------------------------------------------------
// Wiring. Registered, not installed: the shared installer runs at priority 102,
// after every registrar, and add() refuses a registration made afterwards.
//
// Opt-in through HERD_READER_ATTN. An unrequested hook is a cost with no benefit,
// and in a hot attention path it is a cost paid on every layer.
// ---------------------------------------------------------------------------
#if !defined(HERD_NO_AUTOLOAD)
__attribute__((constructor(101))) static void herd_reader_attn_autoload() {
    if (std::getenv("HERD_NO_AUTOLOAD") != nullptr) {
        return;
    }
    // Refused here, at static-init time, before a model is loaded and before main runs. There is no
    // later point that can be trusted to notice: the seam sees only runs that build attention, and
    // an exit handler also captures exits that are legitimately failing. This call cannot return.
    refuse_if_replace_requested();
    // A RETIRED SWITCH MUST NOT BE A NO-OP. `HERD_READER_PUBLISH` used to register a graph input that
    // wrote a row table; it was removed because the tensors it wrote had no node to consume them, so
    // they were never placed and the publish aborted on layer 0. If a script still sets it, silence
    // would read as "the publisher ran and found nothing to say", which is the failure mode this
    // file refuses everywhere else. One line, at static-init time, before a model loads.
    if (std::getenv("HERD_READER_PUBLISH") != nullptr) {
        std::fprintf(stderr,
                     "[herd] reader.publish: HERD_READER_PUBLISH is retired -- this build has no "
                     "row-table publisher. Registering a graph input is not enough to publish into "
                     "it: the scheduler places THE GRAPH, so a tensor no node consumes is never "
                     "placed and `set_input` sees no storage. The row tensors have to live inside "
                     "the attention graph. See the note where the block was in "
                     "patches/herd_reader_attn.cpp.\n");
        std::fflush(stderr);
        std::abort();
    }

    // THE REQUEST IS HONOURED OR REFUSED; IT IS NEVER QUIETLY DROPPED.
    //
    // Two variables, two different requests, and the pair has to be unambiguous:
    //
    //   HERD_READER_ATTN      "install the reader hook". With it unset there is no lane in this
    //                         process at all, so nothing else can be true or false about it.
    //   HERD_READER_EMIT_NOW  "and use it to REPLACE the layer's attention". This build cannot:
    //                         the reasons are in the block above, and the hook refuses fatally at
    //                         the first layer it is asked to replace.
    //
    // The previous version printed the "EMIT_NOW without ATTN" line only under HERD_SEAM_TRACE,
    // which made an explicit request vanish silently in the ordinary case -- the one thing a
    // request must never do. It is now unconditional, because the operator asked and deserves an
    // answer whether or not they also asked for tracing.
    //
    // Why not exit non-zero here as well? Because this is a constructor in a library: it runs
    // before `main` in every program that links it, including ones that never touch attention.
    // `--help`, a converter, a batch tool -- none of them can replace a layer's attention, so
    // none of them is refusing anything. Killing them would punish the wrong program for a request
    // they cannot act on, which is how an environment variable becomes everyone's problem. The
    // line says what happened; the process is otherwise unaffected; a program that DOES run the
    // lane dies at the first layer instead (see the hook).
    // One variable, one seam. An earlier version returned as soon as HERD_READER_ATTN was
    // unset, which put the MHA probe below that return and made `HERD_READER_MHA=1` alone a
    // run with no lines and no explanation -- a request dropped silently, the exact failure
    // mode this file refuses everywhere else.
    if (std::getenv("HERD_READER_ATTN") == nullptr) {
        if (emit_now_requested()) {
            std::fprintf(stderr,
                         "[herd] reader: HERD_READER_EMIT_NOW asks for a layer's attention to be "
                         "replaced, but HERD_READER_ATTN is not set, so this program does not run "
                         "the reader lane and nothing was replaced. Set both to have the request "
                         "refused where it would take effect.\n");
            std::fflush(stderr);
        }
    } else {
        void * target = herd::resolve(kSymbol);
        if (!target) {
            // The lane was asked for, so running without it would silently drop the
            // configuration the operator chose.
            std::fprintf(stderr,
                         "[herd] reader: HERD_READER_ATTN is set but llm_graph_context::build_attn "
                         "(KV overload) was not found; the seam cannot be installed\n");
            std::abort();
        }
        static_cast<void>(herd::add("reader.build_attn", target,
                                    herd::raw(&herd_build_attn),
                                    reinterpret_cast<void **>(&g_orig), nullptr));
    }

    // The MHA seam, registered on its own request. It is NOT implied by HERD_READER_ATTN:
    // that variable asks for the outer seam, whose whole purpose in this build is to refuse a
    // replacement and say why. This one exists to MEASURE the inner seam (shapes, stream
    // count, whether a mask arrives), and an observer that ran because someone asked for a
    // different thing would be a cost nobody requested.
    // `|| replace_enabled()`: a run that asks for the replacement must have the seam
    // installed even without the probe, or the request would be silently dropped --
    // the same defect the ATTn/MHA split above exists to prevent. The seam is one;
    // what differs is whether the body replaces or observes.
    // `|| input_probe_enabled()`, and that is a defect fix rather than tidiness. The graph-input
    // block sits inside `herd_build_attn_mha`, so with only `HERD_READER_INPUT` set the seam was
    // never installed, the body never ran, and the request produced ZERO output and exit code 0 --
    // the drop that this variable's absence from the log was then read as evidence about. Measured
    // before the fix: `HERD_READER_INPUT=1` alone gave exit 0 and no `reader.input` line, which says
    // nothing about whether upstream would have called `set_input`. After the fix the same run
    // reaches the block and reports what it found.
    // `|| input_probe_enabled()`, and that is a defect fix rather than tidiness. The graph-input
    // block sits inside `herd_build_attn_mha`, so with only `HERD_READER_INPUT` set the seam was
    // never installed, the body never ran, and the request produced ZERO output and exit code 0 --
    // a dropped request, which this file refuses everywhere else. Two earlier revisions made exactly
    // that mistake and then read the empty log as a measurement; the variable now names itself in
    // the failure message for the same reason. (`HERD_READER_PUBLISH` used to have the same
    // arrangement; it was removed with its block -- see the note where the block was.)
    if (mha_probe_enabled() || input_probe_enabled() ||
        std::getenv("HERD_READER_GRAPH_K2") != nullptr ||
        // K2c asks for a replacement too, so it must arm the seam for the same reason K2 does:
        // a switch that names a replacement and does not install the seam produces zero output
        // and exit code 0, which reads as "the replacement refused" when it never ran.
        std::getenv("HERD_READER_GRAPH_K2C") != nullptr) {
        void * mha_target = herd::resolve(kSymbolMha);
        if (!mha_target) {
            // Named, not assumed: an earlier revision printed "HERD_READER_MHA is set" even when
            // only another variable had asked, which sends its reader to the wrong switch.
            std::fprintf(stderr,
                         "[herd] reader: %s is set but llm_graph_context::build_attn_mha was not "
                         "found; nothing can be installed\n",
                         mha_probe_enabled() ? "HERD_READER_MHA" : "HERD_READER_INPUT");
            std::abort();
        }
        // The return is CHECKED. `herd::add` is `[[nodiscard]] Seam *` and returns null when the
        // seam cannot be REGISTERED; discarding it (as this line used to) meant a registration
        // failure left no trace at all -- the probe would silently not run, and its absence from
        // the log would mean nothing.
        herd::Seam * const seam = herd::add("reader.build_attn_mha", mha_target,
                                            herd::raw(&herd_build_attn_mha),
                                            reinterpret_cast<void **>(&g_orig_mha), nullptr);
        if (seam == nullptr) {
            std::fprintf(stderr,
                         "[herd] reader: the build_attn_mha seam could not be registered; refusing "
                         "to run the probe\n");
            std::abort();
        }
        // `g_orig_mha` is NOT checked here, and an earlier version of this block wrongly did.
        // `add()` only registers a SeamDesc; the Dobby trampoline that writes `*origin` is created
        // by `herd::install_all()`, which runs later, from main. At this point in the constructor
        // the original is null BY CONSTRUCTION -- for every seam, not just this one -- so testing
        // it here aborted every run, including a plain `HERD_READER_MHA=1` probe that had been
        // working. Checking a value that cannot yet be set is not a safety check; it is a
        // guaranteed failure with a misleading message. Installation failure is caught where it
        // can be observed: `install_all()` aborts when a seam does not take, and the exit-time
        // zero-call check below catches the case where the seam went in but was never entered.
    }

    // The rows seam, on its own request. It answers a different question from the two above -- not
    // "what does build_attn_mha receive" but "what does the row table actually hold" -- so neither
    // variable implies it and it implies nothing else. Asked for and not installed is fatal, for
    // the same reason as the MHA probe: an observation that silently did not happen produces a log
    // whose emptiness means nothing.
    // `|| rows_validate_enabled()`, and that is a DEFECT FIX, not tidiness -- measured in this scope.
    //
    // The install gate was `rows_probe_enabled()` alone. With only `HERD_READER_VALIDATE_ROWS` set,
    // the seam was never installed, the hook never ran, and the run exited 0 having verified nothing
    // and printed nothing: the exact "request dropped in silence" shape this file refuses everywhere
    // else, found by running the switch on its own (`grep reader.validate` -> 0 lines, exit 0) and
    // confirmed by the same run with both switches set (1 line). Each variable now installs the seam
    // it needs, and neither implies the other's effects.
    if (rows_probe_enabled() || rows_validate_enabled()) {
        void * mask_target = herd::resolve(kSymbolMask);
        if (!mask_target) {
            std::fprintf(stderr,
                         "[herd] reader: HERD_READER_ROWS is set but "
                         "llama_kv_cache_context::set_input_attn_mask was not found; the rows seam "
                         "cannot be installed\n");
            std::abort();
        }
        static_cast<void>(herd::add("reader.set_input_attn_mask", mask_target,
                                    herd::raw(&herd_set_input_attn_mask),
                                    reinterpret_cast<void **>(&g_orig_mask), nullptr));
    }

    // The phase seam, on its own switch. It replaces an inference with an observation: "this graph
    // was built inside graph_reserve" is a fact about the call, and the only reliable source for it.
    if (phase_probe_enabled()) {
        void * reserve_target = herd::resolve(kSymbolReserve);
        if (!reserve_target) {
            std::fprintf(stderr,
                         "[herd] reader: HERD_READER_PHASE is set but "
                         "llama_context::graph_reserve was not found; the phase seam cannot be "
                         "installed\n");
            std::abort();
        }
        auto * seam = herd::add("reader.graph_reserve", reserve_target,
                                herd::raw(&herd_graph_reserve),
                                reinterpret_cast<void **>(&g_orig_reserve), nullptr);
        // ONLY THE REGISTRATION IS CHECKED HERE, AND THIS IS THE SAME CORRECTION THE MHA SEAM NEEDED.
        // `herd::add` stores a SeamDesc; the Dobby trampoline that writes `*origin` is created later
        // by `herd::install_all()`, which runs from main. So at this point in the constructor
        // `g_orig_reserve` is null BY CONSTRUCTION -- for every seam. An earlier draft of this block
        // tested it, and every run aborted with the message below, which reads like "the seam did not
        // take" when in fact nothing had been installed yet. Installation failure is caught where it
        // can be observed: `install_all()` aborts when a seam does not take.
        if (seam == nullptr) {
            std::fprintf(stderr,
                         "[herd] reader: the graph_reserve seam could not be registered; the phase "
                         "probe would report every graph as running\n");
            std::abort();
        }
    }

    // The K1 seam, on its own switch. It replaces an assumption with a measurement: that the whole
    // graph is built at one point, so a future takeover can decide for every layer at once. Asked
    // for and not installed is fatal, for the same reason as every seam above -- an identity path
    // that silently never ran proves nothing about the identity path.
    if (graph_k1_enabled()) {
        void * graph_target = herd::resolve(kSymbolGraph);
        if (!graph_target) {
            std::fprintf(stderr,
                         "[herd] reader: HERD_READER_GRAPH_K1 is set but "
                         "llama_model::build_graph was not found; the K1 seam cannot be installed\n");
            std::abort();
        }
        auto * k1 = herd::add("reader.build_graph", graph_target,
                              herd::raw(&herd_build_graph),
                              reinterpret_cast<void **>(&g_orig_graph), nullptr);
        if (k1 == nullptr) {
            std::fprintf(stderr,
                         "[herd] reader: the build_graph seam could not be registered; the K1 "
                         "identity path would not be measured\n");
            std::abort();
        }
    }
}
#endif
