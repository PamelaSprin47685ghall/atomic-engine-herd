// herd/prefab.cpp — the prefab lane's seam.
//
// One hooked function: llama_context::process_ubatch. Everything this file does to
// it is translation:
//
//   shape the ubatch -> ask the station -> (accept) copy it at envelope width
//                                       -> call the original with the copy
//                                       -> throw the copy away
//
// No graph is built here, no slot table is kept, no scheduler is touched. That is
// the whole design: upstream already reuses a compute graph whose topology matches
// the previous one and reallocates when the size changes (see n_reused /
// n_reallocs_diff in llama_perf_context_data). Making the size stable is therefore
// enough to get the reuse, and padding the ubatch is how the size is made stable
// without touching a single upstream file.
//
// Why padding is semantically neutral, as far as this lane is concerned:
//
//   * a pad row carries n_seq_id == 0, so it belongs to no sequence: the memory
//     context writes no KV cell for it and no other row's mask can see it;
//   * a pad row carries output == 0, so no caller asks for its logits;
//   * real rows keep their indices, so llama_get_logits_ith(i) still means the
//     same row after the call;
//   * the station refuses every shape where appending a row would change the
//     meaning of the existing ones (packed sequences, equal-seqs batches, M-RoPE,
//     embeddings).
//
// The lane is off unless LLAMA_PREFAB_ENVELOPES names a ladder, and the reason is
// not caution for its own sake: padding changes what upstream sees, so it must be
// a decision an operator made, not a default they inherited.
#include "herd_patch.hpp"
#include "herd_prefab_station.hpp"

#include "llama-batch.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace {

// Upstream keeps this interface in llama-memory.h, which the ubatch header does
// not pull in. The seam only passes it through, so a forward declaration is the
// honest way to name it without including a second private header.
struct llama_memory_context_i;

// The hooked signature, spelled without llm_graph_type / llm_graph_result so this
// file needs no llama-context.h: the enum's underlying type is int, which is what
// the ABI passes, and the result is an opaque pointer that is only handed back.
using process_ubatch_fn = void * (*)(llama_context *, const llama_ubatch &, int,
                                     llama_memory_context_i *, ggml_status &, bool);

process_ubatch_fn g_orig = nullptr;

// The mangled name of llama_context::process_ubatch(const llama_ubatch&,
// llm_graph_type, llama_memory_context_i*, ggml_status&, bool), verified against
// the built archive with nm. A member function has no header declaration we can
// take the address of without including a private header, so the resolver is the
// route -- and it is verified to work on this pin.
constexpr const char * kSymbol =
    "_ZN13llama_context14process_ubatchERK12llama_ubatch14llm_graph_typeP22llama_memory_context_iR11ggml_statusb";

herd::prefab::shape describe(const llama_ubatch & u) {
    herd::prefab::shape s{};
    s.n_tokens     = u.n_tokens;
    s.n_seqs       = u.n_seqs;
    s.n_seq_tokens = u.n_seq_tokens;
    s.n_pos        = u.n_pos;
    s.equal_seqs   = u.equal_seqs();
    s.has_type     = u.type != nullptr;
    s.has_embd     = u.embd != nullptr;
    return s;
}

// A padded copy that owns its arrays. Built as a value and destroyed when the call
// returns: nothing in this file outlives one ubatch.
llama_ubatch pad(const llama_ubatch & u, uint32_t n_env) {
    auto data = std::make_shared<llama_ubatch::data_t>();
    const uint32_t real = u.n_tokens;
    const uint32_t n_pos = u.n_pos;

    data->token      .assign(u.token, u.token + real);
    data->token      .resize(n_env, u.token[real - 1]);
    data->pos        .resize(std::size_t(n_env) * n_pos);
    data->n_seq_id   .assign(u.n_seq_id, u.n_seq_id + real);
    data->n_seq_id   .resize(n_env, 0);
    data->seq_id     .resize(n_env, nullptr);
    data->output     .assign(u.output, u.output + real);
    data->output     .resize(n_env, 0);
    data->seq_idx    .assign(u.seq_idx, u.seq_idx + LLAMA_MAX_SEQ); // guarded by the caller's pointer check
    data->seq_id_unq .assign(u.seq_id_unq, u.seq_id_unq + u.n_seqs_unq);
    data->batch_idxs.resize(n_env, -1);
    if (u.data) {
        std::copy(u.data->batch_idxs.begin(), u.data->batch_idxs.end(), data->batch_idxs.begin());
    }

    // pos is laid out [n_pos][n_env]: section j holds the j-th position of every
    // row. Real rows are copied per section; pad rows repeat the last real row's
    // position in every section, which keeps the array well-formed without
    // asserting anything about a row that will never be read.
    for (uint32_t j = 0; j < n_pos; ++j) {
        std::memcpy(data->pos.data() + std::size_t(j) * n_env,
                    u.pos + std::size_t(j) * real, real * sizeof(llama_pos));
        for (uint32_t i = real; i < n_env; ++i) {
            data->pos[std::size_t(j) * n_env + i] = u.pos[std::size_t(j) * real + real - 1];
        }
    }

    // seq_id entries point into seq_id_data, exactly as llama-batch.cpp builds
    // them. The capacity has to be the SUM of the per-row counts, not the row
    // count: reserving `real` and then pushing more than that in reallocates the
    // buffer, and every pointer already stored in data->seq_id[] then dangles.
    // Those pointers are what the memory context dereferences to find a row's
    // sequences, so the failure would be a wild read inside upstream, far from
    // here. Reserve the exact total, then never grow it again.
    std::size_t total_ids = 0;
    for (uint32_t i = 0; i < real; ++i) {
        total_ids += static_cast<std::size_t>(u.n_seq_id[i]);
    }
    data->seq_id_data.reserve(total_ids); // exact: push_back below cannot reallocate
    for (uint32_t i = 0; i < real; ++i) {
        // A row with no sequences points at its own (empty) slot. When the whole
        // ubatch has no ids at all the vector's data() is null, so take the
        // address arithmetic only when there is storage.
        data->seq_id[i] = total_ids ? data->seq_id_data.data() + data->seq_id_data.size() : nullptr;
        for (int32_t s = 0; s < u.n_seq_id[i]; ++s) {
            data->seq_id_data.push_back(u.seq_id[i][s]);
        }
    }
    llama_seq_id * end = total_ids ? data->seq_id_data.data() + data->seq_id_data.size() : nullptr;
    for (uint32_t i = real; i < n_env; ++i) {
        // Pad rows belong to no sequence, so their pointer carries no rows and
        // their n_seq_id is 0: upstream reads the count before the pointer.
        data->seq_id[i] = end;
    }

    if (u.decision_order) {
        data->decision_order.assign(u.decision_order, u.decision_order + real);
        data->decision_order.resize(n_env, 0);
    }
    if (u.decision_group) {
        data->decision_group.assign(u.decision_group, u.decision_group + real);
        data->decision_group.resize(n_env, 0);
    }

    llama_ubatch padded{};
    padded.b_equal_seqs = 0;   // rows are independent; that is the station's precondition
    padded.n_tokens     = n_env;
    padded.n_seq_tokens = 1;
    padded.n_seqs       = n_env; // one sequence set per row, pads included
    padded.n_seqs_unq   = u.n_seqs_unq;
    padded.n_pos        = n_pos;
    padded.token        = data->token.data();
    padded.embd         = nullptr;
    padded.embd_state   = nullptr;
    padded.embd_row     = nullptr;
    padded.pos          = data->pos.data();
    padded.n_seq_id     = data->n_seq_id.data();
    padded.seq_id       = data->seq_id.data();
    padded.seq_id_unq   = data->seq_id_unq.data();
    padded.seq_idx      = data->seq_idx.data();
    padded.output       = data->output.data();
    padded.type         = nullptr;
    padded.decision_order = data->decision_order.empty() ? nullptr : data->decision_order.data();
    padded.decision_group = data->decision_group.empty() ? nullptr : data->decision_group.data();
    padded.data         = std::move(data);
    return padded;
}

// The one fact that decides whether this lane may ever pad: has a padded pass been
// shown to produce the same result as the stock pass? It has not. Every recorded
// run declined to pad -- 32 of 32 ubatches, all on "n_seq_tokens != 1" -- so the
// padded path has never executed at all, and "the numbers did not change" is not
// evidence about it. While this is false the padded branch is unreachable, in the
// strongest sense the language offers: a compile-time constant, not a flag that
// some environment variable could turn over.
// PADDING AT THIS SEAM IS IMPOSSIBLE, AND THIS IS THE MEASUREMENT THAT SETTLES IT:
//
//   ./llama-corpus score -m tiny-transformer.gguf --corpus …   with the lane padding
//   → ggml.c/llama-kv-cache.cpp:1619
//     GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream())  failed
//     n_tokens = 8, sinfo.size()*n_stream() = 1                    (SIGABRT, exit 134)
//
// The first reading of this file attributed that to the pad rows' ownership. That was
// the wrong diagnosis, and the right one is structural. Read the decode path:
//
//   llama_context::decode          llama-context.cpp:2704  memory->init_update(this, …)
//                                          :2717  mctx->apply()
//                                          :2764  process_ubatch(ubatch, …, mctx.get(), …)
//
// `init_update` is what runs find_slot/prepare, so the KV cell plan -- one index per
// token -- is computed from the ORIGINAL ubatch and handed to process_ubatch through
// `mctx`. This seam receives that plan as an argument it cannot change. A padded
// ubatch of width 8 arrives with a plan of width 1, and the two must agree; upstream
// aborts, correctly.
//
// So no arrangement of the ubatch at this seam can work: the plan is already fixed.
// Padding has to happen where n_tokens is decided and before the plan is built, which
// is llama_batch_allocr's ubatch construction -- a different seam, and one whose output
// this layer does not control.
//
// What remains here is therefore the observation point, plus the two pieces the future
// implementation will need: herd::prefab::place() (may this ubatch be padded, and what
// does it cost) and herd::prefab::scratch_pool (a ledger of cells a caller intends to
// borrow, and of how many are outstanding). Neither is reachable from a running product,
// and the ledger allocates nothing: the cells themselves must come from a sequence the
// implementation owns and clears with llama_memory_seq_rm, which does not exist yet.
constexpr bool kPaddingProven = false;

const char * padding_refusal_reason() {
    return "padding cannot happen at this seam: the KV cell plan is computed before "
           "process_ubatch runs (llama-context.cpp:2704 init_update -> :2764), so a padded "
           "ubatch always disagrees with the plan it is given (GGML_ASSERT at "
           "llama-kv-cache.cpp:1619). Padding belongs at ubatch construction, where n_tokens "
           "is decided.";
}

// Shape dump for the first few declines. Without it, one merged reason name covers
// several distinct causes and the counter cannot say which fired; with it, a run
// states the actual n_tokens/n_seqs/n_seq_tokens/equal_seqs it refused.
void trace_decline(const herd::prefab::shape & s, herd::prefab::verdict why) {
    if (std::getenv("HERD_PREFAB_TRACE") == nullptr) {
        return;
    }
    static std::atomic<int> printed{0};
    if (printed.fetch_add(1) >= 8) {
        return;
    }
    std::fprintf(stderr,
                 "[herd] prefab: decline %s (n_tokens=%u n_seqs=%u n_seq_tokens=%u n_pos=%u "
                 "equal_seqs=%d embd=%d type=%d)\n",
                 herd::prefab::name(why), s.n_tokens, s.n_seqs, s.n_seq_tokens, s.n_pos,
                 s.equal_seqs ? 1 : 0, s.has_embd ? 1 : 0, s.has_type ? 1 : 0);
}

void * herd_process_ubatch(llama_context * ctx, const llama_ubatch & ubatch, int gtype,
                           llama_memory_context_i * mctx, ggml_status & ret, bool expert_decode) {
    using namespace herd::prefab;
    station & lane = shared();
    const ruling verdict = lane.consider(describe(ubatch));

    // The station judges the shape; this checks the pointers the copy would read.
    // They are separate concerns and both must hold: a token-wise ubatch with a
    // null array is still not safe to copy, and the station cannot see pointers
    // without depending on llama's types. Copying a null array is a wild read
    // inside upstream, so an unknown shape is declined, never attempted.
    const bool copyable = ubatch.token && ubatch.pos && ubatch.n_seq_id && ubatch.seq_id &&
                          ubatch.output && ubatch.seq_idx &&
                          (ubatch.n_seqs_unq == 0 || ubatch.seq_id_unq != nullptr);
    if (!kPaddingProven || verdict.why != herd::prefab::verdict::accept || !copyable) {
        // Counted, and traced only behind HERD_PREFAB_TRACE: a decline is the normal
        // path for most ubatches, so an unconditional fprintf would be a flood.
        lane.note_declined(copyable ? verdict.why : herd::prefab::verdict::mixed_input);
        // The station's verdict and this seam's reason are different facts: the shape
        // may be acceptable while the seam still refuses, because padding itself is
        // refuted. Printing the verdict alone labelled every such run "decline accept",
        // which reads as a contradiction. Say which of the two refused.
        if (verdict.why != herd::prefab::verdict::accept) {
            trace_decline(describe(ubatch), verdict.why);
        } else if (!copyable) {
            trace_decline(describe(ubatch), herd::prefab::verdict::mixed_input);
        } else if (std::getenv("HERD_PREFAB_TRACE") != nullptr) {
            static std::atomic<int> printed{0};
            if (printed.fetch_add(1) < 8) {
                std::fprintf(stderr,
                             "[herd] prefab: shape accepted but not padded -- padding is refuted; "
                             "see kPaddingProven in this file\n");
            }
        }
        return g_orig(ctx, ubatch, gtype, mctx, ret, expert_decode);
    }

    const llama_ubatch padded = pad(ubatch, verdict.n_env);
    void * result = g_orig(ctx, padded, gtype, mctx, ret, expert_decode);
    // The counters move only after the padded call returned, so a lane that never
    // actually ran cannot report success.
    lane.note_ran(verdict.n_env, ubatch.n_tokens);
    return result;
}

} // namespace

// ---------------------------------------------------------------------------
// Accounting on exit, so the lane's effect is observable from outside.
//
// Without this, "the numbers did not change" has two indistinguishable causes:
// the padding was neutral (the wanted result) or the lane never ran at all. The
// first version of this module could not tell them apart, and reported an
// equality it had not earned.
//
// Gated behind HERD_SEAM_TRACE: a product's stderr is part of its output
// contract, and this is a diagnostic.
// ---------------------------------------------------------------------------
namespace {
// Whether this process ever installed the seam. The lane's numbers mean nothing without
// it: a report that says "0 padded, declines: none" while the hook was never installed is
// not a quiet lane, it is a lane that did not exist -- and the two are written by the same
// fprintf. Measured: a clean build links this file into the product, no environment switch
// is set, and the exit report read "0 padded ubatch(es), 0 pad row(s), declines: none",
// which a reader takes as "prefab ran and found nothing to pad". It had not run at all.
//
// No flag is kept here: the installer's `Seam::installed` is the owner of that fact, and a
// copy of it in this file would be a second thing to keep true -- the previous revision's
// copy was never set at all.
struct lane_report {
    ~lane_report() {
        if (std::getenv("HERD_SEAM_TRACE") == nullptr) {
            return;
        }
        // Whether the seam exists and whether the hook took are two facts, and the counters
        // mean nothing unless both hold. This module is linked into every product while the
        // hook is installed only under HERD_PREFAB_OBSERVE, so a plain run printed
        // "0 padded ubatch(es), 0 pad row(s), declines: none": every clause true, the line a
        // lie, because it reads as "every ubatch was examined and none refused".
        //
        // The answer is asked of the installer, not cached here. The previous version of this
        // guard kept its own `g_seam_installed` flag and never set it to true anywhere, so the
        // report said NOT INSTALLED even when the observer was running -- the same defect with
        // the sign flipped. One fact, one owner: `Seam::installed` is already that owner, and
        // install_all() runs at priority 102, after every constructor, so only a query made
        // here (at exit) can see the truth.
        const herd::Seam * seam = herd::find("prefab.process_ubatch");
        if (!seam) {
            std::fprintf(stderr,
                         "[herd] prefab: no seam registered -- this run is not a prefab "
                         "measurement and its counters do not exist\n");
            return;
        }
        if (!seam->installed) {
            // No counters are printed here, deliberately. Zeroes from a lane that was never
            // installed are a false measurement shape: indistinguishable from a healthy
            // run, and the reason the previous version was believed.
            std::fprintf(stderr,
                         "[herd] prefab: seam registered but NOT installed (HERD_PREFAB_OBSERVE "
                         "unset or the hook failed) -- no ubatch was observed, so there is no "
                         "report to give\n");
            return;
        }
        const auto counters = herd::prefab::shared().snapshot();
        std::fprintf(stderr,
                     "[herd] prefab: %llu padded ubatch(es), %llu pad row(s), declines:", 
                     static_cast<unsigned long long>(counters.accepted),
                     static_cast<unsigned long long>(counters.pad_rows));
        bool any = false;
        for (int why = 0; why < 8; ++why) {
            if (counters.declined[why] != 0) {
                std::fprintf(stderr, " %s=%llu", herd::prefab::name(herd::prefab::verdict(why)),
                             static_cast<unsigned long long>(counters.declined[why]));
                any = true;
            }
        }
        std::fprintf(stderr, "%s\n", any ? "" : " none");
        if (!kPaddingProven) {
            // Even installed, "0 padded" has two readings: every shape was refused, or this
            // build refuses to pad at all. Today it is always the second. Without this
            // sentence the zero looks like a measurement of the lane's appetite.
            std::fprintf(stderr,
                         "[herd] prefab: this build pads nothing by construction "
                         "(kPaddingProven is false); the zero above is not a shape verdict\n");
        }
    }
} g_lane_report;
} // namespace

// ---------------------------------------------------------------------------
// Wiring: resolve, register, and let the shared installer hook it.
//
// The lane has no section descriptor because the target is a member function,
// whose address is not a constant expression in the Itanium ABI; registration is
// therefore a call from this constructor, which runs at priority 101 -- before any
// ordinary initializer and before main().
// ---------------------------------------------------------------------------
#if !defined(HERD_NO_AUTOLOAD)
__attribute__((constructor(101))) static void herd_prefab_autoload() {
    if (std::getenv("HERD_NO_AUTOLOAD") != nullptr) {
        return;
    }
    // The environment is read here, not through the station's lazy configure.
    // Measured: with the lazy path the lane reported nothing at all -- no seam, no
    // message -- while LLAMA_PREFAB_ENVELOPES was set. A lane that was asked for
    // and then silently not installed is indistinguishable from a working run, so
    // the request is honoured here or the process stops.
    const char * ladder_text = std::getenv("LLAMA_PREFAB_ENVELOPES");
    if (!ladder_text || !*ladder_text) {
        // The operator did not ask for the lane, and nothing is installed: the exit report
        // says so by asking the installer, rather than reporting zeroes that read as a quiet,
        // successful prefab run.
        return;
    }
    // LLAMA_PREFAB_ENVELOPES is a request for the prefab lane, and the lane does not
    // exist: padding is refuted (kPaddingProven, and why, in this file's banner). A
    // request that installs a pass-through and exits 0 is the worst of the three
    // possible answers -- the operator asked for the feature, got none of it, and the
    // run reports success, so nothing downstream can tell the difference. So the
    // request fails, here, before main() and before a model is read.
    //
    // Installing an observer used to live on this path. That was my mistake in an
    // earlier revision: the diagnostic is genuinely useful, but it is a different
    // capability and must not be reachable by asking for the feature. It now has its
    // own switch below.
    std::fprintf(stderr,
                 "[herd] prefab: LLAMA_PREFAB_ENVELOPES=\"%s\" asks for the prefab lane, which this "
                 "build does not have. %s Unset it to run normally, or set HERD_PREFAB_OBSERVE=1 "
                 "for a shape trace that pads nothing.\n",
                 ladder_text, padding_refusal_reason());
    std::abort();
}

// The observation point, on its own switch and its own name.
//
// It hooks the same function and pads nothing: it reports the ubatch shapes upstream
// produces and why each was not padded. That data is what a future implementation
// needs, and it is only visible from inside the seam.
//
// Why a separate variable rather than a mode of the one above: an operator who sets
// HERD_PREFAB_OBSERVE is asking for a trace and knows nothing changes. An operator
// who sets LLAMA_PREFAB_ENVELOPES is asking for the feature -- and gets a stop, not a
// substitute. One variable, one meaning.
__attribute__((constructor(101))) static void herd_prefab_observe_autoload() {
    if (std::getenv("HERD_NO_AUTOLOAD") != nullptr) {
        return;
    }
    if (std::getenv("HERD_PREFAB_OBSERVE") == nullptr) {
        return;
    }
    const char * ladder_text = std::getenv("LLAMA_PREFAB_ENVELOPES");
    void * target = herd::resolve(kSymbol);
    if (!target) {
        std::fprintf(stderr,
                     "[herd] prefab: HERD_PREFAB_OBSERVE is set but llama_context::process_ubatch "
                     "was not found; the observer cannot be installed\n");
        std::abort();
    }
    static_cast<void>(herd::add("prefab.process_ubatch", target, herd::raw(&herd_process_ubatch),
                                reinterpret_cast<void **>(&g_orig), nullptr));
    std::fprintf(stderr,
                 "[herd] prefab: OBSERVER only (pads nothing), envelopes=%s. %s\n",
                 ladder_text && *ladder_text ? ladder_text : "(none)", padding_refusal_reason());
}
#endif
