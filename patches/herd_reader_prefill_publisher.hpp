// The graph-owned publisher for a PREFILL emission: one `llm_graph_input_i` that writes the gather
// table of every token, in one pass, at the only moment those tensors can be written.
//
// WHY A SEPARATE CLASS AND NOT `publisher` IN A LOOP
// --------------------------------------------------
// `publisher` (herd_reader_publisher.hpp) holds ONE `physical_emit_result`. A prefill produces T of
// them, and `publish_index_inputs` requires each set to tile ONE buffer contiguously -- which is
// exactly why T sets cannot be merged into one call. So the publication is T calls. Putting that
// loop at a call site would put the ordering, the all-or-nothing question and the failure channel in
// three different places; here they are one object with one `set_input`.
//
// THE ORDERING RULE, AND WHY IT MATTERS MORE HERE THAN FOR ONE TOKEN
// -----------------------------------------------------------------
// With T tokens a naive loop writes token 0, then token 1, and so on. If token 3 is not placed,
// tokens 0..2 are already written into a graph that must be discarded -- harmless if the run stops,
// but it means the state after a refusal is not the state before it. So this class is TWO PASSES:
//
//   pass 1  validate every token with a writer that CHECKS the tensor and returns true WITHOUT
//           writing. `publish_index_inputs` does its structural validation (tiling, dtype, shape,
//           `no_storage`) before it calls the writer, so a pass-1 refusal means nothing was written
//           by any token.
//   pass 2  only if pass 1 was clean for all T, write with the backend writer.
//
// A pass-2 failure can still leave a partial publication (the writer failed mid-way), and that is
// stated rather than hidden: the host is stopped, so no compute follows, and the graph is discarded.
// What pass 1 buys is that the COMMON failure -- a caller who published before placement -- costs
// zero writes instead of T.
//
// FAILURE STOPS THE HOST, UNCONDITIONALLY
// ---------------------------------------
// Same reasoning as the single-token publisher, and it is worth not re-deriving: `set_input` is
// `virtual void` (llama-graph.h:117), so there is no return code; the call chain
// `llama_process` (extern "C") -> `decode` -> `process_ubatch` -> `set_inputs` has no `catch` on it,
// so a throw would unwind across a C ABI. `publisher_stop_host` aborts. A flag alone would let the
// host compute a graph gathering through uninitialised indices, which is a wrong answer that looks
// right -- the failure class this whole lane refuses.
//
// LIFETIME
// --------
// The object is a `unique_ptr` inside `llm_graph_result`: it dies with the graph, and the tensors it
// borrows belong to the same context. "The graph died but the publisher still writes" is not
// representable. The statistics are `shared_ptr` so a caller can read them AFTER `reset()` -- which
// is exactly when a diagnostic is wanted, and exactly when a raw pointer would have been a write to
// freed memory.
//
// `can_reuse` returns false. A graph carrying written row tables must not be submitted twice: reuse
// would neither re-register this input nor rewrite the indices, and nothing in `llm_graph_input_i`
// could detect the substitution afterwards.
//
// WHAT THIS DOES NOT DO
// ---------------------
// It does not find the publication window, it does not decide anything about prefill admission (see
// `emit_prefill_attention`'s preflight), and no product path constructs one today. The replacement
// switches still refuse at startup.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "llama-graph.h"

#include <tq/reader/graph_emit_physical.h>
#include <tq/reader/graph_emit_prefill.h>

namespace herd::reader {

// What one prefill publication did. Shared-owned so it outlives the input.
struct prefill_publisher_stats {
    int set_input_calls = 0;   // how many times upstream's `set_inputs` reached this input
    int passes = 0;            // completed publication passes (correct: 0 or 1)
    int tokens_published = 0;  // how many tokens' row tables were written in that pass
    bool destroyed = false;    // set by the input's destructor
    bool attempted = false;
    bool contract_violation = false;

    // Per-token verdicts from the LAST attempt, so a refusal names which token rather than
    // reporting "the prefill publication failed". Empty before the first attempt.
    std::vector<tq::reader::index_publish_refusal> per_token;
    std::string why;
};

using prefill_publisher_stats_ptr = std::shared_ptr<prefill_publisher_stats>;

// The stop channel, identical in behaviour to `publisher_stop_host` in herd_reader_publisher.hpp and
// deliberately duplicated rather than included: that header defines the single-token class, and a
// prefill test that included it would link two independent llm_graph_input_i subclasses into one
// translation unit for no benefit. One three-line function is a cheaper duplication than that
// coupling.
[[noreturn]] inline void prefill_publisher_stop_host(const char * what, const std::string & detail) {
    std::fprintf(stderr,
                 "[herd] reader.prefill_publisher: %s -- %s. The graph must not compute: its per-token "
                 "gather indices were not all written, so the attention would read whatever the "
                 "placement left in those tensors. Aborting instead of returning (a returned status "
                 "cannot stop the host's compute) and instead of throwing (the call chain up to "
                 "llama_process has no catch, so the exception would unwind across the C ABI).\n",
                 what, detail.c_str());
    std::fflush(stderr);
    std::abort();
}

class prefill_publisher final : public llm_graph_input_i {
public:
    // Takes the emission BY VALUE: `prefill_result` is a value returned from the emitter and usually
    // a temporary, and the entries are plain pointers into its tensors -- a reference would dangle at
    // the end of the statement that created it, silently.
    prefill_publisher(tq::reader::prefill_result result, prefill_publisher_stats_ptr stats)
        : result_(std::move(result)), stats_(std::move(stats)) {
        if (!stats_) {
            prefill_publisher_stop_host("constructed with a null stats pointer",
                                        "the destructor would write through it");
        }
        // THE EMISSION ITSELF IS VALIDATED HERE, AND NOT ONLY ITS LENGTH.
        //
        // The previous revision checked `per_token.empty()` and nothing else, so it would ADOPT a
        // failed emission and publish it: `emit_prefill_attention` returns `partial_graph` (ggml grew
        // the arena past the charge) or `inner_refusal` (a token emission refused mid-build) with
        // `out == nullptr` and a `per_token` vector that may be short, empty, or hold refused entries.
        // Publishing such a result means writing row tables for a graph the emitter itself declared
        // broken -- and the host would then compute over them, which is exactly the outcome this lane
        // refuses. The refusal has to happen at CONSTRUCTION, where no graph exists yet and stopping
        // costs nothing, rather than at `set_input` time inside the host's decode.
        if (result_.refusal != tq::reader::prefill_refusal::ok) {
            stats_->contract_violation = true;
            stats_->why = std::string("adopted a FAILED prefill emission: ") +
                          tq::reader::describe(result_.refusal) +
                          (result_.why.empty() ? "" : " -- " + result_.why);
            prefill_publisher_stop_host("a failed prefill emission was adopted", stats_->why);
        }
        // `ok` with no output, or with a per-token list that does not cover one emission per token,
        // is not a usable result either. The emitter promises `out` non-null on `ok`; checking it
        // here means a contract drift surfaces at construction instead of as a null dereference in
        // the readback path.
        if (result_.out == nullptr || result_.weights == nullptr) {
            stats_->contract_violation = true;
            stats_->why = "the emission reports ok but its output (or weights) tensor is null";
            prefill_publisher_stop_host("a prefill emission with no output was adopted", stats_->why);
        }
        // A prefill with no per-token emissions cannot honour the one job this object has, and that
        // is a configuration error at the point of construction -- no graph exists yet, so refusing
        // costs nothing. The alternative is a publisher that writes nothing and lets the graph gather
        // uninitialised memory.
        if (result_.per_token.empty()) {
            stats_->contract_violation = true;
            stats_->why = "adopted a prefill emission with no per-token results";
            prefill_publisher_stop_host("a prefill emission with no per-token results was adopted",
                                        stats_->why);
        }
        // Each per-token emission must itself be a complete, accepted result with entries to write.
        // This is the check that makes "T emissions" a meaningful count rather than a vector length:
        // a refused entry here would be published as if it had indices, and `publish_index_inputs`
        // would then report `no_entries` deep inside pass 1 with the host already committed.
        for (std::size_t t = 0; t < result_.per_token.size(); ++t) {
            const tq::reader::physical_emit_result & e = result_.per_token[t];
            if (e.refusal != tq::reader::physical_emit_refusal::ok || e.out == nullptr ||
                e.indices.empty()) {
                stats_->contract_violation = true;
                stats_->why = "token " + std::to_string(t) + ": " +
                              tq::reader::describe(e.refusal) +
                              (e.out == nullptr ? " (null output)" : "") +
                              (e.indices.empty() ? " (no index entries)" : "");
                prefill_publisher_stop_host("a per-token emission is not usable", stats_->why);
            }
        }
        stats_->per_token.assign(result_.per_token.size(),
                                 tq::reader::index_publish_refusal::ok);
    }

    ~prefill_publisher() override { stats_->destroyed = true; }

    prefill_publisher(const prefill_publisher &) = delete;
    prefill_publisher & operator=(const prefill_publisher &) = delete;
    prefill_publisher(prefill_publisher &&) = delete;
    prefill_publisher & operator=(prefill_publisher &&) = delete;

    void set_input(const llama_ubatch * /*ubatch*/) override {
        ++stats_->set_input_calls;

        // `can_reuse` is false, so upstream must not submit this graph twice. If it does, the row
        // tables would be rewritten underneath a compute that may already be reading them.
        if (stats_->passes > 0) {
            stats_->contract_violation = true;
            if (stats_->why.empty()) {
                stats_->why = "set_input was called again after a complete publication; a reuse would "
                              "need a fresh emission, not a repeat";
            }
            prefill_publisher_stop_host("set_input called twice", stats_->why);
        }

        stats_->attempted = true;

        // PASS 1 -- validate every token, write NOTHING.
        //
        // `publish_index_inputs` runs its structural checks (entry tiling, dtype, shape, `no_storage`)
        // BEFORE it invokes the writer, so a writer that returns true without touching the tensor
        // turns the whole call into a dry run. That is what makes "nothing was written" a property of
        // a refusal instead of a hope.
        for (std::size_t t = 0; t < result_.per_token.size(); ++t) {
            std::string why;
            const auto r = tq::reader::publish_index_inputs(result_.per_token[t], &validate_writer,
                                                            &why, nullptr);
            stats_->per_token[t] = r;
            if (r != tq::reader::index_publish_refusal::ok) {
                stats_->contract_violation = true;
                stats_->why = "token " + std::to_string(t) + ": " +
                              tq::reader::describe(r) + " -- " + why;
                prefill_publisher_stop_host("a token's row table cannot be published", stats_->why);
            }
        }

        // PASS 2 -- every token validated, so now write. A failure here means the backend refused
        // mid-way; the entries before it stay written, which is why the host is stopped and the
        // graph must be discarded rather than retried.
        for (std::size_t t = 0; t < result_.per_token.size(); ++t) {
            std::string why;
            const auto r = tq::reader::publish_index_inputs(result_.per_token[t], &backend_writer,
                                                            &why, nullptr);
            stats_->per_token[t] = r;
            if (r != tq::reader::index_publish_refusal::ok) {
                stats_->contract_violation = true;
                stats_->why = "token " + std::to_string(t) + " failed during the write: " +
                              tq::reader::describe(r) + " -- " + why +
                              " (entries before this one were already written; the graph must be "
                              "discarded)";
                prefill_publisher_stop_host("a token's row table failed during the write",
                                            stats_->why);
            }
            ++stats_->tokens_published;
        }

        ++stats_->passes;
    }

    // Owned by upstream and destroyed with the graph. Not overridden: `false` is the true answer for
    // a graph whose row tables were written for one decode.
    bool can_reuse(const llm_graph_params &) override { return false; }

    const tq::reader::prefill_result & emission() const { return result_; }
    const prefill_publisher_stats & report() const { return *stats_; }
    std::size_t token_count() const { return result_.per_token.size(); }

private:
    // Returns true without writing. Used for pass 1; see the ordering note above.
    static bool validate_writer(ggml_tensor * tensor, const void *, std::size_t, std::size_t) {
        // A tensor with no buffer has not been placed. `publish_index_inputs` checks this itself, so
        // reaching here with a null buffer would be a hole in that check -- reported as false rather
        // than assumed impossible.
        return tensor != nullptr && tensor->buffer != nullptr;
    }

    // Through the backend, never `memcpy` into `data`: a non-null `data` says the tensor was placed,
    // not that the pointer is host memory.
    static bool backend_writer(ggml_tensor * tensor, const void * data, std::size_t offset,
                               std::size_t size) {
        if (tensor == nullptr || tensor->buffer == nullptr) return false;
        ggml_backend_tensor_set(tensor, data, offset, size);
        return true;
    }

    tq::reader::prefill_result result_;
    prefill_publisher_stats_ptr stats_;
};

} // namespace herd::reader
