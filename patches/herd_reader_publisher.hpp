// The production form of the graph-owned index publisher: one `llm_graph_input_i` that writes a
// reader emission's gather table into tensors the scheduler has placed.
//
// WHY IT IS A SEPARATE MODULE AND NOT A TEST FIXTURE
// --------------------------------------------------
// The behaviour was proven in a test (`tq_reader.graph_result_publish`) against upstream's real
// `llm_graph_result`, and then left there. A test that owns the only copy of a mechanism is a
// mechanism nothing can use: the patch layer could not register it, so the publication existed and
// was unreachable. This header is that same object, with the two properties a test never needed and
// production cannot do without.
//
// PROPERTY 1: THE STATISTICS CANNOT DANGLE
// ----------------------------------------
// The obvious shape -- `publisher{result, reader_publisher_stats * stats}` -- forces the caller to
// keep `stats` alive longer than `llm_graph_result`. That ordering is invisible in the type system,
// it is violated the moment someone writes `llm_graph_result res(8192); stats s;` (destruction runs
// in reverse: `res` dies first, its publisher's destructor writes into the dead `s`), and the failure
// is a write to freed memory that usually looks like a correct number.
//
// So the statistics are `std::shared_ptr<publisher_stats>`, copied into the publisher. The input
// owns one reference; the caller keeps another. Whichever dies first, the block stays alive, and
// after `reset()` the caller can still read what the publication did -- which is exactly the moment
// a diagnostic is wanted. Cost: one atomic refcount per graph input, on a path that runs once per
// decode and already creates hundreds of ggml objects.
//
// PROPERTY 2: EVERY FAILURE STOPS THE RUN
// ---------------------------------------
// Three things can go wrong at `set_input` time, and each used to be either invisible or fatal in
// the wrong way:
//
//   * the tensors are not placed yet        -> `no_storage`; nothing is written
//   * the tiling is wrong or a write fails  -> `publish_failed`
//   * `set_input` is called a second time   -> the table a compute may be reading would be rewritten
//
// The test fixture this replaces answered all three with a returned flag, and the third with a plain
// `if (wrote_once) return;`. That is a silent no-op in the one place where silence costs most: a
// caller who thinks a fresh row table was published for a new decode gets the previous one, and every
// later number is plausible and wrong. A flag alone is bookkeeping for a post-mortem; it does not stop
// the host from computing over unwritten indices, and treating it as a safety property was the defect.
//
// So each of the three calls `publisher_stop_host`, which ABORTS. The statistics are still filled in
// first, so the post-mortem has the reason, but the process does not continue to a compute that would
// read whatever the placement left in those tensors. `set_input` is `virtual void`
// (`llama-graph.h:117`), so there is no return code to fail with; and it is reached through
// `llama_process` (`extern "C"`) -> `decode` -> `process_ubatch`, a chain with no `catch` on it, so an
// exception would unwind across a C ABI. Abort is the only signal that cannot be ignored and cannot
// cross a boundary it does not own. See `publisher_stop_host` for the traced call chain.
//
// `can_reuse` is OVERRIDDEN to return `false`, explicitly rather than by relying on the base default:
// a graph carrying this input must not be reused, because reuse would neither re-register the
// publisher nor rewrite the indices, and there is nothing in `llm_graph_input_i` that could detect it
// after the fact. Returning `true` would let upstream submit last decode's graph with this decode's
// data.
//
// WHAT THIS DOES NOT DO
// ---------------------
// It does not find the publication window, and it does not decide anything about prefill. It is
// constructed by whoever is building a graph (from a `build_attn_mha` hook, a prefab lane, or a
// test), registered with `llm_graph_result::add_input`, and then driven by upstream's `set_inputs`.
// No product path constructs one today.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>

#include "llama-graph.h"

#include <tq/reader/graph_emit_physical.h>

namespace herd::reader {

// What one publisher did, and what went wrong. Owned by `std::shared_ptr` so it survives the input.
struct publisher_stats {
    int  set_input_calls = 0;   // how many times upstream's `set_inputs` reached this input
    int  writes = 0;            // how many times a row table was actually written (correct: 0 or 1)
    bool destroyed = false;     // set by the input's destructor: `reset()` freed it

    // The last publication's verdict. `ok` before the first attempt, so a caller that checks it
    // without having published yet is not told "everything is fine" -- see `attempted`.
    tq::reader::index_publish_refusal last = tq::reader::index_publish_refusal::ok;
    bool attempted = false;
    std::string why;

    // Set by any of: a second `set_input`, a publication that failed, or a publication attempted
    // before placement. A non-zero value means the run must not be trusted, and unlike a return code
    // it cannot be lost by an intermediate layer that ignores one.
    bool contract_violation = false;
};

using publisher_stats_ptr = std::shared_ptr<publisher_stats>;

// The one way this class refuses to let the host proceed: ABORT. Unconditionally, and the reason is
// that a throw would have to cross an ABI boundary this project does not own.
//
// The chain from `set_input` back to the application was traced, not assumed:
//
//   llama_process                       third_party/upstream/src/llama-context.cpp:5623  extern "C"
//     -> ctx->decode(*batch)            :5641 (a switch; no try/catch)
//       -> llama_context::process_ubatch :1997
//         -> res->set_inputs(&ubatch)    :2068
//
// Every frame in that chain is free of try/catch -- the 21 catches in llama-context.cpp sit at
// 1306-1531 and 4246-4303, all inside other functions. So an exception thrown here would unwind out
// through `extern "C" int32_t llama_process(...)` into whatever called it. That is the boundary: the
// declaration is C-linkage, the caller is not necessarily C++, and unwinding across a C ABI is at
// best implementation-defined and at worst (once any frame in between is C, as ggml's are) undefined
// behaviour. A patch layer that relies on it is betting the process on a calling convention.
//
// There is a second reason, independent of ABI: `set_input` is `virtual void` (llama-graph.h:117), so
// there is no return code to fail with. The choice is only between "signal that some hosts may
// ignore" and "stop". This project's rule is that a lane which cannot produce a trustworthy number
// must not produce one, so the stop is unconditional -- and it also makes `-fno-exceptions` builds
// identical, because no throw remains to be compiled out.
//
// The cost is real and accepted: a host can no longer turn this into a recoverable decode error. That
// is the intended trade -- a decode whose row table was never written has no correct result to
// recover to, and a retry would write into the same tensors.
[[noreturn]] inline void publisher_stop_host(const char * what, const std::string & detail) {
    std::string message = std::string("[herd] reader.publisher: ") + what + " -- " + detail +
                          ". The graph must not compute: its gather indices were never written, so "
                          "the attention would read whatever the placement left in those tensors. "
                          "Aborting instead of returning (a returned status cannot stop the host's "
                          "compute) and instead of throwing (the call chain up to llama_process has "
                          "no catch, so the exception would unwind across the C ABI).";
    std::fprintf(stderr, "%s\n", message.c_str());
    std::fflush(stderr);
    std::abort();
}

// The graph-owned input. Construction takes the emission by value: the caller's `physical_emit_result`
// is usually a temporary returned from the emitter, and a reference into it would dangle at the end
// of the statement that created it -- silently, because the entries are plain pointers.
class publisher final : public llm_graph_input_i {
public:
    // Refuses a null stats pointer instead of writing through it in the destructor. This is a
    // programming error at construction, not bad input, so the refusal is immediate and fatal:
    // carrying it would mean every later `set_input` writes into a null shared_ptr target.
    publisher(tq::reader::physical_emit_result result, publisher_stats_ptr stats)
        : result_(std::move(result)), stats_(std::move(stats)) {
        if (!stats_) {
            std::fprintf(stderr,
                         "[herd] reader.publisher: constructed with a null stats pointer; the "
                         "destructor would write through it. Refusing at construction rather than "
                         "at teardown.\n");
            std::fflush(stderr);
            std::abort();
        }
        if (result_.indices.empty()) {
            // An emission built without `collect_indices` has nothing to publish, which makes this
            // object unable to honour its one job. That is a construction-time configuration error,
            // so it stops HERE -- before a graph exists -- rather than at decode time: the host has
            // not started building anything, so refusing costs nothing, and the alternative is a
            // publisher that silently writes nothing and lets the graph gather uninitialised memory.
            stats_->contract_violation = true;
            stats_->why = "adopted an emission with no index entries (collect_indices was off?)";
            publisher_stop_host("an emission with no index entries was adopted", stats_->why);
        }
    }

    ~publisher() override { stats_->destroyed = true; }

    publisher(const publisher &) = delete;
    publisher & operator=(const publisher &) = delete;
    publisher(publisher &&) = delete;
    publisher & operator=(publisher &&) = delete;

    // A CONTRACT VIOLATION STOPS THE HOST. IT IS NOT RECORDED AND THEN IGNORED.
    //
    // The previous revision set `contract_violation` and returned, which left the host free to
    // compute a graph whose gather indices were never written. On the CPU those indices are whatever
    // placement left in them, so the graph reads past k/v and returns numbers that are wrong and
    // plausible -- the exact failure class this whole lane exists to refuse. Counting violations is
    // bookkeeping for a post-mortem; it is not a safety property, and treating it as one was the
    // defect.
    //
    // WHY AN EXCEPTION IS THE RIGHT CHANNEL, verified rather than assumed:
    //
    //   * `llm_graph_input_i::set_input` is `virtual void` (llama-graph.h:117), so a return code is
    //     not available; the only in-band signals are throw and abort.
    //   * The decode call site (`llama-context.cpp:2068`, `res->set_inputs(&ubatch)`) is NOT inside
    //     any try block -- the nearest `try {`/`catch (` are at 1525/1531, in an earlier function --
    //     so a thrown exception propagates out of `decode_impl` instead of being swallowed. Nothing
    //     between here and the application turns it into a warning.
    //   * Therefore the run either stops before the compute (host lets it propagate) or is caught by
    //     a host that knows how to fail a decode. Either way the graph never computes over an
    //     unpublished row table.
    //
    // A build without exceptions cannot throw, so it aborts with the same message. Both outcomes stop
    // the run; neither produces a number.
    void set_input(const llama_ubatch * /*ubatch*/) override {
        ++stats_->set_input_calls;
        // A second call: `can_reuse` is false, so upstream must not submit this graph again, and if it
        // does the row table would be rewritten underneath a compute that may already be reading it.
        if (stats_->writes > 0) {
            stats_->contract_violation = true;
            if (stats_->why.empty()) {
                stats_->why = "set_input was called again after a successful publish; the indices "
                              "on this graph were already written and a reuse would need a fresh "
                              "emission, not a repeat";
            }
            publisher_stop_host("set_input called twice", stats_->why);
        }

        stats_->attempted = true;
        std::string why;
        stats_->last = tq::reader::publish_index_inputs(result_, &backend_writer, &why, nullptr);
        if (stats_->last != tq::reader::index_publish_refusal::ok) {
            stats_->contract_violation = true;
            stats_->why = std::string(tq::reader::describe(stats_->last)) + " -- " + why;
            publisher_stop_host("the row table could not be published", stats_->why);
        }
        ++stats_->writes;
    }

    // Owned by upstream and destroyed with the graph. Not overridden: `false` is the true answer for
    // a graph whose indices were written for one decode, and returning `true` would let upstream
    // submit it for another.
    bool can_reuse(const llm_graph_params &) override { return false; }

    const tq::reader::physical_emit_result & emission() const { return result_; }
    const publisher_stats & report() const { return *stats_; }

private:
    // Through the backend, never `memcpy` into `data`: a non-null `data` says the tensor was placed,
    // not that the pointer is host memory. On a device backend it is a device address, and a memcpy
    // there either faults or writes somewhere unrelated while every check on the host copy passes.
    static bool backend_writer(ggml_tensor * tensor, const void * data, std::size_t offset,
                               std::size_t size) {
        if (tensor == nullptr || tensor->buffer == nullptr) return false;
        ggml_backend_tensor_set(tensor, data, offset, size);
        return true;
    }

    tq::reader::physical_emit_result result_;
    publisher_stats_ptr stats_;
};

} // namespace herd::reader
