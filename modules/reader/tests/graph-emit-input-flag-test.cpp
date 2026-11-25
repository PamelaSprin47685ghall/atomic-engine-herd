// Which tensors in a reader emission need GGML_TENSOR_FLAG_INPUT for a real scheduler to place it?
//
// The answer decides whether the reader lane can ever run inside llama.cpp's graph, so it is a
// measurement. Three earlier rounds of this lane disagreed about it:
//
//   1. "the scheduler aborts because the graph is not something llama.cpp builds"      (no cause)
//   2. "the flags are the root cause"                                                  (no control)
//   3. "the flags are NOT required -- a control without them still passed"             (wrong control)
//
// Round 3's control was the instructive one: it deleted three `ggml_set_input(...)` calls that live
// in a DIFFERENT test function, while the scheduler test set its flags with `|=` on the tensors
// after the emission. It changed nothing about the thing under test, so its pass proved nothing.
//
// So the question is asked here as an experiment with four variants, each in its own process because
// the failure mode is GGML_ABORT (ggml-alloc.c:686, `buffer_id >= 0`), which cannot be observed
// in-process:
//
//   variant 0  no flags                     expect: abort
//   variant 1  k, v, q flagged              expect: alloc_graph succeeds
//   variant 2  index tensors flagged only   expect: abort
//   variant 3  everything flagged           expect: alloc_graph succeeds
//
// Measured on this pin: exactly that. So the requirement is on **k, v, q** -- the tensors the
// emission reads as view sources and gathers from -- and NOT on the per-span index tensors. That is
// the opposite of what the word "index" suggests, which is why it is an experiment.
//
// Why k/v/q specifically: `ggml_gallocr_alloc_graph_impl` (ggml-alloc.c:786-820) allocates
// `graph->leafs` first, then walks nodes allocating only what carries the INPUT flag -- the node
// itself, and each `src` that carries it. The cache and query tensors are reached through views
// (`ggml_view_3d(ctx, k, ...)`), and a view is not a leaf, so they are not in that first sweep;
// without the flag nothing allocates them and the first view that needs one aborts. The index
// tensors are leaves -- `get_rows(k, idx)` reads them directly -- so the leaf sweep covers them.
//
// What this does NOT establish: that every other reader graph will have the same membership. It
// establishes the requirement for an emission over a cache and a query, which is the only shape
// this lane builds today. A new operand that arrives through a view rather than as a leaf needs its
// own row in this table.
#include <tq/reader/attention.h>
#include <tq/reader/graph_emit.h>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

constexpr int kQHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 8;
constexpr int kCells = 12;

// Variant outcomes, in the child process's exit code.
enum outcome { kOk = 0, kAllocFailed = 1, kEmissionFailed = 8, kContextFailed = 9 };

// Builds the emission, applies the variant's flags, and asks the scheduler to place it.
int run_variant(int variant) {
    ggml_context * ctx = ggml_init({32u * 1024u * 1024u, nullptr, /*no_alloc=*/true});
    if (!ctx) {
        return kContextFailed;
    }

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCells;
    in.collect_indices = true;

    tq::reader::query_view view;
    tq::reader::span_keys first;
    first.effective_pos = 5;
    first.cells = {0, 3, 7, 9};
    tq::reader::span_keys second;
    second.effective_pos = 6;
    second.cells = {11, 1, 4};
    view.spans = {first, second};

    const auto emitted = tq::reader::emit_reader_attention(in, view, 0.70710678f, nullptr);
    if (emitted.refusal != tq::reader::emit_refusal::ok || !emitted.out) {
        ggml_free(ctx);
        return kEmissionFailed;
    }
    if (emitted.indices.size() != view.spans.size()) {
        // The entries are how this test reaches the index tensors now that there is no callback.
        ggml_free(ctx);
        return kEmissionFailed;
    }

    // The four variants differ ONLY here.
    if (variant == 1 || variant == 3) {
        in.k->flags |= GGML_TENSOR_FLAG_INPUT;
        in.v->flags |= GGML_TENSOR_FLAG_INPUT;
        in.q->flags |= GGML_TENSOR_FLAG_INPUT;
    }
    if (variant == 2 || variant == 3) {
        for (const auto & entry : emitted.indices) {
            entry.index->flags |= GGML_TENSOR_FLAG_INPUT;
        }
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, emitted.out);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    if (!cpu) {
        ggml_free(ctx);
        return kContextFailed;
    }
    ggml_backend_t backends[1] = {cpu};
    // The buffer types are passed explicitly. `nullptr` also works (ggml-backend.cpp:2548 falls back
    // to the default), but passing them states the intent instead of relying on it.
    ggml_backend_buffer_type_t bufts[1] = {ggml_backend_get_default_buffer_type(cpu)};
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends, bufts, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (!sched) {
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return kContextFailed;
    }

    const bool allocated = ggml_backend_sched_alloc_graph(sched, graph);

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
    return allocated ? kOk : kAllocFailed;
}

// Runs one variant in a child process and reports how it ended.
// A signal death is the GGML_ASSERT, and it is the EXPECTED result for variants 0 and 2. A test that
// ran them in-process would take the whole binary down with it.
enum class ending { ok, alloc_failed, aborted, other };

ending run_isolated(int variant, int * exit_code) {
    const pid_t pid = fork();
    if (pid == 0) {
        _exit(run_variant(variant));
    }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) {
        if (exit_code) *exit_code = 128 + WTERMSIG(status);
        return ending::aborted;
    }
    const int code = WEXITSTATUS(status);
    if (exit_code) *exit_code = code;
    if (code == kOk) return ending::ok;
    if (code == kAllocFailed) return ending::alloc_failed;
    return ending::other;
}

const char * name_of(int variant) {
    switch (variant) {
        case 0: return "no flags";
        case 1: return "k/v/q flagged";
        case 2: return "indices flagged only";
        case 3: return "all flagged";
        default: return "?";
    }
}

} // namespace

int main() {
    const struct { int variant; ending expected; } cases[] = {
        {0, ending::aborted},   // nothing flagged: the first view that needs a buffer aborts
        {1, ending::ok},        // k/v/q alone are sufficient
        {2, ending::aborted},   // the index flags are NOT what the allocator needs
        {3, ending::ok},        // and flagging everything obviously works
    };

    for (const auto & c : cases) {
        int code = 0;
        const ending actual = run_isolated(c.variant, &code);
        const char * verdict = actual == ending::ok            ? "alloc_graph ok"
                               : actual == ending::aborted     ? "ABORT"
                               : actual == ending::alloc_failed ? "alloc failed"
                                                                : "other";
        std::fprintf(stderr, "variant %d (%-20s) -> %s (exit %d)\n", c.variant,
                     name_of(c.variant), verdict, code);
        check(actual == c.expected,
              std::string("variant ") + name_of(c.variant) +
                  (c.expected == ending::ok
                       ? ": the scheduler places the graph"
                       : ": the scheduler aborts, which is the measured consequence of withholding "
                         "the flag"));
    }

    // The two positive variants together are what makes this an isolation rather than a demo: if
    // variant 2 also succeeded, the conclusion would be "any flag helps", which is false and would
    // have the next reader marking the wrong tensors.
    check(run_isolated(1, nullptr) == ending::ok && run_isolated(2, nullptr) == ending::aborted,
          "the requirement is on k/v/q and not on the index tensors: variant 1 succeeds while "
          "variant 2 aborts");

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "input flags: k/v/q must carry GGML_TENSOR_FLAG_INPUT for a real scheduler to "
                 "place a reader emission; the per-span index tensors must not be assumed to carry "
                 "it instead\n");
    return 0;
}
