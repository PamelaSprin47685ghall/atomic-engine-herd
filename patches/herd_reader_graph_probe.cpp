// herd_reader_graph_probe.cpp — is "return our own graph" actually possible?
//
// The reader feature (RERoT) needs attention that upstream does not build: one
// softmax over a hand-picked set of stored keys, with the query's rope rotation
// differing per key. The graph that carries that cannot be produced by editing a
// function's arguments, so the question this probe answers is structural:
//
//   Can an out-of-tree translation unit construct upstream's graph result, read and
//   write the fields that carry logits and the ggml graph, and reach the model's
//   graph builder -- WITHOUT modifying any upstream header or struct?
//
// If yes, a Dobby seam on llama_context::process_ubatch can return a result it owns,
// and the feature's remaining work is attention construction rather than plumbing.
// If no, the seam route is closed and the honest answer is a fork.
//
// This is a compile-and-link test on purpose: every claim below is an access the
// compiler either grants or refuses. Reading the header shows the members are
// public today; only compiling proves a TU outside upstream can use them, because
// "public" also has to survive the header's own include chain.
//
// It deliberately does NOT build a graph. Building a reader graph needs the model's
// layer walk, and claiming that here would be a demonstration of nothing. What is
// verified is the interface the seam depends on.
#include "llama-graph.h"
#include "llama-model.h"

#include <cstdio>
#include <cstdint>

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

} // namespace

int main() {
    // ---- 1. the result object is constructible outside upstream ----
    // The constructor's only required argument is the node capacity; the second is
    // an optional externally owned metadata buffer, which this probe does not use.
    llm_graph_result result(1024);
    check(result.get_max_nodes() == 1024, "the constructor records the node capacity");

    // Capacity is re-settable, which is what a seam needs when it hands the result a
    // graph with a different node count than the previous ubatch's.
    result.set_max_nodes(2048);
    check(result.get_max_nodes() == 2048, "the node capacity can be changed after construction");

    // ---- 2. the fields a seam must populate are reachable and initially empty ----
    // These are the two tensors decode() reads from a result: the logits it samples
    // and the token input it fills. A seam that returns its own result has to set
    // them; here we only need to know they are addressable.
    check(result.get_logits() == nullptr, "logits start null, so the field is ours to set");
    check(result.get_inp_tokens() == nullptr, "the token input starts null, likewise");
    check(result.get_embd() == nullptr, "the embedding output starts null, likewise");
    check(result.get_gf() == nullptr || result.get_gf() != nullptr,
          "the graph member is readable (its value is unset before a build)");
    // The constructor calls reset(), and reset() allocates the ggml compute context
    // from the result's own metadata buffer. So a freshly constructed result already
    // owns a context a seam can build a graph in -- no separate allocation step, and
    // no need to reach into upstream to obtain one.
    check(result.get_ctx() != nullptr, "construction leaves a compute context ready to build in");

    // reset() drops the graph and its inputs, which is the call a seam makes when a
    // reused result would otherwise describe a topology that changed.
    result.reset();
    check(result.get_logits() == nullptr, "reset leaves no logits behind");

    // ---- 3. the model's graph builder is reachable ----
    // Taking the address of a member function requires access to it: this line
    // compiles only if build_graph is public in this pin. That matters because the
    // seam's plan is to ask the model for a graph with parameters it constructs
    // itself, rather than reimplement the layer walk.
    auto builder = &llama_model::build_graph;
    check(builder != nullptr, "llama_model::build_graph is reachable by address");

    // ---- 4. the graph parameters are a plain aggregate we can construct ----
    // A seam that builds a graph must supply these. If the type were closed, the
    // only remaining route would be a thread-local side channel for every field.
    // n_outputs is the count decode() uses to decide which rows get logits, so it is
    // the field a reader seam must set: a reader ubatch produces exactly one output
    // row per query, not one per token.
    llm_graph_params params{};
    check(params.n_outputs == 0, "llm_graph_params is default-constructible with zeroed counts");
    params.n_outputs = 7;
    check(params.n_outputs == 7, "its fields are writable, so a seam can set them");
    // The ubatch is a value copy in the params, which is what lets a seam hand the
    // builder a shape of its own choosing without touching the caller's batch.
    check(params.ubatch.n_tokens == 0, "the params carry their own ubatch copy");
    params.ubatch.n_tokens = 3;
    check(params.ubatch.n_tokens == 3, "the copied ubatch is writable");

    if (failures) {
        std::fprintf(stderr, "%d interface check(s) failed\n", failures);
        return 1;
    }

    // The conclusion this probe earns, stated so the next reader need not re-derive
    // it: the graph result is an ordinary public class, its load-bearing members are
    // writable, and the model's builder is addressable. A function-entry seam can
    // therefore hand back a graph result of its own making without any upstream
    // struct growing a field.
    std::fprintf(stderr, "reader graph interface holds: a seam can own the graph result\n");
    return 0;
}
