// The index hand-off through upstream's REAL graph result: `llm_graph_result::add_input` and
// `llm_graph_result::set_inputs`, not a stand-in.
//
// WHY EVERY EARLIER TEST WAS NOT THIS. `index-lifecycle-test` allocated a context it made itself;
// `host-lifecycle-test` drove ggml's scheduler but stood in for the graph-owned input with a stack
// object. Both were honest about it, and both left the same question open: does the publication work
// when the object living through it is the one upstream actually owns? That question is answerable
// without a model, because `llm_graph_result(int64_t max_nodes)` is PUBLIC and builds its own
// pieces -- measured here, not read:
//
//   llm_graph_result::reset()   ctx_compute = ggml_init({ mem_size = meta_size, no_alloc = true })
//                               gf          = ggml_new_graph_custom(ctx_compute, max_nodes, false)
//   add_input(unique_ptr)       inputs.emplace_back(std::move(input)); return inputs.back().get();
//   set_inputs(ubatch)          for (auto & input : inputs) input->set_input(ubatch);
//
// So the result IS the no_alloc context, IS the graph, and IS the owner of the input. Three things
// follow that no stand-in could establish:
//
//   1. The emitter runs inside a context whose LIFETIME upstream controls, and whose `no_alloc` flag
//      is upstream's decision rather than this test's convenience. Asserted via `ggml_get_no_alloc`.
//   2. The publication is invoked by `llm_graph_result::set_inputs` -- the same function upstream's
//      decode path calls -- rather than by a line in a test.
//   3. `reset()` destroys the context and clears `inputs`, so a result that was reset cannot inherit
//      a publish. That is the reuse boundary that IS testable here; `can_reuse(llm_graph_params)`
//      is not, because `llm_graph_params` describes a whole decode and cannot be fabricated honestly.
//
// ORDER, which is the whole point: build into the result's graph -> scheduler places it -> set_inputs
// publishes -> compute. Publishing before placement is REFUSED by name, and that refusal is observed
// through `set_inputs` too, not by calling the writer directly.
//
// WHAT THIS IS NOT: a seam, and not a replacement. It drives upstream's public classes in a test
// process. Nothing is installed; the reader lane changes no number in a decode.
#include <tq/reader/graph_emit_physical.h>
#include <tq/reader/upstream_view.h>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include "llama-graph.h"

#include <cmath>
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

constexpr int kHeadDim = 4;
constexpr int kKvHeads = 2;
constexpr int kQHeads  = 4;
constexpr int kCell    = 8;
constexpr float kScale = 0.5f;

struct fixture {
    std::vector<float> k, v, q;
    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        unsigned s = 0x1234abcdu;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
    }
    float K(int d, int kvh, int cell) const {
        return k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * cell)];
    }
    float V(int d, int kvh, int cell) const {
        return v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * cell)];
    }
    float Q(int d, int h) const { return q[std::size_t(d) + kHeadDim * h]; }
};

// Softmax over the named cells in view order, no mask. From the definition.
void reference(const fixture & f, const std::vector<int> & cells, std::vector<float> & out) {
    const int group = kQHeads / kKvHeads;
    out.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
    for (int h = 0; h < kQHeads; ++h) {
        const int kvh = h / group;
        std::vector<double> score(cells.size());
        double max_s = -INFINITY;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            double s = 0.0;
            for (int d = 0; d < kHeadDim; ++d)
                s += double(f.Q(d, h)) * double(f.K(d, kvh, cells[i]));
            score[i] = s * double(kScale);
            if (score[i] > max_s) max_s = score[i];
        }
        double sum = 0.0;
        for (double & s : score) { s = std::exp(s - max_s); sum += s; }
        for (int d = 0; d < kHeadDim; ++d) {
            double acc = 0.0;
            for (std::size_t i = 0; i < cells.size(); ++i)
                acc += (score[i] / sum) * double(f.V(d, kvh, cells[i]));
            out[std::size_t(d) + std::size_t(kHeadDim) * h] = float(acc);
        }
    }
}

std::vector<int> flatten(const tq::reader::query_view & view) {
    std::vector<int> cells;
    for (const auto & span : view.spans)
        for (std::int32_t c : span.cells) cells.push_back(int(c));
    return cells;
}

// The counters live OUTSIDE the input, and that is not tidiness.
//
// `llm_graph_result` owns its inputs as `unique_ptr`, so `res.reset()` DESTROYS them. An earlier
// version kept the counters as members and read `st.set_input_calls` after `res.reset()` -- a read
// of freed memory that happened to print 3 and passed. A test that dereferences a destroyed object
// proves nothing about the object; it proves something about the allocator's mood. Everything
// observable about the publisher is therefore this struct, which the case owns, and the input holds
// only a pointer to it.
//
// `destroyed` is the flag that makes the reset assertion mean something: it is set by the input's
// destructor, so "reset() destroyed the input" becomes an observed fact rather than an inference
// from `inputs.empty()`.
struct publisher_state {
    int set_input_calls = 0;
    int writes = 0;
    tq::reader::index_publish_refusal last = tq::reader::index_publish_refusal::ok;
    std::string why;
    bool wrote_once = false;
    bool destroyed = false;
};

// The input upstream owns, and the only place `publish_index_inputs` is called from. It records
// rather than aborts, because a test must be able to observe a refusal as a value: this is the same
// discipline `publish_index_inputs` itself uses (it returns a named refusal instead of dying).
struct publisher final : llm_graph_input_i {
    tq::reader::physical_emit_result result;
    publisher_state * state = nullptr;

    publisher(tq::reader::physical_emit_result r, publisher_state * s)
        : result(std::move(r)), state(s) {}

    ~publisher() override { state->destroyed = true; }

    void set_input(const llama_ubatch *) override {
        ++state->set_input_calls;
        // A second write would race a compute. Guarded here AND asserted from outside, because the
        // guard is what a real publisher needs and the assertion is what proves the guard is reached.
        if (state->wrote_once) return;
        state->last = tq::reader::publish_index_inputs(
            result,
            +[](ggml_tensor * t, const void * d, std::size_t o, std::size_t s) -> bool {
                if (t == nullptr || t->buffer == nullptr) return false;
                ggml_backend_tensor_set(t, d, o, s);
                return true;
            },
            &state->why, nullptr);
        if (state->last == tq::reader::index_publish_refusal::ok) {
            state->wrote_once = true;
            ++state->writes;
        }
    }
};

void run_case(const char * name, const tq::reader::query_view & view) {
    const fixture f;

    // THE STATE IS DECLARED BEFORE THE RESULT, AND THE ORDER IS THE WHOLE POINT.
    //
    // Destructors run in reverse declaration order. `llm_graph_result` owns its inputs as
    // `unique_ptr`, and the publisher's destructor writes `state->destroyed = true`. If `st` were
    // declared after `res` -- as it was -- then at scope exit `st` would be destroyed FIRST and the
    // result's destructor would then write into freed memory. That is undefined behaviour on EVERY
    // exit path of this function, not just the reset one: an early `return` after a failed check
    // unwinds the same way.
    //
    // The previous revision fixed the reset-time read of a destroyed input and introduced this,
    // which is why it is written down here rather than left to the reader: "the state outlives the
    // input" is a claim about declaration order, and nothing in the type system enforces it.
    publisher_state st;

    // The REAL object. No llama_context, no model, no memory context: `max_nodes` is the only
    // argument, and everything the publication needs is built by it.
    llm_graph_result res(4096);

    ggml_context * ctx = res.get_ctx();
    if (ctx == nullptr) { check(false, std::string(name) + ": result has no context"); return; }

    // (1) UPSTREAM'S CONTEXT SHAPE, measured rather than assumed. The emitter admits no_alloc
    //     contexts only; if upstream ever changed that, this whole approach would need revisiting and
    //     the failure would be here instead of at a mysterious refusal three layers down.
    check(ggml_get_no_alloc(ctx), std::string(name) + ": upstream's graph-result context is no_alloc");
    check(res.get_gf() != nullptr, std::string(name) + ": the result owns a graph");

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCell;
    in.collect_indices = true;
    ggml_set_input(in.k);
    ggml_set_input(in.v);
    ggml_set_input(in.q);

    const auto em = tq::reader::emit_physical_attention(in, view, kScale);
    if (em.refusal != tq::reader::physical_emit_refusal::ok) {
        check(false, std::string(name) + ": emit refused: " + tq::reader::describe(em.refusal) +
                         " -- " + em.why);
        return;
    }
    // Every later line indexes `em.indices[s]` for s < spans.size(), so the count is checked first:
    // a short list would be an out-of-bounds read inside a test that had already failed.
    if (em.indices.size() != view.spans.size() + 1 || em.out == nullptr || em.weights == nullptr) {
        check(false, std::string(name) + ": emission shape is not spans+1 entries with non-null outputs");
        return;
    }

    // A tensor nothing consumes. Under the scheduler it must stay unplaced -- the negative case that
    // makes "the index tensors were placed" mean something.
    ggml_tensor * orphan = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_input(orphan);

    // (2) REGISTRATION through upstream's own method, which takes ownership.
    //
    // `st` was declared above `res` so it outlives the input on every exit path (see the comment
    // there). `pub` is kept only for the one-time check that `add_input` returned the object it
    // stored; after this point nothing dereferences it, because `res.reset()` destroys it.
    {
        auto owned = std::make_unique<publisher>(em, &st);
        publisher * pub = owned.get();
        llm_graph_input_i * stored = res.add_input(std::move(owned));
        check(stored == pub,
              std::string(name) + ": add_input stored the input and returned its address");
        check(res.inputs.size() == 1, std::string(name) + ": the result now owns exactly one input");
    }

    // (3) BUILD into the result's graph, then let the scheduler place it.
    ggml_build_forward_expand(res.get_gf(), em.out);

    ggml_backend_t be = ggml_backend_cpu_init();
    if (be == nullptr) { check(false, std::string(name) + ": cpu backend"); return; }
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(&be, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    if (sched == nullptr) { check(false, std::string(name) + ": sched_new"); ggml_backend_free(be); return; }

    // (4) BEFORE PLACEMENT, through `set_inputs` -- the real call path. The refusal must be named and
    //     nothing may be written; this is the case that would have written through null storage.
    res.set_inputs(nullptr);
    check(st.set_input_calls == 1, std::string(name) + ": set_inputs reached the input once");
    check(st.last == tq::reader::index_publish_refusal::no_storage,
          std::string(name) + ": publishing before placement is `no_storage`, got " +
              tq::reader::describe(st.last) + " (" + st.why + ")");
    check(st.writes == 0, std::string(name) + ": nothing was written before placement");

    if (!ggml_backend_sched_alloc_graph(sched, res.get_gf())) {
        check(false, std::string(name) + ": sched_alloc_graph failed");
        ggml_backend_sched_free(sched);
        ggml_backend_free(be);
        return;
    }

    check(orphan->buffer == nullptr,
          std::string(name) + ": an unreferenced tensor stays UNPLACED (buffer is not null, so the "
                               "scheduler placed orphans too and the positive case proves nothing)");
    for (const auto & e : em.indices) {
        check(e.index != nullptr && e.index->buffer != nullptr,
              std::string(name) + ": every index tensor IS placed, because a node consumes it");
    }

    // (5) THE PUBLICATION, by upstream's `set_inputs` and nowhere else.
    res.set_inputs(nullptr);
    check(st.last == tq::reader::index_publish_refusal::ok,
          std::string(name) + ": publishing after placement succeeds through set_inputs: " +
              tq::reader::describe(st.last) + " (" + st.why + ")");
    check(st.writes == 1, std::string(name) + ": exactly one write");
    if (!st.wrote_once) {
        // HARD STOP: with no published indices the graph would gather from uninitialised host
        // memory, so the compute below would read past k/v and could crash instead of reporting.
        check(false, std::string(name) + ": not computing on an unpublished row table");
        ggml_backend_sched_free(sched);
        ggml_backend_free(be);
        return;
    }

    // A SECOND `set_inputs` must not re-publish. This is the only reuse boundary that is testable
    // here: `can_reuse(llm_graph_params)` needs a whole decode's parameters, which cannot be
    // fabricated honestly, so the observable rule -- one publication per graph, enforced by the
    // input -- is what gets asserted.
    res.set_inputs(nullptr);
    check(st.set_input_calls == 3, std::string(name) + ": a second set_inputs still reaches it");
    check(st.writes == 1, std::string(name) + ": but it does not write twice (" +
                                std::to_string(st.writes) + " writes)");

    // (6) The cells, read back out of the TENSORS -- what a wrong publish would move.
    {
        const std::vector<int> cells = flatten(view);
        std::vector<std::int32_t> seen;
        for (std::size_t s = 0; s < view.spans.size(); ++s) {
            const auto & e = em.indices[s];
            const std::size_t before = seen.size();
            seen.resize(before + std::size_t(e.n_cells));
            if (e.index->data != nullptr) {
                std::memcpy(seen.data() + before, e.index->data,
                            std::size_t(e.n_cells) * sizeof(std::int32_t));
            } else {
                check(false, std::string(name) + ": gather " + std::to_string(s) +
                                 " has no data after a successful publish");
                seen.resize(before);
            }
        }
        std::vector<std::int32_t> want(cells.begin(), cells.end());
        check(seen == want, std::string(name) + ": published gathers hold the view's cells in order");
    }

    // (7) COMPUTE and compare with the definition. A correct publish plus a correct graph is the only
    //     combination that produces these numbers.
    ggml_backend_tensor_set(in.k, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(in.v, f.v.data(), 0, f.v.size() * sizeof(float));
    ggml_backend_tensor_set(in.q, f.q.data(), 0, f.q.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, res.get_gf()) != GGML_STATUS_SUCCESS) {
        check(false, std::string(name) + ": graph_compute failed after a successful publish");
    } else {
        std::vector<float> got(std::size_t(kHeadDim) * kQHeads, 0.0f);
        std::memcpy(got.data(), em.out->data, got.size() * sizeof(float));
        std::vector<float> want;
        reference(f, flatten(view), want);
        double worst = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) { finite = false; break; }
            worst = std::max(worst, std::fabs(double(got[i]) - double(want[i])));
        }
        check(finite, std::string(name) + ": output is finite");
        check(worst < 1e-6, std::string(name) + ": matches the definition, max|diff| = " +
                                std::to_string(worst));
        double magnitude = 0.0;
        for (float x : want) magnitude = std::max(magnitude, std::fabs(double(x)));
        check(magnitude > 1e-3, std::string(name) + ": reference is non-degenerate (" +
                                   std::to_string(magnitude) + ")");
    }

    // (8) THE RESET BOUNDARY. `reset()` destroys the context and clears `inputs`, so a result that
    //     was reset can neither reach nor write the old tensors -- the new context's tensors are
    //     different objects.
    //
    //     The scheduler is freed FIRST, and the order is load-bearing: it was initialised against
    //     tensors living in the context `reset()` is about to destroy. Freeing it afterwards would
    //     leave it pointing into memory that no longer exists.
    ggml_backend_sched_free(sched);
    ggml_backend_free(be);

    const int calls_before_reset = st.set_input_calls;
    res.reset();
    check(st.destroyed,
          std::string(name) + ": reset() DESTROYED the input (its destructor ran), not merely "
                               "detached it");
    check(res.inputs.empty(), std::string(name) + ": reset() clears the owned inputs");
    check(res.get_ctx() != nullptr && ggml_get_no_alloc(res.get_ctx()),
          std::string(name) + ": reset() leaves a fresh no_alloc context");
    res.set_inputs(nullptr);  // must be a no-op now: nothing is registered
    check(st.set_input_calls == calls_before_reset,
          std::string(name) + ": after reset, set_inputs reaches no input (the old one is gone)");

    std::fprintf(stderr, "%s: %zu entries, published via llm_graph_result::set_inputs, computed\n",
                 name, em.indices.size());
}

}  // namespace

int main() {
    run_case("full coverage", tq::reader::full_coverage_view(kCell));

    tq::reader::query_view sparse;
    tq::reader::span_keys a;
    a.effective_pos = 5;
    a.cells = {6, 2};
    tq::reader::span_keys b;
    b.effective_pos = 5;
    b.cells = {4};
    sparse.spans.push_back(a);
    sparse.spans.push_back(b);
    run_case("two spans", sparse);

    if (failures == 0) {
        std::fprintf(stderr, "graph-result publish: OK\n");
        return 0;
    }
    std::fprintf(stderr, "graph-result publish: %d failure(s)\n", failures);
    return 1;
}
