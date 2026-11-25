// The emitter's index hand-off, walked through the HOST lifecycle rather than a test-only one.
//
// WHY THIS EARLIER TEST WAS NOT ENOUGH. `index-lifecycle-test` allocated every tensor of the
// context with `ggml_backend_alloc_ctx_tensors`, then published, then computed. That is a legal
// sequence, and it proved the writer works -- but it is NOT the lifecycle `build_attn_mha` lives in.
// Upstream allocates through a scheduler, and a scheduler is selective: `ggml_backend_sched_alloc_graph`
// places the tensors that are REACHABLE FROM THE GRAPH and nothing else. A tensor that no node
// consumes keeps `buffer == nullptr` no matter how late the publish runs. That difference is not
// academic -- it is exactly what made the deleted publisher abort on its first layer:
//
//   REFUSED: an entry's tensor has no data: publish ran before
//   ggml_backend_sched_alloc_graph() placed the graph ... (no storage yet)
//
// and it is a difference `alloc_ctx_tensors` cannot express, because that function allocates the
// whole context eagerly and so places orphans too.
//
// So this file does three things the earlier one could not:
//
//   1. It allocates with `ggml_backend_sched_alloc_graph`, the host mechanism.
//   2. It proves that an UNREFERENCED tensor stays unplaced under that mechanism (the negative case
//      that makes the positive one meaningful).
//   3. It publishes through a graph-owned input object -- the shape upstream's `set_inputs` drives,
//      with `add_input`/`set_input` standing in for `llm_graph_result` here because this test links
//      no llama. The ORDER is the same: alloc_graph, then set_input, then compute.
//
// WHAT THIS IS NOT: a seam, and not a replacement. It drives ggml's public scheduler in a test
// process. Nothing here is installed, and the reader lane still changes no number in a decode.
#include <tq/reader/graph_emit_physical.h>
#include <tq/reader/upstream_view.h>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

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
        unsigned s = 0x9e3779b9u;
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

// Softmax over the named cells, in view order, no mask. Written from the definition.
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

// The graph-owned input upstream's `set_inputs` drives. Stands in for `llm_graph_input_i` because
// this test links no llama; the SHAPE is what matters -- constructed at build time, invoked after
// placement, owns the emission it will write.
struct published_once {
    tq::reader::physical_emit_result result;
    int calls = 0;
    bool ok = false;
    std::string why;

    void set_input() {
        ++calls;
        if (calls > 1) return;  // a second write would race a compute; refused by construction
        const auto r = tq::reader::publish_index_inputs(
            result,
            +[](ggml_tensor * t, const void * d, std::size_t o, std::size_t s) -> bool {
                // Through the backend: a non-null `data` says "placed", not "host memory".
                if (t == nullptr || t->buffer == nullptr) return false;
                ggml_backend_tensor_set(t, d, o, s);
                return true;
            },
            &why, nullptr);
        ok = (r == tq::reader::index_publish_refusal::ok);
        if (!ok) why = std::string(tq::reader::describe(r)) + " -- " + why;
    }
};

void host_lifecycle_places_publishes_and_computes(const char * name,
                                                  const tq::reader::query_view & view) {
    const fixture f;
    ggml_init_params ip{};
    ip.mem_size = 8u * 1024u * 1024u;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { check(false, std::string(name) + ": ggml_init"); return; }

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
        ggml_free(ctx);
        return;
    }
    // Guarded before any `back()` or `indices[s]`: an empty entry list would make the assertions
    // below read past the end, which is the difference between a red test and undefined behaviour.
    //
    // The COUNT is checked here, not later, for the same reason. The emitter's contract is
    // `spans.size() + 1` (one gather per span, plus the scatter); the loop further down indexes
    // `indices[s]` for `s < spans.size()`, so a short list would be an out-of-bounds read inside a
    // test that had already reported a failure -- it would crash instead of reporting, and a crash
    // in a red test hides the failure it was about to describe.
    if (em.indices.empty() || em.indices.size() != view.spans.size() + 1) {
        check(false, std::string(name) + ": " + std::to_string(em.indices.size()) +
                         " index entries for " + std::to_string(view.spans.size()) +
                         " span(s), expected spans + 1");
        ggml_free(ctx);
        return;
    }
    // `out` and `weights` are what the graph is built from and what is read back. Null here means
    // the emission is unusable, and every later line would dereference it.
    if (em.out == nullptr || em.weights == nullptr) {
        check(false, std::string(name) + ": emission has a null out or weights tensor");
        ggml_free(ctx);
        return;
    }

    // A tensor NOTHING consumes, created in the same context. Under a scheduler it must stay
    // unplaced -- that is the negative case which makes every later "it was placed" meaningful.
    ggml_tensor * orphan = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
    ggml_set_input(orphan);

    ggml_backend_t be = ggml_backend_cpu_init();
    if (be == nullptr) { check(false, std::string(name) + ": cpu backend"); ggml_free(ctx); return; }
    ggml_backend_sched_t sched = ggml_backend_sched_new(&be, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE,
                                                        false, false);
    if (sched == nullptr) {
        check(false, std::string(name) + ": sched_new");
        ggml_backend_free(be);
        ggml_free(ctx);
        return;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, em.out);

    // (1) BEFORE PLACEMENT: publishing must be refused by name, and the writer must not run at all.
    published_once pub{em};
    {
        const auto r = tq::reader::publish_index_inputs(
            em, +[](ggml_tensor *, const void *, std::size_t, std::size_t) { return true; },
            &pub.why, nullptr);
        check(r == tq::reader::index_publish_refusal::no_storage,
              std::string(name) + ": publish before alloc_graph is `no_storage`, got " +
                  tq::reader::describe(r));
    }

    // (2) HOST PLACEMENT. Selective by construction.
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        check(false, std::string(name) + ": ggml_backend_sched_alloc_graph failed");
        ggml_backend_sched_free(sched);
        ggml_backend_free(be);
        ggml_free(ctx);
        return;
    }

    // The negative case, asserted rather than assumed: the scheduler placed the graph, not the
    // context, so a tensor no node consumes has no buffer. If this ever passes, the scheduler became
    // eager and the rest of this test stops proving what it claims.
    check(orphan->buffer == nullptr,
          std::string(name) + ": an unreferenced tensor stays UNPLACED under the scheduler "
                               "(buffer is not null, so the scheduler placed orphans too)");
    for (const auto & e : em.indices) {
        check(e.index != nullptr && e.index->buffer != nullptr,
              std::string(name) + ": every index tensor IS placed, because a node consumes it");
    }

    // (3) THE GRAPH-OWNED INPUT runs in the host's window: after placement, before compute.
    pub.set_input();
    check(pub.ok, std::string(name) + ": publish after alloc_graph: " + pub.why);
    if (!pub.ok) {
        // HARD STOP. The gather indices were not written, so the graph below would gather with
        // whatever the placement left in those buffers -- on the CPU that is uninitialised host
        // memory, so the compute would read past k/v and could crash instead of reporting. A test
        // that continues after the precondition it needs has failed is not measuring the thing it
        // names, and the crash would replace the diagnosis with a signal.
        check(false, std::string(name) + ": not computing on an unpublished row table");
        ggml_backend_sched_free(sched);
        ggml_backend_free(be);
        ggml_free(ctx);
        return;
    }

    // The cells, read back out of the TENSORS -- what a wrong publish would move.
    {
        const std::vector<int> cells = flatten(view);
        std::size_t total = 0;
        for (const auto & e : em.indices) total += std::size_t(e.n_cells);
        check(total == cells.size() * 2, std::string(name) + ": gathers plus scatter cover 2N");
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
                                 " still has no data after a successful publish");
                seen.resize(before);
            }
        }
        std::vector<std::int32_t> want(cells.begin(), cells.end());
        check(seen == want, std::string(name) + ": published gathers hold the view's cells in order");
    }

    // (4) COMPUTE, and compare with the definition. A correct publish plus a correct graph is the
    //     only combination that produces these numbers.
    ggml_backend_tensor_set(in.k, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(in.v, f.v.data(), 0, f.v.size() * sizeof(float));
    ggml_backend_tensor_set(in.q, f.q.data(), 0, f.q.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
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

    // (5) THE REUSE CLAIM IS GONE, AND THIS IS WHY.
    //
    // What used to be here: `ggml_backend_sched_reset`, a second `ggml_cgraph` over the same tensors,
    // a fresh local `published_once` that published the same cells again, and an assertion that the
    // second publish succeeded. It was described as proving that "a graph carrying a publisher cannot
    // be reused without republishing". It proved nothing of the kind, for three separate reasons:
    //
    //   * `published_once` was a STACK OBJECT created by the test, not an input owned by a graph
    //     result. It never modelled the lifetime it claimed to model, so "the second graph needs its
    //     own publish" restated the test's own call and not a property of the host.
    //   * The second publish wrote the SAME cells into the SAME tensors. That is an idempotent
    //     rewrite, and success is guaranteed once the first publish succeeded -- an assertion that
    //     cannot fail is not evidence.
    //   * `sched_reset` plus a new graph is not reuse. Upstream's reuse is `llm_graph_input_i::
    //     can_reuse` deciding that a graph from a previous decode may be submitted again; ggml's
    //     scheduler has no such notion, so the question cannot even be posed at this layer.
    //
    // Reuse therefore remains UNTESTED, and the honest statement of the requirement (a graph carrying
    // an index publisher must refuse reuse) lives in HANDOFF §7.13/§7.15 as a semantic argument, not
    // as a measurement. Deleting the block is the point: a false green here would be cited later.

    std::fprintf(stderr, "%s: %zu entries, placed by the scheduler, published, computed\n", name,
                 em.indices.size());

    ggml_backend_sched_free(sched);
    ggml_backend_free(be);
    ggml_free(ctx);
}

}  // namespace

int main() {
    host_lifecycle_places_publishes_and_computes("full coverage",
                                                 tq::reader::full_coverage_view(kCell));

    tq::reader::query_view sparse;
    tq::reader::span_keys a;
    a.effective_pos = 5;
    a.cells = {6, 2};
    tq::reader::span_keys b;
    b.effective_pos = 5;
    b.cells = {4};
    sparse.spans.push_back(a);
    sparse.spans.push_back(b);
    host_lifecycle_places_publishes_and_computes("two spans", sparse);

    if (failures == 0) {
        std::fprintf(stderr, "host lifecycle: OK\n");
        return 0;
    }
    std::fprintf(stderr, "host lifecycle: %d failure(s)\n", failures);
    return 1;
}
