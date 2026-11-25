// The emission through a context that allocates nothing, filled the way upstream fills its graph
// inputs -- and the numbers compared with an allocating run.
//
// The root cause this file was written against, established by reading and then by measurement:
//
//   graph_emit.cpp created `ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n)` and required
//   `idx->data != nullptr`, refusing otherwise. In a `no_alloc` context -- the regime llama.cpp's
//   own graph builds in -- `ggml_new_tensor_impl` allocates data only when `!ctx->no_alloc`
//   (ggml.c:1825), so `data` is null BY CONSTRUCTION. The refusal named ggml for what was the
//   caller's missing half of the graph-input contract.
//
// The contract now: the caller sets `collect_indices`, the emitter returns the index tensors (and
// their cells, owned by the result), and the caller writes them after the scheduler has placed the
// graph -- upstream's own `llm_graph_result::set_inputs` order (llama-graph.cpp:1590).
//
// Three things make this a measurement rather than a demonstration:
//
//   * the indices are written ONLY after allocation. If the emitter had used its own null-data
//     tensors, the gather would select whatever the allocator's buffer happened to contain;
//   * the fixture's cells are non-contiguous and the spans are of unequal length, so a gather that
//     used the wrong tensor, order, or span index cannot coincidentally match;
//   * the negative case asserts the REFUSAL, so treating a null-data context as "no indices needed"
//     fails here instead of producing plausible numbers.
//
// The OUTPUT is marked with `ggml_set_output`. That matters more than it looks: `ggml_gallocr` frees
// every node it allocated except the ones the graph declares as outputs, so an unmarked output can
// be recycled before the caller reads it -- which is a use-after-free that still returns the right
// numbers often enough to look like a pass.
#include <tq/reader/attention.h>
#include <tq/reader/graph_emit.h>

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

constexpr int kQHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 8;
constexpr int kCells = 12;
constexpr float kScale = 0.70710678f;

// Cells the emission will gather: not contiguous, unequal span lengths (4 and 3), so a stale or
// reused index tensor cannot produce the right answer by accident. Returned by value ON PURPOSE --
// every caller here passes it straight into the emitter, which is exactly the temporary-view case
// the owned `index_cells` buffer exists to survive.
tq::reader::query_view the_view() {
    tq::reader::query_view view;
    tq::reader::span_keys first;
    first.effective_pos = 5;
    first.cells = {0, 3, 7, 9};
    tq::reader::span_keys second;
    second.effective_pos = 6;
    second.cells = {11, 1, 4};
    view.spans = {first, second};
    return view;
}

struct values {
    std::vector<float> k;   // [head_dim, n_head_kv, n_cell]
    std::vector<float> v;
    std::vector<float> q;   // [head_dim, n_head]

    values() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCells, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        unsigned state = 0x5eed1234u;
        auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return float((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
    }
};

tq::reader::attention_result core_reference(const values & data) {
    tq::reader::kv_buffers kv;
    kv.k = data.k.data();
    kv.v = data.v.data();
    kv.n_cell = kCells;
    kv.n_head = kQHeads;
    kv.n_head_kv = kKvHeads;
    kv.head_dim = kHeadDim;
    const tq::reader::rope_fn no_rope = [](float *, std::size_t, std::int32_t) {};
    return tq::reader::attend(data.q.data(), kv, the_view(), kScale, no_rope);
}

void fill_tensor(ggml_tensor * tensor, const float * source, std::size_t count) {
    check(tensor != nullptr && tensor->data != nullptr, "the tensor has storage before it is filled");
    if (tensor && tensor->data) {
        std::memcpy(tensor->data, source, count * sizeof(float));
    }
}

// ---------------------------------------------------------------------------------------------
// Positive: no_alloc context, real scheduler, indices written after allocation.
// ---------------------------------------------------------------------------------------------
void no_alloc_emission_matches_allocating_emission() {
    const values data;
    const auto expected = core_reference(data);
    check(expected.ok, "the core produced a reference");
    if (!expected.ok) {
        return;
    }

    ggml_context * ctx = ggml_init({32u * 1024u * 1024u, nullptr, /*no_alloc=*/true});
    check(ctx != nullptr, "a no_alloc context was created");
    if (!ctx) {
        return;
    }
    check(ggml_get_no_alloc(ctx), "the context really is no_alloc");

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

    check(in.k && in.k->data == nullptr,
          "a cache tensor under no_alloc exists with no storage -- the condition that used to make "
          "the emission refuse");
    check(in.k->data == nullptr && in.v->data == nullptr && in.q->data == nullptr,
          "k/v/q also have no storage before allocation: they are inputs of the same kind, and the "
          "same ordering rule applies to them");

    // The emission is handed a TEMPORARY view. That is not incidental: it is the call shape that
    // made `index_entry::cells` a dangling pointer before the result owned the cells, and it is the
    // shape a caller writes without thinking.
    const auto emitted = tq::reader::emit_reader_attention(in, the_view(), kScale, nullptr);
    check(emitted.refusal == tq::reader::emit_refusal::ok,
          "the emission succeeds under no_alloc when the caller asks for the entries");
    if (emitted.refusal != tq::reader::emit_refusal::ok || !emitted.out) {
        std::fprintf(stderr, "  (refused: %s -- %s)\n",
                     tq::reader::describe(emitted.refusal), emitted.why.c_str());
        ggml_free(ctx);
        return;
    }

    check(emitted.indices.size() == 2, "two spans produced two entries");
    bool all_null = true;
    for (const auto & entry : emitted.indices) {
        all_null = all_null && entry.index != nullptr && entry.index->data == nullptr;
    }
    check(all_null,
          "every entry's tensor had no storage at emission time: the data is the scheduler's to "
          "provide, not the emitter's assumption");

    // THE DANGLING-POINTER CASE, asserted rather than reasoned about.
    //
    // The view above was a temporary and is gone. If the entries carried pointers into it, this read
    // would be a use-after-free; ASan catches it, and the values would be garbage regardless. They
    // are owned by the result, so they are still the view's cells.
    for (std::size_t i = 0; i < emitted.indices.size(); ++i) {
        const std::int32_t * cells = emitted.cells_of(emitted.indices[i]);
        check(cells != nullptr, "the entry's cells are addressable after the temporary view died");
        if (cells == nullptr) {
            continue;
        }
        const tq::reader::query_view fresh = the_view();
        check(std::equal(cells, cells + emitted.indices[i].n_cells,
                         fresh.spans[i].cells.begin()),
              "and they are that span's cells, in order -- not another span's, not freed memory");
        check(emitted.indices[i].span == i, "the entry records the span position it came from");
        check(emitted.indices[i].n_cells == std::size_t(emitted.indices[i].index->ne[0]),
              "the cell count equals the tensor's own extent");
    }

    // ---- the graph, placed by a real scheduler ----
    //
    // k/v/q carry GGML_TENSOR_FLAG_INPUT and the output carries GGML_TENSOR_FLAG_OUTPUT.
    //
    // The INPUT flags are measured, not assumed: tests/graph-emit-input-flag-test.cpp runs four
    // variants and finds that a scheduler aborts without them (they are view sources, not leafs, so
    // the leaf sweep never allocates them). The OUTPUT flag is the other half -- `ggml_gallocr` frees
    // what it allocated except declared outputs, so an unmarked output can be recycled before the
    // caller reads it, and the numbers would often still be right.
    in.k->flags |= GGML_TENSOR_FLAG_INPUT;
    in.v->flags |= GGML_TENSOR_FLAG_INPUT;
    in.q->flags |= GGML_TENSOR_FLAG_INPUT;
    emitted.out->flags |= GGML_TENSOR_FLAG_OUTPUT;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, emitted.out);

    ggml_backend_t cpu = ggml_backend_cpu_init();
    check(cpu != nullptr, "the CPU backend initialised");
    if (!cpu) {
        ggml_free(ctx);
        return;
    }
    ggml_backend_t backends[1] = {cpu};
    ggml_backend_buffer_type_t bufts[1] = {ggml_backend_get_default_buffer_type(cpu)};
    ggml_backend_sched_t sched =
        ggml_backend_sched_new(backends, bufts, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
    check(sched != nullptr, "a scheduler was created");
    if (!sched) {
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return;
    }

    const bool allocated = ggml_backend_sched_alloc_graph(sched, graph);
    check(allocated, "the scheduler allocated the graph");
    if (!allocated) {
        ggml_backend_sched_free(sched);
        ggml_backend_free(cpu);
        ggml_free(ctx);
        return;
    }

    // ---- now, and only now, the inputs become real ----
    fill_tensor(in.k, data.k.data(), data.k.size());
    fill_tensor(in.v, data.v.data(), data.v.size());
    fill_tensor(in.q, data.q.data(), data.q.size());
    bool index_written = true;
    for (const auto & entry : emitted.indices) {
        const std::int32_t * cells = emitted.cells_of(entry);
        check(entry.index->data != nullptr, "after allocation the index tensor has storage");
        if (entry.index->data == nullptr || cells == nullptr) {
            index_written = false;
            continue;
        }
        std::memcpy(entry.index->data, cells, entry.n_cells * sizeof(std::int32_t));
    }
    check(index_written, "every index tensor was written from the result's own cells");

    void * const k_data = in.k->data;
    void * const v_data = in.v->data;
    void * const q_data = in.q->data;
    std::vector<void *> index_data;
    for (const auto & entry : emitted.indices) index_data.push_back(entry.index->data);

    const ggml_status compute_status = ggml_backend_sched_graph_compute(sched, graph);
    check(compute_status == GGML_STATUS_SUCCESS, "the scheduler computed the graph");

    check(in.k->data == k_data && in.v->data == v_data && in.q->data == q_data,
          "no allocation happened between the fill and the compute: the addresses the values were "
          "written to are the addresses the kernels read");
    bool indices_stable = true;
    for (std::size_t i = 0; i < emitted.indices.size() && i < index_data.size(); ++i) {
        indices_stable = indices_stable && emitted.indices[i].index->data == index_data[i];
    }
    check(indices_stable, "and the index tensors did not move either");

    if (compute_status == GGML_STATUS_SUCCESS) {
        const float * got = static_cast<const float *>(emitted.out->data);
        check(got != nullptr, "the output tensor has storage after computation");
        if (got) {
            double worst = 0.0;
            for (std::size_t i = 0; i < expected.out.size(); ++i) {
                const double delta = std::fabs(double(got[i]) - double(expected.out[i]));
                if (delta > worst) worst = delta;
            }
            std::fprintf(stderr, "no_alloc via scheduler vs core: max|diff| = %.3e\n", worst);
            check(worst < 1e-5,
                  "the emission through a real scheduler allocation reproduces the core's answer: "
                  "the indices the CALLER wrote are the ones the gather used");
        }
    }

    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------------------------
// Negative: no storage and no request must be REFUSED, by name.
// ---------------------------------------------------------------------------------------------
void a_null_data_context_without_a_request_is_refused() {
    ggml_context * ctx = ggml_init({8u * 1024u * 1024u, nullptr, /*no_alloc=*/true});
    if (!ctx) {
        check(false, "a no_alloc context was created");
        return;
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
    // collect_indices stays false: nobody will fill the indices.

    const auto emitted = tq::reader::emit_reader_attention(in, the_view(), kScale, nullptr);
    check(emitted.refusal == tq::reader::emit_refusal::index_data_unavailable,
          "without a request and without storage, the emission is refused by name -- not built on "
          "uninitialised indices");
    check(emitted.out == nullptr, "a refusal produces no tensor");
    check(emitted.indices.empty(),
          "and carries no entries: a refusal must not hand back a set the caller could act on");
    check(emitted.why.find("collect_indices") != std::string::npos,
          "the reason names the flag to set, so a caller knows exactly what it owes");
    std::fprintf(stderr, "no_alloc without a request: %s\n",
                 tq::reader::describe(emitted.refusal));
    ggml_free(ctx);
}

// ---------------------------------------------------------------------------------------------
// An allocating context needs no request, and the entries are an observation when one is made.
// ---------------------------------------------------------------------------------------------
void an_allocating_context_writes_immediately() {
    const values data;
    ggml_context * ctx = ggml_init({32u * 1024u * 1024u, nullptr, /*no_alloc=*/false});
    if (!ctx) {
        check(false, "an allocating context was created");
        return;
    }

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    fill_tensor(in.k, data.k.data(), data.k.size());
    fill_tensor(in.v, data.v.data(), data.v.size());
    fill_tensor(in.q, data.q.data(), data.q.size());
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCells;
    in.collect_indices = true;

    const auto emitted = tq::reader::emit_reader_attention(in, the_view(), kScale, nullptr);
    check(emitted.refusal == tq::reader::emit_refusal::ok,
          "an allocating context needs no request and still succeeds");

    const tq::reader::query_view fresh = the_view();
    bool all_observed = emitted.indices.size() == fresh.spans.size();
    for (std::size_t i = 0; i < emitted.indices.size() && i < fresh.spans.size(); ++i) {
        const auto & entry = emitted.indices[i];
        // Under an allocating context the cells are already in the tensor, and the entry is an
        // observation of where they went -- so both the tensor's contents and the result's copy must
        // be the view's cells.
        const auto * written = static_cast<const std::int32_t *>(entry.index->data);
        const std::int32_t * owned = emitted.cells_of(entry);
        all_observed = all_observed && written != nullptr && owned != nullptr &&
                       std::equal(fresh.spans[i].cells.begin(), fresh.spans[i].cells.end(), written) &&
                       std::equal(fresh.spans[i].cells.begin(), fresh.spans[i].cells.end(), owned);
    }
    check(all_observed,
          "under an allocating context the tensors already carry the view's cells, and the result's "
          "owned copy agrees with them");
    ggml_free(ctx);
}

} // namespace

int main() {
    no_alloc_emission_matches_allocating_emission();
    a_null_data_context_without_a_request_is_refused();
    an_allocating_context_writes_immediately();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "graph inputs: the emission runs under no_alloc when the caller asks for the "
                 "entries and fills them after allocation, and refuses by name when nobody will\n");
    return 0;
}
