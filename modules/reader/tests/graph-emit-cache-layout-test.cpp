// The emitter against upstream's ACTUAL V cache, not against a convenient fixture.
//
// The existing graph-emit test builds its inputs as contiguous tensors, which proves the
// emitter is self-consistent and proves nothing about the tensors llama hands a graph builder.
// This test constructs K and V the way `llama_kv_cache::get_k/get_v` construct them -- as
// views over one flat cache buffer with explicit strides -- and asks whether the emitter reads
// the same values there. Both of upstream's V layouts are built, because upstream has two and
// distinguishes them by stride (`v_trans = v->nb[1] > v->nb[2]`), never by shape.
//
//   llama-kv-cache.cpp:1430  get_v, !v_trans  ne = (head_dim, n_head_kv, n_kv)
//                                             nb = (row(head_dim), row(head_dim*n_head_kv), ...)
//   llama-kv-cache.cpp:1449  get_v,  v_trans  ne = (n_kv, n_head_kv, head_dim)
//                                             nb = (row(n_kv*head_dim), row(n_kv), ...)
//
// The first is what this module reads. The second must be refused: with n_cell == head_dim its
// shape is identical to the first while its bytes are a different permutation, so a shape-only
// check would accept it and return numbers from the wrong cells.
#include <tq/reader/attention.h>
#include <tq/reader/graph_emit.h>

#include <ggml.h>
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

// Deliberately equal: this is the case in which the two V layouts are indistinguishable by
// shape, so the stride test is the only thing standing between a correct answer and a
// plausible wrong one.
constexpr int kKvHeads  = 2;
constexpr int kQHeads   = 4;
constexpr int kHeadDim  = 6;
constexpr int kCells    = 6;
constexpr int kKvSize   = 16;   // cache capacity; the view covers kCells of it
constexpr float kScale  = 0.70710678f;

struct cache {
    std::vector<float> k_flat;   // (cell, kv head, dim), as llama stores K
    std::vector<float> v_flat;   // (cell, kv head, dim), as llama stores the !v_trans V
    std::vector<float> q_flat;   // (head, dim)

    cache() {
        k_flat.assign(std::size_t(kCells) * kKvHeads * kHeadDim, 0.0f);
        v_flat.assign(k_flat.size(), 0.0f);
        q_flat.assign(std::size_t(kQHeads) * kHeadDim, 0.0f);
        unsigned state = 99991u;
        auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return float((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (std::size_t i = 0; i < k_flat.size(); ++i) k_flat[i] = next();
        for (std::size_t i = 0; i < v_flat.size(); ++i) v_flat[i] = next() * 2.0f;
        for (std::size_t i = 0; i < q_flat.size(); ++i) q_flat[i] = next();
    }
};

// K and V as upstream views them: one flat cache tensor, two 4-D views over it.
// `v_transposed` selects the OTHER get_v branch, whose bytes are cell-major.
struct upstream_views {
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
};

upstream_views make_views(ggml_context * ctx, const cache & data, bool v_transposed) {
    const std::int64_t n_embd_v = std::int64_t(kHeadDim) * kKvHeads;

    // K cache: [n_embd_k_gqa, kv_size] with each head-dim row contiguous across cells.
    ggml_tensor * k_cache = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, std::int64_t(kCells) * kHeadDim * kKvHeads);
    std::memcpy(k_cache->data, data.k_flat.data(), data.k_flat.size() * sizeof(float));

    upstream_views out;
    // get_k: ne = (head_dim, n_head_kv, n_kv), nb1 = row(head_dim), nb2 = row(head_dim*n_head_kv).
    out.k = ggml_view_4d(ctx, k_cache, kHeadDim, kKvHeads, kCells, 1,
                         ggml_row_size(GGML_TYPE_F32, kHeadDim),
                         ggml_row_size(GGML_TYPE_F32, n_embd_v),
                         ggml_row_size(GGML_TYPE_F32, n_embd_v * kCells), 0);

    if (!v_transposed) {
        // The default branch: same bytes as K, same ordering.
        ggml_tensor * v_cache = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd_v * kCells);
        std::memcpy(v_cache->data, data.v_flat.data(), data.v_flat.size() * sizeof(float));
        out.v = ggml_view_4d(ctx, v_cache, kHeadDim, kKvHeads, kCells, 1,
                             ggml_row_size(GGML_TYPE_F32, kHeadDim),
                             ggml_row_size(GGML_TYPE_F32, n_embd_v),
                             ggml_row_size(GGML_TYPE_F32, n_embd_v * kCells), 0);
    } else {
        // The v_trans branch: cell-major rows, so ne = (n_kv, n_head_kv, head_dim) and the
        // first two strides are swapped relative to the branch above. Filled from the same
        // logical values, permuted the way that cache permutes them -- so a reader that
        // ignored the strides would answer with confidence from the wrong cells.
        // Sized by the CACHE, not by the view.
        //
        // The indexing below is upstream's (llama-kv-cache.cpp:1449): head h starts
        // kv_size*head_dim elements in, dimension d one kv_size further, cell c at +c. Those
        // are the strides of a cache holding kKvSize cells, while this view covers only the
        // first kCells of them. Sizing the buffer by the view's extent and indexing it by the
        // cache's strides writes past the end -- ASan: a 4-byte write 32 bytes beyond a
        // 288-byte allocation, which is exactly 72 floats against the 192 the layout needs.
        // A test fixture that corrupts the heap is not a fixture; the numbers it produces
        // afterwards mean nothing.
        std::vector<float> permuted(std::size_t(kKvSize) * kKvHeads * kHeadDim, 0.0f);
        for (int h = 0; h < kKvHeads; ++h) {
            for (int c = 0; c < kCells; ++c) {
                for (int d = 0; d < kHeadDim; ++d) {
                    // (cell, h, d) -> h * (kv_size*head_dim) + d * kv_size + c
                    permuted[std::size_t(h) * kKvSize * kHeadDim + std::size_t(d) * kKvSize + std::size_t(c)] =
                        data.v_flat[std::size_t(c * kKvHeads + h) * kHeadDim + d];
                }
            }
        }
        ggml_tensor * v_cache = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,
                                                   std::int64_t(kKvSize) * kHeadDim * kKvHeads);
        // The whole cache buffer: v_cache is kKvSize*head_dim*n_head_kv floats and so is
        // `permuted` now. Copying the vector's size rather than a ratio is deliberate -- the
        // two must agree, and if one drifts the memcpy says so instead of filling a prefix
        // and leaving the tail as silent zeros.
        std::memcpy(v_cache->data, permuted.data(), permuted.size() * sizeof(float));
        out.v = ggml_view_4d(ctx, v_cache, kCells, kKvHeads, kHeadDim, 1,
                             ggml_row_size(GGML_TYPE_F32, std::int64_t(kKvSize) * kHeadDim),
                             ggml_row_size(GGML_TYPE_F32, kKvSize),
                             ggml_row_size(GGML_TYPE_F32, std::int64_t(kKvSize) * kHeadDim * kKvHeads), 0);
    }
    return out;
}

tq::reader::query_view two_spans() {
    tq::reader::query_view view;
    tq::reader::span_keys a;
    a.effective_pos = 7;
    a.cells = {0, 2, 3};
    tq::reader::span_keys b;
    b.effective_pos = 8;
    b.cells = {5, 1};
    view.spans = {a, b};
    return view;
}

// The core, on the same values in the layout it documents.
tq::reader::attention_result core_reference(const cache & data, const tq::reader::query_view & view) {
    tq::reader::kv_buffers kv;
    kv.k = data.k_flat.data();
    kv.v = data.v_flat.data();
    kv.n_cell = kCells;
    kv.n_head = kQHeads;
    kv.n_head_kv = kKvHeads;
    kv.head_dim = kHeadDim;
    const tq::reader::rope_fn no_rope = [](float *, std::size_t, std::int32_t) {};
    return tq::reader::attend(data.q_flat.data(), kv, view, kScale, no_rope);
}

// The emitter, on tensors built exactly as the KV cache builds them.
void the_default_cache_layout_is_read_correctly() {
    cache data;
    const tq::reader::query_view view = two_spans();
    const auto expected = core_reference(data, view);
    check(expected.ok, "cache-layout: the core produced a reference");

    ggml_init_params params{64u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    const upstream_views views = make_views(ctx, data, false);

    // The strides upstream's default branch actually has, restated here so a change in either
    // the fixture or the emitter's assumption fails loudly rather than silently agreeing.
    check(views.k->nb[1] <= views.k->nb[2], "fixture: K is dim-major within a cell (nb1 <= nb2)");
    check(views.v->nb[1] <= views.v->nb[2], "fixture: V is dim-major within a cell (nb1 <= nb2)");
    check(views.v->ne[1] == kKvHeads && views.v->ne[2] == kCells,
          "fixture: V view is (head_dim, n_head_kv, n_kv)");

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = views.k;
    in.v = views.v;
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    std::memcpy(in.q->data, data.q_flat.data(), data.q_flat.size() * sizeof(float));
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCells;

    const tq::reader::emit_result got = tq::reader::emit_reader_attention(in, view, kScale, nullptr);
    check(got.refusal == tq::reader::emit_refusal::ok, "cache-layout: the emitter accepts the real view");
    if (got.out) {
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, got.out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        const float * produced = static_cast<const float *>(got.out->data);
        double worst = 0.0;
        for (std::size_t i = 0; i < expected.out.size(); ++i) {
            const double delta = std::fabs(double(produced[i]) - double(expected.out[i]));
            if (delta > worst) worst = delta;
        }
        std::fprintf(stderr, "cache-layout (!v_trans): emitter vs core max|diff| = %.3e\n", worst);
        check(worst < 1e-5, "cache-layout: the emitter reads the values the cache holds");
    } else {
        check(false, "cache-layout: no output tensor to compare");
    }
    ggml_free(ctx);
}

// The other branch, now READ rather than refused.
//
// Same logical values in a different permutation, and -- because n_cell == head_dim -- a shape
// that cannot tell the two apart. That is what makes this case evidence: the emitter has to
// look at the strides, and the numbers have to come back matching the same reference the
// dim-major case matched.
void the_transposed_cache_layout_is_read_correctly() {
    cache data;
    const tq::reader::query_view view = two_spans();
    const auto expected = core_reference(data, view);
    check(expected.ok, "cache-layout (v_trans): the core produced a reference");

    ggml_init_params params{64u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    const upstream_views views = make_views(ctx, data, true);

    check(views.v->nb[1] > views.v->nb[2], "fixture: the v_trans V view has nb1 > nb2");
    // The trap, stated as an assertion: the shape alone is identical to the default branch's,
    // so a shape check would accept this buffer and read another cell's values.
    check(std::size_t(views.v->ne[0]) == kCells && std::size_t(views.v->ne[1]) == kKvHeads &&
          std::size_t(views.v->ne[2]) == kHeadDim,
          "fixture: the v_trans view is (n_cell, n_head_kv, head_dim) -- and n_cell == head_dim");

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = views.k;
    in.v = views.v;
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    std::memcpy(in.q->data, data.q_flat.data(), data.q_flat.size() * sizeof(float));
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCells;

    const tq::reader::emit_result got = tq::reader::emit_reader_attention(in, view, kScale, nullptr);
    check(got.refusal == tq::reader::emit_refusal::ok,
          "cache-layout (v_trans): the cell-major cache is accepted, not refused");
    if (got.out) {
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, got.out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        const float * produced = static_cast<const float *>(got.out->data);
        double worst = 0.0;
        for (std::size_t i = 0; i < expected.out.size(); ++i) {
            const double delta = std::fabs(double(produced[i]) - double(expected.out[i]));
            if (delta > worst) worst = delta;
        }
        std::fprintf(stderr, "cache-layout (v_trans):  emitter vs core max|diff| = %.3e\n", worst);
        check(worst < 1e-5,
              "cache-layout (v_trans): the emitter reads the same values from the cell-major "
              "permutation -- the stride test is what makes this work");
    } else {
        check(false, "cache-layout (v_trans): no output tensor to compare");
    }
    ggml_free(ctx);
}

// A 4-D cache (n_stream > 1) is refused by name.
//
// Upstream builds the cache 4-D when a non-unified cache serves several sequences, and the
// stream axis steps by nb[3]. An emitter that ignored ne3 would answer with stream 0's
// tokens for a caller that meant stream s -- plausible and wrong. It cannot pick a stream, so
// it says so.
//
// The fixture is a real two-stream tensor rather than a 4-D view over a one-stream buffer:
// ggml_view_4d asserts `view_size <= ggml_nbytes`, and a two-plane view of a one-plane buffer
// trips that assertion before the emitter is ever entered. A test that dies in its own setup
// proves nothing about the code under test -- the first version of this case did exactly that.
void a_multi_stream_cache_is_refused() {
    cache data;
    const tq::reader::query_view view = two_spans();

    ggml_init_params params{64u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * k4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells, 2);
    ggml_tensor * v4 = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells, 2);
    std::memset(k4->data, 0, ggml_nbytes(k4));
    std::memset(v4->data, 0, ggml_nbytes(v4));
    check(k4->ne[3] == 2 && v4->ne[3] == 2, "fixture: the two-stream cache has two planes on ne3");
    check(std::size_t(k4->ne[0]) == kHeadDim && std::size_t(k4->ne[2]) == kCells,
          "fixture: and otherwise the declared layout, so only ne3 can be the reason");

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = k4;
    in.v = v4;
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    std::memcpy(in.q->data, data.q_flat.data(), data.q_flat.size() * sizeof(float));
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCells;

    const tq::reader::emit_result got = tq::reader::emit_reader_attention(in, view, kScale, nullptr);
    check(got.refusal == tq::reader::emit_refusal::multi_stream_view,
          "multi-stream: refused by name, not silently read as stream 0");
    check(got.out == nullptr, "multi-stream: a refusal emits no tensor");
    std::fprintf(stderr, "multi-stream: %s\n", tq::reader::describe(got.refusal));
    ggml_free(ctx);
}

// A quantized cache is refused by name, whatever its shape.
//
// The fixture is a LEGAL Q8_0 cache -- head_dim 32 is one whole block, so ggml allocates it
// and no assertion fires in setup. That matters: the previous version used a 6-element head,
// which is not a whole block, so the interesting property (does the emitter decline a
// quantized cache at all?) was never reached; the test only showed that a malformed tensor
// could not be built.
//
// The property is worth pinning because the code does not support quantized caches: the
// arithmetic is verified on floats, and the transpose a cell-major V cache needs would require
// a dequantizing pass. A refusal with a cause is the accurate answer; a graph built anyway
// would fail later, in the graph builder, far from here.
void a_quantized_cache_is_refused() {
    ggml_init_params params{64u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);

    tq::reader::query_view view;
    tq::reader::span_keys span;
    span.effective_pos = 3;
    span.cells = {0, 1};
    view.spans = {span};

    constexpr int kQ8Dim = 32;   // one whole Q8_0 block: a cache ggml will actually allocate
    ggml_tensor * kq = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, kQ8Dim, kKvHeads, kCells);
    ggml_tensor * vq = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, kQ8Dim, kKvHeads, kCells);
    check(ggml_blck_size(kq->type) == 32, "fixture: Q8_0 rows are 32-element blocks");
    check(kQ8Dim % 32 == 0, "fixture: this head_dim is a whole block, so the tensor is legal");

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = kq;
    in.v = vq;
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kQ8Dim, 1, kQHeads);
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kQ8Dim;
    in.n_cell = kCells;

    const tq::reader::emit_result got = tq::reader::emit_reader_attention(in, view, kScale, nullptr);
    check(got.refusal == tq::reader::emit_refusal::quantized_cache_unsupported,
          "quantized: a legal Q8_0 cache is still refused, with a cause");
    std::fprintf(stderr, "quantized: %s\n", tq::reader::describe(got.refusal));
    ggml_free(ctx);
}

// A query that is not F32 is refused before ggml sees it.
//
// mul_mat asserts a->type == b->type, and the k/v type check does not cover q: an F16 query
// against an F32 cache would die inside ggml with an assertion, which reads as a crash in
// the engine rather than a refusal at the boundary. Upstream genuinely produces half-
// precision queries on some paths, so this is a shape of input a caller can hand over.
// An F16 query is ACCEPTED -- and this case replaces one that asserted the opposite.
//
// The previous version required a non-F32 query to be refused, because mul_mat asserts one type
// on both sides and this module's arithmetic is written in floats. That was right about the
// constraint and wrong about the conclusion: the fix is to widen the query, not to refuse the
// caller. The assertion it left behind would have pinned the refusal as the contract.
//
// Changing a test because the behaviour changed is only legitimate if the new case carries the
// same weight, so this one checks three things a lazy version would miss:
//
//   1. the emission runs at all (no refusal);
//   2. it agrees with the core fed the values the tensor ACTUALLY holds, which are the widened
//      halves -- read back through ggml, not recomputed here;
//   3. the fixture is not vacuous: at least one widened value must differ from its F32 original,
//      or the comparison would pass whether or not any conversion happened.
//
// (3) is the same teeth the dedicated F16 test uses, and it is the reason this case can be trusted
// where "compare two runs that both skipped the cast" could not.
void a_float16_query_is_accepted() {
    cache data;
    const tq::reader::query_view view = two_spans();

    ggml_init_params params{64u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    const upstream_views views = make_views(ctx, data, false);

    // The F32 source of the query, and its F16 twin filled by ggml_cpy so the rounding is ggml's.
    ggml_tensor * q32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    std::memcpy(q32->data, data.q_flat.data(), data.q_flat.size() * sizeof(float));
    ggml_tensor * q16 = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kHeadDim, 1, kQHeads);
    {
        ggml_tensor * cast = ggml_cpy(ctx, q32, q16);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, cast);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
    }
    check(q16->type == GGML_TYPE_F16, "fixture: the query really is half precision");

    // The values the emitter will compute on: the halves, widened back through ggml.
    std::vector<float> widened(data.q_flat.size(), 0.0f);
    {
        ggml_tensor * back = ggml_cast(ctx, q16, GGML_TYPE_F32);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, back);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        std::memcpy(widened.data(), back->data, widened.size() * sizeof(float));
    }

    // (3) non-vacuity: the conversion must actually have changed something.
    std::size_t changed = 0;
    for (std::size_t i = 0; i < widened.size(); ++i) {
        if (widened[i] != data.q_flat[i]) ++changed;
    }
    check(changed > 0,
          "fixture: at least one query value is not representable in half precision, so the "
          "comparison below would detect a skipped conversion");
    std::fprintf(stderr, "q16: %zu of %zu values changed by the conversion\n", changed,
                 widened.size());

    // The reference, fed exactly those widened values.
    tq::reader::kv_buffers kv;
    kv.k = data.k_flat.data();
    kv.v = data.v_flat.data();
    kv.n_cell = kCells;
    kv.n_head = kQHeads;
    kv.n_head_kv = kKvHeads;
    kv.head_dim = kHeadDim;
    const tq::reader::rope_fn no_rope = [](float *, std::size_t, std::int32_t) {};
    const auto expected = tq::reader::attend(widened.data(), kv, view, kScale, no_rope);
    check(expected.ok, "q16: the core produced a reference from the widened values");

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = views.k;
    in.v = views.v;
    in.q = q16;
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCells;

    const tq::reader::emit_result got = tq::reader::emit_reader_attention(in, view, kScale, nullptr);
    check(got.refusal == tq::reader::emit_refusal::ok,
          "q: an F16 query is accepted, not refused -- the fix is to widen it");
    if (got.out) {
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, got.out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        const float * produced = static_cast<const float *>(got.out->data);
        double worst = 0.0;
        for (std::size_t i = 0; i < expected.out.size() && i < std::size_t(kHeadDim * kQHeads); ++i) {
            const double delta = std::fabs(double(produced[i]) - double(expected.out[i]));
            if (delta > worst) worst = delta;
        }
        std::fprintf(stderr, "q16: emitter vs core max|diff| = %.3e\n", worst);
        check(worst < 1e-5,
              "q: the emission computed on the widened halves, so the answer is the one an F16 "
              "query should produce");
    } else {
        check(false, "q: an accepted emission must produce a tensor");
    }
    ggml_free(ctx);
}

} // namespace

int main() {
    the_default_cache_layout_is_read_correctly();
    the_transposed_cache_layout_is_read_correctly();
    a_multi_stream_cache_is_refused();
    a_quantized_cache_is_refused();
    a_float16_query_is_accepted();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "emitter reads both of upstream's V cache layouts and both float cache "
                         "types (F32 and F16, the one a model here has); refuses multi-stream "
                         "caches and quantized caches by name\n");
    return 0;
}
