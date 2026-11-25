// The physical-form emitter, graded by running it.
//
// What makes this a test rather than a demonstration: the emitter's own admission test is checked
// against the allocator, and the numbers are compared to an independent per-head reference written
// from the definition. Neither of those can pass by being empty — the size of each vector is asserted
// before any difference is taken, and the reference's weights are checked to be non-uniform so a
// reference that ignored the scores could not agree with a graph that also ignored them.
//
// The graph runs the way a product runs it: a `no_alloc` context, `ggml_backend_alloc_ctx_tensors`
// for the placement, the gather indices written afterwards — exactly the `set_inputs` pattern the
// emitter's hand-off exists for. That is also why this test cannot be fooled by a large arena: there
// is no arena to be generous with, the backend sizes it from the graph.
#include <tq/reader/attention.h>
#include <tq/reader/graph_emit_physical.h>

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
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

constexpr int kHeadDim = 8;
constexpr int kKvHeads = 2;
constexpr int kQHeads = 4;
constexpr int kCell = 16;
constexpr float kScale = 0.70710678f;
// The no_alloc contexts still need an arena: objects live in it even when no data does. 1 MiB
// reserves about 2.6k object headers, far more than the ~40 this emission creates, and the
// emitter's own admission test is what decides whether it fits.
constexpr std::size_t kArena = 1024u * 1024u;

// 5 visible cells out of 16, deliberately neither a prefix nor contiguous: a scatter that wrote rows
// densely would land them in the wrong places and the comparison would say so.
tq::reader::query_view the_view(int n_spans) {
    tq::reader::query_view view;
    tq::reader::span_keys a;
    a.effective_pos = 5;
    a.cells = {0, 7, 3};
    tq::reader::span_keys b;
    b.effective_pos = 5;
    b.cells = {12, 15};
    view.spans.push_back(a);
    if (n_spans > 1) {
        view.spans.push_back(b);
    }
    return view;
}

struct fixture {
    std::vector<float> k;   // [hd, kvh, n_cell]
    std::vector<float> v;
    std::vector<float> q;   // [hd, n_head]

    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        unsigned state = 0x2468aceu;
        auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return float((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
    }
};

// The exact value a half holds for a float. Used to build a fixture whose F32 numbers ARE the numbers
// an F16 cache would deliver — so the reference and the graph see identical inputs, and any difference
// between them is the graph's arithmetic rather than the cache's precision.
float round_through_f16(float x) {
    ggml_fp16_t h{};
    ggml_fp32_to_fp16_row(&x, &h, 1);
    float y = 0.0f;
    ggml_fp16_to_fp32_row(&h, &y, 1);
    return y;
}

// An independent per-head physical-attention reference, in double, straight from the definition:
// score(cell, head) = scale * q[head] . k[kvh(head), cell]; ONE softmax over the WHOLE cache with the
// cells the reader cannot see weighing exactly zero; out[d, head] = sum_cell w(cell, head) * v[d, kvh].
//
// It is not `physical.h`'s core: that one's scatter takes one score per cell, while this form needs one
// per (cell, head), so driving it here refuses with `length_mismatch` and an earlier revision ended up
// comparing against an EMPTY vector while printing "max|diff| = 0.000e+00". A reference that cannot
// express the shape is not a reference.
void physical_reference(const fixture & f, const tq::reader::query_view & view,
                        std::vector<float> & weights, std::vector<float> & out) {
    std::vector<int> visible(kCell, 0);
    for (const tq::reader::span_keys & span : view.spans) {
        for (std::int32_t cell : span.cells) visible[std::size_t(cell)] = 1;
    }

    weights.assign(std::size_t(kCell) * std::size_t(kQHeads), 0.0f);
    out.assign(std::size_t(kHeadDim) * std::size_t(kQHeads), 0.0f);

    for (int h = 0; h < kQHeads; ++h) {
        const int kvh = h / (kQHeads / kKvHeads);

        std::vector<double> score(std::size_t(kCell), -INFINITY);
        double best = -INFINITY;
        for (std::size_t cell = 0; cell < std::size_t(kCell); ++cell) {
            if (!visible[cell]) { continue; }
            double dot = 0.0;
            for (int d = 0; d < kHeadDim; ++d) {
                dot += double(f.q[std::size_t(d) + kHeadDim * h]) *
                       double(f.k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * cell)]);
            }
            score[cell] = dot * double(kScale);
            best = std::max(best, score[cell]);
        }

        std::vector<double> e(std::size_t(kCell), 0.0);
        double total = 0.0;
        for (std::size_t cell = 0; cell < std::size_t(kCell); ++cell) {
            if (!visible[cell]) { continue; }
            e[cell] = std::exp(score[cell] - best);
            total += e[cell];
        }

        for (std::size_t cell = 0; cell < std::size_t(kCell); ++cell) {
            if (!visible[cell]) { continue; }
            const double w = e[cell] / total;
            // ggml's weights are [n_cell, 1, n_head] => cell + n_cell*head.
            weights[cell + std::size_t(kCell) * std::size_t(h)] = float(w);
            for (int d = 0; d < kHeadDim; ++d) {
                out[std::size_t(d) + kHeadDim * h] +=
                    w * double(f.v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * cell)]);
            }
        }
    }
}

struct run {
    bool                     ok = false;
    tq::reader::physical_emit_refusal refusal = tq::reader::physical_emit_refusal::ok;
    std::string              why;
    std::vector<float>       out;
    std::vector<float>       weights;
    std::size_t              objects = 0;
    std::size_t              arena_bytes = 0;   // what the placement actually reserved
};

// Build → place → write the indices → compute. Every step is checked; a step that fails leaves `ok`
// false and the caller reports it rather than grading garbage.
//
// `cache_type` is the type the K and V tensors are created with, and it is the only thing that varies
// between the F32 run and the F16 one: the values written are the same numbers, narrowed by the test's
// own fixture when it asks for F16 (`round_through_f16`), so an F16 run does not compare a graph
// against numbers the cache never held. Q stays F32 in both — that is the mixture a real graph has,
// where the query follows the accumulator while the cache follows the model's KV type.
run execute(const fixture & f, int n_spans, ggml_type cache_type = GGML_TYPE_F32,
            ggml_type q_type = GGML_TYPE_F32) {
    run r;
    const tq::reader::query_view view = the_view(n_spans);

    ggml_context * ctx = ggml_init({/*mem_size=*/kArena, /*mem_buffer=*/nullptr, /*no_alloc=*/true});
    if (!ctx) { r.why = "ggml_init(no_alloc) failed"; return r; }

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = ggml_new_tensor_3d(ctx, cache_type, kHeadDim, kKvHeads, kCell);
    in.v = ggml_new_tensor_3d(ctx, cache_type, kHeadDim, kKvHeads, kCell);
    in.q = ggml_new_tensor_3d(ctx, q_type, kHeadDim, 1, kQHeads);
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCell;
    in.collect_indices = true;
    ggml_set_input(in.k);
    ggml_set_input(in.v);
    ggml_set_input(in.q);

    const auto emitted = tq::reader::emit_physical_attention(in, view, kScale);
    r.refusal = emitted.refusal;
    r.why = std::string(tq::reader::describe(emitted.refusal)) + " -- " + emitted.why;
    if (emitted.refusal != tq::reader::physical_emit_refusal::ok) {
        ggml_free(ctx);
        return r;
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, emitted.out);

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { r.why = "cpu backend"; ggml_free(ctx); return r; }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        r.why = "the placement refused to allocate the graph's tensors";
        ggml_backend_free(backend);
        ggml_free(ctx);
        return r;
    }
    r.arena_bytes = ggml_backend_buffer_get_size(buf);

    // The hand-off, as a product does it: after placement, write each index tensor the emitter handed
    // back. A refusal here would mean the emitter promised an index it did not create.
    if (emitted.indices.size() != std::size_t(n_spans) + 1) {
        r.why = "the emitter returned " + std::to_string(emitted.indices.size()) +
                " index tensors for " + std::to_string(n_spans) + " spans; expected spans + 1";
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return r;
    }
    for (const auto & entry : emitted.indices) {
        if (!entry.index || !entry.index->data) {
            r.why = "an index tensor came back without storage after placement";
            ggml_backend_buffer_free(buf);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return r;
        }
        const std::int32_t * cells = entry.span < std::size_t(n_spans)
                                         ? view.spans[entry.span].cells.data()
                                         : emitted.index_cells.data();
        if (entry.span > std::size_t(n_spans)) {
            r.why = "an index entry names span " + std::to_string(entry.span) + ", past the view";
            ggml_backend_buffer_free(buf);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return r;
        }
        std::memcpy(entry.index->data, cells, entry.n_cells * sizeof(std::int32_t));
    }

    // THE VALUES GO IN AT THE CACHE'S OWN WIDTH. An F16 tensor's row is 2 bytes per element, so a
    // `memcpy` of the F32 fixture would write twice the tensor's size and half the elements would be
    // garbage. The test narrows through the same helper the fixture uses, so what lands in the tensor
    // is bit-for-bit what the reference was computed from.
    if (cache_type == GGML_TYPE_F32) {
        std::memcpy(in.k->data, f.k.data(), f.k.size() * sizeof(float));
        std::memcpy(in.v->data, f.v.data(), f.v.size() * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> half(std::max(f.k.size(), f.v.size()));
        ggml_fp32_to_fp16_row(f.k.data(), half.data(), std::int64_t(f.k.size()));
        std::memcpy(in.k->data, half.data(), f.k.size() * sizeof(ggml_fp16_t));
        ggml_fp32_to_fp16_row(f.v.data(), half.data(), std::int64_t(f.v.size()));
        std::memcpy(in.v->data, half.data(), f.v.size() * sizeof(ggml_fp16_t));
    }
    // THE QUERY GOES IN AT ITS OWN WIDTH, for the same reason as K/V above: a memcpy of the F32 fixture
    // into an F16 tensor writes twice the tensor's size.
    if (q_type == GGML_TYPE_F32) {
        std::memcpy(in.q->data, f.q.data(), f.q.size() * sizeof(float));
    } else {
        std::vector<ggml_fp16_t> half_q(f.q.size());
        ggml_fp32_to_fp16_row(f.q.data(), half_q.data(), std::int64_t(f.q.size()));
        std::memcpy(in.q->data, half_q.data(), f.q.size() * sizeof(ggml_fp16_t));
    }

    const ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        r.why = "compute failed with status " + std::to_string(int(st));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return r;
    }

    const float * w = static_cast<const float *>(emitted.weights->data);
    const float * o = static_cast<const float *>(emitted.out->data);
    r.weights.assign(w, w + std::size_t(kCell) * kQHeads);
    r.out.assign(o, o + std::size_t(kHeadDim) * kQHeads);
    r.objects = emitted.indices.size();
    r.ok = true;

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);
    return r;
}

// Does `ggml_used_mem` respond at all? If it did not, the emitter's self-check would be a measurement
// that cannot fail, and every number it reports would be theatre.
void probe_instrument() {
    ggml_context * ctx = ggml_init({16u * 1024u, nullptr, /*no_alloc=*/true});
    const std::size_t a = ggml_used_mem(ctx);
    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    const std::size_t b = ggml_used_mem(ctx);
    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8);
    const std::size_t c = ggml_used_mem(ctx);
    std::fprintf(stderr, "instrument: used_mem %zu -> %zu -> %zu (%s per object)\n", a, b, c,
                 b > a ? std::to_string(b - a).c_str() : "NO GROWTH");
    check(b > a && c > b, "ggml_used_mem grows per object, so the emitter's admission test measures "
                          "something that can fail");
    ggml_free(ctx);
}

// THE AUDIT THAT MATTERS: is the emitter's object count a real upper bound, and is one object's cost
// the same for every object the emission creates?
//
// The claim under test is `objects = 13 + 6*spans` charged against a `per_object` measured once from a
// single `ggml_new_tensor_1d`. Two things could make that unsound, and neither is visible by reading
// the emitter alone:
//
//   1. AN OBJECT THAT COSTS A DIFFERENT AMOUNT. `ggml_new_object` is used for TENSOR, GRAPH and
//      WORK_BUFFER objects, and only the first has the size the probe measures. If any op in the
//      emission created one of the others (a graph, a backend work buffer), `per_object` would be the
//      wrong constant and the bound would be meaningless.
//   2. A NODE THE EMITTER DOES NOT CALL. `ggml_mul_mat` and `ggml_soft_max_ext` are free to create
//      helper tensors of their own (a cast, a cont, an implicit mask) inside ggml, and those are
//      objects the emitter never wrote down.
//
// Both are answered by counting, not by argument: create the inputs, emit, and divide the arena growth
// that followed by the measured cost. The result must be (a) an exact multiple of `per_object` — a
// remainder would mean an object of a different size — and (b) no larger than the charge.
void probe_object_count() {
    ggml_context * probe = ggml_init({16u * 1024u, nullptr, /*no_alloc=*/true});
    const std::size_t a = ggml_used_mem(probe);
    ggml_tensor * one = ggml_new_tensor_1d(probe, GGML_TYPE_F32, 8);
    const std::size_t b = ggml_used_mem(probe);
    // A VIEW of the same context: `ggml_reshape_2d` calls the same allocation path with view_src set.
    ggml_reshape_2d(probe, one, 2, 4);
    const std::size_t c = ggml_used_mem(probe);
    ggml_free(probe);

    const std::size_t per_object = b - a;
    check(per_object != 0, "the probe measures a non-zero object cost");
    check(c - b == per_object,
          "a VIEW costs exactly what a tensor costs, so the constant is not specific to one node kind");

    for (int n_spans : {1, 2, 3, 4}) {
        ggml_context * ctx = ggml_init({kArena, nullptr, /*no_alloc=*/true});
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

        // A view built HERE, not `the_view(n)`: that helper deliberately stops at two spans (it exists
        // for the numerical comparison, whose reference covers 1 and 2), so asking it for three or four
        // returns the two-span view and the object count would be measured for a graph that was never
        // asked for. This probe needs the span count it says it needs.
        tq::reader::query_view view;
        for (int s = 0; s < n_spans; ++s) {
            tq::reader::span_keys span;
            span.effective_pos = 5;
            for (int j = 0; j < 4; ++j) {
                span.cells.push_back(std::int32_t(s * 4 + j));
            }
            view.spans.push_back(span);
        }

        const std::size_t before = ggml_used_mem(ctx);
        const auto emitted = tq::reader::emit_physical_attention(in, view, kScale);
        const std::size_t grew = ggml_used_mem(ctx) - before;
        ggml_free(ctx);

        if (emitted.refusal != tq::reader::physical_emit_refusal::ok) {
            check(false, "the emission is admitted at " + std::to_string(n_spans) + " spans");
            continue;
        }

        const std::size_t actual = grew / per_object;
        const std::size_t charged = 13 + 6 * std::size_t(n_spans);
        std::fprintf(stderr, "objects %d spans: actual %zu, charged %zu, remainder %zu\n", n_spans,
                     actual, charged, grew % per_object);
        // A remainder means at least one object of another kind — a graph or work buffer — inside an
        // emission that assumes every object is a tensor.
        check(grew % per_object == 0,
              std::to_string(n_spans) + " spans: every object the emission created is the size the "
                                        "admission test charged for");
        check(actual <= charged,
              std::to_string(n_spans) + " spans: the count charged (" + std::to_string(charged) +
                  ") is at least the count created (" + std::to_string(actual) + "), so the bound "
                  "holds and a wrong count cannot certify itself");
        // And the margin is not accidental: `13 + 6S` is `11 + 6S` with two objects of slack. Pinned,
        // because growth of the real count past the charge is exactly what a future ggml would do.
        check(actual == 11 + 6 * std::size_t(n_spans),
              std::to_string(n_spans) + " spans: the true count is 11 + 6*spans, leaving the charged "
                                        "count two objects of slack");
    }
}

// THE CALLER'S SCAFFOLDING, measured, because it is the reason a "correct" arena still crashed.
//
// A previous audit found the fixture dying with rc=139 at 8, 12 and 16 KiB arenas and could not say
// where, only that it was "inside its own ggml_new_graph". It is measurable, and the number is not
// marginal: a default `ggml_new_graph` on the same context costs 82608 bytes, against 368 for one
// tensor object — 224x this entire emission. So an arena sized to `bytes_required` alone is not a
// tight arena, it is a guaranteed abort, and the emitter's bound was never the thing to blame.
//
// That is what `bytes_required` exists for: the emitter reports its exact cost, the caller adds
// `ggml_new_graph`'s, and only then is the arena large enough. This probe pins both numbers.
void probe_caller_budget() {
    ggml_context * ctx = ggml_init({kArena, nullptr, /*no_alloc=*/true});

    const std::size_t before_graph = ggml_used_mem(ctx);
    ggml_new_graph(ctx);
    const std::size_t graph_cost = ggml_used_mem(ctx) - before_graph;

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

    const std::size_t before = ggml_used_mem(ctx);
    const auto emitted = tq::reader::emit_physical_attention(in, the_view(1), kScale);
    const std::size_t grew = ggml_used_mem(ctx) - before;
    ggml_free(ctx);

    check(emitted.refusal == tq::reader::physical_emit_refusal::ok,
          "the emission is admitted, so the budget report has numbers to carry");
    if (emitted.refusal != tq::reader::physical_emit_refusal::ok) { return; }

    std::fprintf(stderr,
                 "caller budget: graph object %zu B, emission %zu B (%zu objects x %zu B), "
                 "actual growth %zu B\n",
                 graph_cost, emitted.bytes_required, emitted.objects_charged, emitted.object_cost,
                 grew);

    // The bound is an upper bound on what was spent, and the reported arithmetic reproduces it.
    check(grew <= emitted.bytes_required,
          "the emission's actual arena growth stays inside the bound it reported");
    check(emitted.bytes_required == emitted.objects_charged * emitted.object_cost,
          "the reported cost is exactly `objects * object_cost` — a caller can re-derive it against "
          "another ggml build");
    // And the scope of the bound, asserted so nobody reads it as covering the whole context: the
    // graph object is more than two orders of magnitude larger than the emission, so `bytes_required`
    // alone is not a safe arena size and the header says so.
    check(graph_cost > emitted.bytes_required,
          "the caller's own ggml_new_graph costs more than the whole emission — which is why "
          "bytes_required is not a safe arena size on its own");
}

// THE PUBLISH STEP, tested where it can fail. Four facts, each of which is a way a caller can get
// this wrong with no symptom at all if the API only documented a loop instead of owning one:
//
//   1. from a result whose inputs have no storage yet (the natural mistake: publish before
//      `ggml_backend_sched_alloc_graph`), every entry is refused with `no_storage` and NOT ONE
//      write reaches the writer. This is the case the no_alloc fixture reproduces exactly, because
//      a no_alloc context's inputs really do have null `data`.
//   2. with storage present, publishing covers every entry -- N gathers and then the scatter -- and
//      the bytes handed to the writer are the right slice of `index_cells` for each tensor.
//      "Storage present" is simulated by pointing `t->data` at a host buffer: no device is involved,
//      the writer records, and what is under test is the pairing, not ggml's allocator.
//   3. a writer that refuses mid-way reports `write_failed` with the span that failed -- and the
//      entries before it stay written, because inventing a rollback for a device write would be a
//      fiction. The test asserts that ordering rather than pretending otherwise.
//   4. a result from a refusal publishes nothing: `no_entries`, zero writes.
struct record {
    ggml_tensor * tensor = nullptr;
    std::size_t offset = 0;
    std::size_t size = 0;
    std::vector<std::int32_t> bytes;
};

std::vector<record> g_written;

bool recording_writer(ggml_tensor * tensor, const void * data, std::size_t offset, std::size_t size) {
    record r;
    r.tensor = tensor;
    r.offset = offset;
    r.size = size;
    const auto * cells = static_cast<const std::int32_t *>(data);
    r.bytes.assign(cells, cells + size / sizeof(std::int32_t));
    g_written.push_back(std::move(r));
    return true;
}

bool refuses_second(ggml_tensor * tensor, const void * data, std::size_t offset, std::size_t size) {
    if (!recording_writer(tensor, data, offset, size)) { return false; }
    return g_written.size() != 2;   // the second entry fails
}

// An adapter that RAISES instead of returning false — the case the write loop must convert into a
// reported refusal. Throwing an int, not std::runtime_error: the point is that ANY exception is
// caught at this boundary, and including <stdexcept> to demonstrate that would only test the
// include. `catch (...)` catches an int exactly as it catches a class.
bool throwing_writer(ggml_tensor *, const void *, std::size_t, std::size_t) {
    throw 1;
}

void probe_publish() {
    // ---- 1. no storage: refused, and nothing written -----------------------------------------
    {
        ggml_context * ctx = ggml_init({kArena, nullptr, /*no_alloc=*/true});
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

        const auto emitted = tq::reader::emit_physical_attention(in, the_view(2), kScale);
        check(emitted.refusal == tq::reader::physical_emit_refusal::ok, "publish fixture: emitted");
        check(emitted.indices.size() == 3, "publish fixture: two gathers and one scatter");

        g_written.clear();
        std::string why;
        std::size_t failed = 0;
        const auto r = tq::reader::publish_index_inputs(emitted, &recording_writer, &why, &failed);
        check(r == tq::reader::index_publish_refusal::no_storage,
              "publishing before placement is REFUSED as no_storage, not written into a null pointer");
        check(g_written.empty(),
              "and not one write reached the writer: validation covers every entry before any device "
              "traffic, so a refusal leaves the graph untouched");
        check(failed == 0, "a validation refusal does not report a failed span, because none was tried");
        check(!why.empty(), "the refusal explains which entry and why");

        // ---- 2 + 3. storage present -----------------------------------------------------------
        // A host buffer per index tensor stands in for what the scheduler would have attached.
        std::vector<std::vector<float>> storage(kHeadDim * kKvHeads, std::vector<float>(64, 0.0f));
        std::size_t slot = 0;
        for (const auto & entry : emitted.indices) {
            entry.index->data = storage[slot++ % storage.size()].data();
        }

        g_written.clear();
        why.clear();
        const auto ok = tq::reader::publish_index_inputs(emitted, &recording_writer, &why);
        check(ok == tq::reader::index_publish_refusal::ok, "with storage, the whole set publishes");
        check(g_written.size() == emitted.indices.size(),
              "every entry was written, not a subset: N gathers and the scatter");

        // The bytes handed over must be that entry's slice of index_cells, in the same order.
        bool slices_match = true;
        for (std::size_t i = 0; i < emitted.indices.size() && i < g_written.size(); ++i) {
            const auto & entry = emitted.indices[i];
            const auto & got = g_written[i];
            if (got.tensor != entry.index || got.offset != 0 ||
                got.size != entry.n_cells * sizeof(std::int32_t) || got.bytes.size() != entry.n_cells) {
                slices_match = false;
                continue;
            }
            for (std::size_t j = 0; j < entry.n_cells; ++j) {
                if (got.bytes[j] != emitted.index_cells[entry.cells_offset + j]) {
                    slices_match = false;
                }
            }
        }
        check(slices_match,
              "each entry received its own slice of index_cells — the gathers their spans, the scatter "
              "the whole visible set");

        // The scatter's entry is the last one and names all visible cells; the gathers name their own.
        if (emitted.indices.size() == 3 && g_written.size() == 3) {
            check(emitted.indices[2].span == 2 && emitted.indices[2].n_cells == emitted.index_cells.size(),
                  "the scatter is published last and covers the whole visible set");
            check(emitted.indices[0].n_cells + emitted.indices[1].n_cells == emitted.indices[2].n_cells,
                  "the gathers' cells partition the scatter's: same total, no double count");
        }

        // ---- 3. a writer that refuses mid-way --------------------------------------------------
        g_written.clear();
        why.clear();
        failed = 0;
        const auto partial = tq::reader::publish_index_inputs(emitted, &refuses_second, &why, &failed);
        check(partial == tq::reader::index_publish_refusal::write_failed,
              "a failing writer is reported as write_failed, not swallowed");
        check(failed == 1, "failed_span names the entry that failed");
        check(g_written.size() == 2,
              "the writes before the failure stand: a device write has no rollback, and the API says so "
              "rather than offering a retry that would publish two batches");

        ggml_free(ctx);
    }

    // ---- 3b. THE LAYOUT CHECKS HAVE TEETH: a caller that edits the result is refused -------------
    //
    // This is the section the audit was for. `physical_emit_result` is the caller's object, so a
    // caller can erase a gather, re-point an offset, or null a tensor between emit and publish. Every
    // one of those leaves entries that pass a BOUNDS check, and the revision before this one would
    // have written the remainder and returned `ok`. The storage is attached first in each case below,
    // so a passing write is possible — the refusal has to come from the layout, not from luck.
    {
        ggml_context * ctx = ggml_init({kArena, nullptr, /*no_alloc=*/true});
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

        const auto emitted = tq::reader::emit_physical_attention(in, the_view(2), kScale);
        check(emitted.refusal == tq::reader::physical_emit_refusal::ok, "layout fixture: emitted");
        // The result carries the N+1 entries the tiling below depends on. The provenance counter a
        // previous revision kept here is gone — see the field comment in the header — so what is
        // asserted is the shape, which is all `publish_index_inputs` can honestly check.
        check(emitted.indices.size() == 3, "layout fixture: two gathers and one scatter");

        // Attach storage so nothing below is refused for the wrong reason.
        std::vector<std::vector<float>> storage(kHeadDim * kKvHeads, std::vector<float>(64, 0.0f));
        for (std::size_t i = 0; i < emitted.indices.size(); ++i) {
            emitted.indices[i].index->data = storage[i].data();
        }

        // ---- 0. THE DOCUMENTED LIMIT, made observable -------------------------------------------
        //
        // A set that satisfies the SHAPE publishes, whoever assembled it. The previous revision kept
        // a public counter and claimed that made a hand-built result refuse; it did not, because the
        // caller can write the counter. This case is the counter-demonstration: three freshly made
        // I32 inputs, arbitrary cell values, tiled correctly — and `publish_index_inputs` returns ok.
        //
        // That is the whole point of stating the contract as STRUCTURE in the header instead of
        // provenance: the guarantee is "each tensor gets exactly its own slice, and nothing is
        // written unless the whole set is consistent", not "only the emitter can produce this".
        {
            tq::reader::physical_emit_result forged;
            forged.index_cells = {4, 5, 6, 7, 8};
            ggml_tensor * g0 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
            ggml_tensor * g1 = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 2);
            ggml_tensor * sc = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 5);
            std::size_t slot = 0;
            for (ggml_tensor * t : {g0, g1, sc}) {
                ggml_set_input(t);
                t->data = storage[slot++ % storage.size()].data();
            }
            forged.indices = {{0, g0, 0, 3}, {1, g1, 3, 2}, {2, sc, 0, 5}};

            g_written.clear();
            std::string why;
            const auto r = tq::reader::publish_index_inputs(forged, &recording_writer, &why);
            check(r == tq::reader::index_publish_refusal::ok,
                  "a HAND-ASSEMBLED set that tiles publishes: the contract is structure, and the "
                  "header says so rather than claiming provenance a public field cannot supply");
            check(g_written.size() == 3, "and each of its entries received its own slice");
        }

        // ---- 0b. AN ADAPTER THAT RAISES is reported, not propagated ----------------------------
        //
        // Without the try/catch the exception would leave publish_index_inputs with the earlier
        // entries written and no return value, so the caller would learn nothing. The span is set
        // before the call, which is why failed_span is meaningful here at all.
        {
            g_written.clear();
            std::string why;
            std::size_t failed = 99;
            const auto r = tq::reader::publish_index_inputs(emitted, &throwing_writer, &why, &failed);
            check(r == tq::reader::index_publish_refusal::write_failed,
                  "a writer that raises is converted to write_failed at the boundary");
            check(failed == 0, "failed_span names the entry that was in flight (the first)");
            check(why.find("raised") != std::string::npos,
                  "and the message says the writer raised, so the caller can tell the two failure "
                  "modes apart");
            check(g_written.empty(), "nothing reached the recording path, because this writer never "
                                     "calls it");
        }

        // ---- 0c. A NEGATIVE ne[0] is a malformed tensor, named as such -------------------------
        //
        // int64_t -> std::size_t turns -1 into a colossal value, so the size comparison would have
        // rejected it anyway — with a message reading "n_cells 3 against ne[0] -1", which reads like
        // a size disagreement and hides that the tensor itself is broken.
        {
            auto broken = emitted;
            const std::int64_t saved = broken.indices[0].index->ne[0];
            broken.indices[0].index->ne[0] = -1;
            g_written.clear();
            std::string why;
            const auto r = tq::reader::publish_index_inputs(broken, &recording_writer, &why);
            broken.indices[0].index->ne[0] = saved;   // the graph is shared; leave it as it was
            check(r == tq::reader::index_publish_refusal::size_disagrees,
                  "a negative extent is refused as a size disagreement");
            check(why.find("negative") != std::string::npos,
                  "and the message says the extent is negative rather than printing a size mismatch "
                  "that never happened");
            check(g_written.empty(), "and nothing was written");
        }

        // (a) erase a gather: the tiling no longer covers the buffer.
        {
            auto subset = emitted;
            subset.indices.erase(subset.indices.begin());
            g_written.clear();
            std::string why;
            const auto r = tq::reader::publish_index_inputs(subset, &recording_writer, &why);
            check(r == tq::reader::index_publish_refusal::entry_layout,
                  "DELETING A GATHER is refused as entry_layout, not published as if it were the set");
            check(g_written.empty(), "and nothing was written: a subset never reaches the device");
            check(!why.empty(), "the refusal says which count disagreed");
        }

        // (b) an entry whose tensor is not an input: bounds-passing, but nothing would read it.
        {
            auto broken = emitted;
            broken.indices[0].index = nullptr;
            g_written.clear();
            const auto r = tq::reader::publish_index_inputs(broken, &recording_writer, nullptr);
            check(r == tq::reader::index_publish_refusal::not_an_input,
                  "a null tensor is refused as not_an_input");
            check(g_written.empty(), "and nothing was written");
        }

        // (c) a re-pointed offset: in range, but it would put one span's cells into another's tensor.
        {
            auto shifted = emitted;
            shifted.indices[1].cells_offset = 0;   // collides with the first gather
            g_written.clear();
            const auto r = tq::reader::publish_index_inputs(shifted, &recording_writer, nullptr);
            check(r == tq::reader::index_publish_refusal::entry_layout,
                  "a gather pointing at the wrong slice breaks the tiling and is refused");
            check(g_written.empty(), "and nothing was written");
        }

        // (d) a scatter that does not cover the buffer would leave cells unpublished.
        {
            auto short_scatter = emitted;
            short_scatter.indices.back().n_cells -= 1;
            g_written.clear();
            const auto r = tq::reader::publish_index_inputs(short_scatter, &recording_writer, nullptr);
            check(r == tq::reader::index_publish_refusal::entry_layout ||
                      r == tq::reader::index_publish_refusal::size_disagrees,
                  "a scatter short of the buffer is refused rather than leaving cells unwritten");
            check(g_written.empty(), "and nothing was written");
        }

        // (e) the untouched result still publishes: the checks above are not refusing everything.
        {
            g_written.clear();
            std::string why;
            const auto r = tq::reader::publish_index_inputs(emitted, &recording_writer, &why);
            check(r == tq::reader::index_publish_refusal::ok, "the unedited result still publishes");
            check(g_written.size() == 3, "all three entries written");
        }

        ggml_free(ctx);
    }

    // ---- 4. a refusal publishes nothing --------------------------------------------------------
    {
        g_written.clear();
        std::string why;
        const tq::reader::physical_emit_result refused;   // default: refusal = ok, but no entries
        const auto r = tq::reader::publish_index_inputs(refused, &recording_writer, &why);
        check(r == tq::reader::index_publish_refusal::no_entries,
              "an empty result is refused as no_entries rather than treated as a successful no-op");
        check(g_written.empty(), "and writes nothing");
    }
}

} // namespace

int main() {
    probe_instrument();
    probe_object_count();
    probe_caller_budget();
    probe_publish();

    const fixture f;

    // ---- 1. the reference's own invariants, before it is quoted as anything ---------------------
    for (int n_spans : {1, 2}) {
        const tq::reader::query_view view = the_view(n_spans);
        std::vector<float> ref_weights;
        std::vector<float> ref_out;
        physical_reference(f, view, ref_weights, ref_out);

        const std::size_t n_visible = n_spans == 1 ? 3u : 5u;
        check(ref_weights.size() == std::size_t(kCell) * kQHeads, "the reference fills every (cell, head)");
        check(ref_out.size() == std::size_t(kHeadDim) * kQHeads, "the reference fills every (dim, head)");

        std::vector<double> head_sum(std::size_t(kQHeads), 0.0);
        int nonzero_hidden = 0;
        int non_uniform = 0;
        std::vector<int> visible(kCell, 0);
        for (const auto & span : view.spans) {
            for (std::int32_t c : span.cells) visible[std::size_t(c)] = 1;
        }
        for (std::size_t i = 0; i < ref_weights.size(); ++i) {
            const std::size_t cell = i % std::size_t(kCell);
            const std::size_t head = i / std::size_t(kCell);
            head_sum[head] += double(ref_weights[i]);
            if (!visible[cell] && ref_weights[i] != 0.0f) ++nonzero_hidden;
            if (visible[cell] && std::fabs(double(ref_weights[i]) - 1.0 / double(n_visible)) > 1e-7) {
                ++non_uniform;
            }
        }
        bool sums_one = true;
        for (double s : head_sum) sums_one = sums_one && std::fabs(s - 1.0) < 1e-6;
        // 1e-6, not 1e-9: the reference computes in double but STORES float, so a head's sum lands a
        // few float epsilons from 1. Asserting 1e-9 fails — the tolerance was wrong, not the maths.
        check(sums_one, std::to_string(n_spans) + " span(s): each of the reference's heads sums to one");
        check(nonzero_hidden == 0,
              std::to_string(n_spans) + " span(s): the reference leaves the cells outside the view at "
                                        "exactly zero — the softmax's -INFINITYs do their job");
        check(non_uniform > 0,
              std::to_string(n_spans) + " span(s): the reference's weights are NOT all 1/n_visible, so "
                                        "the scores reach them and agreement is evidence about the "
                                        "arithmetic rather than about normalisation");
    }

    // ---- 2. the emitter, through a real no_alloc graph and a real backend -----------------------
    for (int n_spans : {1, 2}) {
        const tq::reader::query_view view = the_view(n_spans);
        std::vector<float> ref_weights;
        std::vector<float> ref_out;
        physical_reference(f, view, ref_weights, ref_out);

        const run r = execute(f, n_spans);
        const std::string tag = std::to_string(n_spans) + " span(s)";

        if (!r.ok) {
            check(false, tag + ": the emission did not produce a graph — " + r.why);
            continue;
        }
        std::fprintf(stderr, "%s: emitted and computed; placement reserved %zu bytes, %zu index tensors\n",
                     tag.c_str(), r.arena_bytes, r.objects);

        // Sizes first. A comparison loop bounded by the reference's size can run zero times and
        // report a difference of nothing.
        check(r.weights.size() == ref_weights.size(), tag + ": one weight per (cell, head)");
        check(r.out.size() == ref_out.size(), tag + ": one value per (dim, head)");

        double worst_w = 0.0;
        for (std::size_t i = 0; i < std::min(r.weights.size(), ref_weights.size()); ++i) {
            worst_w = std::max(worst_w, std::fabs(double(r.weights[i]) - double(ref_weights[i])));
        }
        double worst_o = 0.0;
        for (std::size_t i = 0; i < std::min(r.out.size(), ref_out.size()); ++i) {
            worst_o = std::max(worst_o, std::fabs(double(r.out[i]) - double(ref_out[i])));
        }
        std::fprintf(stderr, "%s: weights max|diff| = %.3e over %zu; out max|diff| = %.3e over %zu\n",
                     tag.c_str(), worst_w, r.weights.size(), worst_o, r.out.size());

        check(worst_w < 1e-6, tag + ": the graph's post-softmax weights match the reference's — the "
                                   "scatter wrote each row at its own cell and ONE softmax ran over "
                                   "the full cache length");
        check(worst_o < 1e-5, tag + ": the attended tensor matches the reference's, so the value "
                                   "reduction ran in the native order");

        // The emitter's own claim, checked from outside: under no_alloc the arena is object headers,
        // so the number it charged is the placement's floor rather than a guess with margin.
        check(r.arena_bytes >= std::size_t(kCell) * kQHeads * sizeof(float),
              tag + ": the placement is at least the size of the largest tensor it must hold");
    }

    // ---- 3. THE MIXTURE A REAL MODEL HANDS IN: F16 cache, F32 query -----------------------------
    //
    // The physical form was F32-only until this revision, and a real cache is not: the tiny
    // transformer in this tree builds F16 (measured cache type 1, ne [8 4 256 1], n_kv 256) while the
    // graph keeps its query in F32. So the case that matters is not "all F32" or "all F16" but the
    // mixture, and it is the one checked here.
    //
    // The fixture's values are first rounded through F16, so the reference and the cache hold the SAME
    // numbers and the comparison measures the graph's arithmetic. A tolerance taken from F16's
    // precision would have been the wrong instrument: F16 -> F32 is exact, so the only error left is
    // F32 accumulation, and 1e-5 is the bound that error already has in the F32 case.
    {
        fixture f16;
        for (float & x : f16.k) x = round_through_f16(x);
        for (float & x : f16.v) x = round_through_f16(x);

        for (int n_spans : {1, 2}) {
            const tq::reader::query_view view = the_view(n_spans);
            std::vector<float> ref_weights;
            std::vector<float> ref_out;
            physical_reference(f16, view, ref_weights, ref_out);

            const run r = execute(f16, n_spans, GGML_TYPE_F16);
            const std::string tag = "F16 cache, F32 q, " + std::to_string(n_spans) + " span(s)";

            if (!r.ok) {
                check(false, tag + ": the emission did not produce a graph — " + r.why);
                continue;
            }
            std::fprintf(stderr, "%s: emitted and computed; placement reserved %zu bytes\n",
                         tag.c_str(), r.arena_bytes);

            check(r.weights.size() == ref_weights.size(), tag + ": one weight per (cell, head)");
            check(r.out.size() == ref_out.size(), tag + ": one value per (dim, head)");

            double worst_w = 0.0;
            for (std::size_t i = 0; i < std::min(r.weights.size(), ref_weights.size()); ++i) {
                worst_w = std::max(worst_w, std::fabs(double(r.weights[i]) - double(ref_weights[i])));
            }
            double worst_o = 0.0;
            for (std::size_t i = 0; i < std::min(r.out.size(), ref_out.size()); ++i) {
                worst_o = std::max(worst_o, std::fabs(double(r.out[i]) - double(ref_out[i])));
            }
            std::fprintf(stderr, "%s: weights max|diff| = %.3e over %zu; out max|diff| = %.3e over %zu\n",
                         tag.c_str(), worst_w, r.weights.size(), worst_o, r.out.size());

            // THE SAME BOUNDS AS THE F32 CASE, deliberately. The widening is exact, so a half cache
            // gets no allowance: if these need slack, the cast is in the wrong place or the wrong
            // type reached a kernel, and loosening the tolerance would hide precisely that.
            check(worst_w < 1e-6, tag + ": the weights match a reference computed from the same F16 "
                                       "values, so the widening preserved them");
            check(worst_o < 1e-5, tag + ": the attended tensor matches, so the value reduction saw the "
                                       "same numbers the cache stores");
        }

        // F16 q, in BOTH cache types. The query is the one operand this emission widens, so it is the
        // one whose cast has to be measured. What stood here before was
        //
        //     check(true, "F16 q is accepted and charged; its cast path is shared with F16 k and v");
        //
        // — a tautology, and worse, a false one: by that revision k and v were no longer cast at all,
        // so "shared with F16 k and v" described code that had been deleted. A passing check asserting
        // nothing, in a sentence misdescribing the implementation, is how a path stays untested while
        // reading as covered. These two runs replace it.
        for (ggml_type cache_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            fixture fq = f16;                                  // k/v already rounded through F16
            for (float & x : fq.q) x = round_through_f16(x);    // and so is the query

            const std::string tag = std::string("F16 q, ") +
                                    (cache_type == GGML_TYPE_F32 ? "F32 cache" : "F16 cache");

            const run r = execute(fq, 1, cache_type, GGML_TYPE_F16);
            std::vector<float> ref_weights;
            std::vector<float> ref_out;
            physical_reference(fq, the_view(1), ref_weights, ref_out);

            if (!r.ok) {
                check(false, tag + ": the emission did not produce a graph — " + r.why);
                continue;
            }
            check(r.weights.size() == ref_weights.size(), tag + ": one weight per (cell, head)");
            check(r.out.size() == ref_out.size(), tag + ": one value per (dim, head)");

            double worst_w = 0.0;
            for (std::size_t i = 0; i < std::min(r.weights.size(), ref_weights.size()); ++i) {
                worst_w = std::max(worst_w, std::fabs(double(r.weights[i]) - double(ref_weights[i])));
            }
            double worst_o = 0.0;
            for (std::size_t i = 0; i < std::min(r.out.size(), ref_out.size()); ++i) {
                worst_o = std::max(worst_o, std::fabs(double(r.out[i]) - double(ref_out[i])));
            }
            std::fprintf(stderr, "%s: weights max|diff| = %.3e over %zu; out max|diff| = %.3e over %zu\n",
                         tag.c_str(), worst_w, r.weights.size(), worst_o, r.out.size());

            // Same bounds as every other case: widening F16 -> F32 is exact, so a half query gets no
            // allowance. If this needed slack, the cast would be in the wrong place.
            check(worst_w < 1e-6, tag + ": the widened query reproduces the weights computed from the " 
                                       "same half-precision query");
            check(worst_o < 1e-5, tag + ": the attended tensor matches, so the value reduction saw the "
                                       "same numbers the cache stores");
        }
    }

    // ---- 3. the refusals a caller needs to tell apart -------------------------------------------
    {
        // An allocating context: refused, because its arena cost cannot be enumerated. This is not a
        // limitation to be patched with a margin; it is the reason the no_alloc path exists.
        ggml_context * ctx = ggml_init({1u * 1024u * 1024u, nullptr, /*no_alloc=*/false});
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
        const std::size_t before = ggml_used_mem(ctx);
        const auto a = tq::reader::emit_physical_attention(in, the_view(1), kScale);
        const std::size_t after = ggml_used_mem(ctx);
        check(a.refusal == tq::reader::physical_emit_refusal::arena_admission_unproven,
              "an allocating context is refused by NAME, not admitted behind an unproven estimate");
        check(after == before, "the allocating-context refusal allocates nothing");
        ggml_free(ctx);
    }
    {
        // no_alloc without the hand-off: there is nowhere to write the gather indices, so the refusal
        // names that instead of failing later inside ggml.
        ggml_context * ctx = ggml_init({kArena, nullptr, /*no_alloc=*/true});
        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim;
        in.n_cell = kCell;
        in.collect_indices = false;
        const auto a = tq::reader::emit_physical_attention(in, the_view(1), kScale);
        check(a.refusal == tq::reader::physical_emit_refusal::index_data_unavailable,
              "no_alloc without collect_indices is refused as a missing hand-off");
        ggml_free(ctx);
    }
    {
        // A duplicate cell would make the scatter write one position twice, and which write wins
        // depends on iteration order — malformed, and distinguishable from unsupported.
        ggml_context * ctx = ggml_init({kArena, nullptr, /*no_alloc=*/true});
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
        tq::reader::query_view dup = the_view(2);
        dup.spans[1].cells = {12, 3};
        const std::size_t before = ggml_used_mem(ctx);
        const auto a = tq::reader::emit_physical_attention(in, dup, kScale);
        check(a.refusal == tq::reader::physical_emit_refusal::duplicate_cell,
              "a cell listed twice is refused as malformed, before any allocation");
        check(ggml_used_mem(ctx) == before, "the malformed-view refusal allocates nothing");
        ggml_free(ctx);
    }

    std::fprintf(stderr, failures
                             ? "%d failure(s)\n"
                             : "physical emitter: a no_alloc graph reproduces the independent "
                               "per-head reference through a real backend, and its arena admission "
                               "is an exact object count against a measured per-object cost\n",
                 failures);
    return failures ? 1 : 0;
}
