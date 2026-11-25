// The emitted graph against the arithmetic core.
//
// attention.h is verified twice over: against a double-precision reference and against
// ggml's own kernels (attention-ggml-test.cpp). What was never verified is the thing the
// seam actually needs -- that a graph built out of ggml nodes from the same view produces
// the same numbers. Those are different programs: the core gathers floats in C++, the
// emitter gathers rows with ggml_get_rows, concatenates, and normalizes in one softmax
// node. A shape that mul_mat accepts can still reduce the wrong axis.
//
// So this test runs both and compares. The rotation is the identity on both sides, on
// purpose: the rotation belongs to upstream's kernel (the emitter takes it as an injected
// capability), and a test that compared my rope with my rope would prove nothing. What is
// measured here is exactly what this module owns -- the gather, the per-span concatenation,
// the single normalization, the value reduction and the head grouping.
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

constexpr int kKvHeads  = 2;
constexpr int kQHeads   = 4;
constexpr int kHeadDim  = 8;
constexpr int kCells    = 24;     // the cache
constexpr float kScale  = 0.70710678f;
constexpr int kGroup    = kQHeads / kKvHeads;

// Deterministic fixture in the reader's layouts:
//   k[cell][kvh][d]  at  (cell * n_head_kv + h) * head_dim + d
//   v[cell][kvh][d]  at  the same place -- and the emitter's v tensor is
//                        [head_dim, n_head_kv, n_cell] over that same flat buffer, because
//                        llama stores both caches in this order. The value axis is NOT
//                        declared as [n_cell, head_dim, ...]: that is a different memory
//                        permutation, and a test that filled it would compare the emitter
//                        with a layout the product never has.
struct board {
    std::vector<float> k, v, q;

    board() {
        k.assign(std::size_t(kCells) * kKvHeads * kHeadDim, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kQHeads) * kHeadDim, 0.0f);
        unsigned s = 987654321u;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (auto & x : k) x = next();
        for (auto & x : v) x = next() * 2.0f;
        for (auto & x : q) x = next();
    }

    tq::reader::kv_buffers core_buffers() const {
        tq::reader::kv_buffers kv;
        kv.k = k.data();
        kv.v = v.data();
        kv.n_cell = kCells;
        kv.n_head = kQHeads;
        kv.n_head_kv = kKvHeads;
        kv.head_dim = kHeadDim;
        return kv;
    }

    // Deliberately not the identity view: two spans, with the second holding keys that sit
    // *after* the first in the cache but are read out of order, and a gap the query must
    // not see. A wrong gather or a per-span softmax changes this answer.
    tq::reader::query_view view() const {
        tq::reader::query_view v_;
        tq::reader::span_keys a;
        a.effective_pos = 5;
        a.cells = {0, 1, 2, 3};
        tq::reader::span_keys b;
        b.effective_pos = 6;
        b.cells = {12, 7, 19, 4};
        v_.spans = {a, b};
        return v_;
    }
};

// Runs the emitter's graph and returns its output.
std::vector<float> run_graph(const board & b, const tq::reader::query_view & view,
                             std::string & why, tq::reader::emit_refusal & refusal) {
    struct ggml_init_params params{32u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);

    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    std::memcpy(k->data, b.k.data(), b.k.size() * sizeof(float));
    std::memcpy(v->data, b.v.data(), b.v.size() * sizeof(float));
    std::memcpy(q->data, b.q.data(), b.q.size() * sizeof(float));

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = k;
    in.v = v;
    in.q = q;
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = kCells;

    const tq::reader::emit_result r = tq::reader::emit_reader_attention(in, view, kScale);
    refusal = r.refusal;
    why = r.why;
    std::vector<float> out;
    if (r.out) {
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, r.out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        out.resize(std::size_t(kHeadDim) * kQHeads);
        // The node is [head_dim, 1, n_head]: element (d, h) at d + head_dim*h, which is the
        // core's out[h * head_dim + d] -- the same index. No reordering here on purpose; a
        // transpose in the test would hide one in the emitter.
        std::memcpy(out.data(), r.out->data, out.size() * sizeof(float));
    }
    ggml_free(ctx);
    return out;
}

void the_graph_matches_the_core() {
    const board b;
    const tq::reader::query_view view = b.view();

    const tq::reader::rope_fn no_rope = [](float *, std::size_t, std::int32_t) {};
    const tq::reader::attention_result core =
        tq::reader::attend(b.q.data(), b.core_buffers(), view, kScale, no_rope);
    check(core.ok, "core: attend succeeded on the view");

    std::string why;
    tq::reader::emit_refusal refusal = tq::reader::emit_refusal::ok;
    const std::vector<float> emitted = run_graph(b, view, why, refusal);
    check(refusal == tq::reader::emit_refusal::ok,
          std::string("emit: accepted the view") + (why.empty() ? "" : " (" + why + ")"));

    double worst = 0.0;
    for (std::size_t i = 0; i < emitted.size() && i < core.out.size(); ++i) {
        worst = std::fmax(worst, std::fabs(double(emitted[i]) - double(core.out[i])));
    }
    std::fprintf(stderr, "emitted graph vs core: max|diff| = %.3e over %zu values\n",
                 worst, emitted.size());
    check(worst < 1e-5,
          "the emitted nodes compute what the core computes (gather, union softmax, reduction)");

    // Anti-tautology: the values must not be a constant or a copy of one head. A graph that
    // dropped the per-head grouping, or reduced the wrong axis, still produces a number.
    double spread = 0.0;
    for (std::size_t i = 1; i < core.out.size(); ++i) {
        spread = std::fmax(spread, std::fabs(double(core.out[i]) - double(core.out[0])));
    }
    check(spread > 1e-3, "the fixture separates the heads and the query dimensions");
}

void the_union_is_one_softmax() {
    // A span appended after another must change the earlier keys' weights: that is what a
    // single normalization over the union means. If the emitter normalized per span, the
    // first span's contribution would be unchanged by the second.
    const board b;

    tq::reader::query_view one = b.view();
    one.spans.pop_back();

    std::string why;
    tq::reader::emit_refusal r1 = tq::reader::emit_refusal::ok;
    tq::reader::emit_refusal r2 = tq::reader::emit_refusal::ok;
    const std::vector<float> a = run_graph(b, one, why, r1);
    const std::vector<float> both = run_graph(b, b.view(), why, r2);
    check(r1 == tq::reader::emit_refusal::ok && r2 == tq::reader::emit_refusal::ok,
          "both views are accepted");

    double worst = 0.0;
    for (std::size_t i = 0; i < a.size() && i < both.size(); ++i) {
        worst = std::fmax(worst, std::fabs(double(a[i]) - double(both[i])));
    }
    std::fprintf(stderr, "one span vs two spans: max|diff| = %.3e\n", worst);
    check(worst > 1e-3,
          "adding a span renormalizes the first span's keys: the softmax spans the union");
}

// The rotation is a capability, so its application is observable: with a rotation that
// flips the sign of q, a per-span rotation at two different positions must produce two
// different rotations. A single rotation would give one.
ggml_tensor * negate(ggml_context * ctx, ggml_tensor * q, std::int32_t pos, void * user) {
    auto * positions = static_cast<std::vector<std::int32_t> *>(user);
    positions->push_back(pos);
    return ggml_neg(ctx, q);
}

void the_rotation_is_injected_per_span() {
    const board b;
    std::vector<std::int32_t> seen;
    std::string why;
    tq::reader::emit_refusal refusal = tq::reader::emit_refusal::ok;

    struct ggml_init_params params{32u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    std::memcpy(k->data, b.k.data(), b.k.size() * sizeof(float));
    std::memcpy(v->data, b.v.data(), b.v.size() * sizeof(float));
    std::memcpy(q->data, b.q.data(), b.q.size() * sizeof(float));

    tq::reader::graph_inputs in;
    in.ctx = ctx; in.k = k; in.v = v; in.q = q;
    in.n_head = kQHeads; in.n_head_kv = kKvHeads; in.head_dim = kHeadDim; in.n_cell = kCells;

    const tq::reader::emit_result r =
        tq::reader::emit_reader_attention(in, b.view(), kScale, &negate, &seen);
    refusal = r.refusal;
    why = r.why;
    check(refusal == tq::reader::emit_refusal::ok,
          std::string("emit with an injected rotation succeeded") + (why.empty() ? "" : " (" + why + ")"));
    check(seen.size() == b.view().spans.size(),
          "the rotation ran once per span, not once per key or once per query");
    check(seen.size() == 2 && seen[0] == 5 && seen[1] == 6,
          "each span was rotated at its own effective_pos, in view order");

    // And it changes the numbers: a rotation that was accepted but never applied would pass
    // the checks above.
    if (r.out) {
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, r.out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);
        const float * flipped = static_cast<const float *>(r.out->data);

        const tq::reader::rope_fn no_rope = [](float *, std::size_t, std::int32_t) {};
        const tq::reader::attention_result plain =
            tq::reader::attend(b.q.data(), b.core_buffers(), b.view(), kScale, no_rope);
        double worst = 0.0;
        for (std::size_t i = 0; i < plain.out.size(); ++i) {
            worst = std::fmax(worst, std::fabs(double(flipped[i]) - double(plain.out[i])));
        }
        std::fprintf(stderr, "negated rotation vs none: max|diff| = %.3e\n", worst);
        check(worst > 1e-3, "the injected rotation reaches the numbers");
    }
    ggml_free(ctx);
}

void the_refusals_are_named() {
    const board b;
    std::string why;
    tq::reader::emit_refusal refusal = tq::reader::emit_refusal::ok;

    struct ggml_init_params params{32u * 1024u * 1024u, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCells);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);

    tq::reader::graph_inputs in;
    in.ctx = ctx; in.k = k; in.v = v; in.q = q;
    in.n_head = kQHeads; in.n_head_kv = kKvHeads; in.head_dim = kHeadDim; in.n_cell = kCells;

    // A cell outside the cache must be refused, not gathered.
    {
        tq::reader::query_view bad = b.view();
        bad.spans[0].cells[1] = kCells + 3;
        const tq::reader::emit_result r = tq::reader::emit_reader_attention(in, bad, kScale);
        refusal = r.refusal;
        check(refusal == tq::reader::emit_refusal::cell_out_of_range,
              "a cell outside the cache is refused by name");
    }
    // A duplicate cell must be refused: weights would count it twice.
    {
        tq::reader::query_view bad = b.view();
        bad.spans[1].cells[0] = bad.spans[0].cells[0];
        const tq::reader::emit_result r = tq::reader::emit_reader_attention(in, bad, kScale);
        refusal = r.refusal;
        check(refusal == tq::reader::emit_refusal::duplicate_cell,
              "a cell twice in the view is refused by name");
    }
    // An empty view.
    {
        tq::reader::query_view empty;
        const tq::reader::emit_result r = tq::reader::emit_reader_attention(in, empty, kScale);
        refusal = r.refusal;
        check(refusal == tq::reader::emit_refusal::empty_view, "an empty view is refused by name");
    }
    // A head layout that cannot be grouped.
    {
        tq::reader::graph_inputs odd = in;
        odd.n_head_kv = 3;
        ggml_tensor * k3 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 3, kCells);
        odd.k = k3;
        const tq::reader::emit_result r = tq::reader::emit_reader_attention(in, b.view(), kScale);
        check(r.refusal == tq::reader::emit_refusal::ok, "the valid layout still passes");
        const tq::reader::emit_result r3 = tq::reader::emit_reader_attention(odd, b.view(), kScale);
        check(r3.refusal == tq::reader::emit_refusal::bad_head_layout ||
              r3.refusal == tq::reader::emit_refusal::shape_mismatch,
              "an ungroupable head layout is refused, not silently regrouped");
    }
    // No context at all.
    {
        tq::reader::graph_inputs none;
        const tq::reader::emit_result r = tq::reader::emit_reader_attention(none, b.view(), kScale);
        check(r.refusal == tq::reader::emit_refusal::no_context,
              "a missing graph context is refused by name");
    }
    ggml_free(ctx);
    (void) why;

    // Every refusal has words; a caller that only learns "false" cannot fix anything.
    for (int i = 0; i <= int(tq::reader::emit_refusal::graph_error); ++i) {
        check(std::strcmp(tq::reader::describe(tq::reader::emit_refusal(i)), "unknown refusal") != 0,
              "every refusal reason is described");
    }
}

} // namespace

int main() {
    the_graph_matches_the_core();
    the_union_is_one_softmax();
    the_rotation_is_injected_per_span();
    the_refusals_are_named();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "emitted reader attention matches the core\n");
    return 0;
}
