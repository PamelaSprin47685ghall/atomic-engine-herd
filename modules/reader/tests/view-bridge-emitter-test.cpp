// The whole chain: ledger -> resolve -> to_query_view -> emit_reader_attention -> numbers.
//
// Every earlier reader test starts after the index. `graph-emit-*` builds a `query_view` by hand;
// `index-ledger` stops at `resolved_view`. So the one thing a product needs — "the cells the ledger
// resolved are the cells the graph read" — was never executed end to end. This test executes it.
//
// The reference is written here in double, from the fixture's own floats, following the emitter's
// documented arithmetic rather than its code:
//
//   scores[s] = kq_scale * sum_d q[d, 0, h] * k[d, kvh, cell]      over every visible cell, in view
//                                                                  order and across spans (the
//                                                                  emitter concatenates spans into
//                                                                  ONE softmax)
//   w         = softmax(scores)                                    over that whole set
//   out[d + HD*h] = sum_i w[i] * v[d, kvh, cell_i]
//
// with `kvh = h / (n_head / n_head_kv)` — the GQA grouping the emitter performs through ggml's own
// broadcast. The rope is `nullptr` (identity), so `effective_pos` is not observable in the numbers;
// what this test measures is that the CELLS reaching the graph are the cells the ledger named, which
// is the half no prior test covered. Dropping a cell, or naming a different one, moves the softmax
// denominator and is caught here.
//
// F32 k/v on purpose: the emitter widens F16 exactly, so an F32 cache removes quantisation from the
// comparison and leaves layout and index as the only things that can disagree.
#include <tq/reader/graph_emit.h>
#include <tq/reader/view_bridge.h>

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

constexpr int kHeadDim = 8;
constexpr int kKvHeads = 2;
constexpr int kQHeads  = 4;
constexpr int kCell    = 16;
constexpr float kScale = 0.70710678f;

struct fixture {
    // k, v : [head_dim, n_head_kv, n_cell]  idx = d + HD*(kvh + KVH*cell)
    // q    : [head_dim, 1, n_head]         idx = d + HD*h
    std::vector<float> k, v, q;
    tq::reader::ledger board;
    tq::reader::query_view view;
    std::vector<int> visible_cells;   // the cells the view names, in view order

    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        unsigned state = 0x9e3779b9u;
        auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return float((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();

        board.set_n_stream(1);
        board.set_n_sequences(2);
        board.set_capacity(kCell);
        // Three past cells in sequence 0 at slots 4, 9, 12 — slots chosen NOT in position order, so a
        // conversion that sorted or assumed contiguity would read a different cell.
        board.put({0, 0}, 4);
        board.put({0, 1}, 9);
        board.put({0, 2}, 12);
        // One cell of another sequence, presented to the rotation at its own virtual slot: it must
        // land in a span of its own, because its effective position differs.
        board.put({1, 7}, 1);

        const std::vector<tq::reader::view_entry> entries{
            {{0, 0}, 0},   // effective 4 + 0 - 0 = 4
            {{0, 1}, 1},   // effective 4 + 1 - 1 = 4
            {{0, 2}, 2},   // effective 4 + 2 - 2 = 4
            {{1, 7}, 3},   // effective 4 + 7 - 3 = 8 -> new span
            {{0, 3}, 4},   // the query itself: no slot yet, dropped
        };
        tq::reader::attention_request req;
        req.query_seq = 0;
        req.query_pos = 3;
        req.query_virtual_pos = 4;
        req.entries = entries.data();
        req.n_entries = entries.size();
        const auto answer = board.resolve(&req, 1);
        check(answer.refusal == tq::reader::resolve_refusal::ok,
              "resolve: " + std::string(tq::reader::describe(answer.refusal)) + " " + answer.why);
        check(answer.views.size() == 1, "one view");
        if (answer.views.size() == 1) {
            view = tq::reader::to_query_view(answer.views[0]);
        }
        for (const auto & span : view.spans) {
            for (std::int32_t cell : span.cells) visible_cells.push_back(cell);
        }
    }
};

// The emitter's arithmetic, in double, from the fixture.
std::vector<double> reference(const fixture & f, std::size_t n_cells) {
    std::vector<double> out(std::size_t(kHeadDim) * kQHeads, 0.0);

    const int group = kQHeads / kKvHeads;
    for (int h = 0; h < kQHeads; ++h) {
        const int kvh = h / group;

        std::vector<double> scores(n_cells, 0.0);
        for (std::size_t i = 0; i < n_cells; ++i) {
            const int cell = f.visible_cells[i];
            double dot = 0.0;
            for (int d = 0; d < kHeadDim; ++d) {
                dot += double(f.q[std::size_t(d) + kHeadDim * h]) *
                       double(f.k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * cell)]);
            }
            scores[i] = double(kScale) * dot;
        }

        // One softmax over the union — the emitter's `ggml_soft_max` sees every span's scores at once.
        double max_score = scores.empty() ? 0.0 : scores[0];
        for (double s : scores) max_score = std::max(max_score, s);
        double total = 0.0;
        for (double & s : scores) { s = std::exp(s - max_score); total += s; }

        for (std::size_t i = 0; i < n_cells; ++i) {
            const int cell = f.visible_cells[i];
            const double w = scores[i] / total;
            for (int d = 0; d < kHeadDim; ++d) {
                out[std::size_t(d) + kHeadDim * h] +=
                    w * double(f.v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * cell)]);
            }
        }
    }
    return out;
}

std::vector<double> emitted(fixture & f, std::string & why, std::size_t * keys_seen) {
    std::vector<double> out;
    ggml_context * ctx = ggml_init({/*mem_size=*/32u * 1024u * 1024u, /*mem_buffer=*/nullptr,
                                    /*no_alloc=*/false});
    if (!ctx) { why = "ggml_init failed"; return out; }

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
    ggml_set_input(in.k);
    ggml_set_input(in.v);
    ggml_set_input(in.q);
    std::memcpy(in.k->data, f.k.data(), f.k.size() * sizeof(float));
    std::memcpy(in.v->data, f.v.data(), f.v.size() * sizeof(float));
    std::memcpy(in.q->data, f.q.data(), f.q.size() * sizeof(float));

    const auto res = tq::reader::emit_reader_attention(in, f.view, kScale, nullptr, nullptr);
    if (res.refusal != tq::reader::emit_refusal::ok) {
        why = std::string(tq::reader::describe(res.refusal)) + " -- " + res.why;
        ggml_free(ctx);
        return out;
    }
    if (keys_seen) *keys_seen = res.n_visible;

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, res.out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 1) != GGML_STATUS_SUCCESS) {
        why = "graph compute failed";
        ggml_free(ctx);
        return out;
    }
    const float * p = static_cast<const float *>(res.out->data);
    out.assign(p, p + std::size_t(kHeadDim) * kQHeads);
    why = "ok";
    ggml_free(ctx);
    return out;
}

}  // namespace

int main() {
    fixture f;

    check(f.view.spans.size() == 2,
          "the ledger's runs become two spans (three cells at effective 4, one at 8); got " +
              std::to_string(f.view.spans.size()));
    check(f.visible_cells == std::vector<int>({4, 9, 12, 1}),
          "cells reach the graph in ledger order [4, 9, 12, 1]; a sorted view would read [1, 4, 9, 12]");

    std::string why;
    std::size_t keys_seen = 0;
    const std::vector<double> got = emitted(f, why, &keys_seen);
    check(!got.empty(), "emission produced no output: " + why);
    if (got.empty()) {
        std::fprintf(stderr, "ledger->emitter: %d failure(s)\n", failures);
        return 1;
    }

    check(keys_seen == f.visible_cells.size(),
          "the softmax saw " + std::to_string(keys_seen) + " keys, the view names " +
              std::to_string(f.visible_cells.size()));

    // The reference follows the CELLS THE VIEW NAMES, and that is the whole point: if the emitter
    // read a different set, the two sides normalise over different denominators and disagree.
    const std::vector<double> want = reference(f, f.visible_cells.size());
    double max_abs = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(got[i] - want[i]));
    }
    check(max_abs < 1e-5, "ledger->emitter vs double reference: max|diff| = " +
                              std::to_string(max_abs));

    // NON-VACUITY, three ways. A tiny diff means nothing if both sides are zero, if the heads are
    // indistinguishable, or if the output ignores which cells were read at all.
    double magnitude = 0.0;
    for (double x : want) magnitude = std::max(magnitude, std::fabs(x));
    check(magnitude > 1e-3, "reference is degenerate: max |value| = " + std::to_string(magnitude));

    double head_spread = 0.0;
    for (int h = 1; h < kQHeads; ++h) {
        for (int d = 0; d < kHeadDim; ++d) {
            head_spread = std::max(head_spread,
                                   std::fabs(want[std::size_t(d) + kHeadDim * h] -
                                             want[std::size_t(d) + kHeadDim * 0]));
        }
    }
    check(head_spread > 1e-3,
          "all heads agree; the comparison cannot detect a wrong GQA grouping (" +
              std::to_string(head_spread) + ")");

    std::fprintf(stderr,
                 "ledger->emitter: %zu spans, %zu cells, keys seen %zu, max|diff| = %.3e, "
                 "|ref|max = %.3e, head spread = %.3e\n",
                 f.view.spans.size(), f.visible_cells.size(), keys_seen, max_abs, magnitude,
                 head_spread);

    if (failures == 0) {
        std::fprintf(stderr, "ledger->emitter: OK\n");
        return 0;
    }
    std::fprintf(stderr, "ledger->emitter: %d failure(s)\n", failures);
    return 1;
}
