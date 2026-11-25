// The estimator against the emitter, on a sweep of shapes, with the check the header promises.
//
// This file exists to hunt a counterexample. The prompt it answers is not "does the estimate look
// reasonable" but "is there ANY shape where it falls below the measured cost" -- because that is
// the only failure mode that matters, and one shape is enough.
//
// The sweep is deliberately unbalanced. A bound written against one convenient fixture can be
// accidentally safe; a bound that survives the four regimes below is a different claim:
//
//   data-dominated     large head_dim and cells (large per-node payload, few nodes)
//   object-dominated   head_dim 1, one cell per span, many heads (tiny payload, many nodes)
//   quadratic          many spans of many cells (the prefix-sum concatenation chain, n^2)
//   span-explosion     32 spans of one cell (the per-span fixed overhead)
//
// Every case also runs `estimate_drift()`, which is the POST-HOC regression check a caller is told
// to use -- not a gate. A gate is impossible here: an over-full arena aborts inside ggml
// (ggml.c:1832, GGML_ASSERT in every build) before any check can run, so a check that follows the
// emission cannot observe the one failure that matters. Its negative directions are asserted
// separately at the end, because a check that always answers "within" is worse than no check: it
// would certify a drift it cannot see.
//
// A shape that is not measured -- skipped by the arena cap, or failed to allocate its arena -- is
// COUNTED and the count must be zero. Returning a sentinel ratio was how an earlier revision let
// every case be skipped while still reporting "no counterexample".
#include <tq/reader/attention.h>
#include <tq/reader/arena_budget.h>
#include <tq/reader/graph_emit.h>

#include <ggml.h>
#include <ggml-cpu.h>

#include <cstdio>
#include <cstring>
#include <limits>
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

struct shape {
    int head_dim;
    int n_head_kv;
    int n_head;
    int cells_per_span;
    int spans;
    int n_cell;      // the cache's extent; must cover spans * cells_per_span
    bool f16;
};

// A cache in its own context, filled through ggml so an F16 cache is rounded by ggml.
ggml_tensor * make_cache(ggml_context * ctx, ggml_type type, int head_dim, int n_head_kv, int n_cell) {
    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, type, head_dim, n_head_kv, n_cell);
    ggml_tensor * source = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, n_head_kv, n_cell);
    std::memset(source->data, 0x30, std::size_t(head_dim) * n_head_kv * n_cell * sizeof(float));
    ggml_tensor * cast = ggml_cpy(ctx, source, tensor);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cast);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    return tensor;
}

// One measurement. Returns estimate/measured, and asserts the two properties that matter.
// `skipped` is an OUTPUT, and it is the difference between a test and a demonstration.
//
// The previous version returned 1e9 for a shape it did not measure, and `worst_ratio` starts at
// 1e9, so `ratio < worst_ratio` was false and the shape left no trace. A run in which EVERY shape
// was skipped would still print "no counterexample" and exit 0. That is the exact failure this file
// exists to prevent, reproduced inside the file: a verdict that survives having measured nothing.
double measure(const shape & s, double * worst_ratio, long long * tightest_margin, int * skipped) {
    ggml_context * cache_ctx = ggml_init({256u << 20, nullptr, false});
    const ggml_type cache_type = s.f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * k = make_cache(cache_ctx, cache_type, s.head_dim, s.n_head_kv, s.n_cell);
    ggml_tensor * v = make_cache(cache_ctx, cache_type, s.head_dim, s.n_head_kv, s.n_cell);

    // THE ARENA IS SIZED FROM THE ESTIMATE, and that is the opposite of circular: it is how a
    // drifted-low estimate becomes a loud failure instead of a comfortable pass.
    //
    // The previous version allocated 8 GiB (8ull << 30). Two problems, both real:
    //
    //   * ggml_init with a non-allocating context mallocs the whole buffer immediately, so an
    //     8 GiB request is an 8 GiB demand on a machine that may have two; the sweep would fail
    //     for a reason unrelated to the estimator (and on Linux with overcommit it may instead
    //     succeed and then be killed while touching pages).
    //   * it hid the estimator's only serious failure. With an arena that large, ANY estimate --
    //     including one ten times too small -- fits, so the case could never produce a
    //     counterexample. The measurement was of the emitter, never of the claim.
    //
    // Now the arena is 2x the estimate, because:
    //     * 2x leaves room for the emission to FINISH when the estimate is somewhat low, so
    //       estimate_drift() gets to report `exceeded` instead of the process dying first;
    //     * a drift larger than 2x aborts at ggml.c:1832 -- which is the counterexample, loudly,
    //       and in a test rather than in a decoder.
    // The cap keeps one shape from claiming the machine, and a shape whose estimate exceeds it is
    // reported and skipped BY NAME rather than silently omitted.
    // The estimate is computed HERE, before the arena exists, because the arena's size comes from
    // it. Every input to it is a property of the shape, so nothing here needs a context -- which is
    // what makes it legitimate to size the arena from a number the arena has not yet constrained.
    tq::reader::budget_inputs sizing;
    std::vector<std::size_t> sizing_cells(std::size_t(s.spans), std::size_t(s.cells_per_span));
    sizing.head_dim = std::size_t(s.head_dim);
    sizing.n_head_kv = std::size_t(s.n_head_kv);
    sizing.n_cell = std::size_t(s.n_cell);
    sizing.n_head = std::size_t(s.n_head);
    sizing.cells_per_span = sizing_cells.data();
    sizing.n_spans = sizing_cells.size();
    sizing.k_elem = s.f16 ? 2 : 4;
    sizing.v_elem = s.f16 ? 2 : 4;
    sizing.q_elem = 4;
    sizing.widen_k = s.f16;
    sizing.widen_v = s.f16;
    sizing.widen_q = false;
    sizing.no_alloc = false;
    sizing.tensor_overhead = ggml_tensor_overhead();
    const auto budget = tq::reader::estimate(sizing);

    constexpr std::size_t kArenaCap = 512u << 20;
    std::size_t arena_bytes = kArenaCap;
    if (budget.status == tq::reader::budget_status::ok && budget.total_bytes > 0) {
        const std::size_t doubled = budget.total_bytes > kArenaCap / 2 ? kArenaCap
                                                                      : budget.total_bytes * 2;
        arena_bytes = doubled < (1u << 20) ? (1u << 20) : doubled;
    }
    if (budget.status == tq::reader::budget_status::ok && budget.total_bytes > kArenaCap / 2) {
        std::fprintf(stderr,
                     "%-3s hd=%3d kvh=%2d nh=%2d sp=%2d cps=%3d nc=%5d | SKIPPED: the estimate "
                     "(%zu bytes) exceeds half the %zu-byte arena cap, so this shape is not "
                     "measured here\n",
                     s.f16 ? "F16" : "F32", s.head_dim, s.n_head_kv, s.n_head, s.spans,
                     s.cells_per_span, s.n_cell, budget.total_bytes, kArenaCap);
        ++*skipped;
        ggml_free(cache_ctx);
        return 1e9;
    }

    // Checked, because ggml_init with a non-allocating context mallocs the whole buffer and returns
    // null when that fails. The unchecked version dereferenced the null on the next line
    // (ggml_new_tensor_3d), which in a build with assertions becomes an abort blamed on the
    // estimator. On this host 2.3 GiB was available while a 512 MiB cap plus the cache contexts
    // compete for it -- a failure here would have looked like a defect in the code under test.
    ggml_context * ctx = ggml_init({arena_bytes, nullptr, false});
    if (!ctx) {
        std::fprintf(stderr,
                     "%-3s hd=%3d kvh=%2d nh=%2d sp=%2d nc=%5d | FAILED: ggml_init could not "
                     "allocate the %zu-byte arena; this shape measured nothing\n",
                     s.f16 ? "F16" : "F32", s.head_dim, s.n_head_kv, s.n_head, s.spans, s.n_cell,
                     arena_bytes);
        ++*skipped;
        ggml_free(cache_ctx);
        return 1e9;
    }

    tq::reader::graph_inputs in;
    in.ctx = ctx;
    in.k = k;
    in.v = v;
    in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.head_dim, 1, s.n_head);
    std::memset(in.q->data, 0x20, std::size_t(s.head_dim) * s.n_head * sizeof(float));
    in.n_head = std::size_t(s.n_head);
    in.n_head_kv = std::size_t(s.n_head_kv);
    in.head_dim = std::size_t(s.head_dim);
    in.n_cell = std::size_t(s.n_cell);

    // Globally distinct cells: the emitter refuses a view that names one twice, and a fixture that
    // restarts at zero per span would be refused for that reason rather than measured.
    tq::reader::query_view view;
    std::vector<std::size_t> counts;
    std::int32_t next_cell = 0;
    for (int sp = 0; sp < s.spans; ++sp) {
        tq::reader::span_keys span;
        span.effective_pos = sp;
        for (int i = 0; i < s.cells_per_span; ++i) {
            span.cells.push_back(next_cell++);
        }
        view.spans.push_back(std::move(span));
        counts.push_back(std::size_t(s.cells_per_span));
    }

    // (The budget was computed above, before the arena existed, so the sizing decision and the
    // drift check below are about the same number rather than two independently derived ones.)

    const std::size_t before = ggml_used_mem(ctx);
    const auto emitted = tq::reader::emit_reader_attention(in, view, 0.70710678f, nullptr);
    const std::size_t after = ggml_used_mem(ctx);

    check(budget.status == tq::reader::budget_status::ok,
          "the shapes describe an emission, so the estimate is usable");
    // A refused emission means this case measured nothing. It is a failure here, not a footnote:
    // every fixture error this file's forerunners hit (a cache extent that did not match n_cell,
    // duplicate cells across spans) would otherwise have passed silently.
    check(emitted.refusal == tq::reader::emit_refusal::ok,
          "the emission ran, so the measurement exists");

    const long long measured = static_cast<long long>(after - before);
    const long long estimate = static_cast<long long>(budget.total_bytes);
    check(measured >= 0, "arena usage did not go backwards");

    // The post-hoc check the header tells callers to use. It must agree with what is asserted
    // here; if it did not, the advertised check would be a different property from the measured
    // one -- and this file would be certifying something it never tested.
    //
    // Note what it is NOT: a gate. `kEstimateAdmitsEmission` is false because an over-full arena
    // aborts inside ggml (ggml.c:1832) before any of this runs. Reaching this line at all means the
    // emission survived, so the only question it can answer is whether the inventory drifted.
    check(tq::reader::estimate_drift(budget, before, after) == tq::reader::drift::within,
          "estimate_drift() agrees the emission fit inside the estimate");

    const double ratio = measured > 0 ? double(estimate) / double(measured) : 0.0;
    if (ratio < *worst_ratio) *worst_ratio = ratio;
    const long long margin = estimate - measured;
    if (margin < *tightest_margin) *tightest_margin = margin;

    std::fprintf(stderr,
                 "%-3s hd=%3d kvh=%2d nh=%2d sp=%2d cps=%3d nc=%5d | est %10lld meas %10lld "
                 "margin %+9lld ratio %.3f\n",
                 s.f16 ? "F16" : "F32", s.head_dim, s.n_head_kv, s.n_head, s.spans,
                 s.cells_per_span, s.n_cell, estimate, measured, margin, ratio);

    ggml_free(ctx);
    ggml_free(cache_ctx);
    return ratio;
}

} // namespace

int main() {
    double worst_ratio = 1e9;
    // Seeded at the maximum, not at zero.
    //
    // The previous initialisation was `0`, and `if (margin < *tightest_margin)` can never fire when
    // every margin is positive -- so the reported "tightest margin" was +0 bytes no matter what the
    // sweep measured. The number is printed and read by whoever maintains the estimate; a statistic
    // that is silently the initialiser is worse than no statistic, because it reads as "the bound
    // has no headroom left anywhere", which is the opposite of what the run showed.
    long long tightest_margin = std::numeric_limits<long long>::max();
    int skipped = 0;

    const shape sweep[] = {
        // data-dominated
        {64, 8, 16, 32, 2, 70, false},
        {128, 8, 16, 64, 2, 134, false},
        {128, 8, 16, 256, 2, 2300, false},
        // object-dominated: one cell per span, one-element heads, many of both
        {1, 1, 1, 1, 1, 4, false},
        {1, 8, 8, 1, 16, 24, false},
        {1, 16, 16, 1, 16, 24, false},
        {1, 16, 16, 1, 40, 48, false},
        // quadratic: many spans of many cells
        {8, 2, 4, 32, 8, 300, false},
        {8, 2, 4, 32, 16, 560, false},
        {8, 2, 4, 16, 32, 560, false},
        // span explosion: per-span fixed overhead at its extreme
        {8, 2, 4, 1, 32, 40, false},
        // head explosion: per-head setup at its extreme
        {1, 32, 64, 2, 1, 8, false},
        // F16, where the widening terms appear
        {128, 8, 16, 64, 2, 134, true},
        {8, 2, 4, 16, 32, 560, true},
        {1, 16, 16, 1, 32, 40, true},
    };

    for (const shape & s : sweep) {
        measure(s, &worst_ratio, &tightest_margin, &skipped);
    }

    // A skipped shape is a shape that measured nothing, and this file's verdict is only as wide as
    // the shapes it actually ran. Asserted before the ratio checks so a run that skipped everything
    // fails on the truthful reason rather than on a coincidental sentinel.
    check(skipped == 0,
          "every shape was measured: a skipped shape contributes no evidence, and enough skips "
          "make 'no counterexample' mean nothing");

    std::fprintf(stderr,
                 "sweep of %zu shapes: worst ratio %.3f, tightest margin %+lld bytes\n",
                 sizeof(sweep) / sizeof(sweep[0]), worst_ratio, tightest_margin);
    check(worst_ratio >= 1.0,
          "every shape stayed inside the estimate: a ratio below 1.0 is a counterexample, and one "
          "is enough to make the bound false");

    // ---- estimate_drift's negative directions ----
    //
    // A predicate that always answers "within" would certify exactly the drift it exists to detect,
    // so each way of being wrong is exercised. Note what these cases are NOT: none of them makes an
    // emission safe. They check the accounting of an emission that already finished. The scope of
    // this function is stated once, in the header, by kEstimateAdmitsEmission == false.
    {
        tq::reader::arena_budget ok;
        ok.status = tq::reader::budget_status::ok;
        ok.total_bytes = 1000;

        check(tq::reader::estimate_drift(ok, 0, 1000) == tq::reader::drift::within,
              "exactly the estimate fits");
        check(tq::reader::estimate_drift(ok, 0, 999) == tq::reader::drift::within,
              "less than the estimate fits");
        check(tq::reader::estimate_drift(ok, 0, 1001) == tq::reader::drift::exceeded,
              "one byte over is EXCEEDED, not within: an off-by-one here certifies a drift");

        // A shrinking context: the subtraction would wrap to a huge number and pass, so it is
        // refused by name rather than by arithmetic. `unusable`, not `exceeded` -- nothing was
        // measured, and reporting a verdict would invent one.
        check(tq::reader::estimate_drift(ok, 500, 400) == tq::reader::drift::unusable,
              "a decreasing used_mem is UNUSABLE, not read as a negative cost");

        // A non-ok status carries no number to compare against; SIZE_MAX especially must not read
        // as generous, because saturation means "do not emit".
        tq::reader::arena_budget saturated;
        saturated.status = tq::reader::budget_status::overflow;
        saturated.total_bytes = SIZE_MAX;
        check(tq::reader::estimate_drift(saturated, 0, 1) == tq::reader::drift::unusable,
              "a saturated budget is UNUSABLE, not a large allowance");

        tq::reader::arena_budget unsupported;
        unsupported.status = tq::reader::budget_status::unsupported;
        check(tq::reader::estimate_drift(unsupported, 0, 0) == tq::reader::drift::unusable,
              "an unsupported budget that computed nothing certifies nothing");

        // The distinction the enum exists for: a caller reading a bool could not tell "the estimate
        // was too small" (fix this file) from "there is nothing to compare" (fix the caller).
        check(tq::reader::estimate_drift(ok, 0, 1001) != tq::reader::estimate_drift(ok, 500, 400),
              "exceeded and unusable are different answers, so a caller can act on them differently");

        // And the admission constant, asserted so a future edit that flips it fails here.
        check(!tq::reader::kEstimateAdmitsEmission,
              "the estimate is not an admission gate: emission failures ABORT inside ggml "
              "(ggml.c:1832), so no post-hoc check can observe them");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "arena budget: no counterexample across the sweep, every shape measured, and "
                 "estimate_drift() refuses every way of being meaningless\n");
    return 0;
}
