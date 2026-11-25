// The emission's arena cost: estimated before, measured after, compared.
//
// The property under test is one inequality, on real runs of the real emitter:
//
//     estimate(inputs) >= ggml_used_mem(after) - ggml_used_mem(before)
//
// An inequality is only worth something if it is also TIGHT enough to be useful, so every case
// prints the ratio and the worst one is reported at the end. A budget that is ten times the
// measurement is safe and useless; the numbers make that visible instead of arguable.
//
// Two regimes, deliberately separate:
//
//   allocating  (!no_alloc)  the emitter runs, writes its index tensors, and materialises every
//                            intermediate. This is where the inequality is proved, because it is
//                            where the data actually exists.
//   no_alloc                 the emitter REFUSES -- see the last case, which is the finding this
//                            file was built to establish rather than assume: `idx->data` is null
//                            under no_alloc, and the emitter requires it.
//
// That refusal is why the no_alloc branch of the estimate cannot be validated end to end yet, and
// the file says so where it matters rather than implying coverage it does not have.
#include <tq/reader/attention.h>
#include <tq/reader/arena_budget.h>
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

constexpr int kQHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 8;

// A cache as one flat buffer, in the emitter's declared layout, filled through ggml so an F16
// cache is rounded by ggml rather than by a second implementation of the rounding rule.
ggml_tensor * make_cache(ggml_context * ctx, ggml_type type, int cells) {
    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, type, kHeadDim, kKvHeads, cells);
    ggml_tensor * source = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, cells);
    std::vector<float> values(std::size_t(kHeadDim) * kKvHeads * cells, 0.0f);
    unsigned state = 1234567u;
    for (float & value : values) {
        state = state * 1664525u + 1013904223u;
        value = float((state >> 9) & 0xFFFF) / 65536.0f - 0.5f;
    }
    std::memcpy(source->data, values.data(), values.size() * sizeof(float));
    ggml_tensor * cast = ggml_cpy(ctx, source, tensor);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, cast);
    ggml_graph_compute_with_ctx(ctx, graph, 1);
    return tensor;
}

// One measured case. Returns the ratio measurement/estimate (<= 1 when the bound holds).
double run_case(ggml_type cache_type, const std::vector<int> & cells_per_span,
                bool no_alloc, const char * label) {
    const int total_cells = [&] {
        int sum = 0;
        for (int c : cells_per_span) sum += c;
        return sum;
    }();

    // The cache lives in its own allocating context: it must be readable, and it is not part of
    // the graph whose cost is being measured.
    ggml_context * cache_ctx = ggml_init({16u * 1024u * 1024u, nullptr, false});

    // The cache is LARGER than the cells the view names, which is the real situation: a model's
    // cache holds every position, and a view names the keys one query may attend to. The emitter's
    // `n_cell` is the cache's extent (it checks k->ne[2] against it), so the spare cells exist
    // but are never gathered -- and a budget that assumed a full-cache gather would be measuring a
    // different operation.
    const int cache_cells = total_cells + 4;

    tq::reader::graph_inputs in;
    in.ctx = nullptr;   // filled below, per regime
    in.k = make_cache(cache_ctx, cache_type, cache_cells);
    in.v = make_cache(cache_ctx, cache_type, cache_cells);
    in.n_head = kQHeads;
    in.n_head_kv = kKvHeads;
    in.head_dim = kHeadDim;
    in.n_cell = std::size_t(cache_cells);

    // The graph context under measurement: a fresh one per case, so the delta is this emission's.
    ggml_init_params graph_params{64u * 1024u * 1024u, nullptr, no_alloc};
    ggml_context * graph_ctx = ggml_init(graph_params);
    in.ctx = graph_ctx;

    in.q = ggml_new_tensor_3d(graph_ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
    std::vector<float> q(std::size_t(kHeadDim) * kQHeads, 0.25f);
    std::memcpy(in.q->data, q.data(), q.size() * sizeof(float));

    tq::reader::query_view view;
    std::int32_t next_pos = 3;
    std::int32_t next_cell = 0;
    for (int c : cells_per_span) {
        tq::reader::span_keys span;
        span.effective_pos = next_pos++;
        span.cells.reserve(std::size_t(c));
        // Cells are GLOBALLY distinct across spans. The emitter refuses a view that names the same
        // cell twice -- the softmax would weight it twice -- and a fixture whose spans both start
        // at zero is refused for that reason rather than for anything about the budget. `cache_cells`
        // is total_cells + 4, so the spare capacity also keeps every index inside the cache.
        for (int i = 0; i < c; ++i) {
            span.cells.push_back(next_cell++);
        }
        view.spans.push_back(std::move(span));
    }

    tq::reader::budget_inputs bi;
    bi.head_dim = kHeadDim;
    bi.n_head_kv = kKvHeads;
    bi.n_cell = std::size_t(cache_cells);
    bi.n_head = kQHeads;
    // The span counts live in a local vector and are handed over as a pointer: the estimator no
    // longer owns a fixed array, so there is nothing to truncate at eight.
    std::vector<std::size_t> span_cells;
    span_cells.reserve(cells_per_span.size());
    for (int c : cells_per_span) {
        span_cells.push_back(std::size_t(c));
    }

    bi.cells_per_span = span_cells.data();
    bi.n_spans = span_cells.size();
    bi.k_elem = ggml_type_size(cache_type);
    bi.v_elem = ggml_type_size(cache_type);
    bi.q_elem = ggml_type_size(GGML_TYPE_F32);
    bi.widen_k = cache_type != GGML_TYPE_F32;
    bi.widen_v = cache_type != GGML_TYPE_F32;
    bi.widen_q = false;
    bi.no_alloc = no_alloc;
    bi.tensor_overhead = ggml_tensor_overhead();
    const tq::reader::arena_budget budget = tq::reader::estimate(bi);

    const std::size_t before = ggml_used_mem(graph_ctx);
    const auto emitted = tq::reader::emit_reader_attention(in, view, 0.70710678f, nullptr);
    const std::size_t after = ggml_used_mem(graph_ctx);
    const std::size_t delta = after - before;

    double ratio = 0.0;
    check(budget.status == tq::reader::budget_status::ok,
          std::string(label) + ": the inputs describe an emission, so the estimate is a bound");
    if (emitted.refusal == tq::reader::emit_refusal::ok) {
        ratio = budget.total_bytes == 0 ? 0.0
                                        : double(delta) / double(budget.total_bytes);
        std::fprintf(stderr,
                     "%-34s objects>=%zu data>=%zu estimate=%zu measured=%zu ratio=%.2f\n",
                     label, budget.objects, budget.data_bytes, budget.total_bytes, delta, ratio);
        check(budget.total_bytes >= delta,
              std::string(label) +
                  ": the estimate covers the measured arena growth (a bound below the "
                  "measurement is a defect in the estimate, not slack in the model)");
    } else {
        // A refusal is a FAILED measurement, not a footnote.
        //
        // This branch used to print and return, so every shape that the emitter declined passed the
        // file silently: the ratio stayed 0, `worst` never rose, and the inequality was never
        // evaluated. A test whose cases can all be skipped by the code under test proves nothing
        // about it -- and the two fixtures that were refused earlier in this file's life (a cache
        // extent that did not match `n_cell`, then duplicate cells across spans) would have been
        // invisible here rather than fixed.
        std::fprintf(stderr, "%-34s REFUSED: %s (%s)\n", label,
                     tq::reader::describe(emitted.refusal), emitted.why.c_str());
        check(false,
              std::string(label) +
                  ": the emission ran, so the measurement exists -- a refused emission means this "
                  "case measured nothing");
    }

    ggml_free(graph_ctx);
    ggml_free(cache_ctx);
    return ratio;
}

} // namespace

int main() {
    double worst = 0.0;
    const auto note = [&worst](double r) { if (r > worst) worst = r; };

    // The shape matrix: the two cache types a model here can have, one and two spans, and a
    // single long span. Deliberately not "one convenient configuration" -- a bound that only
    // holds for the fixture it was written against is not a bound.
    note(run_case(GGML_TYPE_F32, {6}, false, "f32, 1 span"));
    note(run_case(GGML_TYPE_F32, {6, 5}, false, "f32, 2 spans (unequal)"));
    note(run_case(GGML_TYPE_F32, {64}, false, "f32, 1 long span (64 cells)"));
    note(run_case(GGML_TYPE_F16, {6}, false, "f16, 1 span"));
    note(run_case(GGML_TYPE_F16, {6, 5}, false, "f16, 2 spans (unequal)"));
    note(run_case(GGML_TYPE_F16, {64}, false, "f16, 1 long span (64 cells)"));

    // Sixteen spans, and nine: the second is one past the old fixed array, so before this round
    // it was costed as if it had eight entries -- a bound below the real cost. The emitter has no
    // span limit, so the estimator must not invent one. The quadratic union chain makes these the
    // cases where a linear charge would show up.
    note(run_case(GGML_TYPE_F32, {4, 4, 4, 4, 4, 4, 4, 4, 4}, false, "f32, 9 spans"));
    note(run_case(GGML_TYPE_F16, {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}, false,
                  "f16, 12 spans"));

    check(worst <= 1.0, "no case measured more than the estimate");
    std::fprintf(stderr, "worst measured/estimated ratio: %.2f (1.00 would mean no margin)\n",
                 worst);

    // A tighter bound is a better bound: if the estimate were 10x the measurement it would still
    // pass every check above and be useless in a real arena.
    check(worst > 0.25,
          "the estimate is not wildly loose: a bound more than 4x the measurement would waste "
          "more arena than it protects");

    // The empty case: nothing describable means "do not emit", never "free".
    {
        tq::reader::budget_inputs empty;
        empty.tensor_overhead = ggml_tensor_overhead();
        const auto b = tq::reader::estimate(empty);
        check(b.total_bytes == 0 && b.objects == 0 && b.data_bytes == 0,
              "an undescribable emission yields a zero budget, which a caller must read as "
              "'do not emit'");
    }

    // The no_alloc branch, checked the only way it can be while the emitter refuses there.
    //
    // The emission cannot run under no_alloc (next case proves it), so no measurement can confirm
    // what it WOULD have cost. What is checkable is that the estimator honours the flag at all:
    // for identical inputs, the no_alloc budget must be strictly smaller, and by exactly the data
    // term. A first version of this file did not check it, and a mutant that ignored `no_alloc`
    // passed every case -- because it charged data the measurement never contained, which is
    // conservative and therefore invisible to a one-sided bound. A one-sided bound cannot detect
    // excess; this comparison can.
    {
        tq::reader::budget_inputs bi;
        bi.head_dim = kHeadDim;
        bi.n_head_kv = kKvHeads;
        bi.n_cell = 32;
        bi.n_head = kQHeads;
        const std::size_t one_span[1] = {6};
        bi.cells_per_span = one_span;
        bi.n_spans = 1;
        bi.k_elem = ggml_type_size(GGML_TYPE_F16);
        bi.v_elem = ggml_type_size(GGML_TYPE_F16);
        bi.q_elem = ggml_type_size(GGML_TYPE_F32);
        bi.widen_k = true;
        bi.widen_v = true;
        bi.tensor_overhead = ggml_tensor_overhead();

        bi.no_alloc = false;
        const auto allocating = tq::reader::estimate(bi);
        bi.no_alloc = true;
        const auto reserving = tq::reader::estimate(bi);

        check(allocating.data_bytes > 0, "the allocating budget counts data");
        check(reserving.data_bytes == 0,
              "the no_alloc budget counts no data: ggml.c:1825 allocates none when no_alloc is set");
        check(reserving.total_bytes < allocating.total_bytes,
              "so the two regimes give different answers -- an estimator that ignored the flag "
              "would charge data the arena never holds, which a one-sided bound cannot detect");
        check(reserving.objects == allocating.objects,
              "while the object count is the same: no_alloc changes what is stored, not what is "
              "created");
        std::fprintf(stderr, "regimes: allocating=%zu (data %zu), no_alloc=%zu (data 0)\n",
                     allocating.total_bytes, allocating.data_bytes, reserving.total_bytes);
    }

    // ---- the finding this file was built to establish ----
    //
    // A no_alloc context allocates no data (ggml.c:1825), so `ggml_new_tensor_1d(...)->data` is
    // null, and the emitter writes its span indices into exactly that pointer. It therefore
    // refuses on any no_alloc graph -- which is the regime llama.cpp's own graph builds in.
    //
    // This is not a budget defect; it is why the emission cannot be wired into a product yet, and
    // it is asserted here so the claim is a measurement rather than a reading of the code.
    {
        ggml_init_params params{8u * 1024u * 1024u, nullptr, /*no_alloc=*/true};
        ggml_context * ctx = ggml_init(params);
        check(ggml_get_no_alloc(ctx), "fixture: the context really is no_alloc");

        ggml_tensor * probe = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);
        check(probe != nullptr && probe->data == nullptr,
              "fixture: under no_alloc a new tensor carries no data, which is what the emitter "
              "writes its indices into");

        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, 4);
        in.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, 4);
        in.q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        in.n_head = kQHeads;
        in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim;
        in.n_cell = 4;

        tq::reader::query_view view;
        tq::reader::span_keys span;
        span.effective_pos = 1;
        span.cells = {0, 1};
        view.spans.push_back(span);

        const auto emitted = tq::reader::emit_reader_attention(in, view, 0.70710678f, nullptr);
        // The refusal code changed with the contract, and this assertion had to change with it.
        //
        // It used to expect `graph_error` with a message about "the index tensor", because that was
        // what the emitter said: it created the tensor, found `data == nullptr`, and reported a
        // failure of ggml. The code was wrong about the cause -- under `no_alloc`, `data` is null
        // BY CONSTRUCTION (ggml.c:1825) -- so the emitter now refuses with `index_data_unavailable`
        // and names the missing half of the contract (the sink). Pinning the old wording would
        // pin the old misdiagnosis.
        check(emitted.refusal == tq::reader::emit_refusal::index_data_unavailable,
              "under no_alloc with no sink the emission refuses, naming the missing contract half");
        check(emitted.out == nullptr, "and a refusal produces no tensor");
        std::fprintf(stderr, "no_alloc: %s -- %s\n", tq::reader::describe(emitted.refusal),
                     emitted.why.c_str());
        check(emitted.why.find("collect_indices") != std::string::npos,
              "and the reason names what the caller owes -- graph_inputs::collect_indices -- rather "
              "than being told \"the graph failed\"; the wording changed with the contract (the "
              "entries now come back in the result, so there is no sink to name)");
        ggml_free(ctx);
    }

    // ---- structural checks: what the measurement cannot see ----
    //
    // The ratios above prove the bound holds; they cannot prove it holds for the RIGHT REASON.
    // Two mutants survived them -- clamping spans back to eight, and charging the cross-span union
    // once instead of prefix by prefix -- because both errors are absorbed by the 25% slack on
    // these small shapes. A bound whose correctness comes from its margin is not a bound, it is a
    // coincidence with headroom.
    //
    // So two properties are asserted directly, both independent of any measurement:
    {
        constexpr std::size_t perch = 368;   // any plausible overhead; cancels in both comparisons
        const auto make = [&](const std::size_t * cells, std::size_t n) {
            tq::reader::budget_inputs bi;
            bi.head_dim = kHeadDim;
            bi.n_head_kv = kKvHeads;
            bi.n_cell = 64;
            bi.n_head = kQHeads;
            bi.cells_per_span = cells;
            bi.n_spans = n;
            bi.tensor_overhead = perch;
            return tq::reader::estimate(bi);
        };

        // (a) The object count is LINEAR in spans, with an equal increment at every span --
        //     including the ninth. A clamp at eight makes the 8->9 increment vanish while the
        //     9->10 one reappears, so equal increments is exactly the property that fails.
        const std::size_t four[10] = {4, 4, 4, 4, 4, 4, 4, 4, 4, 4};
        const std::size_t o8 = make(four, 8).objects;
        const std::size_t o9 = make(four, 9).objects;
        const std::size_t o10 = make(four, 10).objects;
        std::fprintf(stderr, "objects: 8 spans %zu, 9 spans %zu, 10 spans %zu\n", o8, o9, o10);
        check(o9 > o8, "a ninth span costs objects: the estimator does not stop counting at eight");
        check(o9 - o8 == o10 - o9,
              "and the increment per span is the same at 8->9 and 9->10, which a clamp at eight "
              "cannot produce");

        // (b) The cross-span union is QUADRATIC, because the emitter concatenates in a loop
        //     (`scores = scores ? ggml_concat(scores, kq, 0) : kq`) and each intermediate holds
        //     every cell so far.
        //
        //     The previous assertion was `data(8) > 2 * data(4)`, and it FAILED at 1.99 because the
        //     linear terms (per-span gathers, scores, index tensors) dominate at these sizes. The
        //     tempting repair -- lowering 2 to 1.9 -- would have replaced a wrong claim with a
        //     threshold nobody can justify: the ratio depends on the shapes, so any constant is
        //     arbitrary and the case would stop meaning anything on the next model.
        //
        //     The exact test is the SECOND DIFFERENCE. If data(n) = A*n + B*n(n-1)/2 + C -- which
        //     is what a linear set of per-span charges plus a prefix-sum chain produces -- then
        //       data(4) - 2*data(3) + data(2) = B
        //     for every A and C. B > 0 if and only if the quadratic term is there, with no
        //     threshold, no dependence on which term dominates, and no dependence on the shape.
        //     A union charged once gives B == 0 and this equals zero.
        const std::size_t two_short[2] = {6, 6};
        const std::size_t three_short[3] = {6, 6, 6};
        const std::size_t four_short[4] = {6, 6, 6, 6};
        const std::size_t d2 = make(two_short, 2).data_bytes;
        const std::size_t d3 = make(three_short, 3).data_bytes;
        const std::size_t d4 = make(four_short, 4).data_bytes;
        const long long second_difference =
            static_cast<long long>(d4) - 2LL * static_cast<long long>(d3) +
            static_cast<long long>(d2);
        std::fprintf(stderr,
                     "data: 2 spans %zu, 3 spans %zu, 4 spans %zu; second difference %lld\n",
                     d2, d3, d4, second_difference);
        check(second_difference > 0,
              "the second difference of the data term is positive: a prefix-sum chain is "
              "quadratic, and charging the union once makes this exactly zero (an inequality on "
              "ratios cannot distinguish them, because the linear terms dominate at small n)");
    }

    // ---- the refusals, which are the point of this round ----
    //
    // Each is a case where no honest bound exists, and each must be reported rather than
    // approximated. A saturated total is the conservative direction: every arena refuses it, so
    // the caller cannot mistake the answer for room.
    {
        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
        const std::size_t one[1] = {6};

        tq::reader::budget_inputs bi;
        bi.head_dim = kHeadDim;
        bi.n_head_kv = kKvHeads;
        bi.n_cell = 32;
        bi.n_head = kQHeads;
        bi.cells_per_span = one;
        bi.n_spans = 1;
        bi.tensor_overhead = ggml_tensor_overhead();

        // (a) A rotation is refused, ALWAYS, and the reason is not \"the node count was not
        //     declared\" -- it is that no node count can bound the cost.
        //
        //     `span_rope_fn` receives the ggml_context: one legal implementation is
        //     `ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 100000, 100000)`, forty gigabytes from a
        //     single call. An earlier version accepted a caller-declared node count and charged one
        //     query-sized copy per node; that was a promise the estimator could not keep, and the
        //     field is gone. So both halves of the old case -- \"refused when undeclared\" and
        //     \"usable when declared\" -- collapse into one fact, and the second is asserted here
        //     rather than deleted: a future reader who reintroduces a declared-count field must
        //     fail this line.
        bi.has_rope = true;
        const auto rope_refused = tq::reader::estimate(bi);
        check(rope_refused.status == tq::reader::budget_status::unbounded_rope,
              "a rotation is refused as unbounded: the callback chooses node shapes, so no "
              "node count can bound bytes");
        check(rope_refused.total_bytes == kMax,
              "and its total saturates, so no arena can read it as room");
        check(rope_refused.objects == 0 && rope_refused.data_bytes == 0,
              "and it reports no partial cost either: a half-filled budget invites arithmetic "
              "on it");

        // (b) The same inputs WITHOUT the rotation are usable, which is what makes (a) a statement
        //     about the callback rather than about the shapes.
        bi.has_rope = false;
        const auto rope_absent = tq::reader::estimate(bi);
        check(rope_absent.status == tq::reader::budget_status::ok,
              "with no rotation the identical shapes yield a usable bound");
        check(rope_absent.objects > 0, "with objects charged");

        // (d) Element sizes whose SUM wraps. This is the case that made rejoining them a defect.
        //
        // With `data.add_product({..., k_elem + v_elem})`, two sizes at SIZE_MAX/2 + 1 add to 0:
        // the product collapses, no overflow is recorded, and the bound comes out BELOW the truth
        // -- fail-open. With the two products split, each one overflows and the answer is a
        // saturation, which no arena can read as room.
        //
        // Realistic element sizes (2 and 4) cannot show this: 2 + 4 equals 2 and 4 summed, so a
        // rejoined implementation passes every shape in the matrix above. That is exactly why the
        // case is stated in arithmetic rather than left to the model fixtures.
        {
            const std::size_t one[1] = {6};
            tq::reader::budget_inputs bi;
            bi.head_dim = kHeadDim;
            bi.n_head_kv = kKvHeads;
            bi.n_cell = 32;
            bi.n_head = kQHeads;
            bi.cells_per_span = one;
            bi.n_spans = 1;
            bi.tensor_overhead = ggml_tensor_overhead();
            bi.k_elem = kMax / 2 + 1;
            bi.v_elem = kMax / 2 + 1;
            const auto wrapped = tq::reader::estimate(bi);
            check(wrapped.status == tq::reader::budget_status::overflow,
                  "element sizes whose sum wraps are reported as overflow: joining them into one "
                  "factor collapses the product and puts the bound below the truth");
            check(wrapped.total_bytes == kMax, "and the total saturates");
        }

        // (c) Arithmetic that cannot fit: the product overflows, so the true cost exceeds any
        //     size_t. Saturating is the only answer that stays above the truth.
        bi.has_rope = false;
        bi.head_dim = kMax / 2;
        bi.n_head_kv = 4;
        bi.n_cell = kMax / 2;
        const auto overflowed = tq::reader::estimate(bi);
        check(overflowed.status == tq::reader::budget_status::overflow,
              "a product beyond size_t is reported as overflow, not wrapped into a small number");
        check(overflowed.total_bytes == kMax, "and the total saturates");

        // (d) No span storage at all: the count is positive but there is nothing to read.
        bi.head_dim = kHeadDim;
        bi.n_cell = 32;
        bi.cells_per_span = nullptr;
        const auto no_storage = tq::reader::estimate(bi);
        check(no_storage.status == tq::reader::budget_status::unsupported,
              "a positive span count with no storage is unsupported, not read as zero cells");
        check(no_storage.total_bytes == 0, "and computes nothing");
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "arena budget: the emission's cost is bounded above the measurement across cache "
                 "types and span shapes, and the no_alloc blocker is a measured refusal\n");
    return 0;
}
