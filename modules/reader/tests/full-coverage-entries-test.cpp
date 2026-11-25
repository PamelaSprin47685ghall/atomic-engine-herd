// What the emitter returns for a FULL-COVERAGE view: how many index entries, and are the numbers right.
//
// WHY THIS EARNS A TEST. A previous experiment reported `0 index entr(ies)` for the full-coverage
// view and the note drawn from it was "the gather degenerates, so the hand-off never runs". That
// reading was wrong, and the source says so plainly: `graph_emit_physical.cpp` pushes one entry per
// span and then ALWAYS pushes the scatter (543-545), so `indices` cannot be empty for any view the
// emitter accepts. The zero came from reading `em.indices` AFTER `std::move(em)` had emptied it --
// the same use-after-move that made the replacement log untrustworthy.
//
// That distinction decides real work: if a full view produced no gather, a replacement would need no
// index publication at all; if it produces one, the publisher is on the critical path. So the count
// is measured here for three views, and the numbers are checked against a reference, so neither the
// count nor the arithmetic can be assumed.
//
// The contract asserted:
//   entries.size() == spans.size() + 1     one gather per span, plus the scatter
//   entries[0..S)  tile index_cells[0..N) contiguously, in span order
//   entries[S]     is the scatter: offset 0, n_cells == N
//   index_cells    is the view's cells, in span order, once each
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
        unsigned s = 0x2545f491u;
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

// Softmax over the named cells only (the emitter's compact-axis reference): one global softmax over
// [n_cell] would need the mask, and this test deliberately supplies none.
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

// Emits, then publishes, then computes -- the full life cycle, because a count without placement
// proves nothing about whether the tensors can be written.
void run_case(const char * name, const tq::reader::query_view & view) {
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

    // (1) THE COUNT. One gather per span plus the scatter, always -- the source pushes the scatter
    //     unconditionally, so a different number means the contract changed.
    const std::size_t want_entries = view.spans.size() + 1;
    check(em.indices.size() == want_entries,
          std::string(name) + ": " + std::to_string(em.indices.size()) + " entries, expected " +
              std::to_string(want_entries) + " (one gather per span, plus the scatter)");
    if (em.indices.size() != want_entries) { ggml_free(ctx); return; }

    // (2) THE TILING, checked here so a publish later in this function cannot silently paper over a
    //     wrong layout: gathers tile [0, N) in span order, the scatter covers all of it.
    const std::vector<int> cells = flatten(view);
    check(em.index_cells.size() == cells.size(),
          std::string(name) + ": index_cells holds " + std::to_string(em.index_cells.size()) +
              " of " + std::to_string(cells.size()) + " cells");
    std::size_t expect_offset = 0;
    for (std::size_t s = 0; s < view.spans.size(); ++s) {
        check(em.indices[s].cells_offset == expect_offset,
              std::string(name) + ": gather " + std::to_string(s) + " offset " +
                  std::to_string(em.indices[s].cells_offset) + ", expected " +
                  std::to_string(expect_offset));
        check(em.indices[s].n_cells == view.spans[s].cells.size(),
              std::string(name) + ": gather " + std::to_string(s) + " covers " +
                  std::to_string(em.indices[s].n_cells) + " cells, expected " +
                  std::to_string(view.spans[s].cells.size()));
        expect_offset += view.spans[s].cells.size();
    }
    check(em.indices.back().cells_offset == 0 && em.indices.back().n_cells == cells.size(),
          std::string(name) + ": the scatter must cover the whole buffer");

    // (3) PLACEMENT, then publish, then compute. Before placement the publish must be refused by
    //     name; after it, it must succeed and the numbers must match the definition.
    {
        std::string why;
        const auto r = tq::reader::publish_index_inputs(
            em, +[](ggml_tensor *, const void *, std::size_t, std::size_t) { return true; }, &why,
            nullptr);
        check(r == tq::reader::index_publish_refusal::no_storage,
              std::string(name) + ": publishing before placement is `no_storage`, got " +
                  tq::reader::describe(r) + " (" + why + ")");
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = be ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
    if (buf == nullptr) {
        check(false, std::string(name) + ": placement failed");
        if (be) ggml_backend_free(be);
        ggml_free(ctx);
        return;
    }

    std::string why;
    const auto r = tq::reader::publish_index_inputs(
        em,
        +[](ggml_tensor * t, const void * d, std::size_t o, std::size_t s) -> bool {
            if (t == nullptr || t->buffer == nullptr) return false;
            ggml_backend_tensor_set(t, d, o, s);
            return true;
        },
        &why, nullptr);
    check(r == tq::reader::index_publish_refusal::ok,
          std::string(name) + ": publish after placement: " + tq::reader::describe(r) + " (" + why +
              ")");

    // The values, read back out of the TENSORS: this is what a wrong publish would move.
    {
        std::vector<std::int32_t> seen(cells.size(), -1);
        std::size_t at = 0;
        for (std::size_t s = 0; s < view.spans.size(); ++s) {
            std::memcpy(seen.data() + at, em.indices[s].index->data,
                        std::size_t(em.indices[s].n_cells) * sizeof(std::int32_t));
            at += std::size_t(em.indices[s].n_cells);
        }
        std::vector<std::int32_t> want(cells.begin(), cells.end());
        check(seen == want, std::string(name) + ": published gathers hold the view's cells in order");
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, em.out);
    ggml_backend_tensor_set(in.k, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(in.v, f.v.data(), 0, f.v.size() * sizeof(float));
    ggml_backend_tensor_set(in.q, f.q.data(), 0, f.q.size() * sizeof(float));

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        check(false, std::string(name) + ": compute failed");
    } else {
        std::vector<float> got(std::size_t(kHeadDim) * kQHeads, 0.0f);
        std::memcpy(got.data(), em.out->data, got.size() * sizeof(float));
        std::vector<float> want;
        reference(f, cells, want);
        double worst = 0.0;
        bool finite = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) { finite = false; break; }
            worst = std::max(worst, std::fabs(double(got[i]) - double(want[i])));
        }
        check(finite, std::string(name) + ": output is finite");
        check(worst < 1e-6, std::string(name) + ": matches the definition, max|diff| = " +
                                std::to_string(worst));

        // Non-vacuity: the numbers depend on WHICH cells, so a gather over the wrong ones fails.
        double magnitude = 0.0;
        for (float x : want) magnitude = std::max(magnitude, std::fabs(double(x)));
        check(magnitude > 1e-3, std::string(name) + ": reference is non-degenerate (" +
                                   std::to_string(magnitude) + ")");
    }

    std::fprintf(stderr,
                 "%s: %zu span(s) -> %zu entries, %zu cells, published and computed\n", name,
                 view.spans.size(), em.indices.size(), cells.size());

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
}

// A view naming every cache position: this is the shape a replacement would use, and the shape the
// previous experiment claimed produced no gather.
void full_coverage_view_produces_a_gather() {
    run_case("full coverage", tq::reader::full_coverage_view(kCell));
}

// Two spans with non-monotonic cells: the ordinary sparse shape, where a gather is unavoidable.
void sparse_view_produces_a_gather_per_span() {
    tq::reader::query_view v;
    tq::reader::span_keys a;
    a.effective_pos = 5;
    a.cells = {6, 2};
    tq::reader::span_keys b;
    b.effective_pos = 5;
    b.cells = {4};
    v.spans.push_back(a);
    v.spans.push_back(b);
    run_case("two spans", v);
}

// A single span that names one cell: the degenerate end, where an off-by-one in the tiling shows up
// as an empty gather rather than a wrong number.
void single_cell_span_produces_one_gather() {
    tq::reader::query_view v;
    tq::reader::span_keys a;
    a.effective_pos = 3;
    a.cells = {5};
    v.spans.push_back(a);
    run_case("single cell", v);
}

}  // namespace

int main() {
    full_coverage_view_produces_a_gather();
    sparse_view_produces_a_gather_per_span();
    single_cell_span_produces_one_gather();

    if (failures == 0) {
        std::fprintf(stderr, "full-coverage entries: OK\n");
        return 0;
    }
    std::fprintf(stderr, "full-coverage entries: %d failure(s)\n", failures);
    return 1;
}
