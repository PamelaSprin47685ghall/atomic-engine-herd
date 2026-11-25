// The index hand-off, end to end: emit into a no_alloc arena, let a backend PLACE the tensors, then
// publish, then compute.
//
// WHY THIS TEST EXISTS, given that `publish_index_inputs` already has coverage.
//
// Every existing emitter test writes the gather indices by hand:
//
//     std::memcpy(entry.index->data, src->data(), entry.n_cells * sizeof(std::int32_t));
//
// That line is only legal AFTER placement, and it silently encodes a fact nobody had verified: that
// in a no_alloc context `index->data` is null until the backend allocates, and non-null afterwards.
// A hand-written memcpy also bypasses `publish_index_inputs` entirely, so the function that is
// supposed to own this hand-off had never been run on the path it exists for. Two consequences
// followed, and both are what this file checks:
//
//   * the ORDER was unverified. Publishing before placement writes through a null pointer; publishing
//     after is the only correct window. An API cannot enforce that, so a test has to demonstrate it.
//   * the VALUES were unverified. A published index that reaches the wrong number produces a gather
//     over the wrong cells -- plausible numbers, wrong attention. Comparing against a reference is
//     the only thing that catches it.
//
// The lifecycle here is the same one upstream's own `set_inputs` lives in:
//
//   ggml_init(no_alloc) -> emit (nodes exist, no storage) -> ggml_backend_alloc_ctx_tensors
//     -> write the marked inputs -> ggml_backend_graph_compute
//
// WHAT THIS IS NOT: a seam, and not a replacement. It proves the hand-off works in a test process.
// `build_attn_mha` runs during graph construction, which is INSIDE the first arrow above -- before
// allocation -- so a replacement still has to hand the entries to whoever runs this window. That is
// the remaining design question, not a defect this test can close.
#include <tq/reader/graph_emit_physical.h>

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

// Two spans, non-monotonic cells, one cell named twice would be refused by the emitter so the spans
// are disjoint: {6,2} and {4}. Five cache positions stay unnamed, which is what makes the emitted
// gather NON-identity and therefore worth publishing at all.
tq::reader::query_view the_view() {
    tq::reader::query_view v;
    tq::reader::span_keys a;
    a.effective_pos = 5;
    a.cells = {6, 2};
    tq::reader::span_keys b;
    b.effective_pos = 5;
    b.cells = {4};
    v.spans.push_back(a);
    v.spans.push_back(b);
    return v;
}

struct fixture {
    std::vector<float> k, v, q;
    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        unsigned s = 0x7f4a7c15u;
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

// Softmax over the CELLS THE VIEW NAMES, in view order, one bias of zero (no mask here). Written
// from the definition and indexed through the same cell list the gather should produce.
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

// The writer a caller supplies, in the shape the API actually declares:
//
//     bool (*)(ggml_tensor *, const void * data, std::size_t offset, std::size_t size)
//
// It goes through `ggml_backend_tensor_set` rather than `memcpy` into `data`, and that is not
// stylistic. `data` non-null means the scheduler PLACED the tensor; on a device backend that pointer
// is a device address, so a memcpy either faults or writes somewhere unrelated. Using the backend API
// is what makes this writer correct for the lifetime it is walking through -- and it is also what
// makes the test meaningful: a writer that only appears to work on the CPU host would pass a memcpy
// version and fail on any other backend.
//
// The API takes no user pointer, so the counting state is file-scope. `reset_writer` clears it before
// each attempt so an earlier call cannot satisfy a later assertion.
struct writer_state {
    int calls = 0;
    bool saw_null_buffer = false;
};

writer_state g_writer;

void reset_writer() { g_writer = writer_state{}; }

bool backend_writer(ggml_tensor * tensor, const void * data, std::size_t offset, std::size_t size) {
    ++g_writer.calls;
    if (tensor == nullptr) return false;
    if (tensor->buffer == nullptr) {
        // Placement has not happened. Recorded rather than dereferenced: this is the exact mistake
        // the publish step's own validation exists to catch first.
        g_writer.saw_null_buffer = true;
        return false;
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

void publish_then_compute_matches_the_definition() {
    ggml_init_params ip{};
    ip.mem_size = 8u * 1024u * 1024u;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { check(false, "ggml_init(no_alloc)"); return; }

    const fixture f;
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

    const tq::reader::query_view view = the_view();
    const auto emitted = tq::reader::emit_physical_attention(in, view, kScale);
    if (emitted.refusal != tq::reader::physical_emit_refusal::ok) {
        check(false, std::string("emission refused: ") + tq::reader::describe(emitted.refusal) +
                         " -- " + emitted.why);
        ggml_free(ctx);
        return;
    }

    // (1) BEFORE PLACEMENT the index tensors have no storage. Publishing here must be REFUSED rather
    //     than write through a null pointer, and that is what the publish step's own validation is
    //     for -- so it is exercised at the wrong moment on purpose.
    {
        std::string why;
        reset_writer();
        const auto r = tq::reader::publish_index_inputs(emitted, &backend_writer, &why, nullptr);
        check(r != tq::reader::index_publish_refusal::ok,
              "publishing BEFORE placement is refused, got " + std::string(tq::reader::describe(r)));
        check(g_writer.calls == 0 && !g_writer.saw_null_buffer,
              "and it is refused before the writer runs, not by the writer noticing null storage");
    }

    // (2) PLACEMENT. After this every marked input owns storage written through index->data.
    ggml_backend_t be = ggml_backend_cpu_init();
    if (be == nullptr) { check(false, "cpu backend"); ggml_free(ctx); return; }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (buf == nullptr) { check(false, "placement"); ggml_backend_free(be); ggml_free(ctx); return; }

    for (const auto & entry : emitted.indices) {
        check(entry.index != nullptr && entry.index->data != nullptr,
              "after placement every index tensor has storage");
    }

    // (3) PUBLISH. This is the window a replacement has to reach: constructed, placed, not computed.
    {
        std::string why;
        reset_writer();
        const auto r = tq::reader::publish_index_inputs(emitted, &backend_writer, &why, nullptr);
        check(r == tq::reader::index_publish_refusal::ok,
              std::string("publishing AFTER placement succeeds, got ") +
                  tq::reader::describe(r) + " (" + why + ")");
        if (g_writer.saw_null_buffer) check(false, "the writer saw an unplaced tensor after placement");
        // One call per entry: two gathers plus the scatter for this view.
        check(g_writer.calls == int(emitted.indices.size()),
              "the writer ran for every entry: " + std::to_string(g_writer.calls) + " of " +
                  std::to_string(emitted.indices.size()));
    }

    // The published cells must be the view's cells, in view order. Reading them back from the
    // TENSORS (not from `index_cells`) is what catches a publish that wrote the wrong buffer.
    {
        const std::vector<std::int32_t> expect{6, 2, 4};
        std::vector<std::int32_t> seen(expect.size());
        std::size_t at = 0;
        for (std::size_t s = 0; s < view.spans.size(); ++s) {
            const auto & entry = emitted.indices[s];
            std::memcpy(seen.data() + at, entry.index->data,
                        std::size_t(entry.n_cells) * sizeof(std::int32_t));
            at += std::size_t(entry.n_cells);
        }
        check(at == expect.size(), "the gathers cover every named cell");
        check(seen == expect, "published cells are the view's cells in view order");
    }

    // (4) COMPUTE, and compare with the definition. A correct publish plus a correct graph is the
    //     only combination that produces these numbers; either alone is not enough.
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, emitted.out);
    ggml_backend_tensor_set(in.k, f.k.data(), 0, f.k.size() * sizeof(float));
    ggml_backend_tensor_set(in.v, f.v.data(), 0, f.v.size() * sizeof(float));
    ggml_backend_tensor_set(in.q, f.q.data(), 0, f.q.size() * sizeof(float));

    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        check(false, "compute failed after a successful publish");
    } else {
        std::vector<float> got(std::size_t(kHeadDim) * kQHeads, 0.0f);
        std::memcpy(got.data(), emitted.out->data, got.size() * sizeof(float));
        std::vector<float> want;
        reference(f, {6, 2, 4}, want);

        double worst = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) {
                check(false, "non-finite output at " + std::to_string(i));
                break;
            }
            worst = std::max(worst, std::fabs(double(got[i]) - double(want[i])));
        }
        check(worst < 1e-6, "published-index attention matches the definition: max|diff| = " +
                                std::to_string(worst));

        // Non-vacuity: the numbers must depend on WHICH cells were gathered. Reversing the cell list
        // changes the row order and, since attention is permutation invariant, does NOT change `out`
        // -- so the discriminating check is against the WRONG cells, not the wrong order.
        std::vector<float> other;
        reference(f, {6, 2, 0}, other);   // cell 0 instead of cell 4
        double gap = 0.0;
        for (std::size_t i = 0; i < got.size(); ++i)
            gap = std::max(gap, std::fabs(double(want[i]) - double(other[i])));
        check(gap > 1e-3, "a different cell set changes the result by " + std::to_string(gap) +
                              " -- so the comparison can detect a wrong gather");
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
}

// The order matters, and the API refuses the wrong one. This is the assertion that makes the
// lifecycle a RULE rather than a convention: a caller who publishes during construction gets a
// named refusal instead of a write through null.
void publishing_before_placement_is_named_not_crashed() {
    ggml_init_params ip{};
    ip.mem_size = 4u * 1024u * 1024u;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) { check(false, "ggml_init for the order case"); return; }

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

    const auto emitted = tq::reader::emit_physical_attention(in, the_view(), kScale);
    if (emitted.refusal != tq::reader::physical_emit_refusal::ok) {
        check(false, "order case: emission refused");
        ggml_free(ctx);
        return;
    }

    std::string why;
    reset_writer();
    const auto r = tq::reader::publish_index_inputs(emitted, &backend_writer, &why, nullptr);
    check(r == tq::reader::index_publish_refusal::no_storage,
          std::string("publishing during construction is `no_storage`, got ") +
              tq::reader::describe(r) + " (" + why + ")");
    // The message is checked for the FACT, not for a phrase this test guessed. An earlier revision
    // asserted the words "before"/"place" and went red on "the tensor has no storage yet" -- the
    // diagnosis was right and the assertion was measuring my vocabulary instead of the contract.
    check(why.find("storage") != std::string::npos,
          "and the message names the missing storage: " + why);
    check(g_writer.calls == 0,
          "the writer never ran, so nothing was written through an unplaced tensor");
    ggml_free(ctx);
}

}  // namespace

int main() {
    publish_then_compute_matches_the_definition();
    publishing_before_placement_is_named_not_crashed();

    if (failures == 0) {
        std::fprintf(stderr, "index lifecycle: OK\n");
        return 0;
    }
    std::fprintf(stderr, "index lifecycle: %d failure(s)\n", failures);
    return 1;
}
