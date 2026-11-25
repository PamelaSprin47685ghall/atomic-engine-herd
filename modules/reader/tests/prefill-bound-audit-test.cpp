// The prefill composition's ADMISSION BOUND against what ggml actually builds, and its F16 numbers
// against a reference written from the definition.
//
// WHY THIS FILE EXISTS, AND WHAT THE PREVIOUS VERSION OF IT GOT WRONG
// -------------------------------------------------------------------
// `prefill-emit-test` proves the numbers for ONE configuration: F32 only, spans = 1, T = 3,
// `group = 2`, generous arena. Two consequences followed, and both were defects in the test rather
// than in the emitter:
//
//   * an under-estimated bound aborts inside `ggml_new_object` half-way through a graph, and no
//     numeric comparison can see it -- so the bound has to be compared against the arena's real
//     growth, over configurations the other test never builds (F16 caches, `group` of 1/2/4,
//     1/2/3 spans);
//   * the F16 path was never compared to a reference ANYWHERE. The previous revision of this file
//     said "F16 agreement is asserted in prefill-emit-test", which was false -- that file is F32
//     throughout. An F16 path that ignored a stride or a dtype therefore had no numeric test.
//
// The previous revision also asserted its boundary case in a way that could not fail: `emit_in` did
// its work under `if (want_ok)`, so on the refusal path `built` stayed false and `grew` stayed zero
// no matter what the emitter did. `check(!run.built, ...)` then proved nothing, and a compute
// failure was never recorded at all. Those are fixed by making the refusal path ASSERT its own
// observables instead of skipping them.
//
// The boundary is also now the real one. `bytes_required` counts graph objects only, so a context of
// `bytes_required - 1` bytes does not test "one byte short of the requirement" -- it tests "far too
// small, and the fixtures barely fit". Here the fixture cost is measured, and the pair
// (fixture + required, fixture + required - 1) brackets the requirement exactly.
#include <tq/reader/graph_emit_prefill.h>
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
void check(bool ok, const std::string & what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); ++failures; }
}

constexpr int kHeadDim = 8, kCell = 16;
constexpr float kScale = 0.35355339f;

struct config {
    int n_head;
    int n_head_kv;
    int tokens;
    ggml_type cache_type;
    ggml_type q_type;
    int n_spans;
    const char * label;
};

// Spans that together name every cache cell exactly once, so the mask (and hence the numbers) do not
// depend on how the cells are grouped -- only the node count does.
tq::reader::query_view view_with(int n_spans) {
    tq::reader::query_view v;
    int next = 0;
    for (int s = 0; s < n_spans; ++s) {
        tq::reader::span_keys span;
        span.effective_pos = 5;
        const int take = (kCell - next) / (n_spans - s);
        for (int i = 0; i < take; ++i) span.cells.push_back(std::int32_t(next + i));
        next += take;
        v.spans.push_back(std::move(span));
    }
    for (; next < kCell; ++next) v.spans.back().cells.push_back(std::int32_t(next));
    return v;
}

struct fixture {
    std::vector<float> k32, v32, q32, mask;
    int tokens = 0, n_head = 0, n_head_kv = 0;

    fixture() = default;
    explicit fixture(const config & c) {
        tokens = c.tokens; n_head = c.n_head; n_head_kv = c.n_head_kv;
        const std::size_t kv = std::size_t(kHeadDim) * c.n_head_kv * kCell;
        k32.assign(kv, 0.0f);
        v32.assign(kv, 0.0f);
        q32.assign(std::size_t(kHeadDim) * c.tokens * c.n_head, 0.0f);
        mask.assign(std::size_t(kCell) * c.tokens, 0.0f);
        unsigned s = 0x51ed270bu;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k32) x = next();
        for (float & x : v32) x = next() * 1.5f;
        for (float & x : q32) x = next();
        // Per-token causal ladder: token t sees cells below 6 + 3t, with a finite bias on them.
        for (int t = 0; t < c.tokens; ++t)
            for (int p = 0; p < kCell; ++p)
                mask[std::size_t(p) + std::size_t(kCell) * t] =
                    (p < 6 + 3 * t) ? 0.25f * float(t) : -INFINITY;
    }

    float K(int d, int kvh, int c) const { return k32[std::size_t(d) + kHeadDim * (kvh + n_head_kv * c)]; }
    float V(int d, int kvh, int c) const { return v32[std::size_t(d) + kHeadDim * (kvh + n_head_kv * c)]; }
    float Q(int d, int t, int h) const { return q32[std::size_t(d) + kHeadDim * (t + tokens * h)]; }
    float M(int p, int t) const { return mask[std::size_t(p) + std::size_t(kCell) * t]; }
};

// The definition, per (token, head): one softmax over the whole cache, `-inf` hidden, finite bias
// added, grouped kv head = h / (n_head / n_head_kv). Written from that sentence, not from the graph.
void reference(const fixture & f, std::vector<float> & out) {
    const int group = f.n_head / f.n_head_kv;
    out.assign(std::size_t(kHeadDim) * std::size_t(f.tokens) * std::size_t(f.n_head), 0.0f);
    for (int t = 0; t < f.tokens; ++t) {
        for (int h = 0; h < f.n_head; ++h) {
            const int kvh = h / group;
            std::vector<double> score(static_cast<std::size_t>(kCell), 0.0);
            double max_s = -INFINITY;
            for (int p = 0; p < kCell; ++p) {
                const double b = double(f.M(p, t));
                if (!std::isfinite(b)) { score[std::size_t(p)] = -INFINITY; continue; }
                double s = 0.0;
                for (int d = 0; d < kHeadDim; ++d)
                    s += double(f.Q(d, t, h)) * double(f.K(d, kvh, p));
                score[std::size_t(p)] = s * double(kScale) + b;
                if (score[std::size_t(p)] > max_s) max_s = score[std::size_t(p)];
            }
            double sum = 0.0;
            for (int p = 0; p < kCell; ++p) {
                if (!std::isfinite(score[std::size_t(p)])) continue;
                score[std::size_t(p)] = std::exp(score[std::size_t(p)] - max_s);
                sum += score[std::size_t(p)];
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (int p = 0; p < kCell; ++p) {
                    if (!std::isfinite(score[std::size_t(p)]) || score[std::size_t(p)] == 0.0) continue;
                    acc += (score[std::size_t(p)] / sum) * double(f.V(d, kvh, p));
                }
                out[std::size_t(d) + std::size_t(kHeadDim) *
                                         (std::size_t(t) + std::size_t(f.tokens) * std::size_t(h))] = float(acc);
            }
        }
    }
}

void to_half(const std::vector<float> & src, std::vector<ggml_fp16_t> & dst) {
    dst.resize(src.size());
    ggml_fp32_to_fp16_row(src.data(), dst.data(), std::int64_t(src.size()));
}

// One run. `refusal` and `grew` are recorded on BOTH paths, so the refusal path has observables that
// can fail; the previous revision recorded them only under `if (want_ok)` and proved nothing.
struct run {
    tq::reader::prefill_refusal refusal = tq::reader::prefill_refusal::ok;
    std::string why;
    std::size_t grew = 0;
    std::size_t objects = 0;
    std::size_t cost = 0;
    std::size_t bytes_required = 0;
    bool emitted = false;       // the emitter returned ok
    bool computed = false;      // placement, publication and compute all succeeded
    std::vector<float> out;     // [hd, T, nh], copied out before the context dies
    std::string fail_note;      // a compute/placement failure, recorded instead of ignored
};

// Builds the graph in a fresh no_alloc context. `fixture_bytes` is where the fixtures were placed so
// the caller can bracket the requirement; it is computed by the caller when it needs it.
run emit_in(const config & c, std::size_t arena_bytes, bool admission_only = false) {
    run out;
    ggml_init_params ip{};
    ip.mem_size = arena_bytes;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        out.fail_note = "ggml_init failed";
        ++failures;
        std::fprintf(stderr, "FAIL: ggml_init(%zu) for %s\n", arena_bytes, c.label);
        return out;
    }

    const fixture f(c);
    const tq::reader::query_view view = view_with(c.n_spans);

    ggml_tensor * k = ggml_new_tensor_3d(ctx, c.cache_type, kHeadDim, c.n_head_kv, kCell);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, c.cache_type, kHeadDim, c.n_head_kv, kCell);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, c.q_type, kHeadDim, c.tokens, c.n_head);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, c.tokens, 1, 1);
    // The mask stays F32 in every configuration, so a difference between two runs can only come from
    // the cache or the query width.
    ggml_set_input(k); ggml_set_input(v); ggml_set_input(q); ggml_set_input(m);

    tq::reader::prefill_inputs in;
    in.ctx = ctx;
    in.k = k; in.v = v; in.q = q; in.mask = m;
    in.view = view;
    in.n_head = std::size_t(c.n_head);
    in.n_head_kv = std::size_t(c.n_head_kv);
    in.head_dim = kHeadDim;
    in.n_cell = kCell;
    in.coverage = tq::reader::graph_inputs::mask_coverage::intersect;

    const std::size_t before = ggml_used_mem(ctx);
    const tq::reader::prefill_result r = tq::reader::emit_prefill_attention(in, kScale);
    out.refusal = r.refusal;
    out.why = r.why;
    out.grew = ggml_used_mem(ctx) - before;
    out.objects = r.objects_charged;
    out.cost = r.object_cost;
    out.bytes_required = r.bytes_required;

    if (r.refusal != tq::reader::prefill_refusal::ok) {
        // The refusal path still has to check that nothing was built. This is the assertion the
        // previous revision made vacuously, by never reaching it.
        if (out.grew != 0) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s: refusing built %zu bytes\n", c.label, out.grew);
        }
        ggml_free(ctx);
        return out;
    }
    out.emitted = true;

    // ADMISSION-ONLY: stop here, before the caller's own `ggml_new_graph`.
    //
    // The boundary case needs a context of exactly `fixture + requirement` bytes, and such a context
    // has no room left for the graph object the CALLER creates (~107 KB: `GGML_DEFAULT_GRAPH_SIZE`
    // nodes). That is not the emitter's requirement, and letting it run would abort inside
    // `ggml_new_object` with "not enough space in the context's memory pool" -- which is what the
    // first version of this fix did. So the bracket is asserted on the admission decision, which is
    // exactly the thing `bytes_required` describes, and the graph-backed checks above cover the rest.
    if (admission_only) {
        ggml_free(ctx);
        return out;
    }

    if (r.per_token.empty() || r.out == nullptr) {
        check(false, std::string(c.label) + ": emitted without per-token results or an output");
        ggml_free(ctx);
        return out;
    }

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = be ? ggml_backend_alloc_ctx_tensors(ctx, be) : nullptr;
    if (buf == nullptr) {
        out.fail_note = "placement failed";
        check(false, std::string(c.label) + ": placement failed");
        if (be != nullptr) ggml_backend_free(be);
        ggml_free(ctx);
        return out;
    }

    std::vector<ggml_fp16_t> hk, hv, hq;
    if (c.cache_type == GGML_TYPE_F16) {
        to_half(f.k32, hk);
        to_half(f.v32, hv);
        ggml_backend_tensor_set(k, hk.data(), 0, hk.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v, hv.data(), 0, hv.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(k, f.k32.data(), 0, f.k32.size() * sizeof(float));
        ggml_backend_tensor_set(v, f.v32.data(), 0, f.v32.size() * sizeof(float));
    }
    if (c.q_type == GGML_TYPE_F16) {
        to_half(f.q32, hq);
        ggml_backend_tensor_set(q, hq.data(), 0, hq.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(q, f.q32.data(), 0, f.q32.size() * sizeof(float));
    }
    ggml_backend_tensor_set(m, f.mask.data(), 0, f.mask.size() * sizeof(float));

    bool published = true;
    for (std::size_t t = 0; t < r.per_token.size(); ++t) {
        std::string why;
        auto w = +[](ggml_tensor * tt, const void * d, std::size_t o, std::size_t s) -> bool {
            if (tt == nullptr || tt->buffer == nullptr) return false;
            ggml_backend_tensor_set(tt, d, o, s);
            return true;
        };
        const auto pr = tq::reader::publish_index_inputs(r.per_token[t], w, &why, nullptr);
        if (pr != tq::reader::index_publish_refusal::ok) {
            out.fail_note = "publish: " + std::string(tq::reader::describe(pr)) + " (" + why + ")";
            check(false, std::string(c.label) + ": token " + std::to_string(t) + ": " + out.fail_note);
            published = false;
        }
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r.out);
    if (!published) {
        out.fail_note = "not computed: publication failed";
    } else if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        // A compute failure is a FAILURE. The previous revision wrapped the whole block in
        // `if (published && compute == SUCCESS)` and did nothing in the else branch, so a graph that
        // never ran left `failures` untouched.
        out.fail_note = "compute failed";
        check(false, std::string(c.label) + ": compute failed, so no number below was read");
    } else {
        out.computed = true;
        out.out.assign(std::size_t(kHeadDim) * std::size_t(c.tokens) * std::size_t(c.n_head), 0.0f);
        ggml_backend_tensor_get(r.out, out.out.data(), 0, out.out.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);   // `out` holds values now, not tensor pointers
    return out;
}

// The bytes the four fixture tensors occupy in a context of this shape. Needed to bracket
// `bytes_required`, which counts graph objects only.
std::size_t fixture_bytes(const config & c) {
    ggml_init_params ip{};
    ip.mem_size = 1u * 1024u * 1024u;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) return 0;
    ggml_tensor * k = ggml_new_tensor_3d(ctx, c.cache_type, kHeadDim, c.n_head_kv, kCell);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, c.cache_type, kHeadDim, c.n_head_kv, kCell);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, c.q_type, kHeadDim, c.tokens, c.n_head);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, c.tokens, 1, 1);
    const std::size_t used = (k && v && q && m) ? ggml_used_mem(ctx) : 0;
    ggml_free(ctx);
    return used;
}

} // namespace

int main() {
    // (1) THE GRID: the bound must hold, and be tight enough to be worth publishing, for every
    //     configuration a caller can hand it -- including F16 and the groupings the other test skips.
    const config grid[] = {
        {4, 2, 3, GGML_TYPE_F32, GGML_TYPE_F32, 1, "F32 group2 spans1 T3"},
        {4, 2, 3, GGML_TYPE_F32, GGML_TYPE_F32, 2, "F32 group2 spans2 T3"},
        {4, 2, 3, GGML_TYPE_F32, GGML_TYPE_F32, 3, "F32 group2 spans3 T3"},
        {4, 2, 2, GGML_TYPE_F16, GGML_TYPE_F16, 1, "F16 group2 spans1 T2"},
        {4, 2, 3, GGML_TYPE_F16, GGML_TYPE_F32, 2, "F16cache F32q spans2 T3"},
        {4, 2, 3, GGML_TYPE_F32, GGML_TYPE_F16, 3, "F32cache F16q spans3 T3"},
        {8, 2, 3, GGML_TYPE_F32, GGML_TYPE_F32, 1, "F32 group4 spans1 T3"},
        {8, 2, 3, GGML_TYPE_F16, GGML_TYPE_F16, 2, "F16 group4 spans2 T3"},
        {4, 1, 3, GGML_TYPE_F32, GGML_TYPE_F32, 2, "F32 MQA spans2 T3"},
        {4, 4, 3, GGML_TYPE_F32, GGML_TYPE_F32, 1, "F32 MHA spans1 T3"},
        {4, 2, 5, GGML_TYPE_F32, GGML_TYPE_F32, 3, "F32 group2 spans3 T5"},
    };

    for (const config & c : grid) {
        const run r = emit_in(c, 8u * 1024u * 1024u);
        check(r.emitted, std::string(c.label) + ": emitted, got " + tq::reader::describe(r.refusal) +
                             " -- " + r.why);
        if (!r.emitted) continue;
        check(r.computed, std::string(c.label) + ": computed" +
                              (r.fail_note.empty() ? "" : " (" + r.fail_note + ")"));

        bool ov = false;
        const std::size_t expect = tq::reader::prefill_object_count(
            c.tokens, std::size_t(c.n_spans), c.q_type != GGML_TYPE_F32, &ov);
        check(!ov, std::string(c.label) + ": the bound overflowed");
        check(r.objects == expect,
              std::string(c.label) + ": objects_charged " + std::to_string(r.objects) +
                  " != prefill_object_count " + std::to_string(expect) +
                  " -- the published bound is not the documented one");
        check(r.cost > 0, std::string(c.label) + ": object_cost was not measured");

        // The load-bearing assertion: a charge below the truth aborts inside `ggml_new_object` before
        // this line, so reaching it with `grew <= need` is the evidence that the bound covers it.
        check(r.grew > 0, std::string(c.label) + ": the arena did not grow");
        check(r.grew <= r.bytes_required,
              std::string(c.label) + ": arena grew " + std::to_string(r.grew) + " against a charge of " +
                  std::to_string(r.bytes_required));
        const double ratio = double(r.bytes_required) / double(r.grew);
        std::fprintf(stderr, "  %-26s objects %4zu  grew %7zu  charged %7zu  (%.2fx)\n", c.label,
                     r.objects, r.grew, r.bytes_required, ratio);
        check(ratio < 4.0, std::string(c.label) + ": the charge is " + std::to_string(ratio) +
                               "x the growth -- a guess, not an admission bound");

        // ... AND THE NUMBERS. Half precision loosens the tolerance, it does not remove the check.
        if (r.computed) {
            const fixture f(c);
            std::vector<float> want;
            reference(f, want);
            const double tol = (c.cache_type == GGML_TYPE_F16 || c.q_type == GGML_TYPE_F16)
                                   ? 5e-3 : 1e-5;
            double worst = 0.0;
            int at = -1;
            bool finite = true;
            for (std::size_t i = 0; i < r.out.size() && i < want.size(); ++i) {
                if (!std::isfinite(r.out[i])) { finite = false; break; }
                const double d = std::fabs(double(r.out[i]) - double(want[i]));
                if (d > worst) { worst = d; at = int(i); }
            }
            check(finite, std::string(c.label) + ": output is finite");
            check(r.out.size() == want.size(), std::string(c.label) + ": output size");
            check(worst <= tol, std::string(c.label) + ": matches the definition, max|diff| = " +
                                    std::to_string(worst) + " at " + std::to_string(at) +
                                    " (tol " + std::to_string(tol) + ")");
            double magnitude = 0.0;
            for (float x : want) magnitude = std::max(magnitude, std::fabs(double(x)));
            check(magnitude > 1e-2, std::string(c.label) + ": the reference is non-degenerate (" +
                                       std::to_string(magnitude) + ")");
        }
    }

    // (2) THE BOUNDARY ITSELF: fixture bytes plus exactly the requirement succeeds; one byte less
    //     refuses with `arena_admission` and builds nothing.
    //
    //     The previous revision used `bytes_required - 1` as the whole arena, which is not one byte
    //     short of the requirement -- it is far too small for the fixtures of some configurations and
    //     merely "small" for others, and it allowed `bad_shape` as an acceptable verdict, which is
    //     what a fixture that does not fit would produce.
    {
        const config c{4, 2, 3, GGML_TYPE_F32, GGML_TYPE_F32, 1, "boundary probe"};
        const std::size_t fx = fixture_bytes(c);
        check(fx > 0, "the fixture cost was measured");
        const run probe = emit_in(c, 8u * 1024u * 1024u);
        check(probe.emitted, "the boundary probe emitted, so there is a requirement to bracket");
        if (probe.emitted && fx > 0 && probe.bytes_required > 0) {
            const std::size_t need = fx + probe.bytes_required;

            const run exact = emit_in(c, need, true);
            check(exact.emitted,
                  std::string("at fixture + requirement (") + std::to_string(need) +
                      " bytes) the emission is ADMITTED, got " + tq::reader::describe(exact.refusal) +
                      " -- " + exact.why);

            // THE REFUSAL SIDE, AND WHY ±1 BYTE IS THE WRONG PROBE.
            //
            // `need - 1` was tried and is ADMITTED. The temptation is to call that "ggml_init's
            // alignment" or "accounting granularity"; neither survives the numbers this file already
            // prints. The charge is `objects * per_object`, where `per_object` is measured from a
            // ONE-object probe, so every object's alignment slack is counted in full and the total
            // sits 6-10% ABOVE the growth the graph really causes (the ratios printed above:
            // 1.06-1.10x, i.e. 1500-2200 spare bytes on a 23-25 KB charge). One byte is nowhere near
            // that margin, so `need - 1` was never "one byte short of the requirement" -- it was one
            // byte short of a CONSERVATIVE OVER-ESTIMATE.
            //
            // That makes the byte-exact boundary an artefact of how the bound is computed rather than
            // a property of the arena, and pinning it would assert a number this emitter is not
            // claiming. What is asserted instead is the bracket that carries meaning: the charge is
            // real, because removing all of it refuses.
            const run short_run = emit_in(c, need - probe.bytes_required, true);
            check(!short_run.emitted, "without the graph's charge the emission refuses");
            check(short_run.refusal == tq::reader::prefill_refusal::arena_admission,
                  std::string("the short context reports arena_admission exactly, got ") +
                      tq::reader::describe(short_run.refusal) + " -- " + short_run.why);
            check(short_run.grew == 0, "the refusal built " + std::to_string(short_run.grew) + " bytes");
            std::fprintf(stderr, "  boundary: fixture %zu + required %zu; exact admits, "
                                 "minus the charge refuses\n", fx, probe.bytes_required);
        }
    }

    // (3) THE OVERFLOW PATH: the bound's own arithmetic must refuse rather than wrap. `1 << 40`
    //     tokens did NOT overflow (36 objects each is 3.9e13, which fits in a 64-bit size_t), so the
    //     first version of this check asserted a property the arithmetic never claimed. `1 << 62` is
    //     past the point where the multiply can wrap.
    {
        bool ov = false;
        const std::size_t n = tq::reader::prefill_object_count(std::int64_t(1) << 62, 3, true, &ov);
        check(ov, "a token count at 2^62 sets overflow, got " + std::to_string(n));
        check(n == 0, "and returns 0 rather than a wrapped byte budget, got " + std::to_string(n));
        bool ov2 = false;
        const std::size_t zero = tq::reader::prefill_object_count(1, 1, false, &ov2);
        check(zero == 0, "T == 1 needs no composition, got " + std::to_string(zero));
    }

    if (failures == 0) {
        std::fprintf(stderr, "prefill bound audit: OK\n");
        return 0;
    }
    std::fprintf(stderr, "prefill bound audit: %d failure(s)\n", failures);
    return 1;
}
