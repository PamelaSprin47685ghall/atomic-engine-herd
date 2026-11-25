// The reader's attention arithmetic against ggml's own kernels.
//
// Every other test of this core compares it with a hand-written reference of the same
// definition -- necessary, but not sufficient: if the definition is wrong, both sides are
// wrong the same way and the test passes. This one removes that circularity by computing
// the expected values with ggml itself, on the same layouts llama.cpp builds.
//
// It also settles the one claim the header used to make carelessly. `attention.h` says a
// comparison against stock must not assume a bf16 tolerance, because ggml_prec_set_acc()
// sets the *minimum* accumulator type and the CPU backend does not consult it for mul_mat.
// That is asserted here: setting GGML_PREC_BF16 on the KQ product is ACCEPTED by the API
// and leaves every output bit unchanged.
//
// What is measured, and what each number means:
//
//   ggml(F32) vs the double reference      ~5e-8   float rounding, nothing more
//   ggml(F32) vs the STRIDED mapping       ~4e-1   seven orders larger: the comparison
//                                                  can tell the two mappings apart, so the
//                                                  agreement above is evidence, not a tautology
//   attend()  vs ggml(F32)                 <=1e-6  same arithmetic, double vs float accum
//   prec(BF16) vs prec(off)                 0 bits identical on CPU
#include <tq/reader/attention.h>

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

// llama's shapes, not convenient ones: n_head_kv < n_head, so the head axis is broadcast.
constexpr int kKvHeads = 2;
constexpr int kQHeads  = 4;
constexpr int kHeadDim = 8;
constexpr int kKeys    = 16;
constexpr int kGroup   = kQHeads / kKvHeads;
constexpr float kScale = 0.70710678f;

// Deterministic, asymmetric values: a symmetric fixture can make a wrong gather or a wrong
// head pairing produce the right answer.
struct fixture {
    // ggml layouts (what llama builds):
    //   k : [head_dim, n_keys, n_head_kv]   idx = d + HD*(i + NK*h)
    //   v : [n_keys,   head_dim, n_head_kv] idx = i + NK*(d + HD*h)   <- the value multiply
    //                                          is mul_mat(v, kq), so ne0 must be n_keys
    //   q : [head_dim, 1, n_head_q]         idx = d + HD*h
    std::vector<float> k_ggml, v_ggml, q_ggml;

    fixture() {
        k_ggml.assign(std::size_t(kHeadDim) * kKeys * kKvHeads, 0.0f);
        v_ggml.assign(std::size_t(kKeys) * kHeadDim * kKvHeads, 0.0f);
        q_ggml.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);

        unsigned state = 12345u;
        auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return float((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (int h = 0; h < kKvHeads; ++h) {
            for (int i = 0; i < kKeys; ++i) {
                for (int d = 0; d < kHeadDim; ++d) {
                    k_ggml[d + kHeadDim * (i + kKeys * h)] = next();
                    // Wider range on the values: a wrong head pairing then differs by much
                    // more than a wrong rounding, so the two failures cannot be confused.
                    v_ggml[i + kKeys * (d + kHeadDim * h)] = next() * 2.0f;
                }
            }
        }
        for (int h = 0; h < kQHeads; ++h) {
            for (int d = 0; d < kHeadDim; ++d) {
                q_ggml[d + kHeadDim * h] = next();
            }
        }
    }

    // The reader's layout for the same data: key `cell`, kv head `h`, dim `d` at
    // (cell * n_head_kv + h) * head_dim + d.
    tq::reader::kv_buffers reader_buffers(std::vector<float> & k, std::vector<float> & v) const {
        k.assign(std::size_t(kKeys) * kKvHeads * kHeadDim, 0.0f);
        v.assign(std::size_t(kKeys) * kKvHeads * kHeadDim, 0.0f);
        for (int h = 0; h < kKvHeads; ++h) {
            for (int i = 0; i < kKeys; ++i) {
                for (int d = 0; d < kHeadDim; ++d) {
                    k[std::size_t(i * kKvHeads + h) * kHeadDim + d] = k_ggml[d + kHeadDim * (i + kKeys * h)];
                    v[std::size_t(i * kKvHeads + h) * kHeadDim + d] = v_ggml[i + kKeys * (d + kHeadDim * h)];
                }
            }
        }
        tq::reader::kv_buffers kv;
        kv.k = k.data();
        kv.v = v.data();
        kv.n_cell = kKeys;
        kv.n_head = kQHeads;
        kv.n_head_kv = kKvHeads;
        kv.head_dim = kHeadDim;
        return kv;
    }

    // Double-precision reference. `division` selects the head pairing, so the test can show
    // that the other candidate is NOT what ggml computes.
    std::vector<double> reference(bool division) const {
        std::vector<double> out(std::size_t(kHeadDim) * kQHeads, 0.0);
        for (int h = 0; h < kQHeads; ++h) {
            const int kh = division ? h / kGroup : h % kKvHeads;
            double scores[kKeys];
            double maximum = -1e300;
            for (int i = 0; i < kKeys; ++i) {
                double dot = 0.0;
                for (int d = 0; d < kHeadDim; ++d) {
                    dot += double(k_ggml[d + kHeadDim * (i + kKeys * kh)]) *
                           double(q_ggml[d + kHeadDim * h]);
                }
                scores[i] = dot * double(kScale);
                if (scores[i] > maximum) maximum = scores[i];
            }
            double denominator = 0.0;
            for (int i = 0; i < kKeys; ++i) {
                scores[i] = std::exp(scores[i] - maximum);
                denominator += scores[i];
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (int i = 0; i < kKeys; ++i) {
                    acc += scores[i] * double(v_ggml[i + kKeys * (d + kHeadDim * kh)]);
                }
                out[std::size_t(d + kHeadDim * h)] = acc / denominator;
            }
        }
        return out;
    }

    // ggml's own computation, one thread, identical graph both times.
    std::vector<float> ggml_attention(int * prec_accepted) const {
        struct ggml_init_params params{16u * 1024u * 1024u, nullptr, false};
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKeys, kKvHeads);
        ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kKeys, kHeadDim, kKvHeads);
        std::memcpy(q->data, q_ggml.data(), q_ggml.size() * sizeof(float));
        std::memcpy(k->data, k_ggml.data(), k_ggml.size() * sizeof(float));
        std::memcpy(v->data, v_ggml.data(), v_ggml.size() * sizeof(float));

        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        if (prec_accepted) {
            *prec_accepted = ggml_prec_set_acc(kq, GGML_PREC_BF16) ? 1 : 0;
        }
        ggml_tensor * sm = ggml_soft_max(ctx, ggml_scale(ctx, kq, kScale));
        ggml_tensor * out = ggml_mul_mat(ctx, v, sm);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_graph_compute_with_ctx(ctx, graph, 1);

        std::vector<float> result(std::size_t(kHeadDim) * kQHeads, 0.0f);
        std::memcpy(result.data(), out->data, result.size() * sizeof(float));
        ggml_free(ctx);
        return result;
    }
};

double max_abs_diff(const std::vector<float> & a, const std::vector<double> & b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const double delta = std::fabs(double(a[i]) - b[i]);
        if (delta > worst) worst = delta;
    }
    return worst;
}

void the_pipeline_and_the_mapping() {
    fixture f;
    const std::vector<float> ggml = f.ggml_attention(nullptr);
    const std::vector<double> division = f.reference(true);
    const std::vector<double> strided = f.reference(false);

    const double vs_division = max_abs_diff(ggml, division);
    const double vs_strided = max_abs_diff(ggml, strided);
    std::fprintf(stderr, "ggml vs double ref: division %.3e, strided %.3e\n", vs_division, vs_strided);

    // The reference and the kernels agree, which is what makes everything below meaningful.
    check(vs_division < 1e-6, "ggml's own attention matches the double reference (division mapping)");
    // And they disagree about the other candidate by seven orders of magnitude. Without this
    // the agreement above could be a coincidence of the fixture.
    check(vs_strided > 1e-2,
          "the strided head pairing is NOT what ggml computes (otherwise the check above is vacuous)");
}

void the_reader_matches_ggml() {
    fixture f;
    std::vector<float> k;
    std::vector<float> v;
    const tq::reader::kv_buffers kv = f.reader_buffers(k, v);
    const std::vector<float> ggml = f.ggml_attention(nullptr);

    // Identity rotation: this test is about the arithmetic, not about rope. A no-op rope is
    // the one function whose effect is known exactly, so any difference left is the
    // gathering, the head pairing, the softmax or the reduction.
    const tq::reader::rope_fn no_rope = [](float *, std::size_t, std::int32_t) {};

    const auto view = tq::reader::identity_view(0, kKeys, 0);
    const auto got = tq::reader::attend(f.q_ggml.data(), kv, view, kScale, no_rope);
    check(got.ok, "reader: attend succeeded");
    check(got.out.size() == std::size_t(kHeadDim) * kQHeads, "reader: one block per query head");
    check(got.n_visible == std::size_t(kKeys), "reader: every key was visible");

    // ggml's buffer is [head_dim, 1, n_head_q], so element (d, h) is at d + kHeadDim*h --
    // the same index as the reader's head-major out[h * head_dim + d].
    const double worst = max_abs_diff(got.out, std::vector<double>(ggml.begin(), ggml.end()));
    std::fprintf(stderr, "reader vs ggml: max|diff| = %.3e\n", worst);
    check(worst < 1e-6,
          "reader: matches ggml's kernels on llama's layouts (double vs float accumulation only)");

    // The same teeth as above, against the reader this time: the wrong pairing must be far
    // outside the tolerance, so this comparison is not satisfiable by any mapping.
    check(max_abs_diff(got.out, f.reference(false)) > 1e-2,
          "reader: the strided pairing is rejected by the same tolerance");
}

void the_precision_setting_does_not_change_cpu_results() {
    fixture f;
    const std::vector<float> off = f.ggml_attention(nullptr);
    int accepted = -1;
    const std::vector<float> on = f.ggml_attention(&accepted);

    // The API takes it. That is the trap: a caller sees the call succeed and concludes the
    // accumulator changed. Per ggml.h:1263-1270 the setting is a *minimum* type, and the CPU
    // mul_mat path never reads it (ggml-cpu.c has no ggml_prec_get_acc; only two
    // `prec_get_src` TODOs at :1291,:1586).
    check(accepted == 1, "ggml_prec_set_acc(GGML_PREC_BF16) is accepted by the API");

    std::size_t differing = 0;
    for (std::size_t i = 0; i < off.size() && i < on.size(); ++i) {
        if (std::memcmp(&off[i], &on[i], sizeof(float)) != 0) ++differing;
    }
    std::fprintf(stderr, "prec(BF16) vs off: %zu of %zu floats differ\n", differing, off.size());
    // On a CPU-only build the claim in attention.h is exactly this: bit-identical. A future
    // backend that honours the floor (CUDA with an f16 compute type) would break this test,
    // and that break would be information -- the tolerance for a cross-backend comparison is
    // a property of the backend, which is what the header now says.
    check(differing == 0,
          "on the CPU backend, requesting bf16 accumulation changes no bit (the setting is a floor, "
          "not a mandate)");
}

} // namespace

int main() {
    the_pipeline_and_the_mapping();
    the_reader_matches_ggml();
    the_precision_setting_does_not_change_cpu_results();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "reader attention matches ggml's kernels\n");
    return 0;
}
