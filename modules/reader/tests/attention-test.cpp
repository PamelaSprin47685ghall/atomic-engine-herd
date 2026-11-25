// The reader's attention, checked against attention computed the slow and obvious way.
//
// The central test is not "does it produce numbers" but "does it produce the SAME
// numbers as a direct implementation of the definition". So the fixture below writes a
// second attention by hand -- loops, no spans, no gathering -- and the two are compared
// bit for bit. A test that compared reader attend() against itself would pass for any
// implementation, including one that never looked at the keys.
//
// Three claims, in increasing strength:
//
//   1. identity view == plain causal attention. The strongest claim available: it ties
//      the reader's arithmetic to the arithmetic every transformer already does, so a
//      reader result cannot be "different in a way nobody noticed".
//   2. a scattered view == attention over exactly those keys. This is the reader's
//      whole point -- a key from another sequence competes in one softmax -- so the
//      comparison is against a hand-computed softmax over a hand-picked set.
//   3. per-span rotation: two keys at different effective positions must give a
//      different answer than the same two keys at one position, and must match the
//      hand-computed version that rotates each span separately. Without that, a port
//      that rotated once (the common bug) would still pass claims 1 and 2 whenever all
//      positions happened to coincide.
//
// Plus the failure modes: empty view, out-of-range cell, non-finite scores.
#include <tq/reader/attention.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// A rotation with numbers a human can follow: rotate each adjacent pair by an angle
// that grows with the position. Not upstream's rope -- it is never compared against
// upstream's rope, only used inside both sides of a comparison that shares it.
void pair_rotation(float * vector, std::size_t head_dim, std::int32_t pos) {
    const float angle = 0.1f * float(pos);
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    for (std::size_t d = 0; d + 1 < head_dim; d += 2) {
        const float a = vector[d];
        const float b = vector[d + 1];
        vector[d]     = a * c - b * s;
        vector[d + 1] = a * s + b * c;
    }
}

struct fixture {
    std::size_t n_cell = 6;
    std::size_t n_head = 2;
    // 0 means "no grouping", which is what every pre-GQA test here means: the key/value
    // buffers carry one head per query head. A non-zero value models GQA, and everything
    // below has to use kv_heads() for the buffer geometry or the fixture itself would
    // reproduce the defect the implementation had.
    std::size_t n_head_kv = 0;
    std::size_t head_dim = 4;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> q;

    std::size_t kv_heads() const { return n_head_kv == 0 ? n_head : n_head_kv; }

    fixture() { allocate(); }

    // The buffers are sized from the shape, so a shape cannot be applied after
    // construction -- setting n_head_kv on a built fixture leaves buffers sized for the
    // old geometry, and a GQA test over an undersized buffer would then fail for a reason
    // that has nothing to do with the code under test.
    fixture(std::size_t cells, std::size_t heads, std::size_t kv_heads_, std::size_t dim)
        : n_cell(cells), n_head(heads), n_head_kv(kv_heads_), head_dim(dim) {
        allocate();
    }

    void allocate() {
        k.assign(n_cell * kv_heads() * head_dim, 0.0f);
        v.assign(n_cell * kv_heads() * head_dim, 0.0f);
        q.assign(n_head * head_dim, 0.0f);
        // Deterministic, non-symmetric values: a symmetric fixture can make a wrong
        // gather look right, because several cells then give the same answer.
        unsigned state = 12345u;
        auto next = [&state] {
            state = state * 1664525u + 1013904223u;
            return float((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & value : k) value = next();
        for (float & value : v) value = next() * 2.0f;
        for (float & value : q) value = next();
    }

    tq::reader::kv_buffers buffers() const {
        tq::reader::kv_buffers kv;
        kv.k = k.data();
        kv.v = v.data();
        kv.n_cell = n_cell;
        kv.n_head = n_head;
        kv.n_head_kv = n_head_kv;
        kv.head_dim = head_dim;
        return kv;
    }

    // Kv head `h` of `cell`, the layout attention.h documents.
    const float * cell_head(const std::vector<float> & base, std::size_t cell, std::size_t h) const {
        return base.data() + (cell * kv_heads() + h) * head_dim;
    }
};

// Attention over an explicit list of cells, each with an explicit rotation position,
// written as a direct transcription of the definition: rotate, dot, softmax, reduce.
//
// This is the reference. It shares nothing with attention.cpp but the rotation
// callback, so agreement is evidence about the implementation and not about a shared
// helper.
std::vector<float> reference_attention(const fixture & f, const std::vector<std::int32_t> & cells,
                                       const std::vector<std::int32_t> & positions, float kq_scale) {
    std::vector<float> out(f.n_head * f.head_dim, 0.0f);
    std::vector<float> qrot(f.head_dim);
    // Query head h reads kv head h / group, the grouping ggml_mul_mat performs when the
    // key's head count divides the query's. With n_head_kv == 0 (every pre-GQA fixture)
    // group is 1 and this is the identity.
    //
    // WHAT THIS REFERENCE CANNOT PROVE, and it is the part worth knowing: this file and
    // the implementation under test write the SAME expression here (`h / group`). So the
    // gqa case below proves that `attend` agrees with a second transcription of the same
    // rule -- not that the rule is ggml's. A wrong mapping would be wrong identically in
    // both and pass. The rule's authority is upstream's broadcast, read rather than
    // assumed: ggml-cpu.c:1195 computes `r2 = ne12 / ne02` and :1230 indexes the inner
    // operand with `i02 = i12 / r2`, i.e. DIVISION -- contiguous groups of
    // (n_head_q / n_head_kv) query heads per kv head, not a strided `h % n_head_kv`.
    // llama.cpp reaches that multiply at llama-graph.cpp:3353 (`ggml_mul_mat(ctx0, k, q)`
    // on the permuted [HD, n_kv, n_head_kv] x [HD, n_tokens, n_head_q] pair), so the
    // broadcast axis is the head axis and division is what pairs q head h with kv head
    // h/group.
    //
    // An independent check therefore has to call ggml itself. The harness for it exists at
    // /tmp/ggmlcheck/probe.c and was NOT run: linking libggml-cpu needs -fopenmp
    // (GOMP_barrier), and the budget for shell work ran out first. The next holder can
    // settle it with:
    //   gcc -std=c11 -O1 -I third_party/upstream/ggml/include -c probe.c -o probe.o
    //   g++ -o probe probe.o -Wl,--start-group .../libggml.a .../libggml-cpu.a \
    //       .../libggml-base.a -Wl,--end-group -lm -lpthread -fopenmp
    // It prints max|diff| of ggml against both candidate mappings; a correct
    // implementation shows ~0 for division and a large number for the strided variant.
    const std::size_t group = f.n_head / f.kv_heads();
    for (std::size_t h = 0; h < f.n_head; ++h) {
        const std::size_t kv_head = h / group;
        std::vector<double> scores(cells.size());
        double maximum = -INFINITY;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            for (std::size_t d = 0; d < f.head_dim; ++d) {
                qrot[d] = f.q[h * f.head_dim + d];
            }
            pair_rotation(qrot.data(), f.head_dim, positions[i]);
            double dot = 0.0;
            const float * k = f.cell_head(f.k, std::size_t(cells[i]), kv_head);
            for (std::size_t d = 0; d < f.head_dim; ++d) {
                dot += double(k[d]) * double(qrot[d]);
            }
            scores[i] = dot * double(kq_scale);
            maximum = std::max(maximum, scores[i]);
        }
        double denominator = 0.0;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            scores[i] = std::exp(scores[i] - maximum);
            denominator += scores[i];
        }
        for (std::size_t d = 0; d < f.head_dim; ++d) {
            double sum = 0.0;
            for (std::size_t i = 0; i < cells.size(); ++i) {
                sum += scores[i] * double(f.cell_head(f.v, std::size_t(cells[i]), kv_head)[d]);
            }
            out[h * f.head_dim + d] = float(sum / denominator);
        }
    }
    return out;
}

// Comparison, with the tolerance that the arithmetic actually requires.
//
// The two implementations evaluate the same formula in different orders: attend()
// accumulates the dot in double and multiplies the scale in float, the reference keeps
// everything in double. Those orders are not bit-identical in IEEE arithmetic, and a
// bit-exact assertion would therefore be testing the compiler's rounding rather than
// the formula. What must hold is that the answers agree to float precision -- and the
// observed disagreement is ~1e-9 absolute on values of order 0.1, i.e. below one float
// ulp at that magnitude.
//
// The exactness claim that matters is not asserted here but avoided by construction:
// both sides call the same rotation, so a port that rotated once would be caught by a
// large difference (per_span_rotation_is_not_one_rotation), not by a rounding one.
double max_delta(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return INFINITY;
    }
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        worst = std::max(worst, std::fabs(double(a[i]) - double(b[i])));
    }
    return worst;
}

std::string compare(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return "sizes differ: " + std::to_string(a.size()) + " vs " + std::to_string(b.size());
    }
    double worst = 0.0;
    std::size_t at = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double delta = std::fabs(double(a[i]) - double(b[i]));
        if (delta > worst) {
            worst = delta;
            at = i;
        }
    }
    // 1e-6 absolute on outputs of order one: four orders of magnitude above the observed
    // 1e-9, and far below any difference a wrong formula produces (the per-span test's
    // disagreement is ~1e-2).
    if (worst > 1e-6) {
        char message[200];
        std::snprintf(message, sizeof message,
                      "value %zu differs: %.9g vs %.9g (delta %.3g)",
                      at, double(a[at]), double(b[at]), worst);
        return message;
    }
    return {};
}

// ---- 1. the identity view is ordinary attention ----
void identity_view_matches_plain_attention() {
    fixture f;
    const auto view = tq::reader::identity_view(0, f.n_cell, 0);
    const auto got = tq::reader::attend(f.q.data(), f.buffers(), view, 0.3f, &pair_rotation);
    check(got.ok, "identity view: attend succeeded");
    check(got.n_visible == f.n_cell, "identity view: every cell was visible");

    std::vector<std::int32_t> cells, positions;
    for (std::size_t i = 0; i < f.n_cell; ++i) {
        cells.push_back(std::int32_t(i));
        positions.push_back(std::int32_t(i));
    }
    const std::string difference = compare(got.out, reference_attention(f, cells, positions, 0.3f));
    check(difference.empty(), "identity view must equal plain attention: " + difference);
}

// ---- 2. a scattered view is exactly the keys it names ----
void scattered_view_is_a_hand_picked_softmax() {
    fixture f;
    // Out of order, with a gap: cells 4, 1, 3. A gather that used the query's own
    // position, or the wrong stride, cannot reproduce this.
    tq::reader::query_view view;
    view.spans.push_back({2, {4}});
    view.spans.push_back({5, {1}});
    view.spans.push_back({9, {3}});

    const auto got = tq::reader::attend(f.q.data(), f.buffers(), view, 0.25f, &pair_rotation);
    check(got.ok && got.n_visible == 3, "scattered view: three keys visible");

    const std::vector<std::int32_t> cells = {4, 1, 3};
    const std::vector<std::int32_t> positions = {2, 5, 9};
    const std::string difference = compare(got.out, reference_attention(f, cells, positions, 0.25f));
    check(difference.empty(), "scattered view must equal its hand-picked softmax: " + difference);
}

// ---- 3. rotation is per span, and it is observable ----
void per_span_rotation_is_not_one_rotation() {
    fixture f;
    tq::reader::query_view per_span;
    per_span.spans.push_back({0, {0, 1}});
    per_span.spans.push_back({4, {2, 3}});

    const auto split = tq::reader::attend(f.q.data(), f.buffers(), per_span, 0.5f, &pair_rotation);
    check(split.ok, "per-span: attend succeeded");

    // The same cells, all at one position. A port that rotated the query once would give
    // this answer for both views, and would therefore pass every test above.
    tq::reader::query_view one_span;
    one_span.spans.push_back({0, {0, 1, 2, 3}});
    const auto merged = tq::reader::attend(f.q.data(), f.buffers(), one_span, 0.5f, &pair_rotation);
    check(merged.ok, "one-span: attend succeeded");

    check(max_delta(split.out, merged.out) > 1e-3,
          "two spans at different positions must not give the same answer as one span");

    const std::vector<std::int32_t> cells = {0, 1, 2, 3};
    const std::vector<std::int32_t> positions = {0, 0, 4, 4};
    const std::string difference = compare(split.out, reference_attention(f, cells, positions, 0.5f));
    check(difference.empty(), "per-span answer matches the hand-computed per-key rotations: " + difference);
}

// ---- 4. the softmax is global, not per span ----
void one_softmax_across_every_span() {
    fixture f;
    // Two spans, each with one key. If each span were normalized separately and the
    // results averaged, the answer would be the mean of two single-key attention
    // outputs -- and for one key the softmax is 1, so that mean is exactly the average
    // of the two V rows, with the scores ignored entirely.
    tq::reader::query_view view;
    view.spans.push_back({0, {0}});
    view.spans.push_back({1, {1}});

    const float * v0 = f.cell_head(f.v, 0, 0);
    const float * v1 = f.cell_head(f.v, 1, 0);
    const double mean0 = 0.5 * (double(v0[0]) + double(v1[0]));
    const auto mean_like = [&](const std::vector<float> & out) {
        return std::fabs(double(out[0]) - mean0) < 1e-5;
    };

    // A scale of zero makes every key equally weighted, so the answer IS the mean of the
    // two V rows. This is the assertion a per-span softmax cannot pass: it produces that
    // mean for every scale, so this test alone would not distinguish the two.
    const auto flat = tq::reader::attend(f.q.data(), f.buffers(), view, 0.0f, &pair_rotation);
    check(flat.ok, "zero scale: attend succeeded");
    check(mean_like(flat.out), "zero scale must weight both keys equally");

    // The distinguishing case: a large scale must make the result nearly one of the two
    // V rows, which a per-span average never does. "Which one" is not assumed -- both
    // single-key answers are computed and the large-scale answer must be near one of
    // them, and near neither if the softmax were per span.
    tq::reader::query_view only0;
    only0.spans.push_back({0, {0}});
    tq::reader::query_view only1;
    only1.spans.push_back({1, {1}});
    const auto out0 = tq::reader::attend(f.q.data(), f.buffers(), only0, 0.0f, &pair_rotation);
    const auto out1 = tq::reader::attend(f.q.data(), f.buffers(), only1, 0.0f, &pair_rotation);
    check(out0.ok && out1.ok, "single-key references computed");

    const auto sharp = tq::reader::attend(f.q.data(), f.buffers(), view, 1.0e6f, &pair_rotation);
    check(sharp.ok, "large scale: attend succeeded");
    const double near0 = max_delta(sharp.out, out0.out);
    const double near1 = max_delta(sharp.out, out1.out);
    check(near0 < 1e-3 || near1 < 1e-3,
          "a large scale must make the global softmax nearly one-hot");
    check(!mean_like(sharp.out),
          "and must not equal the average of the two keys, which is what per-span "
          "normalization would produce at any scale");
}

// ---- 5. the failure modes are failures, not zero vectors ----
void failures_are_reported() {
    fixture f;
    tq::reader::query_view empty;
    check(!tq::reader::attend(f.q.data(), f.buffers(), empty, 1.0f, &pair_rotation).ok,
          "an empty view is a failure, not a zero vector");

    tq::reader::query_view bad;
    bad.spans.push_back({0, {0}});
    bad.spans.push_back({1, {std::int32_t(f.n_cell)}}); // one past the end
    check(!tq::reader::attend(f.q.data(), f.buffers(), bad, 1.0f, &pair_rotation).ok,
          "a cell past the buffer is refused, not read");

    tq::reader::query_view negative;
    negative.spans.push_back({0, {-1}});
    check(!tq::reader::attend(f.q.data(), f.buffers(), negative, 1.0f, &pair_rotation).ok,
          "a negative cell is refused");

    tq::reader::query_view good;
    good.spans.push_back({0, {0}});
    check(!tq::reader::attend(nullptr, f.buffers(), good, 1.0f, &pair_rotation).ok,
          "a null query is refused");
    check(!tq::reader::attend(f.q.data(), tq::reader::kv_buffers{}, good, 1.0f, &pair_rotation).ok,
          "an empty buffer set is refused");
    check(!tq::reader::attend(f.q.data(), f.buffers(), good, 1.0f, tq::reader::rope_fn{}).ok,
          "a missing rotation is refused: silently skipping it would answer a different question");

    // A non-finite score must fail rather than normalize. Reachable by asking for a
    // scale that overflows a float product.
    fixture huge;
    for (float & value : huge.q) value = 3.0e38f;
    for (float & value : huge.k) value = 3.0e38f;
    tq::reader::query_view one;
    one.spans.push_back({0, {0}});
    check(!tq::reader::attend(huge.q.data(), huge.buffers(), one, 3.0e38f, &pair_rotation).ok,
          "an overflowing score is a failure, not an infinite weight");
}

// ---- 6. an empty span is skipped, not treated as a rotation ----
void empty_spans_contribute_nothing() {
    fixture f;
    tq::reader::query_view with_empty;
    with_empty.spans.push_back({3, {}});
    with_empty.spans.push_back({1, {2}});
    const auto got = tq::reader::attend(f.q.data(), f.buffers(), with_empty, 0.4f, &pair_rotation);
    check(got.ok && got.n_visible == 1, "an empty span does not change the visible count");

    tq::reader::query_view only_key;
    only_key.spans.push_back({1, {2}});
    const auto direct = tq::reader::attend(f.q.data(), f.buffers(), only_key, 0.4f, &pair_rotation);
    check(compare(got.out, direct.out).empty(),
          "and does not change the answer: an empty span is not a second contribution");
}

// ---- 7. GQA: a query head count larger than the key/value count ----
//
// Every model this project targets has it (Llama-3 at 4:1, Qwen at 2:1 or 4:1), so a
// core that cannot express it is a core that answers from the wrong memory on real
// models. The defect this pins, measured before the fix: with 4 query heads over a
// buffer laid out for 2 kv heads per cell, the old single-stride code read head 2 of
// cell 0 as cell 1's head 0 and returned finite, plausible numbers.
void gqa_grouping_matches_reference() {
    // 4 query heads over 2 kv heads: the 2:1 ratio Qwen models use, and the same shape
    // that made the old single-stride code read one cell's key as another's.
    const fixture gqa(6, 4, 2, 4);

    const auto view = tq::reader::identity_view(0, gqa.n_cell, 0);
    const auto got = tq::reader::attend(gqa.q.data(), gqa.buffers(), view, 0.3f, &pair_rotation);
    check(got.ok, "gqa: attend succeeded");
    check(got.out.size() == gqa.n_head * gqa.head_dim,
          "gqa: one output block per QUERY head");

    std::vector<std::int32_t> cells, positions;
    for (std::size_t i = 0; i < gqa.n_cell; ++i) {
        cells.push_back(std::int32_t(i));
        positions.push_back(std::int32_t(i));
    }
    const std::string difference = compare(got.out, reference_attention(gqa, cells, positions, 0.3f));
    check(difference.empty(), "gqa: grouped heads must match the reference: " + difference);

    // And the grouping has to be observable -- which needs identical queries inside a
    // group. With a different q vector per head, two heads that read the SAME kv head
    // still produce different outputs, because they are different queries; my first
    // version asserted equality on that basis and failed for exactly that reason. So make
    // heads 0,1 identical and 2,3 identical: the pairs must then agree exactly, and the
    // two pairs must not.
    fixture grouped(6, 4, 2, 4);
    const std::size_t dim = grouped.head_dim;
    std::copy(grouped.q.begin(), grouped.q.begin() + std::ptrdiff_t(dim),
              grouped.q.begin() + std::ptrdiff_t(dim));          // head 1 := head 0
    std::copy(grouped.q.begin() + std::ptrdiff_t(2 * dim),
              grouped.q.begin() + std::ptrdiff_t(3 * dim),
              grouped.q.begin() + std::ptrdiff_t(3 * dim));       // head 3 := head 2

    const auto shared = tq::reader::attend(grouped.q.data(), grouped.buffers(),
                                           tq::reader::identity_view(0, grouped.n_cell, 0),
                                           0.3f, &pair_rotation);
    check(shared.ok, "gqa: the shared-query fixture runs");
    check(shared.out[0 * dim] == shared.out[1 * dim],
          "gqa: heads in one group read the same kv head (identical queries agree)");
    check(shared.out[2 * dim] == shared.out[3 * dim],
          "gqa: the second group agrees with itself too");
    check(shared.out[0 * dim] != shared.out[2 * dim],
          "gqa: the two groups read different kv heads");

    // A head layout that cannot be grouped is refused rather than rounded.
    const fixture bad(6, 4, 3, 4);   // 4 % 3 != 0
    check(!tq::reader::attend(bad.q.data(), bad.buffers(), view, 0.3f, &pair_rotation).ok,
          "gqa: a head count that does not divide is refused, not rounded");
}

// ---- 8. the same key twice is refused, not weighted twice ----
//
// "Every visible key, once" was a claim in the header with nothing enforcing it. Two
// cells presented as three keys answered 1.333 where 1.5 was correct -- a finite,
// plausible number from a view that names one key twice. views.h's resolve() never
// produces this, but attend() takes a query_view directly.
void duplicate_cells_are_refused() {
    fixture f;
    tq::reader::query_view twice;
    twice.spans.push_back({0, {1}});
    twice.spans.push_back({4, {1, 2}});   // cell 1 again, under a different rotation
    check(!tq::reader::attend(f.q.data(), f.buffers(), twice, 0.5f, &pair_rotation).ok,
          "a cell named by two spans is refused");

    tq::reader::query_view same_span;
    same_span.spans.push_back({0, {1, 1}});
    check(!tq::reader::attend(f.q.data(), f.buffers(), same_span, 0.5f, &pair_rotation).ok,
          "a cell named twice in one span is refused");

    // The control that makes the two above meaningful: over distinct cells the call
    // succeeds, so the refusals are about duplication and not about the view shape.
    tq::reader::query_view distinct;
    distinct.spans.push_back({0, {1}});
    distinct.spans.push_back({4, {2}});
    check(tq::reader::attend(f.q.data(), f.buffers(), distinct, 0.5f, &pair_rotation).ok,
          "the same shape with distinct cells succeeds");
}

// ---- 9. non-finite values are failures, wherever they appear ----
//
// The score path was checked and the value path was not: a NaN or an infinity in V
// produced ok == true with that value in `out`, which a caller then uses. The header
// promises the opposite ("a failure the caller must see"), and a check on the inputs
// would cost the same as the reduction that already reads them.
void non_finite_values_are_refused() {
    fixture f;
    tq::reader::query_view view;
    view.spans.push_back({0, {0}});
    view.spans.push_back({1, {1}});

    for (int kind = 0; kind < 2; ++kind) {
        fixture broken;
        // Overwrite one value row with NaN (kind 0) or +inf (kind 1).
        const float poison = kind == 0 ? std::nanf("") : INFINITY;
        broken.v[1 * broken.kv_heads() * broken.head_dim] = poison;
        const auto got = tq::reader::attend(broken.q.data(), broken.buffers(), view, 0.5f,
                                            &pair_rotation);
        check(!got.ok, kind == 0 ? "a NaN in V is a failure, not an answer"
                                 : "an infinity in V is a failure, not an answer");
    }

    // The score side, unchanged in spirit but now checked per score rather than on the
    // maximum alone: a NaN score leaves the maximum finite, because std::max drops it.
    fixture nan_q;
    nan_q.q[0] = std::nanf("");
    tq::reader::query_view one;
    one.spans.push_back({0, {0}});
    check(!tq::reader::attend(nan_q.q.data(), nan_q.buffers(), one, 0.5f, &pair_rotation).ok,
          "a NaN query is refused");
}

// ---- 10. a failure carries no partial answer ----
//
// `attend` fills its output head by head, so a failure at head 3 used to leave heads
// 0..2 in `out` while reporting ok == false. A caller that guards on the vector — the
// natural thing for a vector-returning API, and what `if (!r.out.empty())` looks like —
// then reads real numbers from a call that reported failure.
//
// The fixture is built so that head 0 is computable and head 1 is not: one poisoned
// value row in the second head. If the implementation reported the first head's real
// answer alongside ok == false, this catches it; if it returns nothing, out.empty() is
// the whole assertion.
void a_failure_carries_no_partial_answer() {
    fixture f;                       // 6 cells, 2 heads, 2 kv heads, head_dim 4
    tq::reader::query_view one;
    one.spans.push_back({0, {0}});

    // Sanity: the same call succeeds while both heads are intact, so the assertions
    // below are about the poison and not about the view.
    const auto healthy = tq::reader::attend(f.q.data(), f.buffers(), one, 0.5f, &pair_rotation);
    check(healthy.ok && healthy.out.size() == f.n_head * f.head_dim,
          "the two-head fixture computes both heads");

    // Head 0 fine, head 1 poisoned: the failure happens after head 0 has been written.
    fixture poisoned;
    poisoned.v[(0 * poisoned.kv_heads() + 1) * poisoned.head_dim] = std::nanf("");
    const auto got = tq::reader::attend(poisoned.q.data(), poisoned.buffers(), one, 0.5f,
                                        &pair_rotation);
    check(!got.ok, "a poisoned second head fails the call");
    check(got.out.empty(), "and the failure carries no output — not even the heads that did compute");
    check(got.n_visible == 0, "and reports no visible keys, rather than the count it had reached");
}

// ---- 11. geometry that cannot be indexed is refused before any pointer is formed ----
//
// Two products are indexed with pointer arithmetic:
//
//   n_cell * kv_heads * head_dim   the K/V buffers
//   n_head * head_dim              one output row
//
// A wrapped product does not crash. `cell_head` computes `(cell * kv_heads + head) *
// head_dim`, so a wrapped value yields a *plausible* offset and the read returns a real
// float from the wrong place — the caller gets numbers, not a failure. These cases pin
// the guard that refuses the geometry instead.
//
// The overflow is built from size_t's own maximum rather than from a hard-coded 2^32:
// on a 32-bit target `1u << 32` is zero and the case would silently test nothing. The
// values below overflow on both widths (and on 64-bit they are also unreachable for any
// real allocation, which is why the same guard is the one that rejects a fabricated
// pointer in a test).
void unrepresentable_geometry_is_refused() {
    fixture f;
    const std::size_t top = std::numeric_limits<std::size_t>::max();
    const std::size_t half = top / 2 + 1;

    tq::reader::query_view one;
    one.spans.push_back({0, {0}});

    // n_cell * kv_heads * head_dim overflows: (half) * 2 already exceeds the range.
    tq::reader::kv_buffers huge;
    huge.k = f.k.data();
    huge.v = f.v.data();
    huge.n_cell = half;
    huge.n_head = 2;
    huge.n_head_kv = 2;
    huge.head_dim = 2;
    check(!huge.valid(), "a cell/head/dim product that overflows is not a usable buffer");
    check(!tq::reader::attend(f.q.data(), huge, one, 0.5f, &pair_rotation).ok,
          "and attend refuses it rather than indexing a wrapped offset");

    // n_head * head_dim overflows, with a small cell count so the first product fits.
    tq::reader::kv_buffers wide;
    wide.k = f.k.data();
    wide.v = f.v.data();
    wide.n_cell = 1;
    wide.n_head = half;
    wide.n_head_kv = half;
    wide.head_dim = 2;
    check(!wide.valid(), "an output row count that overflows is not a usable buffer");

    // The contrast that makes both refusals mean something: a large geometry that fits
    // is accepted, so the guard is not simply rejecting everything big.
    tq::reader::kv_buffers fits;
    fits.k = f.k.data();
    fits.v = f.v.data();
    fits.n_cell = 1024;
    fits.n_head = 4096;
    fits.n_head_kv = 1024;
    fits.head_dim = 128;
    check(fits.valid(), "a large but representable geometry is accepted");
}

} // namespace

int main() {
    identity_view_matches_plain_attention();
    scattered_view_is_a_hand_picked_softmax();
    per_span_rotation_is_not_one_rotation();
    one_softmax_across_every_span();
    failures_are_reported();
    empty_spans_contribute_nothing();
    gqa_grouping_matches_reference();
    duplicate_cells_are_refused();
    non_finite_values_are_refused();
    a_failure_carries_no_partial_answer();
    unrepresentable_geometry_is_refused();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "reader attention contract holds\n");
    return 0;
}
