// tq/reader/attention.h — the reader's attention, as arithmetic.
//
// What the fork computes (src/llama-graph.cpp:3613-3712, build_attn_reader), reduced
// to what it actually does to numbers:
//
//   for each query row i:
//     for each span in the view:
//       scores_span = K[span.indices] · rope(Q_i, span.effective_pos)
//     all_scores   = concat(scores_span …)        // every visible key, once
//     weights      = softmax(all_scores * kq_scale)  // ONE softmax over all spans
//     out_i        = Σ_j weights_j * V[cell_j]
//
// Three things make it different from ordinary attention, and each is a line here:
//
//   * the key set is *chosen* (span.indices), not "whatever the sequence holds";
//   * the query is rotated per span (span.effective_pos), not once;
//   * the softmax spans every chosen key from every span, so a key from another
//     sequence competes with the query's own history in one normalization.
//
// The rotation itself is a callback. That is deliberate: this core owns the
// gathering, the concatenation, the normalization and the value reduction, and it
// must be tested with a rotation whose numbers are known. The fork's rotation is
// ggml_rope_ext with a dozen parameters (yarn factors, attn scale); re-deriving that
// here would be a second implementation of upstream's convention, and a test that
// compared my rope against my rope would prove nothing. So the caller supplies the
// rotation, and the identity test below uses one where the answer is computable by
// hand.
//
// Deliberately absent: any ggml type, any graph, any allocation policy. This is what
// the values must be, not how they are scheduled.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <vector>

namespace tq::reader {

// One visible key, already resolved to physical storage by views.h's resolve():
// `cell` indexes the K and V buffers, `effective_pos` is the rope position this key
// pretends to occupy for this query.
struct key_ref {
    std::int32_t cell = -1;
    std::int32_t effective_pos = 0;
};

// One query's visible set. Consecutive keys with the same effective_pos share a
// rotation — that is the span abstraction views.h produces, and the reason the fork
// rotates per span rather than per key.
struct span_keys {
    std::int32_t          effective_pos = 0;
    std::vector<std::int32_t> cells;
};

struct query_view {
    std::vector<span_keys> spans;
};

// Rotates `vector` (one head's q, length head_dim) in place for rope position `pos`.
//
// One head at a time: the fork rotates a 3-D query view and lets ggml broadcast over
// heads, but the rotation itself never mixes heads, so this signature is the same
// computation with the head loop made explicit.
using rope_fn = std::function<void(float * vector, std::size_t head_dim, std::int32_t pos)>;

// Flat K and V storage: key `cell`, kv head `h`, dimension `dim` is at
//   k[(cell * n_head_kv + h) * head_dim + dim]
// which is the fork's k_flat layout reshaped as (head_dim, n_head_kv, n_cell) — head
// major within a cell, cells contiguous — so the gather is a copy of `head_dim`
// contiguous floats.
//
// TWO head counts, not one, because every model this project targets has GQA. The
// query carries n_head heads, K and V carry n_head_kv <= n_head, and query head h reads
// kv head h / (n_head / n_head_kv) — the same grouping ggml_mul_mat performs when
// b->ne[1] is a multiple of a->ne[1]. Measured before this was a field: passing the
// query's head count as the single `n_head` made head 2 of cell 0 read cell 1's head 0
// (the stride was `cell * n_head + head` with n_head=4 over a buffer laid out for 2
// heads per cell), so a GQA model did not fail — it answered from the wrong keys, and
// past the last cell it read beyond the buffer.
//
// `n_head_kv == 0` means "no grouping": n_head_kv is taken as n_head, which is the
// multi-head case and what every pre-GQA caller means. A non-zero value must divide
// n_head; the grouping is otherwise undefinable, and the call fails rather than
// rounding the group size.
//
// `head_dim` is ONE number for q, K and V. Upstream allows the query's head width to
// differ from the value's (n_embd_head_k vs n_embd_head_v, as MLA does); this core does
// not model that, so a caller with such a model must not use it. Named here because it
// is a real limit, not an oversight: the alternative is a second stride that no model in
// this project's test set can exercise, and untested flexibility is worse than a stated
// boundary.
struct kv_buffers {
    const float * k = nullptr;   // [n_cell * n_head_kv * head_dim]
    const float * v = nullptr;   // same shape
    std::size_t   n_cell    = 0;
    std::size_t   n_head    = 0;  // query heads
    std::size_t   n_head_kv = 0;  // key/value heads; 0 means "same as n_head"
    std::size_t   head_dim  = 0;

    // True when a * b * c is representable in size_t. Written out rather than left to
    // the multiplication, because a wrapped product is the worst kind of wrong here:
    // `cell_head` would compute a plausible offset into the buffer and read a real float
    // from the wrong place, so the caller gets numbers instead of a failure.
    static bool product_fits(std::size_t a, std::size_t b, std::size_t c) {
        constexpr std::size_t top = std::numeric_limits<std::size_t>::max();
        if (b != 0 && a > top / b) {
            return false;
        }
        const std::size_t ab = a * b;
        if (c != 0 && ab > top / c) {
            return false;
        }
        return true;
    }

    bool valid() const {
        if (!k || !v || n_cell == 0 || n_head == 0 || head_dim == 0) {
            return false;
        }
        const std::size_t kv_heads = n_head_kv == 0 ? n_head : n_head_kv;
        if (kv_heads > n_head || n_head % kv_heads != 0) {
            return false;
        }
        // Two products the loop below indexes with, both checked before any pointer is
        // formed:
        //
        //   n_cell * kv_heads * head_dim   the K and V buffers' element count
        //   n_head * head_dim              one output row
        //
        // On a 64-bit target the first needs 2^64 elements to wrap, which no allocation
        // reaches -- but the check is not defensive theatre: on a 32-bit target
        // 65536 cells x 256 heads x 256 floats is exactly 2^32, and 64k cells of a
        // 256-head model is an ordinary large-context configuration. The same guard also
        // closes the fabricated-pointer case a test can construct on any target.
        if (!product_fits(n_cell, kv_heads, head_dim)) {
            return false;
        }
        if (!product_fits(n_head, head_dim, 1)) {
            return false;
        }
        return true;
    }

    std::size_t kv_heads() const { return n_head_kv == 0 ? n_head : n_head_kv; }
};

// Result of one query's attention: `head_dim` floats per head, laid out
// out[head * head_dim + dim].
struct attention_result {
    std::vector<float> out;      // n_head * head_dim
    std::size_t        n_visible = 0;  // keys the softmax saw
    bool               ok = false;      // false when the view was empty or invalid
};

// The fork's computation, for one query row and one head set.
//
// `q` is the query *before* any rope, one head per `head_dim` block: the fork
// re-derives every rotated query from `q_rope->src[0]` for exactly this reason
// (:3627-3629) — rotating an already-rotated vector would compound the scaling.
//
// ACCUMULATION PRECISION — read this before comparing against stock numbers, and do not
// infer the tolerance from the setting alone.
//
// This function accumulates each score in double and rounds once. Stock attention sets
// `ggml_prec_set_acc(kq, GGML_PREC_BF16)` on the KQ product (llama-graph.cpp:3355), and
// the natural reading — "stock therefore accumulates in bf16, so I may compare at 1e-2"
// — is wrong. What that call means, from ggml.h:1263-1270: it sets the *minimum required*
// accumulator type. "GGML_PREC_BF16 - can accumulate the results in BF16, F32". It is a
// floor, not a mandate: bf16 or better, and F32 already satisfies it.
//
// Where the floor actually lands, per backend:
//
//   CPU, mul_mat     NEVER CONSULTED. ggml-cpu has no `ggml_prec_get_acc` in the mul_mat
//                    path at all (only two TODO comments about `prec_get_src` in
//                    ggml-cpu.c:1291,1586), so the KQ product is accumulated as F32
//                    vec_dot regardless of the setting.
//   CPU, flash_attn  UNDEFINED, F32 and BF16 all dispatch to the same implementation,
//                    whose own comment says "uses F32 accumulators"
//                    (ggml-cpu/ops.cpp:9589-9596).
//   CUDA             a genuine floor: `if (prec == BF16 && compute_type == F16)
//                    compute_type = fast_bf16_hardware_available ? BF16 : F32;`
//                    (ggml-cuda.cu:2203-2205) — "F16 is the only compute type that can
//                    not satisfy a request for BF16". F32 stays F32; bf16 is an upgrade
//                    from f16, never a downgrade from f32.
//   Metal            same shape (ggml-metal-device.m:1532-1535 selects per prec).
//
// MEASURED ON THIS PIN (CPU backend, tests/attention-ggml-test.cpp, registered as
// tq_reader.attention_ggml):
//
//   ggml_prec_set_acc(BF16) returns 1 (the API accepts it)
//   prec on vs off                      0 of 32 outputs differ, bit for bit
//   ggml(F32) vs double ref, h/group    5.9e-08   float rounding only
//   ggml(F32) vs double ref, h%n_head_kv 4.2e-01  seven orders larger
//   reader vs ggml, same layouts        6.0e-08
//
// The first two lines are the trap made concrete: the call succeeds and nothing changes, so
// "I requested bf16 and got it" is false on this backend. The middle pair is what makes the
// last line evidence rather than a tautology -- the comparison separates the two candidate
// head pairings by seven orders of magnitude, so a wrong mapping cannot pass it.
//
// So the honest statement is narrower than "the reader is more accurate": on the CPU
// backend — where every probe in this repository runs — stock accumulates in F32, and the
// reader's double accumulation differs from it at float-rounding scale (relative ~1e-7 on
// the scores), not at bf16 scale. A 1e-2 tolerance there would hide real errors. On CUDA
// or Metal with an f16 compute type the difference can reach bf16 scale, and the same
// tolerance would then be right. The tolerance is a property of the backend and kernel,
// not of this header.
//
// What this does not change: an acceptance comparison should still be against a
// double-precision reference (what the unit test does), or bit-identical within one
// backend with the same setting on both sides. Comparing reader output against stock
// across backends measures the backends, not the reader.
//
// Every guarantee below is an ok == false, never a plausible number:
//
//   * an empty view — the softmax of no keys is undefined, not zero;
//   * a cell index outside [0, n_cell), or a negative one;
//   * a head layout that cannot be grouped (n_head_kv not dividing n_head);
//   * the same cell twice in the view. The header above says "every visible key,
//     once"; a repeated cell is weighted twice, which measurably changes the answer
//     (two cells counted as three gave 1.333 where 1.5 was correct) and looks like an
//     ordinary result. views.h's resolve() never produces duplicates, but this function
//     takes a query_view directly, so the invariant is checked here rather than assumed;
//   * a non-finite score (the product overflowed, or q/K hold NaN or infinity);
//   * a non-finite output — which is how NaN or infinity in V shows up. Checking V
//     element by element would cost the same as the reduction that already reads it, and
//     the output is the value the caller consumes.
//
// Duplicate detection costs one sort of the gathered indices: O(n log n) against the
// n * head_dim * n_head multiply-adds the same call performs (a 8k-key view with 32
// heads of width 128 is 32M flops against ~100k comparisons).
attention_result attend(const float * q,
                        const kv_buffers & kv,
                        const query_view & view,
                        float kq_scale,
                        const rope_fn & rope);

// The identity view: every cell of the sequence, in ascending order, each at its own
// position — which is exactly what a stock causal attention over that sequence sees,
// with virtual_pos == key_pos and one span per distinct position.
//
// Provided because it is the only way to state the central claim as a test: reader
// attention over the identity view must equal ordinary attention over the same keys.
query_view identity_view(std::int32_t first_cell, std::size_t n_keys,
                         std::int32_t first_pos);

} // namespace tq::reader
