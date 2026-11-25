#include "tq/reader/arena_budget.h"

#include <algorithm>
#include <limits>

namespace tq::reader {
namespace {

// The largest value a byte count can take, used as the saturation point.
constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();

// Saturating, overflow-detecting products.
//
// A budget that wrapped would come out SMALLER than the truth, which is the one direction a bound
// must never err in: the caller would read a small number, conclude there is room, and let
// ggml_new_tensor_impl assert on an overfull arena (ggml.c:1832 -- it aborts in every build, so
// the failure is a dead process, not a returned null). So overflow is recorded and the value is
// clamped at the maximum, where every arena refuses it.
//
// `std::size_t` factors rather than a variadic form: an initializer_list keeps the call sites
// readable (which product is being taken, in the order the shapes multiply) with one bounded
// loop, no allocation, and no template machinery in a header that must stay allocation-free.
struct tally {
    std::size_t value = 0;
    bool overflowed = false;

    void add_product(std::initializer_list<std::size_t> factors) {
        if (overflowed) {
            value = kMax;
            return;
        }
        std::size_t product = 1;
        for (const std::size_t factor : factors) {
            // `factor != 0 && product > kMax / factor` is the exact test: the division is only
            // taken when it is defined, and it is the only form that does not overflow itself.
            if (factor != 0 && product > kMax / factor) {
                overflowed = true;
                value = kMax;
                return;
            }
            product *= factor;
        }
        if (value > kMax - product) {
            overflowed = true;
            value = kMax;
            return;
        }
        value += product;
    }
};

} // namespace

// Per-head setup nodes, counted generously.
//
// Reads out of graph_emit.cpp, per kv head:
//   K: ggml_view_3d + ggml_cont + (widen ? ggml_cast)                     = 2 or 3
//   V: ggml_view_3d + (cell-major ? ggml_transpose : nothing) + ggml_cont
//      + (widen ? ggml_cast)                                              = 2..4
//
// The estimator does not know the V layout (it is not in the inputs, because a budget should not
// depend on a fact the caller could get wrong), so it charges the cell-major form for both. Two
// nodes of slack per head, on top: a bound that is exactly right is a bound that breaks when
// someone adds a node.
constexpr std::size_t kSetupNodesPerHead = 11;

// Per-span, per-head nodes: get_rows (K) + concat (K, except the first head) + get_rows (V) +
// transpose (V) + cont (V) + concat (V, except the first head) = 6, +2 slack.
constexpr std::size_t kSpanNodesPerHead = 8;

// Per-span nodes that do not scale with heads: the index tensor, mul_mat, scale, the two
// cross-span concatenations, and one node of rope, plus slack.
constexpr std::size_t kSpanNodesFixed = 9;

// Fixed nodes after the spans: soft_max, mul_mat, and the query's cast.
constexpr std::size_t kTailNodes = 4;

// Widening makes an operand cost four bytes per element whatever the cache held, so the copy term
// is the larger of the cache's element size and four. (A helper for this existed and was used
// once; a function that normalises one number is clearer as the expression it stands for.)

drift estimate_drift(const arena_budget & budget, std::size_t used_before,
                     std::size_t used_after) {
    // A status other than ok carries no number to compare against. `unsupported` computed nothing;
    // the saturated ones mean "do not emit", and treating SIZE_MAX as a roomy allowance would
    // invert their purpose. `unusable`, not `exceeded`: this is not a verdict that the emission
    // cost too much, it is a statement that nothing was measured.
    if (budget.status != budget_status::ok) {
        return drift::unusable;
    }
    // A shrinking context is not a negative cost.
    //
    // gap = used_after - used_before would wrap to a huge number and pass every comparison, which
    // is the failure mode this whole module keeps removing. Two readings are possible and both are
    // caller errors: the measurements came from different contexts, or something reset the arena
    // between them. Neither is a margin.
    if (used_after < used_before) {
        return drift::unusable;
    }
    return used_after - used_before <= budget.total_bytes ? drift::within : drift::exceeded;
}

const char * describe(drift value) {
    switch (value) {
        case drift::within:   return "the emission fit inside the estimate";
        case drift::exceeded: return "the emission cost more than the estimate: the inventory this "
                                     "estimate enumerates has drifted from what the emitter creates";
        case drift::unusable: return "the comparison cannot be made: the budget is not ok, or the "
                                     "two measurements do not come from the same arena";
    }
    return "unknown drift value";
}

arena_budget estimate(const budget_inputs & in) {
    arena_budget out;

    // Nothing describable: no shapes, no spans, no cost fact, or no storage to read the span
    // counts from. `unsupported`, not `ok`: nothing was computed, so the zero total must not be
    // read as a cost.
    if (in.head_dim == 0 || in.n_head_kv == 0 || in.n_cell == 0 || in.n_head == 0 ||
        in.n_spans == 0 || in.tensor_overhead == 0 || in.cells_per_span == nullptr) {
        out.status = budget_status::unsupported;
        return out;
    }

    // An injected rotation with no declared node count has no bound. Refused BEFORE any
    // arithmetic, because the number that follows would be a bound on a computation this
    // function cannot see.
    if (in.has_rope) {
        out.status = budget_status::unbounded_rope;
        out.total_bytes = kMax;
        return out;
    }

    // Every span the caller reported, not the first eight.
    //
    // The previous version clamped to its fixed array. Clamping is the one error a bound cannot
    // make: the emitter iterates `view.spans` with no limit, so a ninth span was costed as if it
    // did not exist.
    const std::size_t spans = in.n_spans;

    // ---- pass 1: prefix sums, and the concatenation chains ----
    //
    // The emitter builds its unions with `scores = scores ? ggml_concat(scores, kq, 0) : kq`
    // (graph_emit.cpp:347), so span i creates a node whose result holds EVERY cell seen so far.
    // The data allocated by that chain is therefore the sum of the prefixes, not the union once:
    // for n equal spans it is n(n+1)/2 times a span, which is quadratic. The earlier version
    // charged the union a single time, so it under-estimated by ~n/2 -- invisible in the tests it
    // shipped with, because those used at most two spans.
    //
    // The same shape appears one level down: the per-head K/V blocks are concatenated on ne2, so
    // head h produces a tensor of (h+1) head-slices. That chain is quadratic in n_head_kv.
    tally total_cells;
    tally span_concat_cells;   // sum of prefixes, over spans (index > 0)
    std::size_t prefix = 0;
    for (std::size_t s = 0; s < spans; ++s) {
        const std::size_t cells = in.cells_per_span[s];
        if (cells > kMax - prefix) {
            out.status = budget_status::overflow;
            out.total_bytes = kMax;
            return out;
        }
        prefix += cells;
        total_cells.add_product({cells});
        if (s > 0) span_concat_cells.add_product({prefix});
        // Both tallies are checked, and `span_concat_cells` in its own right.
        //
        // Checking only `total_cells` happened to be sufficient -- a prefix that overflows is a
        // prefix sum that also overflows a few steps later -- but it required tracing a second
        // variable to see why the first check covered the second accumulator. `tally` records its
        // own overflow, so reading it is free, and a reader no longer has to run that argument.
        if (total_cells.overflowed || span_concat_cells.overflowed) {
            out.status = budget_status::overflow;
            out.total_bytes = kMax;
            return out;
        }
    }
    if (total_cells.value == 0) {
        out.status = budget_status::unsupported;
        return out;
    }

    // Sum over h = 1..n_head_kv-1 of (h+1): the total head-slices materialised by the ne2 chain.
    //
    // Closed form, because the loop that used to compute it was unbounded in its own right: with
    // n_head_kv == SIZE_MAX it would iterate 1.8e19 times. That is a hang, not a slow path, and it
    // is reachable from a caller that reports a nonsense head count -- which is exactly the kind
    // of input a budget is asked about. Summing k-1 terms that run from 2 to k gives (k-1)(k+2)/2,
    // which is three operations and an explicit overflow test.
    tally head_chain;
    if (in.n_head_kv >= 2) {
        // `k + 2` itself can wrap.
        //
        // The closed form was introduced to remove an unbounded loop, and it reintroduced the same
        // class of error one step in: with n_head_kv within two of SIZE_MAX, `in.n_head_kv + 2`
        // wraps to 0 or 1, and the product then collapses to zero -- a bound BELOW the truth, in
        // the one direction that must never happen. The guard is checked first, so no wrapped
        // intermediate is ever formed.
        if (in.n_head_kv > kMax - 2) {
            out.status = budget_status::overflow;
            out.total_bytes = kMax;
            return out;
        }
        head_chain.add_product({in.n_head_kv - 1, in.n_head_kv + 2});
        if (head_chain.overflowed) {
            out.status = budget_status::overflow;
            out.total_bytes = kMax;
            return out;
        }
        // Exact: of two consecutive integers (k-1) and (k+2) one is even, so the product is
        // divisible by two and the halving loses nothing.
        head_chain.value /= 2;
    }
    const std::size_t head_chain_slices = head_chain.value;

    // ---- objects ----
    //
    // The per-head setup happens once per emission (the gathered per-head tensors are reused by
    // every span), which is why it is not multiplied by `spans`. The rope term is the caller's
    // declared bound, charged per span because the emitter calls it once per span.
    tally objects;
    objects.add_product({kTailNodes});
    objects.add_product({in.n_head_kv, kSetupNodesPerHead});
    objects.add_product({spans, kSpanNodesFixed});
    objects.add_product({spans, in.n_head_kv, kSpanNodesPerHead});
    // No rope term: `has_rope` returned at the top of this function. The line that used to charge
    // `spans * rope_nodes_per_span` is gone with the field it read, and it is not replaced by a
    // constant -- a rotation's cost is chosen by code this function cannot see, which is why the
    // whole input is refused rather than estimated.
    if (objects.overflowed) {
        out.status = budget_status::overflow;
        out.total_bytes = kMax;
        return out;
    }
    out.objects = objects.value;

    // ---- data ----
    //
    // Only under an allocating context: ggml.c:1825 allocates data only when `!ctx->no_alloc`,
    // and a view never allocates at all (view_src != NULL). So a no_alloc emission costs objects
    // and nothing else -- which is also why the emitter cannot run there at all: it writes its
    // span indices into `ggml_new_tensor_1d(I32, ...)->data`, and that pointer is null under
    // no_alloc (measured; tests/arena-budget-test.cpp asserts the refusal).
    if (!in.no_alloc) {
        // The copies are charged at the WIDER of the cache's element size and four bytes: a
        // widened operand is copied at its own type and then cast, so both sizes are real and the
        // larger one bounds the pair. (q needs no such term -- its cast output is counted below.)
        const std::size_t k_elem = std::max<std::size_t>(in.k_elem, 4);
        const std::size_t v_elem = std::max<std::size_t>(in.v_elem, 4);

        // The emitter's arithmetic is in F32 before it multiplies, so every intermediate it
        // materialises is four bytes per element regardless of the cache type.
        const std::size_t fp32 = 4;

        tally data;

        // The widening casts: one full copy of the cache per head, both operands.
        if (in.widen_k) data.add_product({in.n_head_kv, in.head_dim, in.n_cell, fp32});
        if (in.widen_v) data.add_product({in.n_head_kv, in.head_dim, in.n_cell, fp32});

        // The per-head contiguous copies. Both are made even without widening (get_rows needs a
        // contiguous axis), so this is charged unconditionally.
        //
        // TWO products, not one with `k_elem + v_elem` as a factor.
        //
        // The sum can wrap: two element sizes at SIZE_MAX/2 each add to a small number, the product
        // then collapses, and the bound lands below the truth -- fail-open, the one direction this
        // function must never take. Splitting also states what is being counted: one copy of K
        // plus one copy of V, each at its own element size. The sum of two upper bounds is an upper
        // bound of the sum, so nothing is lost by the split.
        data.add_product({in.n_head_kv, in.head_dim, in.n_cell, k_elem});
        data.add_product({in.n_head_kv, in.head_dim, in.n_cell, v_elem});

        // The query: one cast when it is widened. (No rope term: `has_rope` returned at the top.)
        if (in.widen_q) data.add_product({in.n_head, in.head_dim, fp32});

        // Per span: the index tensor (I32, written by the emitter), the gathered rows per head,
        // the scores and their scaled copy.
        for (std::size_t s = 0; s < spans; ++s) {
            const std::size_t cells = in.cells_per_span[s];
            data.add_product({cells, 4});
            data.add_product({in.n_head_kv, in.head_dim, cells, fp32});   // gathered K rows
            data.add_product({in.n_head_kv, cells, in.head_dim, fp32});   // gathered V rows
            data.add_product({cells, in.n_head, fp32, 2});                // scores + scaled
            // The head-concatenation chains: each produced a tensor of (h+1) slices.
            data.add_product({in.head_dim, cells, head_chain_slices, fp32});
            data.add_product({cells, in.head_dim, head_chain_slices, fp32});
        }

        // The cross-span unions, prefix by prefix (quadratic in spans; see pass 1).
        data.add_product({span_concat_cells.value, in.n_head, fp32});
        data.add_product({span_concat_cells.value, in.head_dim, in.n_head_kv, fp32});

        // The softmax output and the final multiply.
        data.add_product({total_cells.value, in.n_head, fp32});
        data.add_product({in.head_dim, in.n_head, fp32});

        if (data.overflowed) {
            out.status = budget_status::overflow;
            out.total_bytes = kMax;
            return out;
        }

        // A quarter of slack, on top of the per-node slack above. Two independent margins, both
        // because the bound's job is to be above the measurement on shapes nobody has tried yet.
        out.data_bytes = data.value + data.value / 4;
        if (out.data_bytes < data.value) {   // the slack itself overflowed
            out.status = budget_status::overflow;
            out.total_bytes = kMax;
            return out;
        }
    }

    // ---- total ----
    //
    // Per-object overhead plus data, then an alignment allowance of GGML_MEM_ALIGN (16) per
    // object plus one:
    //
    //   * ggml rounds every object up to GGML_MEM_ALIGN, and `ggml_tensor_overhead()` already
    //     includes that rounding for the tensor struct -- but the object header and the tensor are
    //     laid out inside one allocation whose size is aligned per allocation, so a per-object
    //     allowance is the safe reading rather than a claim about ggml's internals;
    //   * the data region ends on an alignment boundary too, which is what the trailing +1 covers.
    //
    // The earlier single `+ 64` was a fixed pad that happened to cover the tail of one small
    // graph; with hundreds of objects it under-covers, and under-covering is the wrong direction.
    tally total;
    total.add_product({out.objects, in.tensor_overhead});
    total.add_product({out.data_bytes});
    total.add_product({out.objects, 16});
    total.add_product({16});
    if (total.overflowed) {
        out.status = budget_status::overflow;
        out.total_bytes = kMax;
        return out;
    }

    out.total_bytes = total.value;
    out.status = budget_status::ok;
    return out;
}

} // namespace tq::reader
