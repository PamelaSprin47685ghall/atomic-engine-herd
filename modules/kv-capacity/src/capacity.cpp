#include <tq/kv/capacity.h>

#include <limits>

namespace tq::kv {
namespace {

// Upstream's own padding, applied wherever it pads (src/llama-context.cpp:456 for
// n_ctx, and the fork's :976 for the physical capacity). Kept as a named function so
// the two call sites cannot drift.
uint32_t pad256(uint32_t value) {
    return (value + 255u) & ~255u;
}

} // namespace

const char * name(refusal why) {
    switch (why) {
        case refusal::none:                  return "ok";
        case refusal::requires_unified:      return "decoupled capacity requires a unified KV cache";
        case refusal::exceeds_range:         return "capacity exceeds the supported physical KV range";
        case refusal::unsupported_arch:      return "this architecture does not support a separate physical KV capacity";
        case refusal::arena_conflict:        return "separate capacity is not supported with a shared phase arena";
        case refusal::capacity_below_window: return "physical capacity does not cover one complete logical window";
        case refusal::split_would_round:     return "capacity is not an exact multiple of the sequence count and its padded window";
        case refusal::window_zero:           return "a zero logical window is not a configuration";
    }
    return "unknown";
}

const char * name(mode kind) {
    switch (kind) {
        case mode::unified:   return "unified";
        case mode::split:     return "split";
        case mode::decoupled: return "decoupled";
    }
    return "unknown";
}

plan evaluate(const request & req) {
    plan out;

    // A zero window cannot be padded into anything meaningful, and every check below
    // would divide by it.
    if (req.logical_window == 0) {
        out.why = refusal::window_zero;
        return out;
    }

    // Range before padding, for every request shape.
    //
    // pad256 adds 255 before masking, so any window above 2^32 - 256 wraps to a small
    // value -- and a wrapped window then compares as "fits", because the comparison
    // happens after the wrap. Measured before this guard: a window of 4294967290 was
    // accepted with logical = physical = 0, i.e. a configuration no allocator can
    // build was reported as legal and expressible. The fork bounded the same quantity
    // (:959-961); this check applies it to the logical window as well, which the fork
    // got for free because it validated the request where the value entered.
    if (req.logical_window > std::numeric_limits<int32_t>::max() - 255u) {
        out.why = refusal::exceeds_range;
        return out;
    }

    // `physical_capacity == 0` is the "no decoupling requested" spelling, the same
    // convention the fork used (`params.n_ctx_kv != 0` gates the whole block at
    // :955).
    const bool wants_separate = req.physical_capacity != 0 &&
                                req.physical_capacity != req.logical_window;

    // In a split cache the capacity is not a free parameter: upstream derives it as
    // n_ctx_seq * n_seq_max (:466-471). So a caller asking for more capacity than
    // that many windows is asking for the fork's separate field, which needed a
    // unified cache (:956-958) precisely because a split cache already spends the
    // capacity on sequences.
    const uint32_t aligned_window = pad256(req.logical_window);
    const uint32_t sequences      = req.n_seq_max == 0 ? 1u : req.n_seq_max;
    // 64-bit product on purpose: `aligned_window * sequences` overflows uint32 for a
    // large window with many sequences, and an overflowed product makes a request that
    // asks for far MORE capacity than the split shape can hold look like it asks for
    // less. The comparison decides whether to refuse, so it must not be computed in a
    // type that can wrap. The range guard above bounds each factor, not the product.
    if (!req.kv_unified &&
        static_cast<std::uint64_t>(req.physical_capacity) >
            static_cast<std::uint64_t>(aligned_window) * sequences) {
        out.why = refusal::requires_unified;
        return out;
    }

    // The fork's ordering: architecture and arena gates next, because they make the
    // request impossible regardless of numbers, then the range, then the padding.
    //
    // These gates belong to the unified shape only. A split cache reaches its extra
    // capacity by multiplying windows, so `n_ctx_kv requires a unified KV cache` says
    // nothing about it -- applying that gate here refused every legitimate split.
    if (wants_separate && req.kv_unified) {
        if (req.recurrent_arch || req.deepseek4) {
            // :962-964. Recurrent models keep one slot per sequence, so there is no
            // cell pool to enlarge.
            out.why = refusal::unsupported_arch;
            return out;
        }
        if (req.shared_arena) {
            // :965-967. A shared phase arena maps one buffer across phases; its size
            // is fixed by the arena, not by this number.
            out.why = refusal::arena_conflict;
            return out;
        }
        if (req.physical_capacity > std::numeric_limits<int32_t>::max() - 255u) {
            // :959-961. The fork bounded the value, not the product: a capacity that
            // does not fit the padded int32 range cannot be compared after rounding.
            out.why = refusal::exceeds_range;
            return out;
        }
    }

    // Padding. Upstream pads the window to 256 (:456); the fork padded the capacity
    // the same way unless the operator asked for the exact length (:976), and that
    // option exists because a 1024-aligned capacity is what lets attention use the
    // wider pad.
    // The physical capacity's range, checked immediately before it is padded and after
    // the architecture and arena gates -- so a recurrent model with an absurd capacity
    // hears "this architecture has no cell pool" first, because changing the number
    // would not help. Placed here rather than at the top for exactly that ordering, and
    // before pad256 for the wrapping reason above.
    if (req.physical_capacity > std::numeric_limits<int32_t>::max() - 255u) {
        out.why = refusal::exceeds_range;
        return out;
    }

    out.logical = pad256(req.logical_window);
    // What "no separate capacity requested" means depends on the shape, and getting
    // this wrong was a real defect (measured, not argued):
    //
    //   unified: capacity == window. One window over the whole cache.
    //   split:   capacity == n_seq_max windows. Upstream derives it, not the caller:
    //            `n_ctx_seq = PAD(n_ctx / n_seq_max)` and then `n_ctx = n_ctx_seq *
    //            n_seq_max` (llama-context.cpp:461-471). Asking for a 2048 window with
    //            three sequences is reachable by requesting n_ctx = 6144, and the
    //            runtime confirms it (measured: n_ctx=6000,S=3 -> n_ctx_seq=2048,
    //            n_ctx=6144).
    //
    // The earlier version collapsed capacity to the window in both shapes, so every
    // split request with no explicit capacity was refused as split_would_round --
    // while upstream builds it without complaint. A refusal that fires on a legal
    // configuration is worse than no check, because the caller cannot tell it from a
    // genuine limit.
    const uint32_t requested = wants_separate
                                   ? req.physical_capacity
                                   : (req.kv_unified ? out.logical : aligned_window * sequences);
    out.physical = req.runtime_length ? requested : pad256(requested);
    out.n_seq_max = req.n_seq_max == 0 ? 1u : req.n_seq_max;

    if (out.physical < out.logical) {
        // :977-979. Compared after padding, on purpose: a request of "window 4096,
        // capacity 4000" is below the window even though 4000 > 0.
        out.why = refusal::capacity_below_window;
        return out;
    }

    // Which shape is this?
    if (req.kv_unified) {
        // Unified upstream means one window over the whole capacity, so a split of
        // the capacity is not available in this mode. Distinct capacities are the
        // fork's third shape.
        out.kind = (out.physical == out.logical) ? mode::unified : mode::decoupled;
        out.expressible_upstream = (out.physical == out.logical);
    } else {
        // Split: upstream divides the capacity into n_seq_max windows and then forces
        // `n_ctx = n_ctx_seq * n_seq_max` (:466-471), rounding DOWN with a warning
        // when it does not divide. The capacity is therefore derived, not requested:
        // the window is the input, and S windows is the output. A caller that asked
        // for less capacity than S windows gets a rounding they did not choose, so it
        // is refused rather than honoured approximately.
        out.kind = mode::split;
        const uint32_t derived = aligned_window * sequences;
        if (out.physical < derived) {
            out.why = refusal::split_would_round;
            return out;
        }
        out.physical = derived;
        out.logical  = aligned_window;
        out.n_seq_max = sequences;
        out.expressible_upstream = true;
    }

    // Capacity in whole windows, floored. `1` means "exactly one window", `2` means
    // one window plus a spare window's worth of cells -- the quantity the feature
    // exists to provide, and the number to print instead of raw bytes.
    out.spare_windows = out.logical == 0 ? 0 : out.physical / out.logical;
    return out;
}

std::string explain(const plan & result) {
    if (result.why != refusal::none) {
        return std::string("refused: ") + name(result.why);
    }
    std::string text = std::string("accepted as ") + name(result.kind) +
                       ": window " + std::to_string(result.logical) +
                       ", capacity " + std::to_string(result.physical) +
                       " (" + std::to_string(result.spare_windows) + " window(s))";
    if (result.expressible_upstream) {
        text += ", expressible with stock parameters";
    } else {
        // Always said, never implied: this is the case a caller is most likely to
        // assume succeeded, because the numbers look like what they asked for.
        text += ", NOT expressible with stock parameters: the capacity is larger than the "
                "logical window with a unified cache, which only the fork's n_ctx_kv could do";
    }
    return text;
}

} // namespace tq::kv
