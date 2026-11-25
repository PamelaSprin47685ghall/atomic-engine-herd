// The capacity station's decisions, tested without a model, a cache, or a context.
//
// Each case pins one ruling a driver would otherwise re-derive by reading the fork.
// The refusals matter as much as the acceptances: a driver that cannot tell "legal
// but unexpressible" from "illegal" either refuses work it could do, or runs
// something other than what it was asked for.
//
// Two expectations were wrong on the first run and are corrected here, because the
// corrections are the semantics: padding happens BEFORE comparison (so 8000 cells
// against four 2048 windows is exactly 8192 and legal), and "a split cache with
// exactly S windows of capacity" is not a decoupled request at all -- it is the
// ordinary split configuration. The gates only fire when a caller wants more capacity
// than the split shape can spend on sequences.
#include <tq/kv/capacity.h>

#include <cstdio>
#include <string>

using namespace tq::kv;

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

request plain(uint32_t window) {
    request r;
    r.logical_window = window;
    return r;
}

request decoupled(uint32_t window, uint32_t capacity) {
    request r = plain(window);
    r.physical_capacity = capacity;
    return r;
}

request split(uint32_t window, uint32_t capacity, uint32_t sequences) {
    request r = decoupled(window, capacity);
    r.kv_unified = false;
    r.n_seq_max  = sequences;
    return r;
}

void plain_configuration() {
    const plan p = evaluate(plain(4096));
    check(p.why == refusal::none, "a plain window is accepted");
    check(p.kind == mode::unified, "with a unified cache it is one window");
    check(p.logical == 4096 && p.physical == 4096, "window and capacity agree");
    check(p.spare_windows == 1, "exactly one window's worth of cells");
    check(p.expressible_upstream, "and upstream can build it as asked");
    check(explain(p).find("NOT expressible") == std::string::npos,
          "no warning on a plan upstream can honour");
}

void padding_is_reported_not_hidden() {
    const plan p = evaluate(plain(3000));
    check(p.logical == 3072, "3000 pads up to 3072");
    check(p.physical == 3072, "capacity follows the window when none was asked for");

    request exact = decoupled(3000, 5000);
    exact.runtime_length = true;
    const plan q = evaluate(exact);
    check(q.why == refusal::none, "an exact capacity is legal");
    check(q.physical == 5000, "runtime length keeps 5000 cells");
    check(q.logical == 3072, "while the window is still padded");
    check(q.spare_windows == 1, "5000 holds one 3072 window, not two");
}

void decoupled_capacity() {
    const plan p = evaluate(decoupled(4096, 8192));
    check(p.why == refusal::none, "a legal decoupled request is accepted");
    check(p.kind == mode::decoupled, "and identified as the fork's shape");
    check(p.spare_windows == 2, "two windows' worth of cells");
    check(!p.expressible_upstream, "which stock parameters cannot produce");
    check(explain(p).find("NOT expressible") != std::string::npos,
          "and the explanation says so instead of staying quiet");

    // Padding comes first, so 4000 cells against a 4096 window becomes 4096 and is
    // legal -- the fork's own behaviour (:976 pads, :977 compares).
    const plan padded_up = evaluate(decoupled(4096, 4000));
    check(padded_up.why == refusal::none, "padding 4000 up to the window is legal");
    check(padded_up.physical == 4096, "and the plan reports the padded capacity");

    // The exact-length mode skips the padding, which is the only way to remain below
    // the window -- and then it must be refused rather than rounded up.
    request exact_short = decoupled(4096, 4000);
    exact_short.runtime_length = true;
    const plan low = evaluate(exact_short);
    check(low.why == refusal::capacity_below_window, "capacity under the window is refused");
    check(low.logical == 4096 && low.physical == 4000, "the plan reports the numbers it compared");
}

// Upstream CAN express decoupling, as long as the capacity is exactly n_seq_max
// windows of a split cache. That is the honest answer to "decoupled KV is
// unavailable": unavailable in the unified shape the fork offered, available in a
// different shape with a different cost -- n_seq_max sequences exist.
void upstream_can_express_it_when_split_exactly() {
    const plan p = evaluate(split(2048, 8192, 4));
    check(p.why == refusal::none, "an exact split is accepted");
    check(p.kind == mode::split, "as a split cache");
    check(p.logical == 2048, "each sequence keeps its window");
    check(p.spare_windows == 4, "the capacity is four windows");
    check(p.expressible_upstream, "and stock parameters can produce it");

    // 8000 pads to 8192, which is exactly four 2048 windows: the request is honoured,
    // not rounded down. Stated as a case because "8000 is not a multiple of 4" would
    // be the wrong conclusion -- the padding runs first.
    const plan padded = evaluate(split(2000, 8000, 4));
    check(padded.why == refusal::none, "a capacity that pads to an exact split is accepted");
    check(padded.physical == 8192 && padded.logical == 2048, "at the padded numbers");

    // Genuinely short: 4000 pads to 4096, still below four 2048 windows. Upstream
    // would round the window down and warn; the station refuses instead of honouring a
    // configuration nobody asked for.
    const plan short_capacity = evaluate(split(2048, 4000, 4));
    check(short_capacity.why == refusal::split_would_round, "a short split is refused");
    check(!short_capacity.expressible_upstream, "and cannot be claimed as expressible");

    const plan one = evaluate(split(4096, 4096, 1));
    check(one.why == refusal::none, "a one-sequence split is accepted");
    check(one.logical == 4096, "with the whole capacity as its window");
}

void the_fork_gates() {
    // Extra capacity beyond the S windows a split cache can spend: that is the fork's
    // separate field, which required a unified cache.
    check(evaluate(split(4096, 16384, 2)).why == refusal::requires_unified,
          "capacity beyond the split's windows requires a unified cache");

    request recurrent = decoupled(4096, 8192);
    recurrent.recurrent_arch = true;
    check(evaluate(recurrent).why == refusal::unsupported_arch, "a recurrent model is refused");

    request ds4 = decoupled(4096, 8192);
    ds4.deepseek4 = true;
    check(evaluate(ds4).why == refusal::unsupported_arch, "DEEPSEEK4 is refused");

    request arena = decoupled(4096, 8192);
    arena.shared_arena = true;
    check(evaluate(arena).why == refusal::arena_conflict, "a shared phase arena is refused");

    request huge = decoupled(4096, 2147483647u);
    check(evaluate(huge).why == refusal::exceeds_range,
          "a capacity past the padded int32 range is refused");

    check(evaluate(plain(0)).why == refusal::window_zero, "a zero window is refused");
}

// The order of the gates is observable, and it matters for the message a user reads:
// a recurrent model with a bad range should hear about the architecture, because
// fixing the number would not help.
void gate_order_is_fixable_first() {
    request r = decoupled(4096, 2147483647u);
    r.recurrent_arch = true;
    check(evaluate(r).why == refusal::unsupported_arch,
          "the architecture is reported before the range");

    request s = split(4096, 16384, 2);   // extra capacity AND an arena conflict
    s.shared_arena = true;
    check(evaluate(s).why == refusal::requires_unified,
          "the unified requirement is reported before the arena conflict");
}

void spare_windows_is_a_floor() {
    const plan p = evaluate(decoupled(3072, 10000));
    check(p.why == refusal::none, "an intermediate capacity is legal");
    check(p.logical == 3072, "window is the padded request");
    check(p.physical == 10240, "capacity pads to 10240");
    check(p.spare_windows == 3, "10240 / 3072 floors to 3, not 4");
    check(p.spare_windows * p.logical <= p.physical, "the floor never exceeds capacity");
    check((p.spare_windows + 1) * p.logical > p.physical, "and it is the largest such floor");
}

// The split shape as upstream actually builds it, measured against real contexts.
//
// Every row below was read from `llama_n_ctx` / `llama_n_ctx_seq` on a live context
// built with the tiny transformer, because the arithmetic is easy to get backwards:
// the requested n_ctx is the INPUT, the window is derived by division and padding,
// and n_ctx is then rewritten to window * S. So a caller who asks for a window gets
// capacity = window * S, and "no explicit capacity" must mean that product -- not the
// window, which is what an earlier version assumed and then refused as
// split_would_round for every split request.
//
//   requested          runtime
//   n_ctx   S unified  n_ctx  n_ctx_seq
//    6000   3    no     6144   2048
//    8000   3    no     8448   2816
//    8000   4    no     8192   2048
//    4000   4    no     4096   1024
//    4096   2    no     4096   2048
//    6000   1    yes    6144   6144
void split_shape_matches_a_real_context() {
    // n_ctx 6000, S 3 -> window 2048, capacity 6144.
    request r = plain(2048);
    r.kv_unified = false;
    r.n_seq_max  = 3;
    const plan p = evaluate(r);
    check(p.why == refusal::none, "a split request without an explicit capacity is accepted");
    check(p.kind == mode::split, "and is the split shape");
    check(p.logical == 2048, "the window is the one asked for");
    check(p.physical == 6144, "the capacity is three windows, which is what the runtime reports");
    check(p.spare_windows == 3, "capacity in windows is the sequence count");
    check(p.expressible_upstream, "and stock parameters build it");

    // n_ctx 8000, S 3 -> window 2816, capacity 8448. The window is not 8000/3 = 2666:
    // upstream pads AFTER dividing, and then grows n_ctx to match.
    request tall = plain(2816);
    tall.kv_unified = false;
    tall.n_seq_max  = 3;
    const plan q = evaluate(tall);
    check(q.why == refusal::none, "an unaligned-looking window is accepted");
    check(q.logical == 2816 && q.physical == 8448, "matching the runtime's 2816 and 8448");

    // An explicit capacity equal to the product is the same configuration.
    request explicit_product = split(2048, 6144, 3);
    check(evaluate(explicit_product).why == refusal::none, "the same plan stated explicitly");
    check(evaluate(explicit_product).physical == 6144, "resolves to the same capacity");

    // A requested capacity BELOW the product would be grown by upstream, which is a
    // rounding the caller did not choose: refused, and that refusal is the only case
    // this check should fire for.
    request too_small = split(2048, 4000, 3);
    check(evaluate(too_small).why == refusal::split_would_round,
          "a capacity below the product is refused rather than grown silently");
}

void every_reason_is_named_distinctly() {
    const refusal all[] = {
        refusal::requires_unified, refusal::exceeds_range, refusal::unsupported_arch,
        refusal::arena_conflict, refusal::capacity_below_window, refusal::split_would_round,
        refusal::window_zero,
    };
    for (std::size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) {
        const char * text = name(all[i]);
        check(text != nullptr && text[0] != '\0', "a reason has a message");
        check(std::string(text) != "unknown", "and it is not the fallback");
        for (std::size_t j = 0; j < i; ++j) {
            check(std::string(text) != name(all[j]), "and no two reasons share a message");
        }
    }
    check(std::string(name(refusal::none)) == "ok", "acceptance is named separately");
}

// Overflow, and the two different ways it used to be invisible.
//
// pad256 adds 255 before masking, so a window near the top of the range wraps to a
// small number -- and because the comparison runs after the padding, the wrapped value
// then looks legal. Before the range guard, this request was accepted with
// logical = physical = 0: a configuration no allocator could build, reported as legal
// and expressible with stock parameters.
//
// The split branch had the other shape of the same bug: `aligned_window * sequences`
// is computed in 32 bits, so a large product wraps and `physical > window*sequences`
// becomes false for a request far beyond the shape's capacity. Both are now refused
// rather than evaluated.
void overflow_is_refused_not_wrapped() {
    const uint32_t near_top = 4294967290u; // 2^32 - 6: pad256 would wrap to 0

    const plan wrapped = evaluate(plain(near_top));
    check(wrapped.why == refusal::exceeds_range, "a window whose padding would wrap is refused");
    check(wrapped.logical == 0 && wrapped.physical == 0, "and no numbers are reported for it");

    request separate = decoupled(4096, near_top);
    check(evaluate(separate).why == refusal::exceeds_range,
          "a physical capacity whose padding would wrap is refused");

    // The product's own bound, without wrapping: 65536 x 65536 = 2^32 is NOT
    // representable in 32 bits, and in the old code it wrapped to zero so that any
    // capacity at all looked larger than the shape could hold -- refused for the wrong
    // reason. Here the product is 64-bit, so the refusal that arrives is the range
    // one, which is the accurate diagnosis for this request.
    request beyond = split(65536, 0, 65536);
    beyond.physical_capacity = 4294967295u;
    const plan over = evaluate(beyond);
    check(over.why == refusal::exceeds_range,
          "a capacity past the padded int32 range is refused");
    check(over.why != refusal::requires_unified,
          "and not blamed on the shape: a wrapped product used to read as one beyond it");

    // A capacity strictly inside the product is inside the shape, so the shape gate
    // must not fire. (It may still be refused for a real reason -- here the split does
    // not divide evenly -- but that reason must not be requires_unified.)
    request inside = split(4096, 8192, 65536);
    check(evaluate(inside).why != refusal::requires_unified,
          "a capacity inside the product is not mistaken for one beyond it");
}

} // namespace

int main() {
    plain_configuration();
    padding_is_reported_not_hidden();
    decoupled_capacity();
    upstream_can_express_it_when_split_exactly();
    split_shape_matches_a_real_context();
    the_fork_gates();
    gate_order_is_fixable_first();
    spare_windows_is_a_floor();
    every_reason_is_named_distinctly();
    overflow_is_refused_not_wrapped();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "kv capacity contract holds\n");
    return 0;
}
