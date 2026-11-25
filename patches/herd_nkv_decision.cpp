#include "herd_nkv_decision.hpp"

namespace herd::nkv {

ruling decide(const call_facts & facts) {
    ruling out;
    out.window = facts.window;

    // A second context under one request is a caller bug, and the ruling is about the
    // caller rather than about capacity: no widening, and the count is what reports it.
    // Checked before every capacity gate because "already consumed" is not a fact about
    // the numbers — even a perfectly legal request must not be applied twice.
    if (facts.already_consumed) {
        out.act   = ruling::action::refuse_duplicate;
        out.why   = refusal::none; // nothing was refused about the capacity itself
        out.cells = facts.window;
        return out;
    }

    // The model's memory shape. A ring of states (recurrent) or a mixed pool (hybrid,
    // DEEPSEEK4) is sized by the same number that bounds the window, so a wider value
    // enlarges the window rather than adding spare cells.
    auto refuse = [&](refusal why) {
        out.act   = ruling::action::refuse;
        out.why   = why;
        out.cells = facts.window;
        return out;
    };
    if (!facts.memory_supports_decoupled) {
        return refuse(facts.recurrent_shape ? refusal::recurrent : refusal::unsupported_shape);
    }

    // The fork's bound (llama-context.cpp:958): without it an absurd value becomes a
    // cache size and aborts inside the allocator instead of failing here with a reason.
    // Also the guard that keeps the padding arithmetic below from overflowing.
    if (facts.requested > std::uint32_t(INT32_MAX) - 255u) {
        return refuse(refusal::exceeds_range);
    }

    // No more than the window: the window already provides that many cells. Recorded as
    // below_window rather than as a silent success — the caller still has to decide what
    // to report, and "nothing to do" is not "done".
    if (facts.requested <= facts.window) {
        return refuse(refusal::below_window);
    }

    // A split cache with more than one sequence: `n_ctx_seq` is the PER-SEQUENCE window
    // there (llama-context.cpp:461-471), so widening it widens every sequence's window.
    if (!facts.kv_unified && facts.n_seq_max > 1) {
        return refuse(refusal::split_shape);
    }

    // Widen. The capacity is padded to 256 exactly as the fork did (:974); upstream pads
    // n_ctx the same way, and a cache size that is not a multiple of the pad is the case
    // the fork's own comment says only holds on the CPU path.
    out.act   = ruling::action::widen;
    out.why   = refusal::none;
    out.cells = (facts.requested + 255u) & ~std::uint32_t(255u);
    return out;
}

const char * name(refusal why) {
    switch (why) {
        // NOT "applied". `refusal::none` means "no refusal was issued", which is equally
        // true of a request that was honoured and of one that never reached the hook at
        // all -- a scope opened and closed without an init call leaves `reason` at this
        // default, with `consumed == false` and nothing applied. Printing "applied" for
        // that state made the one function whose job is to describe into a source of a
        // false claim, and a driver that formatted `name(reason())` without first checking
        // `consumed()` would have written it into a report.
        //
        // The pair (consumed(), applied()) is the answer; this string only names the
        // absence of a refusal, so it says exactly that.
        case refusal::none:         return "no refusal recorded (this is not a claim that the "
                                           "request was applied; check consumed() and applied())";
        case refusal::below_window: return "the request is not larger than the context's window, which "
                                           "already provides that many cells";
        case refusal::split_shape:  return "this context uses a per-sequence window as its capacity, so "
                                           "widening it would widen every sequence's window";
        case refusal::recurrent:    return "this model keeps one ring of states per sequence and sizes it "
                                           "from the same number, so a larger value is not a separate "
                                           "capacity: it is a larger ring no position may address";
        case refusal::exceeds_range:return "the request exceeds the supported physical KV range";
        case refusal::unsupported_shape:
            return "this model's memory is not a plain KV cell pool (hybrid, or DEEPSEEK4 "
                   "family), so a larger capacity would widen the window rather than add "
                   "spare cells";
        case refusal::allocation_failed:
            return "the cache could not be created at that capacity (allocation failed or the "
                   "memory module refused the shape)";
    }
    return "unknown";
}

void record(request_state & state, const ruling & decision, outcome result) {
    if (decision.act == ruling::action::refuse_duplicate) {
        // Nothing about this context's capacity is recorded: the first call already owns
        // those answers, and this call is only counted.
        ++state.consumptions;
        return;
    }

    state.consumed = true;
    ++state.consumptions;

    if (decision.act == ruling::action::refuse) {
        state.applied       = false;
        state.applied_cells = decision.window;
        state.reason        = decision.why;
        return;
    }

    // widen: the claim follows the call, never precedes it.
    switch (result) {
        case outcome::built:
            state.applied       = true;
            state.applied_cells = decision.cells;
            state.reason        = refusal::none;
            return;
        case outcome::returned_null:
        case outcome::threw:
            // The request was legal and the memory module could not be built. Reporting
            // `applied` here would name a capacity the process does not hold, and the
            // caller would trust a context that does not exist.
            state.applied       = false;
            state.applied_cells = decision.window;
            state.reason        = refusal::allocation_failed;
            return;
        case outcome::not_called:
            // A widen ruling that never reached upstream: the scope was closed before the
            // init call, or the call site returned early. Not an applied capacity either.
            state.applied       = false;
            state.applied_cells = decision.window;
            state.reason        = refusal::allocation_failed;
            return;
    }
}

} // namespace herd::nkv
