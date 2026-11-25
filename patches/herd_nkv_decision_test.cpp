// The NKV ruling and its recording, tested without a model, a cache, or an allocator.
//
// The two claims this file exists for, neither of which a model-backed probe can state:
//
//   1. A legal request that upstream fails to deliver is NOT reported as applied. That
//      covers `create_memory` returning null and `create_memory` throwing — the two paths
//      that a real run can only reach by exhausting memory, which is not a test.
//   2. Every refusal is a refusal *of the request*, with the window — never the requested
//      number — as the capacity reported to the caller. A refusal that echoed the request
//      back would make the driver admit against cells that do not exist.
//
// A third, structural: the ruling is pure, so the same facts give the same answer. The
// hook calls it once per context and must not be able to get two different answers.
#include "herd_nkv_decision.hpp"

#include <cstdio>
#include <string>

using herd::nkv::call_facts;
using herd::nkv::outcome;
using herd::nkv::refusal;
using herd::nkv::request_state;
using herd::nkv::ruling;

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

call_facts plain(std::uint32_t requested, std::uint32_t window) {
    call_facts facts;
    facts.requested = requested;
    facts.window    = window;
    return facts;
}

// ---- the ruling -----------------------------------------------------------

// A request above the window, on a cell-pool model, in one unified cache: widen. The
// target is the padded value, because a cache size that is not a multiple of 256 is the
// configuration upstream documents as CPU-only.
void widening_pads_to_256() {
    const ruling exact = herd::nkv::decide(plain(512, 256));
    check(exact.act == ruling::action::widen, "512 over a 256 window widens");
    check(exact.cells == 512, "an aligned target is unchanged by padding");
    check(exact.why == refusal::none, "and the ruling carries no refusal");

    const ruling padded = herd::nkv::decide(plain(300, 256));
    check(padded.act == ruling::action::widen, "300 over a 256 window widens");
    check(padded.cells == 512, "and the target is padded to 512, not reported as 300");

    const ruling barely = herd::nkv::decide(plain(257, 256));
    check(barely.cells == 512, "one cell over the window still rounds up to a whole pad");

    const ruling at_edge = herd::nkv::decide(plain(std::uint32_t(INT32_MAX) - 255u, 256));
    check(at_edge.act == ruling::action::widen,
          "the largest legal request widens (the range gate is > INT32_MAX-255)");
    check(at_edge.cells <= std::uint32_t(INT32_MAX),
          "and its padded target stays inside the int32 range the cache takes");
}

// Every gate names itself, and every one of them reports the window as the capacity.
// A refusal that echoed the request would let the driver admit against cells that were
// never allocated.
void each_refusal_is_named_and_reports_the_window() {
    struct case_ {
        const char * label;
        call_facts   facts;
        refusal      expected;
    };
    call_facts recurrent = plain(512, 256);
    recurrent.memory_supports_decoupled = false;
    recurrent.recurrent_shape           = true;

    call_facts hybrid = plain(512, 256);
    hybrid.memory_supports_decoupled = false;
    hybrid.recurrent_shape           = false;

    call_facts split = plain(512, 256);
    split.kv_unified = false;
    split.n_seq_max  = 2;

    const case_ cases[] = {
        {"below the window", plain(100, 256), refusal::below_window},
        {"equal to the window", plain(256, 256), refusal::below_window},
        {"past the int32 range", plain(std::uint32_t(INT32_MAX) - 254u, 256), refusal::exceeds_range},
        {"recurrent memory", recurrent, refusal::recurrent},
        {"hybrid or deepseek4 memory", hybrid, refusal::unsupported_shape},
        {"split cache with several sequences", split, refusal::split_shape},
    };

    for (const auto & entry : cases) {
        const ruling got = herd::nkv::decide(entry.facts);
        check(got.act == ruling::action::refuse,
              std::string(entry.label) + ": refused, not widened");
        check(got.why == entry.expected,
              std::string(entry.label) + ": expected refusal " + herd::nkv::name(entry.expected) +
                  ", got " + herd::nkv::name(got.why));
        check(got.cells == entry.facts.window,
              std::string(entry.label) + ": the reported capacity is the window (" +
                  std::to_string(entry.facts.window) + "), not the request (" +
                  std::to_string(entry.facts.requested) + ")");
    }
}

// The duplicate is not a capacity ruling. It must not touch what the first call recorded,
// and its reason must stay `none` — reporting "refused because …" here would blame the
// numbers for a caller's mistake.
void a_duplicate_is_about_the_caller_not_the_capacity() {
    call_facts second = plain(512, 256);
    second.already_consumed = true;
    const ruling got = herd::nkv::decide(second);
    check(got.act == ruling::action::refuse_duplicate, "a consumed request is a duplicate");
    check(got.why == refusal::none, "and carries no capacity refusal");
}

// The same facts must give the same answer: the hook calls this once per context, and a
// ruling that depended on a hidden counter would make two contexts with identical
// arguments behave differently.
void rulings_are_pure() {
    const call_facts facts = plain(600, 256);
    const ruling first  = herd::nkv::decide(facts);
    const ruling second = herd::nkv::decide(facts);
    check(first.act == second.act && first.cells == second.cells && first.why == second.why,
          "the same facts give the same ruling");
}

// ---- the recording --------------------------------------------------------

// The point of the whole exercise, and the one claim a model-backed probe cannot state.
void a_failed_build_is_never_applied() {
    const ruling widening = herd::nkv::decide(plain(512, 256));
    check(widening.act == ruling::action::widen, "premise: this is a widening ruling");

    for (const auto result : {outcome::returned_null, outcome::threw, outcome::not_called}) {
        request_state state;
        herd::nkv::record(state, widening, result);
        const char * label = result == outcome::returned_null ? "null"
                           : result == outcome::threw       ? "throw"
                                                            : "not called";
        check(!state.applied, std::string("a failed build (") + label + ") is not applied");
        check(state.reason == refusal::allocation_failed,
              std::string("a failed build (") + label + ") says why");
        check(state.applied_cells == 256,
              std::string("a failed build (") + label + ") reports the window, not the request");
        check(state.consumed, std::string("a failed build (") + label + ") was still consumed");
        check(state.consumptions == 1, std::string("a failed build (") + label + ") counted once");
    }

    // The contrast, so the assertions above are known to be able to fail: the same ruling
    // with a successful call IS applied, at the padded target.
    request_state ok_state;
    herd::nkv::record(ok_state, widening, outcome::built);
    check(ok_state.applied, "a built widening is applied");
    check(ok_state.applied_cells == 512, "at the padded target");
    check(ok_state.reason == refusal::none, "with no refusal");
}

// A refusal keeps the first call's answer, and a duplicate leaves it untouched.
void a_refusal_records_the_window_and_a_duplicate_changes_nothing() {
    const ruling refused = herd::nkv::decide(plain(100, 256));
    request_state state;
    herd::nkv::record(state, refused, outcome::not_called);
    check(!state.applied, "a refusal is not applied");
    check(state.applied_cells == 256, "and reports the window");
    check(state.reason == refusal::below_window, "with its own reason");
    check(state.consumed && state.consumptions == 1, "and is counted as one consumption");

    // A second context under the same scope: the first call's record must survive.
    const request_state before = state;
    const ruling duplicate = herd::nkv::decide([] {
        call_facts facts = plain(512, 256);
        facts.already_consumed = true;
        return facts;
    }());
    herd::nkv::record(state, duplicate, outcome::not_called);
    check(state.consumptions == 2, "the duplicate increments the consumption count");
    check(state.applied == before.applied, "and leaves `applied` as the first call set it");
    check(state.applied_cells == before.applied_cells, "and its recorded capacity");
    check(state.reason == before.reason, "and its reason");
}

// A request that never reached the hook is not a success, and the one string that used to
// say it was.
//
// `refusal::none` is the default of `request_state::reason`, and it also is what a
// successfully applied request carries. So the text for `none` must not describe the
// outcome -- a scope opened and closed without an init call has `consumed == false`,
// `applied == false`, `applied_cells == 0`, and `reason == none`, and any driver that
// formatted `name(reason())` would have printed a claim about an application that never
// happened. Measured: the string was literally "applied".
//
// The assertion is on the absence of the claim, not on any particular wording: the next
// person may reword it freely, and must not reintroduce "applied".
void the_none_reason_does_not_claim_application() {
    const std::string text = herd::nkv::name(refusal::none);
    // The property is "does not claim success", not "does not contain a word": the
    // disclaimer has to name the two fields that do answer the question, and one of them
    // is called `applied`. My first version asserted the substring's absence and failed
    // against its own disclaimer -- an assertion that forbids the fix rather than the bug.
    check(text != "applied", "name(none) must not be the bare claim 'applied'; got: " + text);
    check(text.find("consumed") != std::string::npos &&
              text.find("applied()") != std::string::npos,
          "name(none) must point at the pair that answers the question; got: " + text);

    // The state such a scope actually carries, stated so the distinction is pinned rather
    // than implied: nothing consumed, nothing applied, and no capacity to report.
    const request_state untouched;
    check(!untouched.consumed, "a request that never reached the hook is not consumed");
    check(!untouched.applied, "and nothing was applied");
    check(untouched.applied_cells == 0, "and no capacity is reported");
    check(untouched.reason == refusal::none, "with no refusal, which is why the text matters");

    // The contrast that keeps this from being vacuous: an applied request also carries
    // `none`, so the string must be readable in that state too -- the pair (consumed,
    // applied) is what distinguishes them, and every refusal has its own text.
    const ruling widening = herd::nkv::decide(plain(512, 256));
    request_state applied;
    herd::nkv::record(applied, widening, outcome::built);
    check(applied.reason == refusal::none, "an applied request also records none");
    check(applied.consumed && applied.applied, "and is distinguished by consumed()/applied(), not by text");
}

// Every combination of ruling and outcome, swept: `applied` is true in exactly one cell of
// the matrix. Written as a sweep rather than as cases because the property is "only
// widen+built", and a future action added to the enum must not silently become applied.
void applied_is_true_only_for_widen_and_built() {
    call_facts widen_facts = plain(512, 256);
    call_facts refuse_facts = plain(100, 256);
    call_facts dup_facts = plain(512, 256);
    dup_facts.already_consumed = true;

    const ruling rulings[] = {herd::nkv::decide(widen_facts),
                              herd::nkv::decide(refuse_facts),
                              herd::nkv::decide(dup_facts)};
    const outcome outcomes[] = {outcome::built, outcome::returned_null, outcome::threw,
                                outcome::not_called};

    for (const ruling & decision : rulings) {
        for (const outcome result : outcomes) {
            request_state state;
            herd::nkv::record(state, decision, result);
            const bool expected = decision.act == ruling::action::widen &&
                                  result == outcome::built;
            check(state.applied == expected,
                  std::string("applied is ") + (expected ? "true" : "false") +
                      " for action " + std::to_string(int(decision.act)) +
                      " with outcome " + std::to_string(int(result)));
            if (state.applied) {
                check(state.applied_cells == decision.cells,
                      "an applied record carries the ruling's target");
            } else if (decision.act != ruling::action::refuse_duplicate) {
                // A duplicate is the exception, and deliberately so: it must leave the first
                // call's record intact — including its capacity — because the two calls are
                // two contexts and only the first was ruled on.
                check(state.applied_cells == decision.window,
                      "a non-applied record carries the window");
            } else {
                check(state.applied_cells == 0,
                      "a duplicate record touches no capacity at all");
            }
        }
    }
}

} // namespace

int main() {
    widening_pads_to_256();
    each_refusal_is_named_and_reports_the_window();
    a_duplicate_is_about_the_caller_not_the_capacity();
    rulings_are_pure();
    a_failed_build_is_never_applied();
    a_refusal_records_the_window_and_a_duplicate_changes_nothing();
    the_none_reason_does_not_claim_application();
    applied_is_true_only_for_widen_and_built();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "nkv decision contract holds\n");
    return 0;
}
