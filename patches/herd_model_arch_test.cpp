// herd/model_arch_test.cpp — the classification, pinned without a model or a GGUF.
//
// Three claims, each of which has been false at some point in this port's history:
//
//   1. The archs the fork refused by name are refused here, and the refusal names the
//      right reason. DEEPSEEK4/41 used to be refused only by accident -- they happen to be
//      members of llm_arch_is_hybrid in this pin -- and the hook's comment called them an
//      uncovered gap. Both the rule and the record were wrong.
//   2. Every arch either membership predicate reports is refused. This is what makes a pin
//      bump that adds an arch to `llm_arch_is_recurrent`/`is_hybrid` safe: the new arch is
//      refused without anyone editing this file.
//   3. The archs that must stay allowed do. DEEPSEEK2 is the case worth naming: matching on
//      a display string ("deepseek") would have refused it, and the fork did not.
//
// The predicates and enumerators come from upstream's `llama-arch.h`, included read-only.
// That is deliberate: a test that hard-coded integer values would pass after a pin bump
// renumbered the enum, and then pin nothing at all.
#include "herd_model_arch.hpp"

#include "llama-arch.h"

#include <cstdio>
#include <string>
#include <vector>

using herd::arch::shape;

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

void expect(int arch, shape expected, const char * label) {
    const shape actual = herd::arch::shape_of_arch(arch);
    if (actual != expected) {
        check(false, std::string(label) + ": expected shape " +
                         std::to_string(int(expected)) + ", got " +
                         std::to_string(int(actual)) + " (" + herd::arch::name(actual) + ")");
    }
}

} // namespace

int main() {
    // ---- 1. the fork's named refusals ----
    expect(LLM_ARCH_DEEPSEEK4,  shape::deepseek4, "DEEPSEEK4");
    expect(LLM_ARCH_DEEPSEEK41, shape::deepseek4, "DEEPSEEK41");
    check(!herd::arch::supports_decoupled_capacity(shape::deepseek4),
          "a DEEPSEEK4-family model may not have a decoupled capacity");

    // ---- 2. the shapes that fail for their own reasons ----
    expect(LLM_ARCH_MAMBA, shape::recurrent, "MAMBA");
    expect(LLM_ARCH_RWKV7, shape::recurrent, "RWKV7");
    check(!herd::arch::supports_decoupled_capacity(shape::recurrent),
          "a recurrent model may not have a decoupled capacity");

    expect(LLM_ARCH_JAMBA, shape::hybrid, "JAMBA");
    expect(LLM_ARCH_LFM2,  shape::hybrid, "LFM2");
    check(!herd::arch::supports_decoupled_capacity(shape::hybrid),
          "a hybrid model may not have a decoupled capacity");

    // ---- 3. the shapes that must stay allowed ----
    expect(LLM_ARCH_LLAMA,     shape::cell_pool, "LLAMA");
    expect(LLM_ARCH_QWEN2,     shape::cell_pool, "QWEN2");
    expect(LLM_ARCH_DEEPSEEK2, shape::cell_pool, "DEEPSEEK2");
    check(herd::arch::supports_decoupled_capacity(shape::cell_pool),
          "a plain cell pool may have a decoupled capacity");
    // The one that a string match would have got wrong: the fork allowed DEEPSEEK2 and
    // refused DEEPSEEK4, which no substring test can distinguish.
    check(herd::arch::shape_of_arch(LLM_ARCH_DEEPSEEK2) !=
              herd::arch::shape_of_arch(LLM_ARCH_DEEPSEEK4),
          "DEEPSEEK2 and DEEPSEEK4 are classified differently");

    // ---- 4. absence is not permission ----
    //
    // `shape_of(nullptr)` is unreachable from the hook (`create_memory` is a member
    // function, so `this` cannot be null), which is exactly why it is worth pinning: an
    // unreachable branch has no run to reveal a wrong answer, and the wrong answer here
    // would be the permissive one. A future diagnostic, probe or refactor that calls this
    // with no model must hear "unknown", not "you may widen the cache".
    check(herd::arch::shape_of(nullptr) == shape::unknown,
          "no model classifies as unknown, not as a cell pool");
    check(!herd::arch::supports_decoupled_capacity(shape::unknown),
          "and an unclassifiable model may not have a decoupled capacity");

    // The other half of the same distinction: an arch this pin does not recognise is
    // allowed, because the fork's rule was a blacklist. Refusing it would be a capability
    // removed, which is the failure mode this project keeps finding elsewhere.
    check(herd::arch::supports_decoupled_capacity(herd::arch::shape_of_arch(int(LLM_ARCH_UNKNOWN))),
          "an unrecognised arch on a loaded model stays permitted, as the fork permitted it");

    // ---- 5. the invariant, over the whole enum ----
    //
    // `LLM_ARCH_UNKNOWN` is the last enumerator (llama-arch.h), so [0, UNKNOWN) is every
    // real arch this pin knows. For each one: if either membership predicate says yes, the
    // classification must refuse it. This is the assertion that survives a pin bump that
    // adds an arch to a predicate's list -- the classification follows automatically --
    // and the one that fails if the explicit DEEPSEEK list and the predicates ever
    // disagree.
    int refused = 0;
    int allowed = 0;
    for (int arch = 0; arch < int(LLM_ARCH_UNKNOWN); ++arch) {
        const shape what = herd::arch::shape_of_arch(arch);
        const bool membership_says_no = llm_arch_is_recurrent(llm_arch(arch)) ||
                                        llm_arch_is_hybrid(llm_arch(arch));
        if (membership_says_no) {
            ++refused;
            if (herd::arch::supports_decoupled_capacity(what)) {
                check(false, "arch " + std::to_string(arch) +
                                 " is recurrent or hybrid but was allowed: " +
                                 herd::arch::name(what));
            }
        } else {
            ++allowed;
        }
        // Every refusal names itself, and no two shapes share a name: a caller deciding
        // whether to change a flag, a model or the memory budget reads that string.
        check(herd::arch::name(what)[0] != '\0', "every shape has a name");
    }
    check(refused > 0 && allowed > 0,
          "the sweep saw both outcomes (refused " + std::to_string(refused) + ", allowed " +
              std::to_string(allowed) + ")");

    // Every name distinct: a shared message would send a reader to the wrong remedy.
    const shape all[] = {shape::cell_pool, shape::recurrent, shape::hybrid, shape::deepseek4,
                         shape::unknown};
    for (std::size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            check(std::string(herd::arch::name(all[i])) != herd::arch::name(all[j]),
                  "two shapes share a name");
        }
    }

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "arch classification holds: %d arch(es) refused by membership, %d allowed\n",
                 refused, allowed);
    return 0;
}
