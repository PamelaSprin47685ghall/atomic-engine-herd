// herd/model_arch.cpp — the classification, read from upstream's own declarations.
//
// Both headers are included read-only: `llama-model.h` for the `arch` field the hook's
// argument carries, `llama-arch.h` for the enumerators and the two membership predicates.
// Nothing upstream is edited, and a pin bump that renames either symbol fails this build
// instead of misclassifying a model at run time.
#include "herd_model_arch.hpp"

#include "llama-arch.h"
#include "llama-model.h"

namespace herd::arch {
namespace {

// The archs the fork refused by name. A separate list from the membership predicates, and
// that is the point: LLM_ARCH_DEEPSEEK4 and LLM_ARCH_DEEPSEEK41 are currently members of
// llm_arch_is_hybrid (llama-arch.cpp:1502-1503), so refusing them *through* that predicate
// works today and would stop working silently if the membership changed. This list states
// the rule itself.
bool is_deepseek4_family(llm_arch arch) {
    return arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41;
}

} // namespace

shape shape_of_arch(int arch) {
    const auto value = static_cast<llm_arch>(arch);

    // Order matters for the reported reason, not for the ruling: a DEEPSEEK4 model would
    // also be reported as hybrid by the predicate below, and "this model has its own
    // memory shape" is the accurate statement about it.
    if (is_deepseek4_family(value)) {
        return shape::deepseek4;
    }
    if (llm_arch_is_recurrent(value)) {
        return shape::recurrent;
    }
    if (llm_arch_is_hybrid(value)) {
        return shape::hybrid;
    }
    // Everything else is a cell pool, and that is a decision rather than a fallthrough.
    //
    // The fork's rule was a blacklist -- refuse LLM_ARCH_DEEPSEEK4 and recurrent models,
    // allow the rest (src/llama-context.cpp:962). Inheriting its default is what keeps
    // this port from removing capabilities: an arch this pin knows but neither predicate
    // lists (and a future pin's new arch) is allowed, exactly as the fork allowed it.
    // Refusing the unknown case instead would look more careful and would be a regression.
    //
    // LLM_ARCH_UNKNOWN lands here too. No loaded model carries it -- it is the field's
    // default before loading, and the loader rejects it -- so the hook never sees it.
    return shape::cell_pool;
}

shape shape_of(const llama_model * model) {
    // Absence is not a cell pool. This is unreachable from the hook (`create_memory` is a
    // member function, so `this` is never null), and it is written this way for the reason
    // unreachable branches still matter: the honest answer to "which model is this" when
    // there is no model is "I do not know", and returning the one shape that permits
    // widening would turn a future misuse -- a diagnostic, a probe, a refactor that calls
    // this from somewhere new -- into a silent permission.
    if (!model) {
        return shape::unknown;
    }
    return shape_of_arch(static_cast<int>(model->arch));
}

bool supports_decoupled_capacity(shape what) {
    // Only a plain cell pool. Everything else either has no cell pool (recurrent), or has
    // one whose size is not the only thing derived from n_ctx_seq (hybrid), or has its own
    // memory shape (deepseek4) -- in every case, widening the number would enlarge the
    // window rather than add spare cells, which is the opposite of decoupled capacity.
    return what == shape::cell_pool;
}

const char * name(shape what) {
    switch (what) {
        case shape::cell_pool:
            return "a KV cell pool";
        case shape::recurrent:
            return "a recurrent memory (one ring of states per sequence, sized by the same "
                   "number, so a larger value is a larger ring no position may address)";
        case shape::hybrid:
            return "a hybrid memory (the cell pool is not the only thing sized by the "
                   "context length)";
        case shape::deepseek4:
            return "a DEEPSEEK4-family memory shape, which does not support a separate "
                   "physical KV capacity";
        case shape::unknown:
            return "no model to classify (the model pointer was null), so nothing may be "
                   "assumed about its memory";
    }
    return "an unrecognised memory shape";
}

} // namespace herd::arch
