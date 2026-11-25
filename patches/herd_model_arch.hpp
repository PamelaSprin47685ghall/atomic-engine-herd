// herd/model_arch.hpp — what a model's memory is, as far as decoupled capacity goes.
//
// The fork refused a separate physical KV capacity for exactly two kinds of model
// (src/llama-context.cpp:962):
//
//     if (model.arch == LLM_ARCH_DEEPSEEK4 || llm_arch_is_recurrent(model.arch))
//         throw std::runtime_error("this model does not support a separate physical KV capacity");
//
// The port's gate asked `llama_model_is_recurrent(model) || llama_model_is_hybrid(model)`
// and left a comment saying DEEPSEEK4 was an uncovered gap. Reading the two predicates
// settles it: llama_model_is_hybrid() is llm_arch_is_hybrid(model->arch)
// (llama-model.cpp:4158), and in THIS pin that function's list includes LLM_ARCH_DEEPSEEK4
// and LLM_ARCH_DEEPSEEK41 (llama-arch.cpp:1502-1503). So the gate did refuse them -- by
// accident of another predicate's membership, not by stating the rule.
//
// That is the difference this file makes. Relying on "hybrid happens to contain
// deepseek4" is a fact about one pin's table, sitting one edit away from being false: a
// future pin can move DEEPSEEK4 out of the hybrid list (it is not a hybrid model in the
// Jamba sense -- it has its own memory shape) and the hook would silently start allowing a
// configuration the fork rejected. So the rule is written down explicitly, and the
// membership predicates are used for what they mean rather than for what they happen to
// contain.
//
// No llama type is named here, so the ruling can be tested without a model or a GGUF.
#pragma once

// Declared at global scope: upstream's own type. A declaration inside herd::arch would
// create an unrelated herd::arch::llama_model, and the definition from llama-model.h would
// then be "an incomplete type" at the one line that reads `model->arch`.
struct llama_model;

namespace herd::arch {

// The model's memory shape, reduced to the four answers that change this project's
// behaviour.
enum class shape {
    cell_pool,   // a cache of KV cells whose size can be decoupled from the window
    recurrent,   // one ring of states per sequence; the ring IS the window
    hybrid,      // a mix; the cell pool is not the only thing sized by n_ctx_seq
    deepseek4,   // its own memory shape, refused by the fork by name
    // No model to classify: `shape_of(nullptr)`.
    //
    // Distinct from a loaded model whose arch is LLM_ARCH_UNKNOWN, and the difference is
    // deliberate. Absence is not evidence of a cell pool, so it must never be read as
    // permission -- but an unknown arch on a *loaded* model must stay permitted, because
    // the fork's rule is a blacklist (refuse DEEPSEEK4 and recurrent, allow the rest) and
    // refusing more than the fork did would be a capability removed. Two questions that
    // look alike and have opposite right answers.
    unknown,
};

// The ruling: may this model's KV capacity be decoupled from its logical window?
//
// Only a plain cell pool may. The other three each fail for a different reason, and the
// hook reports the reason rather than a bare "declined" -- which is what a caller needs
// to know whether to change a flag, a model, or the amount of memory.
bool supports_decoupled_capacity(shape what);

// Name, for the refusal message. One string per shape, distinct, because they have
// distinct fixes.
const char * name(shape what);

// Classification from a bare arch value (an `llm_arch`, passed as its underlying integer
// so this header needs no upstream include). This overload exists for the test: it pins
// the classification itself instead of inferring it from a loaded model.
shape shape_of_arch(int arch);

// Classification of a loaded model. Defined in the .cpp, which includes upstream's private
// headers read-only (a pin bump that renames `llama_model::arch` fails that build, which is
// the wanted direction of failure).
shape shape_of(const llama_model * model);

} // namespace herd::arch
