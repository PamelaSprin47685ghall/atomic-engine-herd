// herd/graph_access.h — the graph context behind an attention call, and nothing else.
//
// Why this file exists
// --------------------
// An attention graph builder needs a ggml_context to create nodes in. Inside upstream that
// context is `llm_graph_context::ctx0`, a PRIVATE member of a class that also declares a
// virtual destructor. So it cannot be reached by naming it, and it cannot be reached by
// guessing an offset either: the vptr occupies the first word of every instance, and a
// hardcoded number would be wrong the moment upstream adds, removes or reorders a field
// ahead of it. A wrong offset in a pointer-to-member context does not fail to compile; it
// reads a neighbouring field and hands back a pointer that is not a ggml_context.
//
// The one reading that cannot drift is the one the compiler computes from upstream's own
// header:
//
//   * `&llm_graph_context::ctx0` is evaluated by the same compiler, against the same
//     header, that laid out the object being inspected -- no number is written down;
//   * if upstream renames or removes the member, this translation unit stops compiling.
//     That is the failure this file wants: loud, at build time, in one place;
//   * if upstream changes the layout, the pointer-to-member is regenerated to match, so
//     there is nothing to update.
//
// The private access itself uses the one mechanism the standard permits: an explicit
// template instantiation is not subject to access control ([temp.spec]/6). The member
// pointer is a friend function of a class template that captures it as a non-type
// template argument. No macro renames `private`, no object layout changes, and the class
// definition is not touched -- which is why this remains a read-only use of upstream.
//
// What this header deliberately does NOT do
// -----------------------------------------
// It does not expose `llm_graph_context` to its includers. Only the definition file
// includes upstream's private header, so a layout change cannot ripple through the tree;
// everything else sees `struct ggml_context` and one function.
#pragma once

#include <cstddef>

// ggml's own type. Forward-declared here so including this header does not drag in ggml.
struct ggml_context;

namespace herd::graph {

// The private member is reachable in this build (i.e. the accessor compiled). False is
// not a runtime condition today -- it exists so a caller can state the dependency
// instead of assuming it, and so a future build that compiles the accessor out has to
// notice.
bool accessor_compiled();

// The ggml_context a build_attn call is creating nodes in.
//
// `self` is the `llm_graph_context *` handed to the hooked member function. It is the
// pointer to the llm_graph_context SUBOBJECT, not to the most-derived object: a member
// function always receives the pointer adjusted for its own class, so the hook does not
// need to know which of the derived graph contexts it was called on. That is what makes
// one accessor serve every build_attn overload.
//
// Returns nullptr when `self` is null. It cannot detect a wrong non-null pointer: see
// verify() for the check that can.
ggml_context * context_of(const void * self);

// The `llm_graph_result *` the same call is building into, as an opaque pointer.
//
// Opaque for the same reason as everything else here: only herd_graph_access.cpp includes
// upstream's private header, so the type cannot ripple. The caller casts, and the cast is the
// claim "I include llama-graph.h in my translation unit".
//
// WHY THIS EXISTS AT ALL, stated because the previous note in this tree claimed the opposite.
// `llm_graph_result::inputs` is private, and an earlier draft concluded from that "the result
// cannot be reached, so an input cannot be registered". Reading the class rather than the one
// line showed two things: `add_input(llm_graph_input_ptr)` is PUBLIC (llama-graph.h:1028), and
// `res` is a private member of `llm_graph_context` -- the SAME situation as `ctx0`, which this
// file already reaches. So the reach is the established mechanism, not a new one, and no
// upstream change and no process-global table is needed.
//
// Null when `self` is null. As with context_of, a wrong non-null pointer cannot be detected
// here; safety comes from the member pointer being computed by the compiler from upstream's
// own header.
void * result_of(const void * self);

// Two checks with different powers, and the difference is the whole point.
//
// What NO runtime check here can do: validate an arbitrary pointer.
//
// The previous comment claimed `ggml_used_mem(ctx) <= ggml_get_mem_size(ctx)` would catch a
// wrong member offset "before any allocation". That is false, and the false claim is worse
// than no check: ggml_used_mem() DEREFERENCES ctx to read its fields. A pointer that is not a
// ggml_context does not fail the comparison -- the comparison never happens, because the read
// already faulted. There is no portable test for "is this pointer a ggml_context": every
// candidate test reads through the pointer, and reading through a wrong pointer is undefined
// behaviour, not a detectable error.
//
// So the safety of `context_of` does not come from runtime validation at all. It comes from
// the member pointer being computed by the compiler from upstream's own header
// (herd_graph_access.cpp): while that file compiles, the offset is right by construction, and
// a rename or removal fails the build. Runtime checks below are diagnostics for a pointer
// that is ALREADY believed correct -- they can catch a truncated or stale arena, and they
// report what they see. They cannot make a wrong pointer safe, and nothing in this module
// claims they do.

// Reads the arena's used/size. Allocates nothing, so it does not perturb the graph.
//
// Returns false when the numbers are self-contradictory (used > size), which is the only
// inconsistency visible without trusting the context further. True means "the two fields are
// consistent", NOT "this is a ggml_context" -- see above. Any dereference of a wrong pointer
// is the caller's fault, and no return value can describe it.
bool arena_usage(ggml_context * ctx, std::size_t * used, std::size_t * size);

// Why probe_one_tensor() produced no tensor. Named, because "ggml returned null" and "the
// arena had no room" and "this build already probed once" are three different facts and a
// caller that cannot tell them apart will report the wrong one.
enum class probe_result {
    created,       // the tensor exists; the arena grew by its header
    exhausted,     // declined before asking: the arena cannot hold one more header
    already_done,  // this process already spent its one deep probe
    null_context,  // the caller passed nullptr
};

// Why `exhausted` is a REFUSAL and not a failure reading, which is the one thing this enum
// had to get right.
//
// ggml_new_object's out-of-space branch (ggml.c:1749-1755) returns NULL when NDEBUG is
// defined and calls GGML_ABORT when it is not. So "ggml_new_tensor_1d returned null because
// the arena was full" is a statement about the build's optimisation setting, not about ggml:
// in a debug or assert-enabled build the call does not return at all. A probe that asked
// first and read the answer afterwards was therefore safe in exactly the configuration this
// project builds and lethal in the one a developer debugs.
//
// probe_one_tensor now compares ggml_tensor_overhead() with the arena's free remainder and
// declines BEFORE the call. `exhausted` means "this function decided not to ask", which is
// the same decision in every build and cannot abort the inference the probe was called from.

// One deep check, ONCE per process, and it must be asked for separately.
//
// It creates a real tensor inside the live graph context, which changes what it measures: the
// graph's arena grows by one tensor header and that memory is never returned (a ggml arena has
// no per-tensor free). Called from a per-layer hook, an un-guarded version would grow the
// arena by one header per layer per decode -- which is why the previous comment's "costs one
// tensor header, in an arena about to be freed" was true per graph and false per run.
//
// Hence: one shot per process, behind its own switch, separate from the allocation-free
// arena_usage(). A diagnostic that mutates the thing it observes must be opt-in twice -- once
// to enable it, once by the operator accepting the mutation.
probe_result probe_one_tensor(ggml_context * ctx, std::size_t * used_after, std::size_t * size);

// Test-only: forget that the one deep probe was spent.
//\n// The one-shot rule is a production property -- a probe called from a per-layer hook must not
// add a tensor header per layer -- and it is also what makes a single test process unable to
// check more than one scenario. Rather than weaken the rule, this resets it, and the same
// convention as the HIP plugin's fault seam applies: the function exists for tests, is not
// reachable from any product path, and says so here so a reader does not mistake it for a
// supported operation.
void reset_deep_probe_for_testing();

} // namespace herd::graph
