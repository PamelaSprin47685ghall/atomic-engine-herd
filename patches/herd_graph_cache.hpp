// herd/graph_cache.hpp — the K/V cache a graph build reads from, and nothing else.
//
// Why this file exists, and why it is short
// -----------------------------------------
// The reader lane needs the K/V cache tensors for a layer: the emitter gathers keys by cell and
// values by cell, and both must come from the live cache, not from the batch's new keys. The
// earlier note in the hook file concluded that reaching them meant "reading inp's private field
// and calling get_k/get_v", with the corollary that the hook would have to REPLACE upstream's
// body.
//
// That conclusion was wrong, and the correction is the whole content of this header. Read out of
// upstream:
//
//   llama-graph.h:1389   build_attn(llm_graph_input_attn_kv * inp, ...)
//                        -- the hooked overload's first parameter is already the CONCRETE type,
//                           not llm_graph_input_i *;
//   llama-graph.h:2,35   class llm_graph_input_attn_kv { public: ... const llama_kv_cache_context * mctx; }
//                        -- `mctx` sits after `public:`, so it is a public data member;
//   llama-kv-cache.h:378 llama_kv_cache_context { public: ... };
//   llama-kv-cache.h:427 ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
//                        -- public, const, and it allocates its views in the caller's context.
//
// So the path is `inp->mctx->get_k(ctx0, il)`: three public symbols, no pointer arithmetic, no
// private access, and no reason to copy a function body. The hook still calls the original, so
// the graph is built by upstream's own code; this module only *reads the same tensors upstream
// reads*, which is what an emission needs in order to be built from them.
//
// What it does NOT claim
// ----------------------
// Calling get_k/get_v allocates view nodes in the live ggml context. That is a real cost and a
// real perturbation: a build that asks for these tensors and then lets the original run will
// carry both sets of views. It is therefore behind its own opt-in switch at the call site, and
// the switch is a measurement, not a product path. It also does not decide whether the reader
// attention is correct: that is the emitter's tests, against ggml's kernels.
#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace herd::graph {

// What a layer's cache looks like through the graph input.
//
// The shapes are recorded rather than assumed, because the emitter's contract is stated in
// terms of them ([head_dim, n_head_kv, n_cell] for K, either that or the cell-major
// permutation for V) and a caller that guesses has no way to notice it guessed wrong.
struct cache_view {
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;

    std::int64_t ne_k[4] = {};
    std::int64_t ne_v[4] = {};
    std::size_t  nb_k[4] = {};
    std::size_t  nb_v[4] = {};

    int type_k = 0;
    int type_v = 0;

    // The number of cells the cache context reports as in use. It is the same number the
    // original body uses to size its views, so a caller comparing it against ne[2]/ne[0] can
    // tell which of upstream's two V layouts it is looking at.
    std::uint32_t n_kv = 0;
};

// Why a read produced no cache view. Named, because "it did not work" has three very different
// causes and only one of them is a bug in the caller.
enum class cache_status {
    ok,
    null_input,        // the hook was handed no graph input
    no_cache_context,  // the input exists but carries no cache context (a build with no memory)
    null_tensor,       // get_k/get_v returned nothing for this layer
    layer_not_cached,  // this layer id is not in the cache's map: upstream would have thrown
    no_room,           // the graph arena cannot hold the two view headers: upstream would have aborted
    arena_inconsistent, // ggml reports used > size: the accounting itself is broken, so no claim about
                        // headroom can be made, and calling into a view would be a guess
};

// Is there room for `views` more tensor objects in an arena with `free_bytes` left?
//
// A pure function, and it exists as one because of what happens when the answer is no:
// ggml_view_4d -> ggml_new_tensor_impl -> ggml_new_object, whose out-of-space branch logs a
// warning and returns NULL *under NDEBUG*, and is followed by `GGML_ASSERT(obj_new)` in
// ggml_new_tensor_impl (ggml.c:1832). The assert runs in EVERY build, so asking for a view that
// does not fit does not return null -- it ends the process. A diagnostic switch that can do that
// from inside a graph build would kill the inference it was measuring.
//
// The per-view cost is ggml_tensor_overhead() (GGML_OBJECT_SIZE + GGML_TENSOR_SIZE, ggml.c:1479):
// a view allocates the tensor object and no data.
//
// Callers pass the numbers rather than the context, so the rule is testable without an arena,
// a cache, or a model -- a rule that can only be exercised on the hardware it guards is a rule
// nobody tests.
bool cache_headroom_ok(std::size_t free_bytes, std::size_t per_view_bytes, std::size_t views) noexcept;

const char * describe(cache_status status);

// Reads layer `layer`'s K and V cache views through `graph_input`.
//
// `graph_input` must be the `llm_graph_input_attn_kv *` the hooked build_attn received, passed
// as void* so this header does not drag llama-graph.h into every includer. The cast back is in
// the definition file, where the real type is known.
//
// The two tensors are allocated in `ctx` and owned by it: they are views, valid as long as the
// context is, which for a graph build is the build.
cache_status cache_tensors(const void * graph_input, ggml_context * ctx, std::int32_t layer,
                           cache_view * out);

} // namespace herd::graph
