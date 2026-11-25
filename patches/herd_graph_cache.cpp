// The only translation unit that touches the graph input's cache context.
//
// It includes upstream's private header for one reason: `llm_graph_input_attn_kv` and
// `llama_kv_cache_context` are declared there, and the access being exercised is entirely
// public API on them (`mctx` after `public:`, `get_k`/`get_v` public and const). No member
// offset is computed, no function body is copied, and nothing here writes to upstream.
#include "herd_graph_cache.hpp"

#include <ggml.h>

#include <exception>
#include <stdexcept>

#include "llama-graph.h"
// llama-graph.h forward-declares llama_kv_cache_context; calling get_k/get_v/get_n_kv on it
// needs the definition. This is the second and last private header in the tree, and it is here
// for the same reason as the first: the entry points are public and the type must be complete.
#include "llama-kv-cache.h"

namespace herd::graph {

const char * describe(cache_status status) {
    switch (status) {
        case cache_status::ok:               return "ok";
        case cache_status::null_input:       return "the build_attn hook was handed no graph input";
        case cache_status::no_cache_context: return "the graph input carries no cache context: this build "
                                                     "has no memory to read from";
        case cache_status::null_tensor:      return "get_k/get_v returned nothing for this layer";
        case cache_status::layer_not_cached: return "this layer id is not in the cache: upstream looks it up "
                                                     "with unordered_map::at, which throws";
        case cache_status::no_room:          return "the graph arena cannot hold the two view headers, and "
                                                     "ggml asserts rather than returning null when a view "
                                                     "does not fit";
        case cache_status::arena_inconsistent: return "ggml reports used > size, so the arena's accounting "
                                                      "is broken and no claim about headroom can be made; a "
                                                      "view built here would be a guess";
    }
    return "unknown status";
}

bool cache_headroom_ok(std::size_t free_bytes, std::size_t per_view_bytes,
                       std::size_t views) noexcept {
    if (views == 0) {
        return true;
    }
    // Division, not `free_bytes >= per_view_bytes * views`, so a caller with an enormous
    // per_view_bytes cannot overflow the multiply into a small number and be told there is room.
    // A zero per_view_bytes means the caller passed nothing useful; refusing is the safe reading.
    if (per_view_bytes == 0) {
        return false;
    }
    return free_bytes / per_view_bytes >= views;
}

cache_status cache_tensors(const void * graph_input, ggml_context * ctx, std::int32_t layer,
                           cache_view * out) {
    if (!out) {
        return cache_status::null_input;
    }
    *out = cache_view{};
    if (!graph_input) {
        return cache_status::null_input;
    }
    if (!ctx) {
        return cache_status::null_input;
    }

    // The cast back to the concrete type the hooked overload declared
    // (llama-graph.h:1389 -- `build_attn(llm_graph_input_attn_kv * inp, ...)`). static_cast from
    // void* is exact: the pointer came from that type and this file knows it.
    //
    // The msa variant derives from this one and passes its own cache context up through this
    // constructor, so reading the base field is correct for it too -- which is what makes a
    // single accessor serve the family rather than needing to identify the derived object.
    const auto * input = static_cast<const llm_graph_input_attn_kv *>(graph_input);

    if (!input->mctx) {
        return cache_status::no_cache_context;
    }

    // THE HEADROOM CHECK, before either view is asked for.
    //
    // get_k/get_v end in ggml_view_4d -> ggml_new_tensor_impl -> ggml_new_object, whose
    // out-of-space branch logs and returns NULL under NDEBUG but is followed by
    // `GGML_ASSERT(obj_new)` (ggml.c:1832). GGML_ASSERT expands to GGML_ABORT -> ggml_abort(...)
    // (ggml.h:114-115), a noreturn function that terminates the process. It is NOT an exception:
    // no catch anywhere in this file can intercept it, and claiming otherwise would be the kind
    // of assurance that costs a run. So the only defence is to not ask: the arithmetic below runs
    // first, and a refusal here is a decision this function owns.
    //
    // FAIL CLOSED on an inconsistent arena. The previous version read
    //   `if (used <= size && !headroom_ok(...))`
    // which means: when ggml reports used > size -- i.e. the accounting is already broken -- the
    // guard is SKIPPED and the calls proceed. That is backwards. used > size is not a benign
    // oddity; it is the one state in which `size - used` cannot even be formed (it would wrap to
    // a huge number and pass every headroom test), so there is no honest answer about room and
    // no honest view to build. Reported as its own status rather than folded into no_room: the
    // two mean different things to whoever reads the log.
    {
        const std::size_t used = ggml_used_mem(ctx);
        const std::size_t size = ggml_get_mem_size(ctx);
        if (used > size) {
            return cache_status::arena_inconsistent;
        }
        if (!cache_headroom_ok(size - used, ggml_tensor_overhead(), /*views=*/2)) {
            return cache_status::no_room;
        }
    }

    // The two calls, with the one failure mode a check cannot prevent.
    //
    // `llama_kv_cache::get_k` starts with `map_layer_ids.at(il)` (llama-kv-cache.cpp:1407), and
    // `unordered_map::at` THROWS for a key that is not present. A layer id that is not in the
    // cache is therefore an exception on the way out of a diagnostic, and an exception escaping
    // into upstream's graph builder would either fail a decode that was working or terminate.
    // Catching it converts both into a named refusal, which is the only outcome a probe may have.
    //
    // The check above and these calls are not atomic together: nothing between them allocates in
    // this thread, but another node cannot be precluded in principle, so the catch handles the
    // case the arithmetic cannot. That is why both exist rather than one.
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    try {
        k = input->mctx->get_k(ctx, layer);
        v = input->mctx->get_v(ctx, layer);
    } catch (const std::out_of_range &) {
        return cache_status::layer_not_cached;
    } catch (const std::exception &) {
        // Anything else is reported, not swallowed: an unnamed catch here would hide the
        // distinction between "this layer is not cached" and "something else went wrong".
        return cache_status::null_tensor;
    }

    if (!k || !v) {
        return cache_status::null_tensor;
    }

    out->k = k;
    out->v = v;
    for (int i = 0; i < 4; ++i) {
        out->ne_k[i] = k->ne[i];
        out->ne_v[i] = v->ne[i];
        out->nb_k[i] = k->nb[i];
        out->nb_v[i] = v->nb[i];
    }
    out->type_k = static_cast<int>(k->type);
    out->type_v = static_cast<int>(v->type);
    out->n_kv = input->mctx->get_n_kv();
    return cache_status::ok;
}

} // namespace herd::graph
