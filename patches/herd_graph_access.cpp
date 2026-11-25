// The only translation unit that knows llm_graph_context exists.
//
// Everything else includes herd_graph_access.hpp and sees `struct ggml_context`. The point
// of the isolation is the header it has to read: llama-graph.h pulls in the model, the KV
// cache and the batch headers, and a layout change there must not ripple across the tree.
// One file pays, at build time, and it is this one.
#include "herd_graph_access.hpp"

#include <ggml.h>

#include <atomic>
#include <cstdio>

// Upstream's private header, read and never written. Including it is what makes the
// offset a compile-time fact instead of a number someone typed.
#include "llama-graph.h"

namespace herd::graph::detail {

// An explicit template instantiation is not subject to access control ([temp.spec]/6), so
// this captures a pointer to a PRIVATE data member without renaming `private` and without
// depending on the class being standard-layout.
//
// The captured object is a pointer-to-member: for a non-virtual data member that is an
// offset, computed by the compiler from the same header that laid out the object. There
// is no number in this file to get stale.
template <typename Tag, typename MemberPtr, MemberPtr Ptr>
struct accessor {
    friend MemberPtr steal(Tag) { return Ptr; }
};

struct ctx0_tag {};

// Declared here, defined by the explicit instantiation below as the friend above. The
// declaration is required: it is what tells the compiler the name exists, and the
// definition arrives as an ADL-only friend.
ggml_context * llm_graph_context::* steal(ctx0_tag);

template struct accessor<ctx0_tag, ggml_context * llm_graph_context::*, &llm_graph_context::ctx0>;

// The same capture for `res`, and it is the same mechanism rather than a second idea: a
// pointer-to-member computed by the compiler from upstream's header, so a rename or a removal
// fails this build instead of reading a neighbouring field.
struct res_tag {};

llm_graph_result * llm_graph_context::* steal(res_tag);

template struct accessor<res_tag, llm_graph_result * llm_graph_context::*, &llm_graph_context::res>;

} // namespace herd::graph::detail

namespace herd::graph {

bool accessor_compiled() {
    // Reachable by construction: this file would not compile if the member were gone. The
    // function exists so a caller can say "this build can see the context" instead of
    // assuming it, and so a build that drops this translation unit has a way to answer.
    return true;
}

ggml_context * context_of(const void * self) {
    if (!self) {
        return nullptr;
    }
    // static_cast from void*, not reinterpret_cast: the pointer is the llm_graph_context
    // subobject by the language's own rule for member functions, and this states that.
    const auto * graph = static_cast<const llm_graph_context *>(self);
    return graph->*detail::steal(detail::ctx0_tag{});
}

void * result_of(const void * self) {
    if (!self) {
        return nullptr;
    }
    const auto * graph = static_cast<const llm_graph_context *>(self);
    // Returned as void*: the type's definition stays in this file. The caller's cast is the
    // statement that its translation unit includes llama-graph.h.
    return static_cast<void *>(graph->*detail::steal(detail::res_tag{}));
}

bool arena_usage(ggml_context * ctx, std::size_t * used, std::size_t * size) {
    if (!ctx || !used || !size) {
        return false;
    }
    // These two calls read the context. That is the limit of this function, and the header
    // says so: a pointer that is not a ggml_context faults inside these reads, and no
    // sequence of checks in this file can change that -- the check would have to dereference
    // the pointer to decide whether it may be dereferenced.
    const std::size_t u = ggml_used_mem(ctx);
    const std::size_t s = ggml_get_mem_size(ctx);
    *used = u;
    *size = s;
    // The only contradiction that is checkable once the reads above succeeded: an arena
    // cannot hold more than it owns. (ggml also keeps its allocations in a list; a truncated
    // context could still pass this, which is why the header refuses to call it validation.)
    return u <= s;
}

namespace {

// The exact charge for one F32x1 tensor in a context, MEASURED per flavour.
//
// Why this replaced ggml_tensor_overhead(). That function is GGML_OBJECT_SIZE +
// GGML_TENSOR_SIZE and does not include the tensor's DATA. What ggml actually charges, read
// out of ggml.c:1820-1834:
//
//     size_t obj_alloc_size = 0;
//     if (view_src == NULL && !ctx->no_alloc) {
//         obj_alloc_size = data_size;                  // 4 bytes for one F32
//     }
//     ggml_new_object(ctx, GGML_OBJECT_TYPE_TENSOR, GGML_TENSOR_SIZE + obj_alloc_size);
//
// and ggml_new_object (ggml.c:1710-1750) then pads that to GGML_MEM_ALIGN and adds
// GGML_OBJECT_SIZE before it compares:
//
//     size_t size_needed = GGML_PAD(size, GGML_MEM_ALIGN);
//     if (cur_end + size_needed + GGML_OBJECT_SIZE > ctx->mem_size) { GGML_ABORT; }
//
// So in a context whose no_alloc is FALSE the requirement is overhead() + 4 + up to
// GGML_MEM_ALIGN-1 bytes of padding. The precheck compared against overhead() alone, so it
// under-counted by exactly the part that is easiest to forget -- the data -- and an arena with
// overhead() bytes free would pass the check and then hit the GGML_ASSERT this precheck exists
// to avoid. The probe would abort the inference it was measuring, in every build.
//
// The fix does not transcribe GGML_OBJECT_SIZE, GGML_TENSOR_SIZE or GGML_MEM_ALIGN. Those are
// internal to ggml-impl.h, and a copy of them here would be a second thing to keep true -- the
// failure mode this project keeps removing. Instead the cost is measured once per process, per
// flavour, from ggml's own accounting: create a throwaway context, read ggml_used_mem before
// and after one ggml_new_tensor_1d(F32, 1), and the delta is the exact per-tensor charge.
//
// The delta is also the requirement for the Nth tensor, not just the first: ggml_new_object
// only ever appends at cur_end, which is already aligned, so every tensor of this shape costs
// the same. That is what makes one measurement sufficient.
//
// Both flavours are measured because they differ, and the difference is the defect: a context
// created with no_alloc = true charges the header only (llama's graph contexts are built that
// way -- the scheduler allocates the data), while no_alloc = false charges header plus data.
std::size_t measure_cost(const bool no_alloc) {
    // The pool has to be supplied by hand for the no_alloc flavour: ggml_init with
    // no_alloc = true and a null buffer asks for zero bytes of memory, which cannot hold even
    // the object header this measurement needs.
    std::vector<char> supplied(64u * 1024u);

    ggml_init_params params{};
    params.mem_size = supplied.size();
    params.mem_buffer = no_alloc ? supplied.data() : nullptr;
    params.no_alloc = no_alloc;

    ggml_context * probe_ctx = ggml_init(params);
    if (!probe_ctx) {
        return 0;
    }
    const std::size_t before = ggml_used_mem(probe_ctx);
    // One tensor, exactly the shape the deep probe asks for. If this aborted, ggml would die
    // here -- in one 64 KiB throwaway context, at first use -- instead of inside a user's
    // inference, which is the right place for a wrong assumption to fail.
    ggml_new_tensor_1d(probe_ctx, GGML_TYPE_F32, 1);
    const std::size_t after = ggml_used_mem(probe_ctx);
    ggml_free(probe_ctx);
    return after > before ? after - before : 0;
}

std::size_t cost_of_one_tensor(ggml_context * ctx) {
    // Function-local statics: thread-safe one-time initialization, and no namespace-scope
    // object whose initialization could be ordered against anything else's.
    static const std::size_t allocated_cost   = measure_cost(false);
    static const std::size_t no_alloc_cost    = measure_cost(true);

    const bool no_alloc = ggml_get_no_alloc(ctx);
    const std::size_t measured = no_alloc ? no_alloc_cost : allocated_cost;
    if (measured != 0) {
        return measured;
    }
    // The measurement failed -- a 64 KiB context could not be created, which means the process
    // is already out of memory. Over-estimate rather than under-estimate: a false refusal
    // costs a diagnostic, an under-estimate costs an abort. Twice overhead() is comfortably
    // above header + data + padding for any alignment ggml uses.
    return 2 * ggml_tensor_overhead();
}

} // namespace

probe_result probe_one_tensor(ggml_context * ctx, std::size_t * used_after, std::size_t * size) {
    if (!ctx) {
        return probe_result::null_context;
    }

    // One shot for the whole process, spent atomically.
    //
    // Called from a per-layer hook, this used to run once per layer per decode: each call
    // appended a tensor header to the live graph's arena, and a ggml arena has no per-tensor
    // free, so the probe's cost accumulated for as long as the run lasted. The previous comment
    // described that cost as "one tensor header, in an arena about to be freed with the graph"
    // -- true for one graph build, false for a run, and the version that actually ran was the
    // second. exchange() makes the spend unforgeable under concurrent hooks: at most one thread
    // sees `false` and proceeds.
    static std::atomic<bool> spent{false};
    if (spent.exchange(true, std::memory_order_acq_rel)) {
        return probe_result::already_done;
    }

    std::size_t before_used = 0;
    std::size_t arena_size = 0;
    if (!arena_usage(ctx, &before_used, &arena_size)) {
        // Contradictory fields. Reported by the caller as an inconsistency, not as "not a
        // ggml_context": this build cannot tell those two apart.
        std::fprintf(stderr,
                     "[herd] graph: arena reports used %zu > size %zu, which no ggml_context "
                     "does; treat the graph context as untrustworthy\n",
                     before_used, arena_size);
        return probe_result::exhausted;
    }

    // THE HEADROOM CHECK, and it is load-bearing rather than tidy.
    //
    // Read out of ggml.c, and the reading changed the conclusion twice:
    //
    //   ggml.c:1749-1755  ggml_new_object's out-of-space branch logs a warning, returns NULL
    //                     under NDEBUG, and GGML_ABORTs without it;
    //   ggml.c:1832-1833  but ggml_new_tensor_impl -- the only path to the function above --
    //                     follows it with `GGML_ASSERT(obj_new)`.
    //
    // So the null never reaches a caller. The assert runs in EVERY build, which I established
    // by building this test in Release and watching `ggml_new_tensor_impl: GGML_ASSERT(obj_new)
    // failed` end the process. The earlier note in this file said a debug build would abort
    // and a release build would return null; that was half right, and the wrong half was the
    // one that mattered -- there is no configuration in which asking for memory that is not
    // there is survivable.
    //
    // A probe called from inside an inference therefore cannot ask first and interpret later.
    // It must decide not to ask, and the decision must be a comparison this function owns.
    //
    // The threshold is ggml_tensor_overhead(), which is GGML_OBJECT_SIZE + GGML_TENSOR_SIZE --
    // exactly the two parts ggml_new_object pads and charges for (ggml.c:1479, :1832). The
    // free remainder is measured, not assumed, so the margin does not depend on the
    // architecture's pointer size or on GGML_MEM_ALIGN.
    // The charge for one tensor in THIS context, measured rather than assumed: the header
    // alone when no_alloc is true, header plus data plus padding when it is false. See
    // cost_of_one_tensor() -- comparing against ggml_tensor_overhead() here was the defect,
    // because that number excludes the 4 bytes of data an allocating context charges.
    const std::size_t overhead = cost_of_one_tensor(ctx);
    const std::size_t free_bytes = arena_size - before_used;
    if (free_bytes < overhead) {
        std::fprintf(stderr,
                     "[herd] graph: declining the deep probe: %zu bytes free, %zu needed for one "
                     "F32x1 tensor in a context with no_alloc=%s. Not attempted, because ggml "
                     "asserts on an overfull arena in every build (ggml.c:1834, reached through "
                     "ggml_new_tensor_impl)\n",
                     free_bytes, overhead, ggml_get_no_alloc(ctx) ? "true" : "false");
        return probe_result::exhausted;
    }

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    if (!tensor) {
        // Now genuinely unexpected, and reported that way.
        //
        // The headroom check above removed the one cause that was predictable -- an arena with
        // no room -- so a null here is no longer explained by space and must not be described
        // as if it were. The previous version spent a paragraph distinguishing "full" from
        // "wrong context" AFTER the call, which was the right worry applied at the wrong time:
        // by then ggml may have aborted already. The distinction now happens before the call,
        // where the answer is still this function's to give.
        std::size_t after_used = 0;
        std::size_t after_size = 0;
        if (arena_usage(ctx, &after_used, &after_size) && after_used <= after_size) {
            std::fprintf(stderr,
                         "[herd] graph: ggml_new_tensor_1d returned null with %zu bytes free, "
                         "more than the %zu a header needs; the arena accounting does not "
                         "explain it\n",
                         after_size - after_used, overhead);
        } else {
            std::fprintf(stderr,
                         "[herd] graph: ggml_new_tensor_1d returned null and the arena's "
                         "accounting is inconsistent\n");
        }
        return probe_result::exhausted;
    }

    // The fields ggml wrote are the check: they are set by the same library that would have
    // to be fooled. `data == nullptr` is expected because a new tensor is unallocated until a
    // graph allocator places it.
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 1 || tensor->ne[1] != 1 ||
        ggml_nbytes(tensor) != sizeof(float) || tensor->data != nullptr) {
        std::fprintf(stderr,
                     "[herd] graph: the created tensor disagrees with the request (type %d, "
                     "ne0 %lld, nbytes %zu, data %p)\n",
                     (int) tensor->type, (long long) tensor->ne[0], ggml_nbytes(tensor),
                     tensor->data);
        return probe_result::exhausted;
    }

    std::size_t final_used = 0;
    std::size_t final_size = 0;
    if (arena_usage(ctx, &final_used, &final_size)) {
        if (used_after) { *used_after = final_used; }
        if (size)       { *size = final_size; }
        std::fprintf(stderr,
                     "[herd] graph: one tensor created in the live arena (%zu -> %zu of %zu "
                     "bytes); this cost is permanent for this graph\n",
                     before_used, final_used, final_size);
    }
    return probe_result::created;
}

} // namespace herd::graph
