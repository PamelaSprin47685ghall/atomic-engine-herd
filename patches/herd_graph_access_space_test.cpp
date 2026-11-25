// The deep probe's refusal path, on an arena that is deliberately too small.
//
// The question this answers was open: `probe_one_tensor` reported `exhausted` when the arena
// had no room, and the previous version discovered that by calling ggml_new_tensor_1d and
// reading a null. That reading is only available in a build where NDEBUG is defined. In an
// assert-enabled build the same call runs GGML_ABORT (ggml.c:1749-1755) and the process dies,
// which for a diagnostic called from inside an inference is the worst possible outcome: the
// probe would kill the run it was measuring, and the log would say "not enough space in the
// context's memory pool" from a library the caller never knowingly invoked.
//
// So the property under test is not "does it return the right enum" but "does the process
// survive an arena that cannot hold one more tensor". A test that only checked the enum would
// pass in Release and never be run in Debug, which is exactly the blind spot that made this a
// defect. The assertion below is reachable in both: the call must return, and it must return
// `exhausted` without the library having been asked.
#include "herd_graph_access.hpp"

#include <ggml.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

} // namespace

int main() {
    // A context with room for exactly one tensor header and nothing more.
    //
    // The size is computed from ggml's own report rather than guessed, so the margin does not
    // depend on the architecture's pointer size or on GGML_MEM_ALIGN.
    const std::size_t overhead = ggml_tensor_overhead();
    std::fprintf(stderr, "ggml_tensor_overhead() = %zu bytes\n", overhead);

    ggml_init_params params{overhead + 64, nullptr, false};
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "a small arena can be created");
    if (!ctx) {
        return 1;
    }

    // Fill it to the brim -- but by asking the arena, never by asking ggml to allocate.
    //
    // `while (ggml_new_tensor_1d(...) != nullptr)` was the obvious loop and it is the one that
    // crashed this test in a RELEASE build: the null it waited for is never returned, because
    // ggml_new_tensor_impl asserts on it (ggml.c:1833). The failure was not in the probe but in
    // the fixture -- a loop that relies on a library's OOM behaviour to terminate is a loop
    // that aborts instead of terminating.
    //
    // So the loop stops when the free remainder can no longer hold one header, which is the
    // same comparison probe_one_tensor makes. If ggml's own bookkeeping disagrees with that
    // comparison by even one header, the probe below would be the thing that finds out.
    int placed = 0;
    for (;;) {
        std::size_t used_now = 0;
        std::size_t size_now = 0;
        if (!herd::graph::arena_usage(ctx, &used_now, &size_now)) {
            break;
        }
        if (size_now - used_now < overhead) {
            break;
        }
        if (ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1) == nullptr) {
            break;
        }
        ++placed;
        if (placed > 64) {
            break;
        }
    }
    std::fprintf(stderr, "filled the arena with %d tensor(s)\n", placed);

    std::size_t used = 0;
    std::size_t size = 0;
    check(herd::graph::arena_usage(ctx, &used, &size), "the arena's accounting is consistent");
    std::fprintf(stderr, "arena: %zu of %zu bytes used, %zu free (one header needs %zu)\n", used,
                 size, size - used, overhead);

    // The property: the probe declines, and the process is still here to see it.
    //
    // If the headroom check were missing, this call would abort here in a non-NDEBUG build --
    // the test would not fail with a message, the process would vanish. That asymmetry is why
    // the test asserts survival as well as the verdict.
    std::size_t after = 0;
    std::size_t after_size = 0;
    const herd::graph::probe_result result = herd::graph::probe_one_tensor(ctx, &after, &after_size);

    check(result == herd::graph::probe_result::exhausted,
          "on a full arena the deep probe declines instead of asking ggml for a header");
    check(std::fprintf(stderr, "probe result: exhausted (the process survived)\n") > 0,
          "reaching this line at all is the assertion that no abort happened");

    // The arena must be untouched: a refusal that quietly consumed something would be a
    // mutation reported as an observation.
    std::size_t used_now = 0;
    std::size_t size_now = 0;
    check(herd::graph::arena_usage(ctx, &used_now, &size_now), "still consistent after the probe");
    check(used_now == used && size_now == size,
          "a declined probe wrote nothing into the arena");

    // And the probe is spent: a second call reports already_done regardless of space. This is
    // the separate property that keeps the deep probe from running once per layer, and it must
    // hold even on the path that never allocated.
    const herd::graph::probe_result second = herd::graph::probe_one_tensor(ctx, &after, &after_size);
    check(second == herd::graph::probe_result::already_done,
          "the one allowed deep probe is spent even when it declined");

    // The null path is answered without touching anything.
    check(herd::graph::probe_one_tensor(nullptr, nullptr, nullptr) ==
              herd::graph::probe_result::null_context,
          "a null context is named, not dereferenced");

    ggml_free(ctx);

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "deep probe: declines on a full arena without aborting, writes nothing, and "
                 "spends its one shot\n");
    return 0;
}
