// The cache probe's arithmetic, on the two cases that would otherwise end the process.
//
// Two upstream facts make this file necessary, and neither is a hypothetical:
//
//   F1  llama_kv_cache::get_k / get_v begin with `map_layer_ids.at(il)`
//       (llama-kv-cache.cpp:1407, 1427). `std::unordered_map::at` THROWS for a key that is not
//       present, so a layer id that is not in the cache is an exception on its way out of a
//       diagnostic -- and an exception escaping into upstream's graph builder would either fail a
//       decode that was working or terminate.
//
//   F2  both calls end in ggml_view_4d -> ggml_new_tensor_impl -> ggml_new_object, whose
//       out-of-space branch returns NULL under NDEBUG and is followed by
//       `GGML_ASSERT(obj_new)` (ggml.c:1832). The assert is in EVERY build, so a view that does
//       not fit does not return null -- it ends the process.
//
// (F2 was measured, not read: an earlier version of the sibling probe test crashed in a RELEASE
// build with "ggml_new_tensor_impl: GGML_ASSERT(obj_new) failed". A claim that "release returns
// null, debug aborts" was half right, and the wrong half was the one that mattered.)
//
// The fix for both is arithmetic done BEFORE the call: `cache_headroom_ok`. That function is pure
// and is what this test measures -- including the boundary, which is the only interesting part,
// and including the overflow that a naive `free >= per_view * views` would get wrong.
//
// What this test cannot show: that a real arena with exactly N free bytes behaves as computed.
// That requires a live graph context and a real cache, and is covered instead by the probe's
// end-to-end run on a model. This file establishes the RULE; that run establishes the wiring.
#include "herd_graph_cache.hpp"

#include <cstdio>
#include <limits>
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
    using herd::graph::cache_headroom_ok;

    // F2's rule, at the boundary. Two views are what cache_tensors asks for, so the interesting
    // values are (room for exactly two), (room for one), (room for none).
    constexpr std::size_t perch = 368;   // a plausible ggml_tensor_overhead() value

    check(cache_headroom_ok(2 * perch, perch, 2),
          "exactly enough for two views is accepted: the check is a floor, not a margin");
    check(!cache_headroom_ok(2 * perch - 1, perch, 2),
          "one byte short of two views is refused: an almost-fits view aborts, it does not fail softly");
    check(cache_headroom_ok(2 * perch, perch, 1),
          "a caller asking for one view is judged against one");
    check(!cache_headroom_ok(perch, perch, 2),
          "room for one view is not room for two");
    check(cache_headroom_ok(0, perch, 0),
          "asking for no views needs no room");

    // A zero per-view cost is a caller passing nothing useful. Refusing is the safe reading:
    // accepting would divide the decision by a value that means nothing.
    check(!cache_headroom_ok(1024, 0, 1),
          "a zero per-view cost is refused rather than treated as unlimited room");

    // The overflow the naive form gets wrong. `free >= per_view * views` with a huge per_view and
    // views == 2 wraps to a small product and reports room; the division form does not.
    constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
    check(!cache_headroom_ok(kMax, kMax / 2 + 1, 2),
          "a per-view cost whose double overflows is not mistaken for a small one");
    check(cache_headroom_ok(kMax, 1, 2),
          "and a genuinely tiny per-view cost still fits");

    // The refusal codes exist and are distinguishable -- a caller that maps them to one message
    // loses the difference between "not cached" (a caller bug) and "no room" (a tight graph).
    check(herd::graph::cache_status::layer_not_cached != herd::graph::cache_status::no_room,
          "the two new refusals are distinct facts");
    const char * not_cached = herd::graph::describe(herd::graph::cache_status::layer_not_cached);
    const char * no_room = herd::graph::describe(herd::graph::cache_status::no_room);
    check(not_cached != nullptr && not_cached[0] != '\0', "layer_not_cached explains itself");
    check(no_room != nullptr && no_room[0] != '\0', "no_room explains itself");
    check(std::string(not_cached) != std::string(no_room),
          "and the two explanations differ, so a log line names the actual cause");

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "cache headroom: the rule refuses a view that cannot fit, before ggml asserts\n");
    return 0;
}
