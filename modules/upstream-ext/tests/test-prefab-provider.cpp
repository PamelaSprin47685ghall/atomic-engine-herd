// The prefab statistics ABI, exercised through its real consumer path.
//
// `tq_upstream_ext.unavailable` proves the refusal when nothing is registered. That is half the
// contract; the half that has never run is everything after a provider EXISTS -- and the
// prefab-split lane registered one, so from now on a driver's `--prefab-stats` goes down this
// code. Two properties matter more than the field list:
//
//   * `captured` is emitted, and it is the provider's value. Without it the JSON presents
//     twenty-nine measured-looking zeros for a lane that counts two things, and the first reader
//     concludes "no reuse, no cache hits" from a run that had no cache. A JSON that cannot
//     distinguish "measured none" from "nobody counts it" lies by omission, so the distinction
//     being PRESENT is what this test asserts.
//
//   * the context the provider is asked about is the context the caller named. `prefab_stats_json`
//     loops over several contexts; a provider that ignored its argument would report context A's
//     numbers under context B's name, and the JSON would look entirely healthy.
//
// Registration is process-global and cannot be undone, so the order of the blocks below is the
// test's structure, not a style choice: every "nothing registered" assertion happens before the
// first successful registration, and the double-registration check happens after it.
#include <herd/upstream-ext.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// Two fake contexts. Addresses only: this build links no llama, and the provider below never
// dereferences them -- which is exactly the property `tq_upstream_ext.unavailable` already
// documents for the library itself ("every entry point takes an opaque llama_context pointer").
const llama_context * const kCtxA = reinterpret_cast<const llama_context *>(std::uintptr_t(0x1000));
const llama_context * const kCtxB = reinterpret_cast<const llama_context *>(std::uintptr_t(0x2000));

// What the last call saw, so the test can prove the argument survived the trip.
const llama_context * g_last_ctx = nullptr;
int g_get_stats_calls = 0;

bool provider_is_enabled(const llama_context * ctx) {
    g_last_ctx = ctx;
    return ctx == kCtxA;   // enabled for A only: enough to tell "was asked" from "was not"
}

void provider_get_stats(const llama_context * ctx, herd::ext::prefab_stats * out) {
    g_last_ctx = ctx;
    ++g_get_stats_calls;
    if (out == nullptr) return;
    *out = herd::ext::prefab_stats{};
    if (ctx != kCtxA) {
        // The real provider's rule: a context this thread has no reservation for reports nothing,
        // and `captured` stays 0. Mirrored here so the JSON for B is checked against a provider
        // that distinguishes the two, which is the whole point of the mask.
        return;
    }
    out->n_ubatch_prefab = 33;
    out->n_pad_rows = 198;
    out->captured = herd::ext::prefab_captured_n_ubatch_prefab |
                    herd::ext::prefab_captured_n_pad_rows;
}

std::size_t provider_n_cached_factors(const llama_context * ctx) {
    g_last_ctx = ctx;
    return ctx == kCtxA ? 0u : 0u;   // zero either way; the field's meaning is per-context
}

const herd::ext::prefab_provider g_provider{&provider_is_enabled, &provider_get_stats,
                                            &provider_n_cached_factors};

// A provider with a missing callback must be refused: a null `get_stats` would crash the first
// caller, and a registry that accepted it would have turned a configuration error into a fault
// far from its cause.
bool absent_callback_is_refused() {
    const herd::ext::prefab_provider broken{&provider_is_enabled, nullptr,
                                            &provider_n_cached_factors};
    return !herd::ext::prefab_register(&broken);
}

bool null_provider_is_refused() { return !herd::ext::prefab_register(nullptr); }

// Pulls `"captured":<n>` out of the JSON. Deliberately a search for the emitted spelling rather
// than a JSON parser: this test owns no JSON library, and the question is whether the field is
// PRESENT with the right value, which a substring with a type check can answer.
bool find_captured(const std::string & json, std::size_t from, std::uint64_t * value, std::size_t * at) {
    const std::string key = "\"captured\":";
    const std::size_t pos = json.find(key, from);
    if (pos == std::string::npos) return false;
    std::size_t i = pos + key.size();
    std::uint64_t v = 0;
    bool any = false;
    while (i < json.size() && json[i] >= '0' && json[i] <= '9') {
        v = v * 10 + std::uint64_t(json[i] - '0');
        ++i;
        any = true;
    }
    if (!any) return false;
    *value = v;
    *at = pos;
    return true;
}

void nothing_registered_yet() {
    check(!herd::ext::prefab_available(), "no provider is registered at process start");
    bool threw = false;
    try {
        herd::ext::prefab_stats_json({});
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "prefab_stats_json throws when nothing is registered");
    check(absent_callback_is_refused(), "a provider with a null get_stats is refused");
    check(null_provider_is_refused(), "a null provider is refused");
    check(!herd::ext::prefab_available(), "a refused registration still leaves nothing registered");
}

void registered_provider_is_exactly_one() {
    check(herd::ext::prefab_register(&g_provider), "the first valid registration succeeds");
    check(herd::ext::prefab_available(), "and the lane reports itself available");
    // A second owner for the same fact silently changes what every later reader sees, and the
    // prefab-split patch aborts when this returns false -- so the refusal is load-bearing.
    check(!herd::ext::prefab_register(&g_provider), "a second registration is refused");
    check(herd::ext::prefab_is_enabled(kCtxA), "is_enabled reaches the provider");
    check(!herd::ext::prefab_is_enabled(kCtxB), "is_enabled is not a constant");
}

void json_carries_captured_and_the_named_context() {
    const std::vector<std::pair<const char *, const llama_context *>> contexts{{"a", kCtxA},
                                                                             {"b", kCtxB}};
    const std::string json = herd::ext::prefab_stats_json(contexts);

    std::uint64_t first = 0, second = 0;
    std::size_t at = 0, at2 = 0;
    check(find_captured(json, 0, &first, &at), "the JSON carries \"captured\": at all");
    check(find_captured(json, at + 1, &second, &at2), "each context entry carries its own captured");
    const std::uint64_t both = herd::ext::prefab_captured_n_ubatch_prefab |
                               herd::ext::prefab_captured_n_pad_rows;
    check(first == both, "context a reports both counters as captured, got " +
                             std::to_string(first) + " want " + std::to_string(both));
    check(second == 0, "context b reports NOTHING captured, got " + std::to_string(second) +
                           " -- a zero that would otherwise read as a measurement");

    check(json.find("\"name\":\"a\"") != std::string::npos, "context a is named in the JSON");
    check(json.find("\"name\":\"b\"") != std::string::npos, "context b is named in the JSON");
    check(json.find("\"n_pad_rows\":198") != std::string::npos,
          "context a's measured value reaches the JSON");
    check(json.find("\"n_pad_rows\":0") != std::string::npos,
          "context b's uncaptured value is the zero it is, not a copy of a's");

    // The argument must survive the trip. Without this, a provider that cached the first context
    // it saw would pass every check above while answering the wrong question for B.
    g_get_stats_calls = 0;
    g_last_ctx = nullptr;
    herd::ext::prefab_get_stats(kCtxB, nullptr);
    check(g_last_ctx == kCtxB, "the provider is asked about the context the caller named");
    check(g_get_stats_calls == 1, "one call, not a loop");
}

}  // namespace

int main() {
    nothing_registered_yet();
    registered_provider_is_exactly_one();
    json_carries_captured_and_the_named_context();

    if (failures == 0) {
        std::fprintf(stderr, "prefab provider ABI: OK\n");
        return 0;
    }
    std::fprintf(stderr, "prefab provider ABI: %d failure(s)\n", failures);
    return 1;
}
