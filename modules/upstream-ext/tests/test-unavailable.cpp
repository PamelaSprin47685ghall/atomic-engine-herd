// The point of this test: an unregistered capability must be an error, not an
// empty result. A driver that wrote a zero-filled stats file would look like a
// successful measurement of a feature that never ran.
#include "herd/upstream-ext.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

std::size_t g_n_cached = 7;
std::uint64_t g_ubatch_prefab = 42;

bool provider_enabled(const llama_context *) { return true; }
void provider_stats(const llama_context *, herd::ext::prefab_stats * out) { out->n_ubatch_prefab = g_ubatch_prefab; }
std::size_t provider_cached(const llama_context *) { return g_n_cached; }

const herd::ext::prefab_provider g_provider{&provider_enabled, &provider_stats, &provider_cached};

} // namespace

int main() {
    check(!herd::ext::prefab_available(), "no provider registered yet");

    bool threw = false;
    try {
        herd::ext::prefab_stats_json({{"ctx", nullptr}});
    } catch (const std::runtime_error & error) {
        threw = std::string(error.what()).find("prefab seam") != std::string::npos;
    }
    check(threw, "json throws naming the missing seam");

    threw = false;
    try {
        herd::ext::prefab_get_stats(nullptr, nullptr);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    check(threw, "get_stats throws while unregistered");

    check(herd::ext::prefab_register(&g_provider), "first registration accepted");
    check(!herd::ext::prefab_register(&g_provider), "second registration refused: one owner per fact");
    check(herd::ext::prefab_available(), "capability present after registration");

    const std::string json = herd::ext::prefab_stats_json({{"ctx", nullptr}});
    check(json.find("\"n_ubatch_prefab\":42") != std::string::npos, "provider counters reach the json");
    check(json.find("\"n_cached_factors\":7") != std::string::npos, "cached-factor count reaches the json");
    check(json.find("\"is_enabled\":true") != std::string::npos, "enabled flag reaches the json");

    const herd::ext::prefab_provider incomplete{&provider_enabled, nullptr, &provider_cached};
    check(!herd::ext::prefab_register(&incomplete), "incomplete provider refused");

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "prefab availability contract holds\n");
    return 0;
}
