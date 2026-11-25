// herd/upstream-ext.h — capabilities the fork put inside llama.h.
//
// The old tree carried extra declarations in include/llama.h: prefab statistics,
// decode-wrapper counters, and friends. In this project those names cannot exist
// in upstream's header (upstream is a pinned submodule and stays byte-identical),
// so the capability moves here, as our own interface, and a seam supplies it at
// runtime.
//
// The rule this header exists to make possible: a capability that no seam
// provides is *absent*, and callers must say so. There is no default no-op and
// no zero-filled struct that pretends a disabled feature produced measurements.
// A driver that needs prefab numbers fails with a message naming the missing
// seam, instead of writing a JSON file full of zeroes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct llama_context;

namespace herd::ext {

// Mirrors the counters the fork's prefab lane reported. Field order is part of
// the JSON contract that tooling downstream already parses, so it is preserved;
// nothing here is computed by upstream, which is why a provider must register.
struct prefab_stats {
    std::uint64_t n_ubatch_prefab;
    std::uint64_t n_ubatch_stock;
    std::uint64_t n_stock_builds;
    std::uint64_t n_stock_reuses;
    std::uint64_t n_envelope_builds;
    std::uint64_t n_envelope_reuses;
    std::uint64_t n_envelope_allocs;
    std::uint64_t n_sub_envelope;
    std::uint64_t n_pad_rows;
    std::uint64_t n_factor_compiles;
    std::uint64_t n_factor_cache_hits;
    std::uint64_t n_factor_splices;
    std::uint64_t n_factor_clones;
    std::uint64_t n_cache_resets;
    std::uint64_t n_fallback_no_scratch;
    std::uint64_t n_fallback_runtime;
    std::uint64_t last_n_env;
    std::uint64_t last_n_tokens;
    std::uint64_t last_graph_nodes;
    std::uint64_t last_inp_tokens_ne0;
    std::uint64_t n_decouple_declines;
    std::uint64_t n_decouple_handled;
    std::uint64_t n_mtp_compose_hits;
    std::uint64_t n_mtp_a_compiles;
    std::uint64_t n_mtp_a_reuses;
    std::uint64_t n_mtp_b_builds;
    std::uint64_t n_mtp_a_clones;
    std::uint64_t n_mtp_declines;
    std::uint64_t build_ns;
    std::uint64_t n_nodes_built;
    std::uint64_t template_hits;

    // Which counters above this provider actually measures. A set bit means the value is
    // measured; a clear bit means the field is zero because NOTHING COUNTS IT, not because
    // the count happened to be zero.
    //
    // This field exists because of a near miss. The prefab-split lane counts exactly two of
    // the counters above -- ubatches widened, and rows padded. Registering a provider for it
    // without this mask would have published twenty-nine measured-looking zeros, and the
    // first reader would have concluded "no reuse, no cache hits, no fallbacks" from a lane
    // that has no cache and no factors at all. A zero that means "not applicable" and a zero
    // that means "measured none" are different facts, and a JSON document that cannot tell
    // them apart is a document that lies by omission.
    //
    // Bit i corresponds to the i-th counter above, in declaration order. Consumers that only
    // read the named fields are unaffected; consumers that want to know whether a number is
    // real read this first.
    std::uint64_t captured;
};

// Bit positions for `prefab_stats::captured`, in declaration order.
enum : std::uint64_t {
    prefab_captured_n_ubatch_prefab = 1ull << 0,
    prefab_captured_n_pad_rows      = 1ull << 8,
};

// Implemented by a patch module (patches/…), never by upstream. All three
// entries must be non-null; registration is one-shot and refuses a second
// provider, so a stale seam cannot silently shadow a live one.
struct prefab_provider {
    bool   (*is_enabled)      (const llama_context *);
    void   (*get_stats)       (const llama_context *, prefab_stats *);
    std::size_t (*n_cached_factors)(const llama_context *);
};

// False until a patch module registers. Query this instead of assuming the
// capability exists: the answer decides whether the driver can honour a request.
bool prefab_available();

// Returns false when a provider was already registered (or the argument is
// incomplete). A seam module should treat false as a fatal configuration error:
// two providers means two owners for one fact.
bool prefab_register(const prefab_provider * provider);

void   prefab_get_stats(const llama_context * ctx, prefab_stats * out);
bool   prefab_is_enabled(const llama_context * ctx);
std::size_t prefab_n_cached_factors(const llama_context * ctx);

// The JSON document the fork's drivers emitted. Throws std::runtime_error when
// no provider is registered: an unavailable capability is an error, not an
// empty result.
std::string prefab_stats_json(const std::vector<std::pair<const char *, const llama_context *>> & contexts);

} // namespace herd::ext
