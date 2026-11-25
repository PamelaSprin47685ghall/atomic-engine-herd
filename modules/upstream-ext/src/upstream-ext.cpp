#include "herd/upstream-ext.h"

#include <stdexcept>

namespace herd::ext {
namespace {

const prefab_provider * g_provider = nullptr;

[[noreturn]] void unavailable(const char * what) {
    throw std::runtime_error(std::string(what) +
                             " requires the prefab seam, which no patch module has registered in this build");
}

} // namespace

bool prefab_available() { return g_provider != nullptr; }

bool prefab_register(const prefab_provider * provider) {
    if (!provider || !provider->is_enabled || !provider->get_stats || !provider->n_cached_factors) {
        return false;
    }
    if (g_provider) {
        return false;
    }
    g_provider = provider;
    return true;
}

void prefab_get_stats(const llama_context * ctx, prefab_stats * out) {
    if (!g_provider) unavailable("herd::ext::prefab_get_stats");
    g_provider->get_stats(ctx, out);
}

bool prefab_is_enabled(const llama_context * ctx) {
    if (!g_provider) unavailable("herd::ext::prefab_is_enabled");
    return g_provider->is_enabled(ctx);
}

std::size_t prefab_n_cached_factors(const llama_context * ctx) {
    if (!g_provider) unavailable("herd::ext::prefab_n_cached_factors");
    return g_provider->n_cached_factors(ctx);
}

std::string prefab_stats_json(const std::vector<std::pair<const char *, const llama_context *>> & contexts) {
    if (!g_provider) unavailable("herd::ext::prefab_stats_json");

    std::string out = "{\"contexts\":[";
    bool first_ctx = true;
    for (const auto & entry : contexts) {
        if (!first_ctx) { out += ","; }
        first_ctx = false;
        prefab_stats ps{};
        prefab_get_stats(entry.second, &ps);
        out += "{\"name\":\"";
        out += entry.first;
        out += "\",\"is_enabled\":";
        out += prefab_is_enabled(entry.second) ? "true" : "false";
        out += ",\"n_cached_factors\":";
        out += std::to_string(prefab_n_cached_factors(entry.second));
        out += ",\"stats\":{";
        const std::pair<const char *, std::uint64_t> fields[] = {
            {"n_ubatch_prefab",       ps.n_ubatch_prefab},
            {"n_ubatch_stock",        ps.n_ubatch_stock},
            {"n_stock_builds",        ps.n_stock_builds},
            {"n_stock_reuses",        ps.n_stock_reuses},
            {"n_envelope_builds",     ps.n_envelope_builds},
            {"n_envelope_reuses",     ps.n_envelope_reuses},
            {"n_envelope_allocs",     ps.n_envelope_allocs},
            {"n_sub_envelope",        ps.n_sub_envelope},
            {"n_pad_rows",            ps.n_pad_rows},
            {"n_factor_compiles",     ps.n_factor_compiles},
            {"n_factor_cache_hits",   ps.n_factor_cache_hits},
            {"n_factor_splices",      ps.n_factor_splices},
            {"n_factor_clones",       ps.n_factor_clones},
            {"n_cache_resets",        ps.n_cache_resets},
            {"n_fallback_no_scratch", ps.n_fallback_no_scratch},
            {"n_fallback_runtime",    ps.n_fallback_runtime},
            {"last_n_env",            ps.last_n_env},
            {"last_n_tokens",         ps.last_n_tokens},
            {"last_graph_nodes",      ps.last_graph_nodes},
            {"last_inp_tokens_ne0",   ps.last_inp_tokens_ne0},
            {"n_decouple_declines",   ps.n_decouple_declines},
            {"n_decouple_handled",    ps.n_decouple_handled},
            {"n_mtp_compose_hits",    ps.n_mtp_compose_hits},
            {"n_mtp_a_compiles",      ps.n_mtp_a_compiles},
            {"n_mtp_a_reuses",        ps.n_mtp_a_reuses},
            {"n_mtp_b_builds",        ps.n_mtp_b_builds},
            {"n_mtp_a_clones",        ps.n_mtp_a_clones},
            {"n_mtp_declines",        ps.n_mtp_declines},
            {"build_ns",              ps.build_ns},
            {"n_nodes_built",         ps.n_nodes_built},
            {"template_hits",         ps.template_hits},
        };
        for (std::size_t k = 0; k < sizeof(fields) / sizeof(fields[0]); ++k) {
            if (k) { out += ","; }
            out += "\"";
            out += fields[k].first;
            out += "\":";
            out += std::to_string(fields[k].second);
        }
        // Emitted inside `stats` so a reader cannot see the numbers without seeing whether
        // they were measured. Named `captured` rather than a list of names: the bit positions
        // are in the header, next to the fields they describe, and a second copy of that
        // mapping here would be a second place to keep right.
        out += ",\"captured\":";
        out += std::to_string(ps.captured);
        out += "}}";
    }
    out += "]}";
    return out;
}

} // namespace herd::ext
