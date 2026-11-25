#include <llama.h>
#include <ggml-backend.h>
#include <tq/scheduling/prefill.h>
#include <tq/scheduling/admission.h>
#include <tq/text/stream.h>
#include "herd_phase.hpp"
#include "herd_nkv.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using tq::scheduling::stage;

struct options {
    std::string model;
    std::string prompts = "-";
    std::uint32_t predict = 32;
    std::uint32_t context = 2048;
    std::uint32_t total_kv = 0;
    std::uint32_t batch = 512;
    std::uint32_t ubatch = 128;
    std::uint32_t threads = 4;
    std::uint32_t gpu_layers = 0;
    std::uint32_t parallel = 0;
    std::optional<std::uint32_t> inject_after;
    std::vector<std::string> stops;
    bool phase_memory = true;
    bool ignore_eog = false;
    bool json = false;
    bool stream = false;
    bool help = false;
};

std::uint32_t number(std::string_view text, const std::string & name, bool allow_zero = false) {
    std::uint32_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        (!allow_zero && value == 0) || value > std::uint32_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(name + " requires a " + (allow_zero ? "non-negative" : "positive") +
                                    " decimal integer <= INT32_MAX");
    }
    return value;
}

void usage() {
    std::cout <<
        "Usage: tq-batch-compose --model MODEL.gguf [options]\n"
        "  --prompts FILE        One raw prompt per line; default: standard input (-).\n"
        "  --predict N           Generated tokens per request; default: 32.\n"
        "  --ctx-size N          Logical context per sequence; default: 2048.\n"
        "  --total-kv N          Explicit physical unified KV capacity in tokens.\n"
        "                        Default: enough for the largest concurrent requests.\n"
        "  --parallel N          Maximum resident requests; default: min(input count,\n"
        "                        batch-size). Queue excess requests in input order.\n"
        "  --batch-size N        Maximum rows per decode call; default: 512.\n"
        "  --ubatch-size N       Maximum physical microbatch; default: 128.\n"
        "  --threads N           CPU threads; default: 4.\n"
        "  --gpu-layers N        Requested offloaded layers; default: 0.\n"
        "  --inject-after N      Start two requests, then admit the remaining prompts\n"
        "                        after N successful generation batches.\n"
        "  --phase-memory        Release compute memory at prompt-to-generation\n"
        "                        transitions (default); errors are fatal.\n"
        "  --no-phase-memory     Keep compute memory for a real comparison.\n"
        "  --ignore-eog          Ignore model EOG tokens; stop strings still apply.\n"
        "  --stop TEXT           Stop at the first completed string; repeatable.\n"
        "  --stream              Flush UTF-8 text deltas; requires --json.\n"
        "  --json                Write JSON Lines for real batches and results.\n";
}

options parse(int argc, char ** argv, bool & json_errors) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") { result.json = true; json_errors = true; continue; }
        if (arg == "--help") { result.help = true; continue; }
        if (arg == "--stream") { result.stream = true; continue; }
        if (arg == "--phase-memory") { result.phase_memory = true; continue; }
        if (arg == "--no-phase-memory") { result.phase_memory = false; continue; }
        if (arg == "--ignore-eog") { result.ignore_eog = true; continue; }
        if (arg != "--model" && arg != "--prompts" && arg != "--predict" && arg != "--ctx-size" &&
            arg != "--total-kv" && arg != "--batch-size" && arg != "--ubatch-size" &&
            arg != "--threads" && arg != "--gpu-layers" && arg != "--inject-after" && arg != "--parallel" && arg != "--stop") {
            throw std::invalid_argument("unknown option: " + arg);
        }
        if (++i == argc) { throw std::invalid_argument("missing value for " + arg); }
        const std::string value = argv[i];
        if (arg == "--model") { result.model = value; }
        else if (arg == "--stop") { result.stops.push_back(value); }
        else if (arg == "--prompts") { result.prompts = value; }
        else if (arg == "--predict") { result.predict = number(value, arg); }
        else if (arg == "--ctx-size") { result.context = number(value, arg); }
        else if (arg == "--total-kv") { result.total_kv = number(value, arg); }
        else if (arg == "--batch-size") { result.batch = number(value, arg); }
        else if (arg == "--ubatch-size") { result.ubatch = number(value, arg); }
        else if (arg == "--threads") { result.threads = number(value, arg); }
        else if (arg == "--gpu-layers") { result.gpu_layers = number(value, arg, true); }
        else if (arg == "--parallel") { result.parallel = number(value, arg); }
        else { result.inject_after = number(value, arg); }
    }
    if (result.help) { return result; }
    if (result.model.empty()) { throw std::invalid_argument("--model is required"); }
    if (result.stream && !result.json) { throw std::invalid_argument("--stream requires --json"); }
    // The fork released compute memory at prompt-to-generation by default, and
    // kept a separate physical KV capacity. This build can do neither, and both
    // are reported here, before a model is loaded, rather than absorbed:
    // a run that quietly used another configuration would succeed while
    // describing something the operator did not ask for. Each message names the
    // supported choice, so the fix is in the message.
    if (result.phase_memory && !herd::phase::release_available()) {
        // Ask the owner instead of repeating the answer here. The previous text hard-coded
        // "no patch module provides it", which went stale the moment the mechanism was
        // measured, and the JSON field beside it was filled from a different source of
        // truth -- so the report and the refusal could disagree. One fact, one owner.
        throw std::invalid_argument(
            std::string("cannot honour --phase-memory: ") +
            herd::phase::release_unavailable_reason() +
            ". Pass --no-phase-memory to select the supported mode explicitly");
    }
    // --total-kv has a mechanism now (patch module nkv), so the check is about what the
    // request MEANS rather than about whether it is possible:
    //
    //   below the logical context  -> refused: a cache smaller than the window would
    //                                 withhold cells the context legitimately uses
    //   equal                      -> the ordinary configuration
    //   above                      -> decoupled capacity, applied through a scoped
    //                                 request around this one init call (see below)
    //
    // The old message said "needs the nkv seam, which no patch module provides", which
    // was true when it was written and is now false. A refusal that states a stale
    // reason is worse than no refusal: it tells the operator to stop asking.
    if (result.total_kv != 0 && std::uint64_t(result.total_kv) < std::uint64_t(result.context)) {
        throw std::invalid_argument(
            "--total-kv " + std::to_string(result.total_kv) + " is smaller than --ctx-size " +
            std::to_string(result.context) + ": a cache below the logical window withholds cells the "
            "context uses. Omit --total-kv, or set it at least as large as --ctx-size");
    }
    const tq::text::generation_stream validated_stops(result.stops);
    if (result.ubatch > result.batch) { throw std::invalid_argument("--ubatch-size exceeds --batch-size"); }
    if (result.parallel > result.batch) { throw std::invalid_argument("--parallel exceeds --batch-size"); }
    if (result.inject_after && result.parallel == 1) { throw std::invalid_argument("--inject-after requires at least two resident requests"); }
    if (result.inject_after && std::uint64_t(*result.inject_after) + 2 > result.predict) {
        throw std::invalid_argument("injection while two requests are generating requires --predict >= --inject-after + 2");
    }
    return result;
}

// Keep valid UTF-8; replace incomplete or invalid bytes in display text. Token
// ids in results and traces remain exact, including a truncated UTF-8 tail.
std::string quote(std::string_view text) {
    std::string result = "\"";
    const char hex[] = "0123456789abcdef";
    for (std::size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
            else if (c < 0x20) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
            else { result += char(c); }
            ++i;
            continue;
        }
        std::size_t count = c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        bool valid = count != 0 && i + count <= text.size();
        std::uint32_t scalar = count == 2 ? c & 31u : count == 3 ? c & 15u : c & 7u;
        for (std::size_t j = 1; valid && j < count; ++j) {
            const auto tail = static_cast<unsigned char>(text[i + j]);
            valid = (tail & 0xc0) == 0x80;
            scalar = (scalar << 6) | (tail & 63u);
        }
        valid = valid && scalar >= (count == 2 ? 0x80u : count == 3 ? 0x800u : 0x10000u) &&
                scalar <= 0x10ffff && !(scalar >= 0xd800 && scalar <= 0xdfff);
        if (valid) { result.append(text.substr(i, count)); i += count; }
        else { result += "\\ufffd"; ++i; }
    }
    result += '"';
    return result;
}

const char * phase_name(stage value) {
    return value == stage::prompt ? "prompt" : "generation";
}

class backend_owner {
public:
    backend_owner() { ggml_backend_load_all(); llama_backend_init(); }
    ~backend_owner() { llama_backend_free(); }
    backend_owner(const backend_owner &) = delete;
    backend_owner & operator=(const backend_owner &) = delete;
};

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

class batch_owner {
public:
    explicit batch_owner(std::uint32_t capacity) : value(llama_batch_init(std::int32_t(capacity), 0, 1)) {
        // llama_batch_is_complete is a fork addition; all it asserted was that
        // the allocation produced every array. Check the arrays themselves, so
        // the test is against the real contract instead of a helper that only
        // the fork's header declared.
        if (!value.token || !value.pos || !value.n_seq_id || !value.seq_id || !value.logits) {
            llama_batch_free(value);
            value = {};
            throw std::bad_alloc();
        }
    }
    ~batch_owner() { llama_batch_free(value); }
    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;
    llama_batch value;
};

struct request {
    std::vector<llama_token> prompt;
    std::vector<llama_token> generated;
    std::string text;
    std::optional<tq::text::generation_stream> text_stream;
    std::optional<std::size_t> stop_index;
    sampler_ptr sampler{nullptr, llama_sampler_free};
    std::optional<tq::scheduling::reservation> reservation;
    std::optional<llama_token> pending;
    std::size_t prompt_done = 0;
    llama_pos position = 0;
    stage state = stage::idle;
    bool finished = false;
    bool eligible = false;
    std::string finish_reason;
};

struct row {
    std::uint32_t request_id;
    // The KV sequence this row is written to. Captured when the station admits
    // the request, so the ticket never reads request state to find it: the plan
    // is complete before the only blocking call begins.
    std::int32_t  slot;
    llama_pos position;
    llama_token token;
    bool logits;
};

struct sampled {
    std::uint32_t request_id;
    llama_token token;
    bool eog;
    bool finished;
    float logit_best;
    float logit_next;
    llama_token candidate_next;
    tq::text::stream_delta output;
};

// A committed decode instruction. The station decides it in one synchronous
// block -- admission, slot selection, row construction -- and records it here
// before anything blocks. It is a value: rows, positions, slots, output flags,
// nothing to look up later.
struct decode_plan {
    stage phase;
    std::vector<row> rows;
};

// The raw answer from the device. Deliberately not interpreted: the ticket
// reports what happened, the station decides what it means.
struct decode_outcome {
    int  code;         // llama_decode return value
    bool synchronized; // the device finished before the call returned
};

// The only place this program blocks. submit() fills the batch, submits it, and
// waits for the device; it reads no request state and writes none. A failure
// therefore leaves the station's books untouched, and reconcile() stays the
// single commit point -- which is what makes the pair safe to reason about
// without a lock.
class decode_ticket {
public:
    decode_ticket(llama_context * context, llama_batch & batch) : context_(context), batch_(batch) {}

    decode_outcome submit(const decode_plan & plan) const {
        if (plan.rows.empty() || plan.rows.size() > std::size_t(llama_n_batch(context_))) {
            throw std::logic_error("invalid planned batch size");
        }
        batch_.n_tokens = std::int32_t(plan.rows.size());
        for (std::size_t i = 0; i < plan.rows.size(); ++i) {
            batch_.token[i] = plan.rows[i].token;
            batch_.pos[i] = plan.rows[i].position;
            batch_.n_seq_id[i] = 1;
            batch_.seq_id[i][0] = llama_seq_id(plan.rows[i].slot);
            batch_.logits[i] = plan.rows[i].logits ? 1 : 0;
        }
        // The scheduler seams learn which context is executing from this scope.
        // It is opened at the only call site that can submit work, so the
        // association is a fact about this function, not an inference from a
        // struct layout upstream owns.
        const herd::phase::scope_guard phase_scope(context_);
        const int code = llama_decode(context_, batch_);
        if (code != 0) {
            return {code, false};
        }
        // Waiting belongs here and nowhere else: this is the one call that can
        // take milliseconds, and it must never happen while the station is
        // updating its own state.
        llama_synchronize(context_);
        return {code, true};
    }

private:
    llama_context * context_;
    llama_batch &   batch_;
};

// State and rulings. Every method below is an ordinary synchronous call: it
// mutates this object's state and returns. Nothing here waits, and nothing here
// prints-and-blocks. Per the ticket-station method the station is a rich
// aggregate -- admission policy, slot selection, sampling, release -- rather than
// an anaemic bag of fields, because all of that is pure memory work.
class station {
public:
    explicit station(options config) : config_(std::move(config)) {
        auto params = llama_model_default_params();
        params.n_gpu_layers = std::int32_t(config_.gpu_layers);
        model_.reset(llama_model_load_from_file(config_.model.c_str(), params));
        if (!model_) { throw std::runtime_error("A could not load the model"); }
        vocab_ = llama_model_get_vocab(model_.get());
        if (!vocab_ || llama_vocab_n_tokens(vocab_) <= 0) { throw std::runtime_error("A returned no vocabulary"); }
        load_prompts();

        active_limit_ = std::uint32_t(std::min(requests_.size(), std::size_t(config_.parallel ? config_.parallel : config_.batch)));
        std::vector<std::uint64_t> needs;
        for (const auto & item : requests_) {
            const auto cells = token_demand(item);
            if (cells > config_.context) {
                throw std::invalid_argument("a prompt plus requested generation exceeds --ctx-size");
            }
            needs.push_back(cells);
        }
        std::sort(needs.rbegin(), needs.rend());
        std::uint64_t needed = 0;
        for (std::size_t i = 0; i < active_limit_; ++i) {
            if (needs[i] > std::numeric_limits<std::uint64_t>::max() - needed) {
                throw std::invalid_argument("concurrent token demand overflow");
            }
            needed += needs[i];
        }
        const auto physical = config_.total_kv ? std::uint64_t(config_.total_kv) :
            std::max(std::uint64_t(config_.context), needed);
        // Auto-expansion, as the previous branch did it.
        //
        // The fork sized the cache as `total_kv ? total_kv : max(n_ctx, needed)` and
        // assigned it unconditionally (fork/batch-compose/src/main.cpp:252-259), so a
        // resident set larger than the window got a larger pool without being asked. A
        // guard once refused that instead; it read as prudence and was in fact a feature
        // removed -- a run the fork completed now stopped with "pass --total-kv".
        //
        // The complaint behind that guard was real: the operator cannot see the
        // expansion. The answer is to report it (`kv_auto_expanded` in the config event),
        // not to refuse work the previous branch did. This is the line the project draws
        // everywhere else -- a capability is not dropped to avoid explaining it.
        if (physical > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("required physical KV exceeds INT32_MAX");
        }
        auto context_params = llama_context_default_params();
        context_params.n_ctx = config_.context;
        // Decoupled capacity, when the resident set needs more cells than the logical
        // context provides. The mechanism is a scoped request around this one init call
        // (patches/herd_nkv.cpp): the cache is sized from a widened copy of cparams while
        // the context keeps its own, so the window every sequence may attend over does
        // not move -- only the number of cells available to hold S sequences.
        //
        // Verified afterwards rather than assumed: a request that the hook never saw, or
        // that the cache's shape refused, turns into an error here instead of a run that
        // silently used the smaller capacity it was told not to use.
        // The window upstream will actually give this context, predicted rather than
        // assumed: it pads the requested length up to a multiple of 256
        // (llama-context.cpp:456), so --ctx-size 64 yields 256 cells. Comparing a
        // capacity request against the REQUESTED figure was a real defect, measured:
        // `--ctx-size 64 --total-kv 128` opened a 128-cell request, the hook found 128
        // below the padded window of 256 and declined to widen anything, and the run
        // aborted with "this memory shape cannot be decoupled" -- a wrong diagnosis of a
        // request that needed no widening at all, because 256 cells already covers 128.
        const std::uint64_t window = (std::uint64_t(config_.context) + 255u) & ~std::uint64_t(255u);
        // Whether the pool ended up larger than the window without the operator asking.
        // Reported in the config event, because the expansion is a real consequence of
        // sizing a cache for the resident set and a reader should not have to infer it
        // from two numbers.
        const bool kv_auto_expanded = config_.total_kv == 0 && physical > window;

        std::unique_ptr<herd::nkv::scope> decoupled;
        if (physical > window) {
            decoupled = std::make_unique<herd::nkv::scope>(static_cast<std::uint32_t>(physical));
        }
        context_params.kv_unified = true;
        context_params.n_seq_max = active_limit_;
        context_params.n_batch = config_.batch;
        context_params.n_ubatch = config_.ubatch;
        context_params.n_outputs_max = active_limit_;
        context_params.n_threads = std::int32_t(config_.threads);
        context_params.n_threads_batch = std::int32_t(config_.threads);
        context_.reset(llama_init_from_model(model_.get(), context_params));
        if (!context_) { throw std::runtime_error("A rejected the context configuration"); }
        // The ONE place this fact is read, on the object it was chosen on.
        //
        // --phase-memory's release re-arms the scheduler by toggling llama_set_embeddings()
        // twice; the toggle restores the value only if the argument is the context's actual
        // setting, and upstream has no getter to consult later. The value is therefore
        // captured here, where it was chosen, and handed on unchanged. Hard-coding `false` at
        // the release site was correct only because this program never sets the field -- a
        // coincidence that a future --embeddings flag, or any second caller, would break
        // silently: the context would come out of the release with embeddings inverted and
        // every later llama_get_embeddings() would throw.
        embeddings_enabled_ = context_params.embeddings;
        // The request must have taken effect. Both failure modes are named, because they
        // mean different things and the operator can act on either:
        //   not consumed -> the hook never ran for this init call (a build without the
        //                   nkv module, or a symbol that moved)
        //   not applied  -> the cache refused the shape (it is not a shape that can be
        //                   decoupled), so the capacity above is a request, not a fact
        bool          decoupled_applied = false;
        // The cells the cache actually has when the hook applied, which is the request
        // padded up to 256 (the hook keeps the fork's alignment). Reporting or admitting
        // on the raw request would name a number no buffer holds.
        std::uint64_t decoupled_cells   = 0;
        if (decoupled) {
            decoupled->close();
            // One scope, one context -- enforced here, because this is the layer that knows
            // how many contexts it created. The hook already refused to widen a second one
            // and said so; the count is what makes "did that happen" a fact the driver can
            // act on instead of a line in a log nobody reads.
            if (decoupled->consumptions() != 1) {
                throw std::runtime_error(
                    "the decoupled-KV request was consumed by " +
                    std::to_string(decoupled->consumptions()) +
                    " contexts; one scope belongs to one init call. The capacity was applied "
                    "to the first context only");
            }
            if (!decoupled->consumed()) {
                throw std::runtime_error(
                    "decoupled KV was requested for " + std::to_string(physical) + " cells but the nkv "
                    "hook never ran during context creation; this build cannot honour --total-kv above "
                    "the context's window");
            }
            if (!decoupled->applied()) {
                // The reason decides the message. "Declined" alone sends the operator to
                // the wrong place: a recurrent model, a split cache and an out-of-range
                // number have three different fixes, and the hook knows which one it hit.
                throw std::runtime_error(
                    "--total-kv " + std::to_string(physical) + " was declined: " +
                    herd::nkv::name(decoupled->reason()) + " (the context kept " +
                    std::to_string(decoupled->applied_cells()) + " cells)");
            }
            decoupled_applied = true;
            decoupled_cells   = decoupled->applied_cells();
        }
        // The request is scoped to the init call and is now spent: leaving it open would
        // attach it to the NEXT context created on this thread, which is the same
        // leak-across-contexts defect that made the environment-variable version wrong.
        decoupled.reset();

        // Two numbers, and they are not interchangeable.
        //
        //   window_actual  what llama_n_ctx_seq() reports: the window ONE sequence may
        //                  attend over. Decoupling never changes it -- that is the point.
        //   pool_capacity  the cells that exist: the widened cache when the hook applied,
        //                  otherwise the same window.
        //
        // A request at or below the pool is honoured, whether or not the hook ran: 256
        // padded cells already cover a request of 128, and calling that a refusal was the
        // defect this check replaces. A request above it means the pool was not widened,
        // and that is a failure worth naming with both numbers.
        const std::uint64_t window_actual = std::uint64_t(llama_n_ctx_seq(context_.get()));
        const std::uint64_t pool_capacity = decoupled_applied ? decoupled_cells : window_actual;
        if (physical > pool_capacity) {
            throw std::runtime_error(
                "--total-kv " + std::to_string(physical) + " asks for more cells than this context holds: " +
                std::to_string(pool_capacity) + " (window " + std::to_string(window_actual) + ", no widening "
                "applied). Omit --total-kv, or ask for at most the window");
        }
        if (!llama_get_memory(context_.get())) { throw std::runtime_error("this runner requires a decoder model with sequence memory"); }
        if (llama_n_batch(context_.get()) < active_limit_ || llama_n_seq_max(context_.get()) < active_limit_) {
            throw std::runtime_error("A cannot hold the configured resident request count");
        }
        // Capacity for admission is the KV the context actually has, not the
        // number requested: upstream pads n_ctx_seq up to a multiple of 256, so
        // --ctx-size 64 yields 256 cells. Using the requested figure would
        // under-fill real memory; using the actual one is why the config event
        // reports both.
        //
        // With a decoupled request the cells available are the physical ones, and the
        // per-sequence window is still llama_n_ctx_seq(): upstream's unified cache lets
        // each of S sequences grow to the window while sharing one pool, so admitting on
        // the window alone would promise less room than exists, and admitting on the
        // physical figure alone would promise each sequence more than it may use.
        // Capacity for admission is the pool: the widened cache when the hook applied, and
        // the window otherwise. `window_actual` still bounds what one sequence may attend
        // over, and reporting the requested figure instead of the pool was the other half
        // of the same defect -- --total-kv 128 under --ctx-size 64 would have admitted
        // against 128 while the context held 256.
        const std::uint64_t admitted = pool_capacity;
        admission_ = std::make_unique<tq::scheduling::admission_budget>(admitted, active_limit_);
        if (config_.inject_after && (active_limit_ < 2 ||
            token_demand(requests_[0]) + token_demand(requests_[1]) > admission_->capacity())) {
            throw std::invalid_argument("--inject-after requires capacity for both initial requests");
        }
        batch_.reset(new batch_owner(llama_n_batch(context_.get())));
        candidates_.resize(std::size_t(llama_vocab_n_tokens(vocab_)));
        if (config_.json) {
            std::cout << "{\"event\":\"config\",\"requests\":" << requests_.size()
                      << ",\"logical_context_requested\":" << config_.context
                      << ",\"logical_context_actual\":" << llama_n_ctx_seq(context_.get())
                      << ",\"total_kv_requested\":" << physical
                      // What the context actually holds, not what was asked for. With a
                      // widened cache that is the widened size; without one it is the
                      // window, which is why `--total-kv 128` under `--ctx-size 64`
                      // reports 256 and `kv_decoupled: false` -- the request was honoured
                      // by padding, and saying otherwise would misreport a legal run.
                      << ",\"total_kv_actual\":" << pool_capacity
                      << ",\"kv_decoupled\":" << (decoupled_applied ? "true" : "false")
                      // True when the cache is larger than the window and `--total-kv` was
                      // not given: the runner sized the pool for its resident set, as the
                      // previous branch did. Reported rather than refused, and reported
                      // rather than left implicit in the two numbers beside it.
                      << ",\"kv_auto_expanded\":" << (kv_auto_expanded ? "true" : "false")
                      << ",\"batch_actual\":" << llama_n_batch(context_.get())
                      << ",\"parallel_actual\":" << active_limit_
                      << ",\"gpu_layers_requested\":" << config_.gpu_layers
                      << ",\"phase_memory\":"
                      << (config_.phase_memory ? "true" : "false")
                      << ",\"stream\":" << (config_.stream ? "true" : "false") << ",\"stops\":[";
            for (std::size_t i = 0; i < config_.stops.size(); ++i) {
                if (i) { std::cout << ','; }
                std::cout << quote(config_.stops[i]);
            }
            std::cout << "]}\n";
            if (config_.stream) { flush_output(); }
        }
    }

    // Event entry: everything the run loop needs before work can be submitted,
    // decided synchronously and returned as a value. Returning nullopt means the
    // station is done with every request; it does not mean "wait".
    //
    // Compute (inject_if_ready / admit_ready / selector) then Commit (change_phase
    // records the transition, rows capture slots and positions) and only then
    // return. The order is what the method calls for: nothing outside this
    // function can observe the state between those steps, and by the time it
    // returns the plan is complete, so the ticket has nothing left to decide.
    std::optional<decode_plan> plan_next() {
        inject_if_ready();
        admit_ready();
        std::vector<tq::scheduling::slot_state> snapshot;
        for (std::size_t i = 0; i < requests_.size(); ++i) {
            if (requests_[i].reservation) {
                snapshot.push_back({std::uint32_t(i), requests_[i].state});
            }
        }
        const auto selected = selector_.choose(snapshot);
        if (!selected.prompt_slot && selected.generation_slots.empty()) {
            if (config_.inject_after && !injected_) {
                throw std::runtime_error("initial requests stopped before injection; reduce --inject-after or use --ignore-eog");
            }
            return std::nullopt;
        }
        change_phase(selected.next);
        decode_plan plan{selected.next, {}};
        if (selected.next == stage::prompt) {
            auto & item = requests_.at(*selected.prompt_slot);
            const auto count = std::min(std::size_t(llama_n_batch(context_.get())), item.prompt.size() - item.prompt_done);
            for (std::size_t i = 0; i < count; ++i) {
                plan.rows.push_back({*selected.prompt_slot, std::int32_t(item.reservation->slot()),
                                     item.position + llama_pos(i), item.prompt[item.prompt_done + i],
                                     item.prompt_done + i + 1 == item.prompt.size()});
            }
        } else {
            for (const auto id : selected.generation_slots) {
                auto & item = requests_.at(id);
                if (!item.pending) { throw std::logic_error("generation has no sampled input token"); }
                plan.rows.push_back({id, std::int32_t(item.reservation->slot()), item.position, *item.pending, true});
            }
        }
        return plan;
    }

    // End of the run: the accounting assertions and the per-request report. Split
    // from plan_next so the loop ticket has one call to make when it stops.
    void finish() {
        llama_synchronize(context_.get());
        if (admission_->active_count() != 0 || admission_->reserved_tokens() != 0) {
            throw std::logic_error("completed requests retained an admission reservation");
        }
        for (std::size_t id = 0; id < requests_.size(); ++id) {
            const auto & item = requests_[id];
            if (!item.finished) { throw std::logic_error("request did not finish"); }
            if (config_.json) {
                std::cout << "{\"event\":\"result\",\"request_id\":" << id
                          << ",\"prompt_tokens\":" << item.prompt.size()
                          << ",\"decoded_rows\":" << item.position
                          << ",\"finish_reason\":" << quote(item.finish_reason) << ",\"tokens\":[";
                for (std::size_t i = 0; i < item.generated.size(); ++i) {
                    if (i) { std::cout << ','; }
                    std::cout << item.generated[i];
                }
                std::cout << "],\"text\":" << quote(item.text) << ",\"stop_index\":";
                if (item.stop_index) { std::cout << *item.stop_index; }
                else { std::cout << "null"; }
                std::cout << "}\n";
            } else {
                std::cout << "[request " << id << "]\n" << item.text << "\n\n";
            }
        }
        if (!std::cout) { throw std::runtime_error("could not write output"); }
    }

private:
    std::uint64_t token_demand(const request & item) const {
        return std::uint64_t(item.prompt.size()) + config_.predict - 1;
    }

    void admit_ready() {
        for (std::size_t id = 0; id < requests_.size(); ++id) {
            auto & item = requests_[id];
            if (!item.eligible || item.finished || item.reservation) { continue; }
            auto held = admission_->try_acquire(id, token_demand(item));
            if (!held) {
                if (config_.json) {
                    std::cout << "{\"event\":\"wait\",\"request_id\":" << id
                              << ",\"tokens_required\":" << token_demand(item)
                              << ",\"tokens_available\":" << admission_->available_tokens()
                              << ",\"active_requests\":" << admission_->active_count() << "}\n";
                }
                break;
            }
            try {
                item.text_stream.emplace(config_.stops);
                item.sampler.reset(llama_sampler_init_greedy());
                if (!item.sampler) { throw std::runtime_error("A could not create a greedy sampler"); }
            } catch (...) {
                item.text_stream.reset();
                item.sampler.reset();
                admission_->release(*held);
                throw;
            }
            item.reservation = std::move(held);
            item.state = stage::prompt;
            if (config_.json) {
                std::cout << "{\"event\":\"admit\",\"request_id\":" << id
                          << ",\"sequence_id\":" << item.reservation->slot()
                          << ",\"generation\":" << item.reservation->generation()
                          << ",\"tokens_reserved\":" << item.reservation->tokens()
                          << ",\"total_reserved\":" << admission_->reserved_tokens() << "}\n";
            }
        }
    }

    void load_prompts() {
        std::ifstream file;
        std::istream * input = &std::cin;
        if (config_.prompts != "-") {
            file.open(config_.prompts);
            if (!file) { throw std::runtime_error("cannot open prompt file: " + config_.prompts); }
            input = &file;
        }
        std::string line;
        while (std::getline(*input, line)) {
            if (!line.empty() && line.back() == '\r') { line.pop_back(); }
            if (line.size() > std::size_t(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("prompt is too large to tokenize");
            }
            request item;
            const auto required = llama_tokenize(vocab_, line.data(), std::int32_t(line.size()), nullptr, 0, true, false);
            if (required == std::numeric_limits<std::int32_t>::min()) { throw std::runtime_error("token count overflow"); }
            item.prompt.resize(std::size_t(required < 0 ? -required : required));
            const auto count = llama_tokenize(vocab_, line.data(), std::int32_t(line.size()), item.prompt.data(),
                                              std::int32_t(item.prompt.size()), true, false);
            if (count <= 0) { throw std::runtime_error("A could not tokenize a nonempty sequence"); }
            item.prompt.resize(std::size_t(count));
            item.eligible = !config_.inject_after || requests_.size() < 2;
            if (requests_.size() >= std::size_t(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("request count exceeds INT32_MAX");
            }
            requests_.push_back(std::move(item));
        }
        if (input->bad()) { throw std::runtime_error("could not read prompts"); }
        if (requests_.empty()) { throw std::invalid_argument("the prompt input has no lines"); }
        if (config_.inject_after && requests_.size() < 3) { throw std::invalid_argument("--inject-after needs at least three prompts"); }
    }

    void inject_if_ready() {
        if (!config_.inject_after || injected_ || generation_batches_ < *config_.inject_after) { return; }
        if (requests_[0].state != stage::generation || requests_[1].state != stage::generation) {
            throw std::runtime_error("two initial requests must still be generating at injection; use --ignore-eog for fixed-length runs");
        }
        for (std::size_t id = 2; id < requests_.size(); ++id) { requests_[id].eligible = true; }
        injected_ = true;
        if (config_.json) {
            std::cout << "{\"event\":\"inject\",\"after_generation_batches\":" << generation_batches_ << ",\"request_ids\":[";
            for (std::size_t id = 2; id < requests_.size(); ++id) {
                if (id > 2) { std::cout << ','; }
                std::cout << id;
            }
            std::cout << "]}\n";
        }
    }

    void change_phase(stage next) {
        if (phase_ && *phase_ == next) { return; }
        llama_synchronize(context_.get());

        // The release happens HERE, on the boundary the fork released on, and only at the
        // one transition that asked for it: prompt -> generate. Doing it on every phase
        // change would free buffers the next decode immediately needs, which costs a
        // full re-reservation per turn for no memory saving.
        const bool requested = config_.phase_memory && next == stage::generation;
        std::uint64_t before = 0;
        std::uint64_t after  = 0;
        const bool known_before = herd::phase::compute_bytes(context_.get(), &before);

        bool released = false;
        const char * refusal = nullptr;
        if (requested) {
            // The value the context was CREATED with, recorded at creation (line ~425) rather
            // than restated here. upstream has no getter for it, so a literal at this site is
            // an assertion about the constructor -- and an assertion that can drift the moment
            // the constructor changes. `false` happened to be right because nothing sets the
            // field; `embeddings_enabled_` is right because it is the field.
            released = herd::phase::release_compute(context_.get(), embeddings_enabled_);
            if (!released) { refusal = herd::phase::release_refusal_reason(); }
        }
        const bool known_after = herd::phase::compute_bytes(context_.get(), &after);

        if (config_.json) {
            // Every field is measured or explicitly null. `release_compute_success` is
            // null when nothing was requested, false when the request was refused (with a
            // cause on stderr), and true only when buffers were actually returned.
            std::cout << "{\"event\":\"phase\",\"from\":" << (phase_ ? quote(phase_name(*phase_)) : "null")
                      << ",\"to\":" << quote(phase_name(next)) << ",\"synchronized\":true"
                      << ",\"release_compute_requested\":" << (requested ? "true" : "false")
                      << ",\"release_compute_success\":"
                      << (requested ? (released ? "true" : "false") : "null")
                      << ",\"release_available\":" << (herd::phase::release_available() ? "true" : "false")
                      << ",\"release_experiment_verified\":"
                      << (herd::phase::release_experiment_verified() ? "true" : "false")
                      << ",\"compute_bytes_before\":";
            if (known_before) { std::cout << before; } else { std::cout << "null"; }
            std::cout << ",\"compute_bytes_after\":";
            if (known_after) { std::cout << after; } else { std::cout << "null"; }
            std::cout << "}\n";
        }
        if (requested && !released) {
            // A refusal under an explicit request is a failed run, not a warning.
            //
            // --phase-memory is a memory contract: the operator asked for the compute
            // buffers to be returned between prompt and generation. Printing a line and
            // carrying on means every downstream number comes from a run that kept the
            // buffers, while the exit status and the JSON config event both say the
            // bounded mode was selected. That is the silent degradation this project has
            // spent its whole budget refusing, and a one-line stderr is not a substitute
            // for a non-zero exit.
            //
            // The JSON phase event above is already emitted, so the cause survives for a
            // consumer that reads the stream; the throw turns it into a failed process.
            throw std::runtime_error(
                std::string("--phase-memory was requested and the release at ") +
                (phase_ ? phase_name(*phase_) : "start") + "->" + phase_name(next) +
                " was refused: " + (refusal ? refusal : "release_compute returned false"));
        }
        phase_ = next;
    }

    std::string piece(llama_token token) const {
        char buffer[128];
        const auto count = llama_token_to_piece(vocab_, token, buffer, sizeof(buffer), 0, false);
        if (count >= 0) { return std::string(buffer, std::size_t(count)); }
        if (count == std::numeric_limits<std::int32_t>::min()) { throw std::runtime_error("token piece size overflow"); }
        std::string result(std::size_t(-count), '\0');
        const auto actual = llama_token_to_piece(vocab_, token, result.data(), std::int32_t(result.size()), 0, false);
        if (actual < 0 || std::size_t(actual) > result.size()) { throw std::runtime_error("A could not detokenize a sample"); }
        result.resize(std::size_t(actual));
        return result;
    }

    sampled sample(std::uint32_t id, std::int32_t batch_row) {
        auto & item = requests_.at(id);
        if (!item.text_stream) { throw std::logic_error("resident request has no text stream"); }
        const auto * logits = llama_get_logits_ith(context_.get(), batch_row);
        if (!logits) { throw std::runtime_error("A returned no logits for a requested output row"); }
        float best = -std::numeric_limits<float>::infinity();
        float next = best;
        llama_token best_id = LLAMA_TOKEN_NULL;
        llama_token next_id = LLAMA_TOKEN_NULL;
        for (std::size_t i = 0; i < candidates_.size(); ++i) {
            if (std::isnan(logits[i]) || logits[i] == std::numeric_limits<float>::infinity()) {
                throw std::runtime_error("A returned invalid logits");
            }
            if (logits[i] > best) {
                next = best;
                next_id = best_id;
                best = logits[i];
                best_id = llama_token(i);
            } else if (logits[i] > next) {
                next = logits[i];
                next_id = llama_token(i);
            }
            candidates_[i] = {llama_token(i), logits[i], 0.0f};
        }
        if (best_id == LLAMA_TOKEN_NULL) { throw std::runtime_error("A returned no finite logit candidate"); }
        llama_token_data_array view{candidates_.data(), candidates_.size(), -1, false};
        llama_sampler_apply(item.sampler.get(), &view);
        if (view.selected < 0 || std::size_t(view.selected) >= view.size) {
            throw std::runtime_error("A greedy sampler returned no selected token");
        }
        const auto token = view.data[view.selected].id;
        llama_sampler_accept(item.sampler.get(), token);
        const bool eog = llama_vocab_is_eog(vocab_, token);
        tq::text::stream_delta output;
        if (eog && !config_.ignore_eog) {
            output = item.text_stream->finish();
            item.finish_reason = output.stop_index ? "stop" : "eog";
            item.finished = true;
        } else {
            item.generated.push_back(token);
            output = item.text_stream->push(piece(token));
            if (!output.finished && item.generated.size() >= config_.predict) {
                auto tail = item.text_stream->finish();
                output.text += tail.text;
                output.finished = tail.finished;
                output.stop_index = tail.stop_index;
            }
            if (output.finished) {
                item.finish_reason = output.stop_index ? "stop" : "length";
                item.finished = true;
            } else {
                item.pending = token;
            }
        }
        item.text += output.text;
        item.stop_index = output.stop_index;
        item.state = item.finished ? stage::idle : stage::generation;
        return {id, token, eog, item.finished, best, next, next_id, std::move(output)};
    }

    static void flush_output() {
        std::cout.flush();
        if (!std::cout) { throw std::runtime_error("could not write output"); }
    }

    void publish_text(const sampled & sample) const {
        if (!config_.stream || (sample.output.text.empty() && !sample.output.finished)) { return; }
        const auto & item = requests_.at(sample.request_id);
        std::cout << "{\"event\":\"text\",\"request_id\":" << sample.request_id
                  << ",\"offset_bytes\":" << sample.output.offset
                  << ",\"text\":" << quote(sample.output.text)
                  << ",\"generated_tokens\":" << item.generated.size()
                  << ",\"finished\":" << (sample.output.finished ? "true" : "false")
                  << ",\"finish_reason\":" << (sample.output.finished ? quote(item.finish_reason) : "null")
                  << ",\"stop_index\":";
        if (sample.output.stop_index) { std::cout << *sample.output.stop_index; }
        else { std::cout << "null"; }
        std::cout << "}\n";
        flush_output();
    }

    void trace_batch(stage phase, const std::vector<row> & rows, int returned, const std::vector<sampled> & samples) const {
        if (!config_.json) { return; }
        std::cout << "{\"event\":\"batch\",\"step\":" << step_ << ",\"phase\":" << quote(phase_name(phase))
                  << ",\"decode_return\":" << returned
                  << ",\"compute_bytes\":";
        std::uint64_t bytes = 0;
        if (herd::phase::compute_bytes(context_.get(), &bytes)) { std::cout << bytes; }
        else { std::cout << "null"; }
        std::cout << ",\"rows\":[";
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (i) { std::cout << ','; }
            const auto & row = rows[i];
            std::cout << "{\"request_id\":" << row.request_id << ",\"position\":" << row.position
                      << ",\"sequence_id\":" << row.slot
                      << ",\"token\":" << row.token << ",\"logits\":" << (row.logits ? "true" : "false") << '}';
        }
        std::cout << "],\"samples\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) {
            if (i) { std::cout << ','; }
            const auto & sample = samples[i];
            std::cout << "{\"request_id\":" << sample.request_id << ",\"token\":" << sample.token
                      << ",\"eog\":" << (sample.eog ? "true" : "false")
                      << ",\"finished\":" << (sample.finished ? "true" : "false")
                      << ",\"logit_best\":" << std::setprecision(std::numeric_limits<float>::max_digits10) << sample.logit_best
                      << ",\"logit_next\":";
            if (std::isfinite(sample.logit_next)) { std::cout << sample.logit_next; }
            else { std::cout << "null"; }
            std::cout << ",\"candidate_next\":" << sample.candidate_next << '}';
        }
        std::cout << "]}\n";
    }

public:

    // The reconciliation half of the sandwich: the ticket brings back the raw
    // device answer and the plan it was given, and this method -- synchronous,
    // the only writer of request state -- decides what it means. Positions move,
    // samples are taken, output is published, finished requests are released.
    void reconcile(const decode_plan & plan, const decode_outcome & outcome) {
        const auto & rows  = plan.rows;
        const auto   phase = plan.phase;
        const int    returned = outcome.code;
        if (returned != 0) {
            trace_batch(phase, rows, returned, {});
            throw std::runtime_error("A llama_decode failed at batch " + std::to_string(step_) + " with code " + std::to_string(returned));
        }
        // Reading logits requires a finished device. The ticket guarantees that
        // when it reports code 0; asserting it keeps the contract from decaying
        // into an assumption if the ticket ever gains an early-return path.
        if (!outcome.synchronized) {
            throw std::logic_error("decode reported success without waiting for the device");
        }
        std::vector<sampled> samples;
        try {
            // Finish A before changing our committed positions or freeing a sequence.
            llama_synchronize(context_.get());
            for (std::size_t i = 0; i < rows.size(); ++i) {
                auto & item = requests_.at(rows[i].request_id);
                ++item.position;
                if (phase == stage::prompt) { ++item.prompt_done; }
                else { item.pending.reset(); }
                if (rows[i].logits) { samples.push_back(sample(rows[i].request_id, std::int32_t(i))); }
            }
        } catch (...) {
            // A already returned success. Keep that fact even when C cannot
            // consume all of its outputs; the next event is a fatal error.
            trace_batch(phase, rows, returned, samples);
            throw;
        }
        trace_batch(phase, rows, returned, samples);
        for (const auto & sample : samples) { publish_text(sample); }
        ++step_;
        // All output rows have been consumed before any memory mutation.
        for (const auto & sample : samples) {
            if (!sample.finished) { continue; }
            auto & item = requests_.at(sample.request_id);
            const auto held = *item.reservation;
            if (!llama_memory_seq_rm(llama_get_memory(context_.get()), llama_seq_id(held.slot()), 0, -1)) {
                throw std::runtime_error("A could not release a completed sequence");
            }
            if (!admission_->release(held)) { throw std::logic_error("completed reservation was not active"); }
            item.reservation.reset();
            item.sampler.reset();
            item.text_stream.reset();
            if (config_.json) {
                std::cout << "{\"event\":\"release\",\"request_id\":" << sample.request_id
                          << ",\"sequence_id\":" << held.slot() << ",\"generation\":" << held.generation()
                          << ",\"total_reserved\":" << admission_->reserved_tokens() << "}\n";
            }
        }
        if (phase == stage::generation) { ++generation_batches_; }
    }

    // Capability hand-out for the wiring layer: the ticket needs these two and
    // nothing else. They are not setters and carry no policy -- the station is
    // not delegating a decision, it is being connected.
    llama_context * context() const { return context_.get(); }
    llama_batch &   batch()   const { return batch_->value; }

private:
    options config_;
    // What llama_context_params::embeddings was when this station created its context.
    //
    // Held because the release needs it and upstream cannot be asked (no getter), and because
    // a value captured where it was chosen cannot disagree with the context it describes. See
    // the assignment in the constructor and the release at change_phase().
    bool embeddings_enabled_ = false;
    backend_owner backend_;
    model_ptr model_{nullptr, llama_model_free};
    context_ptr context_{nullptr, llama_free};
    const llama_vocab * vocab_ = nullptr;
    std::unique_ptr<batch_owner> batch_;
    std::vector<request> requests_;
    std::vector<llama_token_data> candidates_;
    tq::scheduling::prefill_selector selector_;
    std::unique_ptr<tq::scheduling::admission_budget> admission_;
    std::uint32_t active_limit_ = 0;
    std::optional<stage> phase_;
    std::uint64_t step_ = 0;
    std::uint64_t generation_batches_ = 0;
    bool injected_ = false;
};

// The run loop, as a ticket: no domain state, only the two capabilities it was
// assembled with. Each iteration asks the station for one plan, blocks once in
// the ticket, and hands the raw answer back. No branch in here decides anything
// about a request -- if one appeared, it would belong in the station.
class run_ticket {
public:
    run_ticket(station & app, decode_ticket ticket) : app_(app), ticket_(std::move(ticket)) {}

    void execute() {
        while (const auto plan = app_.plan_next()) {
            const auto outcome = ticket_.submit(*plan);
            app_.reconcile(*plan, outcome);
        }
        app_.finish();
    }

private:
    station &     app_;
    decode_ticket ticket_;
};

} // namespace

int main(int argc, char ** argv) {
    bool json = false;
    try {
        // Install the scheduler-capture seams before anything can decode. A
        // failure aborts inside install(): a binary that silently ran without
        // them would report compute_bytes:null forever and look like a model
        // that keeps no compute buffers.
        herd::phase::install();
        auto config = parse(argc, argv, json);
        if (config.help) { usage(); return std::cout ? 0 : 1; }
        station app(std::move(config));
        // Wiring: build the station, hand the ticket its two capabilities, run.
        // Nothing below inspects a request, a stage, or a tool name.
        run_ticket runner(app, decode_ticket(app.context(), app.batch()));
        runner.execute();
        return 0;
    } catch (const std::exception & error) {
        if (json) { std::cout << "{\"event\":\"error\",\"message\":" << quote(error.what()) << "}\n"; }
        std::cerr << "tq-batch-compose: " << error.what() << '\n';
        return 1;
    }
}
