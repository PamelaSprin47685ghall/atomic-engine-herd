// The prefill publisher, driven by upstream's real `llm_graph_result`: all T tokens published in one
// `set_inputs`, the numbers compared against a per-(token, head) reference, and both refusal paths
// observed as process deaths.
//
// WHY THE FAILURE CASES ARE CHILD PROCESSES
// -----------------------------------------
// The refusal is `std::abort()`, and that is deliberate rather than lazy: `set_input` sits under
// `llama_process` (extern "C") -> `decode` -> `process_ubatch` -> `set_inputs`, and no frame on that
// chain catches anything, so a thrown exception would unwind across a C ABI. A process cannot catch an
// abort; a test can only fork and look at how the child died. `expect_abort` does that with
// `fork`/`execv`/`waitpid` rather than `std::system`, because a shell reports a signal death as an
// exit status of 128+n -- `WIFSIGNALED` would be false and the assertion would be checking the shell.
//
// WHAT THE ACCEPTING PATH ASSERTS, AND WHY EACH ONE EARNED ITS PLACE
// ------------------------------------------------------------------
//   * `per_token.size() == T` and every emission has `spans + 1` entries: the composition is intact,
//     not silently short a token.
//   * every index tensor has a buffer after placement: the scheduler placed them because `get_rows`
//     consumes them. A tensor no node reads stays unplaced, and that is the failure the single-token
//     publisher once hit on layer 0.
//   * after `set_inputs`: `passes == 1`, `tokens_published == T`, `contract_violation == false`.
//   * the published cells, read back out of the TENSORS per token, equal that token's view order. A
//     publication that wrote token 2's table into token 1's tensors would pass every count above.
//   * `out` matches a double reference written per (token, head) from the definition.
//   * `reset()` destroys the input and the statistics remain readable -- the shared-ownership point,
//     which a raw pointer would have failed as a write to freed memory.
#include "../patches/herd_reader_prefill_publisher.hpp"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "llama-graph.h"

#include <tq/reader/graph_emit_physical.h>
#include <tq/reader/graph_emit_prefill.h>
#include <tq/reader/upstream_view.h>

#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

constexpr int kHeadDim = 8;
constexpr int kKvHeads = 2;
constexpr int kHeads   = 4;   // group = 2
constexpr int kCell    = 16;
constexpr int kTokens  = 3;
constexpr float kScale = 0.35355339f;

struct fixture {
    std::vector<float> k, v, q, mask;

    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kTokens * kHeads, 0.0f);
        mask.assign(std::size_t(kCell) * kTokens, 0.0f);
        unsigned s = 0x9e3779b9u;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
        // Tokens must differ, or a shared query and a broadcast mask would hide both.
        for (int t = 0; t < kTokens; ++t)
            for (int p = 0; p < kCell; ++p)
                mask[std::size_t(p) + std::size_t(kCell) * t] =
                    (p < 6 + 3 * t) ? 0.25f * float(t) : -INFINITY;
    }

    float K(int d, int kvh, int c) const { return k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float V(int d, int kvh, int c) const { return v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float Q(int d, int t, int h) const { return q[std::size_t(d) + kHeadDim * (t + kTokens * h)]; }
    float M(int p, int t) const { return mask[std::size_t(p) + std::size_t(kCell) * t]; }
};

// The definition, per (token, head): one softmax over the whole cache, `-inf` hidden, finite bias
// added, kv head = h / (n_head / n_head_kv). Written from that sentence, not from the graph.
void reference(const fixture & f, std::vector<float> & out) {
    const int group = kHeads / kKvHeads;
    out.assign(std::size_t(kHeadDim) * kTokens * kHeads, 0.0f);
    for (int t = 0; t < kTokens; ++t) {
        for (int h = 0; h < kHeads; ++h) {
            const int kvh = h / group;
            std::vector<double> score(static_cast<std::size_t>(kCell), 0.0);
            double max_s = -INFINITY;
            for (int p = 0; p < kCell; ++p) {
                const double b = double(f.M(p, t));
                if (!std::isfinite(b)) { score[std::size_t(p)] = -INFINITY; continue; }
                double s = 0.0;
                for (int d = 0; d < kHeadDim; ++d)
                    s += double(f.Q(d, t, h)) * double(f.K(d, kvh, p));
                score[std::size_t(p)] = s * double(kScale) + b;
                if (score[std::size_t(p)] > max_s) max_s = score[std::size_t(p)];
            }
            double sum = 0.0;
            for (int p = 0; p < kCell; ++p) {
                if (!std::isfinite(score[std::size_t(p)])) continue;
                score[std::size_t(p)] = std::exp(score[std::size_t(p)] - max_s);
                sum += score[std::size_t(p)];
            }
            for (int d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (int p = 0; p < kCell; ++p) {
                    if (!std::isfinite(score[std::size_t(p)]) || score[std::size_t(p)] == 0.0) continue;
                    acc += (score[std::size_t(p)] / sum) * double(f.V(d, kvh, p));
                }
                out[std::size_t(d) + std::size_t(kHeadDim) *
                                         (std::size_t(t) + std::size_t(kTokens) * std::size_t(h))] = float(acc);
            }
        }
    }
}

tq::reader::query_view full_view() {
    tq::reader::query_view view;
    tq::reader::span_keys span;
    span.effective_pos = 5;
    for (int c = 0; c < kCell; ++c) span.cells.push_back(std::int32_t(c));
    view.spans.push_back(std::move(span));
    return view;
}

// Everything a host needs, in construction order. `stats` is declared BEFORE `res` on purpose: a
// `unique_ptr` inside `res` destroys the publisher, whose destructor writes `stats->destroyed`, and
// reverse-order destruction would then write into freed memory. It is a `shared_ptr`, so the order is
// not a hazard here -- but the comment stays because the ordering mistake was made once already in
// this tree.
struct host {
    herd::reader::prefill_publisher_stats_ptr stats =
        std::make_shared<herd::reader::prefill_publisher_stats>();
    llm_graph_result res{4096};
    ggml_context * ctx = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * q = nullptr;
    ggml_tensor * m = nullptr;
    tq::reader::prefill_result em{};
    herd::reader::prefill_publisher * raw = nullptr;
    ggml_backend_t be = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool built = false;

    host() {
        ctx = res.get_ctx();
        if (ctx == nullptr || !ggml_get_no_alloc(ctx)) return;

        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kTokens, kHeads);
        m = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kCell, kTokens, 1, 1);
        if (k == nullptr || v == nullptr || q == nullptr || m == nullptr) return;
        ggml_set_input(k); ggml_set_input(v); ggml_set_input(q); ggml_set_input(m);

        tq::reader::prefill_inputs in;
        in.ctx = ctx;
        in.k = k; in.v = v; in.q = q; in.mask = m;
        in.view = full_view();
        in.coverage = tq::reader::graph_inputs::mask_coverage::intersect;
        in.n_head = std::size_t(kHeads);
        in.n_head_kv = std::size_t(kKvHeads);
        in.head_dim = std::size_t(kHeadDim);
        in.n_cell = std::size_t(kCell);

        em = tq::reader::emit_prefill_attention(in, kScale);
        if (em.refusal != tq::reader::prefill_refusal::ok) return;
        if (em.per_token.size() != std::size_t(kTokens) || em.out == nullptr) return;

        raw = static_cast<herd::reader::prefill_publisher *>(
            res.add_input(std::make_unique<herd::reader::prefill_publisher>(em, stats)));
        if (raw == nullptr || res.inputs.size() != 1) return;

        ggml_build_forward_expand(res.get_gf(), em.out);
        be = ggml_backend_cpu_init();
        if (be == nullptr) return;
        sched = ggml_backend_sched_new(&be, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
        built = sched != nullptr;
    }

    ~host() { release_scheduler(); }

    // Frees the scheduler before anything destroys the context its tensors live in, and NULLS the
    // handles so a later explicit call cannot double-free.
    void release_scheduler() {
        if (sched != nullptr) { ggml_backend_sched_free(sched); sched = nullptr; }
        if (be != nullptr) { ggml_backend_free(be); be = nullptr; }
    }

    bool place() { return sched != nullptr && ggml_backend_sched_alloc_graph(sched, res.get_gf()) != 0; }

    bool inputs_placed() const {
        for (const ggml_tensor * t : {k, v, q, m}) {
            if (t == nullptr || t->buffer == nullptr) return false;
        }
        return true;
    }

    void write_inputs(const fixture & f) {
        if (!inputs_placed()) return;
        ggml_backend_tensor_set(k, f.k.data(), 0, f.k.size() * sizeof(float));
        ggml_backend_tensor_set(v, f.v.data(), 0, f.v.size() * sizeof(float));
        ggml_backend_tensor_set(q, f.q.data(), 0, f.q.size() * sizeof(float));
        ggml_backend_tensor_set(m, f.mask.data(), 0, f.mask.size() * sizeof(float));
    }
};

std::string self_exe() {
    char buf[4096] = {};
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    return n > 0 ? std::string(buf, std::size_t(n)) : std::string();
}

void expect_abort(const std::string & self, const char * mode, const std::string & what) {
    const pid_t pid = ::fork();
    if (pid < 0) {
        check(false, what + ": fork failed: " + std::strerror(errno));
        return;
    }
    if (pid == 0) {
        char arg0[4096];
        std::snprintf(arg0, sizeof arg0, "%s", self.c_str());
        char * const argv[] = {arg0, const_cast<char *>(mode), nullptr};
        ::execv(self.c_str(), argv);
        std::fprintf(stderr, "child: execv failed: %s\n", std::strerror(errno));
        ::_exit(127);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            check(false, what + ": waitpid failed: " + std::strerror(errno));
            return;
        }
    }
    const bool signalled = WIFSIGNALED(status);
    const int sig = signalled ? WTERMSIG(status) : 0;
    check(signalled && sig == SIGABRT,
          what + ": expected SIGABRT, got " +
              (signalled ? "signal " + std::to_string(sig)
                         : "exit " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1)));
}

// ---- child mode 1: publish before the scheduler placed the tensors ------------------------------
int abort_before_placement() {
    const fixture f;
    host h;
    if (!h.built) { std::fprintf(stderr, "child: host setup failed\n"); return 2; }
    h.res.set_inputs(nullptr);
    std::fprintf(stderr, "child: set_inputs before placement RETURNED, which must not happen\n");
    return 3;
}

// ---- child mode 2: a second publication on the same graph ---------------------------------------
int abort_repeat() {
    const fixture f;
    host h;
    if (!h.built) { std::fprintf(stderr, "child: host setup failed\n"); return 2; }
    if (!h.place()) { std::fprintf(stderr, "child: placement failed\n"); return 2; }
    h.write_inputs(f);
    h.res.set_inputs(nullptr);
    if (h.stats->passes != 1) {
        std::fprintf(stderr, "child: the first publication did not complete (%d passes)\n",
                     h.stats->passes);
        return 2;
    }
    h.res.set_inputs(nullptr);
    std::fprintf(stderr, "child: the repeat RETURNED, which must not happen\n");
    return 3;
}

// ---- the accepting path, in this process --------------------------------------------------------
void run_accepting_path() {
    const fixture f;
    std::vector<float> want;
    reference(f, want);

    host h;
    check(h.built, "the host built a graph, a prefill emission and a publisher");
    if (!h.built) return;
    check(ggml_get_no_alloc(h.ctx), "upstream's graph-result context is no_alloc");
    check(h.em.per_token.size() == std::size_t(kTokens),
          "the emission carries one result per token, got " +
              std::to_string(h.em.per_token.size()));
    for (std::size_t t = 0; t < h.em.per_token.size(); ++t) {
        check(h.em.per_token[t].indices.size() == full_view().spans.size() + 1,
              "token " + std::to_string(t) + ": spans+1 entries (one gather per span, plus the scatter)");
    }

    // WHERE THE RESULT IS BOUND, NOT JUST CHECKED. `check(h.place(), ...)` followed by a second
    // `h.place()` was the first version of this fix, and it aborted inside `ggml_backend_sched_alloc_graph`
    // on the SECOND call -- allocating an already-allocated graph is not idempotent. `check` only
    // tallies a failure; it does not carry the answer, so the value has to be bound to a name before
    // it can decide a branch. (This is the same defect shape as `herd_reader_publisher_test.cpp`'s
    // missing `placed` variable, which is why the comment is explicit.)
    const bool placed = h.place();
    check(placed, "the scheduler placed the graph");
    if (!placed) return;
    // Placement FIRST: writing an unplaced tensor aborts inside ggml, and the message would not name
    // which tensor. The named check below turns that into a diagnosable failure.
    check(h.inputs_placed(), "every input tensor (k/v/q/mask) is placed");
    if (!h.inputs_placed()) return;
    {
        for (std::size_t t = 0; t < h.em.per_token.size(); ++t) {
            for (std::size_t s = 0; s < h.em.per_token[t].indices.size(); ++s) {
                const auto & e = h.em.per_token[t].indices[s];
                if (e.index == nullptr || e.index->buffer == nullptr) {
                    check(false, "token " + std::to_string(t) + " entry " + std::to_string(s) +
                                     " is not placed: no node consumes it, so writing its row "
                                     "table would fault");
                    return;
                }
            }
        }
    }
    h.write_inputs(f);

    h.res.set_inputs(nullptr);
    check(h.stats->set_input_calls == 1, "set_inputs reached the input once");
    check(h.stats->attempted, "the publication was attempted, not skipped");
    check(!h.stats->contract_violation,
          "a clean publication records no violation" + (h.stats->why.empty() ? "" : " -- " + h.stats->why));
    check(h.stats->passes == 1, "exactly one publication pass, got " + std::to_string(h.stats->passes));
    check(h.stats->tokens_published == kTokens,
          "every token was published, got " + std::to_string(h.stats->tokens_published));
    check(int(h.stats->per_token.size()) == kTokens, "a verdict per token");
    for (std::size_t t = 0; t < h.stats->per_token.size(); ++t) {
        check(h.stats->per_token[t] == tq::reader::index_publish_refusal::ok,
              "token " + std::to_string(t) + ": published, got " +
                  tq::reader::describe(h.stats->per_token[t]));
    }
    if (h.stats->passes != 1) {
        check(false, "not computing: the row tables were not published");
        return;
    }

    // The cells, read back out of the TENSORS, per token. A publication that wrote token 2's table
    // into token 1's tensors passes every count above and fails here.
    {
        const std::vector<std::int32_t> want_cells = [&] {
            std::vector<std::int32_t> cells;
            for (const auto & span : full_view().spans)
                for (std::int32_t c : span.cells) cells.push_back(c);
            return cells;
        }();
        for (std::size_t t = 0; t < h.em.per_token.size(); ++t) {
            std::vector<std::int32_t> seen;
            for (std::size_t s = 0; s < h.em.per_token[t].indices.size() - 1; ++s) {
                const auto & e = h.em.per_token[t].indices[s];
                const std::size_t before = seen.size();
                seen.resize(before + std::size_t(e.n_cells));
                if (e.index->data != nullptr) {
                    std::memcpy(seen.data() + before, e.index->data,
                                std::size_t(e.n_cells) * sizeof(std::int32_t));
                } else {
                    check(false, "token " + std::to_string(t) + " gather " + std::to_string(s) +
                                     " has no data after a successful publish");
                    seen.resize(before);
                }
            }
            check(seen == want_cells,
                  "token " + std::to_string(t) + ": published gathers hold the view's cells in order");
        }
    }

    {
        llm_graph_params params{};
        check(!h.raw->can_reuse(params), "can_reuse is false, so upstream cannot reuse this graph");
    }

    if (ggml_backend_sched_graph_compute(h.sched, h.res.get_gf()) != GGML_STATUS_SUCCESS) {
        check(false, "compute failed after a successful publication");
    } else {
        std::vector<float> got(want.size(), 0.0f);
        ggml_backend_tensor_get(h.em.out, got.data(), 0, got.size() * sizeof(float));
        check(got.size() == want.size(), "output size");
        double worst = 0.0;
        int at = -1;
        bool finite = true;
        for (std::size_t i = 0; i < got.size() && i < want.size(); ++i) {
            if (!std::isfinite(got[i])) { finite = false; break; }
            const double d = std::fabs(double(got[i]) - double(want[i]));
            if (d > worst) { worst = d; at = int(i); }
        }
        check(finite, "out is finite");
        check(worst < 1e-6, "prefill matches the per-(token, head) definition: max|diff| = " +
                                std::to_string(worst) + " at " + std::to_string(at));
        double magnitude = 0.0;
        for (float x : want) magnitude = std::max(magnitude, std::fabs(double(x)));
        check(magnitude > 1e-2, "the reference is non-degenerate (" + std::to_string(magnitude) + ")");
        // Non-vacuity: the tokens must differ from each other, or a broadcast mask and a shared query
        // would let the comparison pass while the per-token axis did nothing.
        double token_gap = 0.0;
        for (int hd = 0; hd < kHeads; ++hd)
            for (int d = 0; d < kHeadDim; ++d) {
                const double a = double(want[std::size_t(d) + std::size_t(kHeadDim) *
                                                               (0 + std::size_t(kTokens) * std::size_t(hd))]);
                const double b = double(want[std::size_t(d) + std::size_t(kHeadDim) *
                                                               (2 + std::size_t(kTokens) * std::size_t(hd))]);
                token_gap = std::max(token_gap, std::fabs(a - b));
            }
        check(token_gap > 1e-3, "tokens differ by " + std::to_string(token_gap) +
                                    " -- per-token visibility is observable");
        std::fprintf(stderr, "  prefill publish: max|diff| = %.3e (at %d), |ref|max = %.3e, "
                             "token gap = %.3e\n", worst, at, magnitude, token_gap);
    }

    // The shared-ownership guarantee: the scheduler goes first, then the graph, then the stats are
    // read -- after the input is gone.
    const int calls_before = h.stats->set_input_calls;
    h.release_scheduler();
    h.res.reset();
    check(h.stats->destroyed, "reset() destroyed the input (its destructor ran)");
    check(h.stats->set_input_calls == calls_before,
          "after reset, set_inputs reaches no input (the old one is gone)");
    check(h.stats->tokens_published == kTokens,
          "and the statistics are still readable AFTER the input died -- shared ownership is what "
          "makes this safe, where a raw pointer would have been a write to freed memory");
    check(h.res.inputs.empty(), "reset() clears the owned inputs");
}

} // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--abort-before-placement") return abort_before_placement();
    if (mode == "--abort-repeat") return abort_repeat();

    const std::string self = self_exe();
    if (self.empty()) {
        std::fprintf(stderr, "FAIL: cannot resolve /proc/self/exe\n");
        return 1;
    }

    expect_abort(self, "--abort-before-placement",
                 "publishing before placement kills the process rather than letting it compute");
    expect_abort(self, "--abort-repeat",
                 "a second set_inputs on a published graph kills the process rather than skipping");

    run_accepting_path();

    if (failures == 0) {
        std::fprintf(stderr, "prefill publisher: OK\n");
        return 0;
    }
    std::fprintf(stderr, "prefill publisher: %d failure(s)\n", failures);
    return 1;
}
