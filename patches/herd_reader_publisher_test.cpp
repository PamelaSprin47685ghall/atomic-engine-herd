// The production publisher, driven by upstream's real `llm_graph_result`: lifecycle, reuse, and the
// numbers that make the publication worth doing.
//
// TWO PROCESSES, BECAUSE THE FAILURE MODE IS A PROCESS DEATH.
//
// An earlier revision asserted these cases by catching a C++ exception in-process. That was wrong for
// a reason this file now encodes structurally: `set_input` sits under
// `llama_process` (extern "C") -> `ctx->decode` -> `process_ubatch` -> `set_inputs`, and no frame on
// that chain catches anything, so the publisher's refusal cannot be an exception without unwinding
// across a C ABI. It is `std::abort()`. A test cannot catch an abort; it can only observe the
// termination of a process that performs one.
//
// So the driver (`main`, no arguments) re-executes itself for the two refusing cases and inspects the
// child's exit status, then runs the accepting case in its own process:
//
//   <self> --abort-before-placement   set_inputs before the scheduler placed the tensors  -> SIGABRT
//   <self> --abort-repeat             a second set_inputs on a graph already published    -> SIGABRT
//   <self>                            the correct path: publish, read back, compute, reset
//
// Using a child process also makes the assertions honest about what a host would experience: the
// refusal is not a value the driver can inspect and ignore, it is the run ending.
#include "../patches/herd_reader_publisher.hpp"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <csignal>

#include "llama-graph.h"

#include <tq/reader/graph_emit_physical.h>
#include <tq/reader/upstream_view.h>

#include <cmath>
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

constexpr int kHeadDim = 4, kKvHeads = 2, kQHeads = 4, kCell = 8;
constexpr float kScale = 0.5f;

struct fixture {
    std::vector<float> k, v, q;
    fixture() {
        k.assign(std::size_t(kHeadDim) * kKvHeads * kCell, 0.0f);
        v.assign(k.size(), 0.0f);
        q.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
        unsigned s = 0x1234abcdu;
        auto next = [&s] {
            s = s * 1664525u + 1013904223u;
            return float((s >> 8) & 0xFFFF) / 65536.0f - 0.5f;
        };
        for (float & x : k) x = next();
        for (float & x : v) x = next() * 1.5f;
        for (float & x : q) x = next();
    }
    float K(int d, int kvh, int c) const { return k[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float V(int d, int kvh, int c) const { return v[std::size_t(d) + kHeadDim * (kvh + kKvHeads * c)]; }
    float Q(int d, int h) const { return q[std::size_t(d) + kHeadDim * h]; }
};

// Softmax over the named cells only (the emitter's compact axis), one query row, no mask.
void reference(const fixture & f, const std::vector<int> & cells, std::vector<float> & out) {
    const int group = kQHeads / kKvHeads;
    out.assign(std::size_t(kHeadDim) * kQHeads, 0.0f);
    for (int h = 0; h < kQHeads; ++h) {
        const int kvh = h / group;
        std::vector<double> score(cells.size(), 0.0);
        double max_s = -INFINITY;
        for (std::size_t i = 0; i < cells.size(); ++i) {
            double s = 0.0;
            for (int d = 0; d < kHeadDim; ++d)
                s += double(f.Q(d, h)) * double(f.K(d, kvh, cells[i]));
            score[i] = s * double(kScale);
            if (score[i] > max_s) max_s = score[i];
        }
        double sum = 0.0;
        for (double & s : score) { s = std::exp(s - max_s); sum += s; }
        for (int d = 0; d < kHeadDim; ++d) {
            double acc = 0.0;
            for (std::size_t i = 0; i < cells.size(); ++i)
                acc += (score[i] / sum) * double(f.V(d, kvh, cells[i]));
            out[std::size_t(d) + std::size_t(kHeadDim) * h] = float(acc);
        }
    }
}

tq::reader::query_view the_view() {
    tq::reader::query_view v;
    tq::reader::span_keys a;
    a.effective_pos = 5;
    a.cells = {6, 2};
    tq::reader::span_keys b;
    b.effective_pos = 5;
    b.cells = {4};
    v.spans.push_back(a);
    v.spans.push_back(b);
    return v;
}

// Everything both modes need: the graph, the emission, and the registered publisher.
struct host {
    llm_graph_result res{4096};
    ggml_context * ctx = nullptr;
    ggml_tensor * k = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * q = nullptr;
    tq::reader::query_view view = the_view();
    tq::reader::physical_emit_result em{};
    herd::reader::publisher_stats_ptr stats = std::make_shared<herd::reader::publisher_stats>();
    herd::reader::publisher * raw = nullptr;
    ggml_backend_t be = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool ok = false;

    host() {
        ctx = res.get_ctx();
        if (ctx == nullptr || !ggml_get_no_alloc(ctx)) return;
        k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, kKvHeads, kCell);
        q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kHeadDim, 1, kQHeads);
        ggml_set_input(k); ggml_set_input(v); ggml_set_input(q);

        tq::reader::graph_inputs in;
        in.ctx = ctx;
        in.k = k; in.v = v; in.q = q;
        in.n_head = kQHeads; in.n_head_kv = kKvHeads;
        in.head_dim = kHeadDim; in.n_cell = kCell;
        in.collect_indices = true;
        em = tq::reader::emit_physical_attention(in, view, kScale);
        if (em.refusal != tq::reader::physical_emit_refusal::ok) return;
        if (em.indices.size() != view.spans.size() + 1 || em.out == nullptr) return;

        raw = static_cast<herd::reader::publisher *>(
            res.add_input(std::make_unique<herd::reader::publisher>(em, stats)));
        if (raw == nullptr || res.inputs.size() != 1) return;

        ggml_build_forward_expand(res.get_gf(), em.out);
        be = ggml_backend_cpu_init();
        if (be == nullptr) return;
        sched = ggml_backend_sched_new(&be, nullptr, 1, GGML_DEFAULT_GRAPH_SIZE, false, false);
        ok = sched != nullptr;
    }

    ~host() { release_scheduler(); }

    // Frees the scheduler and its backend and FORGETS them.
    //
    // Two things make this a method rather than two lines in the destructor. First, the scheduler was
    // created with this graph's tensors, so it must be released before `res.reset()` destroys the
    // context they live in; the accepting path calls `reset()` explicitly, and without this call the
    // scheduler would outlive the tensors it was initialised against on that path only. Second, the
    // handles are nulled, so a second release (destructor after an explicit call) cannot free the same
    // object twice -- the destructor body runs before members are destroyed, which is why the ordering
    // is correct but also why an unguarded explicit call would be a double free.
    void release_scheduler() {
        if (sched != nullptr) {
            ggml_backend_sched_free(sched);
            sched = nullptr;
        }
        if (be != nullptr) {
            ggml_backend_free(be);
            be = nullptr;
        }
    }

    bool place() { return sched != nullptr && ggml_backend_sched_alloc_graph(sched, res.get_gf()) != 0; }

    void write_inputs(const fixture & f) {
        if (k == nullptr || v == nullptr || q == nullptr) return;
        ggml_backend_tensor_set(k, f.k.data(), 0, f.k.size() * sizeof(float));
        ggml_backend_tensor_set(v, f.v.data(), 0, f.v.size() * sizeof(float));
        ggml_backend_tensor_set(q, f.q.data(), 0, f.q.size() * sizeof(float));
    }
};

// Runs the child modes and reports whether each died by SIGABRT.
//
// NOT through `std::system`, and both reasons are concrete:
//
//   * a shell sits between this process and the child, and a shell reports a signal death by
//     EXITING with 128+n. So `WIFSIGNALED` is false and `WEXITSTATUS` reads 134 -- the assertion
//     would see an ordinary exit, and whether it passed would depend on whether it also happened to
//     accept "non-zero". The property under test is "the process was killed", which only `waitpid`
//     on the direct child can report.
//   * `system` concatenates the executable path into a command string. A build directory
//     containing a space would silently run a different program (or none), and the failure would
//     look like a missing abort rather than a malformed command.
//
// `execv` takes `argv` verbatim and does not search `PATH`, which is what we want here: `self` came
// from /proc/self/exe and is already absolute.
void expect_abort(const char * self, const char * mode, const std::string & what) {
    const pid_t pid = ::fork();
    if (pid < 0) {
        check(false, what + ": fork failed: " + std::strerror(errno));
        return;
    }
    if (pid == 0) {
        char arg0[4096];
        std::snprintf(arg0, sizeof arg0, "%s", self);
        char * const argv[] = {arg0, const_cast<char *>(mode), nullptr};
        ::execv(self, argv);
        // Only reachable when exec itself failed, so the driver sees a distinguishable code rather
        // than a SIGABRT it would otherwise credit as a correct refusal.
        std::fprintf(stderr, "child: execv(%s) failed: %s\n", self, std::strerror(errno));
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
                         : "exit " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) +
                               " -- a shell would report 134 here, which is why the child is forked "
                               "directly"));
}

// ---- child mode 1: publish before placement ----------------------------------------------------
int abort_before_placement() {
    const fixture f;
    host h;
    if (!h.ok) { std::fprintf(stderr, "child: host setup failed\n"); return 2; }
    // No `place()` call: the index tensors have no storage yet.
    h.res.set_inputs(nullptr);
    std::fprintf(stderr, "child: set_inputs before placement RETURNED, which must not happen\n");
    return 3;
}

// ---- child mode 2: a second publish on the same graph ------------------------------------------
int abort_repeat() {
    const fixture f;
    host h;
    if (!h.ok) { std::fprintf(stderr, "child: host setup failed\n"); return 2; }
    if (!h.place()) { std::fprintf(stderr, "child: placement failed\n"); return 2; }
    h.write_inputs(f);
    h.res.set_inputs(nullptr);   // the correct publication
    if (h.stats->writes != 1) { std::fprintf(stderr, "child: first publish did not write\n"); return 2; }
    h.res.set_inputs(nullptr);   // must abort
    std::fprintf(stderr, "child: the repeat RETURNED, which must not happen\n");
    return 3;
}

// ---- driver: the accepting path, in this process ----------------------------------------------
void run_accepting_path() {
    const fixture f;
    host h;
    check(h.ok, "the host built a graph, an emission and a publisher");
    if (!h.ok) return;
    check(ggml_get_no_alloc(h.ctx), "upstream's graph-result context is no_alloc");
    check(h.em.indices.size() == h.view.spans.size() + 1,
          "the emission is spans+1 entries (one gather per span, plus the scatter)");

    // Placement FIRST: `ggml_backend_tensor_set` on an unplaced tensor aborts (the context is
    // no_alloc, so `data` does not exist yet). The order here is the same one a host must follow.
    //
    // AND THE FAILURE RETURNS. An earlier revision called `check(h.place(), ...)` and carried on to
    // `h.write_inputs(f)`, so a failed placement wrote into tensors with no storage -- the same
    // ggml abort that the placement-before-write ordering exists to prevent, only now reached by a
    // path the test itself created. `check` records a failure; it does not stop the function, and a
    // test that keeps running after the precondition it needs has failed cannot report the thing it
    // names. It crashes instead, and a crash in a red test hides the diagnosis.
    //
    // AND THE RETURN VALUE IS KEPT. The previous revision wrote `check(h.place(), ...)` and then
    // tested `h.sched == nullptr`, which asks a different question: `sched` is created in the
    // constructor, so it is non-null here whether or not `ggml_backend_sched_alloc_graph` succeeded.
    // A failed placement therefore fell straight through to the writes -- the exact abort the
    // paragraph above is about, reintroduced one line below the comment that describes it. Binding
    // the result to a named `bool` is what makes the early return possible; `check` alone cannot.
    const bool placed = h.place();
    check(placed, "the scheduler placed the graph");
    if (!placed) return;
    for (const auto & e : h.em.indices) {
        if (e.index == nullptr || e.index->buffer == nullptr) {
            check(false, "the graph was not placed: an index tensor has no buffer, so writing the "
                         "row table would fault. Stopping rather than writing into unplaced "
                         "storage.");
            return;
        }
    }
    // The inputs too, not just the indices: `ggml_backend_tensor_set` on an unplaced k/v/q aborts
    // inside ggml, where the message says nothing about which tensor or why. Checking here turns a
    // nameless abort into a named failure.
    for (const auto * t : {h.k, h.v, h.q}) {
        if (t == nullptr || t->buffer == nullptr) {
            check(false, "an input tensor (k/v/q) has no buffer after placement, so writing it "
                         "would abort inside ggml with no indication of which one");
            return;
        }
    }
    h.write_inputs(f);

    h.res.set_inputs(nullptr);
    check(h.stats->set_input_calls == 1, "set_inputs reached the input once");
    check(h.stats->attempted, "the publication was attempted, not skipped");
    check(h.stats->last == tq::reader::index_publish_refusal::ok,
          std::string("publishing after placement succeeds, got ") +
              tq::reader::describe(h.stats->last) + " (" + h.stats->why + ")");
    check(h.stats->writes == 1, "exactly one write");
    check(!h.stats->contract_violation, "a correct publication records no violation");
    if (h.stats->writes != 1) {
        check(false, "not computing: the row table was never written");
        return;
    }

    // The cells, read back out of the TENSORS: what a wrong publish would move.
    {
        const std::vector<int> cells{6, 2, 4};
        std::vector<std::int32_t> seen;
        for (std::size_t s = 0; s < h.view.spans.size(); ++s) {
            const auto & e = h.em.indices[s];
            const std::size_t before = seen.size();
            seen.resize(before + std::size_t(e.n_cells));
            if (e.index->data != nullptr) {
                std::memcpy(seen.data() + before, e.index->data,
                            std::size_t(e.n_cells) * sizeof(std::int32_t));
            } else {
                check(false, "gather " + std::to_string(s) + " has no data after a successful publish");
                seen.resize(before);
            }
        }
        check(seen == std::vector<std::int32_t>(cells.begin(), cells.end()),
              "the published gathers hold the view's cells in order");
    }

    // can_reuse must be false: a graph carrying written indices must not be submitted again. The
    // params block is default-constructed because the override ignores it -- and because a whole
    // decode's parameters cannot be fabricated honestly.
    {
        llm_graph_params params{};
        check(!h.raw->can_reuse(params), "can_reuse is false, so upstream cannot reuse this graph");
    }

    if (ggml_backend_sched_graph_compute(h.sched, h.res.get_gf()) != GGML_STATUS_SUCCESS) {
        check(false, "compute failed after a successful publish");
    } else {
        const std::vector<int> cells{6, 2, 4};
        std::vector<float> got(std::size_t(kHeadDim) * kQHeads, 0.0f);
        ggml_backend_tensor_get(h.em.out, got.data(), 0, got.size() * sizeof(float));
        std::vector<float> want;
        reference(f, cells, want);
        double worst = 0.0;
        int at = -1;
        bool finite = true;
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (!std::isfinite(got[i])) { finite = false; break; }
            const double d = std::fabs(double(got[i]) - double(want[i]));
            if (d > worst) { worst = d; at = int(i); }
        }
        check(finite, "out is finite");
        check(worst < 1e-6, "published-index attention matches the definition: max|diff| = " +
                                std::to_string(worst) + " at " + std::to_string(at));
        double magnitude = 0.0;
        for (float x : want) magnitude = std::max(magnitude, std::fabs(double(x)));
        check(magnitude > 1e-3, "the reference is non-degenerate (" + std::to_string(magnitude) + ")");
    }

    // The shared-ownership guarantee, asserted after the input is gone.
    //
    // The scheduler is released FIRST and explicitly. It was created against this graph's tensors, and
    // `res.reset()` destroys the context they live in; on this path the destructor would only run at
    // scope exit, i.e. after the reset, so relying on it would leave the scheduler pointing into a
    // freed context for the rest of the function. `release_scheduler()` also nulls the handles, so the
    // destructor that follows cannot free them again.
    const int calls_before = h.stats->set_input_calls;
    h.release_scheduler();
    h.res.reset();
    check(h.stats->destroyed, "reset() destroyed the input (its destructor ran)");
    check(h.stats->set_input_calls == calls_before,
          "after reset, set_inputs reaches no input (the old one is gone)");
    check(h.stats->writes == 1,
          "and the statistics are still readable AFTER the input died -- shared ownership is what "
          "makes this safe, where a raw pointer would have been a write to freed memory");
    check(h.res.inputs.empty(), "reset() clears the owned inputs");
}

} // namespace

int main(int argc, char ** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--abort-before-placement") return abort_before_placement();
    if (mode == "--abort-repeat") return abort_repeat();

    char self[4096] = {};
    const ssize_t n = ::readlink("/proc/self/exe", self, sizeof self - 1);
    if (n <= 0) {
        std::fprintf(stderr, "FAIL: cannot resolve /proc/self/exe\n");
        return 1;
    }

    // The two refusing cases, observed as process deaths. This is the property the previous revision
    // asserted by catching an exception it should never have thrown: `set_input` reachest the host
    // through `llama_process` (extern "C") -> `decode` -> `process_ubatch`, with no catch on that
    // chain, so the only failure mode that cannot be ignored is termination.
    expect_abort(self, "--abort-before-placement",
                 "publishing before placement kills the process rather than letting it compute");
    expect_abort(self, "--abort-repeat",
                 "a second set_inputs on a published graph kills the process rather than skipping");

    run_accepting_path();

    if (failures == 0) {
        std::fprintf(stderr, "reader publisher: OK\n");
        return 0;
    }
    std::fprintf(stderr, "reader publisher: %d failure(s)\n", failures);
    return 1;
}
