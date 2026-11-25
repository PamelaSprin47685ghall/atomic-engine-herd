// herd/release_probe.cpp — the measurement the release claim depends on.
//
// The claim: a scheduler's compute buffers can be returned to the system between a
// prompt pass and a generation pass, and the next decode re-acquires them by itself
// through upstream's own retry path, producing the same logits as a run that never
// released anything.
//
// Both halves are measured here in one process on one model:
//
//   A (control)  prompt decode, then one token decode, no release anywhere.
//   B (release)  same decodes, with a release between them.
//
// What is asserted:
//   1. before  > 0        bytes are really held after the prompt
//   2. released           the release reported success
//   3. after   == 0       nothing is held while the buffers are gone
//   4. decode  == 0       the second decode succeeds WITHOUT any reserve of ours
//                         (this is the claim the previous review got wrong)
//   5. again   > 0        memory came back
//   6. logits of B == logits of A, byte for byte
//
// Point 6 is the one that matters. Without it, "it did not crash" would be the whole
// evidence, and a subtly different result would pass.
//
// Usage: herd-release-probe <model.gguf>
#include "herd_buffer_station.hpp"
#include "herd_phase.hpp"

#include <llama.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <malloc.h>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Bytes the process allocator currently holds, from glibc itself.
//
// This is the independent measurement the release claim was missing. The station's
// own counter answers "do I still own an allocator?", which is circular: after a
// release it necessarily says zero, whether or not a byte went back. mallinfo2 asks
// the allocator that actually handed out the memory, so a drop here is evidence and
// a non-drop is a refutation -- neither of which our bookkeeping can supply.
//
// Caveat, and it is why the number is reported rather than trusted alone: glibc
// keeps freed chunks in its arenas, and uordblks counts in-use chunks, so a real
// free does move it; but an alignment allocator that used mmap would move hblkhd
// instead, so both are summed. Under ASan the interposed allocator does not report
// through mallinfo2, so the number is printed with a note and never asserted there;
// the ASan build's evidence is the sanitizer's own use-after-free detection.
//
// Declared here, above its callers: the release block that uses it appears earlier
// in the file than the helper block, and a type used before its definition is a
// compile error -- the first attempt at this measurement produced exactly that.
struct alloc_state {
    std::size_t in_use = 0;
    std::size_t mapped = 0;
    std::size_t total() const { return in_use + mapped; }
};

alloc_state allocation_bytes();

struct run_result {
    bool   ok = false;
    double bytes_before_release = 0;
    double bytes_after_release  = 0;
    std::vector<float> logits;
    std::string error;
};

// One context, one prompt pass, one token pass. `release` decides whether the
// buffers are dropped in between -- the only difference between the two runs.
run_result run_once(llama_model * model, const llama_vocab * vocab,
                    const std::vector<llama_token> & prompt, bool release) {
    run_result result;

    auto params  = llama_context_default_params();
    params.n_ctx   = 128;
    params.n_batch = 64;
    params.n_ubatch = 64;
    llama_context * ctx = llama_init_from_model(model, params);
    if (!ctx) { result.error = "context creation failed"; return result; }

    llama_batch batch = llama_batch_init(64, 0, 1);
    const auto submit = [&](int32_t n, bool logits) {
        batch.n_tokens = n;
        for (int32_t i = 0; i < n; ++i) {
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = logits ? 1 : 0;
        }
        // The scheduler station learns which context is executing from this claim;
        // without it compute_bytes() has no pairing and reports "unknown", which is
        // exactly what the first version of this probe measured.
        const herd::phase::scope_guard guard(ctx);
        return llama_decode(ctx, batch);
    };

    for (std::size_t i = 0; i < prompt.size(); ++i) {
        batch.token[0] = prompt[i];
        batch.pos[0] = llama_pos(i);
        if (submit(1, true) != 0) { result.error = "prompt decode failed"; llama_batch_free(batch); llama_free(ctx); return result; }
    }
    llama_synchronize(ctx);

    if (release) {
        std::uint64_t before = 0;
        if (!herd::phase::compute_bytes(ctx, &before) || before == 0) {
            result.error = "no compute bytes measured before the release";
            llama_batch_free(batch);
            llama_free(ctx);
            return result;
        }
        result.bytes_before_release = double(before);

        // The station is keyed by scheduler; only this layer maps a context to one.
        void * sched = herd::phase::scheduler_of(ctx);
        if (!sched) {
            result.error = "release refused (no scheduler pairing)";
            llama_batch_free(batch);
            llama_free(ctx);
            return result;
        }

        // Fault injection, and its only purpose is to prove the probe measures the
        // right thing: `release_sched` frees the allocator without arming the
        // scheduler's next reservation, which is the state ASan flagged as
        // heap-use-after-free. Running it deliberately is how the *other* path stops
        // being "it did not crash, therefore it is correct".
        const bool arm = std::getenv("HERD_RELEASE_NO_ARM") == nullptr;
        const alloc_state alloc_before = allocation_bytes();
        const bool released = arm ? herd::buffers::release_and_arm(ctx, false)
                                  : herd::buffers::release_sched(sched);
        if (!released) {
            result.error = "release refused (already released, or no allocator)";
            llama_batch_free(batch);
            llama_free(ctx);
            return result;
        }
        const alloc_state alloc_after = allocation_bytes();

        std::uint64_t after = 0;
        const bool known = herd::phase::compute_bytes(ctx, &after);
        result.bytes_after_release = double(after);
        check(known && after == 0, "the station holds no allocator after the release");

        // Independent check: the *process allocator* must have given bytes back. The
        // station's own zero above cannot distinguish "released" from "never had it",
        // which is exactly the circularity this measures away.
        std::fprintf(stderr,
                     "allocator: %zu KiB in use before, %zu KiB after the release (delta %+zd KiB)%s\n",
                     alloc_before.total() / 1024, alloc_after.total() / 1024,
                     (long)((long)alloc_after.total() - (long)alloc_before.total()) / 1024,
                     arm ? "" : "  [no re-arm: fault injection]");
#if defined(__SANITIZE_ADDRESS__)
        // ASan's interposed allocator does not answer mallinfo2; the sanitizer's own
        // use-after-free report is this build's evidence, so nothing here is asserted.
        (void)alloc_before;
        (void)alloc_after;
#else
        check(alloc_after.total() < alloc_before.total(),
              "the process allocator gave memory back (independent of the station)");
#endif
    }

    batch.token[0] = prompt.back();
    batch.pos[0] = llama_pos(prompt.size());
    if (submit(1, true) != 0) { result.error = "post-release decode failed"; llama_batch_free(batch); llama_free(ctx); return result; }

    const float * logits = llama_get_logits_ith(ctx, -1);
    if (!logits) { result.error = "no logits after the second decode"; llama_batch_free(batch); llama_free(ctx); return result; }
    const int n_vocab = llama_vocab_n_tokens(vocab);
    result.logits.assign(logits, logits + n_vocab);
    for (float value : result.logits) {
        if (!std::isfinite(value)) { result.error = "non-finite logit"; break; }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    result.ok = result.error.empty();
    return result;
}

// Bytes the process allocator currently holds, from glibc itself.
//
// This is the independent measurement the release claim was missing. The station's
// own counter answers "do I still own an allocator?", which is circular: after a
// release it necessarily says zero, whether or not a byte went back. mallinfo2 asks
// the allocator that actually handed out the memory, so a drop here is evidence and
// a non-drop is a refutation -- neither of which our bookkeeping can supply.
//
// Caveat, and it is why the number is reported rather than trusted alone: glibc
// keeps freed chunks in its arenas, and uordblks counts in-use chunks, so a real
// free does move it; but an alignment allocator that used mmap would move hblkhd
// instead, so both are summed. Under ASan the interposed allocator does not report
// through mallinfo2, so the number is printed with a note and never asserted there;
// the ASan build's evidence is the sanitizer's own use-after-free detection.
alloc_state allocation_bytes() {
    alloc_state state;
#if defined(__SANITIZE_ADDRESS__)
    return state; // ASan does not report through mallinfo2; zero means "not measured"
#else
    const struct mallinfo2 info = mallinfo2();
    state.in_use = info.uordblks;
    state.mapped = info.hblkhd;
    return state;
#endif
}

// Process resident set, in KiB. Corroboration only, never asserted: the compute
// buffer on the tiny model is 0.075 MiB, which is inside page granularity noise, so
// a changed RSS would not prove anything and an unchanged one would not disprove
// it. It is printed because a number a reader can sanity-check is worth more than a
// claim they must take on faith.
long rss_kib() {
    std::ifstream statm("/proc/self/statm");
    long pages = 0;
    long resident = 0;
    if (!(statm >> pages >> resident)) {
        return -1;
    }
    long page_kib = sysconf(_SC_PAGESIZE) / 1024;
    return resident * (page_kib > 0 ? page_kib : 4);
}

// Two contexts must stay independent, and this is the test that would have caught a
// station keyed the wrong way round: releasing A must not disturb B, and decoding
// B in between must not break A's re-arm. Both directions matter -- a binding by
// allocator rather than by context would make one release silently affect the
// other, which is a wrong answer, not a crash.
void two_context_check(llama_model * model, const llama_vocab * vocab,
                       const std::vector<llama_token> & prompt) {
    auto params = llama_context_default_params();
    params.n_ctx    = 128;
    params.n_batch  = 64;
    params.n_ubatch = 64;

    llama_context * a = llama_init_from_model(model, params);
    llama_context * b = llama_init_from_model(model, params);
    check(a != nullptr && b != nullptr, "two contexts exist");
    if (!a || !b) {
        if (a) llama_free(a);
        if (b) llama_free(b);
        return;
    }

    const auto decode_one = [&](llama_context * ctx, llama_token token, llama_pos pos) {
        llama_batch batch = llama_batch_init(8, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = token;
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        const herd::phase::scope_guard guard(ctx);
        const int rc = llama_decode(ctx, batch);
        if (rc == 0) {
            llama_synchronize(ctx);
        }
        llama_batch_free(batch);
        return rc;
    };

    check(decode_one(a, prompt.back(), llama_pos(prompt.size())) == 0, "context A prompt decode");
    check(decode_one(b, prompt.front(), llama_pos(0)) == 0, "context B seed decode");

    std::uint64_t a_before = 0;
    std::uint64_t b_before = 0;
    const bool a_known = herd::phase::compute_bytes(a, &a_before);
    const bool b_known = herd::phase::compute_bytes(b, &b_before);
    check(a_known && a_before > 0, "A holds compute memory before the release");
    check(b_known && b_before > 0, "B holds compute memory before the release");

    const long rss_before = rss_kib();
    check(herd::buffers::release_and_arm(a, false), "release + re-arm on A");
    const long rss_after = rss_kib();
    std::fprintf(stderr, "rss: %ld KiB before, %ld KiB after the release (corroboration only)\n",
                 rss_before, rss_after);

    // B first: the context that was not released must be untouched by A's release.
    check(decode_one(b, prompt.back(), llama_pos(1)) == 0, "context B decodes after A's release");
    std::uint64_t b_after = 0;
    check(herd::phase::compute_bytes(b, &b_after) && b_after == b_before,
          "B's compute memory is unchanged by A's release");

    // Then A: its scheduler must rebuild, not reuse the freed buffers.
    check(decode_one(a, prompt.back(), llama_pos(prompt.size() + 1)) == 0,
          "context A decodes after its own release");
    std::uint64_t a_after = 0;
    check(herd::phase::compute_bytes(a, &a_after) && a_after > 0,
          "A's compute memory came back after the rebuild");

    const float * logits_a = llama_get_logits_ith(a, -1);
    const float * logits_b = llama_get_logits_ith(b, -1);
    check(logits_a != nullptr && logits_b != nullptr, "both contexts still produce logits");
    if (logits_a && logits_b) {
        bool finite = true;
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
            finite = finite && std::isfinite(logits_a[i]) && std::isfinite(logits_b[i]);
        }
        check(finite, "both contexts produce finite logits");
    }

    llama_free(a);
    llama_free(b);
}

// Two contexts on two threads, each releasing its own scheduler while the other is
// decoding. The sequential check above proves independence of *state*; this one
// proves the station survives contention, and it is the case the design rests on but
// had never executed: the claim stack is thread_local, the station is one mutex, and
// a release on one thread must not disturb the other thread's in-flight decode.
//
// Each thread owns its context exclusively, which is what upstream requires of a
// context anyway; what is being tested is the *station*, not concurrent use of one
// llama_context.
void two_thread_check(llama_model * model, const llama_vocab * vocab,
                      const std::vector<llama_token> & prompt) {
    // Shared by the two workers: reaching 2 means both are ready to diverge.
    std::atomic<int> arrived{0};
    const auto worker = [&](bool release_midway) -> int {
        auto params = llama_context_default_params();
        params.n_ctx    = 128;
        params.n_batch  = 64;
        params.n_ubatch = 64;
        llama_context * ctx = llama_init_from_model(model, params);
        if (!ctx) {
            return 1;
        }
        int failures_here = 0;
        const auto decode_one = [&](llama_token token, llama_pos pos) {
            llama_batch batch = llama_batch_init(8, 0, 1);
            batch.n_tokens = 1;
            batch.token[0] = token;
            batch.pos[0] = pos;
            batch.n_seq_id[0] = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0] = 1;
            const herd::phase::scope_guard guard(ctx);
            const int rc = llama_decode(ctx, batch);
            if (rc == 0) {
                llama_synchronize(ctx);
            }
            llama_batch_free(batch);
            return rc;
        };

        if (decode_one(prompt.back(), llama_pos(prompt.size())) != 0) {
            ++failures_here;
        }

        // Rendezvous before the divergent step. Without it the two threads merely
        // *start* together: the releasing thread can finish its whole release before
        // the other reaches its first decode, and "concurrent" would be a claim about
        // thread creation rather than about overlap. Here both threads block until
        // both have arrived, so the release and the decode below genuinely run side
        // by side -- which is the situation the station's lock has to survive.
        //
        // Spin rather than a condition variable: two participants, one wait, and no
        // scheduler involvement that could serialise the two wake-ups back into a
        // sequence.
        arrived.fetch_add(1, std::memory_order_acq_rel);
        while (arrived.load(std::memory_order_acquire) < 2) {
            std::this_thread::yield();
        }

        if (release_midway) {
            if (!herd::buffers::release_and_arm(ctx, false)) {
                ++failures_here;
            }
        }
        if (decode_one(prompt.front(), llama_pos(prompt.size() + 1)) != 0) {
            ++failures_here;
        }
        std::uint64_t bytes = 0;
        if (!herd::phase::compute_bytes(ctx, &bytes) || bytes == 0) {
            ++failures_here;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            ++failures_here;
        } else {
            for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) {
                if (!std::isfinite(logits[i])) {
                    ++failures_here;
                    break;
                }
            }
        }
        llama_free(ctx);
        return failures_here;
    };

    int released_failures = 0;
    int control_failures  = 0;
    std::thread releasing([&] { released_failures = worker(true); });
    std::thread control([&] { control_failures = worker(false); });
    releasing.join();
    control.join();

    check(released_failures == 0, "a releasing thread finished cleanly while another decoded");
    check(control_failures == 0, "an undisturbed thread finished cleanly beside a releasing one");
    std::fprintf(stderr, "two threads: %d failure(s) releasing, %d failure(s) control\n",
                 released_failures, control_failures);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        llama_backend_free();
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const char * text = "hello world";
    std::vector<llama_token> prompt(64);
    int n = llama_tokenize(vocab, text, int(std::strlen(text)), prompt.data(), int(prompt.size()), true, true);
    if (n <= 0) {
        std::fprintf(stderr, "tokenize failed: %d\n", n);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }
    prompt.resize(std::size_t(n));

    const run_result control = run_once(model, vocab, prompt, false);

    // The default run now measures the release instead of asserting that it is unsafe.
    //
    // That assertion became false when `release_and_arm` was added: the armed path is
    // ASan-clean, produces bit-identical logits, keeps two contexts independent, and
    // survives one thread releasing while another decodes. A probe that instead
    // asserted "unavailable" would pass while contradicting the code beneath it -- the
    // failure mode this file has already had once, where the header's stale warning
    // outlived the defect it described.
    //
    // HERD_RELEASE_NO_ARM=1 still reaches the unarmed path deliberately, so the clean
    // result remains evidence: the sanitizer reports the use-after-free there.
    if (std::getenv("HERD_RELEASE_EXPERIMENT") == nullptr) {
        check(control.ok, "control run completed without any release");
        // The experiment's own claim, not the product's. Reporting availability here is
        // what let a measured mechanism read as a shipped feature.
        check(herd::phase::release_experiment_verified(),
              "the release mechanism reports measured: armed path ASan clean, logits identical");
        check(!herd::phase::release_available(),
              "the product must NOT report the release available: no seam is installed and the "
              "flag is refused, so the two facts cannot be conflated");
        if (!control.ok) { std::fprintf(stderr, "control: %s\n", control.error.c_str()); }
    }
    const run_result released = run_once(model, vocab, prompt, true);

    check(control.ok, "control run completed");
    check(released.ok, "released run completed");
    if (!control.ok) { std::fprintf(stderr, "control: %s\n", control.error.c_str()); }
    if (!released.ok) { std::fprintf(stderr, "released: %s\n", released.error.c_str()); }

    if (control.ok && released.ok) {
        check(control.logits.size() == released.logits.size(), "same vocab width");
        const std::size_t width = std::min(control.logits.size(), released.logits.size());
        std::size_t differences = 0;
        double largest = 0.0;
        for (std::size_t i = 0; i < width; ++i) {
            const double delta = std::fabs(double(control.logits[i]) - double(released.logits[i]));
            if (delta != 0.0) { ++differences; }
            if (delta > largest) { largest = delta; }
        }
        std::fprintf(stderr, "logits: %zu values, %zu differ, largest |delta| = %g\n",
                     width, differences, largest);
        // Bit-identical, not "close": the release changes when memory exists, never
        // what is computed, so any difference is a real defect.
        check(differences == 0, "released run produced bit-identical logits");
    }

    // Independence, which the single-context comparison cannot show: one context's
    // release must leave another context's scheduler, memory and logits alone.
    two_context_check(model, vocab, prompt);
    two_thread_check(model, vocab, prompt);

    std::fprintf(stderr, "%s\n", failures ? "release probe FAILED" : "release probe holds");
    llama_model_free(model);
    llama_backend_free();
    return failures ? 1 : 0;
}
