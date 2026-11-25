// STATUS: passes on a real model. Registered as `herd.embeddings_attestation`, gated on
// HERD_EMBEDDINGS_TEST_MODEL (a GGUF path) because it needs a model and cannot be a default
// ctest; it lives inside the HERD_BUILD_BUFFER_HOOKS block, which is OFF by default, so a
// default build neither compiles nor runs it.
//
// Measured on this pin (tiny transformer, CPU):
//
//   case 1, correct attestation   llama_set_embeddings 0,1 -> embeddings still available
//   case 2, wrong attestation     llama_set_embeddings 1,0 -> embeddings GONE, decode still fine
//
// The two log lines are the proof that the double toggle ran and that the two cases took
// different paths: with the right value the first call is the one that changes the setting,
// with the wrong value it is the second. The cases also constrain each other -- case 1 asserts
// embeddings existed BEFORE the release, so case 2's "they are gone" cannot be satisfied by a
// context that never had them.
//
// Two earlier failures in this file were the harness, and both are worth remembering:
//
//   * batch.token[] was never written. llama_batch_init allocates the arrays and leaves them
//     uninitialised, so llama_decode reported `invalid token[0] = -992467455` and then only the
//     generic "failed to initialize batch". A generic init failure means the cause is on the
//     previous line.
//   * no claim scope was opened. The compute seam learns ctx -> scheduler from a thread-local
//     claim stack, so without `herd::phase::scope_guard` the seam records nothing,
//     scheduler_of() answers null, and release_and_arm() refuses on a context that had just
//     decoded. batch-compose and herd-release-probe both open the guard; this harness did not.
//
// Do NOT weaken the two assertions below to make them green. Case 2 asserts the *hazard* -- that
// a wrong attestation silently disables embeddings -- and a "fix" that makes case 2 pass by
// expecting success there would delete the only evidence the parameter matters.
//
// Does the release leave the context's embeddings setting exactly as it found it?
//
// The mechanism re-arms the scheduler by toggling llama_set_embeddings(), because that is the
// one public lever that sets sched_need_reserve (llama-context.cpp:1583-1590 sets it whenever
// the value CHANGES). Toggling twice restores the value and leaves the flag set, which is what
// the rebuild branch needs.
//
// That is only true if the caller passes the context's ACTUAL value. Upstream has no getter,
// so the value is an attestation, and a wrong one is not harmless: llama_set_embeddings(true)
// on a context that already has embeddings enabled changes nothing, and the second call
// (false) then changes it -- the context comes out of the release with embeddings DISABLED,
// silently, for the rest of the run. Every later llama_get_embeddings() throws "no embeddings".
//
// This test pins both directions on a real model:
//
//   1. a context created with embeddings enabled, released with the CORRECT attestation,
//      still returns embeddings afterwards;
//   2. a context created the same way, released with a WRONG attestation (false), does not --
//      and the test asserts that, so the hazard is a measured fact about the API rather than a
//      sentence in a comment nobody can check.
//
// Case 2 is not a desirable behaviour; it is the behaviour, and recording it is what makes the
// attestation's importance verifiable. If a future upstream gains a getter, case 2 should start
// failing, and that failure is the signal to remove the parameter.
#include "herd_buffer_station.hpp"
#include "herd_phase.hpp"   // the claim scope the compute seam reads

#include <llama.h>

#include <cstdlib>
#include <exception>

#include <cmath>
#include <cstdio>
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

// The vocabulary, recorded once in main. A decode needs real token ids, and "real" means
// inside [0, n_vocab) for THIS model -- there is no universal valid id, so the number cannot
// be invented at the call site.
const llama_vocab * g_vocab = nullptr;

// One decode of `n` tokens, reported as success or as the upstream refusal.
//
// The token ids are the part that was missing, and it cost this harness a build-and-run cycle
// to find: `llama_batch_init` allocates the arrays with malloc and leaves them uninitialised,
// so a batch whose ids were never written carried heap garbage into llama_decode, which
// reported `init: invalid token[0] = -992467455` and then the generic "failed to initialize
// batch". The log line that named the cause was there all along; the generic message is what
// a caller sees.
//
// This is the same shape as every other defect this repository has had to remove: a helper
// that fills most of a structure and leaves one field to chance. The field is filled here,
// from the model's own vocabulary, because only the model knows what a valid id is.
// `base` is where this batch's positions start. A second decode on the same context must
// continue from the KV cache's last position -- upstream refuses otherwise ("it is required
// that the sequence positions remain consecutive: Y = X + 1") -- and a test that always
// starts at zero would fail for a reason unrelated to what it measures.
bool decode(llama_context * ctx, int32_t n, llama_pos base) {
    // The claim scope is not optional, and its absence was why release_and_arm() returned
    // false here while the probe's release worked.
    //
    // The compute seam learns which scheduler belongs to which context from a thread-local
    // claim stack; `current_claim()` returns null when nobody opened one, `observe()` then
    // records nothing, and `scheduler_of(ctx)` answers null -- so release_and_arm() takes its
    // "never decoded" branch and returns false on a context that had just decoded. The probe
    // and batch-compose both open a guard before decoding (patches/herd_release_probe.cpp:113,
    // modules/batch-compose/src/main.cpp:309); this harness did not, which made the release
    // impossible to measure and looked like a defect in the release.
    //
    // Opening it here, per decode, mirrors how a caller that owns a context brackets its work.
    const herd::phase::scope_guard guard(ctx);
    llama_batch batch = llama_batch_init(n, /*embd=*/0, /*n_seq_max=*/1);
    if (!batch.token) {
        return false;
    }
    batch.n_tokens = n;
    // BOS when the model has one, else token 0; wrapped into range so a vocabulary smaller
    // than `n` cannot produce an invalid id. Attention over a repeated token is a perfectly
    // good workload for a memory-release measurement -- this program decodes, it does not
    // generate.
    const int32_t n_vocab = llama_vocab_n_tokens(g_vocab);
    const int32_t bos = llama_vocab_bos(g_vocab);
    const int32_t first = (bos >= 0 && bos < n_vocab) ? bos : 0;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = (first + i) % n_vocab;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n - 1) ? 1 : 0;
        batch.pos[i] = base + i;
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// llama_get_embeddings() throws when embeddings are disabled -- it does not return null. So
// "does this context still produce embeddings" is a question answered by catching.
bool embeddings_available(llama_context * ctx) {
    try {
        return llama_get_embeddings(ctx) != nullptr;
    } catch (const std::exception &) {
        return false;
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    // argv[1] directly. A `std::vector<char> model_path(argv[1], ...)` sat here and was never
    // passed to anything -- dead weight from an earlier revision, and worse than dead weight:
    // it reads as evidence that the loader needs a mutable buffer, which sends the next reader
    // to look for a reason that is not there.
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        return 2;
    }
    // The vocabulary is the model's, so it is read once here rather than guessed per decode.
    g_vocab = llama_model_get_vocab(model);

    // Small enough to be quick, large enough that a decode is a real decode.
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = 8;
    params.n_ubatch = 8;
    params.n_threads = 2;
    params.embeddings = true;   // the fact under test

    const auto fresh = [&]() {
        llama_context * ctx = llama_init_from_model(model, params);
        if (!ctx) {
            std::fprintf(stderr, "context creation failed\n");
            std::exit(2);
        }
        return ctx;
    };

    // ---- case 1: the correct attestation preserves the setting ----
    {
        llama_context * ctx = fresh();
        check(decode(ctx, 2, 0), "correct: the first decode succeeds");

        // The premise. If a model cannot produce embeddings at all, the test would be
        // vacuous: every later "no embeddings" answer would be indistinguishable from the
        // defect it is meant to catch.
        const bool before = embeddings_available(ctx);
        check(before, "correct: the context produces embeddings before the release");

        check(herd::buffers::release_and_arm(ctx, /*embeddings_currently=*/true),
              "correct: release_and_arm reports success");

        check(decode(ctx, 2, 2), "correct: the decode after the release succeeds");
        const bool after = embeddings_available(ctx);
        check(after == before,
              "correct: embeddings are still available after a release that attested true");

        llama_free(ctx);
    }

    // ---- case 2: a wrong attestation silently disables them ----
    //
    // Not a behaviour to preserve: the behaviour to RECORD. A caller that guesses the value
    // wrong gets a context that looks healthy, decodes fine, and has lost the ability to
    // return embeddings -- with no error at the release and none at the next decode.
    {
        llama_context * ctx = fresh();
        check(decode(ctx, 2, 0), "wrong: the first decode succeeds");
        check(embeddings_available(ctx), "wrong: embeddings exist before the release");

        check(herd::buffers::release_and_arm(ctx, /*embeddings_currently=*/false),
              "wrong: release_and_arm still reports success -- it cannot tell");

        check(decode(ctx, 2, 2), "wrong: decoding still works, which is what makes this silent");
        check(!embeddings_available(ctx),
              "wrong: a false attestation leaves embeddings DISABLED; the release cannot "
              "detect it, so the value must come from the caller's own record");

        llama_free(ctx);
    }

    llama_model_free(model);
    llama_backend_free();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "embeddings: a correct attestation preserves them; a wrong one disables them "
                 "silently, which is why the value is the caller's to record\n");
    return 0;
}
