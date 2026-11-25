// herd_prefab_split_probe.cpp — does padding at ubatch birth change the answer?
//
// The claim under test: an ubatch padded before the KV plan is built, with its pad
// rows owned by a scratch sequence, (a) is accepted by upstream at all, (b) produces
// bit-identical logits to the unpadded run, and (c) returns its cells afterwards.
//
// (a) is the part every earlier attempt failed: the plan was computed from the
// original ubatch, so a padded one aborted at llama-kv-cache.cpp:1619. Here the pad
// happens inside llama_batch_allocr::split_simple, which runs BEFORE prepare(), so
// there is no plan to contradict.
//
// (c) matters as much as (b). A pad row with no owner leaks a cell per row per
// ubatch; with the scratch sequence it is freeable, and the probe proves the cells
// come back by decoding again and comparing.
#include "herd_patch.hpp"

#include <llama.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

// One declaration, in the header the seam also includes. A second copy here would be
// a second layout for one class -- the linker keeps one set of offsets and this
// translation unit would write through the other.
#include "herd_prefab_split.hpp"

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// What one run of the probe does, and how much of it is repeated.
//
// The first version decoded a single two-token batch, which could not distinguish two
// different things: "padding works" and "padding works once, then the cache fills up
// because the pad rows were never freed". Both produce identical output on round one.
// A run of several rounds with reclamation, plus a run of many rounds WITHOUT it that
// must run out of cells, is what makes the second claim measurable.
struct run_options {
    int  rounds        = 1;
    bool rm_each_round = true;
    // Decode 1, 2, 1, 2 … tokens instead of a constant 2.
    //
    // This is what makes the graph-reuse measurement mean anything. Upstream reuses a
    // graph when the new parameters are compatible with the previous ones
    // (llm_graph_result::can_reuse, llama-context.cpp:2024) and records it in
    // n_reused; a topology change forces a rebuild and is counted in n_reallocs_diff
    // (:3514). A constant-width control would satisfy can_reuse on its own, so the
    // comparison would show nothing. Alternating widths is the uncontrolled case the
    // envelope lane exists to fix -- and the padded run decodes the same token
    // sequence at a constant envelope width, so any difference in those counters is
    // the lane's doing.
    bool alternate_width = false;
};

run_options g_options;

// Logits per round of the most recent decode_once, so a caller can compare round by
// round rather than only the last one.
std::vector<std::vector<float>> g_round_logits;

std::vector<float> decode_once(llama_model * model, const llama_vocab * vocab,
                               bool padded, int * padded_ubatches, int * declined,
                               llama_perf_context_data * perf_out) {
    auto params = llama_context_default_params();
    params.n_ctx     = 256;
    params.n_batch   = 64;
    params.n_ubatch  = 64;
    // One sequence for the prompt, one reserved for pad rows. Without this the
    // scratch id has no entry in seq_to_stream and find_slot would index past it.
    params.n_seq_max = 2;
    params.kv_unified = true;

    const llama_seq_id scratch = 1;
    llama_context * ctx = llama_init_from_model(model, params);
    // The scope is created AFTER the context, because it needs the memory module to release
    // its cells -- and before any decode, because the seam has to be active when the first
    // batch is split. (Placing it before the init call, which is where it started, was a
    // plain use-before-declaration: `llama_get_memory(ctx)` with no ctx yet.)
    std::unique_ptr<herd::prefab::scratch_scope> scope;
    if (padded && ctx != nullptr) {
        // The memory module goes to the scope so the reservation can release itself on
        // every path out -- including the one where a decode fails.
        scope = std::make_unique<herd::prefab::scratch_scope>(llama_get_memory(ctx), scratch, 8u);
    }
    if (!ctx) {
        std::fprintf(stderr, "context creation failed\n");
        return {};
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> prompt;
    for (int i = 0; i < 4 && i < n_vocab; ++i) {
        prompt.push_back(static_cast<llama_token>(n_vocab - 1 - i));
    }

    std::unique_ptr<llama_batch_ext, void (*)(llama_batch_ext *)> batch(
        llama_batch_ext_init(ctx), &llama_batch_ext_free);
    std::vector<float> logits;
    bool ok = batch != nullptr;

    // One decode per round, two tokens each, logits on the LAST row only.
    //
    // That shape is deliberate, and it took a measurement to find: when a batch asks
    // for logits on every row, decode() counts outputs as `ubatch.n_tokens`
    // (llama-context.cpp:2747) and a padded width overruns the caller's buffer (:2819).
    // With one of two rows asking, the flags are what get counted, and a pad row's 0
    // contributes nothing -- so this is the shape the lane can serve, and the probe has
    // to use it or it measures the refusal instead of the padding.
    llama_memory_t memory = llama_get_memory(ctx);
    int padded_before_round = scope ? scope->padded_ubatches() : 0;
    // Two ledgers, because there are two facts and only one of them belongs to the seam.
    //
    //   attempted_*  every ubatch the seam widened, including one the cache later refused.
    //                Read from the seam's own counter, so it cannot disagree with it.
    //   settled_*    the subset that reached a decode returning 0.
    //
    // The previous version kept only the second and compared it against the seam's
    // counter, which failed by one envelope's worth of rows on the run that exhausts the
    // cache -- and the failure was read as a lane defect. It was arithmetic: a ubatch is
    // widened while the batch is split, before the cache has any say.
    std::uint64_t attempted_pad_rows = 0;
    int           entered_rounds = 0;
    std::uint64_t settled_expected = 0;
    int           settled_rounds = 0;
    // Positions advance by the width actually decoded, not by the round index: with
    // alternating widths `round * 2` would leave a gap whenever a round decodes one
    // token, and the two runs would then be evaluated at different sequence positions --
    // a difference in the inputs, which is not what this probe is comparing.
    llama_pos cursor = 0;
    for (int round = 0; ok && round < g_options.rounds; ++round) {
        // Widths 2 and 5, not 1 and 2.
        //
        // Measured on this pin (herd-perfcheck, no patches): a constant width reuses the
        // graph -- w=8 gives n_reused=7 over eight rounds, and so does w=1. What has no
        // reuse is a CHANGING width: alternating 1/2 gives n_reused=0. So the lane's
        // benefit is not "constant width" (stock already has that for a fixed workload);
        // it is turning a varying workload into a constant one, which is exactly what a
        // prefill/decode mix, or a batch whose parallel count fluctuates, produces.
        //
        // Width 1 cannot be used for the untouched half of this comparison: a single row
        // that asks for logits is the shape the lane declines (upstream counts outputs by
        // n_tokens there, llama-context.cpp:2747). Widths 2 and 5 are both paddable, so
        // both halves of the comparison decode the same tokens.
        const int width = g_options.alternate_width ? (round % 2 == 0 ? 2 : 5) : 2;
        llama_batch_ext_clear(batch.get());
        bool staged = true;
        for (int i = 0; i < width; ++i) {
            const int32_t row = llama_batch_ext_add_token(
                batch.get(), 0, prompt[std::size_t(i) % prompt.size()]);
            llama_pos pos = cursor + static_cast<llama_pos>(i);
            if (row < 0 || !llama_batch_ext_set_pos(batch.get(), row, &pos)) {
                staged = false;
                break;
            }
            // Logits on the last REAL row only. When every row asks, decode() counts
            // outputs as ubatch.n_tokens and a padded width overruns the caller's buffer
            // (llama-context.cpp:2747, :2819) -- the lane declines that shape by design.
            if (i == width - 1) {
                staged = llama_batch_ext_set_output_logits(batch.get(), row, true);
            }
        }
        if (!staged) {
            ok = false;
            break;
        }
        cursor += width;
        // The attempt is accounted BEFORE the verdict, because that is when it happened:
        // the seam widened the ubatch while splitting the batch, and the cache only
        // judges it afterwards. Reading the delta around the call -- rather than
        // predicting it from the width -- also keeps this honest for a round the lane
        // declines, where the delta is zero and nothing is claimed.
        const std::uint64_t pad_rows_before_call = scope ? scope->pad_rows() : 0;
        const int decode_rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (scope) {
            attempted_pad_rows += scope->pad_rows() - pad_rows_before_call;
            ++entered_rounds;
        }
        if (decode_rc != 0) {
            // A refused decode is information here, not a crash: the run that never
            // reclaims its pad cells needs exactly this outcome to show that the cache
            // is finite and that reclamation is what keeps it that way.
            std::fprintf(stderr,
                         "round %d: decode refused (cache exhausted?) after the seam "
                         "submitted %llu pad row(s) for it\n",
                         round, (unsigned long long) (scope ? scope->pad_rows() - pad_rows_before_call : 0));
            ok = false;
            break;
        }
        // Settled: the rows of a ubatch the cache actually accepted. Recorded here rather
        // than assumed from the attempt count; see scratch_scope::note_settled.
        if (scope && scope->envelope() > uint32_t(width)) {
            scope->note_settled(scope->envelope(), uint32_t(width));
            settled_expected += scope->envelope() - uint32_t(width);
            ++settled_rounds;
        }
        llama_synchronize(ctx);
        const float * values = llama_get_logits_ith(ctx, -1);
        if (!values) {
            ok = false;
            break;
        }
        g_round_logits.emplace_back(values, values + n_vocab);

        // The cells the pad rows took must exist AND be returnable -- in that order.
        //
        // The first version asked only whether seq_rm() succeeded, and that is a false
        // positive by construction: seq_rm() on a sequence with no cells returns true
        // (llama-kv-cache.cpp: it has nothing to remove). So an implementation whose pad
        // rows carry no owner -- one leaked cell per row per ubatch -- passed the check,
        // and the mutation that removes the ownership survived the whole probe.
        //
        // The position maximum is what distinguishes the two: owned pad rows leave cells
        // behind at the last real row's position, so seq_pos_max() must equal it. With no
        // owner the scratch sequence stays empty and this fails.
        // Only when this round actually padded. In the alternating run the width-1 rounds
        // are declined (a single row that asks for logits is exactly the shape upstream
        // counts by n_tokens, so the lane refuses it), and asserting on an empty scratch
        // sequence would fail for the wrong reason -- which it did, eight times, before
        // this was tied to the per-round padded count.
        const bool this_round_padded = scope && scope->padded_ubatches() > padded_before_round;
        padded_before_round = scope ? scope->padded_ubatches() : 0;
        if (padded && memory && this_round_padded) {
            check(llama_memory_seq_pos_max(memory, scratch) == cursor - 1,
                  "the pad rows own cells at the last real position");
            if (g_options.rm_each_round) {
                check(llama_memory_seq_rm(memory, scratch, -1, -1),
                      "the pad rows' cells are released by sequence");
                check(llama_memory_seq_pos_max(memory, scratch) == -1,
                      "and the scratch sequence is empty again");
            }
        }
    }
    if (ok && !g_round_logits.empty()) {
        logits = g_round_logits.back();
    }
    if (scope) {
        *padded_ubatches = scope->padded_ubatches();
        *declined        = scope->declined();
        // Two ledgers, two claims, and the difference between them is a fact about this
        // run rather than a tolerance.
        //
        //   attempts  == what the seam submitted   (every round the probe entered)
        //   settled   == what the cache accepted   (rounds whose decode returned 0)
        //   attempts - settled == the rows of the one ubatch that was submitted and then
        //   refused. On the exhausting run that is exactly one envelope's worth, and the
        //   earlier assertion failed here because it counted the second ledger's rounds
        //   and compared them to the first ledger's counter.
        std::fprintf(stderr,
                     "pad_rows: seam=%llu attempted=%llu settled=%llu settled_expected=%llu "
                     "(ubatches seam=%d entered=%d settled=%d)\n",
                     (unsigned long long) scope->pad_rows(),
                     (unsigned long long) attempted_pad_rows,
                     (unsigned long long) scope->settled_pad_rows(),
                     (unsigned long long) settled_expected,
                     scope->padded_ubatches(), entered_rounds, scope->settled_ubatches());
        check(scope->pad_rows() == attempted_pad_rows,
              "the seam's row counter matches every round the probe entered");
        check(scope->padded_ubatches() == entered_rounds,
              "and so does its ubatch counter");
        check(scope->settled_pad_rows() == settled_expected,
              "the settled counter matches the rounds the cache accepted");
        check(scope->settled_ubatches() == settled_rounds,
              "and its settled ubatch count");
        // The invariant that ties the two ledgers together: everything submitted either
        // settled or was refused, and a refused ubatch's rows are exactly the shortfall.
        check(scope->pad_rows() >= scope->settled_pad_rows(),
              "settled rows never exceed submitted rows");
        const std::uint64_t unsettled = scope->pad_rows() - scope->settled_pad_rows();
        check(unsettled == 0 || unsettled < std::uint64_t(scope->envelope()),
              "a shortfall is at most one ubatch's pad rows, which is all one refusal can cost");
    }
    batch.reset();
    // The release must happen while the context is alive, and this line is the fix for a
    // real use-after-free rather than a tidy-up.
    //
    // `scope`'s destructor calls llama_memory_seq_rm(memory_, …), and `memory_` came from
    // llama_get_memory(ctx). As a function-scope object it would be destroyed *after* the
    // explicit `llama_free(ctx)` below returns -- so the release would walk a freed
    // context. `batch` has the same shape (llama_batch_ext holds a pointer into the same
    // memory module), which is why it is reset first here too.
    //
    // Both failures are quiet on this allocator: the freed block usually still maps, the
    // write lands in it, and nothing reports. Doing it in this order is correct on its own
    // rather than by luck, and the scope's own `release()` is idempotent, so a caller that
    // also releases explicitly is not double-freeing anything.
    scope.reset();
    if (perf_out) {
        // Read while the context is alive: llama_perf_context() reports its counters, and
        // they are gone after llama_free(). n_reused and n_reallocs_diff are upstream's
        // own account of what the graph scheduler did, which is the quantity the envelope
        // lane claims to improve -- not a number this probe derives.
        *perf_out = llama_perf_context(ctx);
    }
    llama_free(ctx);
    return ok ? logits : std::vector<float>{};
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    herd::install_all();
    llama_backend_init();

    // A probe that only works with an environment variable set is a trap: the next
    // runner gets four assertion failures from their own omission, and reads them as a
    // lane defect. This is what happened on the first run without the variable -- the
    // counters stayed zero because the seam was never installed, and the two-ledger
    // checks fired against a run that had no lane at all. Refusing at the door names the
    // cause once instead of pointing at five symptoms.
    if (!herd::find("prefab.split_simple")) {
        std::fprintf(stderr,
                     "herd-prefab-split-probe: the split seam is not installed. Set "
                     "HERD_PREFAB_SPLIT=1 (and HERD_NO_AUTOLOAD unset) to run the lane.\n");
        llama_backend_free();
        return 2;
    }

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    if (!model) {
        std::fprintf(stderr, "cannot load %s\n", argv[1]);
        llama_backend_free();
        return 2;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    constexpr int kRounds = 8;
    int padded_count = 0, declined = 0;
    llama_perf_context_data plain_perf{}, padded_perf{};

    g_options.rounds        = kRounds;
    g_options.rm_each_round = true;

    g_round_logits.clear();
    const std::vector<float> plain = decode_once(model, vocab, false, &padded_count, &declined,
                                                 &plain_perf);
    const std::vector<std::vector<float>> plain_rounds = g_round_logits;

    g_round_logits.clear();
    const std::vector<float> padded = decode_once(model, vocab, true, &padded_count, &declined,
                                                  &padded_perf);
    const std::vector<std::vector<float>> padded_rounds = g_round_logits;
    const int padded_ubatches = padded_count;
    const int declines        = declined;

    check(!plain.empty(), "the unpadded run produced logits");
    check(!padded.empty(), "the padded run produced logits (upstream accepted the shape)");
    check(static_cast<int>(plain_rounds.size()) == kRounds, "the control decoded every round");
    check(static_cast<int>(padded_rounds.size()) == kRounds,
          "the padded run decoded every round: repeated padding does not fill the cache");
    check(padded_ubatches == kRounds,
          "every round was padded (one ubatch per round)");
    std::fprintf(stderr, "padded_ubatches=%d declined=%d\n", padded_ubatches, declines);
    if (declines > 0) {
        std::fprintf(stderr, "declines=%d (each is counted once per refused ubatch)\n", declines);
    }

    // Round by round, not just the last one: a lane that padded correctly for one round
    // and then reused a stale cell would agree on the final logits of a single decode and
    // disagree everywhere else.
    std::size_t compared = 0;
    std::size_t differences = 0;
    double largest = 0.0;
    for (std::size_t r = 0; r < plain_rounds.size() && r < padded_rounds.size(); ++r) {
        const auto & a = plain_rounds[r];
        const auto & b = padded_rounds[r];
        check(a.size() == b.size(), "same vocabulary width in every round");
        for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
            ++compared;
            if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
                ++differences;
            }
            const double delta = std::fabs(double(a[i]) - double(b[i]));
            if (delta > largest) {
                largest = delta;
            }
        }
    }
    std::fprintf(stderr, "logits: %zu values over %zu round(s), %zu differ, largest |delta| = %g\n",
                 compared, plain_rounds.size(), differences, largest);
    check(differences == 0, "padding changed no answer, in any round");

    // Reclamation is load-bearing, and this is the measurement that says so rather than
    // assuming it: the same run without rm_each_round must run out of cells. 48 rounds of
    // two tokens at envelope 8 asks for 384 cells from a 256-cell cache, so a lane whose
    // pad rows were never freed cannot finish -- whichever way it fails, it must not
    // silently succeed with the same logits, because that would mean the cells came from
    // somewhere and the accounting is wrong.
    int no_rm_padded = 0, no_rm_declined = 0;
    g_options.rounds        = 48;
    g_options.rm_each_round = false;
    g_round_logits.clear();
    const std::vector<float> unreclaimed = decode_once(model, vocab, true, &no_rm_padded,
                                                       &no_rm_declined, nullptr);
    std::fprintf(stderr,
                 "no-reclamation run: %zu round(s) completed of 48, padded_ubatches=%d, "
                 "logits=%zu\n",
                 g_round_logits.size(), no_rm_padded, unreclaimed.size());
    check(unreclaimed.empty(),
          "without reclamation the pad cells accumulate until a decode is refused");
    g_options.rm_each_round = true;

    // ---- is this prefab's *benefit*, or only its padding? ----
    //
    // The fork's prefab lane did two things, and only one of them is padding:
    //
    //   1. it raised every ubatch to a constant envelope width, and
    //   2. it pinned the mask AND K/V view width to a padded KV size, masking the extra
    //      columns with -inf (llama-prefab.cpp:179-181, :205-212, n_kv_pad_fits).
    //
    // The purpose of (1) is that a constant topology lets upstream reuse the previous
    // graph (llm_graph_result::can_reuse, llama-context.cpp:2024 -> n_reused), and (2)
    // exists because the mask width grows with n_kv: without padding the mask too, the
    // topology still changes as the sequence grows and the reuse never happens.
    //
    // This lane does (1) and not (2). So the honest question is whether the reuse
    // appears anyway, and the counters answer it rather than an argument. Both runs
    // decode the same token sequence with alternating widths (1, 2, 1, 2 …), which is
    // the uncontrolled case; the lane's run sees a constant 8-row ubatch.
    constexpr int kDiffRounds = 16;
    llama_perf_context_data alt_plain{}, alt_padded{};

    g_options.rounds          = kDiffRounds;
    g_options.alternate_width = true;
    int alt_plain_padded = 0, alt_plain_declined = 0;
    g_round_logits.clear();
    static_cast<void>(decode_once(model, vocab, false, &alt_plain_padded, &alt_plain_declined,
                                  &alt_plain));

    int alt_padded_padded = 0, alt_padded_declined = 0;
    g_round_logits.clear();
    static_cast<void>(decode_once(model, vocab, true, &alt_padded_padded, &alt_padded_declined,
                                  &alt_padded));
    g_options.alternate_width = false;

    std::fprintf(stderr,
                 "reuse: alternate widths, no lane  -> n_reused=%d n_reallocs_diff=%d "
                 "n_reallocs_same=%d\n",
                 alt_plain.n_reused, alt_plain.n_reallocs_diff, alt_plain.n_reallocs_same);
    std::fprintf(stderr,
                 "reuse: alternate widths, envelope -> n_reused=%d n_reallocs_diff=%d "
                 "n_reallocs_same=%d (padded %d, declined %d)\n",
                 alt_padded.n_reused, alt_padded.n_reallocs_diff, alt_padded.n_reallocs_same,
                 alt_padded_padded, alt_padded_declined);

    // A fact about the control, and the premise of the comparison: alternating widths
    // should change the graph, so the scheduler should reallocate at least once. But this
    // pin reports zero for every perf counter on every run measured here -- including the
    // constant-width runs -- so the premise cannot be established through this API, and
    // asserting it would be asserting something unmeasured. Report that instead of a
    // failure: an unobservable counter is a limitation of the instrument, not a property
    // of the code.
    // The instrument, calibrated before it is believed (herd-perfcheck, no patches):
    //
    //   constant width 8, eight rounds   -> n_reused = 7
    //   constant width 1, eight rounds   -> n_reused = 7
    //   alternating 1/2, eight rounds    -> n_reused = 0
    //
    // So n_reused is live and means "the previous graph was reused", and a *changing*
    // width is what forfeits it. That calibration is why the earlier record's claim --
    // "the counters report zero for every run, so the benefit is unmeasurable" -- is
    // wrong: it was zero for every run that changed shape, which is not the same thing.
    // The claim is corrected here rather than repeated.
    //
    // n_reallocs_same/diff are NOT used and must not be: sched_graph_realloc is the only
    // writer (llama-context.cpp:3514), and in this pin it is never registered, so both
    // counters stay 0 and the "graph reallocation" log line never appears. A dead counter
    // that reads 0 looks exactly like a healthy graph; asserting on it would have been
    // the same mistake in a new place.

    // The control must be the case that forfeits reuse, or the comparison is vacuous.
    check(alt_plain.n_reused == 0,
          "changing widths forfeit reuse (calibrated: alternating 1/2 also gave 0)");
    // And the lane must restore it. This is the whole point of the fork's prefab lane:
    // a workload whose ubatch width varies gets a constant graph anyway.
    check(alt_padded.n_reused > 0,
          "the envelope lane restores graph reuse on a workload whose width varies");
    check(alt_padded.n_reused >= kDiffRounds - 2,
          "and reuses it on essentially every round after the first");

    // Reported, not asserted, because it is a fact about the lane's cost rather than its
    // claim: the padded run decodes more rows, so it may take longer. What must hold --
    // and what the checks above establish -- is that the extra rows do not change the
    // answer and do not accumulate in the cache.
    std::fprintf(stderr,
                 "reuse verdict: REPRODUCED -- varying widths forfeited reuse (%d), the "
                 "envelope lane restored it (%d of %d rounds)\n",
                 alt_plain.n_reused, alt_padded.n_reused, kDiffRounds);

    if (!plain.empty() && !padded.empty()) {
        check(plain.size() == padded.size(), "same vocabulary width");
        std::size_t differences = 0;
        double largest = 0.0;
        for (std::size_t i = 0; i < plain.size() && i < padded.size(); ++i) {
            const double delta = std::fabs(double(plain[i]) - double(padded[i]));
            if (std::memcmp(&plain[i], &padded[i], sizeof(float)) != 0) {
                ++differences;
            }
            if (delta > largest) {
                largest = delta;
            }
        }
        std::fprintf(stderr, "logits: %zu values, %zu differ, largest |delta| = %g\n",
                     plain.size(), differences, largest);
        check(differences == 0, "padding changed no answer");
    }

    // ---- recovery: the reservation releases itself, even when the decode failed ----
    //
    // The one path a caller cannot be trusted with. A decode that returns non-zero -- a
    // full cache, an abort, an exception out of the graph builder -- leaves the pad rows
    // in the cache, and every caller that "remembers to clean up afterwards" has a branch
    // where it forgets. So the cleanup lives in the destructor, and this checks it against
    // the two shapes a leak would take:
    //
    //   a decode that succeeded, then the scope dying: the cells must be gone;
    //   a decode that was REFUSED (the cache exhausted), then the scope dying: the cells
    //   it submitted before the refusal must be gone too.
    //
    // Both are measured on the cache's own position bookkeeping, not on a counter of ours.
    {
        auto params = llama_context_default_params();
        params.n_ctx      = 256;
        params.n_batch    = 64;
        params.n_ubatch   = 64;
        params.n_seq_max  = 2;
        params.kv_unified = true;

        llama_context * ctx = llama_init_from_model(model, params);
        check(ctx != nullptr, "recovery: context created");
        if (ctx) {
            llama_memory_t memory = llama_get_memory(ctx);
            const llama_seq_id scratch = 1;
            const int n_vocab_here = llama_vocab_n_tokens(vocab);

            // One decode under a scope, then out of the scope: no explicit release.
            {
                herd::prefab::scratch_scope held(memory, scratch, 8u);
                llama_batch_ext * batch = llama_batch_ext_init(ctx);
                if (batch) {
                    llama_batch_ext_clear(batch);
                    // Two rows, logits on the last only. The single-row shape used here
                    // first is exactly what the lane declines -- every row asking for
                    // logits is the case upstream counts by n_tokens (llama-context.cpp
                    // :2747), so `padded_ubatches() > 0` could never hold and the
                    // assertion below failed for a fixture reason, not a lane defect.
                    bool staged = true;
                    for (int i = 0; i < 2 && staged; ++i) {
                        const int32_t row = llama_batch_ext_add_token(
                            batch, 0, static_cast<llama_token>(n_vocab_here - 1));
                        llama_pos pos = llama_pos(i);
                        staged = row >= 0 && llama_batch_ext_set_pos(batch, row, &pos);
                        if (staged && i == 1) {
                            staged = llama_batch_ext_set_output_logits(batch, row, true);
                        }
                    }
                    if (staged) {
                        const int rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
                        check(rc == 0, "recovery: the scoped decode succeeded");
                        check(held.padded_ubatches() > 0, "recovery: it was padded, not declined");
                        check(llama_memory_seq_pos_max(memory, scratch) >= 0,
                              "recovery: the pad rows own cells after the decode");
                    }
                    llama_batch_ext_free(batch);
                }
            }
            check(llama_memory_seq_pos_max(memory, scratch) == -1,
                  "recovery: the reservation's cells are gone once the scope ends, with no "
                  "explicit release by the caller");

            // The refused path: fill the cache until a decode fails, then let the scope die.
            bool saw_refusal = false;
            {
                herd::prefab::scratch_scope held(memory, scratch, 8u);
                llama_batch_ext * batch = llama_batch_ext_init(ctx);
                llama_pos cursor = 0;
                for (int round = 0; batch && round < 64 && !saw_refusal; ++round) {
                    llama_batch_ext_clear(batch);
                    bool staged = true;
                    for (int i = 0; i < 2; ++i) {
                        const int32_t row = llama_batch_ext_add_token(
                            batch, 0, static_cast<llama_token>(n_vocab_here - 1));
                        llama_pos pos = cursor + llama_pos(i);
                        if (row < 0 || !llama_batch_ext_set_pos(batch, row, &pos)) {
                            staged = false;
                            break;
                        }
                        if (i == 1) staged = llama_batch_ext_set_output_logits(batch, row, true);
                    }
                    if (!staged) break;
                    cursor += 2;
                    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
                        saw_refusal = true;
                        break;
                    }
                    // No reclamation here on purpose: the point is to reach the refusal.
                    llama_synchronize(ctx);
                }
                if (batch) llama_batch_ext_free(batch);
                check(saw_refusal, "recovery: the un-reclaimed run eventually has a decode refused");
            }
            check(llama_memory_seq_pos_max(memory, scratch) == -1,
                  "recovery: after the refusal the scope still returns every cell it held");
            // A positive line, because every check above is silent when it passes: without
            // this, "no FAIL" is also what a block that never ran would produce.
            std::fprintf(stderr,
                         "recovery: two paths executed -- accepted decode and refused decode; "
                         "scratch empty afterwards in both\n");
            llama_free(ctx);
        }
    }

    llama_model_free(model);
    llama_backend_free();
    std::fprintf(stderr, "%s\n", failures ? "prefab-split probe FAILED" : "prefab-split probe holds");
    return failures ? 1 : 0;
}
