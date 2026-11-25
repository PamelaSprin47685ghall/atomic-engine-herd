// herd_prefab_split_scope_test.cpp — the reservation's own contract, on a real cache.
//
// The probe exercises the lane end to end; this file exercises the *guard*: what a
// scratch_scope promises about the cells it borrows, and what it does when the promise
// cannot be kept. Each case names the failure it prevents.
//
// Requires the seam (HERD_PREFAB_SPLIT=1) because a reservation with nothing to pad has
// no cells to release -- the interesting states are all reachable only through a real
// decode.
// Two of the three expectations below are observable and one is not, and the difference
// matters more than the passing run.
//
// Observable: release() empties a sequence that held cells; a second release is a no-op;
// a scope whose explicit release already ran is not freed twice by its destructor; an
// envelope wider than the cache produces a refused decode rather than a smaller answer.
//
// NOT observable on this pin: the failing branch of release() -- `seq_rm` refusing, or the
// sequence still holding cells afterwards. A mutation that replaced the postcondition read
// with `empty = true` still passed the whole file, because in every reachable state the
// clear simply works. So that guard is defensive, not covered, and calling it "verified"
// would be the same shape of claim this project keeps finding elsewhere.
#include "herd_patch.hpp"
#include "herd_prefab_split.hpp"

// The statistics ABI. This file is the only one that can test the REAL provider -- it links
// `herd_prefab_split.cpp` itself, so the provider under test is the one a driver's
// `--prefab-stats` reaches, not a stand-in.
#include <herd/upstream-ext.h>

#include <sys/resource.h>
#include <unistd.h>

#include <fstream>

#include <llama.h>

#include <cstdio>
#include <cstring>
#include <memory>

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

llama_context * make_ctx(llama_model * model, uint32_t n_ctx, uint32_t n_seq_max) {
    auto params = llama_context_default_params();
    params.n_ctx      = n_ctx;
    params.n_batch    = 64;
    params.n_ubatch   = 64;
    params.n_seq_max  = n_seq_max;
    params.kv_unified = true;
    return llama_init_from_model(model, params);
}

// Two rows, logits on the last only: the smallest shape the lane can serve. (One row that
// asks for logits is declined by design -- upstream counts outputs by n_tokens then.)
bool decode_pair(llama_context * ctx, llama_token token, llama_pos first) {
    std::unique_ptr<llama_batch_ext, void (*)(llama_batch_ext *)> batch(
        llama_batch_ext_init(ctx), &llama_batch_ext_free);
    if (!batch) {
        return false;
    }
    llama_batch_ext_clear(batch.get());
    for (int i = 0; i < 2; ++i) {
        const int32_t row = llama_batch_ext_add_token(batch.get(), 0, token);
        llama_pos pos = first + llama_pos(i);
        if (row < 0 || !llama_batch_ext_set_pos(batch.get(), row, &pos)) {
            return false;
        }
        if (i == 1 && !llama_batch_ext_set_output_logits(batch.get(), row, true)) {
            return false;
        }
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
        return false;
    }
    llama_synchronize(ctx);
    return true;
}

// The core promise: after release(), the sequence holds nothing -- and that is read from
// the cache, not from the return value. A release that reports success while cells remain
// is the leak this class exists to prevent, and it is exactly what a "seq_rm returned
// true" check cannot distinguish (seq_rm is also true for an empty sequence).
void release_empties_the_sequence_and_is_idempotent(llama_model * model, const llama_vocab * vocab) {
    llama_context * ctx = make_ctx(model, 256, 2);
    check(ctx != nullptr, "context for the release contract");
    if (!ctx) {
        return;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    const llama_seq_id scratch = 1;
    const llama_token token = llama_token(llama_vocab_n_tokens(vocab) - 1);

    {
        herd::prefab::scratch_scope held(memory, scratch, 8u);
        check(decode_pair(ctx, token, 0), "a scoped decode succeeds");
        check(held.padded_ubatches() > 0, "and it was padded, so the reservation holds cells");
        check(llama_memory_seq_pos_max(memory, scratch) >= 0,
              "the scratch sequence owns cells before the release");

        check(held.release(), "release reports success");
        check(held.released(), "and records that it happened");
        check(llama_memory_seq_pos_max(memory, scratch) == -1,
              "and the sequence is empty afterwards -- the postcondition, read from the cache");

        // Idempotence. A second release must not report failure (there is nothing left to
        // do) and must not resurrect anything.
        check(held.release(), "a second release is a no-op success");
        check(llama_memory_seq_pos_max(memory, scratch) == -1, "and the sequence stays empty");
    }

    // Explicit release, then the destructor: the destructor's release() sees the flag and
    // returns without a second seq_rm. Nothing may be freed twice and nothing may come
    // back.
    {
        herd::prefab::scratch_scope held(memory, scratch, 8u);
        check(decode_pair(ctx, token, 2), "a second scoped decode succeeds");
        check(held.release(), "released explicitly");
    }
    check(llama_memory_seq_pos_max(memory, scratch) == -1,
          "after the destructor the sequence is still empty (no double release, no revival)");

    // The context still works, which is the practical meaning of "the cells came back".
    {
        herd::prefab::scratch_scope held(memory, scratch, 8u);
        check(decode_pair(ctx, token, 4), "the context decodes again after two release paths");
        check(held.padded_ubatches() > 0, "still with padding");
    }
    llama_free(ctx);
}

// An envelope wider than the cache. The seam cannot see n_ctx, so it pads anyway; what
// matters is that the result is a REFUSED decode rather than silent damage. This is the
// loudness contract for a mis-sized reservation: the caller reads a non-zero return, not a
// smaller answer.
void an_oversized_envelope_fails_loudly(llama_model * model, const llama_vocab * vocab) {
    llama_context * ctx = make_ctx(model, 256, 2);
    check(ctx != nullptr, "context for the oversized-envelope case");
    if (!ctx) {
        return;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    const llama_token token = llama_token(llama_vocab_n_tokens(vocab) - 1);

    bool refused = false;
    {
        // 4096 rows into a 256-cell cache. find_slot cannot collect the indices, so the
        // decode fails; the reservation then has nothing owned to release.
        herd::prefab::scratch_scope held(memory, llama_seq_id(1), 4096u);
        refused = !decode_pair(ctx, token, 0);
        check(refused, "an envelope larger than the cache is refused, not silently truncated");
        check(held.padded_ubatches() > 0,
              "the seam did attempt the widening (so the refusal came from the cache, not the lane)");
        check(held.settled_ubatches() == 0, "and no ubatch settled");
        // The destructor runs here and must not report a leak it cannot fix; if it does,
        // release() returns false and says so on stderr rather than claiming success.
    }
    // Nothing further to assert about `refused` here: it was checked above, and asserting
    // its negation would be a tautology written as a test (it was, on the first run).
    llama_free(ctx);
}

// A reservation on a sequence that already holds cells must not silently adopt them: the
// cells belong to whoever wrote them, and clearing them would destroy another sequence's
// history. The lane's contract is "the scratch sequence is empty before use", and this
// checks that violating it is visible rather than absorbed.
void a_preoccupied_scratch_sequence_is_not_adopted(llama_model * model, const llama_vocab * vocab) {
    llama_context * ctx = make_ctx(model, 256, 3);
    check(ctx != nullptr, "context for the preoccupied case");
    if (!ctx) {
        return;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    const llama_token token = llama_token(llama_vocab_n_tokens(vocab) - 1);

    // Sequence 2 is used as an ORDINARY sequence first: two token decodes, no padding.
    {
        std::unique_ptr<llama_batch_ext, void (*)(llama_batch_ext *)> batch(
            llama_batch_ext_init(ctx), &llama_batch_ext_free);
        check(batch != nullptr, "batch for the ordinary sequence");
        if (batch) {
            llama_batch_ext_clear(batch.get());
            for (int i = 0; i < 2; ++i) {
                const int32_t row = llama_batch_ext_add_token(batch.get(), 2, token);
                llama_pos pos = llama_pos(i);
                check(row >= 0 && llama_batch_ext_set_pos(batch.get(), row, &pos),
                      "an ordinary row is staged");
                check(llama_batch_ext_set_output_logits(batch.get(), row, true),
                      "with logits");
            }
            check(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0,
                  "the ordinary decode succeeds");
            llama_synchronize(ctx);
        }
    }
    const llama_pos occupied = llama_memory_seq_pos_max(memory, 2);
    check(occupied >= 0, "sequence 2 holds cells of its own");

    // Now reserve THAT sequence for pad rows.
    //
    // This case used to assert the opposite: that the scoped decode ran, that release()
    // cleared the sequence, and that the history was therefore gone. It was written as a
    // documentation of the caller's obligation, and it documented a defect -- the
    // reservation destroyed data it was never given, through its cleanup path, and the
    // suite called that correct.
    //
    // What it asserts now is the fail-closed rule: an occupied sequence is refused at
    // open, nothing is padded, and release() leaves the cells alone. Three separate
    // claims, because each is a place the old behaviour could come back:
    //   the reservation reports itself unusable (so a caller can check before relying);
    //   it never installed as the seam's scope (so no ubatch is padded at all);
    //   release() declines instead of clearing (so the history survives).
    {
        herd::prefab::scratch_scope held(memory, llama_seq_id(2), 8u);
        check(!held.usable(), "a reservation on an occupied sequence is refused");
        check(held.occupied_at_open(), "and reports why");
        check(held.memory() == memory,
              "the reservation remembers the context's memory it was opened for");

        // The decode still succeeds -- the lane simply is not installed, so this is the
        // ordinary path -- but it must not have padded anything.
        check(decode_pair(ctx, token, llama_pos(occupied + 1)),
              "the ordinary decode still runs, without the lane");
        check(held.padded_ubatches() == 0, "nothing was padded by a refused reservation");
        check(held.release() == false, "release declines: the cells are not this object's");
    }
    const llama_pos after = llama_memory_seq_pos_max(memory, 2);
    check(after >= occupied,
          "the history that was there is intact, not cleared by a refused reservation");
    check(llama_memory_seq_pos_max(memory, 2) != -1,
          "and the sequence is still occupied afterwards");

    llama_free(ctx);
}

// The cross-context risk, as a test rather than as a paragraph in a header.
//
// A reservation names one context's memory. Until the decode entry points were hooked,
// the seam had no way to ask whose ubatch it was splitting -- `llama_batch_allocr` carries
// no context (llama-batch.h:167) -- so a reservation held open across a second context's
// decode would pad THAT context's ubatches into this context's scratch sequence. The
// second context's own history, in its own cache, would then be destroyed when this
// reservation released, having never been checked for emptiness.
//
// Four things must hold, and each is a distinct assertion below:
//
//   1. the second context still decodes, with the lane not applied to it;
//   2. nothing was written into the second context's cache at all -- its scratch sequence
//      is still empty, which is the difference between "declined" and "padded elsewhere";
//   3. the reservation still works for its own context (a guard that refused everything
//      would satisfy 1 and 2);
//   4. the second context's history survives this reservation's release.
void a_reservation_does_not_pad_another_contexts_decode(llama_model * model,
                                                       const llama_vocab * vocab) {
    llama_context * a = make_ctx(model, 256, 2);
    llama_context * b = make_ctx(model, 256, 2);
    check(a != nullptr && b != nullptr, "cross-context: both contexts exist");
    if (!a || !b) {
        if (a) llama_free(a);
        if (b) llama_free(b);
        return;
    }
    llama_memory_t memory_a = llama_get_memory(a);
    llama_memory_t memory_b = llama_get_memory(b);
    check(memory_a != memory_b, "cross-context: the two contexts have distinct memories");

    const llama_token token = llama_token(llama_vocab_n_tokens(vocab) - 1);
    const llama_seq_id scratch = 1;

    // B's own history first, on its sequence 0, with no reservation open: this is what
    // must survive later.
    check(decode_pair(b, token, 0), "cross-context: B decodes its own history");
    const llama_pos b_history = llama_memory_seq_pos_max(memory_b, 0);
    check(b_history >= 0, "cross-context: B holds cells of its own");

    {
        herd::prefab::scratch_scope held(memory_a, scratch, 8u);
        check(held.usable(), "cross-context: A's reservation opens (A's scratch is empty)");

        // The decode that must not be padded: same thread, different context.
        check(decode_pair(b, token, llama_pos(b_history + 1)),
              "cross-context: B decodes while A's reservation is open");
        check(held.binding_refusals() >= 1,
              "cross-context: the seam refused to pad another context's ubatch");
        check(held.padded_ubatches() == 0,
              "cross-context: and padded nothing at all for it");

        // The decisive one. A declined padding writes no pad rows; a mis-bound lane would
        // have written them into B's cache under A's scratch id, where B's own sequence 1
        // lives. An empty sequence 1 in B is the difference between the two.
        check(llama_memory_seq_pos_max(memory_b, scratch) == -1,
              "cross-context: nothing was written into B's cache (its scratch sequence is empty)");
        check(llama_memory_seq_pos_max(memory_b, 0) >= b_history,
              "cross-context: B's history is untouched by the refusal");

        // And the lane still serves its own context. Without this, a hook that refused
        // every decode would pass everything above.
        check(decode_pair(a, token, 0), "cross-context: A decodes with its own reservation");
        check(held.padded_ubatches() > 0,
              "cross-context: A's ubatches were padded (the lane is bound, not disabled)");
        check(llama_memory_seq_pos_max(memory_a, scratch) >= 0,
              "cross-context: the pad rows landed in A's cache");
    }

    // The release ran in the destructor, against A's memory. B's must be untouched.
    check(llama_memory_seq_pos_max(memory_a, scratch) == -1,
          "cross-context: A's scratch sequence was released");
    check(llama_memory_seq_pos_max(memory_b, scratch) == -1,
          "cross-context: and B's scratch sequence was never touched");
    check(llama_memory_seq_pos_max(memory_b, 0) >= b_history,
          "cross-context: B's history survived A's reservation entirely");

    llama_free(a);
    llama_free(b);
}

// A reservation that has been released is over -- and the seam has to know that, which
// is not the same thing as the caller remembering to stop using it.
//
// Measured before this case existed: an explicit release() returned true and emptied the
// sequence, the next decode padded again (the seam only tested "is there a scope"), and
// the destructor's release() returned immediately on `released_` -- so the second set of
// pad rows was never cleared. The leak's size is pad_rows() minus settled_pad_rows(), and
// it grew on a run that had already reported a successful release.
//
// Three assertions, because the defect had three faces: padding restarts, the cells stay,
// and opening the next reservation aborts on a stale registration.
void a_released_reservation_stops_padding(llama_model * model, const llama_vocab * vocab) {
    llama_context * ctx = make_ctx(model, 256, 2);
    check(ctx != nullptr, "released-reservation: context exists");
    if (!ctx) {
        return;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    const llama_seq_id scratch = 1;
    const llama_token token = llama_token(llama_vocab_n_tokens(vocab) - 1);

    std::int64_t rows_after_release = 0;
    {
        herd::prefab::scratch_scope held(memory, scratch, 8u);
        check(held.usable(), "released-reservation: it opens (the sequence is empty)");
        check(decode_pair(ctx, token, 0), "released-reservation: a scoped decode succeeds");
        check(held.padded_ubatches() > 0, "released-reservation: it padded, so there is something to stop");
        check(held.release(), "released-reservation: the explicit release succeeds");
        check(held.released() && held.finished(), "released-reservation: and it is finished, not merely released");
        check(llama_memory_seq_pos_max(memory, scratch) == -1, "released-reservation: the cells are back");

        rows_after_release = static_cast<std::int64_t>(held.pad_rows());

        // The decode that used to restart the lane.
        check(decode_pair(ctx, token, 2), "released-reservation: the context decodes again");
        check(decode_pair(ctx, token, 4), "released-reservation: and again");
        check(static_cast<std::int64_t>(held.pad_rows()) == rows_after_release,
              "released-reservation: no pad row was added after the release");
        check(llama_memory_seq_pos_max(memory, scratch) == -1,
              "released-reservation: and the scratch sequence stayed empty -- nothing leaked");

        // The destructor runs here against an already-finished scope. It must not touch a
        // newer scope's registration (there is none yet) and must not resurrect anything.
    }
    check(llama_memory_seq_pos_max(memory, scratch) == -1,
          "released-reservation: the destructor added nothing after the release");

    // The registration must be free now. Before the fix this aborted with "nesting is
    // refused", which is a false diagnosis: the earlier reservation had ended.
    {
        herd::prefab::scratch_scope next(memory, scratch, 8u);
        check(next.usable(), "released-reservation: the next reservation opens without a nesting abort");
        check(decode_pair(ctx, token, 6), "released-reservation: and it pads its own work");
        check(next.padded_ubatches() > 0, "released-reservation: the lane is in force again");
        check(next.release(), "released-reservation: and it releases");
    }
    check(llama_memory_seq_pos_max(memory, scratch) == -1,
          "released-reservation: the second reservation cleaned up after itself");

    llama_free(ctx);
}

// An envelope the process cannot allocate: a declined padding, not a dead process.
//
// Distinct from `an_oversized_envelope_fails_loudly`, and the distinction is the point.
// There, 4096 rows fit in memory and the cache refuses them at find_slot -- the seam built
// its copy and the failure came from upstream. Here the copy itself cannot be built, so the
// failure happens inside the seam, on the one path upstream never throws from: this code
// runs inside `llama_batch_allocr::split_simple`, called by `llama_context::decode`. An
// escaping `std::bad_alloc` unwinds through `llama_process`, and uncaught it terminates the
// process.
//
// How the failure is made deterministic, and why not by asking for 16 GB and hoping:
// under Linux overcommit a huge `operator new` can *succeed*, and the following
// `resize`'s page-touching writes are what die -- via the OOM killer, which is not an
// exception and cannot be caught. So the address space is capped around the call instead
// (`RLIMIT_AS`), which makes `operator new` itself fail immediately, with a
// `std::bad_alloc`, touching nothing. The cap is derived from this process's own peak RSS
// plus generous headroom, so nothing else in the run is squeezed; it is restored before
// the next reservation.
//
// If the cap cannot be set (no permission, or a platform without the limit), the case says
// so and skips rather than passing silently: a test that quietly did nothing is the shape
// this project keeps finding.
void an_unallocatable_envelope_is_declined_not_fatal(llama_model * model,
                                                     const llama_vocab * vocab) {
    llama_context * ctx = make_ctx(model, 256, 2);
    check(ctx != nullptr, "unallocatable: context exists");
    if (!ctx) {
        return;
    }
    llama_memory_t memory = llama_get_memory(ctx);
    const llama_token token = llama_token(llama_vocab_n_tokens(vocab) - 1);

    // The cap is derived from the ADDRESS SPACE, not from resident memory, and that is
    // the whole difference between this test measuring something and measuring nothing.
    //
    // RLIMIT_AS limits virtual address space; `ru_maxrss` is resident set. Loading a GGUF
    // mmaps the weights, so the two diverge as the model grows -- a multi-GB model maps far
    // more address space than it ever faults in. A cap computed from RSS can therefore land
    // *below* what the process already occupies, and then the limit is not retroactive on
    // Linux: every small allocation afterwards succeeds only because glibc can carve it out
    // of the address space already mapped, while the case's real subject (a 16 GB padded
    // copy) fails for an unrelated reason -- or, on a build that asks for fresh address
    // space, `llama_batch_ext_init` fails first inside decode_pair, and the failure is read
    // as if it came from the padding path.
    //
    // The test binary's own mapping for this fixture is small (0.1 GB as printed below), so
    // an RSS-based cap happened to sit above it. That is luck, not measurement, and it is
    // why the cap is now VAS + headroom with both figures reported.
    //
    // Headroom is generous for the decode's own allocations and far below the 16 GB a
    // 4e9-row envelope asks for. If the platform will not let the cap sit above what is
    // already mapped, the case says so and skips, because a test that quietly did nothing is
    // the shape this project keeps finding.
    const auto address_space_bytes = []() -> rlim_t {
        std::ifstream statm("/proc/self/statm");
        long pages_total = 0;
        if (!(statm >> pages_total) || pages_total <= 0) {
            return 0;
        }
        const long page = sysconf(_SC_PAGESIZE);
        return rlim_t(pages_total) * rlim_t(page > 0 ? page : 4096);
    };
    const rlim_t headroom = rlim_t(2) << 30;   // 2 GiB: plenty for one decode, far below 16 GiB
    const rlim_t vas = address_space_bytes();
    if (vas == 0) {
        std::fprintf(stderr,
                     "unallocatable: /proc/self/statm unreadable; the allocation-failure "
                     "branch is NOT exercised by this run\n");
        llama_free(ctx);
        return;
    }
    const rlim_t cap = vas + headroom;

#if defined(__SANITIZE_ADDRESS__)
    constexpr bool asan_active = true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
    constexpr bool asan_active = true;
#else
    constexpr bool asan_active = false;
#endif
#else
    constexpr bool asan_active = false;
#endif

    rlimit saved {};
    bool capped = getrlimit(RLIMIT_AS, &saved) == 0;
    if (capped && saved.rlim_max != RLIM_INFINITY && cap > saved.rlim_max) {
        // Raising rlim_max needs privilege we do not have, so a cap exactly at the hard
        // limit would leave no room above the current mapping and every allocation would
        // fail -- including the ones this case is not about.
        std::fprintf(stderr,
                     "unallocatable: RLIMIT_AS hard limit is %.1f GB, below the %.1f GB cap "
                     "this case needs above a %.1f GB mapping; the allocation-failure branch "
                     "is NOT exercised by this run\n",
                     double(saved.rlim_max) / (1 << 30), double(cap) / (1 << 30),
                     double(vas) / (1 << 30));
        capped = false;
    }
    // Under ASan the address-space cap is meaningless: the ~20 TB shadow reservation
    // dominates `vas`, so `cap` neither fails one `operator new` nor stays below the
    // allocator's own ceiling. The ceiling moves to ASAN_OPTIONS instead (CMakeLists.txt
    // sets `allocator_may_return_null=1:max_allocation_size_mb=1024` on this test).
    if (capped && !asan_active) {
        rlimit want = saved;
        want.rlim_cur = cap;
        capped = setrlimit(RLIMIT_AS, &want) == 0;
    }
    // AddressSanitizer reserves ~20 TB of shadow address space at startup, so RLIMIT_AS
    // can neither sit above the mapping nor reliably make one `operator new` fail: an
    // ASan-internal OOM aborts the process instead of raising `std::bad_alloc`, and the
    // allocator's own ceiling (`max_allocation_size_mb`, `allocator_may_return_null`) still
    // ends in `AddressSanitizer: out-of-memory ... ABORTING` on this path, measured on
    // libasan.so.8 with a 16 GiB request. The case therefore says so and skips under ASan
    // rather than pretending: the negative path is unexercised there, not passing there.
    if (asan_active) {
        std::fprintf(stderr,
                     "unallocatable: ASan build; the allocation-failure branch is NOT "
                     "exercised by this run (RLIMIT_AS is inert under a ~20 TB shadow "
                     "reservation and ASan's allocator aborts rather than returning null "
                     "for this request)\n");
        llama_free(ctx);
        return;
    }
    if (!capped) {
        std::fprintf(stderr,
                     "unallocatable: RLIMIT_AS could not be set on this platform; the "
                     "allocation-failure branch is NOT exercised by this run\n");
        llama_free(ctx);
        return;
    }
    std::fprintf(stderr,
                 "unallocatable: cap %.1f GB over a %.1f GB mapping (%.1f GB of headroom)\n",
                 double(cap) / (1 << 30), double(vas) / (1 << 30),
                 double(headroom) / (1 << 30));

    {
        herd::prefab::scratch_scope held(memory, llama_seq_id(1), 4000000000u);
        check(held.usable(), "unallocatable: the reservation opens (the sequence is empty)");

        // The decode must come back as a failure, not as a dead process. Without the
        // seam's catch block the exception would unwind through llama_process here.
        const bool decoded = decode_pair(ctx, token, 0);
        // The run CONTINUES. That is the contract, and my first version of this line
        // asserted the opposite -- decoded == false -- which would have described a lane
        // that turns an optimisation's allocation failure into a failed run: the defect,
        // not the fix. Declining the padding leaves the ubatch exactly as upstream split
        // it, so the decode proceeds and only the counters show that the lane stood down.
        check(decoded, "unallocatable: the decode still succeeds, unpadded -- the padding "
                       "failed, not the run");
        check(held.padded_ubatches() == 0,
              "unallocatable: nothing was padded -- the copy could not be built");
        check(held.pad_rows() == 0, "unallocatable: no pad row was added");
        check(held.declined() >= 1, "unallocatable: the decline is counted for the driver");
        // The explicit-request contract, as a ruling rather than a log line: the
        // reservation was valid, it was offered work, and it padded nothing. The run
        // continues (the output is correct) but it is NOT the padded configuration that
        // was asked for, and the destructor says so on stderr using this same query.
        check(held.request_unmet(),
              "unallocatable: the request is reported unmet, not silently satisfied");
    }

    // Lift the cap before anything else allocates. Restoring failure here would be worse
    // than the original defect: every later test would fail for an unrelated reason.
    check(setrlimit(RLIMIT_AS, &saved) == 0, "unallocatable: the address-space cap is lifted");

    // And the context is still usable: a failed envelope must not poison the run for the
    // next reservation, which is what a leaked cell or a half-built ubatch would do.
    {
        herd::prefab::scratch_scope ok(memory, llama_seq_id(1), 8u);
        check(ok.usable(), "unallocatable: a sane reservation still opens afterwards");
        check(decode_pair(ctx, token, 2), "unallocatable: and it decodes");
        check(ok.padded_ubatches() > 0, "unallocatable: with the lane in force");
        // Negative control for request_unmet(): a reservation that did deliver must not
        // report itself unmet. Without this, the predicate could be vacuously true and the
        // assertion above would pass for the wrong reason.
        check(!ok.request_unmet(), "unallocatable: a padded reservation is NOT reported unmet");
        check(ok.release(), "unallocatable: and it releases");
    }
    check(llama_memory_seq_pos_max(memory, 1) == -1,
          "unallocatable: the scratch sequence is clean after both reservations");

    llama_free(ctx);
}

// Two contexts, one provider, one thread: the case the ABI's identity check exists for.
//
// `provider_get_stats` once ignored its `ctx` argument and read the thread-local `t_scope`, so it
// answered for whichever context had opened the reservation. A driver that keeps a draft and a
// target model -- two contexts, one thread -- would be handed A's counters under B's name, and the
// JSON would look entirely healthy. The fix is `t_scope->memory() == llama_get_memory(ctx)`; this
// is the run that proves it, against the real provider rather than a stand-in.
//
// Every assertion is about a DIFFERENCE between the two contexts. That is deliberate: a test that
// only asked about A passes just as well against a provider that answers A for everything, which
// is precisely the defect.
void the_provider_reports_only_its_own_context(llama_model * model) {
    check(herd::ext::prefab_available(),
          "provider: the lane registers its statistics provider when it runs");

    llama_context * a = make_ctx(model, 128, 4);
    llama_context * b = make_ctx(model, 128, 4);
    check(a != nullptr && b != nullptr, "provider: two contexts on one thread");
    if (a == nullptr || b == nullptr) {
        if (a) llama_free(a);
        if (b) llama_free(b);
        return;
    }

    llama_memory_t mem_a = llama_get_memory(a);
    llama_memory_t mem_b = llama_get_memory(b);
    check(mem_a != nullptr && mem_b != nullptr && mem_a != mem_b,
          "provider: the two contexts have distinct memories (the identity check is not vacuous)");

    // Outside any reservation, neither is enabled. Without this the two later assertions could be
    // satisfied by a predicate that is simply always true.
    check(!herd::ext::prefab_is_enabled(a) && !herd::ext::prefab_is_enabled(b),
          "provider: with no reservation open, neither context is enabled");

    const llama_token token = 1;
    const std::uint64_t both = herd::ext::prefab_captured_n_ubatch_prefab |
                              herd::ext::prefab_captured_n_pad_rows;

    {
        herd::prefab::scratch_scope held(mem_a, llama_seq_id(1), 8u);
        check(held.usable(), "provider: a reservation on context a");
        check(decode_pair(a, token, 0), "provider: context a decodes");
        check(held.padded_ubatches() > 0 && held.pad_rows() > 0,
              "provider: a's counters are non-zero, so a leak into b would be visible");

        check(herd::ext::prefab_is_enabled(a),
              "provider: a is enabled while its own reservation is open");
        check(!herd::ext::prefab_is_enabled(b),
              "provider: a's reservation does NOT enable b -- the reservation belongs to a's memory");

        herd::ext::prefab_stats sa{};
        herd::ext::prefab_get_stats(a, &sa);
        check(sa.captured == both, "provider: a reports both counters as captured");
        check(sa.n_ubatch_prefab == held.padded_ubatches(),
              "provider: a's ubatch count is its reservation's, not a copy");
        check(sa.n_pad_rows == held.pad_rows(), "provider: a's pad-row count is its reservation's");

        herd::ext::prefab_stats sb{};
        herd::ext::prefab_get_stats(b, &sb);
        check(sb.captured == 0,
              "provider: b reports NOTHING captured while a's reservation is open -- b was not "
              "measured, and a zero that says so is the truth");
        check(sb.n_ubatch_prefab == 0 && sb.n_pad_rows == 0,
              "provider: b's counters are zero, not context a's numbers");

        // The JSON is the surface a driver actually reads: the same two facts, through the emitter.
        const std::vector<std::pair<const char *, const llama_context *>> contexts{{"a", a},
                                                                                  {"b", b}};
        const std::string json = herd::ext::prefab_stats_json(contexts);
        check(json.find("\"name\":\"a\"") != std::string::npos &&
                  json.find("\"name\":\"b\"") != std::string::npos,
              "provider: both contexts are named in the JSON");
        check(json.find("\"captured\":0") != std::string::npos,
              "provider: the JSON carries b's zero captured count");
        check(json.find("\"n_pad_rows\":0") != std::string::npos,
              "provider: and b's zero pad rows -- a's number did not land under b's name");
    }

    // After the reservation closes, A must stop reporting as well. Otherwise `is_enabled` would
    // describe a lane that is no longer running, which is the same false assurance reversed.
    check(!herd::ext::prefab_is_enabled(a),
          "provider: a is no longer enabled once its reservation is gone");
    {
        herd::ext::prefab_stats sa{};
        herd::ext::prefab_get_stats(a, &sa);
        check(sa.captured == 0, "provider: a reports nothing captured after its reservation ends");
    }

    llama_free(b);
    llama_free(a);
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    if (!herd::find("prefab.split_simple")) {
        std::fprintf(stderr,
                     "herd-prefab-split-scope-test: the split seam is not installed. Set "
                     "HERD_PREFAB_SPLIT=1 to run the lane.\n");
        return 2;
    }
    herd::install_all();
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

    release_empties_the_sequence_and_is_idempotent(model, vocab);
    an_oversized_envelope_fails_loudly(model, vocab);
    a_preoccupied_scratch_sequence_is_not_adopted(model, vocab);
    a_reservation_does_not_pad_another_contexts_decode(model, vocab);
    a_released_reservation_stops_padding(model, vocab);
    an_unallocatable_envelope_is_declined_not_fatal(model, vocab);
    the_provider_reports_only_its_own_context(model);

    llama_model_free(model);
    llama_backend_free();

    std::fprintf(stderr,
                 "scope contract: %d case group(s) ran; %s\n",
                 7, failures ? "FAILED" : "holds");
    return failures ? 1 : 0;
}
