// The prefab station's rulings, tested without a model, a backend, or upstream.
//
// Each case pins a decision a reader would otherwise have to trust: which ubatches
// may be padded, what happens at the ladder's edges, and that a decline is counted
// under its own reason rather than lumped into "did nothing".
#include "herd_prefab_station.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <thread>
#include <vector>

using namespace herd::prefab;

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

config ladder(std::initializer_list<uint32_t> widths) {
    config cfg{};
    cfg.enabled = true;
    for (uint32_t width : widths) {
        cfg.envelopes[cfg.n_envelopes++] = width;
    }
    return cfg;
}

shape tokenwise(uint32_t n) {
    shape s{};
    s.n_tokens = n;
    s.n_seqs = n;
    s.n_seq_tokens = 1;
    s.n_pos = 1;
    return s;
}

// A lane that was never configured must not pad. Reporting accept here would make
// every run silently take the padded path.
void disabled_by_default() {
    station s;
    s.configure(config{});
    check(s.consider(tokenwise(3)).why == verdict::disabled, "unconfigured lane declines");
}

// The single decision that makes the lane work: a narrow batch is lifted to the
// smallest envelope that holds it, not to the largest.
void picks_smallest_envelope() {
    station s;
    s.configure(ladder({8, 16, 32}));
    const ruling one = s.consider(tokenwise(1));
    check(one.why == verdict::accept && one.n_env == 8, "one token pads to the smallest envelope");
    const ruling nine = s.consider(tokenwise(9));
    check(nine.why == verdict::accept && nine.n_env == 16, "nine tokens skip the envelope they do not fit");
    const ruling exactly = s.consider(tokenwise(16));
    check(exactly.why == verdict::already_wide, "an exact fit is not re-padded");
}

// Wider than the ladder: the lane declines rather than truncating a batch.
void max_exceeded() {
    station s;
    s.configure(ladder({8, 16}));
    check(s.consider(tokenwise(17)).why == verdict::max_exceeded, "17 tokens exceed a 16 ladder");
    check(s.consider(tokenwise(0)).why == verdict::not_tokenwise, "an empty ubatch is declined");
}

// The padding trick only works when each row is independently addressable and
// owns its sequence set. These are the shapes where appending a row would change
// the meaning of the rows already there.
void shape_preconditions() {
    station s;
    s.configure(ladder({8}));

    shape packed{};
    packed.n_tokens = 4;
    packed.n_seqs = 2;
    packed.n_seq_tokens = 2;
    check(s.consider(packed).why == verdict::not_tokenwise, "a packed ubatch is declined");

    shape equal = tokenwise(4);
    equal.equal_seqs = true;
    check(s.consider(equal).why == verdict::not_tokenwise, "equal-seqs ubatch is declined");

    shape unaligned = tokenwise(4);
    unaligned.n_seqs = 2;
    check(s.consider(unaligned).why == verdict::unaligned_seqs, "n_seqs != n_tokens is declined");

    shape rope = tokenwise(4);
    rope.n_pos = 4;
    check(s.consider(rope).why == verdict::multi_pos, "M-RoPE positions are declined");

    shape embd = tokenwise(4);
    embd.has_embd = true;
    check(s.consider(embd).why == verdict::mixed_input, "embedding input is declined");

    shape mixed = tokenwise(4);
    mixed.has_type = true;
    check(s.consider(mixed).why == verdict::mixed_input, "mixed ubatch is declined");
}

// The counters are the only evidence the lane ran. A padded run must be
// distinguishable from a declined one, and the pad row count must be the actual
// difference, not the envelope width.
void counters_separate_work_from_silence() {
    station s;
    s.configure(ladder({8, 16}));
    s.note_ran(8, 3);
    s.note_ran(16, 9);
    s.note_declined(verdict::max_exceeded);
    s.note_declined(verdict::mixed_input);
    s.note_declined(verdict::mixed_input);

    const counters c = s.snapshot();
    check(c.accepted == 2, "two padded ubatches are counted");
    check(c.pad_rows == (8 - 3) + (16 - 9), "pad rows are the difference per ubatch");
    check(c.declined[static_cast<int>(verdict::max_exceeded)] == 1, "decline reason is counted once");
    check(c.declined[static_cast<int>(verdict::mixed_input)] == 2, "repeated decline accumulates");
    check(c.declined[static_cast<int>(verdict::disabled)] == 0, "unused reasons stay zero");
}

// A ruling must not depend on how many times it was asked: the seam asks once to
// decide and a second time only if a caller re-asks, and a stateful answer would
// make the two disagree.
void rulings_are_pure() {
    station s;
    s.configure(ladder({8}));
    const ruling first = s.consider(tokenwise(5));
    s.note_ran(8, 5);
    const ruling second = s.consider(tokenwise(5));
    check(first.why == second.why && first.n_env == second.n_env, "a ruling is stable across calls");
    check(s.snapshot().accepted == 1, "re-asking does not count work");
}

// The placement half of the lane's decision: can the cache pay for the envelope, and
// what does it cost. Every case here is a way the first padding attempt failed or
// would have failed had it ever been reached.
void placement_accounts_for_cells() {
    // Already at width: nothing to do, and not a failure.
    capacity same{};
    same.n_tokens_real = 8;
    same.n_envelope    = 8;
    same.n_cells_free  = 0;
    const placement_plan exact = place(same);
    check(exact.why == placement::not_padded, "an ubatch already at envelope width is not padded");
    check(exact.cells_required == 8 && exact.cells_permanent == 0, "and costs nothing");

    // The case the measurement landed on: pad rows with no owning sequence.
    capacity unowned{};
    unowned.n_tokens_real = 1;
    unowned.n_envelope    = 8;
    unowned.n_cells_free  = 128;
    check(place(unowned).why == placement::scratch_unowned,
          "pad rows without an owning sequence are refused");
    check(place(unowned).cells_permanent == 7,
          "and the leak is counted: one cell per pad row, per ubatch");

    // Owning them but never clearing them is the same leak one step later.
    capacity unclearable  = unowned;
    unclearable.scratch_owned = true;
    check(place(unclearable).why == placement::scratch_unclearable,
          "owned but unclearable pad cells are refused");
    check(place(unclearable).cells_permanent == 7, "with the same permanent cost");

    // The arithmetic case: a wide envelope with a small cache. `find_slot` needs one
    // index per token and rejects the ubatch when it cannot collect them.
    capacity tight = unowned;
    tight.scratch_owned      = true;
    tight.scratch_clearable  = true;
    tight.n_cells_free       = 4;
    const placement_plan short_of_cells = place(tight);
    check(short_of_cells.why == placement::no_free_cells,
          "fewer free cells than the envelope width is refused");
    check(short_of_cells.cells_required == 8, "the requirement is the envelope width, not the row count");
    check(short_of_cells.cells_permanent == 0, "a refused placement leaks nothing");

    // The accepted case, with its cost reported honestly in both directions.
    capacity fits = tight;
    fits.n_cells_free = 8;
    const placement_plan accepted = place(fits);
    check(accepted.why == placement::fits, "a free run of envelope width is accepted");
    check(accepted.cells_reusable == 8 && accepted.cells_permanent == 0,
          "the cells are borrowed, not consumed");

    // Every refusal names itself, and no two share a message: a lane that cannot tell
    // "no cells" from "unowned cells" falls back for the wrong reason.
    const placement all[] = {placement::fits, placement::no_free_cells,
                             placement::scratch_unowned, placement::scratch_unclearable,
                             placement::not_padded};
    for (std::size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) {
        check(std::string(name(all[i])).size() > 2, "a placement reason has a message");
        for (std::size_t j = 0; j < i; ++j) {
            check(std::string(name(all[i])) != name(all[j]),
                  "and no two placement reasons share a message");
        }
    }
}

// The scratch pool: real state, not arithmetic. The invariants are the ones a padded
// ubatch depends on, so each case names the failure it prevents.
void scratch_pool_borrows_and_returns() {
    scratch_pool pool(8);

    // Nothing held to begin with, and the peak starts at zero rather than at the
    // capacity: a high-water mark that reports the capacity regardless would hide the
    // difference between "borrowed everything once" and "never borrowed".
    check(pool.borrowed() == 0 && pool.live() == 0 && pool.high_water() == 0,
          "a fresh pool holds nothing");

    const auto first = pool.reserve(8);
    check(first != scratch_pool::no_reservation, "the whole capacity can be borrowed");
    check(pool.borrowed() == 8 && pool.live() == 1, "and is accounted for");
    check(pool.high_water() == 8, "the peak follows the borrow");

    // Over-borrow is refused rather than partially granted: a padded ubatch that got
    // five of eight cells has a plan it cannot fill, and the failure would land inside
    // upstream instead of here.
    check(pool.reserve(1) == scratch_pool::no_reservation, "a borrow past the capacity is refused");
    check(pool.borrowed() == 8, "a refusal does not change what is held");
    check(pool.refusals() == 1, "and is counted");

    // Returning is exact, and a double return is refused: crediting the same cells
    // twice would make the next over-borrow invisible.
    check(pool.release(first), "the reservation is released");
    check(pool.borrowed() == 0 && pool.live() == 0, "and the cells are back");
    check(!pool.release(first), "releasing it again is refused");
    check(pool.borrowed() == 0, "and cannot credit the pool twice");

    // The peak survives the release: it is the number that decides whether the
    // envelope fits at worst case, so it must not decay with the current load.
    check(pool.high_water() == 8, "the high-water mark is not reset by a release");

    // Two reservations at once, as two ubatches' worth of scratch would be.
    const auto a = pool.reserve(3);
    const auto b = pool.reserve(5);
    check(a != b && a != scratch_pool::no_reservation && b != scratch_pool::no_reservation,
          "two reservations get distinct ids");
    check(pool.borrowed() == 8 && pool.live() == 2, "both are held");
    check(pool.release(b) && pool.borrowed() == 3, "releasing one leaves the other");
    check(pool.release(a) && pool.borrowed() == 0, "and the pool empties");

    // Teardown.
    check(pool.reserve(4) != scratch_pool::no_reservation, "one reservation before teardown");
    check(pool.release_all() == 1, "release_all reports what it released");
    check(pool.borrowed() == 0 && pool.live() == 0, "and leaves nothing held");
}

// A zero-cell reservation is not a borrow. Accepting it would inflate the live count
// with entries that hold nothing, and the table would fill up first.
void scratch_pool_refuses_empty_and_overflowing_requests() {
    scratch_pool pool(4);
    check(pool.reserve(0) == scratch_pool::no_reservation, "zero cells is refused");
    check(pool.live() == 0, "and creates no entry");

    // The table is a fixed size; when it is full the refusal must not be reported as
    // "out of cells", because the caller would then go looking at the capacity.
    scratch_pool small(1024);
    int granted = 0;
    while (small.reserve(1) != scratch_pool::no_reservation) {
        ++granted;
    }
    check(granted > 0 && granted < 32, "the table's own limit is what stops it, not the capacity");
    check(small.borrowed() == std::uint32_t(granted), "and only the grants are accounted");
    check(small.release_all() == std::uint32_t(granted), "teardown releases them all");
    check(small.borrowed() == 0, "leaving nothing borrowed");
}

// The id counter must never hand out `no_reservation`, and must never hand out an id a
// live reservation already holds. The earlier version did `next_id_++` and failed both
// at the 2^32 wrap: it returned 0, the caller treated it as a refusal, the entry it had
// already written looked empty and could be reused, while `borrowed_` and `live_` kept
// the cells -- an unreleasable borrow. Reaching that in production takes four billion
// reservations, so it is exercised here by placing the counter.
void scratch_pool_ids_never_repeat_or_vanish() {
    scratch_pool pool(64);
    pool.force_next_id_for_test(0xFFFFFFFEu);

    // Read the placed value back before relying on it.
    //
    // This asserts the seam itself, not the pool: if force_next_id_for_test() were a
    // no-op -- because this translation unit's class definition disagrees with the
    // implementation's, or because a stale object file was linked -- the counter would
    // stay at 1 and every assertion below would fail with a message about *ids*, sending
    // a reader to look at id allocation instead of at the build. The read-back names the
    // actual problem.
    //
    // A note from the incident this file came out of: a mutation test once left its
    // mutant in the tree, and a fresh build directory ran a stale binary that failed here
    // with "the counter is honoured". The source was correct by then; the artefact was
    // not. No unit test can catch that -- the guard is rebuilding in a fresh directory
    // and running ctest there, which is exactly how it was found. What this assertion can
    // do is make the *setter* honest, so at least the fixture cannot silently do nothing.
    check(pool.next_id_for_test() == 0xFFFFFFFEu, "the placed counter reads back");

    const auto first = pool.reserve(4);
    check(first == 0xFFFFFFFEu, "the counter is honoured");
    const auto second = pool.reserve(4);
    check(second == 0xFFFFFFFFu, "and advances");

    // Now the wrap. A correct implementation skips 0 and reuses 1, which is free.
    const auto third = pool.reserve(4);
    check(third != scratch_pool::no_reservation, "a reservation survives the wrap");
    check(third != 0, "the reserved id is never the sentinel");
    check(third != first && third != second, "and never duplicates a live id");

    // Every one of the three is still releasable, which is the property the old wrap
    // broke: the id it handed out could not be released at all.
    check(pool.borrowed() == 12 && pool.live() == 3, "three live reservations");
    check(pool.release(first) && pool.release(second) && pool.release(third),
          "all three release after the wrap");
    check(pool.borrowed() == 0 && pool.live() == 0, "and the ledger is empty again");

    // A live id is skipped even when the counter points straight at it.
    const auto held = pool.reserve(2);
    pool.force_next_id_for_test(held.slot_id);
    const auto next = pool.reserve(2);
    check(next != held && next != 0, "a live id is not handed out twice");
    check(pool.release(held) && pool.release(next), "both release");
}

// Two different refusals, two different counters. A caller that only sees a total cannot
// tell "raise the declared capacity" from "hold fewer reservations at once".
void scratch_pool_separates_refusal_causes() {
    scratch_pool pool(4);
    check(pool.reserve(8) == scratch_pool::no_reservation, "over-capacity is refused");
    check(pool.refusals_cells() == 1 && pool.refusals_table() == 0,
          "and counted as a cell shortfall");
    check(pool.refusals() == 1, "the total agrees");

    // Fill the table with one-cell reservations: the capacity allows it, the entry
    // array does not.
    int granted = 0;
    while (pool.reserve(1) != scratch_pool::no_reservation) {
        ++granted;
    }
    check(granted == 4, "the declared capacity limits the grants here");
    check(pool.refusals_cells() == 2, "the fifth cell is refused for capacity");

    // A pool whose capacity is large but whose table is small: now the other cause.
    scratch_pool table_bound(1024);
    int live = 0;
    while (table_bound.reserve(1) != scratch_pool::no_reservation) {
        ++live;
    }
    check(table_bound.refusals_table() == 1 && table_bound.refusals_cells() == 0,
          "a full table is counted as its own cause");
    check(table_bound.borrowed() == std::uint32_t(live), "and only grants are accounted");
}

// The ABA case, as a test rather than a comment.
//
// This is the failure the epoch exists for, and it is reachable WITHOUT any wraparound:
// an id released long ago is handed out again for a different borrow, and a stale holder
// tries to release what it still has. Under the old id-only scheme the release matched
// the live slot and credited cells the stale holder never held -- silently, because the
// ledger stayed self-consistent and the only symptom was cells going missing in an
// unrelated ubatch later.
//
// Reuse is forced by putting the counter back, which is what a wrap does to it.
void scratch_pool_stale_handle_cannot_release_a_recycled_slot() {
    scratch_pool pool(64);

    // 1. A borrow, then its release. The caller keeps the handle -- a log line, a retry
    //    path, a plan that outlived its ubatch.
    const auto stale = pool.reserve(4);
    check(stale.valid(), "the first borrow is valid");
    check(pool.release(stale), "and releases cleanly");
    check(pool.borrowed() == 0 && pool.live() == 0, "leaving the ledger empty");

    // 2. The counter comes back round to the same slot id, as it does after 2^32 borrows.
    pool.force_next_id_for_test(stale.slot_id);

    // 3. A different caller takes a borrow and gets that same id -- a different borrow,
    //    with a different epoch, holding cells nobody else may free.
    const auto live = pool.reserve(8);
    check(live.valid(), "the reused slot id is issued");
    check(live.slot_id == stale.slot_id, "the slot id really was recycled");
    check(live.epoch != stale.epoch,
          "but the epoch advanced, which is the only thing that distinguishes them");
    check(pool.borrowed() == 8 && pool.live() == 1, "the new borrow holds its cells");

    // 4. The stale holder tries to give its borrow back.
    check(!pool.release(stale),
          "a stale handle cannot release the recycled slot");
    check(pool.refusals_stale() == 1, "and the refusal is counted as stale, not as unknown");
    check(pool.borrowed() == 8 && pool.live() == 1,
          "the live reservation is untouched; its cells were not credited away");

    // 5. The live holder can still release, so the guard refuses the wrong handle rather
    //    than locking the right one out.
    check(pool.release(live), "the genuine handle still releases");
    check(pool.borrowed() == 0 && pool.live() == 0, "and the ledger empties exactly once");

    // 6. A teardown must not resurrect old handles either: next_epoch_ is not reset by
    //    release_all(), so a handle that outlived the teardown stays stale.
    const auto before_teardown = pool.reserve(3);
    check(pool.release_all() == 1, "teardown releases the outstanding borrow");
    pool.force_next_id_for_test(before_teardown.slot_id);
    const auto after_teardown = pool.reserve(3);
    check(after_teardown.slot_id == before_teardown.slot_id, "the same slot id comes back");
    check(!pool.release(before_teardown),
          "a handle from before the teardown cannot release the new borrow");
    check(pool.release(after_teardown), "while the current one can");
    check(pool.refusals_stale() == 2, "and both stale refusals are counted");
}

// Cross-pool isolation, as an assertion rather than as a comment.
//
// The `owner` field exists because two pools both mint their first handle as
// {slot 1, epoch 1} -- the documented deployment is two windows with two stations that
// share nothing, so "the other station's handle" is a reachable value, not a hypothetical.
// Without an owner token, releasing it here would credit cells this pool never lent, and
// the books would stay balanced while a live reservation quietly lost its cells.
//
// This asserts all three parts of the fix at once: two pools are told apart even when
// their slot and epoch coincide, the foreign handle is refused and counted as foreign,
// and the genuine one still releases -- because a guard that refuses everything is not a
// guard.
void scratch_pool_isolates_two_pools() {
    scratch_pool a(64);
    scratch_pool b(64);

    const auto from_a = a.reserve(8);
    const auto from_b = b.reserve(8);
    check(from_a.valid() && from_b.valid(), "both pools issue a borrow");
    check(from_a.slot_id == from_b.slot_id && from_a.epoch == from_b.epoch,
          "their first handles collide on slot and epoch, which is why owner exists");
    check(from_a.owner != from_b.owner, "and only the owner tells them apart");
    check(from_a != from_b,
          "so two handles must not compare equal -- this is the assertion the "
          "two-field operator failed");

    // The foreign handle: same slot, same epoch, different ledger.
    check(!b.release(from_a), "a handle from another pool cannot release here");
    check(b.refusals_foreign() == 1, "and the refusal is counted as foreign, not as stale");
    check(b.borrowed() == 8 && b.live() == 1,
          "the local reservation is untouched; its cells were not credited away");
    check(a.borrowed() == 8, "and the pool that issued the handle is likewise untouched");

    // A guard that refuses everything would pass the checks above; this is what shows it
    // refuses the wrong handle rather than locking out the right one.
    check(a.release(from_a), "the genuine handle releases in its own pool");
    check(b.release(from_b), "and likewise in the other");
    check(a.borrowed() == 0 && b.borrowed() == 0, "both ledgers empty exactly once");
    check(a.refusals_foreign() == 0, "the pool that was never crossed records no foreign refusal");
}

} // namespace

// The pool's process-wide identity boundary, executed rather than argued.
//
// The counter's domain is [1, UINT64_MAX] for tokens plus the single "spent" value 0, and
// the rule is that no value is ever issued twice. That gives three observable points, and
// all three are reachable without aborting the test process:
//
//   counter = max-1 : a pool is minted with max-1, and the counter becomes max
//   counter = max   : a pool is minted with max,   and the counter becomes 0 (spent)
//   counter = 0     : minting refuses, and the counter STAYS 0
//
// The last line is the one worth having. Under the previous shape -- fetch_add plus a
// post-hoc check on the value it returned -- the counter had already been incremented to 1
// by the time the refusal was decided, so any refusal that returned instead of killing the
// process would have handed the very next pool the first pool's identity: two live pools
// that no handle can tell apart. Here the counter cannot leave 0, so the duplicate is not
// constructible even if the abort ever becomes a report.
void scratch_pool_pool_identity_never_wraps() {
    using pool = scratch_pool;
    constexpr std::uint64_t top = ~std::uint64_t(0);

    // The counter is process-wide, so its position is part of the test's setup rather than
    // an incidental; every assertion below reads it back instead of assuming.
    pool::force_pool_token_for_test(top - 1);
    check(pool::peek_pool_token_for_test() == top - 1, "the counter is placed at its last step");

    pool last_but_one(8);
    check(last_but_one.owner_for_test() == top - 1, "the pool takes the token the counter held");
    check(pool::peek_pool_token_for_test() == top, "and the counter advances to the maximum");

    pool last(8);
    check(last.owner_for_test() == top, "the maximum is issued exactly once");
    check(pool::peek_pool_token_for_test() == 0, "leaving only the spent state");
    check(last_but_one.owner_for_test() != last.owner_for_test(),
          "no two pools share an identity");

    // Exhaustion. Refused, and refused without moving: the value that must never reappear
    // is 1, because 1 is the token the first pool in the process was given.
    check(!pool::pool_token_available_for_test(), "a spent counter reports no token");
    check(pool::peek_pool_token_for_test() == 0,
          "and asking does not wrap the counter round to the first token");

    // Identity still does its job at the boundary: two pools minted back to back are told
    // apart, so neither may release the other's borrow.
    const auto from_a = last_but_one.reserve(4);
    const auto from_b = last.reserve(4);
    check(from_a.valid() && from_b.valid(), "both borrows are issued");
    check(from_a.owner != from_b.owner, "consecutive pools have different identities");
    check(!last.release(from_a), "a neighbour's borrow cannot be released here");
    check(last.borrowed() == 4, "and its ledger is untouched by the attempt");
    check(last_but_one.release(from_a) && last.release(from_b), "while each releases its own");

    // Leave the process-wide counter where a later test can still mint from it. Not a
    // convenience: a test that exhausts a global and does not restore it makes every
    // subsequent test depend on its position in the run.
    pool::force_pool_token_for_test(top - 2);
}

// The mint itself, at the boundary -- not a predicate over the counter.
//
// The previous round asked pool_token_available_for_test() whether the counter was spent,
// which reads the counter and proves nothing about the decision its owner makes. A
// mutation that moved the refusal AFTER the counter moved survived the whole suite: the
// predicate still answered "spent" because it looks at the raw value, while the mint would
// have handed out token 1 -- the first identity the process ever issued. So the boundary
// is exercised through the minting function, which is the only thing whose behaviour
// matters here.
//
// The window sits below the maximum so this test does not overlap the tokens the earlier
// identity test issues (top-1 and top); no assertion below can then be satisfied by an
// accident of test order.
void scratch_pool_mint_refuses_at_the_boundary() {
    using pool = scratch_pool;
    constexpr std::uint64_t top  = ~std::uint64_t(0);
    constexpr std::uint64_t base = top - 16;

    pool::force_pool_token_for_test(base);

    // 1. An ordinary step, asserted as a pair: the minted token is what the counter held
    //    AND the counter advanced. Either alone is satisfiable by a wrong implementation.
    std::uint64_t issued = 0;
    check(pool::mint_token_for_test(&issued), "a live counter mints");
    check(issued == base, "the minted token is the value the counter held");
    check(pool::peek_pool_token_for_test() == base + 1, "and the counter advances by one");

    // 2. The maximum is issuable exactly once.
    pool::force_pool_token_for_test(top);
    check(pool::mint_token_for_test(&issued), "the maximum itself is issuable");
    check(issued == top, "and is the token handed out");
    check(pool::peek_pool_token_for_test() == 0,
          "leaving the spent state, which is 0 and never a token");

    // 3. Exhaustion refuses, and the refusal does not move the counter.
    //
    //    This is the assertion the earlier suite lacked. A mutant whose check sits after
    //    the store returns true here and writes 0 into the out parameter -- or, worse,
    //    leaves 1 behind for the next caller. The second check catches the move even when
    //    the return value is ignored, and the third catches a bogus out-write.
    std::uint64_t untouched = 12345;
    check(!pool::mint_token_for_test(&untouched), "a spent counter refuses");
    check(pool::peek_pool_token_for_test() == 0, "and the refusal leaves it spent");
    check(untouched == 12345, "a refused mint does not write to the out parameter");

    // 4. Stable, not a one-shot glitch.
    check(!pool::mint_token_for_test(&untouched), "the second attempt also refuses");
    check(pool::peek_pool_token_for_test() == 0, "with the counter still spent");
}

// Several threads minting at once: no two tokens may coincide, and the counter must land
// exactly where the arithmetic says.
//
// What is being protected is the owner field's premise -- distinct pools must be
// distinguishable -- so the check is set cardinality, not "no crash". A compare-and-
// exchange loop that dropped its retry would pass a single-threaded test and fail here.
void scratch_pool_mint_is_unique_under_contention() {
    using pool = scratch_pool;
    constexpr std::uint64_t base      = 1000;
    constexpr int           threads   = 8;
    constexpr int           per_thread = 64;

    pool::force_pool_token_for_test(base);

    std::atomic<int>           refused{0};
    std::vector<std::uint64_t> issued(std::size_t(threads) * per_thread, 0);
    std::vector<std::thread>   workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (int i = 0; i < per_thread; ++i) {
                std::uint64_t token = 0;
                if (!pool::mint_token_for_test(&token)) {
                    ++refused;
                    continue;
                }
                issued[std::size_t(t) * per_thread + std::size_t(i)] = token;
            }
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }

    check(refused.load() == 0, "every thread minted its whole share");
    check(pool::peek_pool_token_for_test() == base + threads * per_thread,
          "the counter advanced by exactly the number of mints");

    std::vector<std::uint64_t> sorted = issued;
    std::sort(sorted.begin(), sorted.end());
    check(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end(),
          "no two threads received the same token");
    check(sorted.front() == base && sorted.back() == base + threads * per_thread - 1,
          "and the tokens cover the window exactly once");

    // The restore, and why it is asserted rather than assumed.
    //
    // Restoring a global is a plain write; if it lands on a value already handed out, the
    // next mint reissues an identity -- the very defect this file is about, reintroduced
    // by the test's own cleanup. So the restore target is chosen outside the window above,
    // and the mint that consumes it is checked to be fresh.
    constexpr std::uint64_t restore = base + 100000;
    pool::force_pool_token_for_test(restore);
    std::uint64_t after = 0;
    check(pool::mint_token_for_test(&after), "the restored counter mints");
    check(after == restore, "with the restored value");
    check(after > sorted.back(), "which is outside every token this test issued");
    pool::force_pool_token_for_test(restore + 1);
}

int main() {
    disabled_by_default();
    picks_smallest_envelope();
    max_exceeded();
    shape_preconditions();
    counters_separate_work_from_silence();
    rulings_are_pure();
    placement_accounts_for_cells();
    scratch_pool_borrows_and_returns();
    scratch_pool_refuses_empty_and_overflowing_requests();
    scratch_pool_ids_never_repeat_or_vanish();
    scratch_pool_stale_handle_cannot_release_a_recycled_slot();
    scratch_pool_isolates_two_pools();
    scratch_pool_pool_identity_never_wraps();
    scratch_pool_mint_refuses_at_the_boundary();
    scratch_pool_mint_is_unique_under_contention();
    scratch_pool_separates_refusal_causes();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr, "prefab station contract holds\n");
    return 0;
}
