// The physical index ledger: the fork's resolution contract, and the failures it prevents.
//
// Every case is derived from `llama_kv_cache::prepare_attention_views` in the fork, and from the
// header's one-sentence invariant: "logical identities are resolved after KV slot preparation; no
// physical index survives a decode call."
//
// So the cases divide by what a stale or mis-resolved index would do:
//
//   ADDRESS REUSE  a slot taken by a different cell must not be readable through the old cell's
//                  name, and the ledger must refuse to hand a live cell's address to another.
//   MIGRATION      a cell moving to a new slot must invalidate the resolutions that named the old.
//   ISOLATION      a cell of one sequence must never resolve as another's; two views of one
//                  sequence in one batch must be refused rather than one silently winning.
//   ORDER          virtual positions are positional across a span, so an out-of-order list is not a
//                  permutation of the same attention.
//   PRESENCE       every view must include its own token, or the softmax normalises over the past
//                  without the present, and a re-decode must be refused because it would pass that
//                  test for the wrong reason.
//
// FIXTURE RULE, learned the hard way: the cache holds positions 0..k-1, and every query is at
// position k or later. A query at a cached position is a re-decode and is refused first, which
// masks whatever the case was actually testing. Seven of the first run's fifteen failures were
// this, and the kernel was right every time -- worth recording, because a test whose fixture is
// wrong fails identically to a kernel that is wrong.
#include <tq/reader/index_ledger.h>

#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

// COPYING A LEDGER IS REFUSED AT COMPILE TIME, and that is a property no runtime case can assert —
// so it is asserted here, where the compiler checks it. A mutation experiment confirmed the gap: with
// the deleted copy constructor replaced by `= default`, every case still passed, because a second
// object with the same identity and the same epoch is exactly what the runtime tests cannot see. The
// two `static_assert`s below fail the BUILD instead, which is the only place this can be caught.
//
// Move stays legal and must: `ledger board = make_board();` is the ordinary construction, and the
// moved-to ledger keeps the identity (see the move case at the end).
static_assert(!std::is_copy_constructible<tq::reader::ledger>::value,
              "a ledger must not be copyable: a copy shares the identity and the epoch, so a "
              "resolution from one would be accepted by the other while the two diverge");
static_assert(!std::is_copy_assignable<tq::reader::ledger>::value,
              "nor copy-assignable, for the same reason");
static_assert(std::is_move_constructible<tq::reader::ledger>::value,
              "but movable: returning a ledger by value is the ordinary way to build one");

namespace {

int failures = 0;

void check(bool condition, const std::string & what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
        ++failures;
    }
}

using tq::reader::attention_request;
using tq::reader::cell_id;
using tq::reader::ledger;
using tq::reader::resolution;
using tq::reader::resolve_refusal;
using tq::reader::view_entry;

view_entry entry(std::int32_t seq, std::int32_t pos, std::int32_t virtual_pos) {
    view_entry e;
    e.id = cell_id{seq, pos};
    e.virtual_pos = virtual_pos;
    return e;
}

attention_request request(std::int32_t seq, std::int32_t pos, std::int32_t virtual_pos,
                          std::vector<view_entry> & entries) {
    attention_request r;
    r.query_seq = seq;
    r.query_pos = pos;
    r.query_virtual_pos = virtual_pos;
    r.entries = entries.data();
    r.n_entries = entries.size();
    return r;
}

// Sequences of the cache, cells (0,0)..(0,k-1) at slots 100+, and a declared capacity well above
// what any case names. Every query in this file must sit at position >= k.
ledger with_cached_cells(std::size_t n_sequences, int k) {
    ledger board;
    board.set_n_sequences(n_sequences);
    board.set_capacity(std::size_t(k) + 16);
    for (int i = 0; i < k; ++i) {
        check(board.put(cell_id{0, i}, 100 + i) == resolve_refusal::ok, "setup: the cell was cached");
    }
    return board;
}

} // namespace

// ---------------------------------------------------------------------------------------------
// 1. The happy path, and what it must produce: physical slots, in entry order.
// ---------------------------------------------------------------------------------------------
void a_plain_view_resolves_to_slots_in_entry_order() {
    ledger board = with_cached_cells(2, 3);
    std::vector<view_entry> entries = {
        entry(0, 0, 0), entry(0, 1, 1), entry(0, 2, 2),   // three past tokens
        entry(0, 3, 3),                                   // and the query itself
    };
    const attention_request r = request(0, 3, 3, entries);
    const resolution answer = board.resolve(&r, 1);

    check(answer.refusal == resolve_refusal::ok, "a consistent view resolves");
    check(answer.views.size() == 1, "one request produces one view");
    if (!answer.views.empty() && !answer.views[0].spans.empty()) {
        const auto & span = answer.views[0].spans[0];
        check(span.effective_pos == 3,
              "the effective position is the query's, because every key's virtual position equals "
              "its own position here");
        check(span.indices == std::vector<std::int32_t>({100, 101, 102}),
              "the slots are in entry order, and the query's own cell contributes none: it is not "
              "in the cache, and its key arrives as the graph's native K");
    }
    check(board.still_valid(answer), "and the resolution is still valid: nothing moved");
}

// ---------------------------------------------------------------------------------------------
// 2. ADDRESS REUSE, sequential — the failure this whole class exists for.
// ---------------------------------------------------------------------------------------------
void a_reused_slot_is_never_readable_through_the_old_cell() {
    ledger board;
    board.set_n_sequences(2);
    board.set_capacity(64);
    check(board.put(cell_id{0, 7}, 500) == resolve_refusal::ok, "cell A takes slot 500");

    std::vector<view_entry> entries = {entry(0, 7, 7), entry(1, 0, 8)};
    const attention_request r = request(1, 0, 8, entries);
    const resolution before = board.resolve(&r, 1);
    check(before.refusal == resolve_refusal::ok, "the view resolves while A owns slot 500");
    check(!before.views.empty() && !before.views[0].spans.empty() &&
              before.views[0].spans[0].indices[0] == 500,
          "and names slot 500");

    board.evict(cell_id{0, 7});
    check(!board.still_valid(before),
          "the old resolution is no longer valid: the epoch moved, so a consumer that checks "
          "before using it is stopped");
    check(board.put(cell_id{1, 5}, 500) == resolve_refusal::ok,
          "B may take slot 500 once A released it -- reuse by announcement");

    // A's name cannot reach B's data: the cell is simply gone.
    std::vector<view_entry> stale = {entry(0, 7, 7), entry(1, 1, 9)};
    const attention_request by_old_name = request(1, 1, 9, stale);
    check(board.resolve(&by_old_name, 1).refusal == resolve_refusal::entry_not_cached,
          "naming A after eviction is refused by name, not answered with B's slot: the durable "
          "identity is the cell, not the address");
}

// ---------------------------------------------------------------------------------------------
// 3. ADDRESS REUSE, unannounced — `put` refuses to hand a live cell's slot to a different cell.
// ---------------------------------------------------------------------------------------------
void taking_a_live_cells_slot_is_refused() {
    ledger board;
    board.set_n_sequences(1);
    check(board.put(cell_id{0, 0}, 42) == resolve_refusal::ok, "A owns slot 42");
    check(board.put(cell_id{0, 1}, 42) == resolve_refusal::slot_conflict,
          "a different cell cannot take a live cell's slot: reuse has to be announced by evicting "
          "first, so it is a decision rather than something that overtakes the owner");
    check(board.slot_of(cell_id{0, 1}) == -1, "and the refused cell was not recorded");
    check(board.slot_of(cell_id{0, 0}) == 42, "while A still owns its slot");
}

// ---------------------------------------------------------------------------------------------
// 4. MIGRATION.
// ---------------------------------------------------------------------------------------------
void a_migrated_cell_invalidates_the_old_resolution() {
    ledger board;
    board.set_n_sequences(1);
    board.set_capacity(64);
    check(board.put(cell_id{0, 0}, 10) == resolve_refusal::ok, "A at slot 10");

    std::vector<view_entry> entries = {entry(0, 0, 0), entry(0, 1, 1)};
    const attention_request r = request(0, 1, 1, entries);
    const resolution before = board.resolve(&r, 1);
    check(before.refusal == resolve_refusal::ok, "resolves naming slot 10");

    check(board.put(cell_id{0, 0}, 77) == resolve_refusal::ok,
          "A moves to slot 77 -- no eviction needed, the ownership is A's either way");
    check(!board.still_valid(before), "the earlier resolution is invalid: it names an address A no "
                                      "longer occupies");
    check(board.slot_of(cell_id{0, 0}) == 77, "and the ledger reports the new address");
    check(board.put(cell_id{0, 2}, 10) == resolve_refusal::ok,
          "the old address is free, so another cell may take it without a spurious conflict -- a "
          "move releases, it does not leak");

    std::vector<view_entry> again = {entry(0, 0, 0), entry(0, 1, 1)};
    const attention_request r2 = request(0, 1, 1, again);
    const resolution after = board.resolve(&r2, 1);
    check(after.refusal == resolve_refusal::ok, "a fresh resolution succeeds");
    check(!after.views.empty() && !after.views[0].spans.empty() &&
              after.views[0].spans[0].indices[0] == 77,
          "and names the new slot: the caller re-resolves rather than reusing the old answer");
}

// ---------------------------------------------------------------------------------------------
// 5. ISOLATION across sequences. A cross-sequence read is the POINT of reader attention, so the
//    refusal here is not "another sequence" -- it is a cell that is not there, or a duplicate slot.
// ---------------------------------------------------------------------------------------------
void cross_sequence_reads_are_allowed_but_must_be_present_and_distinct() {
    ledger board;
    board.set_n_sequences(3);
    board.set_capacity(64);
    check(board.put(cell_id{0, 4}, 11) == resolve_refusal::ok, "sequence 0 has a cell at 4");
    check(board.put(cell_id{1, 5}, 13) == resolve_refusal::ok, "sequence 1 has a cell at 5");
    check(board.put(cell_id{2, 9}, 12) == resolve_refusal::ok, "sequence 2 has a cell at 9");

    // Virtual positions must be strictly increasing, and `query_virtual_pos` must be the largest of
    // them: the rotation offset is `query_virtual_pos + key_pos - virtual_pos`, so a key whose
    // virtual position exceeds the query's would rotate to a point in the future. Giving every key
    // `virtual_pos == key_pos` and the query a large virtual position keeps all four effective
    // positions equal (20), which is what makes them one span -- and one span is the case worth
    // asserting here, since the grouping is tested where the positions differ.
    std::vector<view_entry> mixed = {entry(0, 4, 4), entry(1, 5, 5), entry(2, 9, 9),
                                     entry(1, 6, 20)};
    const attention_request r = request(1, 6, 20, mixed);
    const resolution answer = board.resolve(&r, 1);
    check(answer.refusal == resolve_refusal::ok,
          "a view reading two other sequences' tokens resolves: that is reader attention");
    if (!answer.views.empty() && !answer.views[0].spans.empty()) {
        check(answer.views[0].spans.size() == 1,
              "and the four entries, all rotating to the same effective position, form one span");
        check(answer.views[0].spans[0].indices == std::vector<std::int32_t>({11, 13, 12}),
              "and the slots are those cells', in entry order");
    }

    std::vector<view_entry> future = {entry(1, 9, 9), entry(1, 10, 10)};
    const attention_request f = request(1, 7, 7, future);
    check(board.resolve(&f, 1).refusal == resolve_refusal::future_key,
          "a key from the query's own sequence sits after the query: refused, because it could "
          "only resolve to a stale occupant");
}

// ---------------------------------------------------------------------------------------------
// 6. ORDER.
// ---------------------------------------------------------------------------------------------
void out_of_order_virtual_positions_are_refused() {
    ledger board = with_cached_cells(1, 3);
    std::vector<view_entry> shuffled = {entry(0, 0, 2), entry(0, 1, 1), entry(0, 2, 2)};
    const attention_request r = request(0, 3, 3, shuffled);
    check(board.resolve(&r, 1).refusal == resolve_refusal::unordered_virtual_pos,
          "a virtual position that does not exceed the previous one is refused: the entries are "
          "positional, so out-of-order is a different rotation per key, not a permutation");

    std::vector<view_entry> beyond = {entry(0, 0, 0), entry(0, 1, 9), entry(0, 2, 2)};
    const attention_request b = request(0, 3, 3, beyond);
    check(board.resolve(&b, 1).refusal == resolve_refusal::entry_beyond_query,
          "a virtual position past the query's own is refused");
}

// ---------------------------------------------------------------------------------------------
// 7. PRESENCE.
// ---------------------------------------------------------------------------------------------
void presence_rules() {
    ledger board = with_cached_cells(1, 3);

    std::vector<view_entry> no_own = {entry(0, 0, 0), entry(0, 1, 1), entry(0, 2, 2)};
    const attention_request missing = request(0, 3, 3, no_own);
    check(board.resolve(&missing, 1).refusal == resolve_refusal::missing_own_entry,
          "a view without its own cell is refused: the softmax would normalise over the past "
          "without the present");

    std::vector<view_entry> twice = {entry(0, 0, 0), entry(0, 3, 3), entry(0, 3, 3)};
    const attention_request dup = request(0, 3, 3, twice);
    const resolution r = board.resolve(&dup, 1);
    check(r.refusal == resolve_refusal::unordered_virtual_pos ||
              r.refusal == resolve_refusal::duplicate_own_entry,
          "claiming the own cell twice is refused -- by order first, since the second copy repeats "
          "a virtual position, and by identity if the order were relaxed");

    ledger cached;
    cached.set_n_sequences(1);
    cached.set_capacity(64);
    check(cached.put(cell_id{0, 5}, 60) == resolve_refusal::ok, "the token is already written");
    std::vector<view_entry> later = {entry(0, 5, 5), entry(0, 6, 6)};
    const attention_request rr = request(0, 6, 6, later);
    check(cached.resolve(&rr, 1).refusal == resolve_refusal::ok,
          "querying a LATER token against a cached one is fine -- that is a normal decode");

    std::vector<view_entry> self_cached = {entry(0, 4, 4), entry(0, 5, 5)};
    const attention_request sc = request(0, 5, 5, self_cached);
    check(cached.resolve(&sc, 1).refusal == resolve_refusal::query_already_in_cache,
          "but querying a token that is ITSELF in the cache is a re-decode, and refused: it would "
          "pass the own-entry test for the wrong reason");
}

// ---------------------------------------------------------------------------------------------
// 8. BATCH RULES.
// ---------------------------------------------------------------------------------------------
void batch_rules() {
    ledger board = with_cached_cells(3, 4);   // cells (0,0)..(0,3); every query below is at 4+

    std::vector<view_entry> a = {entry(0, 0, 0), entry(0, 4, 4)};
    std::vector<view_entry> b = {entry(0, 1, 1), entry(0, 5, 5)};
    attention_request both[2] = {request(0, 4, 4, a), request(0, 5, 5, b)};
    check(board.resolve(both, 2).refusal == resolve_refusal::duplicate_query_seq,
          "two views for one sequence in one batch is refused: which of them is the reader?");

    std::vector<view_entry> c = {entry(0, 0, 0), entry(0, 4, 4)};
    attention_request same[2] = {request(0, 4, 4, c), request(1, 4, 4, c)};
    const resolution dup_pos = board.resolve(same, 2);
    check(dup_pos.refusal == resolve_refusal::entry_not_cached,
          "the same POSITION in two sequences is not a duplicate -- (seq, pos) is the identity -- "
          "so this fails later, on the second view's uncached own cell, and that is the point: the "
          "batch check let it through");

    std::vector<view_entry> good = {entry(0, 0, 0), entry(0, 4, 4)};
    std::vector<view_entry> bad = {entry(0, 1, 1), entry(0, 9, 9)};   // 0,9 is not cached
    attention_request mixed[2] = {request(0, 4, 4, good), request(2, 2, 2, bad)};
    const resolution partial = board.resolve(mixed, 2);
    check(partial.refusal != resolve_refusal::ok, "a batch with one bad view is refused");
    check(partial.views.empty(),
          "and NO views are handed back: a partial answer is indistinguishable from a complete one "
          "at the call site");
    check(partial.offending_view == 1,
          "and the refusal names which view failed: the fork's bare false costs a debugging "
          "session, this does not");
}

// ---------------------------------------------------------------------------------------------
// 9. CACHE-LEVEL PRECONDITIONS.
// ---------------------------------------------------------------------------------------------
void cache_level_preconditions() {
    ledger multi = with_cached_cells(1, 2);
    multi.set_n_stream(2);
    std::vector<view_entry> entries = {entry(0, 0, 0), entry(0, 1, 1), entry(0, 2, 2)};
    const attention_request r = request(0, 2, 2, entries);
    check(multi.resolve(&r, 1).refusal == resolve_refusal::multi_stream,
          "a multi-stream cache is refused: a slot's owner is ambiguous when sequences interleave");

    ledger shifted = with_cached_cells(1, 2);
    shifted.set_shifted(true);
    check(shifted.resolve(&r, 1).refusal == resolve_refusal::shifted_cache,
          "a shifted cache is refused: stored positions no longer mean what they did");

    // The configuration is part of the epoch: changing it invalidates resolutions taken under the
    // old answer, even though no cell moved.
    ledger config = with_cached_cells(1, 2);
    std::vector<view_entry> e2 = {entry(0, 0, 0), entry(0, 1, 1), entry(0, 2, 2)};
    const attention_request r2 = request(0, 2, 2, e2);
    const resolution before = config.resolve(&r2, 1);
    check(before.refusal == resolve_refusal::ok, "the view resolves under the original config");
    check(config.still_valid(before), "and is valid there");
    config.set_shifted(true);
    check(!config.still_valid(before),
          "and invalid once the cache is declared shifted: a resolution's meaning depends on the "
          "configuration it was taken under, not only on the cells");
}

// ---------------------------------------------------------------------------------------------
// 10. COUNTS AND BOUNDS.
// ---------------------------------------------------------------------------------------------
void counts_and_bounds() {
    ledger board = with_cached_cells(1, 2);   // capacity 18, holds (0,0),(0,1)
    std::vector<view_entry> entries = {entry(0, 0, 0), entry(0, 1, 1), entry(0, 2, 2)};
    const attention_request r = request(0, 2, 2, entries);

    check(board.resolve(nullptr, 1).refusal == resolve_refusal::empty_batch, "a null batch");
    check(board.resolve(&r, 0).refusal == resolve_refusal::empty_batch, "an empty batch");

    attention_request empty_view = r;
    empty_view.n_entries = 0;
    check(board.resolve(&empty_view, 1).refusal == resolve_refusal::bad_entry_count,
          "a view with no entries");

    attention_request null_entries = r;
    null_entries.entries = nullptr;
    check(board.resolve(&null_entries, 1).refusal == resolve_refusal::bad_entry_count,
          "a view with null entries");

    attention_request too_many = r;
    too_many.n_entries = 19;   // one past the declared capacity of 18
    check(board.resolve(&too_many, 1).refusal == resolve_refusal::bad_entry_count,
          "more entries than the cache can address is refused before any lookup: a view cannot "
          "read more cells than exist");

    // With no capacity configured the bound is `occupied + 1`, because the view names its own cell
    // and that cell is not written yet. Both bounds are real; the fork uses capacity.
    ledger unset;
    unset.set_n_sequences(1);
    unset.put(cell_id{0, 0}, 5);
    std::vector<view_entry> two = {entry(0, 0, 0), entry(0, 1, 1)};
    const attention_request fits = request(0, 1, 1, two);
    check(unset.resolve(&fits, 1).refusal == resolve_refusal::ok,
          "occupied + 1 entries is the ordinary decode, not an overrun");
    attention_request over = fits;
    over.n_entries = 3;
    check(unset.resolve(&over, 1).refusal == resolve_refusal::bad_entry_count,
          "but occupied + 2 is refused even with no capacity declared");

    attention_request bad_seq = r;
    bad_seq.query_seq = 5;
    check(board.resolve(&bad_seq, 1).refusal == resolve_refusal::bad_query_seq,
          "a query sequence outside the cache");

    std::vector<view_entry> neg = {entry(0, 0, 0), entry(0, -1, -1)};
    const attention_request negative = request(0, 2, 2, neg);
    check(board.resolve(&negative, 1).refusal == resolve_refusal::bad_key_pos,
          "a negative entry position");

    std::vector<view_entry> bad_entry_seq = {entry(0, 0, 0), entry(7, 1, 1)};
    const attention_request bes = request(0, 2, 2, bad_entry_seq);
    check(board.resolve(&bes, 1).refusal == resolve_refusal::bad_key_seq,
          "an entry naming a sequence outside the cache");
}

// ---------------------------------------------------------------------------------------------
// 11. DUPLICATE PHYSICAL.
// ---------------------------------------------------------------------------------------------
void two_entries_naming_one_slot_are_refused() {
    ledger board;
    board.set_n_sequences(2);
    board.set_capacity(64);
    check(board.put(cell_id{0, 0}, 30) == resolve_refusal::ok, "one cell at slot 30");
    // Two DIFFERENT cells cannot share a slot (`put` refuses), so the reachable form is the same
    // cell twice with different virtual positions -- which the batch's identity rules do not
    // catch, because the cell is legitimately one cell.
    std::vector<view_entry> entries = {entry(0, 0, 0), entry(0, 0, 1), entry(1, 0, 2)};
    const attention_request r = request(1, 1, 2, entries);
    check(board.resolve(&r, 1).refusal == resolve_refusal::duplicate_physical,
          "the same cell twice resolves to one slot twice, and is refused: one of the two would be "
          "read twice and counted twice by the softmax");
}

// ---------------------------------------------------------------------------------------------
// 12. The epoch changes exactly when it must.
// ---------------------------------------------------------------------------------------------
void the_epoch_changes_exactly_when_it_must() {
    ledger board;
    board.set_n_sequences(1);
    board.put(cell_id{0, 0}, 5);
    const std::uint64_t after_put = board.epoch();

    check(board.put(cell_id{0, 0}, 5) == resolve_refusal::ok,
          "a write of the same cell to the same slot is accepted");
    check(board.epoch() == after_put,
          "and does not bump the epoch: a decode that rewrites its own token must not invalidate "
          "resolutions taken before it -- false staleness is as bad as missed staleness");

    check(board.put(cell_id{0, 0}, 6) == resolve_refusal::ok, "a move is accepted");
    check(board.epoch() > after_put, "and bumps the epoch, because the address changed");

    const std::uint64_t before_evict = board.epoch();
    board.evict(cell_id{9, 9});
    check(board.epoch() == before_evict,
          "evicting a cell that is not there does not bump the epoch: a host that clears a range "
          "and then trims it calls this twice, and the second call must be free");

    board.evict(cell_id{0, 0});
    check(board.epoch() > before_evict, "evicting a present cell does");

    const std::uint64_t before_clear = board.epoch();
    board.clear();
    check(board.epoch() == before_clear, "clearing an already-empty ledger is free");
    board.put(cell_id{0, 0}, 5);
    const std::uint64_t before_real_clear = board.epoch();
    board.clear();
    check(board.epoch() > before_real_clear, "clearing a populated one is not");
}

// ---------------------------------------------------------------------------------------------
// 13. A shared prefix: one physical cell, several sequences.
//
// This is the case the exclusive reverse map could not represent, and it is not exotic —
// `llama_kv_cache::seq_cp` (llama-kv-cache.cpp:2631) exists to create it. It walks the source
// sequence's cells and calls `cells.seq_add(i, seq_id_dst)`: the same physical cell now carries two
// sequence ids, so `cell_id{0,5}` and `cell_id{1,5}` are two identities at one address.
//
// Two failures followed from modelling that as a conflict, and both were silent:
//   * a legitimate `put` of the second sequence's cell was refused;
//   * `evict` of the first released the address while the second still named it, so the next `put`
//     of a third cell there was accepted — slot reuse nobody announced.
void a_shared_cell_is_two_identities_at_one_address() {
    ledger board;
    board.set_n_sequences(3);
    board.set_capacity(64);

    check(board.put(cell_id{0, 5}, 40) == resolve_refusal::ok, "the first sequence's cell takes slot 40");
    check(board.put(cell_id{1, 5}, 40) == resolve_refusal::ok,
          "a second sequence's cell at the SAME position and address is accepted: that is a shared "
          "prefix, not a conflict");
    check(board.slot_of(cell_id{0, 5}) == 40 && board.slot_of(cell_id{1, 5}) == 40,
          "both identities name the one address");

    // A third sequence at a DIFFERENT position cannot take that address: this module's rule, and it
    // is what keeps an address from changing hands quietly.
    check(board.put(cell_id{2, 9}, 40) == resolve_refusal::slot_conflict,
          "a cell at a DIFFERENT position is refused the address: position is the discriminator, "
          "because `slot_info` carries one position per slot upstream");

    // Per-owner eviction. The address is released only when the last owner leaves — `seq_rm`'s
    // `seq[i].none()` test.
    check(board.evict(cell_id{0, 5}) == resolve_refusal::ok, "one owner leaves");
    check(board.slot_of(cell_id{1, 5}) == 40, "and the address is still the other's");
    check(board.put(cell_id{2, 9}, 40) == resolve_refusal::slot_conflict,
          "so a third cell still cannot take it: an eviction that freed the address would be the "
          "silent reuse this class exists to refuse");
    check(board.evict(cell_id{1, 5}) == resolve_refusal::ok, "the last owner leaves");
    check(board.put(cell_id{2, 9}, 40) == resolve_refusal::ok,
          "and now the address is free for another cell");
}

// ---------------------------------------------------------------------------------------------
// 14. Sharing is a property of the CACHE; double counting is a property of the VIEW.
//
// The fork keeps a `std::set<int32_t> physical` per view and does `!physical.insert(slot).second` on
// every non-own entry (llama-kv-cache.cpp:2636,2648), so two DIFFERENT cells that resolve to one
// address inside a single view are refused even though their sharing is legitimate. The reason is
// arithmetic, not ownership: one view feeds one softmax, and the same physical row appearing twice in
// it contributes twice.
//
// A test that only exercised the same-cell-twice form would leave "which key does the set hold?"
// undecided, so this case uses two distinct cells whose sharing the ledger accepts.
void two_distinct_cells_at_one_address_are_refused_inside_a_view() {
    ledger board;
    board.set_n_sequences(3);
    board.set_capacity(64);
    // A shared prefix: (0,2) and (1,2) both live at slot 7.
    check(board.put(cell_id{0, 2}, 7) == resolve_refusal::ok, "sequence 0's cell");
    check(board.put(cell_id{1, 2}, 7) == resolve_refusal::ok, "sequence 1's cell, same address");

    // A view over both, from sequence 2. Both entries are cached and both resolve to slot 7, so the
    // gather would read the same row twice.
    // Virtual positions must be STRICTLY increasing, and that is the first rule `resolve` applies —
    // so the two shared cells need different ones. The fixture bug this replaces used the same
    // virtual position twice and was refused as `unordered_virtual_pos`, which is a different fact
    // from the one this case exists to establish: a fixture and its contract have to agree before
    // either can say anything.
    std::vector<view_entry> entries = {entry(0, 2, 1), entry(1, 2, 2), entry(2, 3, 3)};
    const attention_request r = request(2, 3, 3, entries);
    const resolution answer = board.resolve(&r, 1);
    check(answer.refusal == resolve_refusal::duplicate_physical,
          "two distinct cells at one address in one view is refused by SLOT: the set holds "
          "addresses, so sharing does not make the double count legitimate");
    check(answer.views.empty(), "and no partial view is handed back");

    // The same view minus one of the two resolves: the refusal was the duplication, not the sharing.
    std::vector<view_entry> single = {entry(0, 2, 1), entry(2, 3, 3)};
    const attention_request ok = request(2, 3, 3, single);
    check(board.resolve(&ok, 1).refusal == resolve_refusal::ok,
          "naming one of the two shared cells resolves: sharing across sequences is normal");
}

// ---------------------------------------------------------------------------------------------
// 15. The epoch ceiling, made reachable.
//
// 2^64 mutations cannot be looped through, so the seam advances the counter. What is asserted here is
// the STATE at the ceiling: every mutator and every setter refuses, `resolve` still issues (nothing
// can mutate afterwards, so the addresses it names cannot be reused), `still_valid` keeps answering
// true for a resolution taken there, and the seam refuses to rewind.
void the_epoch_ceiling_is_a_state_not_an_argument() {
    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();

    ledger board;
    board.set_n_sequences(1);
    board.set_capacity(64);
    check(board.put(cell_id{0, 0}, 5) == resolve_refusal::ok, "one cell before the ceiling");

    // Forwards to the ceiling; the seam is documented as forward-only and that is asserted below.
    check(board.force_epoch_for_test(kMax), "the seam advances to the ceiling");

    check(board.put(cell_id{0, 1}, 6) == resolve_refusal::epoch_exhausted, "put refuses");
    check(board.evict(cell_id{0, 0}) == resolve_refusal::epoch_exhausted, "evict refuses");
    check(board.clear() == resolve_refusal::epoch_exhausted, "clear refuses");
    check(board.set_n_stream(2) == resolve_refusal::epoch_exhausted, "set_n_stream refuses");
    check(board.set_shifted(true) == resolve_refusal::epoch_exhausted, "set_shifted refuses");
    check(board.set_n_sequences(2) == resolve_refusal::epoch_exhausted, "set_n_sequences refuses");
    check(board.set_capacity(128) == resolve_refusal::epoch_exhausted, "set_capacity refuses");
    check(board.epoch() == kMax, "and the counter did not move: a refused change is not a change");

    // A no-op still reports ok: nothing was asked for, so nothing is owed — including at the ceiling.
    check(board.set_n_stream(1) == resolve_refusal::ok,
          "setting the value it already has is ok even at the ceiling");

    // Resolve at the ceiling: sound, and for a statable reason — nothing can mutate afterwards, so
    // the addresses this names cannot be reused.
    std::vector<view_entry> entries = {entry(0, 0, 0), entry(0, 1, 1)};
    const attention_request r = request(0, 1, 1, entries);
    const resolution answer = board.resolve(&r, 1);
    check(answer.refusal == resolve_refusal::ok, "resolve still issues at the ceiling");
    check(answer.epoch == kMax, "carrying the ceiling as its epoch");
    check(board.still_valid(answer), "and it stays valid, because nothing can invalidate it");

    // The seam refuses to rewind, which is the property that keeps it from being the ABA hole it
    // would otherwise be: an epoch that goes backwards can equal one an old resolution remembers.
    check(!board.force_epoch_for_test(kMax - 1), "the seam refuses to rewind");
    check(board.epoch() == kMax, "and a refused rewind changes nothing");
}

// ---------------------------------------------------------------------------------------------
// 16. Every setter reports whether it took effect.
//
// The `void` versions returned silently at the ceiling, so a caller resolved against a configuration
// it believed it had changed. Each setter is checked here for the two facts a caller acts on: it
// returns `ok` when it changed something, and the epoch moved; setting the current value returns `ok`
// without moving it.
void every_setter_reports_what_it_did() {
    ledger board;
    const std::uint64_t start = board.epoch();

    check(board.set_n_stream(2) == resolve_refusal::ok, "set_n_stream reports ok");
    check(board.set_shifted(true) == resolve_refusal::ok, "set_shifted reports ok");
    check(board.set_n_sequences(4) == resolve_refusal::ok, "set_n_sequences reports ok");
    check(board.set_capacity(32) == resolve_refusal::ok, "set_capacity reports ok");
    check(board.epoch() == start + 4, "each effective change moved the epoch exactly once");

    const std::uint64_t settled = board.epoch();
    check(board.set_n_stream(2) == resolve_refusal::ok &&
              board.set_shifted(true) == resolve_refusal::ok &&
              board.set_n_sequences(4) == resolve_refusal::ok &&
              board.set_capacity(32) == resolve_refusal::ok,
          "re-setting the same values still reports ok: nothing was asked for, nothing is owed");
    check(board.epoch() == settled, "and none of them moved the epoch");
}

// ---------------------------------------------------------------------------------------------
// 17. A resolution belongs to the ledger that issued it, and the epoch cannot say so.
//
// This is the case the previous version of `still_valid` could not pass. It compared
// `answer.epoch == epoch_` and nothing else, so with two ledgers at the same epoch — constructed
// identically, or mutated the same number of times — a resolution from one was accepted by the other.
// The caller then indexed cache B with cache A's slot numbers: in range, existing addresses, wrong
// answer, and nothing downstream to report it.
//
// The fixture is built to make the epoch half VACUOUS rather than merely coincidental: both ledgers
// receive the same configuration and the same cells and the same addresses, so their epochs are
// asserted EQUAL before the cross-check. Without that assertion the case could pass for the wrong
// reason on a day the two epochs happened to differ.
void a_resolution_belongs_to_the_ledger_that_issued_it() {
    ledger a;
    ledger b;
    for (ledger * board : {&a, &b}) {
        check(board->set_n_sequences(1) == resolve_refusal::ok, "configure");
        check(board->set_capacity(64) == resolve_refusal::ok, "configure capacity");
        check(board->put(cell_id{0, 0}, 100) == resolve_refusal::ok, "cache a cell");
        check(board->put(cell_id{0, 1}, 101) == resolve_refusal::ok, "cache another");
    }

    std::vector<view_entry> entries = {entry(0, 0, 0), entry(0, 1, 1), entry(0, 2, 2)};
    const attention_request r = request(0, 2, 2, entries);

    const resolution from_a = a.resolve(&r, 1);
    check(from_a.refusal == resolve_refusal::ok, "ledger A resolves");
    check(b.epoch() == a.epoch(),
          "fixture: the two ledgers are at the SAME epoch, so the epoch half of the check cannot "
          "tell their resolutions apart -- without this assertion the case could pass for the wrong "
          "reason");
    check(from_a.epoch == b.epoch(), "fixture: and the answer's epoch equals B's");

    check(a.still_valid(from_a), "A accepts its own resolution (the non-vacuous half)");
    check(!b.still_valid(from_a),
          "B refuses A's resolution even at an identical epoch: a slot number is an address in ONE "
          "cache, and answering \"still valid\" here is how the wrong addresses get used");

    // And the reverse, so neither direction is an artefact of construction order.
    const resolution from_b = b.resolve(&r, 1);
    check(b.still_valid(from_b), "B accepts its own");
    check(!a.still_valid(from_b), "A refuses B's");

    // A default-constructed resolution has no origin at all, so no ledger answers for it.
    check(!a.still_valid(resolution{}),
          "a resolution nobody issued is refused by every ledger: an epoch of 0 is not a credential");
}

// ---------------------------------------------------------------------------------------------
// 18. Moving a ledger keeps its identity; the moved-from ledger keeps nothing.
//
// Both halves follow from what a move means. The cache is the same cache, so a resolution taken
// before the move describes it correctly and must stay valid — a move that invalidated resolutions
// would make `ledger x = make_ledger();` unusable for any caller that had already resolved. And the
// source is a husk: it owns no identity, so it must not answer for anything, including its own past
// resolutions.
//
// The second half is the one worth testing rather than reasoning about, because a moved-from
// `shared_ptr` is null and the previous `still_valid` — which never looked at identity — would have
// said "valid" for a ledger that no longer describes anything.
void moving_a_ledger_keeps_its_identity_and_not_its_neighbours() {
    ledger original;
    check(original.set_n_sequences(1) == resolve_refusal::ok, "configure");
    check(original.set_capacity(64) == resolve_refusal::ok, "configure capacity");
    check(original.put(cell_id{0, 0}, 70) == resolve_refusal::ok, "cache a cell");

    std::vector<view_entry> entries = {entry(0, 0, 0), entry(0, 1, 1)};
    const attention_request r = request(0, 1, 1, entries);
    const resolution before = original.resolve(&r, 1);
    check(before.refusal == resolve_refusal::ok, "the original resolves");
    check(original.still_valid(before), "and accepts its own answer");

    ledger moved = std::move(original);

    check(moved.still_valid(before),
          "the moved-to ledger accepts it: the move carried the identity, and a move does not "
          "change the cache");
    check(moved.slot_of(cell_id{0, 0}) == 70, "and the cells moved with it");
    check(!original.still_valid(before),
          "the moved-from ledger accepts nothing, its own past answer included: it owns no identity, "
          "so it describes no cache");

    // AND IT IS STILL A LEDGER — which it was not, before this round.
    //
    // Measured on the previous build: `original` accepted configuration and writes after the move
    // (both `ok`, epoch advancing), and `original.resolve(...)` returned `ok` with an `origin` copied
    // from a null token — so `still_valid` refused the answer it had just produced. Reporting success
    // and then disowning the result is worse than reporting nothing, because the caller believes it
    // holds a resolution.
    //
    // The fix mints a fresh identity at the first resolve. What is asserted here is both halves: the
    // revived ledger answers for its OWN answers, and it still does not answer for the ones that
    // travelled with the move.
    check(original.set_n_sequences(1) == resolve_refusal::ok, "the moved-from ledger can be configured");
    check(original.set_capacity(64) == resolve_refusal::ok, "and given a capacity");
    check(original.put(cell_id{0, 0}, 71) == resolve_refusal::ok, "and written to");
    const resolution revived = original.resolve(&r, 1);
    check(revived.refusal == resolve_refusal::ok, "and it resolves");
    check(original.still_valid(revived),
          "and accepts that answer: a resolution it produced must not be disowned by the ledger that "
          "produced it");
    check(!original.still_valid(before),
          "while the pre-move answer stays refused — it belongs to the destination, and two ledgers "
          "must not both claim it");
    check(!moved.still_valid(revived),
          "and the destination refuses the revived ledger's answer, in the other direction");

    // ---- THE SEAL: an edited copy is not the payload that was issued ----
    //
    // The previous round made `origin` private, which stopped a caller from FABRICATING an answer,
    // and left the payload editable. `views`, `refusal` and `epoch` are public, a plain copy carries
    // all three unchanged, so this compiled, ran, and was accepted:
    //
    //     resolution forged = moved.resolve(&r, 1);   // genuine origin, epoch and refusal
    //     forged.views[0].spans[0].indices[0] = 999;  // not the address that was issued
    //     still_valid(forged)                         // true, before this round
    //
    // Each of these was a runtime assertion in the previous two rounds, and each is now a COMPILE-TIME
    // one — because the thing being asserted is that the expression does not exist. This is the whole
    // shift: detection became prevention, and a prevention is checked by the compiler at the site
    // instead of by a hash at run time.
    {
        const resolution genuine = moved.resolve(&r, 1);
        check(genuine.refusal == resolve_refusal::ok && genuine.views.size() == 1 &&
                  genuine.views[0].spans.size() == 1 && !genuine.views[0].spans[0].indices.empty(),
              "fixture: the answer has an index, which the compile-time checks below name");

        const resolution copy = genuine;             // copying a RESOLUTION is fine
        check(moved.still_valid(copy), "an unedited copy is valid: the payload is the same value");

        // THE FOUR EDITS THAT USED TO PASS. Each `static_assert` below would fail to compile if the
        // corresponding mutation were expressible, so the build itself is the test — and the errors the
        // previous build produced at these exact four expressions are the evidence that the property
        // holds: "assignment of read-only location", "assignment of member ... in read-only object",
        // "passing 'const std::vector<int>' as 'this' argument discards qualifiers".
        //
        // A runtime case cannot express this. Asking `still_valid(edited)` requires first constructing
        // `edited`, and constructing it is what the type system refuses — so the assertion has to move
        // to where the refusal happens, which is the compiler.
        static_assert(!std::is_assignable<decltype((std::declval<resolution &>().views[0]
                                                       .spans[0]
                                                       .indices[0])),
                                         std::int32_t>::value,
                      "one index cannot be rewritten: that is the substitution a copy made possible, "
                      "and the one origin and epoch could never see");
        static_assert(!std::is_assignable<decltype((std::declval<resolution &>().views[0]
                                                       .spans[0]
                                                       .effective_pos)),
                                         std::int32_t>::value,
                      "the effective position cannot be adjusted: the rotation offset is part of what "
                      "was issued, not a detail a caller may tune");
        static_assert(!std::is_assignable<decltype((std::declval<resolution &>().views[0]
                                                       .spans[0]
                                                       .indices)),
                                         std::vector<std::int32_t>>::value,
                      "the index vector cannot be replaced: no re-sort, no dedupe, no truncated copy");
        static_assert(!std::is_assignable<decltype((std::declval<resolution &>().views[0]
                                                       .query_pos)),
                                         std::int32_t>::value,
                      "and a view's query coordinate cannot be edited either: the view's identity is "
                      "part of the payload a caller reads");
        static_assert(!std::is_assignable<decltype((std::declval<resolution &>().epoch)),
                                         std::uint64_t>::value,
                      "nor the epoch -- the field the 64-bit fold never covered, so a forger who "
                      "rewrote it passed both the epoch check and the seal");
        static_assert(!std::is_assignable<decltype((std::declval<resolution &>().refusal)),
                                         resolve_refusal>::value,
                      "nor the verdict: an answer cannot be promoted to success after the fact");

        // The payload a caller holds IS the payload that was issued, and there is nothing left to
        // check at run time to prove it. What CAN be checked is that the escape hatch is explicit:
        // `const_cast` compiles, and naming it here is what keeps the comment above from reading as
        // "impossible" when the honest word is "not expressible without a deliberate cast".
        check(true, "`const_cast` remains available and is the caller's deliberate act; a library "
                    "cannot defend against it and does not claim to");
    }

    check(original.still_valid(before) == false,
          "and the pre-move answer is still refused by the source, immutability or no immutability");

    // A resolution nobody issued is not a resolution, and `refusal` now says so.
    //
    // It used to default to `ok` with zero views: the documented path was safe (`still_valid` refused
    // it, no origin), but `r.refusal == ok` passed — the first check a caller writes and the one that
    // survives review. The default is `unset`, so both questions give the same answer.
    {
        const resolution nothing;
        check(nothing.refusal == resolve_refusal::unset,
              "a default-constructed resolution does not claim success");
        check(std::string(tq::reader::describe(nothing.refusal)).find("never issued") !=
                  std::string::npos,
              "and its description says it was never issued, rather than naming a cache condition");
        check(!original.still_valid(nothing), "and no ledger validates it");
    }

    // WHAT IS NO LONGER POSSIBLE, and why there is no runtime case for it: forging an answer from a
    // genuine one. `resolution::origin` used to be public, so a caller holding any real resolution
    // could copy its origin and epoch into a hand-built resolution with invented views — measured
    // accepted, with slot 999 substituted for the real 10. The member is private now, so the forgery
    // does not compile; the compiler is the test, at every use site, and a runtime assertion here
    // could only re-derive what the access specifier already enforces.
}

// ---------------------------------------------------------------------------------------------
// 19. A failed insertion leaves the two maps agreeing, witnessed rather than argued.
//
// `put` records the cell's claim on the slot in the reverse map BEFORE it adds the cell to the
// forward map. If the second step fails and the first is not undone, `cells_` no longer names the
// slot but `slots_` still says this cell owns it — and that membership is permanent: the address can
// never be given to another cell at a different position, and nothing reports why. The rollback
// exists for that; this case makes it run.
//
// The witness has to be a DIFFERENT POSITION, because a lingering membership of the same position
// would be accepted as legitimate sharing and the test would pass while the rollback was broken.
void a_failed_insert_leaves_the_maps_agreeing() {
    ledger board;
    check(board.set_n_sequences(2) == resolve_refusal::ok, "configure");
    check(board.set_capacity(64) == resolve_refusal::ok, "configure capacity");
    const std::uint64_t before = board.epoch();

    board.fail_next_cell_insert_for_test();
    bool threw = false;
    try {
        board.put(cell_id{0, 1}, 55);
    } catch (const std::bad_alloc &) {
        threw = true;
    }
    check(threw, "the injected failure propagates: put does not swallow an allocation failure");
    check(!board.holds(cell_id{0, 1}), "and the cell that failed to insert is not in the ledger");
    check(board.epoch() == before, "and the epoch did not move: a failed mutation is not a mutation");

    // THE HALF THAT MATTERS. If the rollback had not run, cell (1,9) — a different position, so not
    // covered by the sharing rule — would be refused this address forever, with `slot_conflict` as
    // the only explanation and no way for a caller to clear it.
    check(board.put(cell_id{1, 9}, 55) == resolve_refusal::ok,
          "the address is free: the failed put left no membership behind, so another position's "
          "cell can take it");
    check(board.slot_of(cell_id{1, 9}) == 55, "and it belongs to the cell that took it");

    // And the ledger is still usable for the cell that failed, on a later attempt.
    check(board.put(cell_id{0, 1}, 56) == resolve_refusal::ok,
          "the cell that failed can be written on a later attempt, at a different address");
}

int main() {
    a_plain_view_resolves_to_slots_in_entry_order();
    a_reused_slot_is_never_readable_through_the_old_cell();
    taking_a_live_cells_slot_is_refused();
    a_migrated_cell_invalidates_the_old_resolution();
    cross_sequence_reads_are_allowed_but_must_be_present_and_distinct();
    out_of_order_virtual_positions_are_refused();
    presence_rules();
    batch_rules();
    cache_level_preconditions();
    counts_and_bounds();
    two_entries_naming_one_slot_are_refused();
    the_epoch_changes_exactly_when_it_must();
    a_shared_cell_is_two_identities_at_one_address();
    two_distinct_cells_at_one_address_are_refused_inside_a_view();
    the_epoch_ceiling_is_a_state_not_an_argument();
    every_setter_reports_what_it_did();
    a_resolution_belongs_to_the_ledger_that_issued_it();
    moving_a_ledger_keeps_its_identity_and_not_its_neighbours();
    a_failed_insert_leaves_the_maps_agreeing();

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::fprintf(stderr,
                 "index ledger: resolves logical cells to physical slots, refuses reuse, migration "
                 "and cross-sequence mistakes by name\n");
    return 0;
}
