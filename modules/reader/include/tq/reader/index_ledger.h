// tq/reader/index_ledger.h — the reader lane's physical index ledger.
//
// THE CONTRACT THIS FILE EXISTS TO ENFORCE, quoted from the fork
// -------------------------------------------------------------
// `src/llama-attention-view.h` says, in one sentence, above the owned-view struct:
//
//     "Logical identities are resolved after KV slot preparation.
//      No physical index survives a decode call."
//
// and `llama_kv_cache::prepare_attention_views` is the function that makes it true: it takes
// logical `(key_seq, key_pos)` pairs, looks each one up in the live cache, and returns a fresh set
// of physical slot indices that the caller uses immediately and throws away.
//
// That sentence is the whole design, and it is a DISCIPLINE, not a mechanism — the cache cannot
// stop a caller from keeping the integers. So this file provides the mechanism half: the durable
// identity here is `cell_id{seq, pos}`, never a slot number, and a resolved span carries the epoch
// it was computed in so that use after a mutation is detectable rather than merely forbidden.
//
// WHY A SLOT NUMBER IS NOT AN IDENTITY
// ------------------------------------
// A KV cache slot is an address. It gets reused: eviction frees it, a new write takes it, and a
// migration move reassigns it without any write at all. Three failures follow, and they are
// different, so they get different machinery:
//
//   1. Two identities, one slot, sequentially. `evict(A)` then `put(B, slot_A)` — a consumer still
//      holding A's resolved index reads B's data and gets a finite, plausible, wrong answer.
//      Defence: `put` refuses to take a slot that another cell still owns. Reuse must be
//      announced by evicting first, so it is a decision rather than an accident.
//
//   2. One identity, two slots. `put(A, 5)` then `put(A, 7)` — A moved. Any resolution computed
//      between the two names slot 5, which no longer holds A.
//      Defence: every mutation bumps the epoch, and a resolution carries the epoch it read.
//
//   3. Nothing at all. The host reassigns a slot underneath this ledger without telling it — a
//      compaction, a defragmentation, a backend-side reallocation. This module cannot see that,
//      and says so rather than pretending otherwise: (2)'s epoch only catches mutations that pass
//      through here. That is why the fork's sentence is a discipline for the caller and this file
//      only strengthens it.
//
// WHAT A CALLER MUST DO, in the order the fork actually uses
// ---------------------------------------------------------
//   1. the previous decodes' cells are already in the cache (that is what `put` records);
//   2. call `resolve()` ONCE for the whole batch — BEFORE this decode writes its own token;
//   3. hand the spans to the graph, then run the decode, which writes this token's K/V cell;
//   4. `put` that cell afterwards, so the next decode can attend to it;
//   5. discard the spans. Do not cache, do not compare across decodes, do not store them.
//
// THE ORDER OF (2) AND (3) IS THE WHOLE OF `query_already_in_cache`, and an earlier revision of this
// comment had it backwards — it said "finish KV slot preparation (write this decode's key/value
// cells)" and then resolve. A caller who follows that instruction puts the query's own cell into the
// ledger first, every decode then hits `query_already_in_cache`, and the only correct reading of that
// refusal ("this token is already attended to") becomes unreachable. The check is faithful to the
// fork and the instruction was wrong, which is the worse of the two mistakes: a wrong check fails
// loudly, a wrong instruction is obeyed.
//
// The fork's own sequence, read from `llama_context::set_attention_views` (llama-context.cpp:2957):
// the host resolves first, stores the owned views on the cache, and only then calls `decode`, whose
// graph writes the token's cell. That is why `prepare_attention_views` can test
// `cells.count({query_seq, query_pos}) == 0` and mean "not yet written" rather than "never written".
//
// `resolve()` refuses the whole batch rather than returning a partial answer, which is the fork's
// behaviour too (`prepare_attention_views` has one `return false` for every failure and never
// reports which view failed — this version names it, because the fork's silence costs a debugging
// session).
//
// NOT A THREAD-SAFE OBJECT. One ledger belongs to one decoding host. The station is synchronous
// and lock-free on that assumption, which is the same one the cache itself makes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace tq::reader {

// The durable identity of a cache cell: which sequence it belongs to, and its position there.
//
// Ordered by (seq, pos) so a ledger's iteration is deterministic — a hash map would report the
// same set in a different order on every run, and a test that compares reported indices would then
// be comparing map internals.
struct cell_id {
    std::int32_t seq = 0;
    std::int32_t pos = 0;
};

inline bool operator<(const cell_id & a, const cell_id & b) {
    return a.seq != b.seq ? a.seq < b.seq : a.pos < b.pos;
}
inline bool operator==(const cell_id & a, const cell_id & b) {
    return a.seq == b.seq && a.pos == b.pos;
}
inline bool operator!=(const cell_id & a, const cell_id & b) { return !(a == b); }

// One entry of a view: the cell to read, and the position it presents to the rotation.
//
// `virtual_pos` is what makes cross-sequence reading possible at all: a token from another
// sequence occupies a different slot in the current sequence's timeline, and the rotation must use
// that offset rather than the token's own position. It is strictly increasing across a view's
// entries (see `unordered_virtual_pos`) and never exceeds the query's own virtual position.
struct view_entry {
    cell_id  id;
    std::int32_t virtual_pos = 0;
};

// One query token's view.
//
// Pointer + count rather than a vector, matching the fork's `llama_attention_view`: the caller
// already holds the entries, and copying them per decode would be work for nobody. The pointer is
// read only during `resolve`.
struct attention_request {
    std::int32_t       query_seq = 0;
    std::int32_t       query_pos = 0;
    std::int32_t       query_virtual_pos = 0;
    const view_entry * entries = nullptr;
    std::size_t        n_entries = 0;
};

// One span of a resolved view: the effective position for the rotation, and the physical slots.
struct resolved_span {
    std::int32_t              effective_pos = 0;
    std::vector<std::int32_t> indices;
};

// A view after resolution. `effective_pos = query_virtual_pos + key_pos - virtual_pos`, computed
// exactly as `prepare_attention_views` computes it, in int64 so the range check is meaningful.
struct resolved_view {
    std::int32_t                query_seq = 0;
    std::int32_t                query_pos = 0;
    std::int32_t                query_virtual_pos = 0;
    std::vector<resolved_span>  spans;
};

// Why a batch was refused. Every one of these is a distinct fact, because the caller's next move
// differs: `entry_not_cached` means the token is not where you think, `duplicate_physical` means
// two of your entries name one slot, `stale_epoch` means you held a resolution too long.
//
// The names follow the fork's checks one for one; the ones the fork folds into a bare `false` are
// split here, and the split is the reason this module exists.
enum class resolve_refusal {
    ok,
    // NO RESOLUTION WAS PRODUCED. The default value, and the reason it is not `ok`:
    //
    // `resolution r;` used to read as a successful answer with zero views. `still_valid` refused it
    // (no origin), so the documented path was safe — but the same object passed `r.refusal == ok`,
    // which is the first thing a caller checks and the one check that survives a review. A default
    // that claims success is a trap for the reader who checks once.
    //
    // `unset` makes the two questions agree: a resolution nobody produced is not a resolution, and
    // saying so costs one enumerator. `still_valid` still refuses it, and for the same reason as
    // before — no origin — so the two tests are consistent rather than merely both false.
    unset,
    multi_stream,              // n_stream != 1: the cache interleaves sequences, so a slot's owner
                               // is ambiguous
    shifted_cache,             // the cache has been shifted; positions no longer mean what they did
    empty_batch,               // count == 0, or `requests` is null
    batch_too_large,           // count > max_sequences
    bad_query_seq,             // query_seq outside [0, n_sequences)
    bad_query_pos,             // query_pos < 0, or query_virtual_pos < 0
    bad_entry_count,           // n_entries == 0, or entries is null, or n_entries > cache size
    duplicate_query_seq,       // two views for one sequence: which one is the reader?
    duplicate_query_pos,       // the same (seq, pos) queried twice in one batch
    query_already_in_cache,    // the query's own cell is already written: this decode is a re-decode
                               // of a token that exists, and the own-entry test below would pass
                               // for the wrong reason
    bad_key_seq,               // entry seq outside [0, n_sequences)
    bad_key_pos,               // entry pos < 0
    future_key,                // an entry from the query's own sequence sits after the query
    unordered_virtual_pos,     // virtual positions are not strictly increasing: the entries are
                               // order-sensitive and the caller got the order wrong
    entry_beyond_query,        // an entry's virtual position is past the query's own
    effective_pos_out_of_range,// query_virtual_pos + key_pos - virtual_pos left [0, INT32_MAX]
    own_entry_misplaced,       // the query's own entry is not at its virtual position
    duplicate_own_entry,       // more than one entry claims to be the query's own cell
    missing_own_entry,         // no entry is the query's own cell: every view must include its key,
                               // or the softmax normalises over the past without the present
    entry_not_cached,          // the named cell is not in the cache (evicted, or never written)
    negative_physical,         // the cache reported a negative slot for a cell it holds
    duplicate_physical,        // two entries resolve to one slot: one of them would be read twice
    slot_conflict,             // put(): the slot is still owned by a different cell
    stale_epoch,               // a resolution was used after the ledger moved

    // THE EPOCH IS FINITE, and the ceiling is refused instead of wrapped.
    //
    // `epoch_` is a `std::uint64_t` bumped by every mutation, and `still_valid` decides staleness by
    // comparing it. A counter that wraps returns to a value an ancient resolution remembers, and
    // that resolution would then pass the test while naming addresses that have since been reused —
    // the ABA failure this whole class exists to prevent, in the one field whose entire job is to
    // detect change. "2^64 mutations will not happen" is a statement about run length, not a proof;
    // the same argument was rejected for the whiteboard's epoch, and accepting it here would be an
    // inconsistency rather than a decision.
    //
    // So at the ceiling the ledger stops: `put`, `evict` and `clear` refuse with this code, `resolve`
    // refuses, and the `set_*` calls do not take effect. Nothing can be observed as stale because
    // nothing can be issued — the alternative, refusing to *issue* resolutions while still mutating,
    // is worse: a caller would see a mutation "succeed" and a later resolve refuse with no
    // explanation that ties the two.
    //
    // A `std::uint64_t` of mutations is not reachable from a decode loop, so this is a correctness
    // bound rather than an operational concern. It is refused rather than asserted for the ordinary
    // reason: an assertion in a library kills a host that cannot do anything about it.
    epoch_exhausted,
};

const char * describe(resolve_refusal refusal);

// The answer to one `resolve()`. Carries the epoch it read, so use after a mutation is detectable.
// AN ANSWER, ONCE ISSUED, CANNOT BE EDITED — and the `const` on these members is the whole of that.
//
// Three rounds tried to *detect* edits instead of preventing them:
//
//   1. `still_valid` compared an epoch. A copy of a genuine answer passed it, with different indices.
//   2. `origin` was made private. That stopped FABRICATION and left EDITING: `views` was public, so
//      `resolution f = genuine; f.views[0].spans[0].indices[0] = 999;` still passed.
//   3. a 64-bit FNV fold over the payload ("seal") was checked. That is a change detector, not an
//      authenticator, and calling it "true only for the unmodified payload" was a probability
//      claim dressed as a guarantee:
//        * two different payloads CAN fold to one value — 64 bits, no collision argument available
//          to a library that does not control the inputs;
//        * an in-process caller can recompute the fold, so the seal stopped nobody;
//        * and it did not cover `epoch` at all, so a forged answer that rewrote the epoch to the
//          ledger's current value passed BOTH the epoch comparison and the seal.
//
// Detection was the wrong instrument. A value that cannot be modified needs no detector, and `const`
// members make the modification impossible for any code that does not deliberately cast it away. The
// seal is therefore gone, along with the claim — not weakened, removed.
//
// What remains true is exactly what can be checked:
//
//   * `still_valid` answers "did THIS ledger issue an answer, and has it not moved since?" — origin
//     and epoch, nothing about the payload, and it now says so in its name (`issued_and_current`);
//   * the payload a caller reads is bit-for-bit the payload that was issued, because there is no
//     operation that could change it;
//   * a copy of a resolution is safe: the copy carries the same immutable payload, so it validates
//     exactly as the original does. There is nothing to corrupt and nothing to bless.
//
// A caller who WANTS to change the indices builds a `std::vector<resolved_view>` of its own. That is
// the honest shape: a modified payload is the caller's own data, not a resolution, and the type system
// now says so instead of relying on a hash to notice.
struct resolution {
    // `const` on every field: the payload and the verdict are fixed at construction, and the only way
    // to violate that is `const_cast`, which is a deliberate act a library cannot defend against and
    // should not pretend to.
    const resolve_refusal      refusal;
    const std::uint64_t        epoch;          // the ledger's epoch when this was computed
    const std::size_t          offending_view; // which view refused; 0 when ok
    const std::string          why;            // the refusal in words
    const std::vector<resolved_view> views;

    // A resolution nobody issued. It exists because `resolution r;` has to mean something, and
    // "unset" is the only honest meaning.
    resolution()
        : refusal(resolve_refusal::unset), epoch(0), offending_view(0), why(), views() {}

    // WHICH LEDGER ANSWERED — and without it the epoch alone is not a validity test.
    //
    // `still_valid` used to compare `answer.epoch == epoch_` and nothing else. Two independently
    // constructed ledgers both sit at epoch 0 until they are configured, and any two that receive
    // the same number of mutations sit at the same epoch: a resolution from one then passes the
    // other's check, and the caller uses ledger A's slot numbers against cache B. The numbers are
    // in range, the addresses exist, and the answer is wrong — the failure this whole class exists
    // to refuse, arrived at through the one field that was supposed to prevent it.
    //
    // The token is a `weak_ptr` to an object owned by the ledger that issued the answer. Pointer
    // identity has the two properties an epoch cannot: it is unique without a counter (so nothing
    // to wrap, and no ABA to argue about), and it is unforgeable from the outside (a caller cannot
    // fabricate a `shared_ptr` that compares equal to a live ledger's).
    //
    // `weak_ptr` rather than `shared_ptr` because a resolution must not keep a ledger alive. An
    // expired token means the issuing ledger is gone, and every question about staleness then has
    // one answer: no.
    //
    // PRIVATE, and that is the correction this round made. It used to be a public member, which made
    // the sentence above false: a caller holding ANY genuine resolution could read its `origin` and
    // its `epoch`, build a second `resolution` with invented `views`, and hand it back — and
    // `still_valid` would accept it, because both fields it checks were copied from a real answer.
    // Measured before the fix: a resolution naming slot 999 in place of the real 10 was accepted.
    //
    // What `still_valid` establishes is narrower than "these indices are the ones I computed": it is
    // "this ledger issued an answer at this epoch". Authenticating the payload as well would need the
    // views to be sealed into the token, which means a digest over caller-visible data — a different
    // design, and one that buys nothing here, because the caller who would fabricate a payload is the
    // caller holding the only key. What the private member does buy is the removal of a documented
    // capability that was not true.
    //
    // A caller cannot name this member at all; that is checked by the compiler at every use site,
    // which is why there is no runtime case for it.
private:
    friend class ledger;

    // THE ONLY WAY TO MAKE ONE. Constructing a resolution with all fields at once is what allows them
    // to be `const`: there is no sequence of statements in which a half-built answer exists, and the
    // previous design — build a default, then assign five fields — is exactly the shape that let a
    // caller do the same to a copy.
    resolution(resolve_refusal verdict, std::uint64_t at_epoch, std::size_t offending,
               std::string reason, std::vector<resolved_view> resolved, std::weak_ptr<const int> issuer)
        : refusal(verdict),
          epoch(at_epoch),
          offending_view(offending),
          why(std::move(reason)),
          views(std::move(resolved)) {
        origin = std::move(issuer);
    }

    std::weak_ptr<const int> origin;

    // THE SEAL IS GONE. It is not weakened, it is removed — see the note on `struct resolution` for
    // the three ways "true only for the unmodified payload" was not a guarantee: 64-bit folds collide,
    // an in-process caller recomputes them, and the fold never covered `epoch`, so a forger who also
    // rewrote the epoch passed both checks. Immutability makes detection unnecessary; keeping a
    // detector that cannot authenticate would leave the false claim standing next to the fix.
    //
    // The declaration is gone and so is the definition: this file no longer contains any fold, and
    // nothing in the tree calls one. A comment claiming otherwise would be the same defect in a
    // smaller place.
};

// The ledger: the cache's cell-to-slot mapping, as the reader lane sees it.
//
// ONE SLOT, MANY CELLS — and the previous version of this comment said the opposite.
//
// It declared the reverse map as `map<slot, cell_id>` and called a second claim on one address a
// conflict. That model is false, and the fork's own source says so:
//
//   llama_kv_cache::seq_cp (llama-kv-cache.cpp:2631) does NOT copy cells. It walks the source
//   sequence's cells and calls `cells.seq_add(i, seq_id_dst)` — the same physical cell now carries
//   two sequence ids. And `llama_kv_cells::seq_rm` (llama-kv-cells.h:289) only frees the cell when
//   `seq[i].none()`: the cell's sequences are a `std::bitset<LLAMA_MAX_SEQ>`, and one owner leaving
//   does not release the address for anyone else.
//
// That is not an obscure path. A shared prefix — the ordinary thing `seq_cp` exists for — puts two
// sequences in one physical cell, which makes `cell_id{0, 5}` and `cell_id{1, 5}` two *different*
// identities at one address. An exclusive reverse map cannot represent it, and the two failures it
// produced were both silent:
//
//   1. `put(cell{1,5}, 7)` after `put(cell{0,5}, 7)` was refused as a conflict. A caller doing what
//      the cache does every day was told it was doing something illegal.
//   2. `evict(cell{0,5})` erased slot 7 from the reverse map while `cell{1,5}` still held it, so
//      `slot_of({1,5}) == 7` with no owner recorded — and the next `put` of a third cell to 7 was
//      accepted, handing a live cell's address away. That is the exact silent reuse this class
//      exists to refuse, arrived at by the fix for the previous point.
//
// So the reverse map is `map<slot, set<cell_id>>`, and ownership is per cell:
//
//   * `put` refuses a slot only when another cell holds it AT A DIFFERENT POSITION. Two cells of
//     one position in different sequences are a shared cell and are accepted — the only sharing the
//     cache itself performs (`slot_info` carries one position and a set of sequences).
//   * `evict(cell)` removes THAT cell's ownership. The address is released only when the last owner
//     leaves, which is what `seq_rm`'s `seq[i].none()` test means.
//   * `duplicate_physical` in `resolve` keeps the fork's rule EXACTLY, and the rule is by SLOT, not
//     by cell. `prepare_attention_views` carries a `std::set<int32_t> physical` per view and does
//     `!physical.insert(found->second).second` on every non-own entry (llama-kv-cache.cpp:2636,
//     2648) — so two *different* cells that happen to resolve to one address inside a single view
//     are refused, even though sharing that address across sequences is legitimate.
//     That is not an inconsistency: the double count is what is being prevented. One view feeds one
//     softmax, and the same physical row appearing twice in it contributes twice regardless of which
//     sequences own the cells. Sharing is a property of the CACHE; double counting is a property of
//     the VIEW.
//
// The `slot_conflict` rule that remains (different positions, one address) is this module's, not
// upstream's — a cell has exactly one position there, so the situation cannot arise and the fork
// cannot have a rule about it. It is kept because eviction order is the caller's to state, and
// labelled here so nobody reads it as a fork contract.
class ledger {
public:
    // Configuration. Each is a property of the cache this ledger describes, not of a decode, so it
    // is set once and changing it bumps the epoch like any other mutation.
    //
    // EACH RETURNS A CODE, and that is a correction: they used to be `void` and silently did
    // nothing at the epoch ceiling. A setter that reports success and does not take effect is the
    // failure mode this project keeps removing — the caller then resolves against a configuration
    // it believes it changed. `epoch_exhausted` says so instead.
    resolve_refusal set_n_stream(int n_stream);
    resolve_refusal set_shifted(bool shifted);
    resolve_refusal set_n_sequences(std::size_t count);

    // The cache's cell CAPACITY, which is not the number of cells it currently holds.
    //
    // The fork bounds a view's entry count by `get_size()`, and `get_size()` is how many cells the
    // cache can address — not how many are written. The two differ on every decode: a view names
    // its own token, which is not in the cache yet, plus the tokens already there, so
    // `n_entries == occupied + 1` is the ordinary case and an occupied-count bound would refuse
    // every valid view. That mistake was in this file's first draft and the first test caught it.
    //
    // Zero means "not configured", and then the bound is not policed — a caller that has not said
    // how large its cache is has not asked for that check, and guessing a capacity would be worse
    // than declining to apply it.
    // Returns a code for the same reason as the three above: it used to be `void` and silently
    // early-returned at the ceiling, which is the one setting a caller would then believe had taken
    // effect. Same rule, same reporting.
    resolve_refusal set_capacity(std::size_t cells);

    // THE TEST SEAM. Advances the epoch to a chosen value so the ceiling is reachable.
    //
    // 2^64 mutations cannot be looped through, so a guard that only fires there is untestable by
    // any test that uses the API — which is how a "protective by construction" label ends up
    // covering a guard nobody has run. One setter, marked as what it is, is cheaper than that.
    //
    // It also makes the ceiling testable as a STATE rather than as an argument: set the epoch to the
    // maximum, then assert that every mutator and every setter refuses, that `resolve` still issues
    // (and why that is sound), and that `still_valid` keeps answering true for a resolution taken
    // there. Those are four claims the previous round could only reason about.
    // FORWARD ONLY, and that restriction is the point rather than a courtesy.
    //
    // The seam exists so the ceiling is reachable: 2^64 mutations cannot be looped through, so a
    // guard that only fires there would be untestable by any test that uses the API. Setting the
    // epoch to `kMaxEpoch` is a forward move and is accepted.
    //
    // Rewinding is REFUSED, because a rewind is precisely the ABA failure the ceiling exists to
    // prevent: an epoch that goes backwards can equal one an old resolution remembers, and that
    // resolution then passes `still_valid` while naming addresses reused since. A seam that could do
    // that in a shipping binary would hand any caller the one capability the rest of this class is
    // built to deny — so it cannot, whether or not the caller promises to be a test.
    //
    // Returns false when the request would rewind. A caller that wants the old epoch back must build
    // a new ledger, which is what "the past is not available" means here.
    bool force_epoch_for_test(std::uint64_t epoch) {
        if (epoch < epoch_) {
            return false;
        }
        epoch_ = epoch;
        return true;
    }

    // THE SECOND TEST SEAM: make the next `put()`'s cell insertion fail.
    //
    // `put` has two allocation points — creating the slot node, and adding the cell to its set — and
    // the second one is the one with a rollback, because by then the reverse map already records this
    // cell's claim. Without a way to fail exactly there, the rollback is "protective by construction",
    // which is this project's way of saying nobody has run it.
    //
    // It throws from INSIDE the try, deliberately: the exception then travels through the shipping
    // catch and the shipping `release_membership` call, so what a test exercises is the code a real
    // `bad_alloc` would reach, not a re-enactment of it.
    //
    // One-shot, and it resets itself as it fires, so a test cannot leave a ledger that fails forever
    // by forgetting to clear a flag.
    void fail_next_cell_insert_for_test() { fail_next_cell_insert_ = true; }

    // --- mutations: each one bumps the epoch ---
    //
    // A KV write. `slot` must be free or already this cell's: taking a slot another live cell owns
    // is refused with `slot_conflict`, so reuse has to go through `evict` first and is therefore a
    // decision the caller makes rather than something that happens to it.
    //
    // `slot_conflict` IS AN ADDITION, not a fork rule — and the distinction matters, because a
    // stricter check that pretends to be the contract is how a caller ends up refusing
    // configurations the original accepts. `prepare_attention_views` reads a cell→slot map and has
    // no reverse map, so it cannot notice that two cells claim one address: the second `put` would
    // simply overwrite the first in llama's own bookkeeping, and the divergence would show up later
    // as a wrong number. This ledger keeps the reverse map precisely so that the overwrite is
    // reported at the moment it happens. A host that reassigns a slot underneath this module — a
    // compaction, a defragmentation, a backend-side reallocation — must say so by evicting first,
    // and that requirement is this file's, not upstream's.
    //
    // EXCEPTION SAFETY: strong. The only step that can throw is the first one; every later step is
    // an assignment or an erase on an existing node, and the new-cell path rolls its insertion back
    // if the cell map cannot grow. A `bad_alloc` mid-update would otherwise leave `cells_` naming a
    // slot that no `slots_` entry owns, and the next `put` of a different cell to that address would
    // be accepted — the silent reuse this class exists to refuse, arrived at by a path nobody tests.
    resolve_refusal put(const cell_id & cell, std::int32_t slot);

    // An eviction, or the undo of a reservation. Absent is not an error: a host that clears a
    // range and then trims it will call this twice, and the second call has nothing to do.
    //
    // EVICTION IS PER SEQUENCE, which is the whole reason the reverse map is a set. The fork's
    // `seq_rm(cell, seq)` resets one bit and frees the address only when `seq[i].none()`; so here,
    // removing `{0,5}` from a slot shared with `{1,5}` leaves `{1,5}` holding it. An implementation
    // that erased the slot outright would report the address free while a live cell still names it,
    // and the next `put` of a third cell there would be accepted — silent reuse.
    //
    // Returns `epoch_exhausted` at the ceiling (see that enumerator) and otherwise `ok`. It returns
    // a code rather than nothing because "the eviction did not happen" is a fact the caller must be
    // able to see: a silent no-op would let a host believe a slot was freed while the ledger still
    // refused to reuse it.
    resolve_refusal evict(const cell_id & cell);

    // Everything gone, epoch bumped. A new sequence's decode starting from an empty cache.
    resolve_refusal clear();

    // --- the read ---
    //
    // Pure: no mutation, no epoch bump. The whole batch is validated before any view is resolved,
    // and a refusal returns no partial set of views -- the fork's own property, kept because a
    // partial answer is indistinguishable from a complete one at the call site.
    resolution resolve(const attention_request * requests, std::size_t count) const;

    // May this resolution still be used — ON THIS LEDGER?
    //
    // The caller MUST ask this before reading `views`' indices. It is not automatic: C++ has no way
    // to make an integer index carry its own validity, and pretending otherwise with a proxy type
    // would only move the check somewhere the compiler cannot see either.
    //
    // THE QUESTION HAS TWO HALVES NOW, and the list used to have three because the third one was a
    // seal that could not do its job.
    //
    //   "Did this ledger issue an answer at all?"          — the token in `resolution::origin`.
    //       Without it two ledgers at the same epoch are indistinguishable by number alone, so a
    //       resolution from one was accepted by the other and its slots were used to index a
    //       different cache.
    //   "Has this ledger moved since that epoch?"          — the epoch comparison.
    //
    // The third — "is this the payload it issued?" — is answered by the TYPE rather than by this
    // function: `resolution`'s members are `const`, so the payload a caller holds is the payload that
    // was issued, and no check is needed to establish it. Three rounds tried to detect edits (epoch,
    // private token, 64-bit fold) and each left a way through; the fourth removed the possibility.
    // See the note on `struct resolution`, which also records why the fold never covered `epoch` —
    // the hole a forger walked through while both the epoch check and the fold said yes.
    //
    // So `still_valid` answers provenance and currency, and says nothing about the payload because
    // there is nothing left to say. A caller reading the indices is reading what was issued.
    //
    // A `resolution` taken before a MOVE of the ledger stays valid afterwards: the move carries the
    // identity object along, and a move does not change the cache. A resolution taken before a COPY
    // does not exist, because copying a LEDGER is refused — see the deleted copy operations on the
    // member block. Copying a RESOLUTION is allowed and safe: the copy holds the same immutable
    // payload, so it validates exactly as the original does; an edit to the copy does not "break a
    // seal", it fails to compile.
    bool still_valid(const resolution & answer) const;

    // --- queries, for diagnostics and tests ---
    std::uint64_t epoch() const { return epoch_; }
    std::size_t   size() const { return cells_.size(); }
    bool          holds(const cell_id & cell) const { return cells_.count(cell) != 0; }
    std::int32_t  slot_of(const cell_id & cell) const;   // -1 when absent

private:
    // One check, one place: everything `resolve` may reject, in the order the fork applies it, so
    // a caller reading either file sees the same sequence of questions.
    resolve_refusal check_batch(const attention_request * requests, std::size_t count,
                                std::size_t * offending, std::string * why) const;

    std::map<cell_id, std::int32_t> cells_;
    // THE REVERSE MAP IS A SET PER SLOT, because a physical cell belongs to a `std::bitset` of
    // sequences upstream (llama-kv-cells.h:289-320) and a shared prefix puts several cells of one
    // position at one address. `map<slot, cell_id>` could not represent that, and the two ways it
    // failed — refusing legitimate sharing, and releasing an address a live cell still held — are
    // described on the class.
    std::map<std::int32_t, std::set<cell_id>> slots_;
    std::uint64_t epoch_ = 0;
    int           n_stream_ = 1;
    bool          shifted_ = false;
    std::size_t   n_sequences_ = 0;
    std::size_t   capacity_ = 0;      // 0 = unconfigured; the entry-count bound is then not applied

public:
    // MOVE YES, COPY NO — and both halves are decisions about validity, not about style.
    //
    // A copy would be a second object with the same identity, the same epoch and the same maps: a
    // resolution from one passes the other's `still_valid`, and from the first mutation onwards the
    // two describe different caches while sharing an identity. That is the cross-ledger acceptance
    // this class now refuses, reintroduced by the copy constructor — so the fix is to not have one.
    // A caller that wants a snapshot builds a new ledger and feeds it the cells it needs, which
    // states what the snapshot IS instead of implying it is the same cache.
    //
    // The move is what makes the type usable at all: `ledger board = make_board();` is the ordinary
    // way to construct one, and it must stay legal. A move carries `identity_` along, so a
    // resolution taken before the move stays valid after it — correct, because a move does not
    // change the cache. A moved-from ledger keeps a null identity, so it validates nothing: every
    // answer it is asked about came from somebody else.
    // MOVE YES, COPY NO — and both halves are decisions about validity, not about style.
    //
    // A copy would be a second object with the same identity, the same epoch and the same maps: a
    // resolution from one passes the other's `still_valid`, and from the first mutation onwards the
    // two describe different caches while sharing an identity. That is the cross-ledger acceptance
    // this class now refuses, reintroduced by the copy constructor — so the fix is to not have one.
    // A caller that wants a snapshot builds a new ledger and feeds it the cells it needs, which
    // states what the snapshot IS instead of implying it is the same cache.
    //
    // THE MOVE IS WRITTEN OUT, not `= default`, and that is a correction. The defaulted move claims
    // to leave the source "valid but unspecified" for its maps, and the previous build relied on it
    // being empty: measured, a moved-from ledger accepted `put` and `resolve` and reported `ok`, so
    // its maps were reachable state. A type whose post-move contract is "whatever the standard
    // library felt like" cannot make the promise this one makes — an empty ledger with its own
    // identity — so the move states it: identity travels, maps are emptied, epoch travels (it
    // describes how much this ledger has seen, and the destination is the continuation).
    //
    // A resolution taken before the move stays valid after it (same identity, same epoch, and a move
    // does not change the cache or the immutable payload). The source refills its identity at the
    // first resolve — see `ensure_identity` — and from then on is a distinct, empty ledger that
    // answers for nothing the destination holds.
    ledger() = default;
    ledger(const ledger &) = delete;
    ledger & operator=(const ledger &) = delete;

    ledger(ledger && other) noexcept
        : cells_(std::move(other.cells_)),
          slots_(std::move(other.slots_)),
          epoch_(other.epoch_),
          n_stream_(other.n_stream_),
          shifted_(other.shifted_),
          n_sequences_(other.n_sequences_),
          capacity_(other.capacity_),
          identity_(std::move(other.identity_)) {
        // State the source's end of the contract rather than inheriting "unspecified": a moved-from
        // ledger is an EMPTY ledger with no identity, and `ensure_identity` refills the identity on
        // its first resolve. Clearing the maps is not bookkeeping — it is what makes "the source
        // holds nothing the destination holds" true by construction instead of by observation of
        // what the standard library happened to do.
        other.cells_.clear();
        other.slots_.clear();
        other.identity_.reset();
        other.fail_next_cell_insert_ = false;
    }

    ledger & operator=(ledger && other) noexcept {
        if (this != &other) {
            cells_ = std::move(other.cells_);
            slots_ = std::move(other.slots_);
            epoch_ = other.epoch_;
            n_stream_ = other.n_stream_;
            shifted_ = other.shifted_;
            n_sequences_ = other.n_sequences_;
            capacity_ = other.capacity_;
            identity_ = std::move(other.identity_);
            other.cells_.clear();
            other.slots_.clear();
            other.identity_.reset();
            other.fail_next_cell_insert_ = false;
        }
        return *this;
    }

private:
    // THE LEDGER'S IDENTITY, minted once and never reused.
    //
    // A `shared_ptr` to a one-element object whose value is never read: only its address matters, and
    // address allocation cannot hand the same address to two live objects. That is what makes it
    // unique without a counter — no wrap, no ABA, and nothing to argue about — and what makes it
    // unforgeable from outside, since a caller cannot construct a `shared_ptr` that compares equal to
    // a live ledger's.
    //
    // It is deliberately not `uint64_t`: a counter would need the same exhaustion argument the epoch
    // needed, and a second one is a second place to get wrong. `resolution::origin` holds a weak
    // reference to this, so an expired token means the ledger is gone and the answer is stale.
    //
    // A MOVED-FROM LEDGER HAS NO IDENTITY, and that used to leave it in a state with no honest
    // answer. Measured: after `ledger dst = std::move(src)`, `src` accepted configuration and writes
    // (`ok`, epoch advancing), and `src.resolve(...)` returned `ok` — with `origin` copied from a
    // null token, so `still_valid` refused its own fresh answer. A ledger that reports success and
    // then rejects what it just produced is worse than one that reports nothing: the caller sees a
    // resolution, and every subsequent question about it answers "not mine".
    //
    // `ensure_identity` mints a fresh one at the first resolve, so a moved-from ledger becomes what a
    // move leaves it: an EMPTY ledger with its own identity. The resolutions that travelled with the
    // move still belong to the destination, and this object refuses them — which is the behaviour the
    // move case asserts.
    mutable std::shared_ptr<const int> identity_ = std::make_shared<const int>(0);

    // Mints the identity if a move took it away. Const-ref return so `origin = ensure_identity()`
    // copies a weak reference without a second temporary.
    const std::shared_ptr<const int> & ensure_identity() const {
        if (!identity_) {
            identity_ = std::make_shared<const int>(0);
        }
        return identity_;
    }
    // The seam's one-shot flag; `std::exchange` clears it as it is read, so a throwing `put` cannot
    // leave it set and a caller cannot accidentally create a ledger that fails forever.
    bool fail_next_cell_insert_ = false;
};

} // namespace tq::reader
