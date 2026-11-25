#include "tq/reader/index_ledger.h"

#include <limits>

namespace tq::reader {
namespace {

// Mirrors llama's own bound. Kept as a constant here rather than including llama.h, because this
// module is arithmetic and a llama dependency would make it untestable without a model.
constexpr std::size_t kMaxSequences = 512;   // LLAMA_MAX_SEQ

// The epoch ceiling. Every mutation bumps `epoch_`, and `still_valid` compares it, so a wrap would
// let an ancient resolution pass the test while naming reused addresses. Refused instead of wrapped:
// see `resolve_refusal::epoch_exhausted`.
constexpr std::uint64_t kMaxEpoch = std::numeric_limits<std::uint64_t>::max();

std::string numbered(const char * what, std::int64_t a) {
    return std::string(what) + " " + std::to_string(a);
}

} // namespace

const char * describe(resolve_refusal refusal) {
    switch (refusal) {
        case resolve_refusal::ok:                          return "ok";
        case resolve_refusal::unset:                       return "no resolution was produced: this answer was never issued by a ledger";
        case resolve_refusal::multi_stream:                return "the cache has more than one stream, so a slot's owner is ambiguous";
        case resolve_refusal::shifted_cache:               return "the cache has been shifted, so stored positions no longer mean what they did";
        case resolve_refusal::empty_batch:                 return "the batch is empty";
        case resolve_refusal::batch_too_large:             return "the batch has more views than there are sequences";
        case resolve_refusal::bad_query_seq:               return "a query sequence is outside the cache";
        case resolve_refusal::bad_query_pos:               return "a query position is negative";
        case resolve_refusal::bad_entry_count:             return "a view's entry count is zero, null, or larger than the cache";
        case resolve_refusal::duplicate_query_seq:         return "two views name the same query sequence: which one is the reader?";
        case resolve_refusal::duplicate_query_pos:         return "the same (sequence, position) is queried twice in one batch";
        case resolve_refusal::query_already_in_cache:      return "the query's own cell is already in the cache: this is a re-decode, not a decode";
        case resolve_refusal::bad_key_seq:                 return "an entry names a sequence outside the cache";
        case resolve_refusal::bad_key_pos:                 return "an entry names a negative position";
        case resolve_refusal::future_key:                  return "an entry from the query's own sequence sits after the query";
        case resolve_refusal::unordered_virtual_pos:       return "virtual positions are not strictly increasing: the entries are order-sensitive";
        case resolve_refusal::entry_beyond_query:          return "an entry's virtual position is past the query's own";
        case resolve_refusal::effective_pos_out_of_range:  return "query_virtual_pos + key_pos - virtual_pos left the representable range";
        case resolve_refusal::own_entry_misplaced:         return "the query's own entry is not at the query's virtual position";
        case resolve_refusal::duplicate_own_entry:         return "more than one entry claims to be the query's own cell";
        case resolve_refusal::missing_own_entry:           return "no entry is the query's own cell: the softmax would normalise over the past without the present";
        case resolve_refusal::entry_not_cached:            return "an entry names a cell the cache does not hold (evicted, or never written)";
        case resolve_refusal::negative_physical:           return "the cache reported a negative slot for a cell it holds";
        case resolve_refusal::duplicate_physical:          return "two entries resolve to one slot: one of them would be read twice";
        case resolve_refusal::slot_conflict:               return "the slot is still owned by a different cell; evict it first so reuse is announced";
        case resolve_refusal::stale_epoch:                 return "the resolution was used after the ledger moved";
        case resolve_refusal::epoch_exhausted:             return "the ledger's epoch counter is at its maximum; it accepts no further changes rather than wrapping into a value an old resolution would match";
    }
    return "unknown refusal";
}

// ---------------------------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------------------------

resolve_refusal ledger::set_n_stream(int n_stream) {
    // Bumps the epoch: whether the cache interleaves streams changes what a slot means, so every
    // resolution computed under the old answer is void.
    //
    // EACH SETTER RETURNS A CODE. They used to be `void` and silently did nothing at the ceiling,
    // which is worse than refusing: a caller would then resolve against a configuration it believes
    // it changed, and the mismatch would surface as a wrong number rather than as a refusal. The
    // no-op case (setting the value it already has) still reports `ok`, because nothing was asked
    // for and nothing is owed.
    if (n_stream_ == n_stream) return resolve_refusal::ok;
    if (epoch_ == kMaxEpoch) return resolve_refusal::epoch_exhausted;
    n_stream_ = n_stream;
    ++epoch_;
    return resolve_refusal::ok;
}

resolve_refusal ledger::set_shifted(bool shifted) {
    if (shifted_ == shifted) return resolve_refusal::ok;
    if (epoch_ == kMaxEpoch) return resolve_refusal::epoch_exhausted;
    shifted_ = shifted;
    ++epoch_;
    return resolve_refusal::ok;
}

resolve_refusal ledger::set_n_sequences(std::size_t count) {
    if (n_sequences_ == count) return resolve_refusal::ok;
    if (epoch_ == kMaxEpoch) return resolve_refusal::epoch_exhausted;
    n_sequences_ = count;
    ++epoch_;
    return resolve_refusal::ok;
}

resolve_refusal ledger::set_capacity(std::size_t cells) {
    // Bumps the epoch for the same reason as the other configuration: the entry-count bound is part
    // of what a resolution was validated against, so a resolution taken under one capacity cannot
    // be assumed legal under another.
    if (capacity_ == cells) return resolve_refusal::ok;
    if (epoch_ == kMaxEpoch) return resolve_refusal::epoch_exhausted;
    capacity_ = cells;
    ++epoch_;
    return resolve_refusal::ok;
}

// THE CEILING, stated once so the seven guards above and below mean something together.
//
// At `epoch_ == kMaxEpoch` this ledger stops changing: every mutation refuses with
// `epoch_exhausted`, and every configuration setter returns without taking effect. The reason a
// setter must also stop is the one that is easy to miss — a setter that bumped anyway would wrap the
// counter to 0, and a resolution taken at the very beginning of the ledger's life (epoch 0) would
// then compare equal again and pass `still_valid` while naming addresses that had been reused
// billions of times since. That is the ABA failure in the one field whose whole job is to detect
// change.
//
// What is NOT refused is `resolve`: it still issues a resolution, carrying epoch == kMaxEpoch. That
// is sound, and for a reason worth stating rather than assuming — nothing can mutate afterwards, so
// the addresses that resolution names cannot be reused, and the resolution stays correct for as long
// as it exists. Refusing there as well would be the more obvious choice and would be wrong: it would
// make a frozen ledger unusable for a decode that is perfectly able to proceed.
//
// (Reaching the ceiling takes 2^64 mutations, so this is a correctness bound and not an operational
// concern. It is refused rather than asserted for the reason given on `epoch_exhausted`.)

// ---------------------------------------------------------------------------------------------
// mutations
// ---------------------------------------------------------------------------------------------

// Removes one cell's claim on one slot, dropping the slot node when its last owner leaves.
//
// This is the whole of "eviction is per sequence", in one function so `put` and `evict` cannot
// disagree about it. `llama_kv_cells::seq_rm` (llama-kv-cells.h:289) resets ONE bit of the cell's
// sequence bitset and frees the address only `if (seq[i].none())`; the shape here is the same, with
// the set of owners standing in for the bitset.
//
// Erasing the slot node outright — what the exclusive map did — reports the address free while
// another live cell still names it, and the next `put` of a third cell there is then accepted. That
// is silent slot reuse, arrived at by the fix for a false conflict.
void release_membership(std::map<std::int32_t, std::set<cell_id>> & slots,
                        std::int32_t slot, const cell_id & cell) noexcept {
    const auto found = slots.find(slot);
    if (found == slots.end()) {
        return;
    }
    found->second.erase(cell);
    if (found->second.empty()) {
        slots.erase(found);
    }
}

// Whether any cell OTHER than `cell`, at a DIFFERENT POSITION, already holds `slot`.
//
// Position, not sequence, is the discriminator — and that is what the cache's own structure says.
// `seq_cp` gives several cells of the SAME position to one physical slot (a shared prefix), and that
// is legitimate. Two cells at DIFFERENT positions can never share one slot upstream: `slot_info`
// carries a single `pos`, so the situation cannot arise there and the fork has no rule about it. This
// module keeps the rule because a host that reassigns slots must announce it (evict first) rather
// than let an address change hands quietly — and because the rule is ours, it is documented as ours
// rather than presented as upstream's contract.
bool held_at_other_position(const std::map<std::int32_t, std::set<cell_id>> & slots,
                            std::int32_t slot, const cell_id & cell) {
    const auto found = slots.find(slot);
    if (found == slots.end()) {
        return false;
    }
    for (const cell_id & owner : found->second) {
        if (owner != cell && owner.pos != cell.pos) {
            return true;
        }
    }
    return false;
}

resolve_refusal ledger::put(const cell_id & cell, std::int32_t slot) {
    if (slot < 0) {
        // A negative slot is the cache's "no address" marker; storing it would make `slot_of`
        // return a value that looks like a real index to everyone but the bounds check.
        return resolve_refusal::negative_physical;
    }

    const auto existing = cells_.find(cell);
    if (existing != cells_.end() && existing->second == slot) {
        // Idempotent: a re-write of the same cell to the same address changes nothing, so it must
        // not bump the epoch. A decode that rewrites its own token would otherwise invalidate
        // resolutions taken before it, which would be a false staleness.
        //
        // And it changes no ADDRESS, which is what a resolution names: the fork's own sentence is
        // that no physical index survives a decode call, so a rewrite is the caller's obligation to
        // use the index promptly, not a reason for the ledger to invalidate it. Bumping here would
        // be false staleness, which is as bad as missed staleness.
        return resolve_refusal::ok;
    }

    // SHARING IS ALLOWED; TAKING ANOTHER POSITION'S ADDRESS IS NOT. See held_at_other_position for
    // why position is the discriminator and why this rule is this module's rather than the fork's.
    if (held_at_other_position(slots_, slot, cell)) {
        return resolve_refusal::slot_conflict;
    }

    // The ceiling is tested before anything is written, so a refused call cannot leave a partial
    // update behind. See `resolve_refusal::epoch_exhausted` for why the counter is refused instead
    // of wrapped.
    if (epoch_ == kMaxEpoch) {
        return resolve_refusal::epoch_exhausted;
    }

    // THE FALLIBLE STEPS FIRST, so nothing throws after the maps start to disagree.
    //
    // Two inserts can throw: creating the slot node, and adding this cell to its set. Both happen
    // before any observation changes — `cells_` is untouched until they succeed — and each failure
    // path undoes exactly what that step did. A step that left `cells_` naming a slot the reverse map
    // did not hold would let the next `put` of a different cell to that address be accepted: silent
    // reuse, reached by a path no test exercises without an injected bad_alloc.
    auto node = slots_.emplace(slot, std::set<cell_id>{});
    const bool created_node = node.second;
    try {
        node.first->second.insert(cell);
    } catch (...) {
        if (created_node) {
            slots_.erase(node.first);
        }
        throw;
    }

    if (existing == cells_.end()) {
        try {
            // The seam throws from HERE, so the shipping catch and the shipping rollback are what a
            // test exercises. Injected before the emplace rather than instead of it: an exception
            // from a std::map is not something a test can arrange, and the path being tested is what
            // happens when the insertion does not take.
            if (std::exchange(fail_next_cell_insert_, false)) {
                throw std::bad_alloc();
            }
            cells_.emplace(cell, slot);
        } catch (...) {
            release_membership(slots_, slot, cell);
            throw;
        }
    } else {
        // A move. The new claim is already recorded, so this step only has to drop the old one, and
        // it is noexcept: erase from a set, erase a node, assign an int.
        release_membership(slots_, existing->second, cell);
        existing->second = slot;
    }
    ++epoch_;
    return resolve_refusal::ok;
}

resolve_refusal ledger::evict(const cell_id & cell) {
    const auto found = cells_.find(cell);
    if (found == cells_.end()) {
        // Absent is not an error: clearing a range and then trimming it calls this twice.
        return resolve_refusal::ok;
    }
    if (epoch_ == kMaxEpoch) {
        return resolve_refusal::epoch_exhausted;
    }
    // PER OWNER, not per address. A shared prefix puts several cells of one position at one slot, so
    // removing one of them must leave the address held by the others — `seq_rm`'s `seq[i].none()`
    // test, expressed as "drop the node when its set empties".
    release_membership(slots_, found->second, cell);
    cells_.erase(found);
    ++epoch_;
    return resolve_refusal::ok;
}

resolve_refusal ledger::clear() {
    if (cells_.empty()) {
        return resolve_refusal::ok;
    }
    if (epoch_ == kMaxEpoch) {
        return resolve_refusal::epoch_exhausted;
    }
    cells_.clear();
    slots_.clear();
    ++epoch_;
    return resolve_refusal::ok;
}

std::int32_t ledger::slot_of(const cell_id & cell) const {
    const auto found = cells_.find(cell);
    return found == cells_.end() ? -1 : found->second;
}

// (The 64-bit payload fold that used to live here is deleted, not disabled.)
//
// It was a change detector, and three of its properties made "true only for the unmodified payload"
// false: 64-bit folds collide and this library controls none of the inputs; an in-process caller can
// recompute the fold; and it never covered `epoch`, so a forger who also rewrote the epoch passed
// both the epoch check and the fold. `resolution`'s members are `const` now, so modification is
// unrepresentable and detection has nothing left to do. Keeping an unreachable copy of the
// arithmetic would leave the refuted claim standing next to the fix.

bool ledger::still_valid(const resolution & answer) const {
    // THE ANSWER IS NOW "DID THIS LEDGER ISSUE IT, AT THIS EPOCH" — and nothing about the payload,
    // because there is nothing to say: `resolution`'s members are `const`, so the payload a caller
    // holds is the payload that was issued. See the note on the struct for why three rounds of
    // detectors (epoch, private token, 64-bit fold) were replaced by making modification impossible.
    //
    // The fold is gone from this check deliberately, not weakened. It could not do the job: 64-bit
    // folds collide; a caller who can read the source recomputes one; and it never covered `epoch`,
    // so a forger who also rewrote the epoch passed both the epoch comparison and the fold. A
    // detector that cannot authenticate, sitting next to an immutability guarantee that needs no
    // detector, is a false claim kept alive by habit.
    if (answer.refusal != resolve_refusal::ok) {
        return false;
    }
    const std::shared_ptr<const int> issuer = answer.origin.lock();
    if (!issuer || issuer != identity_) {
        return false;
    }
    return answer.epoch == epoch_;
}

// ---------------------------------------------------------------------------------------------
// resolve
// ---------------------------------------------------------------------------------------------

resolve_refusal ledger::check_batch(const attention_request * requests, std::size_t count,
                                    std::size_t * offending, std::string * why) const {
    // The cache-level preconditions, in the fork's order: a cache it cannot address at all is
    // refused before any view is examined, because every per-view question is unanswerable.
    if (n_stream_ != 1) {
        *offending = 0;
        *why = describe(resolve_refusal::multi_stream);
        return resolve_refusal::multi_stream;
    }
    if (shifted_) {
        *offending = 0;
        *why = describe(resolve_refusal::shifted_cache);
        return resolve_refusal::shifted_cache;
    }
    if (!requests || count == 0) {
        *offending = 0;
        *why = describe(resolve_refusal::empty_batch);
        return resolve_refusal::empty_batch;
    }
    if (count > kMaxSequences) {
        *offending = 0;
        *why = std::string(describe(resolve_refusal::batch_too_large)) + " (" + std::to_string(count) + " > " +
               std::to_string(kMaxSequences) + ")";
        return resolve_refusal::batch_too_large;
    }

    // Batch-wide sets. `readers` makes "one view per sequence" explicit instead of relying on the
    // caller; `pending` makes "the same token twice" explicit, which matters because two views of
    // one token would each be resolved and one of them silently discarded by the caller's map.
    std::set<std::int32_t> readers;
    std::set<cell_id>      pending;

    for (std::size_t i = 0; i < count; ++i) {
        const attention_request & view = requests[i];
        *offending = i;

        if (view.query_seq < 0 || std::size_t(view.query_seq) >= n_sequences_) {
            *why = numbered("query sequence", view.query_seq) + " is outside a cache of " +
                   std::to_string(n_sequences_) + " sequences";
            return resolve_refusal::bad_query_seq;
        }
        if (view.query_pos < 0 || view.query_virtual_pos < 0) {
            *why = numbered("query position", view.query_pos) + " / virtual " +
                   std::to_string(view.query_virtual_pos) + " is negative";
            return resolve_refusal::bad_query_pos;
        }
        if (!view.entries || view.n_entries == 0) {
            *why = "the view has no entries";
            return resolve_refusal::bad_entry_count;
        }
        // The bound is `occupied + 1`, not `occupied`, and the +1 is not slack.
        //
        // A view names the query's own cell — not in the cache yet, it is what this decode writes —
        // plus the cells already there. So `n_entries == occupied + 1` is the ORDINARY case, and a
        // bound of `occupied` refuses every valid view. That was this file's first bug, and the
        // first test caught it before anything else ran.
        //
        // The fork bounds by `get_size()` instead, which is the cache's cell CAPACITY: `occupied`
        // is how many are written, `get_size()` is how many can be. Both are real bounds and they
        // coincide at a full cache; when the caller has told us the capacity we use it, because
        // that is the fork's own rule and the tighter of the two.
        const std::size_t entry_bound = capacity_ != 0 ? capacity_ : cells_.size() + 1;
        if (view.n_entries > entry_bound) {
            *why = numbered("the view has", std::int64_t(view.n_entries)) +
                   " entries but the bound is " + std::to_string(entry_bound) +
                   (capacity_ != 0 ? " (the cache's capacity)"
                                   : " (the cache holds " + std::to_string(cells_.size()) +
                                         ", plus this view's own cell)");
            return resolve_refusal::bad_entry_count;
        }
        if (!readers.insert(view.query_seq).second) {
            *why = numbered("sequence", view.query_seq) + " already has a view in this batch";
            return resolve_refusal::duplicate_query_seq;
        }
        const cell_id own{view.query_seq, view.query_pos};
        if (!pending.insert(own).second) {
            *why = "sequence " + std::to_string(view.query_seq) + " at position " +
                   std::to_string(view.query_pos) + " is queried twice";
            return resolve_refusal::duplicate_query_pos;
        }
        // The query's own cell must NOT already be in the cache.
        //
        // This is the difference between "decode this token" and "re-decode a token that exists".
        // In the second case the own-entry lookup below would succeed and the view would resolve,
        // producing attention over a token that has already been attended to — which is a
        // different operation wearing the same shape.
        if (cells_.count(own) != 0) {
            *why = "sequence " + std::to_string(view.query_seq) + " at position " +
                   std::to_string(view.query_pos) +
                   " is already in the cache: this is a re-decode";
            return resolve_refusal::query_already_in_cache;
        }
    }
    return resolve_refusal::ok;
}

resolution ledger::resolve(const attention_request * requests, std::size_t count) const {
    // THE ANSWER IS CONSTRUCTED WHOLE, and that is what the `const` members require — and, one level
    // up, what makes the payload immutable rather than merely "not edited by well-behaved callers".
    //
    // The previous design built a default `resolution` and assigned five fields in sequence. The same
    // five assignments were available to any caller holding a copy, which is how an answer with
    // substituted indices passed validation. Now the only expression that produces a resolution is
    // this constructor, and it takes every field at once.
    //
    // `origin` is stamped on the way out for a refused answer too: a caller inspecting a refusal can
    // check which ledger produced it, and a refusal claiming no origin would be indistinguishable
    // from a default-constructed one.
    const std::weak_ptr<const int> issuer = ensure_identity();

    // THE DRAFT, and the name is the point: these fields are mutable while the answer is being
    // assembled and unreachable from outside while that happens. `resolution` itself has `const`
    // members, so a caller cannot repeat this assembly on a copy — which is the whole reason the
    // mutation that used to pass validation no longer compiles.
    struct draft_resolution {
        resolve_refusal              refusal = resolve_refusal::ok;
        std::size_t                  offending_view = 0;
        std::string                  why;
        std::vector<resolved_view>   views;
    } draft;

    const auto finish = [&]() {
        return resolution(draft.refusal, epoch_, draft.offending_view, std::move(draft.why),
                          std::move(draft.views), issuer);
    };

    std::size_t offending = 0;
    std::string why;
    const resolve_refusal batch = check_batch(requests, count, &offending, &why);
    if (batch != resolve_refusal::ok) {
        return resolution(batch, epoch_, offending, std::move(why), {}, issuer);
    }

    // Per-view resolution. Nothing is appended to `answer.views` until the whole batch has passed,
    // so a refusal cannot leave a partial set behind: the caller cannot mistake half a batch for a
    // batch.
    std::vector<resolved_view> resolved;
    resolved.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const attention_request & view = requests[i];
        resolved_view out;
        out.query_seq = view.query_seq;
        out.query_pos = view.query_pos;
        out.query_virtual_pos = view.query_virtual_pos;

        // SPANS GROUP ENTRIES BY EFFECTIVE POSITION, and the grouping has to be real because the
        // fork's resolved view is a vector of spans rather than one.
        //
        // `effective = query_virtual_pos + key_pos - virtual_pos` is per ENTRY, not per view. Two
        // entries in one view can demand different rotations — a key from another sequence sits at
        // a different point in this sequence's timeline than its own position suggests — and a span
        // is the unit the emission rotates once and normalises alongside the others. Assuming one
        // span per view would silently apply the first entry's rotation to every key, which
        // produces a wrong answer with no shape error anywhere.
        //
        // Consecutive entries with equal effective positions form one span. A span with no
        // resolvable indices is dropped: that is the query's own cell, which is not in the cache
        // yet and whose key arrives as the graph's native K rather than through a slot.
        std::vector<resolved_span> spans;
        resolved_span              span_acc;
        bool                       have_current = false;

        const auto flush_span = [&]() {
            if (!have_current) return;
            if (!span_acc.indices.empty()) {
                spans.push_back(std::move(span_acc));
            }
            span_acc = resolved_span{};
            have_current = false;
        };
        const auto ensure_span = [&](std::int32_t effective) {
            if (!have_current || span_acc.effective_pos != effective) {
                flush_span();
                span_acc.effective_pos = effective;
                have_current = true;
            }
        };

        std::set<std::int32_t> physical;   // per view: no slot may be read twice
        bool                   own = false;
        std::int32_t           previous_virtual = -1;

        for (std::size_t j = 0; j < view.n_entries; ++j) {
            const view_entry & entry = view.entries[j];
            draft.offending_view = i;

            if (entry.id.seq < 0 || std::size_t(entry.id.seq) >= n_sequences_) {
   draft.refusal = resolve_refusal::bad_key_seq;
                draft.why = numbered("entry sequence", entry.id.seq) + " is outside a cache of " +
                             std::to_string(n_sequences_) + " sequences";
                return finish();
            }
            if (entry.id.pos < 0) {
   draft.refusal = resolve_refusal::bad_key_pos;
                draft.why = numbered("entry position", entry.id.pos) + " is negative";
                return finish();
            }
            // A key from the query's own sequence may not sit in the future. It has not been
            // written yet, so it could only resolve to whatever previously occupied that position
            // — which is the stale-address failure with a plausible cause.
            if (entry.id.seq == view.query_seq && entry.id.pos > view.query_pos) {
    draft.refusal = resolve_refusal::future_key;
                draft.why = "entry " + std::to_string(entry.id.pos) +
                             " is after the query at " + std::to_string(view.query_pos) +
                             " in the same sequence";
                return finish();
            }
            // Strictly increasing virtual positions. The fork checks this, and the reason is worth
            // stating: the entries are positional across the span, so an out-of-order list is not a
            // permutation of the same attention — it is a different rotation per key.
            if (entry.virtual_pos <= previous_virtual) {
    draft.refusal = resolve_refusal::unordered_virtual_pos;
                draft.why = numbered("virtual position", entry.virtual_pos) +
                             " does not exceed the previous entry's " +
                             std::to_string(previous_virtual);
                return finish();
            }
            if (entry.virtual_pos > view.query_virtual_pos) {
    draft.refusal = resolve_refusal::entry_beyond_query;
                draft.why = numbered("virtual position", entry.virtual_pos) +
                             " is past the query's " +
                             std::to_string(view.query_virtual_pos);
                return finish();
            }
            previous_virtual = entry.virtual_pos;

            // int64 arithmetic, then one range check: the fork computes in int64 and compares, and
            // doing it in int32 would make the comparison itself undefined for extreme inputs.
            const std::int64_t effective = std::int64_t(view.query_virtual_pos) + entry.id.pos -
                                           std::int64_t(entry.virtual_pos);
            if (effective < 0 || effective > std::numeric_limits<std::int32_t>::max()) {
    draft.refusal = resolve_refusal::effective_pos_out_of_range;
                draft.why = "effective position " + std::to_string(effective) +
                             " cannot be represented";
                return finish();
            }

            const bool current = entry.id.seq == view.query_seq && entry.id.pos == view.query_pos;
            if (current) {
                if (own) {
        draft.refusal = resolve_refusal::duplicate_own_entry;
                    draft.why = "more than one entry is the query's own cell";
                    return finish();
                }
                if (entry.virtual_pos != view.query_virtual_pos) {
        draft.refusal = resolve_refusal::own_entry_misplaced;
                    draft.why = numbered("the own entry's virtual position", entry.virtual_pos) +
                                 " differs from the query's " +
                                 std::to_string(view.query_virtual_pos);
                    return finish();
                }
                own = true;
                // The own cell is NOT in the cache (check_batch ensured that), so it has no slot:
                // the reservation is what `put` does after the write, and its key reaches the
                // attention as the graph's native K, not through a resolved index. It still
                // establishes an effective position, so entries sharing it stay in one span.
                ensure_span(std::int32_t(effective));
                continue;
            }

            const auto found = cells_.find(entry.id);
            if (found == cells_.end()) {
    draft.refusal = resolve_refusal::entry_not_cached;
                draft.why = "cell (" + std::to_string(entry.id.seq) + ", " +
                             std::to_string(entry.id.pos) + ") is not in the cache";
                return finish();
            }
            if (found->second < 0) {
    draft.refusal = resolve_refusal::negative_physical;
                draft.why = "the cache reports slot " + std::to_string(found->second) +
                             " for a cell it holds";
                return finish();
            }
            if (!physical.insert(found->second).second) {
    draft.refusal = resolve_refusal::duplicate_physical;
                draft.why = "slot " + std::to_string(found->second) +
                             " is named by two entries of one view";
                return finish();
            }
            ensure_span(std::int32_t(effective));
            span_acc.indices.push_back(found->second);
        }

        if (!own) {
            draft.refusal = resolve_refusal::missing_own_entry;
            draft.why = "the view does not include the query's own cell";
            return finish();
        }

        flush_span();
        out.spans = std::move(spans);
        resolved.push_back(std::move(out));
    }

    draft.views = std::move(resolved);
    return finish();
}

} // namespace tq::reader
