#include "tq/rerot/whiteboard.h"

#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tq::rerot {

// The board's whole state, behind one mutex. Kept in this translation unit and reached only
// through shared ownership so that a step cannot dangle when its station dies: the board's
// lifetime is the last handle to it, not the station's.
struct board {
    board(std::size_t stages, const std::vector<std::uint8_t> & bound_stages)
        : committed(stages), bound(bound_stages) {
        for (std::uint8_t b : bound) {
            if (b) ++n_bound_stages;
        }
    }

    mutable std::mutex      mutex;
    std::vector<token_view> committed;
    // Which stages this episode binds. Immutable after construction: a step validates against
    // the set it opened with, so the set cannot change underneath one.
    std::vector<std::uint8_t> bound;
    std::size_t             n_bound_stages = 0;
    std::uint64_t           epoch    = 0;
    std::uint64_t           commits  = 0;
    std::uint64_t           discards = 0;
    std::uint64_t           refusals = 0;
};

const char * describe(step_status status) {
    switch (status) {
        case step_status::ok:                return "ok";
        case step_status::stale_epoch:       return "the board committed after this step opened; recompute the token";
        case step_status::stage_out_of_range: return "the stage id is not on this board";
        case step_status::duplicate_stage:   return "the same stage was submitted twice in one step";
        case step_status::stage_not_bound:   return "the stage is on this board but is not bound in this "
                                                    "episode; it has no sequence and cannot publish";
        case step_status::refused_submission: return "an earlier submission into this step was refused; "
                                                      "the step cannot publish, because the update it was "
                                                      "built from is incomplete";
        case step_status::missing_stage:     return "the step does not carry one token for every bound "
                                                      "stage; a body step advances all of them or none";
        case step_status::sealed_epoch:      return "the board's epoch counter is exhausted; it refuses "
                                                      "further commits rather than wrap into a value a "
                                                      "live step could mistake for its own";
        case step_status::nothing_to_publish: return "the step has no submissions";
        case step_status::already_closed:    return "the step already committed or was discarded";
    }
    return "unknown status";
}

// ---------------------------------------------------------------------------
// step
// ---------------------------------------------------------------------------

step::step(std::shared_ptr<board> owner, std::uint64_t epoch)
    : board_(std::move(owner)), epoch_(epoch), closed_(false) {}

step::step(step && other) noexcept
    : board_(std::move(other.board_)), epoch_(other.epoch_),
      pending_(std::move(other.pending_)), closed_(other.closed_), tainted_(other.tainted_) {
    // The source must not discard what it no longer owns: after a move its board is empty,
    // and a destructor that discarded anyway would count an event that never happened.
    other.closed_ = true;
    other.epoch_ = 0;
    other.pending_.clear();
    // The taint moves with the step. `station::open()` returns by value, so the usual step is
    // a moved one: a taint dropped here would hand a caller a step that had been refused a
    // submission and could now publish the rest of it.
    other.tainted_ = false;
}

step & step::operator=(step && other) noexcept {
    if (this != &other) {
        // An open step being overwritten is a step abandoned, and it is counted as one.
        discard();
        board_ = std::move(other.board_);
        epoch_ = other.epoch_;
        pending_ = std::move(other.pending_);
        closed_ = other.closed_;
        tainted_ = other.tainted_;
        other.closed_ = true;
        other.epoch_ = 0;
        other.tainted_ = false;
        other.board_.reset();
        other.pending_.clear();
    }
    return *this;
}

step::~step() {
    // Covers the early return, the throw, and the path a caller forgot. Destroying an open
    // step cannot publish: AXIOM 3 has no exception for "the caller went away".
    discard();
}

namespace {

} // namespace

epoch_verdict check_epoch(std::uint64_t board_epoch, std::uint64_t opened_at) {
    // Exhaustion first: at the maximum there is no next value, so the board stops accepting
    // commits instead of wrapping to one a live step might remember. Refusing is the only
    // answer that keeps "epoch differs from the one I opened at" a sufficient stale test --
    // a wrapped counter turns the guard into a comparison of two unrelated runs.
    if (board_epoch == std::numeric_limits<std::uint64_t>::max()) {
        return epoch_verdict::sealed;
    }
    return board_epoch == opened_at ? epoch_verdict::commit : epoch_verdict::stale;
}

bool step::missing_bound_stage() const {
    if (!board_) return false;
    // Over the BOUND set only, and without allocating: the loop walks the board's mask and
    // asks whether each bound stage is present.
    //
    // This is the correction. The contract binds a subset: LEMMA 2 covers "each bound stage",
    // and LEMMA 3 leaves the logical-stage-to-sequence mapping to the caller, so a board can
    // declare four stages while the episode binds two. Requiring [0, n_stages) made that
    // correct configuration uncommittable, and reported it as the caller's incompleteness.
    const std::size_t n = board_->committed.size();
    for (std::size_t stage = 0; stage < n; ++stage) {
        if (!board_->bound[stage]) continue;
        bool present = false;
        for (const auto & entry : pending_) {
            if (entry.first == stage) {
                present = true;
                break;
            }
        }
        if (!present) return true;
    }
    return false;
}

step_status step::submit(std::size_t stage, std::int32_t token) {
    // A closed step is a caller bug, not a board event, so it is not counted as a refusal.
    if (closed_ || !board_) return step_status::already_closed;

    std::lock_guard<std::mutex> guard(board_->mutex);
    if (tainted_) {
        // Said now, not at the commit: a caller that keeps submitting into a step that can
        // never publish is collecting tokens for a step it will have to throw away, and
        // telling it at submit is the difference between a bug found in the loop that caused
        // it and a surprise three calls later.
        board_->refusals += 1;
        return step_status::refused_submission;
    }
    if (stage >= board_->committed.size()) {
        board_->refusals += 1;
        taint();
        return step_status::stage_out_of_range;
    }
    // An unbound stage has no sequence in this episode and can never hold a token here. The
    // refusal taints: accepting the token and dropping it at commit would be the same silent
    // partial publish, with the drop moved one call later.
    if (!board_->bound[stage]) {
        board_->refusals += 1;
        taint();
        return step_status::stage_not_bound;
    }
    for (const auto & entry : pending_) {
        if (entry.first == stage) {
            // Last-wins would silence the ambiguity: the caller has two tokens for one stage
            // and does not know which it meant, which is a bug to surface, not to resolve.
            board_->refusals += 1;
            taint();
            return step_status::duplicate_stage;
        }
    }
    try {
        pending_.emplace_back(stage, token);
    } catch (...) {
        // A token that could not be recorded is a hole in the set this step is going to
        // publish, and a hole is what AXIOM 3 forbids. Taint FIRST -- taint() is noexcept and
        // allocates nothing, so it cannot fail on the failing path -- and only then let the
        // failure out. A caller that swallows the bad_alloc and keeps computing now works
        // with a step that can never publish, instead of one that will publish a prefix.
        taint();
        throw;
    }
    return step_status::ok;
}

step_status step::commit() {
    if (closed_ || !board_) return step_status::already_closed;

    std::lock_guard<std::mutex> guard(board_->mutex);

    // A step that was refused a submission cannot publish the rest of itself.
    //
    // This is AXIOM 3 one level down, and it was the real defect in the first version: the
    // calls that follow a refused submit (compute the remaining stages, check every output,
    // then commit) were written by a caller that believed all of its submissions landed. If
    // commit publishes the ones that did, the board receives exactly the thing AXIOM 3
    // forbids -- part of an update -- and no counter shows it, because each individual write
    // looked legitimate.
    //
    // Reported before the stale check and not after: a tainted step can never commit, so
    // telling its owner "recompute" would send it round a loop whose exit is unchanged.
    if (tainted_) {
        board_->refusals += 1;
        return step_status::refused_submission;
    }
    // Emptiness is reported AFTER the taint, and the order matters: a tainted step usually
    // holds nothing at all (its only submission was refused, so nothing was recorded), and
    // answering "you submitted nothing" would hide the cause the caller needs -- a refusal
    // that happened, not a step that was never filled.
    if (pending_.empty()) {
        board_->refusals += 1;
        return step_status::nothing_to_publish;
    }
    // LEMMA 2, enforced rather than assumed: one body step advances EACH bound stage by one
    // token, so the publish set is the whole bound set. A commit carrying a prefix would be a
    // partial update -- AXIOM 3's prohibition one level below the forward pass -- and the
    // stages that did land would look legitimate in every counter, because each individual
    // write is legal. The earlier version allowed it, and its test asserted the allowance;
    // that assertion was the defect, not the coverage.
    //
    // Reported before the epoch check: an incomplete set is a caller bug that recomputing
    // cannot fix, while a stale epoch is only a retry.
    if (missing_bound_stage()) {
        board_->refusals += 1;
        return step_status::missing_stage;
    }
    switch (check_epoch(board_->epoch, epoch_)) {
        case epoch_verdict::sealed:
            // The counter cannot advance without wrapping into a value a live step may
            // remember. Reported, not wrapped: `stale_epoch` would invite a retry that can
            // never succeed, and wrapping would silently convert the ABA guard into noise.
            board_->refusals += 1;
            return step_status::sealed_epoch;
        case epoch_verdict::stale:
            // Something committed while this step was computing. Its submissions were
            // derived from peers' older tokens, so publishing them would make the whiteboard
            // a set of independent completions that happen to share a file (AXIOM 1).
            board_->refusals += 1;
            return step_status::stale_epoch;
        case epoch_verdict::commit:
            break;
    }

    // Validate every submission before writing any. A refusal must leave the board exactly
    // as it was; a loop that wrote as it walked would publish a prefix on the failing path,
    // which is the same defect as AXIOM 3's partial update in a smaller place.
    for (const auto & entry : pending_) {
        if (entry.first >= board_->committed.size()) {
            board_->refusals += 1;
            return step_status::stage_out_of_range;
        }
    }

    for (const auto & entry : pending_) {
        board_->committed[entry.first] = token_view{true, entry.second};
    }
    board_->epoch += 1;
    board_->commits += 1;
    closed_ = true;
    pending_.clear();
    return step_status::ok;
}

void step::discard() {
    if (closed_ || !board_) {
        closed_ = true;
        return;
    }
    pending_.clear();
    closed_ = true;
    std::lock_guard<std::mutex> guard(board_->mutex);
    board_->discards += 1;
}

token_view step::peek(std::size_t stage) const {
    if (!board_) return {};
    std::lock_guard<std::mutex> guard(board_->mutex);
    if (stage >= board_->committed.size()) return {};
    return board_->committed[stage];
}

std::vector<token_view> step::peers_of(std::size_t stage) const {
    if (!board_) return {};
    std::lock_guard<std::mutex> guard(board_->mutex);
    // An unbound stage has no peers in this episode, and the answer is empty -- not "every
    // bound stage, because self is not among them".
    //
    // The previous code returned exactly that: for an unbound `stage`, `i != stage` was true
    // for every bound stage, so the caller received the whole bound set. Those entries are
    // usually invalid ("no token yet"), which reads as "my peers have not committed" -- a
    // meaning the caller would act on, when the truth is that it is not an insertion point in
    // this episode and has no token to compute. Empty says the read set is empty, and the
    // caller's actual mistake is reported at submit(), which refuses an unbound stage by name.
    std::vector<token_view> peers;
    if (stage >= board_->committed.size() || !board_->bound[stage]) return peers;
    // Exact, not "one less than every stage": with a partial binding the old reservation
    // over-allocated by the number of unbound stages on every call.
    peers.reserve(board_->n_bound_stages - 1);
    for (std::size_t i = 0; i < board_->committed.size(); ++i) {
        if (board_->bound[i] && i != stage) peers.push_back(board_->committed[i]);
    }
    return peers;
}

std::vector<token_view> step::snapshot() const {
    if (!board_) return {};
    std::lock_guard<std::mutex> guard(board_->mutex);
    return board_->committed;
}

// ---------------------------------------------------------------------------
// station
// ---------------------------------------------------------------------------

station::station(std::size_t n_stages)
    // Constructed directly, NOT by delegating with an empty list and then replacing the board.
    //
    // The delegating version built a board with an empty mask, then built a second board with
    // the identity mask and assigned it over the first. Every all-bound station therefore paid
    // for two boards and discarded one -- and the discarded one meant something else: an empty
    // mask is "no stage is bound", a legal and very different configuration. For one statement
    // the two constructions disagreed about what they were building. One allocation, one
    // meaning.
    : board_(std::make_shared<board>(n_stages, std::vector<std::uint8_t>(n_stages, 1))) {}

station::station(std::size_t n_stages, std::initializer_list<std::size_t> bound)
    : board_(nullptr) {
    std::vector<std::uint8_t> mask(n_stages, 0);
    for (std::size_t stage : bound) {
        if (stage >= n_stages) {
            // A binding outside the board is a caller bug with no safe reading: dropping it
            // would make the episode's completeness rule quietly weaker than the caller's.
            throw std::out_of_range("station: bound stage " + std::to_string(stage) +
                                    " is outside a board of " + std::to_string(n_stages));
        }
        if (mask[stage]) {
            throw std::invalid_argument("station: stage " + std::to_string(stage) +
                                        " is bound twice; the count is what completeness is "
                                        "judged against");
        }
        mask[stage] = 1;
    }
    board_ = std::make_shared<board>(n_stages, mask);
}

step station::open() const {
    // The epoch is read under the lock so that a step is opened at a defined point: reading
    // it unlocked could observe a commit's epoch but not its tokens, opening a step whose
    // very first read is already one step behind the number it claims to validate against.
    std::lock_guard<std::mutex> guard(board_->mutex);
    return step(board_, board_->epoch);
}

std::size_t station::n_stages() const { return board_->committed.size(); }

std::size_t station::n_bound() const { return board_->n_bound_stages; }

bool station::is_bound(std::size_t stage) const {
    return stage < board_->bound.size() && board_->bound[stage] != 0;
}

std::uint64_t station::epoch() const {
    std::lock_guard<std::mutex> guard(board_->mutex);
    return board_->epoch;
}

token_view station::peek(std::size_t stage) const {
    std::lock_guard<std::mutex> guard(board_->mutex);
    if (stage >= board_->committed.size()) return {};
    return board_->committed[stage];
}

std::vector<token_view> station::peers_of(std::size_t stage) const {
    std::lock_guard<std::mutex> guard(board_->mutex);
    // An unbound stage has no peers in this episode, and the answer is empty -- not "every
    // bound stage, because self is not among them".
    //
    // The previous code returned exactly that: for an unbound `stage`, `i != stage` was true
    // for every bound stage, so the caller received the whole bound set. Those entries are
    // usually invalid ("no token yet"), which reads as "my peers have not committed" -- a
    // meaning the caller would act on, when the truth is that it is not an insertion point in
    // this episode and has no token to compute. Empty says the read set is empty, and the
    // caller's actual mistake is reported at submit(), which refuses an unbound stage by name.
    std::vector<token_view> peers;
    if (stage >= board_->committed.size() || !board_->bound[stage]) return peers;
    // Exact, not "one less than every stage": with a partial binding the old reservation
    // over-allocated by the number of unbound stages on every call.
    peers.reserve(board_->n_bound_stages - 1);
    for (std::size_t i = 0; i < board_->committed.size(); ++i) {
        if (board_->bound[i] && i != stage) peers.push_back(board_->committed[i]);
    }
    return peers;
}

std::vector<token_view> station::snapshot() const {
    std::lock_guard<std::mutex> guard(board_->mutex);
    return board_->committed;
}

std::uint64_t station::commits() const {
    std::lock_guard<std::mutex> guard(board_->mutex);
    return board_->commits;
}

std::uint64_t station::discards() const {
    std::lock_guard<std::mutex> guard(board_->mutex);
    return board_->discards;
}

std::uint64_t station::refusals() const {
    std::lock_guard<std::mutex> guard(board_->mutex);
    return board_->refusals;
}

} // namespace tq::rerot
