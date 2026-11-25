// tq/rerot/whiteboard.h — the shared whiteboard as a station, not a struct full of cursors.
//
// The contract this file implements is LEMMA 2 and LEMMA 3 of modules/rerot-compose/AGENTS.md:
//
//   AXIOM 1  each insertion point reads its peers' last committed token on its next step
//   AXIOM 3  a failed forward must not publish part of an update
//   LEMMA 2  one body step advances each bound stage by one token; check every submission,
//            then commit; partial execution is fail-stop
//
// The shape that falls out of those three sentences is a transaction, and that is what this
// station is. A step captures the epoch it opened at, reads the board through that epoch,
// collects one token per stage, and publishes all of them or none. Publication is refused if
// the board moved while the step was being computed -- because a token computed against
// peers' older tokens is not the token AXIOM 1 asks for, and publishing it anyway is how a
// shared whiteboard quietly becomes several independent completions.
//
// Two things are deliberately not here:
//
//   * No llama, no ggml, no sequence handles. This is the ledger, not the engine. A stage id
//     is an index; what a stage is bound to is the caller's mapping (LEMMA 3).
//   * No retry loop. Retrying is a policy (how many times, with what backoff, and whether to
//     recompute at all), and the station that owned the policy would be the station that
//     hides the fact that a step was recomputed. The caller sees `stale_epoch` and decides.
//
// Lifetime: a `step` holds the board it belongs to, so a step that outlives its station is
// still safe to inspect and to close -- there is no dangling pointer to discover under load.
// The station is a handle, not an object with identity: copies name the same board.
//
// Thread safety: every read and the commit take one mutex. Reads through a step and reads
// outside one differ only in what the caller may conclude, not in what they see:
//
//   reads through a step   are validated by that step's commit (epoch check). This is the
//                          combination AXIOM 1 needs, and it is the combination to use.
//   reads outside a step   are a hint about the board's current contents. Using them to
//                          build a token without opening a step gives up the validation.
//
// The difference is structural rather than documented only: `step::peek` exists so the
// consistent path is the one that reads naturally.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace tq::rerot {

// Why a submission or a commit did not take effect.
//
// Every refusal is named, because the alternative -- a `bool` -- is what lets a caller
// conflate "the board moved underneath me" (recompute) with "this stage does not exist"
// (a bug) and retry the one that cannot succeed.
enum class step_status {
    ok,
    stale_epoch,          // the board committed after this step opened; recompute, do not retry blindly
    stage_out_of_range,   // the stage id is not in [0, n_stages)
    duplicate_stage,      // the same stage twice in one step: two tokens for one stage is ambiguous
    stage_not_bound,      // the stage exists on this board but is not bound in this run
    refused_submission,   // an earlier submit into this step was refused: the step is tainted
    missing_stage,        // the step does not carry one token for every bound stage
    nothing_to_publish,   // a commit with no submissions would bump the epoch for no one
    sealed_epoch,         // the board's epoch counter is exhausted; wrapping it would let a
                          // stale step match a fresh epoch, so this board accepts no more commits
    already_closed,       // this step already committed or discarded
};

const char * describe(step_status status);

// The commit's epoch rule, as a pure function.
//
// It is a free function because its boundary cannot be driven through the station: reaching
// it would take 2^64 commits, and a claim that "wraparound needs 584 years, so it cannot
// happen" is not a proof of safety -- it is a statement about how long the program runs. The
// failure it hides is real: `epoch` is the whole of the ABA guard, so an epoch that returns to
// a value a live step remembers makes that step's stale tokens look current.
//
//   board_epoch == max            -> sealed   (advancing would wrap; refuse, name it, stop)
//   board_epoch == opened_at      -> commit
//   otherwise                     -> stale
//
// `sealed` is checked first and does not depend on `opened_at`: at the maximum there is no
// safe next value, whether or not this step opened there.
enum class epoch_verdict { commit, stale, sealed };

epoch_verdict check_epoch(std::uint64_t board_epoch, std::uint64_t opened_at);

// A stage's last committed token, or the fact that it has none.
//
// Not an int32 with a sentinel: "no token yet" and "token 0" are different answers, and a
// peer that cannot tell them apart starts its completion with the wrong token. AXIOM 1 is
// about reading the last COMMITTED token, which does not exist before the first commit.
struct token_view {
    bool         valid = false;
    std::int32_t token = 0;
};

struct board;

// One body step: the read set, the submissions, and the all-or-nothing publish.
//
// Move-only, and owned by one thread at a time: a step is the caller's working state, so
// sharing one between threads would need a lock around `pending_` and would gain nothing --
// concurrent steps are what the board is for. What is thread-safe is the board, not the step.
//
// Move-only. Destruction with the step still open discards it and counts the discard: a
// caller that returns early, throws, or forgets cannot have published half an update, and
// the counter says it happened rather than leaving the run to look clean.
class step {
public:
    // A default step publishes nothing and owns nothing. It exists so a member can be
    // declared before the station that will produce it, not as a way to skip opening one.
    step() = default;
    step(step && other) noexcept;
    step & operator=(step && other) noexcept;
    step(const step &) = delete;
    step & operator=(const step &) = delete;
    ~step();

    // Collects one token for `stage`. Submission order does not matter; the commit publishes
    // the set. A stage may appear once: twice is refused rather than last-wins, because
    // which token the stage meant is exactly the information the caller lost.
    //
    // A refusal TAINTS the step, and every later submit into it returns refused_submission
    // without accepting anything. The alternative -- keep collecting, commit what landed --
    // publishes part of an update that the caller believed was complete, which is AXIOM 3's
    // prohibition one level below the forward pass. Fail-stop is the only reading of
    // LEMMA 2's "checks all outputs" that survives an output that could not be submitted.
    step_status submit(std::size_t stage, std::int32_t token);

    // Publishes every submission, or none. On any refusal the board is untouched, the epoch
    // does not move, and the step stays open for the caller to discard or to submit into a
    // fresh step.
    step_status commit();

    // Abandons the submissions. No state changes; counted so a caller can see that a step
    // was computed and thrown away.
    void discard();

    bool          open() const { return !closed_; }
    std::uint64_t epoch() const { return epoch_; }   // the epoch this step reads through

    // Reads on behalf of this step. The values are the board's contents now; what makes them
    // usable as AXIOM 1 inputs is that commit() refuses if the board moved since epoch_.
    token_view              peek(std::size_t stage) const;
    std::vector<token_view> peers_of(std::size_t stage) const;

    // Every stage's token, read in one lock acquisition.
    //
    // `peek` twice is two points in time, and a caller that compares two stages needs one:
    // this test caught exactly that, where a reader comparing stage 0 with stage 1 through
    // two `peek` calls reported a torn commit that the board never published. The publish was
    // atomic; the observation was not. This is the call that makes an observation atomic.
    std::vector<token_view> snapshot() const;

private:
    friend class station;
    step(std::shared_ptr<board> board, std::uint64_t epoch);

    // The taint is a bool, and it is set by a function that cannot fail.
    //
    // The previous version recorded it as a sentinel entry inside `pending_`, on the argument
    // that a move would then carry it for free. That argument was wrong twice:
    //
    //   * appending a sentinel ALLOCATES. A refusal that runs out of memory threw out of
    //     submit() with the refusal counted and the taint unrecorded, so a caller that caught
    //     the failure and kept going could commit the rest -- precisely the partial publish
    //     the taint exists to forbid. A safety flag must not depend on memory.
    //   * the move constructor and move assignment already carry every field by hand; a bool
    //     is one more line there, and it is a line a test can check. Encoding state in a
    //     container to avoid that line bought a hidden allocation instead.
    void taint() noexcept { tainted_ = true; }

    // True when some BOUND stage is absent from pending_. Reads the board, so the caller must
    // hold its mutex (commit does). Allocates nothing: this runs on the refusal path, which
    // must not itself be able to fail.
    //
    // Named for what it checks. It was `missing_stage` over [0, n_stages), which demanded a
    // token for stages the episode never bound -- a refusal of correct work, with a message
    // that put the fault on the caller's completeness rather than on this board's definition
    // of it.
    bool missing_bound_stage() const;

    std::shared_ptr<board> board_;
    std::uint64_t          epoch_ = 0;
    std::vector<std::pair<std::size_t, std::int32_t>> pending_;
    bool                   closed_ = true;
    bool                   tainted_ = false;
};

// The board every stage shares. Fixed stage count: a stage that appears after the run began
// is a different run, and growing the vector underneath a reader would be the one operation
// here that cannot be made atomic without a second lock.
//
// STAGES AND BINDINGS ARE NOT THE SAME SET, and the contract is explicit about which one a
// commit covers: LEMMA 2 says one body step advances each **bound** stage, and AXIOM 1 says
// each **active** insertion point reads its peers. A board may hold more logical stages than
// the episode binds -- LEMMA 3 puts the logical-stage-to-sequence mapping in the caller -- so
// requiring every stage id to submit was an over-correction: with four stages declared and two
// insertion points active, no step could commit at all, and the refusal named the wrong fault
// ("missing stage") for a configuration that was correct.
//
// Bindings are fixed at construction, like the stage count, and for the same reason: a step
// validates against the set it opened with, and a set that could change underneath it would
// make "did I cover everything?" a question about when you asked.
class station {
public:
    // Every stage bound. The usual case, and the one an caller that has no notion of inactive
    // insertion points means to say.
    explicit station(std::size_t n_stages);

    // Only `bound` participate. Each entry is a stage id in [0, n_stages); a duplicate is a
    // caller bug and aborts rather than being silently deduplicated, because the count is
    // what completeness is judged against. An empty set is legal and means nobody publishes
    // -- a board that exists but carries no lane this episode.
    station(std::size_t n_stages, std::initializer_list<std::size_t> bound);

    std::size_t   n_stages() const;      // logical stages on this board
    std::size_t   n_bound()  const;      // stages a step must carry: what completeness means
    bool          is_bound(std::size_t stage) const;

    // Opens a step at the current epoch. Cheap: one lock, no allocation beyond the step.
    step open() const;

    std::uint64_t epoch() const;     // number of committed steps since construction

    token_view              peek(std::size_t stage) const;
    // The step's active peers: bound stages other than `stage`, in stage order. AXIOM 1 is
    // about insertion points that are reading, so an unbound stage is not a peer -- it has no
    // sequence and can never hold a token, and listing it as an absent entry would invite a
    // caller to treat "not bound" as "not ready".
    //
    // Asking for the peers of an UNBOUND stage returns an empty vector, which is the read set
    // of a stage that is not an insertion point this episode. (It used to return every bound
    // stage, because `self` was not among them; those entries are normally "no token yet", so
    // the caller was told its peers had not committed rather than that it was not bound.)
    std::vector<token_view> peers_of(std::size_t stage) const;

    // Every stage's token, under one lock. Use this rather than two `peek` calls whenever the
    // answer depends on more than one stage: two calls are two points in time, and the board
    // between them may have committed. This is the observation-side counterpart of the
    // commit's atomicity, and it exists because a test that compared two stages through two
    // `peek` calls reported a torn commit the board never published.
    std::vector<token_view> snapshot() const;

    // Diagnostics. A run that reports its output without these cannot say whether any step
    // was refused, and a refused step is the difference between the whiteboard working and
    // the stages having completed independently.
    std::uint64_t commits() const;
    std::uint64_t discards() const;
    std::uint64_t refusals() const;

private:
    std::shared_ptr<board> board_;
};

} // namespace tq::rerot
