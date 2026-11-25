// herd/prefab_split.cpp — envelope padding at the point where ubatches are BORN.
//
// The earlier attempts padded a ubatch after it existed, and the measurement said
// why that cannot work: llama_context::decode asks the memory module for a plan
// first, and that plan is built from the ubatch by llama_kv_cache::prepare ->
// find_slot, before process_ubatch is ever called. A ubatch rewritten later always
// disagrees with the plan it arrives with (GGML_ASSERT at llama-kv-cache.cpp:1619).
//
// The birth is here:
//
//   llama_kv_cache::init_batch(balloc, n_ubatch, embd_all)      llama-kv-cache.cpp:735
//     while (true) { ubatch = balloc.split_simple(n_ubatch); … }        :747
//     auto sinfos = prepare(ubatches);                                  :761
//
// `prepare` computes one cell index per token of the ubatch it is given. So a
// ubatch padded HERE is planned at its padded width: the plan and the ubatch agree
// by construction, which is the property every previous attempt lacked.
//
// What padding costs, and why the pad rows need an owner.
//
// find_slot reserves one KV cell per token. A pad row with n_seq_id == 0 gets a cell
// with no sequence, so llama_kv_cache::apply_ubatch (:1191) sets its position and
// adds no owner: nothing can ever free it, one leaked cell per pad row per ubatch.
// The fork avoided this with a scratch sequence and a `n_fallback_no_scratch`
// counter. Same here: the pad rows belong to a sequence the caller reserved for
// them, and the caller removes them afterwards with llama_memory_seq_rm.
//
// So the caller's contract is three parts, and all three are checked rather than
// assumed:
//
//   1. n_seq_max is at least (sequences in use + 1), so the scratch id exists in
//      seq_to_stream (find_slot indexes it at :1023 without a range test);
//   2. the scratch sequence holds nothing before the decode;
//   3. it is cleared after, with llama_memory_seq_rm(mem, scratch, -1, -1).
//
// STATUS: an experiment with its own probe, linked into no product. The lane is
// inert unless a herd::prefab::scratch_scope is open on the calling thread, and it
// is only exercised by herd-prefab-split-probe, which compares padded output
// against unpadded output bit for bit on a real transformer.
#include "herd_patch.hpp"

#include "llama-batch.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include "herd_prefab_split.hpp"

// Declared at namespace scope, not only as a friend: a friend declaration inside the class
// is found by argument-dependent lookup but not by the qualified name the seam uses, and a
// seam that cannot name its own reporting function is not a composition problem worth
// solving twice.
namespace herd::prefab {
void prefab_note_binding_refusal(scratch_scope & scope);
} // namespace herd::prefab

namespace {

thread_local herd::prefab::scratch_scope * t_scope = nullptr;

// The memory module of the context whose decode is running, published by the decode
// entry hooks below.
//
// Why this exists at all: the seam that pads ubatches is
// `llama_batch_allocr::split_simple`, and `llama_batch_allocr` carries no context and no
// memory handle (llama-batch.h:167 -- `n_pos_per_embd` and `allow_mixed`, nothing else).
// So the seam cannot ask whose ubatch it is splitting. Left at that, a reservation opened
// for context A would also pad context B's ubatches into A's scratch sequence, and B's
// real KV history in ITS cache would be destroyed at A's release -- the failure the
// constructor refuses within one context, reached by a route it cannot see.
//
// The decode entry point can answer the question, so it is hooked: `llama_context::decode`
// (both overloads -- `llama_process` and the deprecated `llama_decode` dispatch through
// them, verified in llama-context.cpp:5623-5628 and :5546-5549) publishes
// `llama_get_memory(this)` for the duration of the call and restores the previous value
// afterwards.
//
// A value of nullptr means "no decode is in progress". The seam then declines, which is
// correct: padding only ever applies to a decode batch, and an encode or a direct
// allocr use must not be padded on the strength of a reservation that happens to be open.
thread_local llama_memory_t t_decoding_memory = nullptr;

// Saves and restores rather than just assigning: an eval callback can decode on the same
// thread, and that inner decode must not leave the outer one looking unbound when it
// returns.
struct decoding_scope {
    llama_memory_t previous;
    explicit decoding_scope(llama_context * ctx)
        : previous(t_decoding_memory), value(llama_get_memory(ctx)) {
        t_decoding_memory = value;
    }
    ~decoding_scope() { t_decoding_memory = previous; }
    decoding_scope(const decoding_scope &) = delete;
    decoding_scope & operator=(const decoding_scope &) = delete;

private:
    llama_memory_t value;
};

// llama_ubatch llama_batch_allocr::split_simple(uint32_t n_ubatch)
//
// A non-trivial return type (it holds a shared_ptr) means the Itanium ABI passes a
// hidden destination pointer as the first argument. Our replacement therefore
// writes *out and never returns a value, which is what keeps it ABI-compatible
// with the function Dobby jumped over.
using split_simple_fn = void (*)(llama_ubatch * out, llama_batch_allocr * self, uint32_t n_ubatch);

split_simple_fn g_orig = nullptr;

std::atomic<long> g_calls{0};

bool trace_enabled() {
    static const bool enabled = std::getenv("HERD_PREFAB_TRACE") != nullptr;
    return enabled;
}

// A padded copy of `u`, whose pad rows belong to `scratch`.
//
// Every field the memory module and the graph read is filled; nothing is left to a
// default, because a default here is a wild read: find_slot indexes seq_to_stream
// by seq_id_unq and seq_idx by sequence id, and the graph reads pos and output for
// every row.
llama_ubatch pad_owned(const llama_ubatch & u, uint32_t n_env, llama_seq_id scratch) {
    auto data = std::make_shared<llama_ubatch::data_t>();

    const uint32_t real = u.n_tokens;
    data->token      .assign(u.token, u.token + real);
    data->token      .resize(n_env, u.token[real - 1]);
    data->pos        .assign(u.pos, u.pos + off_t(real) * u.n_pos);
    data->pos        .resize(std::size_t(n_env) * u.n_pos, u.pos[off_t(real - 1) * u.n_pos]);
    data->output     .assign(u.output, u.output + real);
    data->output     .resize(n_env, 0);   // no caller reads a pad row's logits
    data->seq_idx    .assign(LLAMA_MAX_SEQ, -1);
    // seq_idx maps a sequence id to its row group in [0, n_seqs_unq). The real rows
    // keep the mapping upstream gave them; the scratch id is appended.
    for (uint32_t s = 0; s < u.n_seqs_unq; ++s) {
        data->seq_idx[u.seq_id_unq[s]] = u.seq_idx[u.seq_id_unq[s]];
    }
    if (u.data) {
        data->batch_idxs.assign(u.data->batch_idxs.begin(), u.data->batch_idxs.end());
    }
    data->batch_idxs .resize(n_env, -1);

    // Sequence ownership. The real rows keep theirs; every pad row owns exactly
    // `scratch`, which is what lets the caller free the cells afterwards.
    data->n_seq_id.assign(u.n_seq_id, u.n_seq_id + real);
    data->n_seq_id.resize(n_env, 1);
    std::size_t total_ids = 0;
    for (uint32_t i = 0; i < real; ++i) {
        total_ids += std::size_t(u.n_seq_id[i]);
    }
    total_ids += n_env - real; // one id per pad row
    data->seq_id_data.reserve(total_ids);
    data->seq_id.resize(n_env, nullptr);
    for (uint32_t i = 0; i < real; ++i) {
        data->seq_id[i] = data->seq_id_data.data() + data->seq_id_data.size();
        for (int32_t s = 0; s < u.n_seq_id[i]; ++s) {
            data->seq_id_data.push_back(u.seq_id[i][s]);
        }
    }
    for (uint32_t i = real; i < n_env; ++i) {
        data->seq_id[i] = data->seq_id_data.data() + data->seq_id_data.size();
        data->seq_id_data.push_back(scratch);
    }

    const bool already = std::find(u.seq_id_unq, u.seq_id_unq + u.n_seqs_unq, scratch) !=
                         u.seq_id_unq + u.n_seqs_unq;
    data->seq_id_unq.assign(u.seq_id_unq, u.seq_id_unq + u.n_seqs_unq);
    if (!already) {
        data->seq_id_unq.push_back(scratch);
    }
    if (data->seq_idx[scratch] < 0) {
        data->seq_idx[scratch] = int32_t(data->seq_id_unq.size() - 1);
    }

    llama_ubatch padded{};
    // b_equal_seqs stays 0: the rows are no longer one uniform sequence set, and the
    // asserts that trust that flag (llama-kv-cache.cpp:1026-1027) would index by a
    // layout this ubatch does not have.
    padded.b_equal_seqs = 0;
    padded.n_tokens     = n_env;
    padded.n_seq_tokens = 1;
    padded.n_seqs       = n_env;
    padded.n_seqs_unq   = uint32_t(data->seq_id_unq.size());
    padded.n_pos        = u.n_pos;
    padded.token        = data->token.data();
    padded.pos          = data->pos.data();
    padded.n_seq_id     = data->n_seq_id.data();
    padded.seq_id       = data->seq_id.data();
    padded.seq_id_unq   = data->seq_id_unq.data();
    padded.seq_idx      = data->seq_idx.data();
    padded.output       = data->output.data();
    padded.data         = std::move(data);

    if (trace_enabled()) {
        std::fprintf(stderr,
                     "[herd] prefab-split: padded %u real row(s) to %u, scratch seq %d, "
                     "n_seqs_unq %u\n",
                     real, n_env, (int) scratch, padded.n_seqs_unq);
    }
    return padded;
}

void herd_split_simple(llama_ubatch * out, llama_batch_allocr * self, uint32_t n_ubatch) {
    ++g_calls;
    g_orig(out, self, n_ubatch);

    herd::prefab::scratch_scope * scope = t_scope;
    if (!scope) {
        return; // no reservation: ordinary splitting, byte for byte
    }
    // A reservation that has been released, or that could not be opened, is not in force.
    // `finished()` covers the first case -- without it, padding continues after release and
    // the destructor's release() short-circuits, so those cells are never returned;
    // `usable()` covers the occupied-at-open case, which never registers anyway and is
    // checked here so the rule does not depend on that registration detail.
    if (!scope->usable() || scope->finished()) {
        scope->note_declined("this reservation is not in force (released, or refused at open)");
        return;
    }
    if (out->n_tokens == 0) {
        return; // end of the batch
    }

    // Whose ubatch is this?
    //
    // The reservation names one context's memory module; the decode entry hook published
    // the memory module of the context whose decode is running. Only when the two agree
    // may this seam touch the cache: padding another context's ubatch would write pad
    // rows into a sequence of THAT cache, which the reservation never checked for
    // emptiness and will never release -- so the other context's real history would be
    // destroyed at a release it does not own.
    //
    // Declining (not aborting) is the safe answer: the other context simply runs without
    // the lane, byte for byte, and the reservation reports the refusal so a driver can
    // see that it held a reservation across two contexts.
    if (t_decoding_memory != scope->memory()) {
        herd::prefab::prefab_note_binding_refusal(*scope);
        return;
    }

    // The lane exists for the narrow case it can serve: row-addressable token rows
    // with no embedding and a single position each. Everything else is declined with
    // a reason, never padded into a shape whose meaning would differ.
    const auto n_env = scope->envelope();
    const llama_seq_id scratch = scope->sequence();
    const bool row_wise = out->n_seq_tokens == 1 && !out->b_equal_seqs && out->n_pos == 1 &&
                          out->token != nullptr && out->embd == nullptr && out->type == nullptr;
    if (!row_wise) {
        scope->note_declined("not row-addressable (packed, embedding or multi-position rows)");
        return;
    }
    if (out->n_tokens >= n_env) {
        scope->note_declined("already at or beyond the envelope width");
        return;
    }
    if (scratch < 0) {
        scope->note_declined("no scratch sequence reserved");
        return;
    }
    // The id must be one the cache can represent. The scope refuses an out-of-range
    // reservation at construction, so this is a second gate for the case where the scope
    // was built before that check existed -- or by a build that has a different
    // LLAMA_MAX_SEQ. Cheap, and the alternative is an out-of-bounds read of seq_to_stream
    // (llama-kv-cache.cpp:1023 indexes it with no bound check).
    if (scratch >= LLAMA_MAX_SEQ) {
        scope->note_declined("scratch sequence id is outside the representable range");
        return;
    }
    // A pad row takes a cell from the same pool as the real rows, and `find_slot` will
    // evict a live SWA-masked cell to get one (llama-kv-cache.cpp:1082-1096: `can_use`
    // is true for an empty cell, or for a single-sequence cell whose entry the sliding
    // window has masked out). For a real sequence that eviction is the cache doing its
    // job; for a pad row it destroys an entry nobody asked to lose, and a later decode
    // that attends to it silently sees less context.
    //
    // The seam cannot see the cache, so it cannot check this -- and the caller's promise
    // ("there is room") is exactly the kind of thing this project refuses to take on
    // faith. What it can do is say so: a lane whose pad rows are competing for cells is
    // reported through the scope's counters, and the caller compares `pad_rows()` against
    // the free cells it measured. Production integration must do that comparison; this
    // file only keeps the numbers honest.

    // Second contract, found by measurement: when the batch asked for logits on every
    // row, decode() counts outputs as `ubatch.n_tokens` rather than by the output flags
    // (llama-context.cpp:2747 `if (n_outputs_all == n_tokens_all)`), and the padded
    // width then exceeds what the caller can receive:
    //   GGML_ASSERT(n_outputs_prev + n_outputs <= n_outputs_all)   :2819
    //
    // There is no shape that satisfies both sides of that test, because n_tokens IS the
    // count upstream reads. So the lane declines exactly the ubatches where every row
    // asked for logits -- an inference it can make from the flags it was given -- and
    // pads the rest, where the flags are what get counted and a pad row's 0 adds
    // nothing.
    bool all_rows_want_logits = true;
    for (uint32_t i = 0; i < out->n_tokens; ++i) {
        if (out->output[i] == 0) {
            all_rows_want_logits = false;
            break;
        }
    }
    if (all_rows_want_logits) {
        scope->note_declined("every row asked for logits; upstream counts ubatch.n_tokens "
                             "as outputs in that case, so a padded width cannot satisfy it");
        return;
    }

    const uint32_t n_real = out->n_tokens;

    // Padding allocates, and this seam runs where upstream never throws.
    //
    // `pad_owned` copies the ubatch's arrays and widens them to `n_env`, which comes from
    // the caller's reservation and is bounded by nothing the seam can see. The allocation
    // can therefore fail -- and this code runs inside `llama_batch_allocr::split_simple`,
    // which `llama_context::decode` calls. Upstream never throws from that path: an
    // impossible configuration aborts, a batch that will not fit returns an empty ubatch.
    // A `std::bad_alloc` escaping here unwinds through `llama_process`, which no caller of
    // a C API is prepared for, and an uncaught one terminates the process.
    //
    // So a failed allocation is a DECLINED PADDING, not a failed run. The lane is an
    // optimisation; an optimisation that can kill the process is a defect. Measured
    // before this guard: a reservation with a 4e9-row envelope threw out of the first
    // decode.
    //
    // Built outside `*out` on purpose: `pad_owned` returns a new ubatch, so a throw leaves
    // the caller's ubatch untouched and the fall-through below is the ordinary, unpadded
    // split -- byte for byte what upstream produced.
    llama_ubatch padded;
    try {
        padded = pad_owned(*out, n_env, scratch);
    } catch (const std::exception & error) {
        scope->note_declined("could not build the padded ubatch (allocation failed)");
        std::fprintf(stderr,
                     "[herd] prefab-split: padding declined, envelope %u: %s\n",
                     n_env, error.what());
        return;
    } catch (...) {
        // Not reachable through the standard library, and caught anyway: an unknown
        // exception type escaping into a C caller is the same defect with no message.
        scope->note_declined("could not build the padded ubatch (unknown failure)");
        return;
    }
    *out = std::move(padded);
    scope->note_padded(n_env, n_real);
}

// _ZN18llama_batch_allocr12split_simpleEj, read from the archive with nm. Note the
// archive also carries a `.cold` clone of the same function; the hook goes on the
// hot body (T), and the clone is only reached on paths the padding lane declines.
constexpr const char * kSymbol = "_ZN18llama_batch_allocr12split_simpleEj";

} // namespace

namespace herd::prefab {

scratch_scope::scratch_scope(llama_memory_t memory, llama_seq_id scratch, uint32_t envelope)
    : memory_(memory), scratch_(scratch), envelope_(envelope) {
    // The id must be representable before anything uses it. Every consumer asserts it
    // (llama-kv-cells.h: `assert(seq_id < LLAMA_MAX_SEQ)` on every seq_add/seq_rm), but
    // find_slot reads `seq_to_stream[seq_id]` with no check at all
    // (llama-kv-cache.cpp:1023) -- so an out-of-range id is not a clean assert, it is an
    // out-of-bounds read that happens to look like a stream number. Refused here, where
    // the caller can act on it, instead of inside a decode.
    if (scratch < 0 || scratch >= LLAMA_MAX_SEQ) {
        std::fprintf(stderr,
                     "[herd] prefab-split: scratch sequence %d is outside [0, %d); refusing to "
                     "open a reservation that could not be represented\n",
                     (int) scratch, (int) LLAMA_MAX_SEQ);
        std::abort();
    }
    if (memory == nullptr) {
        // Without the memory module there is nothing to clear, and a scope that cannot
        // clean up after itself is the leak this class exists to prevent.
        std::fprintf(stderr,
                     "[herd] prefab-split: a reservation needs the context's memory module "
                     "(llama_get_memory) so its cells can be released\n");
        std::abort();
    }
    if (t_scope != nullptr) {
        std::fprintf(stderr,
                     "[herd] prefab-split: a scratch scope is already open on this thread; "
                     "nesting is refused because it makes the reserved sequence ambiguous\n");
        std::abort();
    }

    opened_ = true;

    // The precondition, checked against the cache instead of trusted: the sequence must
    // be EMPTY when the scope opens.
    //
    // This was the defect. Nothing checked it, and release() clears the whole sequence
    // (`seq_rm(…, -1, -1)`), so a caller who pointed a reservation at a sequence holding
    // real history had that history destroyed at release time -- by the cleanup path,
    // silently, some time after the reservation was opened. The test suite even asserted
    // the destruction as correct behaviour, which is how a defect becomes a contract.
    //
    // Fail closed, not "clear and carry on": clearing would destroy data this object was
    // never given, and `seq_rm` succeeding would make it look intentional. A refused
    // scope does not register as the seam's scope at all (so no ubatch is padded), and
    // release() leaves the sequence alone -- see both below.
    //
    // The query is upstream's own: seq_pos_max returns -1 for a sequence with no cells,
    // and it is the same call release() uses to verify its postcondition, so "empty at
    // open" and "empty at close" are decided by one rule.
    occupied_at_open_ = llama_memory_seq_pos_max(memory_, scratch_) != -1;
    if (occupied_at_open_) {
        std::fprintf(stderr,
                     "[herd] prefab-split: scratch sequence %d already holds cells (up to "
                     "position %d); refusing the reservation rather than clearing data this "
                     "object was not given. The padding lane stays off for this run.\n",
                     (int) scratch_, (int) llama_memory_seq_pos_max(memory_, scratch_));
        return;
    }

    t_scope = this;
}

scratch_scope::~scratch_scope() {
    // Only if this scope still owns the registration. After an explicit release() the slot
    // belongs to nobody, and clearing it unconditionally would clobber a newer scope that
    // the caller opened after that release -- the same class of bug as the leak, one level
    // up: a destructor acting on state it no longer owns.
    if (opened_ && t_scope == this) {
        t_scope = nullptr;
    }

    // The explicit-request contract, reported once per reservation instead of once per
    // ubatch.
    //
    // Opening a scope IS the request: "pad the ubatches of this context". A reservation
    // that was valid, was offered work, and padded nothing has failed to deliver what it
    // was asked for -- and the failure mode matters, because the run still produces correct
    // output (unpadded), so nothing downstream would notice. The per-ubatch declines each
    // print a line; this is the one line that says the request as a whole went unmet.
    //
    // Not an abort: the output is correct and an optimisation that kills the process is the
    // defect this file already fixed once. Not silent either: a caller that pipes stderr
    // away still has declined()/binding_refusals() to check, which the header now names as
    // an obligation.
    //
    // The condition is precise. `usable()` excludes a reservation refused at open (it never
    // asked for anything, and it already said why). `padded_ubatches() == 0` is "nothing
    // delivered". The sum over both refusal reasons is "there was work and it was turned
    // away", which distinguishes this from a scope that simply saw no decode.
    if (request_unmet()) {
        std::fprintf(stderr,
                     "[herd] prefab-split: reservation on scratch sequence %d ended having "
                     "padded NOTHING (%d shape decline(s), %d cross-context refusal(s)); this "
                     "run is not the padded configuration that was requested\n",
                     (int) scratch_, declined_, binding_refusals_);
    }

    // Unconditional, and that is the point: this runs on every path out of the caller's
    // scope -- a returned decode, a failed one, or a stack unwind -- so the pad rows'
    // cells come back regardless of how the caller left.
    release();
}

bool scratch_scope::release() {
    if (memory_ == nullptr) {
        // Unreachable: the constructor refuses a null memory module. Reported as success
        // because there is nothing to release -- not because anything was verified.
        return true;
    }
    if (released_) {
        return true;
    }
    if (occupied_at_open_) {
        // Not ours to clear. Reporting false rather than true: "released" would claim
        // this object holds no cells, but the truth is that it never owned any -- and a
        // true here is what made an earlier version look like a successful cleanup while
        // it was destroying somebody else's history.
        return false;
    }

    // Two claims, both checked, because either one alone is the kind of true statement
    // that hides a leak.
    //
    //   seq_rm() returning true means "the removal succeeded", and it also returns true
    //   for a sequence that holds nothing -- so on its own it cannot distinguish "I
    //   released the cells" from "there was nothing there". That is the same false
    //   positive that made an earlier probe pass with unowned pad rows.
    //
    //   seq_pos_max() == -1 means the sequence is empty NOW. It is the postcondition the
    //   caller actually needs, and it is read from the cache rather than inferred.
    //
    // `released_` is set only when both hold. The earlier version set it unconditionally,
    // which had two consequences worth naming: a failed clear was never retried (the
    // destructor's release() short-circuited on the flag), and a second release() would
    // report success for cells that were still occupied. A flag that records the attempt
    // instead of the outcome is worse than no flag -- it makes the leak quiet.
    const bool cleared = llama_memory_seq_rm(memory_, scratch_, -1, -1);
    const bool empty   = llama_memory_seq_pos_max(memory_, scratch_) == -1;
    released_ = cleared && empty;

    // The reservation ends here, and the seam must know it -- not because the caller
    // stopped using it, but because continuing to pad would write rows nobody will ever
    // clear. Measured before this line existed: a decode after an explicit release padded
    // again, and the destructor's release() returned immediately on `released_`, so those
    // cells were held until the context died. The counter that reports this leak is
    // pad_rows() minus settled_pad_rows(), and it grew on a run that had already released.
    finished_ = true;

    // Deregister while still holding the only reference to the slot: `t_scope` is the
    // seam's view of "a reservation is in force". Leaving it pointing at a finished scope
    // had two observable consequences -- the leak above, and a spurious "nesting is
    // refused" abort when the caller opened the next reservation on the same thread,
    // because the constructor only tested `t_scope != nullptr`.
    if (t_scope == this) {
        t_scope = nullptr;
    }

    if (!released_) {
        std::fprintf(stderr,
                     "[herd] prefab-split: scratch sequence %d is still occupied after the "
                     "release (seq_rm=%d, empty=%d); its cells stay taken and later decodes "
                     "will run short of room\n",
                     (int) scratch_, (int) cleared, (int) empty);
    }
    return released_;
}

void scratch_scope::note_padded(uint32_t n_env, uint32_t n_real) {
    ++padded_ubatches_;
    // The rows actually added, which is what a caller reporting cost needs. The earlier
    // version ignored both arguments and added 1 per ubatch, so a scope that padded eight
    // two-token ubatches reported "8 pad rows" when it had added 48 -- a number that looks
    // plausible and is wrong by a factor of the envelope width.
    pad_rows_ += (n_env > n_real) ? (n_env - n_real) : 0;
}

// The completion side, recorded by whoever ran the decode.
//
// The seam cannot fill this in: it returns from split_simple before the cache has
// accepted or refused anything. Deriving it from the attempt counters instead --
// "assume every attempt settled" -- is what made an earlier probe assert 192 rows
// against a counter reading 198 and report it as a lane defect. The gap was the one
// ubatch submitted and then refused; it belongs in the arithmetic, not in a tolerance.
void scratch_scope::note_settled(uint32_t n_env, uint32_t n_real) {
    ++settled_ubatches_;
    settled_pad_rows_ += (n_env > n_real) ? (n_env - n_real) : 0;
}

void scratch_scope::note_declined(const char * why) {
    ++declined_;
    last_decline_ = why;
}

} // namespace herd::prefab

namespace herd::prefab {

// The seam's way of saying "this reservation was held across another context's decode".
//
// A free function rather than a member because the seam is not a member of the class and
// a public mutator would invite callers to fabricate the count. `friend` grants exactly
// this one function the access, so the recorded state cannot be written from anywhere
// else -- the same reason the other counters are only advanced by the seam.
void prefab_note_binding_refusal(scratch_scope & scope) {
    ++scope.binding_refusals_;
    scope.last_decline_ = "the decode in progress belongs to a different context than this "
                          "reservation; the lane stays off for that context";
}

} // namespace herd::prefab

namespace {

// The two decode entry points, hooked to publish which context is decoding.
//
//   int32_t llama_context::decode(const llama_batch &)                  libllama.a: T _ZN13llama_context6decodeERK11llama_batch
//   int32_t llama_context::decode(const llama_batch_ext &, unsigned long)  T _ZN13llama_context6decodeERK15llama_batch_extm
//
// Both are real overloads of the same member (`llama_process` dispatches to them at
// llama-context.cpp:5623-5628, and the deprecated `llama_decode` at :5546-5549), so a
// driver that still uses the old call is covered too. Missing either one would leave one
// entry path publishing a stale memory and the seam padding the wrong cache -- which is
// the failure this file exists to close.
using decode_batch_fn = std::int32_t (*)(llama_context *, const llama_batch &);
using decode_ext_fn   = std::int32_t (*)(llama_context *, const llama_batch_ext &, std::size_t);

decode_batch_fn g_orig_decode_batch = nullptr;
decode_ext_fn   g_orig_decode_ext   = nullptr;

std::int32_t herd_decode_batch(llama_context * ctx, const llama_batch & batch) {
    const decoding_scope published(ctx);
    return g_orig_decode_batch(ctx, batch);
}

std::int32_t herd_decode_ext(llama_context * ctx, const llama_batch_ext & batch, std::size_t n) {
    const decoding_scope published(ctx);
    return g_orig_decode_ext(ctx, batch, n);
}

// Resolved from the archive, not taken by address: both are members of a class declared in
// a private header, and a pin bump that renames or reorders the overloads must fail here
// instead of hooking the wrong one.
constexpr const char * kDecodeBatchSymbol = "_ZN13llama_context6decodeERK11llama_batch";
constexpr const char * kDecodeExtSymbol   = "_ZN13llama_context6decodeERK15llama_batch_extm";

} // namespace

// The statistics ABI. Included here rather than at the top of the file because this TU is also
// compiled into probes that link no upstream-ext, and the registry must not become a link
// dependency of a program that never asks for stats. The CMake targets that build this file add
// `modules/upstream-ext/include` for the same reason: without it, the provider below would be
// code that has never been compiled.
#include <herd/upstream-ext.h>

// ---------------------------------------------------------------------------
// The statistics seam: the first time this lane has been observable from outside.
//
// Before this, `HERD_PREFAB_SPLIT=1` could be running a padded decode and nothing outside
// the process could tell. The counters existed (`padded_ubatches()`, `pad_rows()`), but no
// provider registered them, so `herd::ext::prefab_available()` was permanently false and a
// driver asking for `--prefab-stats` got a hard error -- correct, but useless.
//
// The lane counts TWO things and knows them for certain: ubatches widened, and pad rows
// written. It counts nothing else, because there is nothing else here -- no graph cache, no
// factor cache, no MTP. So the two are published and the rest are reported as NOT CAPTURED
// through `prefab_stats::captured`. Without that mask this registration would have published
// twenty-nine zeros that look exactly like measurements, and the first reader would have
// concluded "no reuse, no cache hits" from a lane that has no cache at all.
//
// `is_enabled` answers a narrower question than a reader might expect, and the narrower one is
// the true one: is a reservation open on THIS thread for THIS context's memory. The lane pads
// per decode call on the decoding thread, so that is exactly the condition under which padding
// can happen. A reservation opened on another thread reports false here, because on this
// thread it would not pad -- and "true" would make a driver believe a run was covered when it
// was not. The cost is that the answer is thread-relative; the benefit is that it is never
// optimistic about a guarantee the lane cannot make.
namespace {

bool provider_is_enabled(const llama_context * ctx) {
    if (ctx == nullptr || t_scope == nullptr) return false;
    return t_scope->memory() == llama_get_memory(ctx);
}

void provider_get_stats(const llama_context * ctx, herd::ext::prefab_stats * out) {
    if (out == nullptr) return;
    *out = herd::ext::prefab_stats{};  // value-initialised: every field is a real zero
    // The SAME identity check `is_enabled` makes, and it is not redundant: a thread-local
    // reservation belongs to one context's memory, but `get_stats` is asked about a context
    // the caller names. Without this line, a thread with an open reservation for context A
    // would report A's counters when asked about B -- a wrong answer from the right thread,
    // which is harder to notice than no answer. (A second context on the same thread is not
    // hypothetical: a driver that keeps a draft and a target model does exactly that.)
    if (ctx == nullptr || t_scope == nullptr || t_scope->memory() != llama_get_memory(ctx)) {
        // No reservation for THIS context on this thread. Nothing was measured for it, and
        // `captured` stays 0 so the reader is told exactly that instead of being handed a page
        // of plausible zeros.
        return;
    }
    out->n_ubatch_prefab = static_cast<std::uint64_t>(t_scope->padded_ubatches());
    out->n_pad_rows      = t_scope->pad_rows();
    out->captured        = herd::ext::prefab_captured_n_ubatch_prefab |
                           herd::ext::prefab_captured_n_pad_rows;
}

std::size_t provider_n_cached_factors(const llama_context *) {
    // A measured zero, not an unmeasured one: this lane has no factor cache, so the honest
    // count of cached factors is none. The distinction is the same one `captured` exists for.
    return 0;
}

const herd::ext::prefab_provider g_prefab_provider{&provider_is_enabled, &provider_get_stats,
                                                   &provider_n_cached_factors};

} // namespace

#if !defined(HERD_NO_AUTOLOAD)
__attribute__((constructor(101))) static void herd_prefab_split_autoload() {
    if (std::getenv("HERD_NO_AUTOLOAD") != nullptr) {
        return;
    }
    if (std::getenv("HERD_PREFAB_SPLIT") == nullptr) {
        return; // opt-in: an unused hook in the batch splitter is a cost with no benefit
    }
    void * target = herd::resolve(kSymbol);
    if (!target) {
        std::fprintf(stderr,
                     "[herd] prefab-split: HERD_PREFAB_SPLIT is set but "
                     "llama_batch_allocr::split_simple was not found\n");
        std::abort();
    }
    static_cast<void>(herd::add("prefab.split_simple", target, herd::raw(&herd_split_simple),
                                reinterpret_cast<void **>(&g_orig), nullptr));

    // The decode entry points, which publish whose ubatch is being split. Registered here
    // because the binding is useless without them: the seam would pad on the strength of
    // an open reservation alone, which is the cross-context destruction this closes.
    //
    // Failures are loud rather than skipped. A run with HERD_PREFAB_SPLIT=1 that silently
    // lost this hook would pad whatever context happened to decode, which is the failure
    // mode the hook exists to prevent -- so an unresolvable symbol stops the process
    // instead of leaving a lane that looks installed.
    void * decode_batch = herd::resolve(kDecodeBatchSymbol);
    void * decode_ext   = herd::resolve(kDecodeExtSymbol);
    if (!decode_batch || !decode_ext) {
        std::fprintf(stderr,
                     "[herd] prefab-split: HERD_PREFAB_SPLIT is set but the decode entry "
                     "point(s) were not found (%s); the lane cannot tell whose ubatch it is "
                     "and must not run\n",
                     decode_batch ? kDecodeExtSymbol : kDecodeBatchSymbol);
        std::abort();
    }
    static_cast<void>(herd::add("prefab.decode_batch", decode_batch,
                                herd::raw(&herd_decode_batch),
                                reinterpret_cast<void **>(&g_orig_decode_batch), nullptr));
    static_cast<void>(herd::add("prefab.decode_ext", decode_ext,
                                herd::raw(&herd_decode_ext),
                                reinterpret_cast<void **>(&g_orig_decode_ext), nullptr));

    // The statistics seam. Registered here rather than in the constructor's body above so that
    // a run which did not ask for the lane registers nothing: `prefab_available()` stays false
    // and a driver asking for stats still gets the hard error it is written to expect. Once the
    // lane IS asked for, "the lane is running but cannot be observed" becomes a defect, so a
    // failed registration aborts for the same reason the symbol lookups do.
    if (!herd::ext::prefab_register(&g_prefab_provider)) {
        std::fprintf(stderr,
                     "[herd] prefab-split: a prefab statistics provider was already registered; "
                     "two providers would mean two owners for one fact\n");
        std::abort();
    }
}
#endif
