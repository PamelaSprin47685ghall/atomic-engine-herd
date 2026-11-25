// tq/kv/capacity.h — decoupled KV capacity, as a decision rather than a field.
//
// What the fork added: `llama_context_params.n_ctx_kv`, a physical KV capacity
// independent of the logical window. Its purpose was not "a bigger cache" -- it was
// spare cells for auxiliary sequences (readers, drafts) that must not inflate
// `n_ctx`, because `n_ctx` also sizes the attention mask and the worst-case compute
// buffers (src/llama-context.cpp:954-982 in the fork).
//
// Why it is not a seam -- and what was WRONG in the earlier version of this comment.
//
// The first version said the const reference to llama_cparams made the size
// unreachable. That reasoning was wrong, and it was wrong in a way worth recording:
// `create_memory(const llama_cparams & cparams)` (src/llama-model.cpp:3068, declared
// at llama-model.h:1074) stops the CALLEE from modifying the argument. It does not
// stop a caller from passing a different one. A hook owns the call, so it may copy
// the struct, change n_ctx_seq, and invoke the original with the copy -- the same
// technique that made the prefab lane's re-reservation reachable. Constness of a
// parameter never decides whether an out-of-tree caller can supply another value.
//
// The next version of this comment said the obstacle was upstream's data model: ONE
// number, cells ARE the window, so only upstream could separate them. That was also
// wrong, and it is wrong by measurement rather than argument. patches/herd_nkv.cpp hooks
// create_memory, passes a copied cparams with a larger n_ctx_seq, and reports:
//
//   control               n_ctx=256 n_ctx_seq=256 kv_bytes=98304   second sequence rc=1
//   HERD_NKV_PHYSICAL=512 n_ctx=256 n_ctx_seq=256 kv_bytes=196608  second sequence rc=0
//
// kv_bytes is llama_memory_i::memory_breakdown() -- the cache's own account of the
// buffers it allocated -- so the doubling is the allocator speaking, not this layer
// echoing the number it passed. The window is identical in both runs, the logits are
// byte-identical, and a decode past the window is refused in both.
//
// Why one field does not mean one reading. cparams is consumed twice, at two moments:
// the context keeps its own copy (llama_n_ctx/n_ctx_seq, decode bounds, get_rope_factors
// read THAT), while create_memory reads the reference it is GIVEN, once, only to size the
// caches. A widened copy reaches the second and never the first. The third leg holds too:
// the mask is sized by cells in use, not by capacity -- llama_kv_cache::get_n_kv() is
// min(cells.size(), PAD(used_max_p1, 256)) -- so spare cells cost cache memory and
// nothing else. That is why the fork's n_ctx_kv can be reproduced without adding a field.
//
// Limits of the evidence, so it is not over-read: CPU only, one buffer type, one
// llama-architecture model. Hybrid and recurrent memories allocate per-sequence state
// rather than a cell pool, so extra capacity is meaningless for them; n_seq_max > 2
// combined with kv_unified=false was not measured. Those are untested, not impossible.
// The experiment is linked into no product and its hook stays inert unless
// HERD_NKV_PHYSICAL is set (HERD_BUILD_NKV is OFF by default).
//
// What this file provides is therefore the decision, not the allocation: the arithmetic
// that says what a requested configuration means, whether it is legal at all, and -- for
// the subset stock parameters can honour -- which of its two shapes expresses it. That is
// not a consolation prize. It is the difference between a driver refusing a request with
// a name and a driver silently running a different configuration than the one asked for,
// which is the defect this project keeps finding: batch-compose collapsing `--total-kv`
// onto `--n-ctx`, rerot-compose refusing "decoupled capacity" as one lumped inability.
//
// The rules are ported from the fork's own checks, each with the line it came from,
// so the refusals are the fork's refusals and not an invention.
#pragma once

#include <cstdint>
#include <string>

namespace tq::kv {

// How a configuration ends up being expressed.
enum class mode {
    unified,    // one logical window covering the whole capacity (n_ctx_seq == n_ctx)
    split,      // capacity divided into n_seq_max equal windows (n_ctx_seq == n_ctx/S)
    decoupled,  // physical > logical with a single window; the fork's addition
};

// Why a request was refused. Each value names the condition a caller violated, so a
// driver can print something a reader can act on instead of "unsupported".
enum class refusal {
    none,
    requires_unified,        // decoupled capacity with a split cache (fork :956-958)
    exceeds_range,           // beyond INT32_MAX-255 (fork :959-961)
    unsupported_arch,        // recurrent or DEEPSEEK4 models (fork :962-964)
    arena_conflict,          // shared CUDA phase arena (fork :965-967)
    capacity_below_window,   // physical must cover one complete window (fork :977-979)
    split_would_round,       // capacity is not n_seq_max * an aligned window
    window_zero,             // a zero logical window is not a configuration
};

const char * name(refusal why);
const char * name(mode kind);

struct request {
    uint32_t logical_window    = 0;   // per-sequence window the model may attend over
    // Total KV cells. 0 means "let the shape decide", which is NOT the same thing in
    // both shapes: unified gives capacity == window, split gives capacity ==
    // n_seq_max * window, because upstream derives one from the other
    // (llama-context.cpp:461-471). Measured: a 2048 window with three sequences is
    // built from n_ctx = 6144 and reports n_ctx_seq = 2048.
    uint32_t physical_capacity = 0;
    uint32_t n_seq_max         = 1;   // sequences the context offers to serve
    bool     kv_unified        = true;   // upstream's single-window mode
    bool     runtime_length    = false;  // fork: keep the exact capacity, do not pad to 256
    bool     recurrent_arch    = false;
    bool     deepseek4         = false;
    bool     shared_arena      = false;
};

struct plan {
    refusal  why        = refusal::none;
    mode     kind       = mode::unified;
    uint32_t logical    = 0;   // actual window after padding
    uint32_t physical   = 0;   // actual capacity after padding
    uint32_t n_seq_max  = 1;
    // Capacity expressed in whole logical windows. 2 means one extra window's worth
    // of cells is available for auxiliary sequences -- which is the entire point of
    // the feature, and the number a driver reports instead of a bare byte count.
    uint32_t spare_windows = 0;
    // True when upstream's own parameters can produce this plan with no patch. That
    // is: unified with physical == logical, or split with physical == S * logical and
    // S dividing evenly. Everything else is a configuration upstream cannot express.
    bool expressible_upstream = false;
};

// The decision. Pure: no allocation, no state, no logging.
plan evaluate(const request & req);

// A one-line explanation a driver can print. Empty for an accepted plan that is not
// expressible upstream? No -- never empty: an unexpressible plan must always say so,
// because that is the case where a caller is most likely to assume it worked.
std::string explain(const plan & result);

} // namespace tq::kv
