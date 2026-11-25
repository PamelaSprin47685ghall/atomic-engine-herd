// herd_nkv_probe.cpp — does a wider cell pool change anything a caller can observe?
//
// The claim under test: passing create_memory a copied cparams with a larger n_ctx_seq
// enlarges the KV pool while the context's logical window, its compute buffers and its
// output all stay exactly as they were.
//
// The probe is deliberately dumb: it prints the numbers and does one decode. Run it twice
// — once with HERD_NKV_PHYSICAL unset (control) and once set — and diff the output. Any
// difference other than the state size is a refutation.
//
// Three observations, in the order they matter:
//
//   A. llama_n_ctx() and llama_n_ctx_seq() must be IDENTICAL with and without the hook.
//      They return the context's own cparams (llama-context.cpp:1174), which the hook
//      never touches. If they moved, the hook reached further than intended and the
//      "logical window unchanged" claim is false.
//
//   B. llama_state_get_size() must be LARGER with the hook. It reports the serialisable
//      state, which includes the KV cells, so it is the cheapest honest evidence that the
//      pool actually grew rather than the call being quietly ignored.
//
//   C. The logits must be BIT-IDENTICAL. Same tokens, same positions, same arithmetic;
//      only the size of the array they live in changed. A difference here would mean the
//      mask or rope selection moved, which is the failure mode this whole experiment has
//      to rule out.
//
// Plus one negative check: decoding past the LOGICAL window must still be refused (or at
// least not succeed silently), because the pool is wider but the window is not. That is
// the difference between decoupled capacity and simply setting a larger --ctx-size.
#include "herd_patch.hpp"
#include "herd_nkv.hpp"

#include <llama.h>

// llama_get_memory() hands back a llama_memory_i*, whose only declaration of
// memory_breakdown() is in this private header. Included, never edited: the probe
// measures what the allocator reports, and that report is the whole point.
#include "llama-memory.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char * what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Position of the first row whose logits differ, or -1. Bit comparison on purpose: this
// experiment changes no arithmetic, so an epsilon would hide exactly the drift it exists
// to detect.
int first_difference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        return 0;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }

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

    // A window small enough that the pool's growth dominates it, and big enough to hold
    // the prompt plus a few tokens.
    auto cparams = llama_context_default_params();
    cparams.n_ctx     = 128;
    cparams.n_batch   = 32;
    cparams.n_ubatch  = 32;
    cparams.n_seq_max = 1;

    // The decoupled request, if any, is scoped to this one init call. Argument, not
    // environment: a probe that shares the production mechanism is the only kind whose
    // result carries over to the product, and an environment variable cannot say which
    // context it means.
    std::uint32_t physical = 0;
    // A second model, for the leak check: it needs a model whose memory is a cell pool
    // (so a request CAN apply) to prove that a refused request does not linger.
    const char * leak_model_path = nullptr;
    for (int i = 2; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--physical") == 0) {
            physical = static_cast<std::uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
        } else if (std::strcmp(argv[i], "--leak-model") == 0) {
            leak_model_path = argv[i + 1];
        }
    }
    std::unique_ptr<herd::nkv::scope> request;
    if (physical > 0) {
        request = std::make_unique<herd::nkv::scope>(physical);
    }

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "context creation failed\n");
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    // ---- observation A: the logical window is the context's own business ----
    // What the scoped request reported, before any other number: a request that was not
    // consumed means the hook never ran for this init call, and a request that was
    // consumed but not applied means the cache refused the shape. Both are failures of
    // the experiment, and neither may be reported as "decoupled".
    if (request) {
        // close() captures what the hook did; reading before it returns the empty
        // initial state, which is why the first run of this probe reported consumed=0
        // while kv_bytes had already doubled. The number that disagreed with reality was
        // ours, not upstream's.
        request->close();
        std::fprintf(stderr, "request: consumed=%d applied=%d cells=%u (asked %u)\n",
                     (int) request->consumed(), (int) request->applied(),
                     request->applied_cells(), physical);
        check(request->consumed(), "the hook consumed the scoped request");
        check(request->applied(), "the cache was sized from the request");
        check(request->applied_cells() == physical,
              "and the applied capacity is the one requested");
    }

    const std::uint32_t n_ctx     = llama_n_ctx(ctx);
    const std::uint32_t n_ctx_seq = llama_n_ctx_seq(ctx);
    const std::size_t   state     = llama_state_get_size(ctx);

    // The independent measurement of physical capacity, and the reason this probe
    // exists in this shape.
    //
    // `state_size` cannot answer it here: on this checkpoint it reported 17 bytes and
    // was identical with and without the hook, because the serialisable state does not
    // include the KV cells. Printing our own "physical=512" would only prove that
    // create_memory was handed 512 -- the question is whether the cache then ALLOCATED
    // 512 cells' worth of buffers.
    //
    // memory_breakdown() is upstream's own accounting: llama_kv_cache reports the size
    // of every buffer it allocated, summed over layers, and llama_memory_i declares it
    // virtual (llama-memory.h). So the number below is computed by the module that
    // allocated the memory, from its own buffer handles. If the pool grew, this grows;
    // if the hook was quietly ignored, this does not move.
    std::size_t kv_bytes = 0;
    if (llama_memory_t memory = llama_get_memory(ctx)) {
        for (const auto & entry : memory->memory_breakdown()) {
            kv_bytes += entry.second;
        }
    }
    std::fprintf(stderr, "n_ctx=%u n_ctx_seq=%u state_bytes=%zu kv_bytes=%zu\n",
                 n_ctx, n_ctx_seq, state, kv_bytes);

    // Token ids taken from the model's own vocabulary size, not hard-coded.
    //
    // The literal 259..262 named the four word tokens the tiny transformer emits after its
    // 256 byte-fallback entries. That is one model's layout: on a 262-token checkpoint id
    // 262 does not exist, and the probe exited before reaching anything it was asked to
    // measure -- which is how this was found, while pointing it at a recurrent model to
    // exercise the refusal path. Whatever the vocabulary, the last four ids are valid and
    // distinct, and no assertion here depends on which words they are.
    const int n_vocab_probe = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> prompt;
    for (int i = 0; i < 4 && i < n_vocab_probe; ++i) {
        prompt.push_back(static_cast<llama_token>(n_vocab_probe - 1 - i));
    }
    check(!prompt.empty(), "the prompt has tokens");

    std::unique_ptr<llama_batch_ext, void (*)(llama_batch_ext *)> batch(
        llama_batch_ext_init(ctx), &llama_batch_ext_free);
    if (!batch) {
        std::fprintf(stderr, "cannot create batch\n");
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 2;
    }

    // One row at a time, positions 0..n-1, logits on the final row: the simplest path
    // through decode, so a difference cannot be blamed on batching.
    for (std::size_t i = 0; i < prompt.size(); ++i) {
        llama_batch_ext_clear(batch.get());
        const int32_t row = llama_batch_ext_add_token(batch.get(), 0, prompt[i]);
        if (row < 0) {
            std::fprintf(stderr, "cannot stage token %zu\n", i);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        const llama_pos pos = static_cast<llama_pos>(i);
        if (!llama_batch_ext_set_pos(batch.get(), row, &pos) ||
            !llama_batch_ext_set_output_logits(batch.get(), row, true)) {
            std::fprintf(stderr, "cannot set position/logits for token %zu\n", i);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
        const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            std::fprintf(stderr, "decode failed at token %zu with %d\n", i, rc);
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return 2;
        }
    }
    llama_synchronize(ctx);

    // ---- observation C: the numbers, printed for an external diff ----
    const float * logits = llama_get_logits_ith(ctx, -1);
    check(logits != nullptr, "the final row produced logits");
    std::vector<float> row;
    if (logits) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        row.assign(logits, logits + n_vocab);
        double sum = 0.0;
        bool finite = true;
        for (float v : row) {
            sum += v;
            finite = finite && std::isfinite(v);
        }
        check(finite, "every logit is finite");
        // A byte digest, not a summary. Two runs are compared by this number, and a
        // digest over the raw float bytes cannot agree by rounding: if the mask or the
        // rope selection moved for any position, the bytes move with them. The first two
        // values and the sum are printed as well, because a human reading one run wants
        // something to recognise -- but the digest is what decides.
        std::uint64_t digest = 1469598103934665603ull; // FNV-1a offset basis
        for (float value : row) {
            unsigned char bytes[sizeof(float)];
            std::memcpy(bytes, &value, sizeof(float));
            for (unsigned char byte : bytes) {
                digest ^= byte;
                digest *= 1099511628211ull;
            }
        }
        std::fprintf(stderr,
                     "decoded_tokens=%zu logits[0]=%.9g logits[1]=%.9g sum=%.9f digest=%016llx\n",
                     prompt.size(), (double) row[0], (double) row[1], sum,
                     (unsigned long long) digest);

        // Optional raw dump so an external `cmp` can confirm the digest: a hash is only
        // as trustworthy as the code computing it, and one command settles that.
        if (argc >= 3) {
            std::FILE * out = std::fopen(argv[2], "wb");
            if (!out) {
                std::fprintf(stderr, "cannot open %s for writing\n", argv[2]);
                ++failures;
            } else {
                const std::size_t written = std::fwrite(row.data(), sizeof(float), row.size(), out);
                std::fclose(out);
                check(written == row.size(), "every logit reached the dump file");
            }
        }
    }

    // ---- the negative check: a wider pool must not widen the window ----
    // Decoding at a position past the logical window. Whether upstream refuses it or
    // writes it anyway is a fact worth printing rather than asserting: either answer is
    // information, and asserting the wrong one would be a false claim about behaviour.
    {
        llama_batch_ext_clear(batch.get());
        const int32_t row = llama_batch_ext_add_token(batch.get(), 0, prompt.back());
        llama_pos pos = static_cast<llama_pos>(n_ctx + 16);
        const bool staged = row >= 0 &&
                            llama_batch_ext_set_pos(batch.get(), row, &pos) &&
                            llama_batch_ext_set_output_logits(batch.get(), row, true);
        if (!staged) {
            std::fprintf(stderr, "beyond_window: staging refused\n");
        } else {
            const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
            std::fprintf(stderr, "beyond_window: position %d decode rc=%d%s\n",
                         (int) pos, rc, rc == 0 ? " (accepted)" : " (refused)");
            llama_synchronize(ctx);
        }
    }

    // ---- the decisive usability check: do the extra cells hold a second sequence? ----
    //
    // A larger pool that nothing can address would be a number, not a capability. In
    // stock upstream the only addressing unit is a cell, and every cell needs an owner
    // sequence; positions are bounded by n_ctx per sequence. So the honest question is
    // whether a SECOND sequence can occupy its own window while the first keeps its own
    // -- which is exactly what the fork's separate capacity was for.
    //
    // Setup: a fresh context with n_ctx = 256 (after padding) and n_seq_max = 2, unified.
    // Fill sequence 0 to its full window, then decode one token into sequence 1.
    //
    //   control (pool 256)  sequence 0 owns every cell; sequence 1 cannot be placed and
    //                       the decode fails inside find_slot -> upstream reports an
    //                       allocation failure.
    //   hook   (pool 512)   sequence 1 gets its own cells; the decode succeeds.
    //
    // Without the hook this test is expected to FAIL. That is the point: it is a
    // differential experiment, and its outcome is printed either way rather than
    // asserted, because both outcomes are information.
    if (std::getenv("HERD_NKV_USABILITY") != nullptr) {
        auto two = llama_context_default_params();
        two.n_ctx     = 256;
        two.n_batch   = 128;
        two.n_ubatch  = 128;
        two.n_seq_max = 2;
        two.kv_unified = true;

        llama_context * shared = llama_init_from_model(model, two);
        if (!shared) {
            std::fprintf(stderr, "usability: context creation failed\n");
        } else {
            std::unique_ptr<llama_batch_ext, void (*)(llama_batch_ext *)> fill(
                llama_batch_ext_init(shared), &llama_batch_ext_free);
            bool filled = fill != nullptr;
            for (llama_pos base = 0; base < 256 && filled; base += 128) {
                llama_batch_ext_clear(fill.get());
                for (llama_pos i = 0; i < 128; ++i) {
                    const int32_t row = llama_batch_ext_add_token(
                        fill.get(), 0, static_cast<llama_token>(259 + (i % 4)));
                    llama_pos pos = base + i;
                    if (row < 0 || !llama_batch_ext_set_pos(fill.get(), row, &pos)) {
                        filled = false;
                        break;
                    }
                }
                if (!filled || llama_process(shared, LLAMA_PROCESS_TYPE_DECODE, fill.get()) != 0) {
                    filled = false;
                }
            }
            std::fprintf(stderr, "usability: sequence 0 filled=%d (256 positions)\n", (int) filled);

            // The second sequence, at position zero: this needs one free cell of its own.
            llama_batch_ext_clear(fill.get());
            const int32_t row = llama_batch_ext_add_token(fill.get(), 1, 259);
            llama_pos pos = 0;
            bool staged = row >= 0 && llama_batch_ext_set_pos(fill.get(), row, &pos) &&
                          llama_batch_ext_set_output_logits(fill.get(), row, true);
            const int32_t rc = staged ? llama_process(shared, LLAMA_PROCESS_TYPE_DECODE, fill.get())
                                      : -2;
            std::size_t split_kv_bytes = 0;
            if (llama_memory_t memory = llama_get_memory(shared)) {
                for (const auto & entry : memory->memory_breakdown()) {
                    split_kv_bytes += entry.second;
                }
            }
            std::fprintf(stderr,
                         "usability: second sequence decode rc=%d%s (kv_bytes=%zu)\n",
                         rc, rc == 0 ? " (has cells of its own)" : " (no cells available)",
                         split_kv_bytes);
            llama_synchronize(shared);
            // The batch goes first. `llama_batch_ext` holds a `llama_memory_i *` into the
            // context's memory (llama-batch.h:102-108), and `llama_batch_ext_free` is a
            // plain delete today -- so freeing it after the context happens to work, and
            // that is exactly the kind of accident this probe must not rely on. Nothing
            // here needs the batch once the decode is done, so it is released while its
            // context is still alive.
            fill.reset();
            llama_free(shared);
        }
    }

    // ---- the shape gate: a split cache must not be widened ----
    //
    // In a non-unified cache `n_ctx_seq` IS the per-sequence window
    // (llama-context.cpp:456-471), so the same hook that adds cells under kv_unified
    // would widen every sequence's window here -- "a bigger --ctx-size" wearing the
    // name of decoupled capacity, and it would double the cache while the context kept
    // reporting the old window. The gate refuses; this asserts it fired rather than
    // trusting that the numbers would have looked different (they would not).
    if (std::getenv("HERD_NKV_SPLIT_TEST") != nullptr && herd::nkv::calls() > 0) {
        auto split = llama_context_default_params();
        split.n_ctx      = 256;
        split.n_batch    = 32;
        split.n_ubatch   = 32;
        split.n_seq_max  = 2;
        split.kv_unified = false;

        const int refusals_before = herd::nkv::shape_refusals();
        const std::uint32_t window_before = herd::nkv::physical_cells_used();

        llama_context * narrow = llama_init_from_model(model, split);
        if (!narrow) {
            std::fprintf(stderr, "split: context creation failed\n");
        } else {
            std::size_t bytes = 0;
            if (llama_memory_t memory = llama_get_memory(narrow)) {
                for (const auto & entry : memory->memory_breakdown()) {
                    bytes += entry.second;
                }
            }
            std::fprintf(stderr,
                         "split: n_ctx=%u n_ctx_seq=%u kv_bytes=%zu refusals_before=%d "
                         "refusals_now=%d physical_reported=%u\n",
                         llama_n_ctx(narrow), llama_n_ctx_seq(narrow), bytes,
                         refusals_before, herd::nkv::shape_refusals(),
                         herd::nkv::physical_cells_used());
            check(herd::nkv::shape_refusals() > refusals_before,
                  "the shape gate refused the split context's request");
            // Upstream's own numbers for n_ctx=256, S=2: the window is PAD(256/2, 256)
            // = 256 (the 256 padding floor, not 128) and the cache holds 256 x 2 = 512
            // cells. Asserted from the runtime that produced them rather than from the
            // division alone -- the first version of this test expected 128 and was
            // wrong about upstream, not about the hook.
            check(llama_n_ctx_seq(narrow) == 256,
                  "the split window is upstream's own 256, not the requested 512");
            check(llama_n_ctx(narrow) == 512,
                  "and the total capacity is the window times the sequence count");
            check(bytes == 196608,
                  "the split cache allocated 512 cells, which is its own shape's size");
            llama_free(narrow);
        }
    }

    // ---- ownership: a refused request must not linger into the next context ----
    //
    // The request is thread-local state, and the failure mode if it outlived its scope is
    // the one that made the environment-variable version wrong: the NEXT context created
    // on this thread silently inherits a capacity nobody asked for. That is silent by
    // construction -- the run succeeds, the cache is bigger, and the only trace is a
    // config field the operator has no reason to inspect.
    //
    // The sequence, and why each step is needed:
    //
    //   1. open a request against a model that refuses it (the caller's model here is a
    //      recurrent one when this flag is used with the mamba checkpoint): consumed, not
    //      applied -- the loud refusal, on a real init call;
    //   2. destroy the scope: close() copies the outcome and clears the thread-local;
    //   3. create a context on a model that CAN be decoupled, with no scope open. It must
    //      come out undecoupled, and `applied_requests()` must not have moved.
    //
    // Step 3 is the assertion; step 1 only puts the thread-local into the state that would
    // leak. Model order is given by the caller because it is the caller who has both.
    if (leak_model_path != nullptr) {
        auto leak_params = llama_model_default_params();
        leak_params.n_gpu_layers = 0;
        llama_model * second = llama_model_load_from_file(leak_model_path, leak_params);
        if (!second) {
            std::fprintf(stderr, "leak: cannot load %s\n", leak_model_path);
            ++failures;
        } else {
            const int applied_before = herd::nkv::applied_requests();
            auto leak_ctx_params = llama_context_default_params();
            leak_ctx_params.n_ctx    = 256;
            leak_ctx_params.n_batch  = 8;
            leak_ctx_params.n_ubatch = 8;

            // 1. a request the hook will refuse: the first model here is the recurrent one
            //    when this mode is used as intended.
            {
                herd::nkv::scope refused(512);
                llama_context * first_ctx = llama_init_from_model(model, leak_ctx_params);
                refused.close();
                std::fprintf(stderr,
                             "leak: first (refused) consumed=%d applied=%d cells=%u\n",
                             (int) refused.consumed(), (int) refused.applied(),
                             refused.applied_cells());
                check(refused.consumed(), "the refused request reached the hook");
                check(!refused.applied(), "and was not applied");
                if (first_ctx) {
                    llama_free(first_ctx);
                }
            }   // scope destroyed here

            // 3. no request open: this context must be the ordinary, undecoupled one.
            llama_context * clean = llama_init_from_model(second, leak_ctx_params);
            check(clean != nullptr, "the second context is created");
            if (clean) {
                const std::uint32_t window = llama_n_ctx_seq(clean);
                std::fprintf(stderr,
                             "leak: second context window=%u applied_requests=%d (before %d)\n",
                             window, herd::nkv::applied_requests(), applied_before);
                check(window == 256, "the second context keeps the model's own window");
                check(herd::nkv::applied_requests() == applied_before,
                      "the closed scope did not apply to the next context");
                llama_free(clean);
            }
            llama_model_free(second);
        }
    }

    // ---- one scope, two contexts: the second must not inherit the request ----
    //
    // The isolation claim, executed. A scope names one context's capacity; upstream calls
    // create_memory once per init (llama-context.cpp:602 is the only call site), so a
    // second context created while the scope is open is a caller error -- and the one
    // thing that must not happen is that it inherits the widening. Measured through the
    // cache's own buffer accounting, because that is what the widening actually changes:
    // 256 cells are 98304 bytes and 512 are 196608 on this model.
    if (std::getenv("HERD_NKV_TWO_CONTEXTS") != nullptr) {
        auto pair_params = llama_context_default_params();
        pair_params.n_ctx    = 256;
        pair_params.n_batch  = 8;
        pair_params.n_ubatch = 8;

        const int applied_before = herd::nkv::applied_requests();
        herd::nkv::scope one(512);
        llama_context * first  = llama_init_from_model(model, pair_params);
        llama_context * second = llama_init_from_model(model, pair_params);
        one.close();

        const auto bytes_of = [](llama_context * ctx) -> std::size_t {
            std::size_t total = 0;
            if (llama_memory_t memory = llama_get_memory(ctx)) {
                for (const auto & entry : memory->memory_breakdown()) {
                    total += entry.second;
                }
            }
            return total;
        };

        std::fprintf(stderr,
                     "two_contexts: consumptions=%d first=%zu bytes second=%zu bytes\n",
                     one.consumptions(), first ? bytes_of(first) : 0,
                     second ? bytes_of(second) : 0);
        check(first != nullptr && second != nullptr, "both contexts were created");
        check(one.consumed(), "the request was consumed");
        check(one.consumptions() == 2, "both init calls reached the hook");
        if (first && second) {
            check(bytes_of(first) == 196608, "the first context was widened to 512 cells");
            check(bytes_of(second) == 98304,
                  "the second kept the window's own 256 cells: the request did not leak");
        }
        check(herd::nkv::applied_requests() == applied_before + 1,
              "exactly one application, not two");
        if (second) { llama_free(second); }
        if (first)  { llama_free(first); }
    }

    // ---- what the hook did, if it was installed ----
    if (herd::nkv::calls() > 0) {
        std::fprintf(stderr, "hook: calls=%d logical=%u physical=%u shape_refusals=%d\n",
                     herd::nkv::calls(), herd::nkv::logical_window_seen(),
                     herd::nkv::physical_cells_used(), herd::nkv::shape_refusals());
        check(herd::nkv::logical_window_seen() == n_ctx_seq,
              "the hook saw the same window the context reports");
    } else {
        std::fprintf(stderr, "hook: not installed\n");
    }

    // Batch before context: see the note in the usability block. `batch` owns a
    // heap object that names this context's memory, and the ordering is a fact about
    // upstream's destructor that this probe should not depend on being benign.
    batch.reset();
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    std::fprintf(stderr, "%s\n", failures ? "nkv probe FAILED" : "nkv probe holds");
    return failures ? 1 : 0;
}
