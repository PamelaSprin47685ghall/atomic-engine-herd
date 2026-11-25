cmake_minimum_required(VERSION 3.16)
project(TQReader VERSION 0.1.0 LANGUAGES CXX)

# The reader lane's rules: what a valid attention view is, and which physical cells
# it resolves to. No llama, no ggml, no cache: the inputs are a view set and a
# (seq, pos) -> cell map, so the rules are testable without a model.
#
# This is a domain core, not a driver. The graph construction that would consume
# these spans needs fields in upstream's llm_graph_input_attn_kv and a build
# function that emits per-span gather/rope nodes; see README.md for why that is not
# reachable from a function-entry seam.
add_executable(tq_reader_index_ledger_test tests/index-ledger-test.cpp)
target_link_libraries(tq_reader_index_ledger_test PRIVATE tq_reader)
target_compile_options(tq_reader_index_ledger_test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
add_test(NAME tq_reader.index_ledger COMMAND tq_reader_index_ledger_test)

# The ledger's answer in the shape the graph takes. Until this existed, `resolved_view` and
# `query_view` were structurally identical and nothing crossed between them, so every emitter test
# was fed a hand-built fixture and the index the ledger computes could not reach a graph at all.
# The test asserts exact span boundaries and cell ORDER on cases where a plausible conversion
# (sorting the cells, one span per cell, one span for all) produces a view the emitter accepts and
# a number that is quietly different.
add_executable(tq_reader_view_bridge_test tests/view-bridge-test.cpp)
target_link_libraries(tq_reader_view_bridge_test PRIVATE tq_reader)
target_compile_options(tq_reader_view_bridge_test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
add_test(NAME tq_reader.view_bridge COMMAND tq_reader_view_bridge_test)

# The chain end to end: ledger -> resolve -> to_query_view -> emit_reader_attention -> numbers.
# Every other reader test starts after the index (hand-built query_view) or stops at the answer
# (resolved_view), so "the cells the ledger resolved are the cells the graph read" had never been
# executed. The reference is a double-precision transcription of the emitter's documented
# arithmetic over the SAME fixture, with three non-vacuity guards: the reference is not near zero,
# the heads differ (so a wrong GQA grouping is visible), and the softmax width is asserted to equal
# the number of cells the view names (so a dropped cell cannot pass as agreement).
add_executable(tq_reader_view_bridge_emitter_test tests/view-bridge-emitter-test.cpp)
target_link_libraries(tq_reader_view_bridge_emitter_test PRIVATE tq_reader tq_reader_graph ggml
    ggml-cpu ggml-base)
target_compile_options(tq_reader_view_bridge_emitter_test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
add_test(NAME tq_reader.view_bridge_emitter COMMAND tq_reader_view_bridge_emitter_test)

# The reader's physical index ledger: cell_id -> slot, with the reverse map that makes slot reuse
# reportable.
#
# Not gated on ggml, because the module has no ggml dependency by design: it is the discipline the
# fork states in one sentence ("no physical index survives a decode call") turned into a mechanism
# -- durable identity is (seq, pos), and a slot is an address that can be taken away.
#
# What the cases pin, each against the fork's own `prepare_attention_views` (llama-kv-cache.cpp):
# every precondition in the fork's order, its own-entry rule (exactly one, at the query's virtual
# position, never resolved to a slot), the batch rules (one view per sequence, no token queried
# twice), and all-or-nothing resolution with the offending view named. Two things here are this
# module's additions and are labelled as such rather than presented as upstream's contract:
# `slot_conflict` (the fork has no reverse map, so it cannot notice two cells claiming one
# address) and the epoch ceiling.
add_library(tq_reader STATIC src/views.cpp src/attention.cpp src/graph_station.cpp src/arena_budget.cpp src/physical.cpp src/index_ledger.cpp src/seq_cell_projection.cpp)

# Attention over upstream's RUNTIME row table (`kv_rows`) and mask. Its own library because it needs
# ggml headers and links ggml, while `tq_reader` stays light enough for tests that never build a graph.
add_library(tq_reader_rows STATIC src/rows_attention.cpp src/rows_validation.cpp)
add_library(TQReader::rows ALIAS tq_reader_rows)
target_include_directories(tq_reader_rows PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tq_reader_rows PUBLIC ggml)

add_executable(tq_reader_rows_attention_test tests/rows-attention-test.cpp)
target_link_libraries(tq_reader_rows_attention_test PRIVATE tq_reader_rows ggml ggml-cpu)
target_compile_options(tq_reader_rows_attention_test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
add_test(NAME tq_reader.rows_attention COMMAND tq_reader_rows_attention_test)

# The host-side VERIFICATION of upstream's runtime row table: every non-negative row inside the cache,
# every negative row invisible to every token of every slice. It is a check and not a compaction,
# because the table is a graph INPUT -- its values arrive at compute time, so the host cannot rewrite
# what the kernel will read; it can only verify the invariant that makes the kernel's `max(row, 0)`
# clamp harmless. The cases that matter are the ones a plausible implementation gets wrong: padding
# visible to ONE token out of three (a single-column check certifies it), a row exactly AT the cache
# depth, NaN and +inf at a padding column (neither is `-inf`), a finite value in the stride gap (which
# fails a scan that reads the mask contiguously), and a second slice whose table and mask differ from
# the first's.
add_executable(tq_reader_rows_validation_test tests/rows-validation-test.cpp)
target_link_libraries(tq_reader_rows_validation_test PRIVATE tq_reader_rows ggml)
target_compile_options(tq_reader_rows_validation_test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
add_test(NAME tq_reader.rows_validation COMMAND tq_reader_rows_validation_test)

# The seam's DESCRIPTION contract, driven with no ggml and no tensor.
#
# The checks used to be inline in `patches/herd_reader_attn.cpp`, where each refusal called
# `abort()`. A function that aborts cannot be tested case by case: a process death reports an
# exit code, not which check fired, so "the rank check runs before the stride check" and
# "this case is decided before any byte is read" were both unobservable -- and the only path
# ever executed was the accepting one. Extracting a pure function over numbers is what makes
# those assertions possible, and the test below exercises every refusal by name.
#
# It links `tq_reader` only: `seam_contract.h` is header-only and ggml-free precisely so a
# case can be a hand-built description rather than a constructed tensor.
add_executable(tq_reader_seam_contract_test tests/seam-contract-test.cpp)
target_link_libraries(tq_reader_seam_contract_test PRIVATE tq_reader)
target_compile_options(tq_reader_seam_contract_test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
add_test(NAME tq_reader.seam_contract COMMAND tq_reader_seam_contract_test)

# Upstream's own row-table operator, executed on the CPU against arithmetic written from its
# documented contract. The reader lane previously had no test that ran `ggml_flash_attn_ext_rows`
# at all: `upstream-attention-contrast` replays the operands of a graph WE build, and
# `rows_attention` validates a row list without computing. This one asks whether the operator a
# replacement would depend on actually behaves as ggml.h says -- with a non-monotonic row list,
# interleaved -1 padding, F16 mask entries carrying finite biases, and GQA.
add_executable(tq_reader_flash_attn_rows_test tests/flash-attn-rows-test.cpp)
target_link_libraries(tq_reader_flash_attn_rows_test PRIVATE ggml ggml-cpu)
target_compile_options(tq_reader_flash_attn_rows_test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
add_test(NAME tq_reader.flash_attn_rows COMMAND tq_reader_flash_attn_rows_test)

# The SAME operator with SEVERAL query tokens, which the test above cannot cover: there `q` is
# `[head_dim, 1, n_head]` and the mask is `[n_kv, 1, 1, 1]`, so the token axis is a constant and the
# two things a multi-token call can get wrong are invisible -- reading one token's visibility for
# another, and indexing the row table per token instead of per column. Here T = 3 with NON-NESTED
# visible sets, a NON-IDENTITY non-monotonic row table `{5,-1,2,7,-1,0}` with padding interleaved at
# columns 1 and 4, an F16 mask (the width the CPU kernel asserts), and GQA with group 2. The
# reference is written per (token, head) from the definition, and a control run reads row `c` instead
# of row `kv_rows[c]` to prove the indirection is actually exercised.
add_executable(tq_reader_flash_attn_rows_multi_test tests/flash-attn-rows-multi-test.cpp)
target_link_libraries(tq_reader_flash_attn_rows_multi_test PRIVATE ggml ggml-cpu)
target_compile_options(tq_reader_flash_attn_rows_multi_test PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
add_test(NAME tq_reader.flash_attn_rows_multi COMMAND tq_reader_flash_attn_rows_multi_test)
add_library(TQReader::reader ALIAS tq_reader)
set_target_properties(tq_reader PROPERTIES EXPORT_NAME reader)
target_compile_features(tq_reader PUBLIC cxx_std_17)
target_compile_options(tq_reader PRIVATE $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
target_include_directories(tq_reader PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# The emission half: the graph nodes that produce the numbers. Defined HERE rather than inside
# `if(BUILD_TESTING)` because a product that wants the reader lane has to be able to link it, and a
# library that only exists when tests are on is a library no product can depend on. That was not a
# hypothetical: the publisher patch needed it, added the link, and every configuration with
# `-DBUILD_TESTING=OFF` failed at generate time with an unknown target.
#
# The gate is `TARGET ggml`, not the test switch. The emitter takes a `ggml_context` by
# construction, so a build that consumes an installed llama package without ggml sources genuinely
# cannot have it -- and in that configuration this target simply does not exist, which the linking
# side must check with `if(TARGET tq_reader_graph)` rather than assume.
if(TARGET ggml AND TARGET ggml-cpu)
    # EXCLUDE_FROM_ALL: the target must EXIST in every configuration (that is what lets a linking
    # side test `if(TARGET tq_reader_graph)`), but a build that never links it must not pay for two
    # translation units of graph emitter. Without this, `-DBUILD_TESTING=OFF` compiled both files on
    # a plain `all` even when no target referenced them -- dead weight that also meant a product's
    # build could fail on emitter code it does not use.
    #
    # Tests are unaffected: `all` still builds the test executables, and each of them links this
    # library, so CMake builds it as their dependency.
    add_library(tq_reader_graph STATIC EXCLUDE_FROM_ALL
        src/graph_emit.cpp src/graph_emit_physical.cpp src/graph_emit_prefill.cpp)
    add_library(TQReader::graph ALIAS tq_reader_graph)
    target_compile_features(tq_reader_graph PUBLIC cxx_std_17)
    target_compile_options(tq_reader_graph PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
    target_link_libraries(tq_reader_graph PUBLIC tq_reader PRIVATE ggml)
    target_include_directories(tq_reader_graph PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
endif()

include(CTest)
if(BUILD_TESTING)
    # The transcription of upstream's own occupancy-bitmap -> row-table conversion, plus the two
    # joins that make a row table safe to hand to a graph: the identity round trip (projected cell ->
    # ledger slot -> the same cell) and the `-1` padding, refused by `validate_runtime_rows` when a
    # token can still see it. Links `tq_reader_rows` because the third case drives the validator; the
    # projection itself stays ggml-free, in `tq_reader`.
    add_executable(tq_reader_seq_cell_projection_test tests/seq-cell-projection-test.cpp)
    target_link_libraries(tq_reader_seq_cell_projection_test PRIVATE tq_reader_rows tq_reader ggml)
    target_compile_options(tq_reader_seq_cell_projection_test PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
    add_test(NAME tq_reader.seq_cell_projection COMMAND tq_reader_seq_cell_projection_test)

    # The projection against UPSTREAM'S OWN `get_seq_cells`, executed, not described.
    #
    # `tq_reader.seq_cell_projection` compares the projection with a fixture written from the same
    # reading of `llama-kv-cells.h:460` that the projection is made of -- one paragraph transcribed
    # twice. Agreement there means the transcription is self-consistent; only running the original
    # says whether the paragraph was read correctly. This target includes
    # `third_party/upstream/src/llama-kv-cells.h` and calls the real member on a real
    # `llama_kv_cells`, the same way `tq_reader.graph_result_publish` includes that directory for
    # `llm_graph_result`. No model, no context, no ggml: `llama_kv_cells` is a self-contained class.
    add_executable(tq_reader_seq_cell_projection_ab_test tests/seq-cell-projection-ab-test.cpp)
    target_include_directories(tq_reader_seq_cell_projection_ab_test PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/upstream/src
        ${CMAKE_SOURCE_DIR}/third_party/upstream/include
        # `llama-kv-cells.h` includes `llama.h`, which includes `ggml.h`: the class is not
        # self-contained after all, and the FIRST build of this test failed exactly here. The
        # directory is listed explicitly as well as inherited through `ggml` below, because the
        # include that broke was reached before any target-level propagation could help.
        ${CMAKE_SOURCE_DIR}/third_party/upstream/ggml/include)
    # `ggml` is linked for its HEADERS, not its kernels: nothing here builds a graph, but
    # `llama.h` declares `ggml_tensor` and the reader is deliberately ggml-free, so the dependency
    # stops at this test target.
    target_link_libraries(tq_reader_seq_cell_projection_ab_test PRIVATE tq_reader ggml)
    target_compile_options(tq_reader_seq_cell_projection_ab_test PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
    add_test(NAME tq_reader.seq_cell_projection_ab
             COMMAND tq_reader_seq_cell_projection_ab_test)

    add_executable(tq_reader_views_test tests/views-test.cpp)
    target_link_libraries(tq_reader_views_test PRIVATE tq_reader)
    add_test(NAME tq_reader.views COMMAND tq_reader_views_test)

    # The attention arithmetic, checked against a hand-written reference of the same
    # definition. No model, no ggml: the comparison is between two implementations of
    # one formula, which is the only way to test a numerical core without a checkpoint.
    add_executable(tq_reader_attention_test tests/attention-test.cpp)
    target_link_libraries(tq_reader_attention_test PRIVATE tq_reader)
    add_test(NAME tq_reader.attention COMMAND tq_reader_attention_test)

    # The request ledger: one request per graph build, refusals named and counted.
    #
    # No ggml, no model, no cache -- the station holds the question "was a reader emission
    # asked for during this build, and with what view", and nothing else. That is what makes
    # it testable here rather than only inside a run that has a checkpoint loaded, and it is
    # the part of the lane that must be right before any emission is wired: a request that
    # leaks across builds produces a plausible graph built from the wrong view.
    add_executable(tq_reader_graph_station_test tests/graph-station-test.cpp)
    target_link_libraries(tq_reader_graph_station_test PRIVATE tq_reader)
    target_compile_options(tq_reader_graph_station_test PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Werror>)
    add_test(NAME tq_reader.graph_station COMMAND tq_reader_graph_station_test)

    # The reader's arithmetic against ggml's own kernels.
    #
    # The test above compares the core with a hand-written reference of the same
    # definition, which cannot catch a wrong definition: both sides would be wrong the same
    # way. This one computes the expected values with ggml on llama's layouts, so the head
    # pairing, the gather and the reduction are checked against the thing they must match.
    #
    # Gated on the in-tree ggml targets: the core stays ggml-free, and a build that consumes
    # an installed llama package without ggml sources simply does not get this test.
    if(TARGET ggml AND TARGET ggml-cpu)
        add_executable(tq_reader_attention_ggml_test tests/attention-ggml-test.cpp)
        target_link_libraries(tq_reader_attention_ggml_test PRIVATE tq_reader ggml ggml-cpu)
        target_compile_options(tq_reader_attention_ggml_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.attention_ggml COMMAND tq_reader_attention_ggml_test)

        # The emission half of the same feature: the graph nodes that produce the numbers.
        #
        # `tq_reader_graph` is defined ABOVE, outside this switch, because a product has to be able
        # to link it. Only the tests live here.

        add_executable(tq_reader_graph_emit_test tests/graph-emit-test.cpp)
        target_link_libraries(tq_reader_graph_emit_test PRIVATE tq_reader_graph ggml ggml-cpu)
        add_test(NAME tq_reader.graph_emit COMMAND tq_reader_graph_emit_test)

        # The emitter against upstream's REAL cache views, not a convenient contiguous fixture:
        # both branches of llama_kv_cache::get_v, with the strides they actually have. The
        # v_trans branch is the one whose shape can coincide with the default's, so only a
        # stride test separates a correct read from a plausible wrong one.
        add_executable(tq_reader_graph_emit_cache_layout_test tests/graph-emit-cache-layout-test.cpp)
        target_link_libraries(tq_reader_graph_emit_cache_layout_test PRIVATE tq_reader_graph ggml ggml-cpu)
        target_compile_options(tq_reader_graph_emit_cache_layout_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.graph_emit_cache_layout
                 COMMAND tq_reader_graph_emit_cache_layout_test)

        # The emission through a context that allocates nothing, filled the way upstream fills its
        # graph inputs.
        #
        # The property is not "the emission ran": it is that the numbers produced through a real
        # allocator match an allocating run. Indices are written ONLY after allocation through the
        # tensor pointers the sink recorded, so an emitter that had silently used its own null-data
        # tensors would select whatever the allocator's buffer happened to contain and the outputs
        # would differ.
        #
        # It also pins the refusal: a null-data context with no sink is `index_data_unavailable`,
        # not a graph built on uninitialised indices.
        add_executable(tq_reader_graph_emit_noalloc_test tests/graph-emit-noalloc-test.cpp)
        target_link_libraries(tq_reader_graph_emit_noalloc_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_graph_emit_noalloc_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.graph_emit_noalloc COMMAND tq_reader_graph_emit_noalloc_test)

        # Which tensors need GGML_TENSOR_FLAG_INPUT for a real scheduler to place a reader emission.
        #
        # Four variants, each in its own child process, because the failure is GGML_ABORT
        # (ggml-alloc.c:686) and an in-process test would take the binary down with it. Measured on
        # this pin: no flags -> abort; k/v/q flagged -> alloc_graph succeeds; index tensors only ->
        # abort; everything flagged -> succeeds.
        #
        # This exists because three rounds of this lane disagreed about the answer, and the round
        # that "disproved" the requirement had deleted flags from a DIFFERENT test function -- a
        # control that changed nothing about the thing under test. The isolation is now an
        # experiment that runs on every build rather than a claim in a comment.
        add_executable(tq_reader_graph_emit_input_flag_test tests/graph-emit-input-flag-test.cpp)
        target_link_libraries(tq_reader_graph_emit_input_flag_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_graph_emit_input_flag_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.graph_emit_input_flag COMMAND tq_reader_graph_emit_input_flag_test)

        # The reader's PHYSICAL-form attention as nodes: scatter the span scores into a cache-length
        # tensor filled with -INFINITY, ONE global softmax over that length, then the native value
        # reduction — the form the fork's own `build_attn_reader` builds, and for its stated reason
        # ("a tiny change in a compact sum can cross a later quantization barrier").
        #
        # Not the same test as `graph_emit` next door: that one emits the COMPACT form (gather the named
        # cells, softmax over those). Both are kept because the two produce different floats, and a port
        # that only offers the compact one is not the fork's behaviour.
        #
        # The cases pin the values against the domain core AND the two properties that make the form
        # what it is: hidden cells weigh exactly zero, and the full-length weights sum to n_head. They
        # also pin the refusals this emitter owes rather than approximating — a mixed set of effective
        # positions (the per-span rope rebuild it does not do), a cell-major V cache (the transpose it
        # does not do), and a no_alloc context (the N+1 index handoff it does not do).
        add_executable(tq_reader_graph_emit_physical_test tests/graph-emit-physical-test.cpp)
        target_link_libraries(tq_reader_graph_emit_physical_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_graph_emit_physical_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.graph_emit_physical COMMAND tq_reader_graph_emit_physical_test)

        # The index hand-off across the LIFECYCLE, not around it.
        #
        # Every existing emitter test writes gather indices with a hand-written `memcpy` into
        # `entry.index->data`. That line is legal only after the scheduler places the tensors, and it
        # bypasses `publish_index_inputs` -- the function that is supposed to own this hand-off had
        # therefore never been run on the path it exists for. This test walks the window upstream's
        # own `set_inputs` lives in (construct in no_alloc -> place -> publish -> compute), asserts
        # that publishing BEFORE placement is refused by name rather than writing through null, reads
        # the published cells back out of the TENSORS, and compares the computed result against the
        # definition. A publish that reached the wrong buffer, or a graph that gathered the wrong
        # cells, fails here and nowhere else.
        add_executable(tq_reader_index_lifecycle_test tests/index-lifecycle-test.cpp)
        target_link_libraries(tq_reader_index_lifecycle_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_index_lifecycle_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.index_lifecycle COMMAND tq_reader_index_lifecycle_test)

        # The physical emitter's MASK path. It was wired in (a `graph_inputs::mask` passed straight to
        # `ggml_soft_max_ext`, which is the operator upstream hands `kq_mask` to) and then never run:
        # the compiler saw it, no test did. This grades it against a reference written from the
        # description -- a per-column BIAS, with `-inf` meaning invisible -- and also asserts the
        # limitation the construction implies: a finite bias cannot open a position no span names,
        # because the emitter fills the cache with `-inf` and `-inf + finite` is still `-inf`.
        add_executable(tq_reader_physical_mask_test tests/physical-mask-test.cpp)
        target_link_libraries(tq_reader_physical_mask_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_physical_mask_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.physical_mask COMMAND tq_reader_physical_mask_test)

        # The index-entry contract for a FULL-COVERAGE view: how many entries, in what layout, and
        # whether they can be published and computed. A previous experiment logged `0 entries` for
        # this view and that zero was read as "the gather degenerates", which would have meant the
        # publisher is not on the critical path. The emitter pushes one entry per span and then the
        # scatter unconditionally, so zero is impossible for an accepted view -- the log had read a
        # moved-from vector. This test measures the count instead of reasoning about it, and it does
        # so for three shapes so that neither the count nor the tiling can be a coincidence of one.
        add_executable(tq_reader_full_coverage_entries_test tests/full-coverage-entries-test.cpp)
        target_link_libraries(tq_reader_full_coverage_entries_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_full_coverage_entries_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.full_coverage_entries
                 COMMAND tq_reader_full_coverage_entries_test)

        # The index hand-off through the HOST lifecycle: `ggml_backend_sched_alloc_graph` instead of
        # `ggml_backend_alloc_ctx_tensors`, and a graph-owned input object standing in for upstream's
        # `set_inputs`. The earlier test placed the whole context eagerly, which cannot distinguish
        # "a node consumes this tensor" from "it happens to exist" -- and that distinction is exactly
        # what made the deleted publisher abort on layer 0. This test also asserts the NEGATIVE case
        # (an unreferenced tensor stays unplaced) so the positive one means something.
        # There is deliberately NO reuse assertion here: an earlier version had
        # one, and it published the same cells into the same tensors through a stack-local object,
        # which is an idempotent rewrite and says nothing about a graph being reused. Upstream's
        # reuse is `llm_graph_input_i::can_reuse`, which ggml's scheduler does not have, so reuse is
        # untested and is described as such in HANDOFF §7.16 rather than faked here.
        add_executable(tq_reader_host_lifecycle_test tests/host-lifecycle-test.cpp)
        target_link_libraries(tq_reader_host_lifecycle_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_host_lifecycle_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.host_lifecycle COMMAND tq_reader_host_lifecycle_test)

        # The index hand-off through upstream's OWN graph result: `llm_graph_result::add_input` and
        # `llm_graph_result::set_inputs`, with the scheduler placing the graph that result owns.
        #
        # This links `llama` and includes `third_party/upstream/src`, which no other reader test does,
        # and both are required for the reason the test exists: `llm_graph_result` is the class
        # upstream's decode path uses to own graph inputs, so testing a stand-in could only ever
        # restate the stand-in. `llm_graph_result(int64_t max_nodes)` is PUBLIC and builds its own
        # no_alloc context and graph, so no model, context or memory context is needed -- which is
        # what makes a real test possible at all instead of a mock.
        add_executable(tq_reader_graph_result_publish_test tests/graph-result-publish-test.cpp)
        target_include_directories(tq_reader_graph_result_publish_test PRIVATE
            ${CMAKE_SOURCE_DIR}/third_party/upstream/src
            ${CMAKE_SOURCE_DIR}/third_party/upstream/include)
        target_link_libraries(tq_reader_graph_result_publish_test PRIVATE
            tq_reader_graph tq_reader llama ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_graph_result_publish_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.graph_result_publish
                 COMMAND tq_reader_graph_result_publish_test)

        # The two ggml LAYOUT facts a multi-token emission rests on. Asserted, not argued: the two
        # multi-token attempts were both removed and both were blamed on "the concat order", which
        # was never measured. What is measured here is that a dim-1 concat of [hd,1,nh] operands is
        # TOKEN-major (so the concat chain was never the problem) while cont(permute(q,0,2,1,3)) is
        # HEAD-major (so "reshape instead of permute" reads another token's value). Distinct values
        # per (t,h,d), so either mistake moves a number. No attention, no mask, no scheduler.
        add_executable(tq_reader_tensor_layout_probe_test tests/tensor-layout-probe-test.cpp)
        target_link_libraries(tq_reader_tensor_layout_probe_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_tensor_layout_probe_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.tensor_layout_probe COMMAND tq_reader_tensor_layout_probe_test)

        # Prefill BY COMPOSITION, measured. T tokens each through the verified single-query emitter,
        # joined along dim 1. Two multi-token emitters were written and removed here -- the first
        # mis-charged its own gathers, the second produced a 0.49 difference nobody could attribute.
        # This test does not add graph code: it answers the numeric question with the emitter that
        # already exists, and asserts the two slice recipes (mask at t*nb[1]; query after one
        # whole-tensor permute to [hd,nh,T]) against the standalone per-token tensors, so a slice
        # error shows up as a named mismatch instead of as an unexplained difference.
        add_executable(tq_reader_prefill_compose_test tests/prefill-compose-test.cpp)
        target_link_libraries(tq_reader_prefill_compose_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_prefill_compose_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.prefill_compose COMMAND tq_reader_prefill_compose_test)

        # Prefill by COMPOSITION, with the whole-graph arena bound computed before the first node and
        # each token's gather table published separately. Two bespoke multi-token emitters were
        # written and removed here (an under-charged bound; an unattributable difference); this one
        # adds no arithmetic to the verified single-query emitter and checks the four things
        # composition adds: the total bound, the numbers against a per-(token, head) reference, the
        # per-token publication through the real scheduler window, and that visibility is read per
        # token. A too-small arena must be refused with zero bytes built.
        add_executable(tq_reader_prefill_emit_test tests/prefill-emit-test.cpp)
        target_link_libraries(tq_reader_prefill_emit_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_prefill_emit_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.prefill_emit COMMAND tq_reader_prefill_emit_test)

        # The prefill ADMISSION BOUND against what ggml actually builds, over the configurations the
        # test above never reaches: F16 caches, `group` of 1/2/4, and 1/2/3 spans. An under-estimated
        # bound aborts inside `ggml_new_object` half-way through a graph, and no numeric comparison can
        # see it -- so the bound is compared against the arena's real growth, the reported charge is
        # compared against the header's own `prefill_object_count`, and a context one byte short of
        # the requirement must refuse with zero growth. The 2 KiB case in the sibling test cannot
        # establish that: a bound that over-reserved by a factor of a thousand would pass it too.
        add_executable(tq_reader_prefill_bound_audit_test tests/prefill-bound-audit-test.cpp)
        target_link_libraries(tq_reader_prefill_bound_audit_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_prefill_bound_audit_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.prefill_bound_audit COMMAND tq_reader_prefill_bound_audit_test)

        # The three preflight gates that keep a STRUCTURAL refusal out of the node-building phase:
        # an allocating context, spans disagreeing on `effective_pos`, and a cell-major `v`. Each is
        # driven with a real legal fixture, and each asserts BOTH the refusal code and that the call
        # grew the arena by ZERO BYTES -- "refused before a node existed" is a claim about
        # `ggml_used_mem`, and a gate one line too late would return the same code having already
        # built `cont(permute(q))`. A well-formed control runs first so the file cannot pass by
        # making everything refuse.
        add_executable(tq_reader_prefill_gates_test tests/prefill-gates-test.cpp)
        target_link_libraries(tq_reader_prefill_gates_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_prefill_gates_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.prefill_gates COMMAND tq_reader_prefill_gates_test)

        # A runtime row-table emitter was built here and REMOVED (2026-11-25), because it could not
        # be made fail-closed and the alternative was an API that reads out of bounds in silence.
        #
        # The operator it wrapped (`ggml_flash_attn_ext_rows`) resolves a row as
        # `std::max<int32_t>(rows[i], 0)` and then walks `k->data + row*nb[1]` with NO upper bound
        # (ggml-cpu/ops.cpp). A negative row is safe -- row 0, whose mask entry must be -inf, so its
        # weight is exactly zero. A POSITIVE row at or past `k->ne[1]` is an out-of-bounds read that
        # returns garbage and asserts nothing.
        #
        # The wrapper could not check it: `kv_rows` is a `ggml_set_input` tensor whose `data` is null
        # until the scheduler places the graph (measured: `input 1 data no`), so the values do not
        # exist when the build-time checks run. A host-side check exists and is optional -- which is
        # the defect: an optional check on values that only appear after the graph is committed means
        # every caller who forgets it gets plausible numbers from beyond the cache. Shipping that is
        # worse than shipping nothing, so the emitter, its header and its test are gone.
        #
        # What replaced it is honest and already verified: `tq_reader.flash_attn_rows` drives the RAW
        # operator with a non-monotonic, interleaved-padding row table and compares against an
        # independent reference, so the operator's actual contract (including the `-1` handling) is
        # pinned by measurement. What is NOT provided -- and what a future attempt must solve first --
        # is a way to make the positive-row bound a property of the graph rather than of the caller's
        # diligence. That needs the values visible, i.e. a build-time row table (the ledger + bridge
        # path) or a flag from whoever fills the tensor.

        # A strided `q` is refused, the refusal costs ZERO nodes, and `ggml_cont` is faithful.
        #
        # The middle claim is the one this test exists for: the emitter publishes an admission bound
        # in OBJECTS, so a refusal that builds nodes first makes that bound wrong with no way for the
        # caller to notice. An earlier revision put this check after `cont(k)`, and the fix moved it
        # above every `ggml_cont`/`ggml_new_*`.
        #
        # The third claim corrects a previous note: `ggml_cont` on the offending view is faithful
        # (0 of 16 differ, read element-wise through the parent's strides). The old "8 of 16 differ"
        # came from `ggml_backend_tensor_get` on a STRIDED view, which ignores `nb`.
        add_executable(tq_reader_q_strided_refusal_test tests/q-strided-refusal-test.cpp)
        target_link_libraries(tq_reader_q_strided_refusal_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_q_strided_refusal_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.q_strided_refusal COMMAND tq_reader_q_strided_refusal_test)

        # Multi-token (prefill) attention: attempted three times, removed each time. The last attempt's
        # STRUCTURE was measured and its NUMBERS were not: per-token emissions with their own
        # index_cells, per-token publication (cells read back per token -- passed), the arena bound
        # checked against real growth, and the refusals (T == 1, a single mask row, a 2 KiB arena
        # leaving used_mem unchanged) -- all passed; `out` differed from the per-(token, head)
        # reference by 0.49 and the cause was not diagnosed. Two candidates, neither checked:
        # `ggml_concat`'s dim-1 order for a 3-D tensor, and the head stride of a [hd, T, nh] query view
        # (the naive form trips ggml's `view_size + view_offs <= ggml_nbytes` assertion, so that part
        # is at least known and needs a whole-tensor permute to [hd, nh, T]). Deleted rather than left
        # red: a prefill emitter producing numbers nobody can attribute is worse than none. The next
        # attempt starts from that structure and dumps four layers of [hd, T, nh] for ONE layer first.

        # ---------------------------------------------------------------------------------------
        # REBUILT REGISTRATIONS (2026-11-25) -- rebuilt, NOT restored verbatim.
        #
        # A script that deleted the multi-token block spliced the file with
        # `s[:start] + s[start:start]`, whose right-hand side is empty: everything after that point
        # was DROPPED. Seven test registrations went with it. They were found by set difference --
        # `tests/*.cpp` on disk MINUS names in an `add_test` -- which is a checkable criterion, so
        # the list below is certainly not short of anything that exists as a source file.
        #
        # It is nevertheless a RECONSTRUCTION, not the original text, and the distinction matters
        # enough to write down:
        #
        #   * The block order, comment wording and link lists here are authored, not recovered.
        #     `build/modules/reader/CTestTestfile.cmake` cannot help -- it is regenerated from the
        #     truncated file, so it agrees with the truncation.
        #   * The session archive was searched for the pre-truncation text and does not contain it;
        #     this tree has no commit to ask (`HEAD` does not exist), and no `CMakeLists` copy
        #     survives in `build/` or `/tmp`.
        #   * What IS established is functional completeness: 26 test sources, 26 `add_test`, no
        #     duplicate registration, no executable without a link, and its own test suite green.
        #   * What is NOT established: that the file has no other content that once followed
        #     `graph_result_publish` (a custom target, an install rule). A full-config generate
        #     succeeding makes that unlikely; it does not make it impossible.
        #
        # The link list is uniform on purpose. Each of these tests includes ggml or tq/reader headers,
        # and `tq_reader_graph` carries `tq_reader` (PUBLIC) plus the include directory, so the uniform
        # list is a superset of what each one needs. If one of these fails to build, that is a real
        # defect to look at, not a link list to trim.
        # ---------------------------------------------------------------------------------------
        add_executable(tq_reader_upstream_attention_contrast_test
            tests/upstream-attention-contrast-test.cpp)
        target_link_libraries(tq_reader_upstream_attention_contrast_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_upstream_attention_contrast_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.upstream_attention_contrast
                 COMMAND tq_reader_upstream_attention_contrast_test)

        add_executable(tq_reader_graph_emit_f16_test tests/graph-emit-f16-test.cpp)
        target_link_libraries(tq_reader_graph_emit_f16_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_graph_emit_f16_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.graph_emit_f16 COMMAND tq_reader_graph_emit_f16_test)

        add_executable(tq_reader_graph_emit_identity_test tests/graph-emit-identity-test.cpp)
        target_link_libraries(tq_reader_graph_emit_identity_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_graph_emit_identity_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.graph_emit_identity COMMAND tq_reader_graph_emit_identity_test)

        add_executable(tq_reader_physical_test tests/physical-test.cpp)
        target_link_libraries(tq_reader_physical_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_physical_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.physical COMMAND tq_reader_physical_test)

        add_executable(tq_reader_physical_safety_test tests/physical-safety-test.cpp)
        target_link_libraries(tq_reader_physical_safety_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_physical_safety_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.physical_safety COMMAND tq_reader_physical_safety_test)

        add_executable(tq_reader_arena_budget_test tests/arena-budget-test.cpp)
        target_link_libraries(tq_reader_arena_budget_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_arena_budget_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.arena_budget COMMAND tq_reader_arena_budget_test)

        add_executable(tq_reader_arena_budget_sweep_test tests/arena-budget-sweep-test.cpp)
        target_link_libraries(tq_reader_arena_budget_sweep_test PRIVATE
            tq_reader_graph tq_reader ggml ggml-cpu ggml-base)
        target_compile_options(tq_reader_arena_budget_sweep_test PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
        add_test(NAME tq_reader.arena_budget_sweep COMMAND tq_reader_arena_budget_sweep_test)
    endif()
endif()

