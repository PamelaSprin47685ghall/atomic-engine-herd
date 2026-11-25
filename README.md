# atomic-engine-herd

An extension layer around [Atomic Engine](https://github.com/AtomicBot-ai/atomic-engine)
(the llama.cpp-family inference engine). It installs inline hooks into the engine's hot
paths and drives them from small, independently testable C++17 libraries.

The code is organised so that the *state* lives in synchronous stations and the
*asynchrony* lives in tickets: every station method is a plain synchronous function that
never `await`s, and all waiting happens inside coroutine-style ticket functions. That keeps
state transitions atomic under a single-threaded event loop without locks, queues, or a
central scheduler.

## Requirements

- Linux, x86-64
- CMake 3.20+, a C++17 compiler, Ninja or Make
- Network access at configure time (Dobby is fetched by CMake)

## Repository layout

| Path | Contents |
| --- | --- |
| `CMakeLists.txt` | Top-level build: `herd_seams` object library, probe, module libraries, tests |
| `third_party/upstream` | Git submodule: Atomic Engine, pinned at `fb69fff02101b048a0a165a3b502712dd2ca978f` |
| `patches/` | Inline hook layer installed into the engine at runtime |
| `modules/` | Self-contained C++17 libraries (see below) |
| `tools/` | `patch-dobby.sh`, `patch-dobby-process-map.py` |
| `tests/` | Test harness and the `tiny-transformer` fixture generator |
| `lsan.supp` | LeakSanitizer suppressions for third-party process-lifetime allocations |

### `patches/`

Hook installers and the seams that expose engine internals:

- `herd_patch.cpp` / `.hpp` — runtime hook installation (`herd::install_all()`)
- `herd_phase*`, `herd_phase_station*` — prefill/decode phase tracking
- `herd_buffer_station*` — KV buffer ownership
- `herd_reader_*`, `herd_reader_prefill_publisher*` — attention reader rules and publishers
- `herd_prefab*`, `herd_prefab_split*` — prefabricated graph selection
- `herd_nkv*`, `herd_nkv_decision*` — KV head-count decisions
- `herd_model_arch*` — model architecture detection
- `herd_graph_cache*`, `herd_graph_access*` — graph cache and graph access
- `herd_release_*` — release-mode probes and decisions

### `modules/`

| Group | Modules |
| --- | --- |
| Scheduling & admission | `scheduling`, `relay`, `relay-plan`, `relay-probe`, `relay-transport`, `relay-hip` |
| KV, reader & capacity | `rerot`, `rerot-compose`, `rerot-whiteboard`, `reader`, `kv-capacity` |
| Composition & generation | `compose`, `batch-compose`, `corpus`, `corpus-compose`, `tp5-compose`, `sampling`, `text` |
| Calibration & quantization | `calibration`, `imatrix-compose`, `logits-compose` |
| Infrastructure | `alloc-ledger`, `upstream-ext` |

Each module is built as its own library target and links only what it needs; `rerot`,
`corpus`, and `calibration` are standard-library only and do not depend on llama or ggml.

## Build

```sh
git clone --recurse-submodules <this-repo>
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j"$(nproc)"
```

All hook translation units are compiled with
`-fpatchable-function-entry=16, -fno-inline, -fno-ipa-*` and without LTO, because the hooks
must survive the optimizer.

### CMake options

| Option | Default | Effect |
| --- | --- | --- |
| `HERD_BUILD_PROBE` | `ON` | Build the `herd-probe` executable |
| `HERD_BUILD_MODULES` | `ON` | Build the `modules/` libraries and their tests |
| `HERD_BUILD_NKV` | `OFF` | Enable the KV head-count hook set |
| `HERD_BUILD_PREFAB_SPLIT` | `OFF` | Enable the prefab split hook set |
| `HERD_BUILD_BUFFER_HOOKS` | `OFF` | Enable the KV buffer hook set |

## Test

```sh
ctest --test-dir build --output-on-failure -j"$(nproc)"
```

The `tiny-transformer` fixture is generated into the build tree at test time from a small
Python generator in `tests/tiny-transformer/`.

### Sanitizer build

```sh
cmake -S . -B build-asan-full -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_FLAGS="-fsanitize=address,undefined" \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
cmake --build build-asan-full -j"$(nproc)"

ASAN_OPTIONS=detect_leaks=1 \
LSAN_OPTIONS=suppressions="$PWD/lsan.supp" \
ctest --test-dir build-asan-full --output-on-failure -j"$(nproc)"
```

`lsan.supp` suppresses only two families of process-lifetime allocations that are never
freed by design: the Dobby trampolines installed by `herd::install_all()`, and ROCm/HSA
runtime singletons. It deliberately does not suppress any project file, so a real leak in
this repository still fails the run.

Note that LeakSanitizer's suppressions must be passed through `LSAN_OPTIONS`; `ASAN_OPTIONS`
rejects `leak:` rules and aborts at startup.

## License

MIT. See [LICENSE](LICENSE). Individual modules keep their own `LICENSE` files.
