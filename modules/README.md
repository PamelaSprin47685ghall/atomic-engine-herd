# modules — 自研层（B/C）

**这些是从原分叉 `atomic-llama-cpp-turboquant/fork/` 原样复制过来的既有代码，不是
clean-room 重写。** 逐字节未改（除下文列出的 4 处 CMake 适配）。它们属于"自有代码"，
但"自有"≠"重新设计"：这批源码今天仍带着原分叉的假设，只有搬进本仓库这一事实是新的。

`third_party/upstream` 一字未改，也未复制。

## 布局

扁平布局是刻意保留的：`tp5-compose` 用 `../alloc-ledger/include` 直接引用账本的值类型，
`tp5-compose` 的测试按 `../../tests/rsz/...` 找夹具。改成 `libs/` + `apps/` 会让这些
相对路径失效，并迫使我们在两处维护同一事实。

**但那个夹具没有跟着迁过来**：`modules/tests/` 目前只有 `isolation.py` 和
`package-consumer`，`tests/rsz/` 仍在原分叉仓库里。所以 `tp5-compose` 的 CMake
即便绕过 relay 头的问题，配置阶段也会因为夹具缺失而失败。搬夹具或改路径是待办，
不是已完成事项。

```
modules/
  sampling/ corpus/ scheduling/ text/ calibration/ relay/ rerot/ alloc-ledger/   # B：纯 C++17，零上游依赖
  compose/ corpus-compose/ batch-compose/ logits-compose/ imatrix-compose/
  rerot-compose/ tp5-compose/ relay-probe/                                        # C：只走 llama/ggml 公开接口
```

B 层 8 个包 = **9432 行**（全文 29133 行 = B 9432 + C 19619）。B 层不含任何上游符号
（`grep -E '\b(llama_|ggml_)'` 只命中 `alloc-ledger/AGENTS.md` 的一句文档文字）。

C 层是驱动，`llama_*` 只在这里出现。但**"只走公开接口"只对 `llama_*` 成立**：

| 依赖 | 上游存在？ | 状态 |
|---|---|---|
| `llama_get_embeddings_nextn_ith` / `llama_set_embeddings_nextn` / `llama_state_seq_*_ext` / `llama_model_n_layer_nextn` | 是，在 `include/llama.h` | 公开接口，OK |
| `<ggml-relay.h>` / `<ggml-relay-plan.h>`（`tp5-compose/src/runtime.cpp:2-3`） | **否**，是原分叉新增的头 | **硬断点**，见下 |

因此 `tp5-compose` 今天无法在本仓库编译：它引用的两个 relay 头不在 submodule 里，
也不在任何已迁入目录里。这正是 A 层要变成外挂补丁的地方，不是 C 层的适配问题。

## 与根项目的契约

根目录 `add_subdirectory(modules)`，并保证 `llama` 目标已存在（来自
`add_subdirectory(third_party/upstream)`）。**这句话今天不等于"整层可用"**：
`TQ_HERD_BUILD_APPS` 默认 ON，会 `add_subdirectory(tp5-compose)`，而它引用的
relay 头不存在，于是全树配置直接红。要拿到可用构建，先设
`-DTQ_HERD_BUILD_APPS=OFF`（得到 8 个库），或逐个显式打开 6 个非 TP5 的 app。
`modules/CMakeLists.txt` 负责：

- `ggml-base` → `ggml::ggml-base` 别名归一（源码树内 ggml 没有命名空间别名，
  安装包才有；不归一，`imatrix-compose` 的 `find_package(ggml)` 会失败）。
- 把 `ggml::ggml-base` 的接口包含目录导出为 `GGML_INCLUDE_DIR`，满足
  `imatrix-compose` 的按名取用。
- 选项：`TQ_HERD_BUILD_APPS`（默认 ON）、`TQ_HERD_INSTALL_EXPORTS`（默认 OFF）。
  后者关掉是因为 `tq-compose`/`tq_tp5_runtime` 链接了未安装的 `llama`，
  `install(EXPORT)` 会要求把 `llama` 塞进导出集；源码树内构建不需要安装包。
- 独立验证：`TQ_HERD_AUTO_UPSTREAM=ON` 时自己 `add_subdirectory(third_party/upstream)`。

## 已实测（临时超级构建，不在本 worktree 里）

所有验证都跑在 `/tmp/herd-check`——一个把 `third_party/upstream` 和 `modules` 拼起来
的临时 CMake 工程，不是 `atomic-engine-herd/build/`。数字如下，不要当成整层已验：

| 项 | 结果 |
|---|---|
| 8 个 B 库 | 编译链接通过（`tq_sampling` `tq_corpus` `tq_scheduling` `tq_text` `tq_calibration` `tq_relay` `tq_rerot` `tq_alloc`） |
| 该临时构建注册的测试 | 41 个（`ctest -N`），覆盖 B 层各包 + `tq_relay_probe.cli` 等 |
| 实际跑过的子集 | 15 个：`tq_sampling.*` `tq-corpus` `tq_scheduling.*` `tq_text.*` `tq_calibration.*` `tq_relay.*` `tq_alloc.*`，全通过（含 python crosscheck、events） |
| 未跑 | 其余 26 个；`tq-corpus-help` 是 C 层二进制，需要先编出上游 `llama` |

"15/15 通过"只是 41 个注册测试里的 15 个，不是全绿。C 层除 `tq-corpus-help` 外的
二进制在本次迁移中也没有一次成功链接的记录。

## 已知阻塞（不是本层缺陷）

上游 pin `fb69fff` 在本机工具链（**g++ 16.2.1**，不是 GCC 15）下自身编译失败，
原因是两个头各缺一个标准头，且互不覆盖：

| 位置 | 缺什么 |
|---|---|
| `src/llama-prefill-window-worker.h`（多处 `std::runtime_error`） | `<stdexcept>` |
| `src/llama-prefill-wide.h`（`std::system_error`） | `<system_error>`——该头只 include 了 `<stdexcept>`，够不着它 |

修法只能落在编译选项上，不改源码。根 `CMakeLists.txt` 当前对 `llama` 目标同时注入：

```cmake
target_compile_options(llama PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-include;system_error>
    $<$<COMPILE_LANGUAGE:CXX>:-include;stdexcept>)
```

强制包含是"编译参数"，submodule 的字节仍未被动过。只注入 `stdexcept` 是不够的——
`llama-prefill-wide.h` 报的错会换一个名字继续存在。升级 pin 时应先复查上游是否已自行补上，
再决定是否撤掉这两行。若某天改用 Clang/MSVC，同样的手段换成 `/FI` 或 `-include` 等价物。

## 未迁移

- A 层 340 条上游改动路径（`tp5_*`、`union_*`、`nkv_*`、prefab 等）不在本层：
  它们要改成 Dobby 补丁或结构体侧表，属另一条线。
- `overlay/`、`micro/`、`diag/`、`audit/`、`patches/ws5`、`mtp-progress-*`：
  证据与探针产物，不属功能层。
- `tests/a-seams`：绑定 A 层接缝，等补丁层落地后再迁。

## 驱动层已做的接口迁移（实测）

`corpus-compose` 与 `logits-compose` 原先依赖原分叉在 `include/llama.h` 里新增的东西。
在本仓库里分两类处理，两类都不改上游：

**1. 上游本身已有替代品 → 直接改用**

`corpus-compose` 的三处：

| 原写法 | 现在的写法 | 理由 |
|---|---|---|
| `if (!llama_memory_clear(mem, true))` | 调用后断言 `llama_memory_seq_pos_max(mem, 0) == -1` | 上游返回 `void`；那个 `bool` 代表的后置条件就是"序列为空"，故检查状态而非返回值 |
| `llama_batch_get_one(&tok, 1)` | `llama_batch_ext_init` 一次 + 每 token `clear`/`add_token` | `llama_batch_get_one` 已标记 DEPRECATED |
| `llama_decode(ctx, batch)` | `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)` | 同上 |

位置不再依赖"自动跟踪"的措辞：显式 `llama_batch_ext_set_pos(&pos)`，取值 `seq_pos_max+1`。
batch 在 `sequence` 构造时分配一次、每 token 复用，因此单 token 解码路径零分配。

**2. 上游没有对应物 → 变成我们自己的可选能力**

`--prefab-stats` 用的 `llama_prefab_*`（5 个符号 + 一个 31 字段结构）只存在于原分叉。
现在它们是 `modules/upstream-ext`（`herd::ext`）：

- 无 patch 模块注册 provider 时，`prefab_available()` 为假，所有取值函数 `throw`，
  异常文本点名缺失的接缝；
- `--prefab-stats` 在**解析参数时**就失败，早于加载模型、早于创建输出文件——
  否则会留下一个空文件，看起来像"跑过但计数为零"的测量；
- `prefab_stats_json` 也改成先格式化再开文件，双保险。

这不是降级：能力不存在就报错。等 prefab 接缝落地，patch 模块调 `herd::ext::prefab_register`
即可让同一个驱动输出真实计数，驱动侧无需改动。测试 `tq_upstream_ext.unavailable`
锁定三条契约：未注册必抛、重复注册被拒、provider 计数确实进入 JSON。

**仍留在旧 API 的地方（有意）**：`logits-compose` 内部还有两处 `llama_decode`。
它给 `--mtp-ctx` 喂合成的 embedding 平面，而新 API 走 `llama_batch_ext_add_embd`，
行的落位需要真实 Qwen4Exp MTP 模型才能证明等价，没有模型就是在猜。映射关系写在
`main.cpp` 调用点上方，有模型的人可以照着换。

### 实测结果（本 worktree 的 build/，微型 GGUF）

模型有两种来源，都是真的 GGUF，不是 mock：

- `tests/tiny-transformer/make_tiny_transformer.py`（`general.architecture = llama`，199296 B）。
  CMake 把它注册成 build 树内的共享夹具 `${CMAKE_BINARY_DIR}/fixtures/tiny-transformer.gguf`，
  经 `FIXTURES_SETUP tq_tiny_llama` 自动生成，`ctest` 单跑某一条测试时也会先建。这是默认值。
- 操作者显式指定的 Mamba 架构 GGUF。`tests/rsz/make_tiny_rsz_mamba.py` **不在磁盘上**，
  本树也没有能写 Mamba GGUF 的东西（无 torch、无 gguf 包）。因此
  `tq-tp5-mtp-state-real` 需要 `-DTP5_TEST_MODEL=<path>` 才会真正运行，否则以返回码 77 显式 SKIP。
  早期文档把"现场生成 30 KiB Mamba 模型"写成既成事实，那从未发生过。

下表的工作量证明针对前者（llama 架构）：

| 命令 | 结果 |
|---|---|
| `llama-corpus build -m tiny.gguf --input prompts` | `{"requests":2,"nodes":32,"request_tokens":36}` |
| `llama-corpus score --corpus` 连跑两次 | 两次输出逐字节相同（`nll_sum=212.26086379601378`）→ `clear()` 真的重置了状态，否则位置累积会改变结果 |
| `llama-corpus score --reference` | 同一 `nll_sum`，`decoded_tokens` 36 对 32（完整重放多解码）→ `clear()` 与前缀重放的 `save`/`restore` 语义一致 |
| `tq-logits-compose --prefill 3` | `rows=32 bytes=33696 decode_calls=32`，tape 落盘 |
| `tq-logits-compose --prefab-stats` | 明确报错，且 `/tmp/pf.json` 未被创建 |

ctest：`tq_upstream_ext.unavailable`、`tq-corpus-help`、`tq_logits_compose.writer`、
`tq_logits_compose.cli` 共 4 项通过。

### 未解决

- 没有旧分叉的构建产物可对比，所以"数值与非外挂实现逐位相同"没有证据，只有内部一致性
  （前缀重放 = 完整重放）和确定性（两次运行相同）。
- `batch-compose`、`tp5-compose` 仍未编译通过：它们用 `n_ctx_kv`、`llama_batch_is_complete`、
  `ggml-backend-impl.h` 等 A 层专有符号，属于"结构体扩容/内部头"那类，必须靠外挂状态表
  重新设计，不在驱动层适配范围内。
- `--mtp-ctx` 的 embedding 行顺序未验证（需 Qwen4Exp MTP 模型）。
