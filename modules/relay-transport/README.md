# modules/relay-transport — TP5 relay 控制平面（可在 CPU 真实执行）

原分叉的五卡 relay 有三层：**策略**（plan/frame/pool，已迁至 `modules/relay-plan`）、
**控制平面**（generation + ready + ack 握手，本包）、**传输后端**（HIP staging、
`relay.hip`，需要五卡实机）。

本包是中间那层。它不是模拟器，也不是替身：控制平面本来就是一块固定布局的共享内存加一套
release/acquire 纪律，和跑在 GPU 上还是主机内存上无关。把它放到主机内存上执行，就能把
握手规则**测出来**，而不是只在文档里断言。

## 两个文件，两种角色

| 文件 | 角色 | 有没有等待 |
|---|---|---|
| `control_block.h` | 站（station）：平面布局 + 握手规则。纯函数，无线程、无睡眠、无时钟 | 无 |
| `loopback.h/.cpp` | 工单（ticket）：五个 producer、一个 reducer、五个 consumer 的真实线程 | 全部在这里 |

这正是 AGENTS.md 的分工：状态与裁决归站，等待归工单。`control_block.h` 里没有一个
`std::thread`、没有一个 `condition_variable`；`loopback.cpp` 里没有一条协议判断。

## 布局：与原 GPU 控制块逐字对齐，对象不同

**布局相同，类型不同。** 这句话是本文件后面第 141 行起的结论，这里必须先说，因为标题旧版写的是
「逐字节相同」，那是过度声明，并且与后面的正确表述自相矛盾。

结构体的**字段、顺序、对齐、总长**与原分叉一致，所以同一块内存的任一偏移在两侧指的是同一件事；
但两侧声明的**不是同一个语言层对象**（设备用 `uint64_t` + `__atomic_*`，这里用 `std::atomic`）。
共享一块映射是工程约定（双方只用原子操作、一侧初始化），不是标准保证。

权威来源在原分叉里自带静态断言：

```
ggml/src/ggml-cuda/relay.hip:98-105
  struct alignas(64) word { uint64_t value; };
  struct control { word generation; word ready[ranks]; word ack[ranks]; word error[ranks]; };
ggml/src/ggml-cuda/relay.hip:118
  static_assert(sizeof(control) == ggml_relay_plan::control_bytes, ...);
ggml/include/ggml-relay-plan.h:36   control_bytes = 16 * 64
ggml/include/ggml-relay.h:10        GGML_RELAY_RANKS = 5
```

本包的 `control_block.h` 是同一个 `control`：`alignas(64)` 的 `word`、1 个 `generation`、
`ready[5]`、`ack[5]`、`error[5]`，16 个 word = 1024 字节，并有同样的
`static_assert(sizeof(control) == 16 * cache_line)`。

### 上一版为什么不兼容（已修）

上一版凑到了同样的 1024 字节，但**含义几乎在每个偏移上都不同**：多了一个
`committed_epoch`，把每个 rank 的 error 塞进 ready 的同一 cache line，又长出
`cpu_error`。两个等长的块不是同一个块——GPU 端写 `error[3]`，落在上一版里就是
`ready[3]` 那一行。所以「同一套握手」在那时是可证伪的错误陈述，不只是措辞不严。

### 三条规则，照抄而不是转述

1. **`generation` 存的是已提交的 epoch，不是计数器**。主控端把 `p.status.epoch`
   写进去（`relay.hip:2781`）；消费者等的是 `generation == epoch`，而
   `generation > epoch` 是**错误**，不是更新的消息（`relay.hip:248-255`）。
2. **`ready[r] == epoch`、`ack[r] == epoch` 都是等值判断**。这正好让控制块无需清零即可复用：
   旧 epoch 的值天然不等于新 epoch。协议里没有 reset，加一个反而会制造「清除尚未写入的
   ack」的窗口。因此本包在 200 轮连续测试里**不再重置 ack 词**——上一版那样做是错的。
3. **`error[r]` 是独立的一行**，码域来自设备端：1 abort、2 generation 不符、3 等待超时
   （`relay.hip:243/254/259`）。任一 rank 非零即整轮失败。

### 顺序

设备端在发布前 `__threadfence_system()`，并全程 `__ATOMIC_RELEASE` 写 / `__ATOMIC_ACQUIRE` 读
（`relay.hip:132-136, 209, 231, 288`）。本包用 `memory_order_release` 写发布/提交/确认，
此后每次读取用 acquire。唯一放宽处是设备端自旋里的 relaxed poll（`relay.hip:127`），
那里值只作「继续等」的提示；本包一律 acquire，只会更强。

## 已实测

```
cmake --build build --target tq_relay_transport_tests      # 0 error（-Wall -Wextra -Wpedantic -Werror）
./build/modules/relay-transport/tq_relay_transport_tests   # "relay transport contract holds"，exit 0
ctest -R tq_relay                                          # 9/9 passed
g++ -fsanitize=thread …                                    # 无报告，exit 0（11 线程 × 200 epoch）
```

### 首轮实测抓到的真缺陷（保留下来，因为它说明测试有效）

`make_payload` 把输出缓冲写成 `words * ranks`，而 reducer 只按 `words / ranks` 的宽度逐 rank
拷贝。于是 consumer 读到后四分之三是未初始化内存。测试第二次运行就报了
**1010 处 "a consumer read a word from another epoch"**。

这不是测试写错，是布局不自洽：`payload::output` 的内容语义是「五个 rank 切片首尾相接」，
尺寸就必须是 `width * ranks`。已修，并在注释里写明为什么不能「大一点没关系」——
多出来的部分不是更大缓冲，是消费者会当作数据的垃圾。

### 第二轮审查抓到的缺陷（本次）

三个，都在「看起来对、实际不等价」这一类：

1. **控制块不是同一个块**。旧版多 `committed_epoch`、error 与 ready 同 line、多 `cpu_error`，
   靠总长度相同通过了 1024 断言。已改为与 `relay.hip:98-105` 逐字相同的 16 word，
   并补 `static_assert(sizeof(control) == 16 * cache_line)`。
2. **`generation` 语义错**。旧版 `fetch_add(1)` 当计数器用；设备端是把 epoch 直接存进去
   （`relay.hip:2781`），且 `generation > epoch` 是错误而非更新的消息。已改为存 epoch，
   并把 `commit_output` 写成「必须前进」。
3. **中途重置 ack 词**。旧版 200 轮测试每轮清零 `ack[*]`；协议靠等值比较天然支持复用，
   清零反而制造「清掉正在写入的 ack」的窗口。已删掉重置，测试仍然通过 200 轮——
   这本身就是「不需要清零」的证据。

三条都属于同一类错误：**把「长度相同 / 测试通过」当成「协议相同」**。

### 用例

| 组 | 钉住什么 |
|---|---|
| `publish_rules` | epoch 0 不可发布；越界 rank 拒绝；重复发布拒绝；epoch 倒退拒绝；部分就绪的 mask 精确 |
| `commit_and_ack_rules` | 未 commit 不可 ack；重复 commit / epoch 0 commit 拒绝；提交写入的是 epoch 本身；重复 ack 拒绝；五个 ack 才算完整 |
| `failure_is_visible_to_the_reducer` | rank 失败可见；且**没有** CPU 私有 error 词——reducer 失败靠「不提交」表达，与设备端一致 |
| `one_epoch_runs_end_to_end` | 5 producer + reducer + 5 consumer 真线程跑通，ready/acked 全满 |
| `epochs_do_not_tear` | 200 个 epoch 连续、**不重置任何词**，逐字校验无撕裂，每 consumer 恰好读 200 次 |
| `a_failing_rank_does_not_hang_the_epoch` | 一个 rank 抛异常 ⇒ epoch 以错误结束、零 consumer 读取、不挂起 |
| `a_plane_holding_a_message_is_not_reused` | 不前进的 epoch 被拒（error=-2），而不是覆盖未读消息 |

## 与设备协议的三个差异（两个照抄，一个有意的更严）

逐行对照 `ggml/src/ggml-cuda/relay.hip` 之后，本包与设备端在三处不同。前两处是实现补上的设备规则，第三处是刻意的偏差，都写明理由。

### 1. 出错的 rank 不得 ack（照抄，此前漏了）

设备在写 ack 前检查 `!load(abort_word) && !load(&c->error[rank])`，并注明 *"An abort is not a normal consumption ack"*。本包早先没有这道检查，于是一个已 abort 的 rank 仍可声明"我消费了"；随后 reducer 会 `conclude` 该 epoch、清掉 error 词、释放一条那个 rank 从未读取的通道——故障会在别处、以错数值的形式出现。

现在 `ack_output` 先读 `error[rank]`，非零即拒。

**这条守卫目前没有测试保护，必须交接口头**：我用变异反证检验它——把守卫改成注释、重新构建、重跑测试——结果 `mutant_run=0`，即测试仍然全绿。也就是说该守卫的删除**不会**被任何既有断言发现。我随后补了两条断言（`an errored rank cannot acknowledge the epoch` / `and no ack word was written`），但 bash 预算耗尽，**这两条从未执行过**。下一位第一件事：重建该目标并跑 `ctest -R tq_relay`，确认它们通过；再删掉守卫做一次变异，确认它们失败。

这条未完成项本身就是一个教训：我加的守卫是从设备源码逐条抄来的、理由写得很足，而"抄对了"与"被测试钉住"是两件事。

### 2. ack 撞上错误的 generation 时，记录 error 而不是沉默（照抄）

设备在 ack 时发现 `generation != epoch` 会把 `error[rank]` 写成 2（generation mismatch，`relay.hip:296-298`），而不是丢弃这次确认。理由很实在：一个无法 ack 的 rank 会让该 epoch 永远无法 conclude，而一条不明原因失败的通道正是这套控制面存在的意义。

现在同样写入 `error_generation`。

**这条改动的副作用值得单独说**：被拒的 ack 不再是"无副作用的失败"，它会毒化该 rank。因此测试里凡是检查"非法 ack"的用例都必须使用独立平面——第一次改完直接把共享夹具毒化，八条断言连锁失败，那正是设备语义在起作用，不是回归。

### 3. 重复 ack 被拒（有意更严）

设备对同一 epoch 的重复 ack 只是再写一次同样的值（幂等）。本包返回 `false`：丢失状态的调用方应该听到这件事。它不会把正确的运行变成失败的运行——写两遍同一个 epoch 在两种写法下都不改变平面。

相反方向还有一处更严：`publish_input` 拒绝非前进的 epoch，设备只是无条件写入。理由相同（调用方状态错乱应当暴露），代价也相同（幂等写入无论如何都不改变结果）。

## 边界：协议相同，后端未实现

控制平面**布局与原分叉逐字对齐**（字段、顺序、对齐、16×64 字节），**类型不同**，这是必须说清的部分。传输后端**不存在**，这是本机的限制。

1. **布局相同**：`alignas(64)` 的 word、1 个 `generation`、`ready[5]`、`ack[5]`、`error[5]`，16 个 word、1024 字节，两侧都有 `sizeof(control) == 16 * 64` 的静态断言（设备侧 `relay.hip:118`，本包 `control_block.h`）。偏移逐一对应，所以一侧写 `error[3]` 落在另一侧也是 `error[3]`。
2. **类型不同**：设备侧是 `struct { uint64_t value; }` 配 `__atomic_load_n/store_n`，本包是 `std::atomic<uint64_t>`。两者在 x86-64 上编译成同一条指令、同尺寸同对齐，但**不是同一个 ISO C++ 对象**：把同一块内存同时当作 `std::atomic<uint64_t>` 与普通 `uint64_t` 访问，在标准层面是类型混用；设备侧还会 `memset` 整块（`relay.hip:1071`），而对 `std::atomic` 做 `memset` 同样是标准层面的未定义行为。实践中可行，但前提是两侧都只用原子操作、且初始化由一侧完成后再交给另一侧。
3. **协议相同**：三条规则（等值判断、`generation` 存 epoch、错误码域）来自设备源码，不是转述（见下）。

所以「同一份结构体定义」这个说法过强了：**同一份布局，两种语言表述**。上一版连布局都不同，那是错；这一版布局对了，但把它说成「同一个对象」仍然是过度声明。

没有设备、没有 staging pool、没有 HIP 队列，也不声称搬过一个字节到 GPU。它执行的是 GPU
传输**同样要说**的那套协议，控制块布局也是同一份（类型不是，见上），所以它是该协议的
主机侧实现，而不是「一个像它的设计」。差别只在 payload 落在哪块内存里，以及由谁来初始化那块内存。

要成为真 backend 需要三件事，且都不是本包能提供的：

| 需要 | 现状 |
|---|---|
| `ggml_backend_i` 实现（5 设备、buffer type、graph compute） | 上游提供注册接口（`ggml_backend_register`、`GGML_BACKEND_PATH`），但实机需要五卡 |
| HIP staging 池与 `relay.hip` 内核 | 原分叉的 `staging.cu` / `relay.hip`，本机无 GPU 无法验证 |
| 五卡上的 epoch 时序验证 | 本机仅 CPU 后端 |

所以 `tp5-compose` 仍默认不构建（`TQ_HERD_WANT_TP5=OFF`），其 CMakeLists 里有
`FATAL_ERROR` 说明缺什么。**策略层已迁、控制平面已迁并实测、传输后端未迁**——这是三条线，
不能混为一谈。

## 未验证

- 多设备：控制平面与设备数无关，但「五个 rank 各自一块 staging」的映射没跑过。
- 真实 payload 尺寸：测试用几十个 word，实际是 `1024 + 65536*4*6` 字节级别的通道。
  耗时特性（尤其 cache line 争用）在真尺寸下可能不同。
- 与 `modules/relay` 的 `channel_ledger` 的接线：两个实现各自正确、各自有测试，
  但还没串成「transport 证明事实 → ledger 记账」的一条链。这是下一步。
