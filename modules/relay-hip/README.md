# modules/relay-hip — TP5 设备面，作为 ggml 后端插件

原分叉把五卡 relay 塞在 `ggml-cuda` 里：一份 CUDA 后端的拷贝加上 `relay.hip`，靠
`upstream-exceptions.json` 里的一条例外维持。本工程的上游是只读 submodule，那条路不存在。

它也不需要存在。

## 纯外挂：上游自己的加载器就是接线

ggml 把后端当共享库加载，全树的后端（BLAS、CUDA、Vulkan、RPC）都走同两个符号：

| 位置 | 符号 | 作用 |
|---|---|---|
| `ggml-backend-reg.cpp:229` | `ggml_backend_score` | 返回 0 表示本机不支持，**在调用 init 之前**就拒绝 |
| `ggml-backend-reg.cpp:234` | `ggml_backend_init` | 返回 `ggml_backend_reg_t` |
| `ggml-backend-reg.cpp:242` | `reg->api_version` | 必须等于 `GGML_BACKEND_API_VERSION`（本 pin 为 3） |

`ggml_backend_load(path)` 做的就这三步。所以 TP5 后端是一个独立 `.so`，上游不知道它是什么，
照样加载。**没有 hook，没有 trampoline，没有被打补丁的编译单元。**

## 已实现，且已实测

本机：AMD Radeon 760M（gfx1103），ROCm 7.2，`/dev/kfd` + `renderD128`。

```
$ ./tq-relay-hip-probe ./libggml-tp5.so
loaded backend: name=TP5 devices=0
devices registered: 0
process-wide device count: 1
roundtrip 4096 bytes: ok (stage=complete) 4096 bytes round-tripped through device memory
roundtrip 1048576 bytes: ok (stage=complete) 1048576 bytes round-tripped through device memory
```

`devices=0` 与 `process-wide device count: 1` 是同一个事实的两面：插件注册，但不注册设备，
所以进程里只有 CPU 后端，普通推理行为与未加载插件时**完全相同**。设备事实（名称、显存）
仍然真实，只是不再通过“注册一个设备”来暴露——原因见下一节。

四项，每项都有对应的真实调用：

| 能力 | 实现 | 证据 |
|---|---|---|
| 设备事实 | `hipGetDeviceCount` / `hipGetDeviceProperties` / `hipMemGetInfo` | 9216 MiB 来自 `hipMemGetInfo`；APU/独显来自 `props.integrated`，不靠产品名 |
| 设备内存 | `hipMalloc` / `hipFree` | 真实设备分配；成功返回后指针被清空（见下） |
| 主机 ↔ 设备传输 | `hipMemcpy` H2D 与 D2H | 4096 B 与 1 MiB 逐字节比对一致 |
| 经上游加载器注册 | `ggml_backend_load` | 探针**不**自己 dlopen，用的是 llama.cpp 用的那个函数 |
| **注册 0 个设备** | `hip_reg_device_count` 返回 0 | 进程级 `ggml_backend_dev_count()` 仍为 1（只有 CPU） |

### 为什么注册 0 个设备，而不是 1 个

上一版报告了 1 个设备。那条路会让普通推理崩溃，链条是从上游读出来的：

1. `llama.cpp:224-265` 默认选择会把 IGPU 类型设备放进 `igpus`；
2. `llama-model.cpp:1802-1808` 对每个入选设备调用 `make_gpu_buft_list`；
3. `llama-model.cpp:1352` **无条件**执行 `buft_list.emplace_back(dev, ggml_backend_dev_buffer_type(dev))`——没有 null 检查；
4. 该列表随后用于分配模型张量。

本插件的 `get_buffer_type` 返回 null（这是诚实的：没有算子能读 hipMalloc 出来的权重）。于是 null buft 进入分配列表，
第一个模型加载就走进空指针。本机没有独显，TP5 设备会成为唯一的 igpu，也就是说**把插件放进搜索路径就会破坏普通 CPU 推理**，
而崩溃点落在模型加载器里，没有任何线索指回这里。

注册 0 个设备把这个危害**从构造上消除**：没有设备，就没有东西进入 `gpus`/`igpus`，`ggml_backend_dev_count()` 不变。
设备面仍可用，因为 round-trip 挂在**注册表**的 proc-address 钩子上，不需要设备——探针正是这样测的
（`ggml_backend_load` 成功、名称 TP5、devices=0、4096 B 与 1 MiB 真设备往返、ptr=`nil`）。

设备在**第一个能读自己缓冲区的算子落地那天**才出现，即 `get_buffer_type()` 能返回一个算子可读的 buft 时。

### 已修的两个形状错误：悬垂句柄与泄漏

`device_roundtrip` 早先在 `hipFree` 之后仍把地址留在结果里，探针还断言它非空——**把悬垂指针当成证据**。
随后的一版只在成功路径清空它，并声称“失败但尚未释放的路径保留指针”；那句话与代码相反：H2D、D2H、
compare 三条失败路径都先 `hipFree(device)` 再返回，指针照样是悬垂的，而那些 `hipFree` 的返回值被丢弃，
于是“释放失败”就是一次没人知道的泄漏。

现在所有权只有一个：函数内的 `device_allocation` 在**每条**路径上只释放一次，释放结果并入返回的
`detail`（“freeing … failed, so it may still be held”），失败时也不重试（重试失败的空闲会带来双重释放）。
结果类型里**不再有**地址字段——不能返回的句柄无法被误用；调用方需要的 stage 与原因都在 `detail` 里。

传输验证的严格处：发送缓冲与接收缓冲是两个不同的 `std::vector`，图案是位置相关的
`(i*31+7)&0xFF`。常量填充会让"每个地址都同一个字节"的实现通过；复用同一缓冲会让
"根本没拷贝"的实现通过。两种假通过都被排除。

## 明确未实现

**算术内核不存在。** `supports_op` 对任何算子都返回 false，`init_backend` 返回 `nullptr` 并打印原因。
这不是保守，是正确性：llama.cpp 按 `supports_op` 调度，一个谎称支持的插件会把 transformer 的节点
接过去，然后在没有 kernel 的 `graph_compute` 里失败——错误落在 decode 深处，离本文件很远。

探针用一条真实算子（四元素 F32 加法）去问，并断言答案是 false。将来第一个 kernel 落地时，
这条断言会失败，加 kernel 的人必须**有意识地**改它。

**五卡 collective 未接设备面。** `generation`/`ready`/`ack` 握手与 staged all-gather 在
`modules/relay-transport`（主机内存，已测）与 `modules/relay-plan`（策略）。把它们接到这块设备面
——设备侧 kernel 发布并轮询控制块——是剩下的工作。本包提供的是它们将要站立的地面。

**`buf_type` 返回 null，不是主机内存替身。** 一个只由 `hipMalloc` 支撑的 buffer type 是另一种谎言：
llama.cpp 会把模型权重分进去，然后发现没有算子能读它们。relay staging 需要的那块内存通过
`device_roundtrip()` 与 proc 地址暴露，它的保证写在上面并被测量，而不是塞进一个它不满足的 ggml 契约里。

## 一个已被修掉的形状错误，留作记录

`score_from()` 曾被我写进匿名 namespace。它声明在公开头里，实现却在内部链接中——`.so` 里没有该符号，
链接探针时才以 `undefined reference to tq::tp5::score_from` 暴露。**公开声明的函数必须在公开链接里定义**；
否则头文件里的声明就是一句没人兑现的承诺。

## 构建

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target tq_relay_hip tq-relay-hip-probe
./build/modules/relay-hip/tq-relay-hip-probe build/modules/relay-hip/libggml-tp5.so
```

模块在 `find_package(hip)` 失败时**自行跳过并打印原因**：没有 ROCm 的机器上，一个"TP5 后端"只能是
报告零设备的桩，而报告零设备的桩不是这个模块要做的事。

`GGML_BACKEND_DL=1` 是必须的：定义它，`GGML_BACKEND_DL_IMPL` 才会展开出上游 `dlsym` 寻找的那两个符号。
不定义时宏展开为空——构建成功、加载"成功"、什么都不导出，这个失败在构建期沉默、加载期费解。
