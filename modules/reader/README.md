# modules/reader — RERoT 的 reader 规则（领域核）

RERoT 的定义是：一次查询只看一组手动指定的已存 key，一次 softmax 覆盖整组，
而 Q 的 rope 位置**逐 key 不同**（`virtual_pos`）。原分叉把这段逻辑写在
`src/llama-graph.cpp` 与 `src/llama-kv-cache.cpp` 里，两者都不可能在不改上游的
前提下复用或测试——所以规则本身被搬到这里，成为纯函数。

## 这里有什么

| 文件 | 内容 |
|---|---|
| `include/tq/reader/views.h` | `entry` / `view` / `span` / `cell_map` / `refusal`，以及 `validate`、`resolve`、`validate_batch` |
| `src/views.cpp` | 规则的实现 |
| `tests/views-test.cpp` | 6 组用例，逐条钉住每个拒绝理由 |

输入是 `(views, cell_map, limits)`，输出是 `(refusal, spans)`。没有 llama 类型、
没有 ggml、没有缓存、没有模型。**它是领域核，不是驱动。**

逐条对照原分叉的三处规则（`llama-kv-cache.cpp:2649 / :2709 / :2740`）：

- `prepare_attention_views` 的全部校验；
- `validate_attention_batch` 的批量一致性（两步：先比对批次，再重跑整组校验）；
- `resolve_attention_view` 的跨度折叠（`effective = query_virtual + key_pos - virtual`，
  相邻同 `effective` 合并为一个 span）。

**"对照"不是"逐字"**。原实现是 `(views, v_cells, seq_to_stream)` 直接读缓存私有字段，
这里换成显式参数 `(views, cell_map, limits)`；原实现返回 `bool`，这里返回具名的 `refusal`。
语义逐条对齐，形状不同——写清楚，免得下一位以为能按行 diff。

搬移时补上或澄清的四件事：

1. **拒绝理由具名**。原实现只返回 `false`，调用方无从知道是"key 不在缓存"还是
   "virtual 位置不递增"。现在 27 个 `refusal`，每个对应一条可执行的修正方向。
2. **批量一致性有自己的理由**。`batch_count_mismatch` / `batch_has_embd` /
   `token_without_view` / `token_multi_seq` 不借用校验理由——把"token 比 view 少一个"
   报成"同一位置重复"会让人改错地方。
3. **两阶段必须分开**。`validate()` 要求查询自身的位置**不在**缓存（它的 key 还没写），
   `resolve()` 要求它**已在**缓存（写完了）。原分叉的 `llama-attention-view.h` 原文：
   *"Logical identities are resolved after KV slot preparation. No physical index survives
   a decode call."* 早先的移植把两阶段压成一段，为调和矛盾给自身条目编了个 `-1` 标记；
   原实现从不产生该标记，而 `-1` 交给 `ggml_get_rows` 是野行索引。已改为：`resolve()`
   只做第二阶段，自身条目从写后 cell map 取真实 cell，缺失即 `key_missing`，
   负数（歧义身份，原 `attention_cells` 的约定）即 `key_cell_negative`。
4. **批量校验的覆盖缺口要写明**。原实现还查 `!batch.pos / !batch.n_seq_id / !batch.seq_id /
   !batch.token` 与 `!batch.seq_id[i]`。`std::vector` 传引用表达不了空内指针，
   这五项**由调用方负责**。这是真实的覆盖缺口，写在这里而不是装作没有。

## 一条容易被忽视的规则

`virtual_pos` 必须**严格递增**，包括查询自身那条。原因是 span 按条目顺序折叠：
顺序一旦乱掉，跨度里的旋转就不单调，图会按查询从未面对过的位置去读 key。
第一版测试的夹具违反了它，被校验器拒绝——这是校验器在工作，不是测试在挑刺。

## 为什么接线不在这个包里

消费这些 span 的是图构建：每个 span 要一次独立的 Q rope、一次对 KV 的按索引 gather、
一次掩码求和。上游没有这些概念，而它需要的三样东西都不是函数入口能提供的：

| 需要 | 上游现状 | 为什么 Dobby 改不了 |
|---|---|---|
| `llm_graph_input_attn_kv` 里能挂 per-query 的 span 张量 | 结构体没有这些字段 | 函数级补丁不能给别的 TU 拥有的结构体加字段 |
| `build_attn` 按 span 发 gather/rope 节点 | 只有单一 mask 路径 | 节点要在建图期追加，且要替换返回值 |
| KV 缓存按 `(seq, pos)` 暴露 cell | 私有 `v_cells` | 没有公开查询 |

结论是精确的：**规则可实现、可测、已实现；图构建需要上游一个结构体字段 + 一个建图入口。**
`tq-rerot-compose` 因此在 `CMakeLists.txt` 里默认不构建（`TQ_HERD_WANT_REROT`），
`src/native.cpp` 顶部有 `#error` 列出缺什么。这不是偷懒，是不把"缺子系统"伪装成"功能被关掉"。

## 已验证

```
cmake --build build --target tq_reader_views_test   # 0 error（-Wall -Wextra -Wpedantic -Werror）
./build/modules/reader/tq_reader_views_test          # "reader view contract holds"，exit 0
ctest -R tq_reader                                   # 1/1 passed
```

用例覆盖：跨度按 `effective` 折叠；跨序列 key 进入同一 softmax 并在自己的旋转处
开新 span；不同旋转产生多个 span 且**保持原序不排序**；两阶段的互斥（写前 map 通过
`validate`、写后 map 才能 `resolve`）；以及 18 条拒绝路径（空视图、双流、已移位、
查询位置已存、同序列两视图、key 不在缓存、同 cell 重复、旋转非递增、自身条目旋转不符、
旋转超过查询、缺自身条目、key 在查询之后、key 是批内另一查询、批次数量不符、
n_seq_id 长度不符、嵌入批次、token 多序列、批次与失效视图集相符）。

**变异反证**（上一轮失败、本轮补上）：把 `validate_batch` 第二步改成直接
`return refusal::none;`，编译零错误，测试立即失败两条——

```
FAIL: a batch that agrees with an invalid view set is refused, naming the view
FAIL: step 2 reports the missing key, not a batch fault
```

还原后重建通过、`reader view contract holds`、exit 0。所以这两条断言确实在测第二步，
不是与实现一起写就的橡皮章。

## 未做

- 图构建接线（见上表）：需要上游字段或入口，没有伪造。
- 批量校验五项空指针检查：由调用方负责（见上第 4 点），本核不覆盖。
- 与原分叉的**差分**运行：原树未构建，所以"语义等价"是逐条读代码得出的，不是对比跑出来的。
- 与 `tq-rerot-compose` 的端到端联调：驱动默认不构建。
- GPU 路径：本机只有 CPU 后端；reader 的收益只在长上下文多读者场景才显著。

## 注意力算子：规则之外的算术（`src/attention.cpp`）

规则层（`views.*`）回答"哪些 key 可见、解析到哪一格"。本文件回答"那之后的值是多少"——
即原分叉 `llm_graph_context::build_attn_reader`（`src/llama-graph.cpp:3613-3712`）真正对数字做的事：

```
对每个 query 行 i：
  对每个 span：scores_span = K[span.cells] · rope(Q_i, span.effective_pos)
  all_scores = concat(所有 span 的 scores)          // 每个可见 key 恰好一次
  weights    = softmax(all_scores × kq_scale)       // 一次 softmax，跨所有 span
  out_i      = Σ_j weights_j · V[cell_j]
```

三处与普通注意力的差别，各对应一行实现：**key 集是挑出来的**（`span.cells`）、
**Q 按 span 旋转**（`effective_pos`）、**softmax 覆盖全部 span**（别的序列的 key 与自己的历史
在同一次归一化里竞争）。

rotable 由调用方以 `rope_fn` 传入，这不是偷懒：分叉的旋转是 `ggml_rope_ext` 带十几个参数
（yarn 因子、attn scale），在这里重写一份就是在复刻上游约定，而"我用自己的 rope 验自己的 rope"
什么都证明不了。核心拥有的是**聚集、拼接、归一化、值归约**，这几件事可以脱离任何 rope 定式被检验。

### 数值参照测试（`tests/attention-test.cpp`，6 组，实测绿）

对着**手写的第二份实现**比，不是对着自己比：

| 组 | 钉住什么 | 变异后是否失败 |
|---|---|---|
| 恒等视图 == 普通因果注意力 | reader 算术与每个 transformer 已在做的算术同源 | 是（旋转一次 → delta 6.5e-3） |
| 打散视图 == 手挑 key 的 softmax | 聚集用的是 `span.cells`，不是 query 自身位置或线性下标 | 是（忽略下标 → delta 0.8） |
| 每 span 分别旋转 ≠ 旋转一次 | 分叉最容易犯的那个 bug（只在 query 位置旋转一次） | 是（→ 三个断言全挂） |
| softmax 是全局而非逐 span | 逐 span 归一化把分数丢掉，答案恒为各 V 行的均值 | 是（→ 恒等视图与打散视图同时失败） |
| 失败即失败（空视图 / 越界 cell / 空 query / 缺 rope / 溢出） | 返回零向量会被读成"查询没看见任何东西" | — |
| 空 span 不参与 | 空 span 不应开一次旋转 | — |

容差是 1e-6 绝对值（输出量级 ~1）：两份实现只在浮点结合顺序上不同（`attend` 的 dot 用 double
累加、scale 在 float 里乘；参照实现全程 double），观测差 ~1e-9，即该量级下一个 float ulp 以下。
**这不是把断言放宽到看不出错**：错误公式的偏差是 1e-2 量级，高出四个数量级。

变异是**实测**做的（在 `/tmp` 的副本上编译，仓库文件未改）：三处变异各自 exit=1，失败断言如上表。

### 数值边界（实测确认，不再是推断）

| 情形 | 现在的行为 |
|---|---|
| GQA（`n_head_kv < n_head`） | 支持：query head `h` 读 kv head `h / (n_head / n_head_kv)`，与 `ggml_mul_mat` 的分组一致 |
| `n_head_kv` 不整除 `n_head` | 拒绝（分组无法定义），不四舍五入 |
| 同一 cell 在视图里出现两次 | 拒绝。此前会被**重复加权**：两个 cell 当成三个 key 时得 1.333，正确值 1.5——数值有限、形态正常 |
| V 里有 NaN / ∞ | 拒绝。此前 `ok == true` 并把 NaN 交出去，而分数路径却检查了有限性——不对称即缺陷 |
| 分数非有限（溢出、q/K 含 NaN） | 逐分数检查。`std::max` 会**丢弃** NaN 并保持累加器，只查最大值的写法漏掉这一种 |
| q 与 V 的 `head_dim` 不同（MLA） | 不支持，且写在头文件里：这是具名边界，不是遗漏——为它加第二条 stride 而没有任何模型能验证，比说明局限更糟 |

"每个可见 key 恰好一次"原是注释里的断言，现在由一次排序检查兜住（`n log n` 对比同一次调用的 `n · head_dim · n_head` 次乘加，8k key × 32 头 × 128 维是 100k 次比较对 32M 次浮点运算）。

## 图发射器（`src/graph_emit.cpp`，已存在并实测）

本核是**算术**；把它发成一串 ggml 节点的是 `graph_emit.{h,cpp}`，有独立测试：

| 用例（`tests/graph-emit-cache-layout-test.cpp`） | 钉住什么 | 实测 |
|---|---|---|
| 默认 V 缓存（`!v_trans`，ne = `(head_dim, n_head_kv, n_kv, ns)`） | 按上游 `get_v` 的真实 stride 读同一批字节 | 对核心核 `max\|diff\| = 4.5e-08` |
| 转置 V 缓存（`v_trans`，ne = `(n_kv, n_head_kv, head_dim, ns)`） | 同一批逻辑值、另一种排列；`n_cell == head_dim` 时**形状完全相同**，只能靠 `nb[1] > nb[2]` 分辨 | `4.5e-08` |
| 多 span（3 + 2 个 key） | 一次 softmax 覆盖全部 span，值归约同序 | 同上（跨 span 拼接生效） |
| 2 个流（`ne[3] == 2`） | 具名拒绝 `multi_stream_view`，绝不读第 0 个流冒充 | 拒绝 |
| Q8_0 32 宽（**合法**张量，非畸形夹具） | 具名拒绝 `quantized_cache_unsupported`：转置量化缓存需要 dequant 遍，本模块没有 | 拒绝 |
| F16 的 q | 具名拒绝：`mul_mat` 要求两侧同类型，否则死在 ggml 的断言里 | 拒绝 |

两处**由实测纠正的错误**（都在同一文件里，留作记录）：

1. `ggml_get_rows(a,b)` 沿 **ne1** 选行（`a->data + i*nb[1]`，结果 `[a->ne0, b->ne0]`），
   而第一版把每个 head 摆成 `[n_cell, head_dim]`（cell 在 ne0）。"key 轴该在 ne0" 是**乘法**的要求，
   不是**聚集**的要求：聚集取到的是 head 维，之后形状自洽、数值全错。
2. 拼接 head 时错用 dim 0（key 轴），得到 `[S*n_head_kv, head_dim]`，`mul_mat` 断言失败。
   **单 kv head 时两种读法结果相同**，所以只有双 head 夹具能分辨——测试用 2 个。

变异验证（`/tmp` 副本，仓库源未改）：head 拼到 dim 0 → `GGML_ASSERT(ggml_can_mul_mat)`；
转置分支不转置 → `max|diff| 4.85e-01`（对比 4.5e-08）。

## 接线到上游图：剩下的确切工作

本核与发射器都**没有接进 `build_attn`**：一条请求路径都没有，所以不存在静默回退到普通注意力的入口。
原分叉那段代码的每一步——`ggml_permute`、`ggml_get_rows`、
`ggml_reshape_3d`、`ggml_soft_max_ext`、`ggml_mul_mat`——都是 **ggml 公开 API**。也就是说，
"从外部造出同一串节点"在接口层面是可行的，与"改上游结构体"无关。缺的是两样东西，都由读码确定：

1. **图上下文**：节点要挂在 `llm_build_context::ctx0`（`ggml_context*`）上并 `ggml_build_forward_expand(gf, …)`。
   这两个成员在 `src/llama-graph.h` 里是 public 的（此前已用编译期探针确认 `llm_graph_result`
   可构造、字段可写、`llama_model::build_graph` 可取地址），所以一个只读包含即可拿到；
2. **接缝选择**：`llm_build_context::build_attn` 有 **八处重载**（`llama-graph.cpp:3442/3574/3692/3800/3984/4005/4087/4589`），
   reader 只对应 KV 那一支。请求开着而模型走别的重载，会静默执行普通注意力——
   这正是 `patches/herd_reader_attn.cpp` 里把"请求"整个删掉、只留观察点的原因。
   要重新引入请求，必须先覆盖全部八处或用 `HERD` 侧的强断言证明当前模型只走 KV 支。

因此接线顺序是：**先做恒等接管**（hook `llama_model::build_graph`，原样返回 `g_orig` 的
`ggml_cgraph*`，验收 = logits 逐位相同且节点数相同），**再替换注意力内容**（用本核的 spans 发
gather/rope/softmax 节点）。跳过第一步就等于在没有基线的图上改数学。

> 早期版本的这一行写成"hook build_attn，用 `model.build_graph` 造同一张图并返回自己的
> `llm_graph_result`"，那是两处笔误：`build_attn` 的返回类型是 `ggml_tensor*`，永远不可能返回
> `llm_graph_result`（后者是 `llm_graph_context` 派生类的函数内临时物，生命周期不出 `build_graph`）；
> 而恒等接管的落点因此只能是 `llama_model::build_graph(const llm_graph_params&) const -> ggml_cgraph*`
> ——它也是唯一能表达"全层替换或全不换"的位置，因为只有它一次返回整张图。
> 该接缝已实现为 `HERD_READER_GRAPH_K1=1`（`patches/herd_reader_attn.cpp`，符号
> `_ZNK11llama_model11build_graphERK16llm_graph_params`）。

### K2：真的替换了（`HERD_READER_GRAPH_K2=1`）

K1 只观察整图；K2 在**同一张图里**用本模块的 `emit_physical_attention` 换掉 attention，并把
索引发布挂到上游唯一的窗口（`llm_graph_result::set_inputs`）。两个开关互斥，同开即 abort ——
一个只声称观察的接缝不该报出替换次数。

接缝位置是 `llm_build_context::build_attn_mha`，不是 `build_attn`。原因不是偏好：
`build_attn` 体内还有 `cpy_k`/`cpy_v`（`llama-graph.cpp:3707`）与四项后处理，在它上面替换会让
cache 停止前进——首 token 正确、其后全错，且无处报错。`build_attn_mha` 在 KV 写回**之后**。

实测推翻了一条早先的读码结论，记在这里因为它是 K2 能成立的全部理由：`build_attn_mha` 收到的
`k`/`v` **已经是 cache 全量视图**（`ne[2] == n_ctx == 2048`），不是本 batch 的新 key。所以不需要
`mctx`、不需要上移到整层接缝、也不需要 `seq_cell_projection` 求 cell 列表——一个命名全部 cell 的
单 span 恒等视图就是完整视图。

`view_covers_finite_mask` 的校验点只有一个：`k2_input::set_input`。构造期 `kq_mask->data` 为
`nullptr`（`ggml_set_input` 张量），值是 `set_input_attn_mask` 在放置之后才填的。F16 由调用方用
`ggml_fp16_to_fp32` 手扩成 F32——该函数故意没有重载，因为扩宽需要一个调用方拥有的目标缓冲。

**边界（实测）**：

| 条件 | 行为 |
|---|---|
| `--parallel 1`，decode 单 token | 每层 **REPLACED**，输出与基线逐字相同 |
| `--parallel 2` 或 prefill（`q->ne[2] > 1`） | 该图**整体回退**上游（打印 `falls back (single_token 0)`） |
| `K1` 与 `K2` 同开 | `abort`，互斥 |

`q->ne[2]` 就是 n_tokens 轴，所以「decode-only」不是形容词而是**进程参数**：验收必须固定
`--parallel 1`。混合状态（decode 用 emit、prefill 用上游）在同一进程内共存且输出仍逐字相同，
但这一致性是靠输出对照证明的，不是被结构强制的——`K2c`（prefill 发射）未实现，这是边界不是缺陷。

验收标准是「与 K2 关闭时 logits 逐位相同」，**不是** RERoT 收益：`emit` 按名字拒绝
`mixed_effective_pos`，且 `q` 到达时已旋转，所以逐 key 不同 rope 需要上移到 rope 之前（记为 K3）。

### 本核尚未覆盖

- 一次 `attend()` 只处理一个 query 行。多行批处理在分叉里由 `q_raw->ne[2]` 循环承担，属于调用方的循环。
- `sinks`（`ggml_soft_max_add_sinks`）、`attn_soft_cap`、`self_k_rot`（Hadamard）未在本核内。
  它们在图上是独立的节点，接在 softmax 前后；本核被调用方组合，不需要内建。
- 与真实 K/V 张量的端到端对照（需要图接线落地）。

