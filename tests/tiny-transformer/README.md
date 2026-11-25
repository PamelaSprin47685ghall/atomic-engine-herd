# tests/tiny-transformer — 一个真的会走注意力的微型模型

`make_tiny_transformer.py` 写出一份真正的 `LLM_ARCH_LLAMA` 检查点（3 层、n_embd 32、
4 头、F32 权重、263 词表含 256 个字节回退 token），约 195 KiB，固定种子，因此两次运行可比。

## 为什么需要它

reader 注意力的接缝钩的是 `llm_graph_context::build_attn`（KV 重载）。这个函数只在
**有注意力层的模型**上才会被调用。在这份模型出现之前，本机唯一的 GGUF 是
`tiny-mamba.gguf`——`architecture=mamba`，图里根本没有注意力层，于是接缝诚实地报告

```
[herd] reader: seam observed 0 build_attn call(s)
```

也就是说：当时"车道开/关的 nll 相同"这条观察，分母是 0 次调用，**不能**作为透明性证据。
这一点被明确撤回，而不是当作结论留着。

## 复现

```bash
python3 tests/tiny-transformer/make_tiny_transformer.py /tmp/tiny-transformer.gguf
# → wrote /tmp/tiny-transformer.gguf: arch=llama layers=3 n_embd=32 n_head=4 n_rot=8
#   n_ff=64 vocab=263 tensors=30

printf 'a b c\nb c d\n' > /tmp/tt-p.txt
./build/modules/corpus-compose/llama-corpus build \
    -m /tmp/tiny-transformer.gguf --input /tmp/tt-p.txt --corpus /tmp/tt-corpus
```

## 消融实测（唯一变量是环境变量）

| 运行 | 环境 | seam 注册 | 观测到的调用 | `nll_sum` |
|---|---|---|---|---|
| A 车道关 | — | 0 | 0 | 381.49004042046039 |
| B 车道开 | `HERD_READER_ATTN=1` + trace | 1（`reader.build_attn @ 0x…`） | **24** | 381.49004042046039 |

B 的 `seam(s) live` 从 4 变 5（4 个来自 `herd_seams` 段，第 5 个由注册器加入）。
逐层轨迹可见，说明钩子确实在路径上、且看到的张量形状正常：

```
[herd] reader: layer 0, scale 0.353553
[herd] reader: q_cur    ne = [8, 4, 1, 1] type 0
[herd] reader: k_cur    ne = [8, 4, 1, 1] type 0
[herd] reader: layer 1, scale 0.353553
...
[herd] reader: seam observed 24 build_attn call(s)
```

`ne = [8, 4, 1, 1]` 正是 `n_rot=8`、`n_head=4`、单 token 的 K/Q 形状，与写出模型时的参数一致。

### 24 不是 22×3

22 个 decode 只产生 24 次 `build_attn`（若每次 decode 每层都建图，应是 66 次）。
差值是**图复用**：上游在拓扑未变时复用上一次的图，不重建，因此接缝只在建图时触发。
这既是钩子行为的直接证据，也说明"每次 decode 都会调一次钩子"这个假设是错的——
排查计数异常时这一点值得先想到。

## 结论的边界

- **已证**：接缝在有注意力的模型上确实安装并执行（24 次，逐层可查），且执行对输出
  零影响（nll 17 位逐字相同）。
- **未证**：接缝的当前实现只做观察后转调，所以"透明"是它唯一的行为；
  reader **发射**（按 span gather K 行、每 span 一次 Q rope、单次 softmax）仍是零行实现。
- **未证**：八处 `build_attn` 重载只钩了 KV 一支。这是为什么请求入口被删除——
  一个能被普通注意力静默满足的请求，比没有请求更危险。上一轮的 `herd_reader_attn.hpp`
  因此被删除，接缝退化为观察点。
- **未测**：GPU 路径；并发（接缝内部已无共享状态，没有可竞态的对象）。

## 这个生成器自身的可信度

容器按 GGUF 规范手写，与仓库里已被使用的 `make_tiny_rsz_mamba.py` 同构。加载之所以可信，
是因为上游自己接受了它：`llama-corpus build` 走完了 tokenizer 与词表路径，`score` 完成了
22 次真实解码并给出有限 nll。如果容器或张量表有问题，这两步会在加载期失败，而不是
产生一个数字。

权重是随机高斯（固定种子），所以模型的输出没有语言意义——这不是缺陷：本测试断言的是
**钩子被调用**与**输出未被改变**，而不是模型好不好。
