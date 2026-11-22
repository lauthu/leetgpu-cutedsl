# Vector Addition

[LeetGPU 题目](https://leetgpu.com/challenges/vector-addition) · challenge 1 · 官方定义 `1_vector_add` · 语言 `cute`

## 题目

`A`、`B`、`C` 是 GPU 上长度为 `N` 的连续一维 `float32` Tensor。计算 `C[i] = A[i] + B[i]`，结果直接写入 `C`。约束为 `1 ≤ N ≤ 100,000,000`。

- 性能测试：`N = 25,000,000`，`A`、`B` 在 [-1000, 1000) 上均匀分布
- 判定：`atol = 1e-05`，`rtol = 1e-05`

## 结果

### H100（快照 2026-11-19，排名基数 108）

| 版本 | 做法 | 提交次数 | 最好 (ms) | 中位 (ms) | 相对首版 | Percentile |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| [v1](v1_scalar.py) | 每线程 1 个元素 | 1 | 0.27299 | 0.27299 | 1.00× | 75.93 |
| [v2](v2_vectorized.py) | 每线程 4 个连续元素，`autovec_copy` 搬运 | 2 | 0.23426 | 0.24470 | 1.12× | 97.22 |

CuTe 排行榜前几名：0.23048、0.23426、0.2479 ms，v2 排第 2。

B200 上还有 1 次提交，记录见 [submissions.json](submissions.json)，不在评审范围内。

## 评审

### 结论

- **现状：** v2 的最好成绩 0.23426 ms 是 CuTe 在 H100 上的第 2 名，比第 1 名慢 1.6%，小于 v2 自身 8.9% 的波动。v2 已经在 CuTe 现有解法的前沿。
- **版本之间：** v1 和 v2 未能区分：v2 两次提交相差 8.9%，大于它与 v1 最保守的差距 6.5%，而 v1 只有一次测量。

### 代码

- `v2_vectorized.py:40-56` **测试覆盖**：官方 13 个功能测试里，只有 `N = 10,000` 会进入向量路径（9 个完整 tile），其余 12 个都小于 1024，只走尾部。下一版：tile 不超过 8192 个元素，否则向量路径只剩性能测试能走到。
- `v2_vectorized.py:84-87` **多余**：未对齐的回退分支让每次提交多编译一个 kernel。平台的输入是新建的 torch Tensor，这个分支很可能从不执行（未验证）。下一版：保留并注明代价，或删掉。
- `v2_vectorized.py:53-56` **未验证**：对齐声明是为了让 `autovec_copy` 生成 128-bit 读写，但生成的指令没有检查过，平台也不提供 CuTe 的 PTX/SASS。
- `v2_vectorized.py:37` **可读性**：`base` 只在尾部分支用到，可以移进 `else`。第 37、40、60 行的 `cute.Uint32(...)` 转换可以省略：下标在 `N ≤ 100,000,000` 内不会超出 `Int32`，与 `N` 比较时会自动按 `Uint32` 进行。
- `v1_scalar.py:3` **多余**：`import cutlass` 没有用到。`v1_scalar.py:17` 的下标是 `Int32`，在 `N ≤ 100,000,000` 内不会溢出。
- **做得好的：** `threads` 和 `values` 是 `Constexpr` 参数（`v2_vectorized.py:31-32`），换配置只需改 `solve` 里的两个数；先在运行时证明对齐（第 76-77 行），再向编译器声明。

### 性能机会

单次结果低于 0.21339 ms 才算比 v2 快。

1. **v3：每线程 8 个元素**：假设是每线程的工作量偏小。v2 用 24,415 个 block，每个线程只读写 16 字节。依据：从 v1 到 v2，block 数由 97,657 减到 24,415，中位 latency 由 0.27299 降到 0.24470 ms，方向一致，但未能区分。实验：只把 `solve` 里的 `values` 从 4 改为 8，block 数再减半；tile 为 2048 个元素，功能测试 `N = 10,000` 仍会走到向量路径。

即使 v3 追平 CuTe 第 1 名的 0.23048 ms，也只比 v2 快 1.6%，一次提交看不出来。所以除非 v3 的变化超过 8.9%，这道题在 H100 上没有值得花配额的空间，建议把配额留给下一道题。

## 版本

### v1 · Scalar（`v1_scalar.py`）

每个线程处理一个元素：下标为 `bidx * block_size + tidx`，检查越界后执行 `C[i] = A[i] + B[i]`。block 为 256 线程，grid 为 `ceil(N / 256)`。这一版用来对应 CUDA 执行模型和 CuTe Tensor 索引，没有用到 CuTe 的 layout 和分块能力。

### v2 · Vectorized（`v2_vectorized.py`）

相对 v1，每个 block 处理 1024 个元素，256 个线程各取 4 个连续的 `float32`。用 `(线程, 元素)` layout 划分 tile，`autovec_copy` 在全局内存和寄存器 fragment 之间搬运。三个基地址都满足 16 字节对齐时，向编译器声明这一对齐；不足一个 tile 的尾部逐元素处理并检查越界。目的是减少访存指令数和 block 数。

## 记录

仓库文件与平台保存的代码不是逐字节相同：平台上的 v1 没有模块 docstring（去掉它之后 Python AST 相同），仓库里的 v2 比平台多一个结尾换行。

提交记录见 [submissions.json](submissions.json)，学习中的问答见 [NOTES.md](NOTES.md)。
