# Matrix Multiplication · 学习笔记

学习中的问答，针对 [v4_aligned.py](v4_aligned.py)。评测结果、评审和版本说明见 [README.md](README.md)。

## Q&A

### 为什么把 tile 从 64 × 64 加大到 128 × 128？

这个改动出现在 v3（照 CuTe 的 `ampere/sgemm.py` 示例，tile 是 128 × 128 × 8），v4 沿用。原因是 tile 越大，每搬运一个值能做的乘加越多：一个 CTA 每一步 k 读入 `BM + BN` 个值，用它们做 `BM × BN` 次乘加。两边各加倍，每个值对应的乘加次数也加倍。

对性能测试的规模 8192 × 6144 × 4096，下表是按公式算的，不是测出来的：

| | 64 × 64（v1、v2） | 128 × 128（v3、v4） |
| --- | ---: | ---: |
| 每个线程的输出 | 4 × 4 = 16 | 8 × 8 = 64 |
| 每一步 k，每个线程读 smem | 8 个值，做 16 次乘加 | 16 个值，做 64 次乘加 |
| A 的每个元素被读入的次数 | 64 | 32 |
| B 的每个元素被读入的次数 | 128 | 64 |
| 从全局内存读入的总量 | 25.8 GB | 12.9 GB |
| CTA 数 | 8192 | 2048 |

- **为什么不再加大：** 256 个线程算 256 × 256 的 tile 时，每个线程要 16 × 16 = 256 个累加器，超过一个线程能用的 255 个寄存器（CUDA 的上限，这里没有另行核对）。128 × 128 时是 64 个累加器加 16 个操作数。
- **k-tile 同时从 16 减到 8**，所以一个 k-tile 仍是每个矩阵 1024 个值，每个线程仍然各拷贝 A、B 的 4 个值。
- **这个收益没有单独测量过。** v3 同时改了 tile、流水线和布局，结果比 v2 慢；v4 只去掉了边界处理。没有哪两个版本只差 tile 的大小。

### `mB` 为什么是 B 的转置视图？为什么计算结果不变？

`solve` 没有移动或拷贝 B，只是给同一块内存套上一个两个坐标互换的 layout（[第 216-220 行](v4_aligned.py#L216-L220)）：

```python
mA = cute.make_tensor(A.iterator, cute.make_layout((M, N), stride=(N, 1)))   # (m, k)
mB = cute.make_tensor(B.iterator, cute.make_layout((K, N), stride=(1, K)))   # (n, k)
mC = cute.make_tensor(C.iterator, cute.make_layout((M, K), stride=(K, 1)))   # (m, n)
```

题目和 CuTe 用的是两套名字：

| | 题目 | CuTe |
| --- | --- | --- |
| C 的行 | `M` | m |
| C 的列 | `K` | n |
| 被求和的维度 | `N` | k |
| A | `[M, N]` | (m, k) |
| B | `[N, K]` | (n, k) |
| C | `[M, K]` | (m, n) |

**步长为什么是 `(1, K)`。** 题目按行存放 `B[N, K]`，元素 `B[i][j]` 的偏移是 `i·K + j`。用 CuTe 的名字，`i` 是 k，`j` 是 n。要让 `mB(n, k)` 指向同一个元素，偏移就是 `n·1 + k·K`，也就是 shape `(K, N)`、stride `(1, K)`。偏移的算式没有变，变的只是两个坐标的先后：`mB(n, k)` 就是 `B[k][n]`。

每个 mode 的大小和步长都没变，只是换了排列次序，所以按行存放的 `(N, K) : (K, 1)` 变成 `(K, N) : (1, K)`，shape 和 stride 一起交换：

| Mode | 大小 | 步长 | 在 `B[i][j]` 里 | 在 `mB(n, k)` 里 |
| --- | ---: | ---: | --- | --- |
| k（题目的 N） | `N` | `K` | 第一个坐标 | 第二个坐标 |
| n（题目的 K） | `K` | `1` | 第二个坐标 | 第一个坐标 |

`N = 2`、`K = 3` 时，`B[1][2]` 的偏移是 1·3 + 2 = 5，`mB(2, 1)` 的偏移是 2·1 + 1·3 = 5。

**结果为什么不变。** CuTe 的乘法不把 B 当作"行、列"来读，它的约定是两个操作数的第二个坐标都是 k：

```text
C(m, n) = Σₖ A(m, k) · B(n, k)
```

把交给它的两个张量当作普通矩阵看，这个式子算的是 A · Bᵀ。我们交给它的是题目里 B 的转置视图，两次转置抵消，得到 A · B。把 `mB(n, k) = B[k][n]` 代进去就是题目自己的定义：

```text
C(m, n) = Σₖ mA(m, k) · mB(n, k) = Σₖ A[m][k] · B[k][n]
```

用 2 × 2 的 `C[0][1]` 验算：定义是 `A[0][0]·B[0][1] + A[0][1]·B[1][1]`；内核算的是 `mA(0,0)·mB(1,0) + mA(0,1)·mB(1,1)`，而 `mB(1,0)` 是 `B[0][1]`，`mB(1,1)` 是 `B[1][1]`。

如果不转置，直接把 B 当作 `(N, K) : (K, 1)` 传进去，内核会把题目的 N 当成 n、K 当成 k，算出别的东西，或者因为形状对不上而失败。

**好处和代价。**

- A 和 B 在内核里形式相同：k 都是第二个 mode，所以一个 `CTA_TILER = (BM, BN, BK)` 能同时切两者，`partition_A`、`partition_B` 拿到的正是它们要的 (m, k) 和 (n, k)。
- 代价是两个矩阵连续的方向不同：A 沿 k 连续，B 沿 n 连续。所以两个拷贝的线程布局不一样，`thr_layout_A = (32, 8) : (8, 1)`，`thr_layout_B = (128, 2) : (1, 128)`，线程号都沿各自连续的方向递增。
- 两套名字同时出现的地方只有 `solve` 里的尺寸判断：`(M % BM == 0) & (K % BN == 0) & (N % BK == 0)`，题目的 `K` 对 `BN`，题目的 `N` 对 `BK`。

### `tiled_mma` 的定义每一部分是什么意思？

```python
atoms_layout = cute.make_layout((THREADS // 16, 16, 1), stride=(16, 1, 0))
permutation_m = cute.make_layout((atoms_layout.shape[0], 4), stride=(4, 1))
permutation_n = cute.make_layout((atoms_layout.shape[1], 4), stride=(4, 1))
tiled_mma = cute.make_tiled_mma(
    cute.nvgpu.MmaUniversalOp(cutlass.Float32),
    atoms_layout,
    permutation_mnk=(permutation_m, permutation_n, None),
)
```

`tiled_mma`（[第 241-248 行](v4_aligned.py#L241-L248)）是一张编译期的表：哪个线程算 C 的 tile 里的哪些元素，因而读 A、B 的哪些值。它由三部分组成。

**1. 原子：`MmaUniversalOp(Float32)`。** 一个线程做一次 `c += a * b`，在 (m, n, k) 上的形状是 (1, 1, 1)。作为对比，v5 的 tensor core 原子是 32 个线程做一个 16 × 8 × 16 的乘积。

**2. 原子在线程上的排布：`atoms_layout = (16, 16, 1) : (16, 1, 0)`。** 把 256 个原子摆在 (m, n) 的 16 × 16 网格上，并给出每个原子由哪个线程执行：线程号 = 16·m + n。`make_tiled_mma` 要求这个 layout 恰好有 m、n、k 三个 mode，否则报错；k 这一维大小是 1，步长 0 表示它不影响线程号。这样铺一遍覆盖 16 × 16 个输出，每个线程一个。

**3. 原子在数值上的排布：置换。** C 的 tile 是 128 × 128，上面的图案要重复。不加置换时，一个线程拿到的是每隔 16 行、每隔 16 列的元素，v1 和 v2 就是这样。

`permutation_m = (16, 4) : (4, 1)` 是 0..63 上的一个函数：`i → 4·(i mod 16) + i div 16`。CuTe 先用它给 m 轴重新编号，再把位置分给线程，于是一个线程的四个位置变成相邻的四行：

```text
m             0  1  2  3  4  5  6  7  8 ... 15 16 17 ...
不置换    a = 0  1  2  3  4  5  6  7  8 ... 15  0  1 ...   a = m mod 16
置换      a = 0  0  0  0  1  1  1  1  2 ...  3  4  4 ...   a = (m mod 64) div 4
```

`a` 是线程的行号（线程号 div 16）。`permutation_n` 对列做同样的事，`None` 表示 k 不置换。例如线程 37 = 16·2 + 5，拥有第 8–11 行和第 72–75 行、第 20–23 列和第 84–87 列，共 8 × 8 个输出。

**为什么要置换。** smem 里 A 沿 m 连续。置换之后，一个线程在某个 k 上要读的 8 个 A 是两段、每段 4 个相邻的值，`autovec_copy` 可以用一次 128-bit 读取读完一段，而不是 4 次标量读取。B 沿 n 同理。这些都在编译期确定，运行时没有代价。

**内核里怎么用**（[第 89-98 行](v4_aligned.py#L89-L98)）。

- `tiled_mma.get_slice(tidx)` 取出本线程的那一份。
- `partition_A(sA)` 得到 `tCsA`，形状是 (1, 8, 8, 3)：原子内的值、沿 m 的重复、沿 k 的重复、流水线的段。`partition_B(sB)` 沿 n 同理。
- `partition_C(gC)` 得到 `tCgC`，形状是 (1, 8, 8)。
- `cute.gemm` 对一个 k 取 8 个 A 和 8 个 B，把每一对相乘，64 个乘积累加进 8 × 8 的累加器。k 是共用的，m 和 n 是输出的坐标。

**依据。** 置换的规则取自 CuTe C++ 版 `TiledMMA::thrfrg_C`：先按置换做 `logical_divide`，再按原子的形状和线程数切分。DSL 把这一步放在编译好的 IR 里，没有可读的 Python。没有在平台上打印过划分结果来核对；"128-bit 读取"也没有在生成的指令里确认过。在 `solve` 里 `print(tiled_mma)` 可以打印线程布局和置换。

### 为什么图案是 64 宽，而不是 128 宽？

因为置换本身只有 64 个位置：`(16, 4) : (4, 1)` 的大小是 16 × 4。16 是线程的行数，4 是每个线程一段相邻值的长度。图案的宽度由这两个数决定，与 tile 无关；128 宽的 tile 放得下两个图案，所以每个线程得到两段、每段 4 个。

128 宽的图案是可以写的：`(16, 8) : (8, 1)` 让每个线程得到一段 8 个相邻的值。但在 v4 里它不会减少读取次数，因为段长 4 正好是一次拷贝能带走的量：

- `autovec_copy` 取的宽度要同时整除段长 × 32 bit、指针的对齐和 256 bit 的上限（宽度的公式见 [Vector Addition 的笔记](../1_vector_add/NOTES.md)）。
- v4 的 `sA`、`sB` 按 16 字节对齐，所以一次拷贝最多 128 bit，即 4 个 `float32`。
- 一段 8 个值仍然要分成两次、每次 4 个，与两段 4 个相同。

要从 8 个一段里得到好处，smem 的每一列都得按 32 字节对齐；现在列的步长是 132 个 `float32`，即 528 字节，是 16 的倍数而不是 32 的倍数，填充也得跟着改。256-bit 的拷贝在 GPU 上是不是一条读取指令，也没有核对过。128 宽的写法没有试过。

### 64 宽的图案怎样铺满 128 的 tile？在哪里指定的？

没有显式指定，是隐含的。`make_tiled_mma` 从头到尾没有见过 128，它只描述图案的一个周期：64 × 64 × 1。重复发生在后面的 `partition_A`、`partition_B`、`partition_C`（[第 93-95 行](v4_aligned.py#L93-L95)），它们拿到的张量带着范围：

- **128 从哪里来：** [第 37-38 行](v4_aligned.py#L37-L38)的 `BM, BN, BK = 128, 128, 8`。`local_tile` 用它把 `gC` 切成 (128, 128)，smem 的 layout 用它定出 `sA`、`sB` 的形状。
- **partition 做什么：** 把图案在张量上能铺几遍就铺几遍。沿 m 是 128 / 64 = 2 个周期；沿 k 原子宽 1，所以是 8 / 1 = 8。
- **结果是什么：** 两个周期并不单独占一个 mode，而是和线程在一个周期内的 4 个位置合并成一个长度 4 × 2 = 8 的 mode，也就是 `tCsA: (MMA, MMA_M, MMA_K, PIPE)` 里的 `MMA_M`。这个 mode 的下标 `r` 对应第 `4a + r mod 4 + 64·(r div 4)` 行。

有没有置换，重复的次数不同，每个线程得到的数量相同：

| | 周期 | 128 内的周期数 | 每个周期内线程的行数 | 每个线程的行数 |
| --- | ---: | ---: | ---: | ---: |
| 不置换 | 16 | 8 | 1 | 8 |
| 置换 `(16, 4) : (4, 1)` | 64 | 2 | 4 | 8 |

沿 n 相同，所以不置换时重复 (8, 8) 次，置换后重复 (2, 2) 次。两种情况下每个线程都是 8 × 8 个输出，partition 的形状都是 (1, 8, 8)；置换改变的是线程拿到哪些行和列。

DSL 的源码里能看到这种分工：`TiledMma.get_tile_size` 在有置换时返回置换的大小（这里是 64），不知道更大的范围；`partition_shape_A/B/C` 则以张量的形状为参数。

- 同一个 `tiled_mma` 可以用在别的 tile 上：64 × 64 时每个线程 4 × 4 个输出，256 × 256 时 16 × 16 个，只需要改 `BM`、`BN`。
- tile 必须是 64 的倍数。不是倍数时 partition 应当报错，但没有试过。
- tiled copy 也是这样：`thr_layout_A` 描述一个 (32, 8) 的小块，`partition_S`、`partition_D` 把它在 (128, 8) 的 tile 上重复 4 次。

### 共享内存为什么按 16 字节对齐？

16 字节是 4 个 `float32`，正好是置换给每个线程的一段，也是一次 128-bit 读取的宽度。[第 76-77 行](v4_aligned.py#L76-L77)的对齐参数让 `autovec_copy` 能用一次读取读完一段。

这个参数在分配器里做两件事：把张量的起始地址放在 16 字节的倍数上；把这个对齐记在指针的类型里，`autovec_copy` 就是从那里读到它的。

默认的对齐是 1 字节，再提高到元素的大小，对 `float32` 是 4 字节。那样 `autovec_copy` 只能退到 32 bit，每次读一个值，置换出来的 4 个一段就没有用处了。

三件事必须一致，16 是把它们连起来的那个数：

- **段长：** 置换给每个线程 4 个一段的 `float32`。
- **起始地址的对齐：** 16 字节，来自这个参数。
- **每一段都从 16 字节的边界开始：** 一段从第 `4a` 行开始，即 `16a` 字节。沿 k 的各列相隔 132 个 `float32`，即 528 字节 = 33 × 16，这就是 `PAD_A` 取 4 而不取 1 的原因（[第 42-46 行](v4_aligned.py#L42-L46)）。`sB` 的列相隔 128 个 `float32`，即 512 字节。

更大的值（例如 32）在这里没有好处：没有哪次拷贝一次读超过 4 个 `float32`，528 字节的列步长也不是 32 的倍数。

## 参考

- [LeetGPU：Matrix Multiplication](https://leetgpu.com/challenges/matrix-multiplication)
- [NVIDIA CUTLASS 4.4.2：`ampere/sgemm.py` 示例](https://github.com/NVIDIA/cutlass/blob/v4.4.2/examples/python/CuTeDSL/ampere/sgemm.py)
- [NVIDIA CUTLASS 4.4.2：`make_tiled_mma`、`TiledMma` 和 `partition_A/B/C`](https://github.com/NVIDIA/cutlass/blob/v4.4.2/python/CuTeDSL/cutlass/cute/atom.py)
- [NVIDIA CUTLASS 4.4.2：C++ 版 `TiledMMA::thrfrg_C`，置换和切分的定义](https://github.com/NVIDIA/cutlass/blob/v4.4.2/include/cute/atom/mma_atom.hpp)
- [NVIDIA CUTLASS 4.4.2：`autovec_copy` 实现](https://github.com/NVIDIA/cutlass/blob/v4.4.2/python/CuTeDSL/cutlass/cute/algorithm.py)
- [NVIDIA CUTLASS 4.4.2：`SmemAllocator.allocate_tensor`](https://github.com/NVIDIA/cutlass/blob/v4.4.2/python/CuTeDSL/cutlass/utils/smem_allocator.py)

上述实现细节核对自 CUTLASS v4.4.2 的源码（`make setup` 取回的 `.cache/cutlass`，以及同一版本的 C++ 头文件）。线程拥有哪些元素是按这些定义推出来的，没有在平台上打印核对；最终生成的硬件指令尚未检查。
