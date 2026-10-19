# Vector Addition

[LeetGPU 题目](https://leetgpu.com/challenges/vector-addition)

输入 `A`、`B` 和输出 `C` 是 GPU 上长度为 `N` 的连续一维 `float32` Tensor，题目约束为 `1 <= N <= 100,000,000`，性能测试使用 `N = 25,000,000`。计算 `C[i] = A[i] + B[i]`，其中 `0 <= i < N`，结果直接写入 `C`。

## 版本与提交历史

2026-10-19 整理入库。**已有平台结果，待核对版本后同步。** 当前尚未取得可对应到以下源码版本的 LeetGPU 提交时间、评测状态、latency 和 GPU 信息；本次整理日期与平台提交日期分别记录。

| 版本 / 源码 | 整理入库日期 | LeetGPU 提交时间 | 评测状态 | Latency | GPU |
|---|---|---|---|---|---|
| [v1 · Scalar](v1_scalar.py) | 2026-10-19 | 未取得 | 待核对已有平台记录 | 待同步 | 待同步 |
| [v2 · Vectorized](v2_vectorized.py) | 2026-10-19 | 未取得 | 待核对已有平台记录 | 待同步 | 待同步 |

## 演进过程

**v1：一个线程处理一个元素。** 使用 256 个线程的 block，通过 `bidx * block_size + tidx` 计算下标，检查边界后直接读 A/B、相加、写 C。这版用于建立 CUDA 执行模型与 CuTe Tensor 索引的对应关系。

**v2：一个线程处理四个元素。** 每个 block 处理 1024 个元素，用 thread/value layout 分配元素，构造每线程视图，通过寄存器 fragment 与 `autovec_copy` 表达搬运和计算。启动前检查三个基地址的 16 字节对齐，满足时向编译器补充对齐信息；完整 tile 使用向量化路径，尾部逐元素检查边界。这些改变旨在减少访存指令与 block 调度开销，是否带来加速以对应平台结果为准。

本次本地检查范围为 Python 语法与源码静态检查；本地环境没有 CUDA GPU，未在本地完成 CuTe 编译、GPU 正确性或性能验证。已有平台实测结果需要与保留的源码版本核对后再补入上表。

## Q&A

### 最初的实现里，CuTe 起了什么作用？用了什么 layout？

最初的版本仍是 CUDA 的逐线程执行模型：每个线程算一个 `i`，执行 `C[i] = A[i] + B[i]`。`@cute.kernel` 把 DSL 函数编译成 GPU kernel，`@cute.jit` 和 `.launch()` 表达主机端的编译、启动逻辑，`cute.arch` 提供线程和 block 索引。它没有使用 CuTe 的分块和线程分配能力，也不会仅因为使用了 CuTe 就更快。

这里有两种映射：

- **数据的 layout** 随输入 `cute.Tensor` 传入。连续一维向量的 layout 是 `shape=(N,)`、`stride=(1,)`，把逻辑下标 `i` 映射到元素偏移 `i`。
- **线程到元素的映射** 是手写的 `i = bidx * block_size + tidx`，没有显式构造 thread/value layout。

连续线程访问连续元素，因此原版已经可以合并访存（coalescing）。合并访存发生在 warp 的多个线程之间；每个线程一次搬运多个元素的向量化是另一件事。

### 怎样利用 CuTe 的特性优化这个 challenge？

向量化版本使用 256 个线程处理一个 1024 元素的 tile，每个线程处理 4 个连续的 `float32`：

```python
tv_layout = cute.make_layout((256, 4), stride=(4, 1))
# (thread, value) -> 4 * thread + value
```

`local_tile` 选出 block 的数据块，`composition` 和切片生成每线程视图，`autovec_copy` 根据布局、类型和对齐信息生成适合的 copy。完整 tile 使用向量化路径，最后不足 1024 个元素的 tile 使用边界检查。尾部使用 `base + tidx + k * threads`，让同一次标量访问中的相邻线程访问相邻元素。

有望减少的是访存指令数和 block 调度开销。数据量仍为读 A、读 B、写 C，共 `12 * N` 字节。小输入可能受启动开销限制，大输入可能受显存带宽限制；更少的 block 也不一定总是更快。性能是否提升、提升多少，必须在目标 GPU 上测量。

### 地址检查为什么是三个地址按位或，再与 15？

```python
addresses = A.iterator.toint() | B.iterator.toint() | C.iterator.toint()
if (addresses & 15) == 0:
    ...
```

`iterator.toint()` 得到数据指针的整数形式，单位是字节；它没有读取张量内容。**16 字节对齐**意味着地址是 16 的倍数，二进制最低 4 位全为 0。`15 == 0b1111`，所以 `addr & 15` 只保留最低 4 位。

按位或的结果中，某一位为 0 当且仅当三个输入的这一位都为 0。因此，这个条件等价于：

```python
addr_a % 16 == 0 and addr_b % 16 == 0 and addr_c % 16 == 0
```

`addresses` 只是用于判断的位运算结果，不是随后要解引用的新指针。这里的单位是 **16 字节 / 128 bit**，不是 16 bit。

对于连续 `float32` 输入，线程负责的第一项地址为：

```text
A 的起始地址 + 4 * (1024 * bidx + 4 * tidx)
= A 的起始地址 + 4096 * bidx + 16 * tidx
```

后两项都是 16 的倍数，因此基地址对齐时，每个线程的四元素组也对齐。这一证明需要结合布局；仅检查基地址不能证明任意 stride 的视图都适合 128-bit copy。

### `_with_alignment_16` 做了什么？没有它，`autovec_copy` 就不会触发吗？

`_with_alignment_16` 用同一个地址、元素类型和地址空间创建指针，声明 `assumed_align=16`，再用原来的 layout 创建 Tensor。它**不移动地址、不分配新数据，也不把地址向上或向下取整**。

需要区分两个事实：

1. 地址实际满足 16 字节对齐。
2. 编译器知道该指针至少满足 16 字节对齐。

运行时的掩码判断证明第一点，辅助函数把这项保证提供给编译器。在当前实现中，不能只依赖 `if` 判断自动增强指针类型的对齐信息。

去掉辅助函数，`autovec_copy` 仍然执行；变化可能是它能选择的 copy 宽度。在布局兼容、目标寄存器 fragment 对齐充分时：

| 编译器已知的源指针对齐 | 四个连续 `float32` 的 copy 分组 |
|---|---|
| 至少 16 字节 | 可以选择一组 128 bit |
| 8 字节 | 通常为两组 64 bit |
| 4 字节 | 通常为四组 32 bit |

如果调用方已经通过 `from_dlpack(..., assumed_align=16)` 提供了正确的对齐信息，这个辅助函数就可能冗余。`assumed_align` 是必须真实满足的保证，不能为了向量化而随意加大；例如从对齐 buffer 的第二个 float 开始取 view，地址就偏移了 4 字节。

### `autovec_copy` 如何选择宽度？一定是 `float4` 吗？

它在**编译时**分析传入的源和目标 Tensor，选择双方都安全支持的向量化宽度。它不会在运行时重新检查地址、试跑多个宽度并选择最快的版本，也不会把其他线程的数据自动并入当前线程。

CuTe DSL 4.4.2 与本次核对的本地源码都要求源和目标的 shape 为静态，元素位宽相等。它先用 `max_common_layout(src, dst)` 找到公共的连续分组，再综合 layout 的对齐约束、两端指针的可证明对齐和实现上限：

```python
# 源码逻辑的简化表示。
group = gcd(src.layout.max_alignment,
            dst.layout.max_alignment,
            common_elements)
copy_bits = gcd(group * element_bits,
                src.iterator.max_alignment * 8,
                dst.iterator.max_alignment * 8,
                256)
```

这里用 `gcd` 保证统一分组能满足各项约束；当前实现的策略上限是 256 bit。`max_alignment` 还会考虑指针的 swizzle 等限制，不能只看原始分配地址。

对于布局兼容、其他约束充分的连续 `float32` 视图：

| 每线程元素数 | 两端有效对齐的较小值 | 策略可选宽度 | 分组 |
|---:|---:|---:|---|
| 4 | 4 字节 | 32 bit | 4 组，每组 1 个 float |
| 4 | 8 字节 | 64 bit | 2 组，每组 2 个 float |
| 4 | 16 字节 | 128 bit | 1 组，每组 4 个 float |
| 4 | 32 字节 | 128 bit | 总共只有 4 个 float |
| 8 | 16 字节 | 128 bit | 2 组，每组 4 个 float |
| 8 | 32 字节 | 256 bit | 1 组，每组 8 个 float |

找不到适合向量化的公共分组时会退回基础 copy。本例每线程只有 4 个 float，且全局指针声明了 16 字节对齐，所以具备选择 128-bit copy 的条件；这可以类比为 `float4` 搬运，但不表示 Python 创建了 CUDA C++ 的 `float4` 对象。

表格是策略层的分组宽度，不能直接当成最终硬件指令数。尤其是 256 bit，最终是否拆分取决于目标架构和后端，需要检查 PTX/SASS；这里没有实测生成的指令。

### 为什么把 `bidx` 转成 `cute.Uint32`？

```python
if cute.Uint32(bidx) < N // tile_size:
```

`block_idx()` 提供的 `bidx` 是 `Int32`，参数 `N` 是 `Uint32`。显式转换让比较按无符号 32 位整数进行，也明确索引的含义。在本题的索引范围中，block 编号非负，转换不改变数值。

按当前检查的 CuTe 类型提升实现，等宽 `Int32` 与 `Uint32` 混合运算会提升到 `Uint32`，所以这个比较中的显式转换可以省略，不能把它视为必需的性能优化。这个结论只针对该比较和本题范围，不意味着所有有符号索引运算都可以任意改成无符号。

### `local_tile`、`tv_layout`、`composition` 和 `tA/tB/tC` 到底是什么关系？

以 A 为例，把代码分开看：

```python
tile_A = cute.local_tile(A, (1024,), (bidx,))
tv_layout = cute.make_layout((256, 4), stride=(4, 1))
tv_A = cute.composition(tile_A, tv_layout)
tA = tv_A[tidx, None]
```

- `tile_A` 是整个 block 的全局内存视图：连续输入时覆盖 `A[1024*bidx : 1024*(bidx+1)]`。
- `tv_layout` 是映射 `(tid, value) -> 4*tid + value`，本身不含 A 的地址。
- `tv_A` 是组合后的 Tensor 视图，接受 `(tid, value)` 两个坐标。
- `tA` 固定了 `tid=tidx`，只保留当前线程的四个 value，接受一个坐标 `value`。

组合的方向是 `tile_A ∘ tv_layout`：先应用右边的线程/元素映射，再应用左边的数据 layout。语义关系为：

```text
(tid, value)
    -> tv_layout: tile 内的逻辑下标 4*tid + value
    -> tile_A.layout: 相对于 tile 起点的存储偏移
    -> tile_A.iterator: 实际地址

tv_A[tid, value] ≡ tile_A[4*tid + value]
tA[value]        ≡ tile_A[4*tidx + value]
```

CuTe 中这里的 `None` 表示保留该维度，更接近 NumPy 的 `:`；它不是 NumPy 中用于增加维度的 `None`。

例如 `bidx=2, tidx=3`，`tA` 对应 `A[2060]` 到 `A[2063]`。因此，“给定 `(tid, localValueId)` 映射到底层存储”描述的是 `tv_A`；`tA` 已经固定 tid，其贡献计入了切片后的指针。

若 A 的元素 stride 为 `s`、起始字节地址为 `pA`，则：

```text
tA 起始地址 = pA + 4*s*(1024*bidx + 4*tidx)
tA.shape    = (4,)
tA.stride   = (s,)
address(tA[v]) = pA + 4*s*(1024*bidx + 4*tidx + v)
```

最前面的 `4` 是 `float32` 的字节数。`s=1` 时四个元素连续，`s=2` 时它们间隔存放。`composition` 保留并组合原始数据布局，**不会重新排列底层数据**。

`tA/tB/tC` 因而不是单独的 layout，而是包含指针、layout、元素类型等信息的 Tensor 视图。`tA/tB` 描述输入位置，`tC` 描述输出位置。这些视图和 layout 的构造不读取 A/B 的元素，也不分配一个新的全局内存 tile；真正搬运数据的是后面的 copy。

### `fragment` 是什么？它与 Tensor、layout、`TensorSSA` 有什么关系？

在本例中，fragment 是**当前线程持有、用于计算的一小组张量数据**。它描述这组数据的角色，不是与 Tensor 互斥的独立类型：一个 fragment 可以由寄存器 Tensor 表示。

```python
rA = cute.make_rmem_tensor((4,), A.element_type)
```

这个 `rA` 就是当前线程的四元素 register fragment。每个线程有自己的 `rA`；256 个线程各持有 4 个元素，共同覆盖完整的 1024 元素 tile。fragment 也可以保存中间结果和累加值，不限于输入副本。

| 对象 | 表示什么 |
|---|---|
| `tA` | 全局内存视图：当前线程从 A 的哪些位置读取 |
| `rA` | 寄存器 Tensor：存放当前线程读入的数据和计算结果 |
| `rA.load()` | `TensorSSA`：以不可变、线程局部的值语义表达这一组元素，用于计算 |

fragment 自己也有 layout，用来组织**线程内部**的数据；`tv_layout` 则描述**线程与 tile 元素之间**的分配。fragment layout 不应理解为对 GPU 物理寄存器编号的直接指定。最终物理寄存器分配由编译器决定。

源数据与 fragment 的 layout 可以不同，例如全局内存 stride 为 2 的四个输入可以复制到紧凑排列的 fragment。四元素 fragment 的大小也不等于 copy 指令宽度：它既可以由一组 128-bit copy 填满，也可以由四组 32-bit copy 填满。

### 从 `rA/rB` 的建立到写回 C，实际发生了什么？

```python
rA = cute.make_rmem_tensor((4,), A.element_type)
rB = cute.make_rmem_tensor((4,), B.element_type)
cute.autovec_copy(tA, rA)
cute.autovec_copy(tB, rB)
rA.store(rA.load() + rB.load())
cute.autovec_copy(rA, tC)
```

1. 建立两个线程私有的寄存器 fragment；这时还没有有效输入数据。
2. 两个 `autovec_copy` 从全局内存读取本线程的 A/B 元素，填入 fragment。
3. `load()` 获取 fragment 的值，`+` 做四组逐元素加法，`store()` 将结果保存回 `rA`。
4. 最后一个 `autovec_copy` 把结果写入 `tC` 指定的全局内存位置。

这里 fragment 的 `load()`/`store()` 不会再次访问 A/B 的全局内存，也不必各自对应一条独立硬件 load/store 指令；编译器可以把它们整理成寄存器数据流。修改 `rA` 不会修改 A，直到明确执行到全局内存的 copy 才会写出结果。

完整 tile 路径已由 `bidx < N // tile_size` 保证所有访问在范围内；尾部单独检查 `i < N`。线程之间没有读写依赖，因此这个 kernel 不需要共享内存或 block 同步屏障。

## 参考

- [LeetGPU：Vector Addition](https://leetgpu.com/challenges/vector-addition)
- [NVIDIA CUTLASS 4.4.2：`autovec_copy` 实现](https://github.com/NVIDIA/cutlass/blob/v4.4.2/python/CuTeDSL/cutlass/cute/algorithm.py)
- [NVIDIA CUTLASS：Tensor、寄存器 fragment 与 `TensorSSA`](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/python/CuTeDSL/cutlass/cute/tensor.py)
- [NVIDIA CUTLASS：数值类型与类型提升](https://github.com/NVIDIA/cutlass/blob/0b55a2f691d69981583568fd9eb69687b1f0de8a/python/CuTeDSL/cutlass/base_dsl/typing.py)

上述实现细节核对自本地 CUTLASS 源码快照 `0b55a2f691d69981583568fd9eb69687b1f0de8a`。2026-10-19 页面显示 CuTe DSL 4.4.2，已另行核对该版本的 `autovec_copy`：本例使用的向量宽度公式与 256-bit 上限相同。页面当前环境不等于所有历史提交的环境；最终生成的硬件指令尚未检查。
