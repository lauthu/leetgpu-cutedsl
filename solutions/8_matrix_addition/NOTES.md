# Matrix Addition · 学习笔记

学习中的问答，针对 [v1.py](v1.py)、[v2_per_thread.py](v2_per_thread.py) 和 [v3_minimal.py](v3_minimal.py)。评测结果、评审和版本说明见 [README.md](README.md)。

标"自测"的数字不是提交记录，量法见 README 的评审一节。"源码"指 CuTe DSL 4.4.2，在 `.cache/cutlass/python/CuTeDSL/cutlass/cute/` 下；"官方示例"在 `.cache/cutlass/examples/python/CuTeDSL/ampere/` 下。

## Q&A

### 第一版为什么差点编译不过？

`v1.py` 是从 Vector Addition 的 `v2_vectorized.py` 改来的，内核没有动，只在 `solve` 里先把矩阵摊平。最初有两行把 `cute.make_tensor` 写成了 `cute.make_layout`：

```python
B1 = cute.make_layout(B.iterator, cute.make_layout(N1, stride=1))  # 错
B1 = cute.make_tensor(B.iterator, cute.make_layout(N1, stride=1))  # 对
```

`make_layout` 只接受一个位置参数（shape），`stride` 必须用关键字传，所以 trace `solve` 时会抛 `TypeError`。本地的预检和 lint 都通过了：它们只查语法错误和未定义的名字，查不出"函数存在，但用错了"。这种错要到平台上才报，会白花一次 run。这一次是在第一次 run 之前读代码发现的。

### 把矩阵摊平成一维，对吗？

对。逐元素相加与行、列无关，每个位置各算各的。三个矩阵都是按行连续存放的 `N × N`，所以各自就是 `N²` 个连续的 `float32`：

```python
cute.make_tensor(A.iterator, cute.make_layout(N * N, stride=1))
```

`A.iterator` 是指向第一个元素的指针，layout `N²:1` 的意思是"`N²` 个元素，步长 1"。内存没有动，只是换了一个视图，与 Matrix Transpose 里的 `Bt` 是同一种做法。

官方示例 `elementwise_add.py` 保留二维：tile 是二维的，矩阵边上不完整的 tile 用谓词处理。`N` 任意时那样要处理两个方向的边界，摊平之后只剩末尾一处。

### 我用的 API 对吗？

对。逐个对照了源码和官方的两个逐元素示例：

| v1 里的写法 | 结论 | 依据 |
| --- | --- | --- |
| `cute.composition(tile, tv_layout)[tidx, None]` | 是惯用法 | 官方示例 `elementwise_apply.py` 用同样的方式取出一个线程的元素 |
| `cute.autovec_copy` | 用对了 | 源码里它按指针声明的对齐选访问宽度：16 字节对齐时是 128-bit，4 字节对齐时是 32-bit |
| 检查地址之后 `cute.make_ptr(..., assumed_align=16)` | 用对了 | `from_dlpack` 默认的对齐是一个元素的大小（4 字节），只有调用方能改；这里调用方是平台 |
| `cute.make_tensor(A.iterator, cute.make_layout(N * N))` | 用对了，也是最简单的摊平方式 | 见上一条 |

不整洁的不是 API，而是 API 周围的结构。

### `A`、`A1`、`A16` 这样改名很丑，怎么整理？

v1 里同一块内存有三个名字：平台给的 `A`，摊平后的 `A1`，声明了 16 字节对齐的 `A16`。v2 用一个 helper 把摊平和对齐声明一次做完，`solve` 里一块内存只有一个名字：

```python
def _flat(x, count, align):
    ptr = cute.make_ptr(
        x.element_type, x.iterator.toint(), x.iterator.memspace, assumed_align=align
    )
    return cute.make_tensor(ptr, cute.make_layout(count, stride=1))
```

v2 另外整理了三处：

- **下标只用 `Int32`。** `N ≤ 4096`，元素个数最多 2²⁴，放得进 `Int32`。在 `solve` 里转一次（`cutlass.Int32(N * N)`），内核里就不需要 `cute.Uint32(...)` 这样的转换了。
- **按线程选择路径，不按 block。** 每个线程拥有 4 个连续的元素：4 个都在数据内就一起搬运，否则逐个处理并检查越界。1024 个元素的 tile 和尾部里的 `k * threads` 都不需要了。
- **测试覆盖顺带变好了。** 官方 11 个功能测试的 `N ≤ 32`。v1 里只有 32 × 32（正好一个 tile）进入向量路径；按线程选择之后有 10 个进入，`make run` 的 2 × 2 示例也是。

### 为什么去掉内存对齐的处理？

**它原来的用途。** 平台给的 Tensor，指针只保证 4 字节对齐。一条 128-bit 的读写指令一次搬 4 个 `float32`，但地址必须是 16 的倍数，而且只有指针的类型保证了这一点，编译器才会生成它。所以 v1 和 v2 先检查三个真实地址，都是 16 的倍数时用 `assumed_align=16` 重建指针，`autovec_copy` 就选 128-bit；否则退到 4 次 32-bit。

**去掉的原因是量出来没有差别。** 自测，v2 的内核，`N = 4096`：

| 视图 | 各轮中位数 (ms) |
| --- | --- |
| 16 字节对齐（128-bit 访问） | 0.0878–0.0923 |
| 4 字节对齐（32-bit 访问） | 0.0873–0.0905 |

解释（推断）：这个内核受内存带宽限制。它要搬 201.3 MB，时间取决于这些字节在内存和 GPU 之间走得多快，扣掉启动开销约 2.9 TB/s。一个 warp 不管用哪种宽度，碰的都是同样连续的 512 字节，内存上的流量一样；多出来的指令花的是计算时间，而计算不是瓶颈。

**它的代价。** 一次地址检查、第二条 launch 路径、第二个要编译的 kernel（编译时间计入平台约 40 秒的时限），还有二十来行要读的代码。

两点要记住：

- **去掉一个声明总是安全的。** 不声明时编译器只假定 4 字节对齐，这对 `float32` 的缓冲区永远成立。危险的是反过来：地址不是 16 的倍数却声明了 16。
- **这不是通用的结论。** Matrix Multiplication 的 v6 里，128-bit 的 copy atom 没有 16 字节的声明根本编译不过；受计算限制的内核里，指令数也会有影响。

没有验证的一点：`load()` 在 4 字节对齐的指针上生成了什么指令没有看过，只知道在真的没有对齐的地址上结果是对的。

### 为什么去掉寄存器 Tensor？

**它原来的用途。** `autovec_copy` 是拷贝，需要一个目的 Tensor。要把数据拿进线程，就先拷到寄存器 Tensor，再用 `load()` 把它的内容变成可以做运算的值；写回去时反过来。v2 的路径是：

```text
全局内存 → 拷贝 → 寄存器 Tensor → load() → 值
值 + 值
值 → store() → 寄存器 Tensor → 拷贝 → 全局内存
```

**去掉的原因。** `load()` 和 `store()` 不是寄存器 Tensor 专用的。源码里它们接受寄存器、共享内存和全局内存里的 Tensor，条件是形状是静态的、layout 不是 composed layout。一个线程的 4 个元素形状是静态的，所以中间那一段是绕路：

```text
全局内存 → load() → 值
值 + 值
值 → store() → 全局内存
```

也就是 `tC.store(tA.load() + tB.load())`，官方示例 `elementwise_apply.py` 就是这样写的。自测里它是 0.0865–0.0907 ms，与带寄存器 Tensor 的写法没有差别。

**v1 为什么有它。** 内核是围绕 `autovec_copy` 写的，而 `autovec_copy` 的本事是按对齐选最宽的访问。对齐处理去掉之后，用它的理由没有了，寄存器 Tensor 也跟着不需要了。

仍然需要寄存器 Tensor 的情况：

- 用 `cute.copy` 配 copy atom 或 tiled copy 时，包括用谓词处理不完整的 tile（Matrix Transpose）。
- 数据要跨几步保留时，例如 Matrix Multiplication 里 MMA 的累加器。
- 在循环里逐个元素更新时。

### v2 为什么比 v1 慢？

没有定论。四次提交都在 2026-11-23（UTC）：

| 时间 | 版本 | Latency (ms) |
| --- | --- | ---: |
| 07:12 | v1 | 0.13778 |
| 07:39 | v2 | 0.18821 |
| 07:56 | v2 | 0.21289 |
| 07:57 | v3 | 0.21318 |

能确定的：

- **v3 相对 v2 没有代价。** 两次提交相隔 40 秒，差 0.1%。
- **同一份代码会差 13.1%。** 这是 v2 两次提交之间的差距，相隔 17 分钟。
- **v1 领先的幅度超过这个波动。** 所以按提交记录，"v2 比 v1 慢"成立。

与之矛盾的：自测里 v1 和 v2 在同一个进程里交替计时，是 0.0879–0.0897 和 0.0878–0.0923 ms，没有差别。完整的一组元素，两版都是 block `b` 的线程 `t` 访问第 `1024·b + 4·t` 到 `+3` 个元素，地址完全一样。

两种解释，现有数据分不开（推测）：

- **平台在这 45 分钟里变慢了。** 四个数字随时间一路上升，同一分钟里提交的两份不同的代码几乎一样。
- **v2 和 v3 在平台的计时条件下确实比 v1 慢。** 平台每次计时都用新生成的输入并先清 L2 cache，自测没有模仿这些条件。但想不出机制。

v1 只有最早的一次提交。要分开这两种解释，需要一个与其他几次同一时段的 v1 结果。

还有一点值得记住：提交比自测（约 0.089 ms）多出 0.049–0.124 ms，这部分在内核之外，而且在变。它是绝对量：0.05 ms 是这个内核的 36%，放在 3.6 ms 的矩阵乘法上只有 1.4%。平台的 FAQ 也说，单次结果不要看得太重。

### shared memory 在这道题里有用吗？

没有用，很可能还会稍慢一点（没有量过，是从已经量到的东西推的）。

shared memory 在前面几道题里有三种用处，这里一种也用不上：

| 用处 | 例子 | 这道题 |
| --- | --- | --- |
| 复用：从全局内存读一次，被很多线程用很多次 | Matrix Multiplication 的 tile | 每个元素只被一个线程读一次、用一次 |
| 换顺序：读和写各自沿自己连续的方向 | Matrix Transpose | `A`、`B`、`C` 的访问顺序相同，一个 warp 本来就在读写相邻的内存 |
| 喂给从 shared memory 取操作数的指令 | H100 上的 warpgroup MMA | 每个元素只做一次加法 |

代价：内核受内存带宽限制，shared memory 不会减少要在全局内存上来回搬的 201.3 MB，只会多出一次写入、一次读出，以及每个 block 一次 `sync_threads`。
