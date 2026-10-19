# leetgpu-cutedsl

通过 CuTe DSL 逐题解决 [LeetGPU Challenges](https://leetgpu.com/challenges)，记录从正确实现到性能优化的完整学习过程。

## 目标

- 用 CuTe DSL 完成全部 LeetGPU challenges，持续维护题目与进度。
- 理解 layout、tensor、线程与数据映射、内存访问和并行计算。
- 保留解题思路、正确性验证、性能实验和失败尝试，让每次优化可以复现。

当前处于仓库初始化阶段，尚未提交题目解答。题目范围以 LeetGPU 平台为准，进度见 [progress.md](progress.md)。

## 目录

```text
solutions/                   # 每道题独立目录：实现、测试、基准和解题笔记
notes/                       # 跨题的 CuTe DSL 知识总结与学习日志
templates/solution-notes.md   # 单题解题与优化记录模板
progress.md                  # 解题进度索引
```

单题目录约定见 [solutions/README.md](solutions/README.md)，知识笔记入口见 [notes/README.md](notes/README.md)。

## 每道题的工作流

1. 从 LeetGPU 选择题目，在 `progress.md` 记录原题链接与状态。
2. 新建 `solutions/<challenge-slug>/`，复制 `templates/solution-notes.md` 为该题的 `README.md`。
3. 先理解输入输出和数值要求，再编写最容易验证的 CuTe DSL 实现。
4. 添加正确性测试，覆盖边界尺寸、非整除尺寸及题目要求的数据类型。
5. 在 LeetGPU 提交验证；单独记录本地测试和平台通过结果。
6. 测量性能，再逐项优化，记录假设、改动、结果及对应 commit。
7. 更新进度，把可复用的知识整理到 `notes/`。

先建立正确性基线，再优化性能；保留失败实验的原因，不只保存最终答案。

## 环境与复现

安装和运行环境以 NVIDIA 的 [CuTe DSL Quick Start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html) 为准。实际开始解题时，再根据目标 GPU 配置并固定依赖版本。

每次有效的测试和性能记录应包含 GPU 型号、驱动、CUDA、Python、CuTe DSL 版本，以及输入尺寸、dtype、运行命令和计时方法。只有在可比较的环境和测量条件下，才报告加速比。

## 学习资料

- [LeetGPU Challenges](https://leetgpu.com/challenges)
- [LeetGPU 官方题库与测试](https://github.com/AlphaGPU/leetgpu-challenges)
- [NVIDIA CuTe DSL 文档](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html)
- [NVIDIA CuTe DSL 示例](https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL)

本仓库用于个人学习，与 LeetGPU、NVIDIA 官方无隶属关系。
