# LeetGPU · CuTe DSL

记录用 CuTe DSL 解 LeetGPU challenge 的过程：从第一版可理解的实现，到后续优化，以及每次提交的结果和学习中的具体问题。

```text
README.md
solutions/
└── vector-addition/
    ├── index.md
    ├── v1_scalar.py
    └── v2_vectorized.py
```

## Solutions

- [Vector Addition](solutions/vector-addition/index.md)：逐线程标量实现 → 使用 layout、寄存器 fragment 和自动向量化的实现。

每个 challenge 使用一个 `solutions/<challenge>/` 目录。`index.md` 集中记录题意、版本与提交历史、latency 等评测结果、演进过程和 Q&A；同目录保存各版本可直接提交的完整源码。新增迭代时保留旧版本，在 `index.md` 增加对应记录。平台记录尚未取得或尚未与源码对应时，明确标为待同步。
