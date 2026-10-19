# LeetGPU · CuTe DSL

记录用 CuTe DSL 解 LeetGPU challenge 的过程：从第一版可理解的实现，到后续优化，以及每次提交的结果和学习中的具体问题。

```text
README.md
AGENTS.md
tools/
└── leetgpu/                 # 认证、历史查询和提交 CLI
solutions/
└── vector-addition/
    ├── README.md
    ├── submissions.json
    ├── v1_scalar.py
    └── v2_vectorized.py
```

## Solutions

- [Vector Addition](solutions/vector-addition/)：逐线程标量实现 → 使用 layout、寄存器 fragment 和自动向量化的实现。

每个 challenge 使用一个 `solutions/<challenge>/` 目录。`README.md` 集中记录题意、版本与提交历史、latency 等评测结果、演进过程和 Q&A；同目录保存各版本可直接提交的完整源码。新增迭代时保留旧版本，在 `README.md` 增加对应记录。平台记录尚未取得或尚未与源码对应时，明确标为待同步。

所有变更在独立分支上完成，通过 PR 合入 `main`；合并前解决讨论，使用 Squash merge 保持线性历史，合并后自动删除分支。当前按个人学习仓库维护，PR 所需批准数为 0。`main` 禁止强推和删除，规则也适用于管理员。

## CLI 与 coding agents

当前重点为 **H100 / CuTe DSL**。仓库自带无 npm 依赖的非官方 LeetGPU 工具，需要 Node.js 22+：

```sh
./tools/leetgpu/leetgpu login
./tools/leetgpu/leetgpu status --json
./tools/leetgpu/leetgpu submissions list --challenge 1 --language cute --gpu H100
```

macOS 可用 `login --existing-browser` 从已登录的 Chrome 标签页获取会话；无桌面的远程机器支持 SSH 转发登录页和 `login --stdin` 导入。凭据保存在用户目录，不进入 Git。完整用法、前置条件和限制见 [工具 README](tools/leetgpu/README.md)。

[AGENTS.md](AGENTS.md) 是 coding agents 的仓库入口，包含认证、H100 查询/提交和结果核验流程。不自动读取此文件的 agent，可以在任务中明确要求先阅读它。工具实现放在 `tools/`，便于任何终端或 agent 复用。
