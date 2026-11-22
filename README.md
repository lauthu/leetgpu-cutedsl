# LeetGPU · CuTe DSL

用 CuTe DSL 解 [LeetGPU](https://leetgpu.com/) challenge 的学习记录。每道题保留从第一版到后续优化的各版源码、平台评测结果，以及学习中遇到的具体问题。当前重点是 **H100** 上的 CuTe DSL（语言参数 `cute`）。

代码在本机编写，在 LeetGPU 平台上编译、运行和评测。CuTe DSL 只提供 Linux 的 wheel，所以没有 CUDA 的机器（包括 macOS）无法在本地运行 kernel。

Coding agents 请先读 [AGENTS.md](AGENTS.md)：认证、提交授权和记录规则都在那里。

## 解法

| Challenge | 版本演进 |
| --- | --- |
| [Vector Addition](solutions/1_vector_add/) | 逐线程标量 → layout、寄存器 fragment 与自动向量化 |
| [Matrix Multiplication](solutions/2_matrix_multiplication/) | 手写下标 → tiled copy 与 tiled MMA → `cp.async` 流水线 → FP16 tensor core → warpgroup MMA（H100）→ tcgen05 MMA 与 tensor memory（B200） |

## 快速开始

需要 Python 3、Node.js 22+、Git 和 GNU Make。

```sh
make setup    # 建 .venv、装固定版本的 ruff、取回 CuTe DSL 源码和官方题目定义
make login    # 登录 LeetGPU 一次，之后自动续期
make run    FILE=solutions/1_vector_add/v2_vectorized.py   # 在平台上运行，不产生提交
make submit FILE=solutions/1_vector_add/v2_vectorized.py   # private 提交并写本地回执
make check    # lint + 提交记录校验 + 工具测试，不需要登录或 GPU
```

- 题目目录与[官方仓库](https://github.com/AlphaGPU/leetgpu-challenges)的目录同名，例如 `1_vector_add`。challenge ID 取自数字前缀，也可以用 `CHALLENGE=<ID>` 指定。
- GPU 默认是 `H100`，用 `GPU=B200` 这样的参数覆盖。
- `run` 只执行题面的示例测试，用来看编译错误和一个小例子的结果；只有 `submit` 跑全部测试并产生带 latency 的提交记录。两者都占用平台 GPU。
- `run` 和 `submit` 会先在本地检查语法错误和未定义名称，检查失败时不联网。
- `make leaderboard CHALLENGE=1_vector_add` 列出 CuTe 在该 GPU 上的前几名成绩，用来看离最好还有多远。
- LeetGPU 每天的提交有配额，所以不重复提交同一份代码，每次提交都应该是一个值得一试的新版本。
- `make help` 列出全部目标。

保持登录、自动续期、历史查询和 `run` 都已在 2026-11-19 用真实账号验证，详情见[工具 README](tools/leetgpu/README.md)。

## 新增一个版本

1. 新题目先建目录 `solutions/<id>_<name>/`，名字照抄 `.cache/leetgpu-challenges/challenges/<难度>/` 下的官方目录，例如 `2_matrix_multiplication`。那里还有题面、起始代码、容差和性能测试规模。
2. 新建 `vN_<说明>.py`，不覆盖旧版本。
3. 反复 `make run FILE=...` 直到通过，再 `make submit FILE=...`。
4. 让 agent review 这道题，例如"review 1_vector_add"。Review 评审每个版本的代码，在每个有提交的 GPU 上对照 CuTe 的排行榜找出差距和下一步的性能机会，同时同步提交历史，并按固定模板重写该题的 `README.md`。它不花提交配额。流程和模板在 [.claude/skills/review-solution/](.claude/skills/review-solution/)。
5. 运行 `make check`，然后开 PR。

## 登录

在 macOS 上，`make login` 打开一个独立的 Chrome 窗口。在里面登录 LeetGPU，窗口会在几秒后自动关闭；refresh token 存入钥匙串，之后的命令自动续期，不需要再登录。

| 情况 | 做法 |
| --- | --- |
| 网络无法直接访问登录提供商 | `make login PROXY=http://proxy.example:3128`：只让登录窗口走代理 |
| 登录提供商拒绝该窗口 | `make login LOGIN_FLAGS=--persist`：在无痕窗口登录后粘贴，同样保持登录 |
| 不想保存 refresh token | `make login LOGIN_FLAGS=--existing-browser`：读取已登录的 Chrome 标签页，每小时重新登录 |
| 远程 Linux 或 SSH | `make login` 提供一个经 SSH 端口转发访问的导入页 |
| 退出登录 | `./tools/leetgpu/leetgpu logout`：删除本机的 access token 和 refresh token |

凭据只保存在用户目录和钥匙串，不进入仓库。各种登录方式的前置条件和限制见[工具 README](tools/leetgpu/README.md)。

## VS Code

打开仓库后安装提示的推荐扩展（Ruff、Python、Pylance）。

- 保存时自动格式化并排序 import，使用 `.venv` 里固定版本的 ruff。
- `Cmd/Ctrl+Shift+B` 对当前文件执行 `make run`。**Tasks: Run Task** 里还有 submit、history、login、check 和 format。
- CuTe DSL 的补全、悬停和跳转来自 `make setup` 取回的 `.cache/cutlass`。
- 类型检查是关闭的。CuTe 代码被 trace，而不是当作普通 Python 执行，类型检查器会对能通过评测的 kernel 报错。代价是拼错的属性名不会被标出，要靠补全和 `make run` 发现。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `solutions/<id>_<name>/` | `README.md`（题目、结果、评审、版本说明）、`NOTES.md`（学习问答）、`submissions.json`（提交记录）和各版本源码 |
| `.claude/skills/review-solution/` | review（代码评审和性能机会）的流程、写作规则、README 模板和统计脚本（`make stats`） |
| `tools/leetgpu/` | 非官方 LeetGPU CLI：认证、历史查询、运行和提交，无 npm 依赖 |
| `tools/check_records.py` | 校验已记录源码的 SHA-256 |
| `Makefile` | 上面所有命令的入口，人、VS Code 任务和 agent 共用 |
| `pyproject.toml`、`requirements-dev.txt` | ruff 的规则和固定版本 |
| `.vscode/` | 编辑器设置、任务和推荐扩展 |
| `.github/workflows/` | CI：lint、记录校验和工具测试 |
| `AGENTS.md` | coding agents 的规则 |
| `.cache/`（不入库） | `make setup` 取回的 CuTe DSL 源码和官方题目定义，只供阅读 |

## 约定

- Latency 是平台在性能测试上 15 次计时的中位数，包含 `solve` 调用的 Python 启动开销和一次设备同步，不包含编译。编译、全部测试和计时合起来约有 40 秒的时限，超过就是 `timeout`。出处是[官方仓库](https://github.com/AlphaGPU/leetgpu-challenges) README 的 FAQ。
- 每道题都一样的规则（计时方式、配额、怎样才算更快）只写在本文件和 review skill 里，题目的 `README.md` 不重复它们。
- 题目的 `README.md` 只写数据支持的结论：版本之间的差距不超过同一份代码重复提交的波动时，写"未能区分"。有提交的每个 GPU 分开评审，不同 GPU 的 latency 不放在一起比较；只与 CuTe 的成绩比较。
- 已写入 `submissions.json` 的源码文件是冻结的，不格式化也不修改。`make check` 和 CI 会校验它的 SHA-256；要改动就新建版本。
- 评测结果只引用平台记录。H100 与其他 GPU 分开记录，同一份代码的重复测量不算新的优化，percentile 是查询时的快照。
- 平台记录尚未取得，或尚未与源码对应时，明确标为“待同步”。
- 所有变更在独立分支上通过 PR 合入 `main`：使用 Squash merge，合并前解决全部 review 讨论，合并后自动删除分支。`main` 禁止强推和删除，对管理员同样生效；所需批准数为 0。
