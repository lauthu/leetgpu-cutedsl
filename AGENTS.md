# Coding agents

本仓库保存 LeetGPU 的 CuTe DSL 解法、学习笔记和可核验的提交记录。当前重点 GPU 是 **H100**，语言参数为 `cute`。先阅读根目录 README 和目标 challenge 的 README。

## 工具与认证

使用仓库自带的 `./tools/leetgpu/leetgpu`，需要 Node.js 22+，无需 npm install。实现位于 `tools/leetgpu/`，不依赖某个 coding agent 的私有运行时。详细认证、SSH 和 API 说明见 [工具 README](tools/leetgpu/README.md)。

1. 先运行 `./tools/leetgpu/leetgpu status --json`。退出码 0 表示本机会话未过期，或已保存 refresh token、下一条命令会自动续期（`renewable: true`），都不需要登录；退出码 1 时按错误原因处理；退出码 127 表示找不到 Node.js 22+，这不是会话问题，应安装 Node 或设置 `LEETGPU_NODE`。这个检查不访问服务端。
2. macOS 的默认方式是保持登录：`login --browser --persist`（即 `make login`）打开独立的 Chrome 窗口，用户登录一次后，refresh token 存入 macOS 钥匙串，之后的命令自动续期。网络无法直接访问登录提供商时加 `--proxy http://host:port`（`make login PROXY=...`），它只影响这个登录窗口；代理地址由用户提供，不写入仓库。登录提供商拒绝该窗口时，改用 `login --persist`（在无痕窗口登录后粘贴到本地页面），或只用一小时的 `login --existing-browser`（读取已登录的普通 Chrome 标签页，需要允许 Apple Events JavaScript；若未开启，说明具体设置并等待用户操作，不擅自更改浏览器权限）。`--persist` 必须使用独占的会话，不能与 `--existing-browser` 合用：每次续期都会更换 refresh token，与正在使用的标签页共用会话会使其中一方被登出。
3. Linux/SSH 机器使用 `login --no-browser --port 8765`，让用户在本机通过同端口 SSH 转发访问打印的临时 URL；或者按工具 README 用 `login --stdin` 经 SSH 导入本机已授权的短期会话。认证界面上的用户操作仍需用户完成。
4. 不让用户把密码、token 或完整会话粘贴到聊天、issue、PR 或命令参数中。不打印 session 文件，不提交凭据、原始认证数据或个人回执。只在用户指定的机器间传递其授权会话。
5. 默认只保存 access token 和过期时间，有效期 1 小时。只有用户明确选择 `--persist` 时才保存 refresh token：macOS 存入钥匙串，其他平台需显式设置 `LEETGPU_SECRET_STORE=file`（0600 明文文件）。不读取、不打印、不转存钥匙串条目或该文件。续期被服务端拒绝时工具会删除失效的 refresh token 并要求重新登录；API 返回 401 时同样重新认证，不循环重试失效凭据。

## 本地开发流程

仓库根目录的 `Makefile` 是人、VS Code 任务和 coding agent 共用的入口，`make help` 列出全部目标。

```sh
make setup                                         # .venv（固定版本的 ruff）+ CuTe DSL 源码 + 官方题目定义，后两者只供阅读
make fmt                                           # 排序 import 并格式化
make check                                         # lint + 提交记录校验 + 工具测试，不需要登录或 GPU
make run FILE=solutions/<challenge>/<version>.py     # 在平台上运行，不产生提交
make submit FILE=solutions/<challenge>/<version>.py  # private 提交并写回执
```

- `run`/`submit` 先做本地预检（语法错误、未定义名称），失败时不联网、不登录；随后只在会话既过期又无法自动续期时按平台默认方式登录，再调用 CLI。默认 `GPU=H100 LANGUAGE=cute`，可在命令行覆盖。
- 题目目录与官方仓库的目录同名：`solutions/<id>_<name>/`，例如 `solutions/1_vector_add/`。challenge ID 取自数字前缀，不做名称匹配；CLI 会核对该 ID 存在，并在 stderr 打印 ID 和标题，提交前确认它是目标题目。目录名不符合这个形式时，用 `CHALLENGE=<ID>` 指定。
- 官方题目定义在 `.cache/leetgpu-challenges/challenges/<难度>/<id>_<name>/`（`make challenges` 取回或更新）：`challenge.py` 给出容差、`solve` 签名和性能测试规模，`challenge.html` 是题面，`starter/starter.cute.py` 是起始代码。它的许可禁止再分发，只引用事实，不把内容复制进本仓库。
- `run` 对应网站的 Run 按钮，用来看编译错误和样例结果；`submit` 才产生带 metrics 的提交记录。`run` 同样占用平台 GPU，只在用户要求运行或评测时执行。
- macOS 没有 CUDA，CuTe DSL 也没有 macOS 的 wheel，本地无法编译或运行 kernel。`make setup` 取回的 `.cache/cutlass` 只供阅读和编辑器补全，不要尝试导入执行。
- ruff 规则见 `pyproject.toml`；Pylance 配置在 `.vscode/settings.json`，不要在 `pyproject.toml` 里加 `[tool.pyright]`（Pylance 不会实时重读它，而且该表一出现就会把类型检查从默认的关闭变为开启）。类型检查有意关闭：CuTe 代码是被 trace 的，类型检查器会对平台上通过的 kernel 报出大量误报。

## 查询与提交

```sh
./tools/leetgpu/leetgpu challenges --json
./tools/leetgpu/leetgpu gpus --json
./tools/leetgpu/leetgpu submissions list --challenge 1 --language cute --gpu H100 --json
./tools/leetgpu/leetgpu submissions code SUBMISSION_ID --file solution.py --output /tmp/leetgpu-source.py
```

`latest` 是账户最近一次提交，可能来自其他 GPU；H100 历史必须使用显式过滤的 `list`。返回记录不保证覆盖完整历史，空列表也不证明账号没有历史。

用户要求提交/评测时，在其授权范围内执行，无需对每次已授权操作重复确认。只要求调研、查询或改代码的任务不自动触发 GPU 提交。示例：

```sh
./tools/leetgpu/leetgpu submissions run --language cute --gpu H100 --file solutions/1_vector_add/v2_vectorized.py --json
./tools/leetgpu/leetgpu submissions submit --challenge 1 --language cute --gpu H100 --file solutions/1_vector_add/v2_vectorized.py --json
```

`--challenge` 可以是数字 ID、官方目录名（`1_vector_add`）或标题 slug；`run`/`submit` 省略它时按源码所在目录推断。需要机器可读结果时加 `--json`；不加时平台输出（编译错误、测试结果）会实时打印到 stderr。

提交固定为 private，并保存本地回执。退出码 2 表示平台确认的失败结果；退出码 1 需检查错误/回执。连接中断或超时后，先查询历史核对回执中的 ID、源码哈希和时间，不能直接重试，因为服务端可能已接收。成功后回读历史取得 metrics，再通过 code 接口核对源码。

## Review

用户要求 review、复盘或分析某道题、某个版本或其提交时，按 [.claude/skills/review-solution/SKILL.md](.claude/skills/review-solution/SKILL.md) 执行：拉取提交历史，更新 `submissions.json`，用 `make stats` 得到数字，再按同目录的 `assets/README.template.md` 重写该题的 `README.md`。不支持 skill 的 agent 直接阅读该文件。Review 不修改版本源码，也不触发 run 或 submit。

## 维护记录与验证

- 每个解法版本保留独立源码，新增迭代不覆盖旧版本。题目的 `README.md` 按 review 模板写题目、结果、分析和版本说明；学习问答放在同目录的 `NOTES.md`。
- 已写入 `submissions.json` 的源码文件是冻结的：不格式化、不修 lint、不改注释，否则 `make records`（CI 也会运行）会因 SHA-256 不符而失败。需要改动时新建版本文件；冻结文件里的 lint 问题在 `pyproject.toml` 的 per-file-ignores 中忽略。
- 提交元数据只保存必要字段：ID、GPU、语言、状态、时间、metrics、版本和源码匹配证据。保留原始 UTC 时间，显示时间注明时区。源码的精确 SHA-256 与经过归一化的匹配要区分。
- H100 与其他 GPU 的结果分开记录。相同代码重新测量不记作新优化；percentile 是查询快照，不是永久固定的排名。
- 工具修改运行 `make test`（即 `node --test tools/leetgpu/*.test.mjs`）；这些测试使用假会话、模拟 API 和临时目录，不需要登录或消耗 GPU。不要为测试调用真实 run 或 submit。
- 本地没有 CUDA 时，只报告实际完成的语法/静态检查；GPU 正确性和性能引用平台记录，不声称本地验证通过。
- 所有变更在 `codex/` 或用户指定分支上通过 PR 合入 `main`；保留用户未提交的修改。提交前检查 diff，避免凭据和机器专用路径进入仓库。
