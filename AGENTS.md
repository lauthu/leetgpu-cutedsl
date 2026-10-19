# Coding agents

本仓库保存 LeetGPU 的 CuTe DSL 解法、学习笔记和可核验的提交记录。当前重点 GPU 是 **H100**，语言参数为 `cute`。先阅读根目录 README 和目标 challenge 的 README。

## 工具与认证

使用仓库自带的 `./tools/leetgpu/leetgpu`，需要 Node.js 22+，无需 npm install。实现位于 `tools/leetgpu/`，不依赖某个 coding agent 的私有运行时。详细认证、SSH 和 API 说明见 [工具 README](tools/leetgpu/README.md)。

1. 先运行 `./tools/leetgpu/leetgpu status --json`。退出码 0 表示本机会话未过期；退出码 1 时按错误原因处理。这个检查不访问服务端。
2. macOS 上用户已在普通 Chrome 登录时，使用 `login --existing-browser`。Chrome 需要允许 Apple Events JavaScript；若未开启，说明具体设置并等待用户操作，不擅自更改浏览器权限。
3. Linux/SSH 机器使用 `login --no-browser --port 8765`，让用户在本机通过同端口 SSH 转发访问打印的临时 URL；或者按工具 README 用 `login --stdin` 经 SSH 导入本机已授权的短期会话。认证界面上的用户操作仍需用户完成。
4. 不让用户把密码、token 或完整会话粘贴到聊天、issue、PR 或命令参数中。不打印 session 文件，不提交凭据、原始认证数据或个人回执。只在用户指定的机器间传递其授权会话。
5. 工具只保存 access token 和过期时间；会话过期后重新登录。API 返回 401 时同样重新认证，不循环重试失效凭据。

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
./tools/leetgpu/leetgpu submissions submit --challenge 1 --language cute --gpu H100 --file solutions/vector-addition/v2_vectorized.py --json
```

提交固定为 private，并保存本地回执。退出码 2 表示平台确认的失败结果；退出码 1 需检查错误/回执。连接中断或超时后，先查询历史核对回执中的 ID、源码哈希和时间，不能直接重试，因为服务端可能已接收。成功后回读历史取得 metrics，再通过 code 接口核对源码。

## 维护记录与验证

- 每个解法版本保留独立源码，新增迭代不覆盖旧版本。在 challenge README 中说明变化、测试与实际结果。
- 提交元数据只保存必要字段：ID、GPU、语言、状态、时间、metrics、版本和源码匹配证据。保留原始 UTC 时间，显示时间注明时区。源码的精确 SHA-256 与经过归一化的匹配要区分。
- H100 与其他 GPU 的结果分开记录。相同代码重新测量不记作新优化；percentile 是查询快照，不是永久固定的排名。
- 工具修改运行 `node --test tools/leetgpu/*.test.mjs`；这些测试使用假会话、模拟 API 和临时目录，不需要登录或消耗 GPU。不要为测试调用真实 submit。
- 本地没有 CUDA 时，只报告实际完成的语法/静态检查；GPU 正确性和性能引用平台记录，不声称本地验证通过。
- 所有变更在 `codex/` 或用户指定分支上通过 PR 合入 `main`；保留用户未提交的修改。提交前检查 diff，避免凭据和机器专用路径进入仓库。
