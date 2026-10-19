# LeetGPU CLI helper

非官方的 LeetGPU 认证、历史查询和提交工具，使用网站现有 API。支持 macOS 与 Linux（包括 SSH 无桌面机器），需要 **Node.js 22+**，无需 npm 依赖或本机 CUDA。Windows 原生使用未验证。

从仓库根目录运行 `./tools/leetgpu/leetgpu --help`。默认在 PATH 查找 `node`；也可设置 `LEETGPU_NODE=/path/to/node`，或直接执行 `node tools/leetgpu/cli.mjs ...`。工具不依赖 Codex 或其他 agent 的私有目录。

## 认证

先运行：

```sh
./tools/leetgpu/leetgpu status --json
```

返回 `authenticated`、`expiresAt`、`sessionPath`；没有有效会话时另有 `reason`。退出码 0 表示本地保存的会话未过期，1 表示缺失/过期或其他错误。这只检查本地有效期；服务端仍可能撤销会话。请求返回 401 时重新登录。

### 已登录的 macOS Chrome：自动读取

在普通 Chrome 中登录 [LeetGPU](https://leetgpu.com/)，保留标签页，然后运行：

```sh
./tools/leetgpu/leetgpu login --existing-browser
```

Chrome 必须开启 **View → Developer → Allow JavaScript from Apple Events**（显示 → 开发者 → 允许来自 Apple 事件的 JavaScript），macOS 可能另行请求 Chrome 自动化权限。工具不会修改这些设置，完成后可以关闭该选项。

此模式只访问已打开且 origin 为 `https://leetgpu.com` 的标签页，读取已知的会话 localStorage 键，不扫描浏览器配置文件、cookies 或剪贴板。

### 普通浏览器：本地导入页

```sh
./tools/leetgpu/leetgpu login
```

工具打开临时本地页面。使用普通浏览器登录 LeetGPU，按页面说明从开发者工具 Application → Local Storage → `https://leetgpu.com` 复制 `sb-yhdtysacdkqoquvkdwdd-auth-token` 的完整 Value，粘贴到**本地页面**。支持会话 JSON 或 access-token JWT，不需要把 token 发给 agent。

接收服务仅监听 `127.0.0.1`，使用随机 URL，检查 Host/Origin，10 分钟超时。接收后关闭；最终是否成功保存以终端提示为准。打开浏览器失败时服务继续等待，可手动打开打印的地址。

`login --browser` 是可选的独立 Chrome/Chromium 临时 profile 自动捕获模式，可用 `LEETGPU_HISTORY_CHROME` 指定浏览器路径。Google 等登录提供商可能拒绝带 remote debugging 的浏览器；遇到该情况使用上面的普通浏览器路径。此模式只读取 LeetGPU 会话，结束后关闭专用窗口并删除临时 profile。

### 远程机器：SSH 转发导入页

1. 在远程仓库内启动接收页：

   ```sh
   ./tools/leetgpu/leetgpu login --no-browser --port 8765
   ```

2. 在本机另一个终端建立转发，将 `user@remote-host` 换成实际目标：

   ```sh
   ssh -N -L 127.0.0.1:8765:127.0.0.1:8765 user@remote-host
   ```

3. 用本机浏览器打开远程命令打印的完整 `http://127.0.0.1:8765/<随机路径>`，按页面说明导入。终端显示成功后可关闭转发。

本地与远程端口必须相同，URL 必须保留 `127.0.0.1` 和随机路径，否则 Host/Origin 检查会拒绝请求。端口占用时，两处同时换用其他端口。默认 `--port 0` 可自动分配端口，命令也会打印相应转发示例。无需对外开放端口，也不需要远程机器有浏览器。

### 远程机器：通过 SSH 标准输入导入已有会话

已在本机用上述任一方式认证后，可以将该短期会话传到自己指定的远程机器。以下示例假设本机未自定义配置路径、远程 checkout 位于 `~/src/leetgpu-cutedsl`：

```sh
ssh user@remote-host 'cd ~/src/leetgpu-cutedsl && ./tools/leetgpu/leetgpu login --stdin' \
  < "${XDG_CONFIG_HOME:-$HOME/.config}/leetgpu-history/session.json"
```

输入不会出现在命令参数或终端输出中；远程工具重新验证有效期，并只保存 access token 和过期时间。也可用 `login --stdin < /path/to/session.json` 导入预先提供的文件。标准输入拒绝交互式 TTY、无效/过期会话及超过 100000 字节的内容。不要把 token 写成命令参数、环境变量值、聊天消息或 PR 文本。

### 保存位置与过期

默认会话为 `~/.config/leetgpu-history/session.json`（文件 `0600`，目录 `0700`），不保存密码或 refresh token。令牌过期后需要重新认证；工具不会自动续期。`logout` 删除当前机器的本地 token，不撤销服务端会话或其他机器上的副本。

| 设置 | 默认值 / 用途 |
|---|---|
| `LEETGPU_NODE` | `node`；指定 Node.js 可执行文件 |
| `LEETGPU_CONFIG_DIR` | `${XDG_CONFIG_HOME:-~/.config}/leetgpu-history`；会话目录 |
| `LEETGPU_DATA_DIR` | `${XDG_DATA_HOME:-~/.local/share}/leetgpu-history`；回执目录的父目录 |
| `LEETGPU_HISTORY_CONFIG_DIR` | 旧配置目录变量；仅在没有 `LEETGPU_CONFIG_DIR` 时使用 |
| `LEETGPU_HISTORY_CHROME` | 可选，独立浏览器模式的 Chrome/Chromium 可执行路径 |

目录名沿用 `leetgpu-history`，兼容早期本地工具。配置覆盖也应指向仓库之外。凭据、回执和导出的原始响应不应提交到 Git。

## 历史、代码和提交

```sh
./tools/leetgpu/leetgpu challenges --json
./tools/leetgpu/leetgpu gpus --json
./tools/leetgpu/leetgpu submissions latest --json
./tools/leetgpu/leetgpu submissions list --challenge 1 --language cute --gpu H100 --json
./tools/leetgpu/leetgpu submissions code SUBMISSION_ID --json
./tools/leetgpu/leetgpu submissions code SUBMISSION_ID --file solution.py --output /tmp/saved-solution.py
./tools/leetgpu/leetgpu submissions submit --challenge 1 --language cute --gpu H100 --file solutions/vector-addition/v2_vectorized.py --json
./tools/leetgpu/leetgpu logout
```

- `latest` 返回账户最近提交，不能按 GPU 筛选；查看 H100 使用显式带 challenge、language、gpu 的 `list`。`cute` 表示 CuTe DSL；用 `gpus` 查看当前 GPU/语言兼容性。
- 读取命令的 `--json` 保留 API 原始结构和元数据，可能含账户字段；提交到仓库前应筛选必要字段。表格显示的 `metrics.executionTime` 单位为毫秒，percentile 是查询时排名。
- 不保证 API 返回完整历史；没有实现未经确认的分页或全账户遍历。空结果只表示当前过滤条件下服务端没有返回记录。
- `code` 默认输出单个文件原文，不执行它；多文件需 `--file` 精确选取或 `--json` 取得完整数据。`--output` 只写显式指定的新文件，拒绝覆盖已有文件。
- `submit` 从 `--file` 读取一个源码文件（CuTe 上传名为 `solution.py`），固定 **private**，显式选择 GPU，发起一次请求并等待终态，不自动重试。先确认用户已要求提交/评测。成功后用 `list` 回读 metrics、用 `code` 核对源码。

提交回执保存于数据目录下 `receipts/`，包括请求 ID、服务端 ID、源码路径/SHA-256、状态及去除 token 的事件。`submit --json` 输出最终状态、事件和回执路径。中断信号会尝试保存最新已知 ID 和不确定状态；强制终止或机器掉线可能只留下初始回执。若没有确认终态，先按 GPU、时间、源码和 ID 查询历史，避免重复提交。

| 退出码 | 含义 |
|---|---|
| `0` | 命令成功；submit 为平台确认 success |
| `1` | 参数、认证、网络或其他错误；submit 可能已经被服务端接收，需看错误和回执 |
| `2` | submit 已收到平台确认的非 success 终态，例如编译/测试失败 |

## API 与实现边界

HTTP 固定访问 `https://api.leetgpu.com`，GET 不跟随重定向，20 秒超时；错误不包含响应正文或 token。

| 命令 | 接口 |
|---|---|
| `challenges` | `GET /api/v1/challenges` |
| `gpus` | `GET /api/v1/settings/accelerated/accelerators` |
| `submissions latest` | `GET /api/v1/submissions/latest` |
| `submissions list` | `GET /api/v1/submissions?challengeId=…&language=…&accelerator=…` |
| `submissions code` | `GET /api/v1/submissions/<id>/code` |
| `submissions submit` | `wss://api.leetgpu.com/api/v1/ws/submit`；等待上限 180 秒 |

这些是网站使用的接口，可能改变，不是官方 CLI 扩展。实现参考已核对的网站请求行为；接口调研资料（2026-10-19）：[LeetGPU web client](https://leetgpu.com/)、[社区 API 实现](https://github.com/GlowLED/LeetGPU-for-VsCode/blob/cdc7930f27a669680ef5eb394910841a3b249ce7/src/services/apiClient.ts#L53-L64)、[官方认证 feature request](https://github.com/AlphaGPU/leetgpu-challenges/issues/304)。

## 测试

```sh
node --test tools/leetgpu/*.test.mjs
sh -n tools/leetgpu/leetgpu
```

测试使用假 token、模拟 HTTP/WebSocket、临时配置目录和本地回环接收服务，不访问账号或消耗 GPU。CI 在 macOS/Linux 与 Node.js 22/24 上运行，也检查解法 Python 语法；不将语法检查当作 CuTe 编译或 GPU 测试。
