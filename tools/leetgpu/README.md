# LeetGPU CLI helper

非官方的 LeetGPU 认证、历史查询和提交工具，使用网站现有 API。支持 macOS 与 Linux（包括 SSH 无桌面机器），需要 **Node.js 22+**，无需 npm 依赖或本机 CUDA。Windows 原生使用未验证。

从仓库根目录运行 `./tools/leetgpu/leetgpu --help`。默认在 PATH 查找 `node`；也可设置 `LEETGPU_NODE=/path/to/node`，或直接执行 `node tools/leetgpu/cli.mjs ...`。工具不依赖 Codex 或其他 agent 的私有目录。

## 认证

先运行：

```sh
./tools/leetgpu/leetgpu status --json
```

返回 `authenticated`、`expiresAt`、`sessionPath`、`renewable`；没有有效会话时另有 `reason`。退出码 0 表示本地保存的会话未过期，或已保存 refresh token、下一条命令会自动续期（`renewable: true`）；1 表示既过期又无法续期，或其他错误；127 表示启动脚本找不到 Node.js 22+。这只做本地检查；服务端仍可能撤销会话。请求返回 401 时重新登录。

### 保持登录：`--persist`

```sh
./tools/leetgpu/leetgpu login --browser --persist   # 独立 Chrome 窗口，登录一次
./tools/leetgpu/leetgpu login --persist             # 或：在无痕窗口登录后粘贴到本地页面
```

默认登录只保存 1 小时有效的 access token。加 `--persist` 后还会保存 refresh token，此后任何需要认证的命令在 access token 过期（或 60 秒内将过期）时先自动续期，再发出请求。它可与 `--browser`、本地导入页（含 `--no-browser`）和 `--stdin` 合用。

- **必须是独占的会话。** 每次续期都会更换 refresh token。若与正在使用的浏览器标签页共用同一会话，后续期的一方会失败并被登出，因此 `--persist` 拒绝与 `--existing-browser` 合用。`--browser` 使用临时 profile 并在结束后删除，天然独占；粘贴方式请在无痕/隐私窗口登录，提交后直接关闭窗口，不要点击 Sign Out。`--persist` 需要含 `refresh_token` 的完整 JSON，不接受单独的 access token。
- **保存位置。** macOS 存入登录钥匙串（服务名 `leetgpu-history`，每个配置目录一个条目），写入时密文只经标准输入传给 `/usr/bin/security`，不出现在命令参数中。其他平台默认不支持；可显式设置 `LEETGPU_SECRET_STORE=file`，改存到配置目录下 0600 的明文文件 `refresh-token`，请自行权衡。钥匙串保护的是静态数据（磁盘、备份、误读文件）；以同一用户身份运行的程序仍可通过 `security` 读取。
- **续期。** 请求 `POST https://yhdtysacdkqoquvkdwdd.supabase.co/auth/v1/token?grant_type=refresh_token`，使用网站公开的 anon key，refresh token 只在请求正文中。先保存新的 refresh token，再保存 access token。并发的命令通过配置目录下的锁文件串行续期，不会重复消耗同一个 refresh token。
- **失效。** 服务端拒绝（HTTP 400/401/403）时，工具删除已失效的 refresh token 并提示重新登录；网络错误、429 或 5xx 时保留它，下次再试。refresh token 的有效期由 LeetGPU 的设置决定，未经核实。
- **关闭。** `logout` 同时删除 access token 和 refresh token；之后做一次不带 `--persist` 的登录也会删除先前保存的 refresh token。两者都不会撤销服务端会话。

2026-11-19 已用真实账号验证：`login --browser --persist` 经 Google 登录成功，refresh token 写入钥匙串；随后一次续期成功，服务端返回新的 access token，钥匙串里的 refresh token 被替换，用新 token 的 API 请求正常。对无效 refresh token 的拒绝（HTTP 400）也已验证。refresh token 最长能用多久仍未知。

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

`login --browser` 用一个临时 profile 的独立 Chrome/Chromium 完成登录，可用 `LEETGPU_HISTORY_CHROME` 指定浏览器路径。它分两步：

1. **登录窗口。** 以普通方式启动 Chrome，不带任何调试或自动化参数。Google 会拒绝在带 remote debugging 的浏览器里登录（提示 "This browser or app may not be secure"），所以这一步不能带。
2. **读取会话。** 检测到该 profile 的本地存储里写入了 LeetGPU 会话后，工具关闭登录窗口，再以无窗口（headless）方式重新打开同一个 profile，这次带上只监听回环地址的调试端口，在 `https://leetgpu.com/robots.txt` 这个不运行网站脚本的同源页面上读取会话。

登录完成后窗口通常在几秒内自动关闭。如果一直不关，手动退出那个 Chrome；在交互式终端里也可以按回车。结束后删除临时 profile，所以浏览器里不会留下这次会话。

2026-11-19 已用真实账号验证：改成两步之前，Google 在登录窗口里提示 "This browser or app may not be secure" 并拒绝登录；改成两步之后，同一账号经 Google 登录成功，会话被正确读回。其他登录提供商未测试。窗口仍被拒绝时，使用上面的普通浏览器粘贴方式。

所在网络无法直接访问登录提供商（例如 Google）时，让这个窗口走代理：

```sh
./tools/leetgpu/leetgpu login --browser --persist --proxy http://proxy.example:3128
make login PROXY=http://proxy.example:3128    # 等价写法
```

`--proxy` 接受 `http`、`https`、`socks4`、`socks5` 代理，写成 `协议://主机:端口`，不接受带用户名、密码或路径的地址。它只能与 `--browser` 合用，并且只影响上面两步启动的 Chrome：其中的所有请求（包括 LeetGPU 和登录提供商）都经过代理，CLI 自己的 API、续期和提交请求仍然直连。2026-11-19 已在本机用 headless Chrome 和一个本地监听端口验证，请求确实经由该代理发出；经真实代理完成一次登录尚未验证。

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

默认会话为 `~/.config/leetgpu-history/session.json`（文件 `0600`，目录 `0700`），其中不保存密码或 refresh token。access token 的有效期为 1 小时（2026-11-19 核对本机会话的 `exp - iat` 为 3600 秒）；未使用 `--persist` 时过期后需要重新认证。仓库根目录的 `make run`/`make submit` 只在会话既过期又无法续期时调用 `login`。`logout` 删除当前机器的本地 access token 和已保存的 refresh token，不撤销服务端会话或其他机器上的副本。

| 设置 | 默认值 / 用途 |
|---|---|
| `LEETGPU_NODE` | `node`；指定 Node.js 可执行文件 |
| `LEETGPU_CONFIG_DIR` | `${XDG_CONFIG_HOME:-~/.config}/leetgpu-history`；会话目录 |
| `LEETGPU_DATA_DIR` | `${XDG_DATA_HOME:-~/.local/share}/leetgpu-history`；回执目录的父目录 |
| `LEETGPU_HISTORY_CONFIG_DIR` | 旧配置目录变量；仅在没有 `LEETGPU_CONFIG_DIR` 时使用 |
| `LEETGPU_HISTORY_CHROME` | 可选，独立浏览器模式的 Chrome/Chromium 可执行路径 |
| `LEETGPU_SECRET_STORE` | `--persist` 的 refresh token 存放处：`keychain`（macOS 默认）、`file`（0600 明文文件）、`none`（其他平台默认，禁用 `--persist`） |

目录名沿用 `leetgpu-history`，兼容早期本地工具。配置覆盖也应指向仓库之外。凭据、回执和导出的原始响应不应提交到 Git。

## 历史、代码和提交

```sh
./tools/leetgpu/leetgpu challenges --json
./tools/leetgpu/leetgpu gpus --json
./tools/leetgpu/leetgpu leaderboard --challenge 1_vector_add --gpu H100
./tools/leetgpu/leetgpu submissions latest --json
./tools/leetgpu/leetgpu submissions list --challenge 1 --language cute --gpu H100 --json
./tools/leetgpu/leetgpu submissions code SUBMISSION_ID --json
./tools/leetgpu/leetgpu submissions code SUBMISSION_ID --file solution.py --output /tmp/saved-solution.py
./tools/leetgpu/leetgpu submissions run --language cute --gpu H100 --file solutions/1_vector_add/v2_vectorized.py
./tools/leetgpu/leetgpu submissions submit --challenge 1 --language cute --gpu H100 --file solutions/1_vector_add/v2_vectorized.py --json
./tools/leetgpu/leetgpu logout
```

- `--challenge` 接受三种写法：数字 ID；官方仓库的目录名 `<id>_<name>`（如 `1_vector_add`，只使用数字前缀，不比较名称）；标题 slug（如 `matrix-multiplication`，与网站 URL 一致）。`run`/`submit` 省略它时，从 `--file` 向上最多三级目录，取第一个符合后两种写法的目录名；找不到或有歧义时报错，不会猜测。后两种写法需要一次公开的 `challenges` 请求，用来确认 ID 存在并取得标题，解析结果打印在 stderr。本仓库的题目目录采用官方目录名。
- `run` 对应网站的 Run 按钮：与 `submit` 使用同一连接和同一 private 载荷，只是 `action` 为 `run`，不写回执，退出码规则相同。它只执行题面的示例测试，用来看代码能否编译并跑通一个小例子；通过 `run` 不代表能通过 `submit` 的全部功能测试。它仍占用平台 GPU。
- 不加 `--json` 时，`run`/`submit` 把平台输出（编译错误、测试结果）去除 ANSI 和控制字符后实时打印到 stderr，状态变化时打印一行 `[status]`；加 `--json` 时 stdout 只有 JSON，平台输出在 `events` 中。

- `leaderboard` 列出平台上各语言在指定 GPU 上的前几名成绩（毫秒），最快的语言在前，`*` 标出自己的成绩。省略 `--language` 时查询该 GPU 支持的全部语言。它的 `--json` 与其他读取命令不同，不是原始响应：只输出名次、时间和 `mine` 标记，不包含任何人的名字。2026-11-19 已对真实服务验证。
- `latest` 返回账户最近提交，不能按 GPU 筛选；查看 H100 使用显式带 challenge、language、gpu 的 `list`。`cute` 表示 CuTe DSL；用 `gpus` 查看当前 GPU/语言兼容性。
- 读取命令的 `--json` 保留 API 原始结构和元数据，可能含账户字段；提交到仓库前应筛选必要字段。表格显示的 `metrics.executionTime` 单位为毫秒，percentile 是查询时排名。
- 不保证 API 返回完整历史；没有实现未经确认的分页或全账户遍历。空结果只表示当前过滤条件下服务端没有返回记录。
- `code` 默认输出单个文件原文，不执行它；多文件需 `--file` 精确选取或 `--json` 取得完整数据。`--output` 只写显式指定的新文件，拒绝覆盖已有文件。
- `submit` 从 `--file` 读取一个源码文件（CuTe 上传名为 `solution.py`），固定 **private**，显式选择 GPU，发起一次请求并等待终态，不自动重试。先确认用户已要求提交/评测。成功后用 `list` 回读 metrics、用 `code` 核对源码。

提交回执保存于数据目录下 `receipts/`，包括请求 ID、服务端 ID、源码路径/SHA-256、状态及去除 token 的事件。`submit --json` 输出最终状态、事件和回执路径。中断信号会尝试保存最新已知 ID 和不确定状态；强制终止或机器掉线可能只留下初始回执。若没有确认终态，先按 GPU、时间、源码和 ID 查询历史，避免重复提交。

| 退出码 | 含义 |
|---|---|
| `0` | 命令成功；run/submit 为平台确认 success |
| `1` | 参数、认证、网络或其他错误；submit 可能已经被服务端接收，需看错误和回执 |
| `2` | run/submit 已收到平台确认的非 success 终态，例如编译/测试失败 |
| `127` | 启动脚本找不到 Node.js 22+；安装 Node 或设置 `LEETGPU_NODE` |

## API 与实现边界

HTTP 固定访问 `https://api.leetgpu.com`，GET 不跟随重定向，20 秒超时；错误不包含响应正文或 token。

| 命令 | 接口 |
|---|---|
| `challenges` | `GET /api/v1/challenges` |
| `gpus` | `GET /api/v1/settings/accelerated/accelerators` |
| `leaderboard` | `GET /api/v1/challenges/<id>/leaderboard?language=…&accelerator=…`，每种语言一次；另用 `GET /api/v1/me/display-name` 标出自己的成绩 |
| `submissions latest` | `GET /api/v1/submissions/latest` |
| `submissions list` | `GET /api/v1/submissions?challengeId=…&language=…&accelerator=…` |
| `submissions code` | `GET /api/v1/submissions/<id>/code` |
| `submissions submit` | `wss://api.leetgpu.com/api/v1/ws/submit`，`action: "submit"`；等待上限 180 秒 |
| `submissions run` | 同一 WebSocket，`action: "run"`；等待上限 180 秒 |

`run` 的消息格式取自[社区 VS Code 扩展的传输实现](https://github.com/GlowLED/LeetGPU-for-VsCode/blob/cdc7930f27a669680ef5eb394910841a3b249ce7/src/services/submissionTransport.ts#L20-L38)，平台输出字段（`type: "stdout"`、`output`）取自 2026-10-19 一次真实 submit 的回执。2026-11-19 已对真实服务验证：`make run` 在 H100 上运行 `solutions/1_vector_add/v2_vectorized.py`，约 17 秒返回 `success`。平台只执行题面的示例测试（本题为 `N = 4`），并把输入和 "Test passed" 打印出来；不跑功能测试和性能测试，所以没有 latency。这次 run 没有出现在提交历史里，`latest` 也没有变化。同一天，`list`（含按官方目录名指定 challenge）、`code` 和 `latest` 也已用续期后的会话执行成功。

这些是网站使用的接口，可能改变，不是官方 CLI 扩展。实现参考已核对的网站请求行为；接口调研资料（2026-10-19）：[LeetGPU web client](https://leetgpu.com/)、[社区 API 实现](https://github.com/GlowLED/LeetGPU-for-VsCode/blob/cdc7930f27a669680ef5eb394910841a3b249ce7/src/services/apiClient.ts#L53-L64)、[官方认证 feature request](https://github.com/AlphaGPU/leetgpu-challenges/issues/304)。

## 测试

```sh
make test   # sh -n tools/leetgpu/leetgpu && node --test tools/leetgpu/*.test.mjs
```

测试使用假 token、模拟 HTTP/WebSocket、临时配置目录和本地回环接收服务，不访问账号或消耗 GPU。CI 在 macOS/Linux 与 Node.js 22/24 上运行工具测试，另有一个任务用固定版本的 ruff 检查解法（含 Python 语法）并校验已记录源码的 SHA-256；不将这些静态检查当作 CuTe 编译或 GPU 测试。
