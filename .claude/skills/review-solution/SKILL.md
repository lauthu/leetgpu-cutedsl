---
name: review-solution
description: Review a LeetGPU challenge in this repository for code excellence and new performance opportunities, on every GPU that has submissions (each reviewed on its own) and within CuTe DSL only, without spending any submissions. Pull its submission history and the CuTe leaderboard, review each version's code, rank what could make it faster, and rewrite solutions/<challenge>/README.md from a fixed template. Use this whenever the user asks to review, 复盘, 回顾, 总结 or analyze a challenge, a solution, a version or its submissions; asks whether the code is good, how a version performed, whether a change helped, or what to optimize next; or wants a solution README written or updated after new submissions, even if they never say "review".
---

# Review a solution

这个仓库用来学习。一次 review 要回答四个问题：

1. **代码好不好？** 正确、清楚、没有多余的东西吗？
2. **离最好成绩多远？** 在每个有提交的 GPU 上，与 CuTe 的最好成绩差多少？
3. **还有哪些性能机会？** 哪个最值得先试，怎么验证？
4. **现有数字支持什么结论？**

只有历史统计、没有代码评审和性能机会的 review 是不完整的。即使没有新的提交，前三个问题也要重新回答。

## 三条边界

1. **不花配额。** LeetGPU 每天的提交有配额。Review 期间不 `run`、不 `submit`，只读平台、只改文档和记录，也不修改已有的版本源码。建议里也不出现只为测量而做的重复提交：每一次提交都应该是一个值得一试的新版本。
2. **每个 GPU 分开评审。** 评审范围是提交历史里出现过的全部 GPU，从 `submissions.json` 和平台历史里取，不靠假设。每个 GPU 有自己的结果表、结论和性能机会。不同 GPU 的 latency 不放在一起比较，也不把"换一个 GPU"当作优化。一个版本可能只在某一代 GPU 上能编译，评审时写明它属于哪个 GPU。
3. **只在 CuTe DSL 内比较。** 这个仓库是用来练习 CuTe DSL 的，参照是 CuTe 在同一个 GPU 上的排行榜。不与 CUDA 或其他语言比较，也不把"换一种语言"当作机会。

## 流程

### 1. 确定题目和评审范围

题目目录是 `solutions/<id>_<name>/`，与官方仓库的目录同名，数字前缀就是 challenge ID。

```sh
make stats CHALLENGE=<id>_<name>
```

开头两行给出各 GPU 的提交次数和评审范围：有提交记录的全部 GPU，提交最多的排在前面。之后每个 GPU 一段统计。用户只要某一个 GPU 时，用 `make stats CHALLENGE=<id>_<name> SKU=<GPU>`。平台上有、而 `submissions.json` 里还没有的 GPU，要到第 2 步同步之后才会出现，所以同步完再运行一次。

官方定义在 `.cache/leetgpu-challenges/challenges/<难度>/<id>_<name>/`（不存在时运行 `make challenges`）。从这里取事实，不要凭记忆写：

- `challenge.py`：标题 `name`、`atol`、`rtol`、`get_solve_signature`，以及示例、功能和性能测试各自的规模与输入分布。
- `challenge.html`：题面和约束。

官方仓库的许可禁止再分发，所以只引用事实（规模、容差、约束），不把题面或测试代码复制进本仓库。

### 2. 同步提交历史

```sh
./tools/leetgpu/leetgpu status --json
./tools/leetgpu/leetgpu submissions list --challenge <id>_<name> --language cute --gpu <GPU> --json   # 每个 GPU 一次
./tools/leetgpu/leetgpu submissions code <提交 ID> --output <仓库外的临时目录>/<提交 ID>.py
```

- 退出码和登录方式见 AGENTS.md。无法登录时不要卡住：用 `submissions.json` 里已有的记录继续，并在汇报里说明历史和排行榜没有刷新。
- 原始响应含账户字段，只放在仓库外的临时目录。
- 对 `submissions.json` 里还没有的每个提交，取回代码并与各版本文件比较 SHA-256：
  - 相同：`sourceVersion` 填该文件名，`sourceMatch.exactBytes` 为 `true`。
  - 不同：依次尝试"只差结尾换行"和"Python AST 相同"，把实际采用的规则写进 `sourceMatch.method`，`exactBytes` 为 `false`。
  - 都不匹配：说明平台上有仓库没有的代码。`sourceVersion` 填 `null`，问用户是否把它存为新版本，不要猜。
- 同一个 GPU 的所有记录，`metrics` 取自同一次 `list` 响应。Latency 不会变，percentile 和它的排名基数 `rankedCount` 会变，所以整组一起更新，并记下取数日期。
- 对评审范围内的每个 GPU 各查询一次，再加上用户提到的、还没有记录的 GPU。`gpus --json` 列出平台支持 `cute` 的 GPU。
- `list` 不保证返回完整历史；空结果不代表没有提交。

`submissions.json` 的每条记录只保留这些字段：

```json
{
  "id": "<完整提交 ID>",
  "language": "cute",
  "accelerator": "H100",
  "status": "success",
  "createdAt": "<平台返回的 UTC 时间>",
  "challengeId": 1,
  "metrics": { "executionTime": 0.23426, "percentile": 97.22, "rankedCount": 108 },
  "sourceVersion": "v2_vectorized.py",
  "hash": { "algorithm": "sha256", "submittedCode": "<平台代码>", "repositoryCode": "<仓库文件>" },
  "sourceMatch": { "exactBytes": false, "method": "<匹配规则>" }
}
```

### 3. 取数：自己的统计和 CuTe 的参照

```sh
make stats CHALLENGE=<id>_<name>                         # 结果表、波动、相邻版本能否区分、单次提交的门槛
make leaderboard CHALLENGE=<id>_<name> GPU=<GPU>          # CuTe 在该 GPU 上的前几名，* 是自己的；每个 GPU 一次
```

排行榜回答"还有多少空间"：第 1 名是多少，自己排第几，差距与波动相比是大是小。只取时间，不记录也不引用别人的名字。

### 4. 代码评审

逐个读每个版本，每个 GPU 上最新的那个读得最细。对照三样东西：题目约束、官方测试（哪些规模会走到哪条代码路径）、平台的计时规则。版本只在某些 GPU 上能编译或只在某个 GPU 上提交过时，在它的发现里写明。

计时规则对每道题都一样（官方仓库 `README.md` 的 FAQ）：latency 是性能测试上 15 次计时的中位数，每次用新生成的输入；计时包含 `solve` 调用的 Python 启动开销和一次设备同步，不包含编译。编译、全部测试和计时合起来约有 40 秒的时限，超过就是 `timeout`。

检查：

- **正确性**：每个下标是否恰好被写一次；整数类型在约束范围内是否会溢出；边界和尾部。
- **测试覆盖**：官方的功能测试实际执行了哪些分支。只被性能测试走到的路径，正确性没有被小规模用例检查过，要指出来。
- **多余**：平台上永远走不到的分支、重复的转换、不再需要的防御代码，以及它们的代价（编译时间、行数、理解成本）。
- **可读性和惯用法**：名字、注释是否说明了原因，CuTe 的抽象是否用得恰当。
- **未验证的假设**：代码依赖但没人检查过的事，例如"生成的是 128-bit 读写"。
- **做得好的**：一两条，让下一版保留。

每条发现写成一行：`文件:行`、类别、问题，以及下一版怎么改。版本文件是冻结的，评审意见用于下一个版本，不回头改旧文件。没有发现就写"没有发现问题"，不要凑数。

### 5. 找性能机会

对每个 GPU 分别按这个顺序想，不要直接列优化技巧：

1. **空间**。自己的最好成绩与 CuTe 第 1 名差多少？差距小于波动时，已经在 CuTe 现有解法的前沿，要直接说出来。题目有简单的物理量时（搬运的字节数、浮点运算次数），算出"工作量 ÷ latency"，帮助判断瓶颈在哪里。
2. **候选**。每个候选对应代码里一个具体的成本：线程和 block 数、每字节的指令数、访存次数和宽度、`solve` 里的主机端工作、同步。只考虑 CuTe DSL 能表达的改动。
3. **配额**。每个机会是一个新版本，只花一次提交。不影响 latency 的清理（删掉多余分支、改名）并进同一个版本，不单独提交。不要建议重复提交同一份代码来量波动。
4. **判定**。用 `make stats` 给出的"单次提交的门槛"：新版本一次提交的结果低于它才算改进，否则当作没有差别，不追加提交。预期收益明显达不到门槛的候选不值得花配额，写明这一点。

每个机会写三样：假设、依据、实验（改什么，越小越好）。判定对同一个 GPU 的所有机会都一样，就是那个 GPU 的门槛，在它的机会前面写一次，不在每个机会里重复。一个机会默认是一次提交；只有例外才注明，例如并进别的版本、或排在另一个机会之后。每个 GPU 最多三个，最值得做的排第一。同一个改动在两个 GPU 上都适用时，写在证据更强的那个 GPU 下，另一个只提一句。没有值得花配额的机会时就这样说，并建议把配额留给下一道题，这也是有用的结论。

### 6. 写 README

以 `assets/README.template.md` 为准，章节的名称和顺序不变，按"写作规则"填写。README 描述当前状态，直接改写旧内容，不追加日志；历史在 Git 里。新数据推翻了旧结论时，改掉旧结论。

模板里没有通用的说明文字，是有意的：README 只写这道题的事（写作规则 6）。

已有的学习问答放在同目录的 `NOTES.md`，原样保留。Review 不改写笔记，只在发现笔记与当前代码、数据或 CuTe DSL 源码（`.cache/cutlass`）矛盾时告诉用户。

### 7. 检查

- `make check` 通过（其中 `records` 会校验版本文件没有被改动）。
- 逐个核对 README 里的数字与 `make stats`、`make leaderboard` 的输出一致。
- 每个 GPU 的数字只出现在它自己的小节里；README 里没有跨 GPU 的 latency 比较，也没有与其他语言的比较。
- 代码评审里的每个行号指向所说的那一行。
- README 里没有原样放进另一道题也成立的句子。
- 除表格外，README 不超过 80 行；评审范围每多一个 GPU，可以多 20 行。

### 8. 汇报

用几句话告诉用户：评审了哪些 GPU，代码评审里最重要的一两条，在每个 GPU 上离 CuTe 最好成绩多远，每个 GPU 最值得做的一个实验和它需要几次提交，以及哪些结论证据不足。说明历史和排行榜是否刷新，有没有无法对应到源码的提交。Review 本身没有花任何配额，这一点不必重复。

## 写作规则

规则不多，每条都是为了让读的人能相信这份 README。

1. **分清事实、推断和推测。** 事实是平台数据、官方定义和源码里能指出来的东西。推断是由事实推出、说得出推理过程的结论。推测是还没有证据的解释。后两种要标出来；不标的就是事实，不用写"事实"两个字。
2. **只写数据支持的结论。** 只有新版最慢的一次提交比旧版最快的一次还快，并且差距超过波动，才说"更快"。否则写"未能区分"，并且不为了分出高下而建议重复提交。平台是共享 GPU，同一份代码两次提交可以相差百分之几。
3. **波动用实测值。** 波动取同一份代码在同一 GPU 上重复提交的 (最大 − 最小) / 最小；没有重复提交时按 5% 并写明是假定值。
4. **各 GPU 分开写。** 每个 GPU 一张结果表、一个快照日期，结论和性能机会也按 GPU 分开。Percentile 是取数当天的排名，要写出排名基数。不同 GPU 的 latency 不相除、不排名次：硬件不同，差距不说明代码的好坏。
5. **数字保留平台给的位数**，单位写明。每个数字都来自 `make stats`、`make leaderboard` 或 `submissions.json`；派生量给出算式。来自别处的数字（例如某次 `run` 里打印的自测结果）在同一句里写明出处，并说明它不是提交记录。
6. **只写这道题的事。** 每道题都一样的内容不进 README：计时方式、配额、怎样才算更快、`submissions.json` 的格式，都在本文件和根目录的 README 里。这次 review 的过程（登录是否失效、历史有没有刷新）写在汇报里。判断方法：一句话原样放进另一道题的 README 也成立，就不属于这里。同一个事实在 README 里只出现一次：性能机会已经写了的成本，不在代码评审里再写一条。
7. **短。** "做法"一栏是一个短语；每个版本两到四句；每条评审意见一行；每个性能机会两到三句。不写形容词，写数字。
8. **机会是实验，不是愿望，而且每个实验都要花配额。** 见第 5 步。
9. **不写本机没做过的验证。** 本机没有 CUDA，正确性和性能只引用平台记录。
