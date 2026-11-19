---
name: review-solution
description: Review a LeetGPU challenge in this repository. Pull its submission history, compare the recorded versions with computed numbers, and rewrite solutions/<challenge>/README.md from a fixed template. Use this whenever the user asks to review, 复盘, 回顾, 总结 or analyze a challenge, a solution, a version or its submissions; asks how a version performed or whether a change actually helped; or wants a solution README written or updated after new submissions, even if they never say "review".
---

# Review a solution

这个仓库用来学习：每道题保留多个版本，review 把提交历史变成一份简短、准确的说明，回答三个问题：每个版本做了什么，数字支持什么结论，下一步该试什么。

Review 只读平台、只改文档和记录。不要修改已有的版本源码，不要 `run` 或 `submit`，除非用户另外要求。

## 流程

### 1. 确定题目

题目目录是 `solutions/<id>_<name>/`，与官方仓库的目录同名，数字前缀就是 challenge ID。

官方定义在 `.cache/leetgpu-challenges/challenges/<难度>/<id>_<name>/`（不存在时运行 `make challenges`）。从这里取事实，不要凭记忆写：

- `challenge.py`：标题 `name`、`atol`、`rtol`、`get_solve_signature`、`generate_performance_test` 的规模和输入分布。
- `challenge.html`：题面和约束。

官方仓库的许可禁止再分发，所以只引用事实（规模、容差、约束），不把题面或测试代码复制进本仓库。

### 2. 同步提交历史

```sh
./tools/leetgpu/leetgpu status --json
./tools/leetgpu/leetgpu submissions list --challenge <id>_<name> --language cute --gpu H100 --json
./tools/leetgpu/leetgpu submissions code <提交 ID> --output <仓库外的临时目录>/<提交 ID>.py
```

- 退出码和登录方式见 AGENTS.md。无法登录时不要卡住：用 `submissions.json` 里已有的记录继续，并在汇报里说明历史没有刷新。
- 原始响应含账户字段，只放在仓库外的临时目录。
- 对 `submissions.json` 里还没有的每个提交，取回代码并与各版本文件比较 SHA-256：
  - 相同：`sourceVersion` 填该文件名，`sourceMatch.exactBytes` 为 `true`。
  - 不同：依次尝试"只差结尾换行"和"Python AST 相同"，把实际采用的规则写进 `sourceMatch.method`，`exactBytes` 为 `false`。
  - 都不匹配：说明平台上有仓库没有的代码。`sourceVersion` 填 `null`，问用户是否把它存为新版本，不要猜。
- 同一个 GPU 的所有记录，`metrics` 取自同一次 `list` 响应。Latency 不会变，percentile 和它的排名基数 `rankedCount` 会变，所以整组一起更新，并记下取数日期。
- 其他 GPU 只在记录里已经出现，或用户要求时才查询，每个 GPU 单独一次 `list`。
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

### 3. 统计

```sh
make stats CHALLENGE=<id>_<name>
```

输出每个 GPU 的结果表、波动，以及相邻版本能否区分。README 里的每个数字都来自这份输出或 `submissions.json`，不要心算，也不要沿用旧 README 里的数字。

### 4. 读代码

读每个版本的源码，并比较相邻版本的差别。"版本"一节写的是代码实际做了什么，不是文件名或注释声称的内容。

### 5. 写 README

以 `assets/README.template.md` 为准，章节的名称和顺序不变，按"写作规则"填写。README 描述当前状态，直接改写旧内容，不追加日志；历史在 Git 里。

已有的学习问答放在同目录的 `NOTES.md`，原样保留。Review 不改写笔记，只在发现笔记与当前代码或数据矛盾时告诉用户。

### 6. 检查

- `make check` 通过（其中 `records` 会校验版本文件没有被改动）。
- 逐个核对 README 里的数字与 `make stats` 的输出一致。
- 除表格外，README 不超过 60 行。

### 7. 汇报

用几句话告诉用户：结论是什么，哪些地方证据不足，建议的下一步实验。说明历史是否刷新，以及有没有无法对应到源码的提交。

## 写作规则

规则不多，每条都是为了让读的人能相信这份 README。

1. **只写数据支持的结论。** 只有新版最慢的一次提交比旧版最快的一次还快，并且差距超过波动，才说"更快"。否则写"未能区分"，并说明还差什么测量。这样做是因为平台是共享 GPU，同一份代码两次提交可以相差百分之几。
2. **波动用实测值。** 波动取同一份代码在同一 GPU 上重复提交的 (最大 − 最小) / 最小；没有重复提交时按 5% 并写明是假定值。
3. **原因要分清已证实和推测。** "向量化读写更快"这类解释，没有对照提交或生成的指令作证据时，写成"推测"。
4. **不同 GPU 不比较。** 每个 GPU 一张表、一个快照日期。Percentile 是取数当天的排名，不是固定值；写出它的排名基数，97 在 100 个提交里和在 10,000 个里含义不同。
5. **数字保留平台给的位数**，单位写明。派生量（倍数、带宽、吞吐）给出算式或来源。
6. **短。** "做法"一栏是一个短语；每个版本两到四句；分析每条一到三句，没有内容的条目删掉。不写形容词，写数字。
7. **下一步是实验，不是愿望。** 最多三个，每个写清改什么、预期什么、什么结果算证伪。优先选能消除当前最大不确定性的那个，通常是重复提交。
8. **不写本机没做过的验证。** 本机没有 CUDA，正确性和性能只引用平台记录。
