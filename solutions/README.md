# 解答目录

每道题使用一个与题目名称对应的目录，优先采用原题 URL 的 slug：

```text
solutions/<challenge-slug>/
  README.md          # 题意理解、解题过程、验证和优化记录
  solution.py        # 当前 CuTe DSL 解答
  test_solution.py   # 正确性与边界测试
  benchmark.py       # 需要本地性能测量时添加
```

开始新题时，复制 [单题笔记模板](../templates/solution-notes.md) 到题目目录的 `README.md`，并更新 [进度表](../progress.md)。代码和测试在实际解题时添加。

`solution.py` 的入口、参数和提交格式遵循该题当时的平台要求。可复用的辅助代码在出现实际需求后再提取。

保留清晰的 Git 提交历史，并在笔记中引用关键版本的 commit。引用原题与资料来源，说明自己的理解和实现选择。
