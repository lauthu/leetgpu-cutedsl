#!/usr/bin/env python3
"""Summarize solutions/<challenge>/submissions.json for a review.

A review looks at one GPU. This script picks it from the submission history
(or takes it as the second argument), then prints a Markdown results table,
whether each version is distinguishable from the previous one, and the
latency a single new submission must beat to count as an improvement. Every
number in a review should come from this output, so that nothing is retyped
or estimated.

Exit codes: 0 done, 1 unreadable records, 2 usage, 3 the target GPU is
ambiguous and the user has to choose.

Standard library only, Python 3.9+, no network and no credentials.
"""

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# Used when no code was submitted twice on a GPU. The platform's maintainers
# describe run-to-run variance as "a few percent".
ASSUMED_NOISE_PERCENT = Decimal("5")


def version_key(name):
    match = re.match(r"v(\d+)", name)
    return (int(match.group(1)) if match else 10**6, name)


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def ms(value):
    return str(value.quantize(Decimal("0.00001"), rounding=ROUND_HALF_UP))


def percent(value):
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) + "%"


def pick_gpu(submissions):
    """Return (history line, target GPU or None, reason).

    The target is the GPU the user has been working on: the one with the most
    submissions, provided the latest submission is on it too. Anything less
    clear is left to the user.
    """
    counts, latest = {}, {}
    for entry in submissions:
        gpu = entry.get("accelerator")
        if not gpu:
            continue
        counts[gpu] = counts.get(gpu, 0) + 1
        latest[gpu] = max(latest.get(gpu, ""), str(entry.get("createdAt") or ""))
    if not counts:
        return "提交历史：没有记录。", None, "没有提交记录"
    order = sorted(counts, key=lambda gpu: (-counts[gpu], gpu))
    history = "；".join(
        f"{gpu} {counts[gpu]} 次（最近 {latest[gpu][:10] or '未知'}）" for gpu in order
    )
    most = [gpu for gpu in order if counts[gpu] == counts[order[0]]]
    newest = max(latest, key=lambda gpu: latest[gpu])
    if len(most) == 1 and most[0] == newest:
        return f"提交历史：{history}。", newest, "提交最多，最近一次提交也在这个 GPU 上"
    reason = f"提交最多的是 {'、'.join(most)}，最近一次提交在 {newest}"
    return f"提交历史：{history}。", None, reason


def summarize(submissions):
    """Group successful, source-matched runs by GPU, then by version file."""
    groups = {}
    skipped = {}
    ranked = {}
    for entry in submissions:
        gpu = entry.get("accelerator")
        latency = (entry.get("metrics") or {}).get("executionTime")
        version = entry.get("sourceVersion")
        if entry.get("status") != "success" or latency is None or not version:
            skipped[gpu] = skipped.get(gpu, 0) + 1
            continue
        runs = groups.setdefault(gpu, {}).setdefault(version, [])
        runs.append((Decimal(str(latency)), entry["metrics"].get("percentile")))
        if entry["metrics"].get("rankedCount") is not None:
            ranked.setdefault(gpu, set()).add(entry["metrics"]["rankedCount"])
    # A percentile is a rank among this many submissions. Rows read at different
    # times can disagree, and then no single base describes the table.
    bases = {gpu: counts.pop() for gpu, counts in ranked.items() if len(counts) == 1}
    return groups, skipped, bases


def report(gpu, versions, skipped, first_version, base_count):
    names = sorted(versions, key=version_key)
    total = sum(len(versions[name]) for name in names)
    ranking = f"，percentile 的排名基数 {base_count}" if base_count else ""
    lines = [f"### {gpu}（{total} 次成功提交{ranking}）", ""]
    lines.append("| 版本 | 提交次数 | 最好 (ms) | 中位 (ms) | 相对首版 | Percentile |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    # "相对首版" compares against the challenge's first version, on the same GPU.
    # Without a run of that version on this GPU there is nothing to compare to.
    base = None
    if names[0] == first_version:
        base = median([latency for latency, _ in versions[first_version]])
    noise, noise_source = None, None
    for name in names:
        latencies = [latency for latency, _ in versions[name]]
        best = min(versions[name], key=lambda run: run[0])
        center = median(latencies)
        speedup = "—"
        if base is not None:
            ratio = (base / center).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            speedup = f"{ratio}×"
        shown_percentile = "—" if best[1] is None else best[1]
        lines.append(
            f"| `{name}` | {len(latencies)} | {ms(best[0])} | {ms(center)} "
            f"| {speedup} | {shown_percentile} |"
        )
        if len(latencies) > 1:
            spread = (max(latencies) - min(latencies)) / min(latencies) * 100
            if noise is None or spread > noise:
                noise, noise_source = spread, f"`{name}` 的 {len(latencies)} 次提交"
    lines.append("")
    if noise is None:
        noise = ASSUMED_NOISE_PERCENT
        lines.append(f"波动：没有重复提交，按 {percent(noise)} 假定。")
    else:
        lines.append(
            f"波动：{percent(noise)}（同一份代码重复提交的 (最大 − 最小) / 最小，"
            f"来自 {noise_source}）。"
        )
    # Conservative comparison: a version only counts as faster when its slowest
    # run beats the older version's fastest run by more than the noise, and as
    # slower when its fastest run loses to the older version's slowest run.
    for index in range(1, len(names)):
        older, newer = names[index - 1], names[index]
        old = [latency for latency, _ in versions[older]]
        new = [latency for latency, _ in versions[newer]]
        gain = (min(old) - max(new)) / min(old) * 100
        loss = (min(new) - max(old)) / max(old) * 100
        if gain > noise:
            detail = (
                f"新版最慢 {ms(max(new))} 比旧版最快 {ms(min(old))} 快 {percent(gain)}"
            )
            verdict = "超过波动 → 新版更快"
        elif loss > noise:
            detail = (
                f"新版最快 {ms(min(new))} 比旧版最慢 {ms(max(old))} 慢 {percent(loss)}"
            )
            verdict = "超过波动 → 新版更慢"
        else:
            detail = (
                f"旧版 {ms(min(old))}–{ms(max(old))}，"
                f"新版 {ms(min(new))}–{ms(max(new))}，"
                f"最保守的差距 {percent(max(gain, loss))}"
            )
            verdict = "不超过波动 → 未能区分"
        lines.append(f"`{older}` → `{newer}`：{detail}，{verdict}。")
    # Submissions are rationed, so the next version gets one. This is the
    # latency it must beat for that single result to mean something.
    newest = names[-1]
    best = min(latency for latency, _ in versions[newest])
    lines.append(
        f"单次提交的门槛：下一版要低于 {ms(best * (1 - noise / 100))} ms "
        f"才算比 `{newest}` 快（它的最好成绩 {ms(best)} ms 再快 {percent(noise)}）。"
    )
    if skipped:
        lines.append(
            f"未计入：{skipped} 条记录（非 success、缺少 latency 或未对应到源码）。"
        )
    return "\n".join(lines)


def main():
    if len(sys.argv) not in (2, 3):
        print("usage: stats.py solutions/<challenge> [GPU]", file=sys.stderr)
        return 2
    path = Path(sys.argv[1]) / "submissions.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        groups, skipped, bases = summarize(data["submissions"])
        history, target, reason = pick_gpu(data["submissions"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: cannot read {path} ({type(error).__name__})", file=sys.stderr)
        return 1
    print(history)
    if len(sys.argv) == 3:
        target, reason = sys.argv[2], "由调用者指定"
    if target is None:
        print(f"目标 GPU 不明确：{reason}。请向用户确认，再用 SKU=<GPU> 重新运行。")
        return 3
    print(f"目标 GPU：{target}（{reason}）。\n")
    if target not in groups:
        print(f"{target} 上没有可统计的成功提交。")
        return 0
    first_version = min(
        (name for versions in groups.values() for name in versions), key=version_key
    )
    print(
        report(
            target,
            groups[target],
            skipped.get(target, 0),
            first_version,
            bases.get(target),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
