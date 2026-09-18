"""反馈池增益实验：量化「人工审核回流的样本」对分诊准确率的实际贡献。

设计（同一批测试样本，仅切换反馈池）：
    A 组：反馈池为空      → 仅动态 few-shot（检索历史工单）
    B 组：反馈池 = P 条人工确认样本 → 人工样本占 few-shot 前 2 位 + 检索补齐
两组除反馈池外完全一致，差值即为闭环带来的净收益。

跑法：
    python scripts/eval_feedback_gain.py --n 100 --pool 60 --workers 6
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from app.agents import feedback  # noqa: E402
from app.agents.triage import triage  # noqa: E402
from app.config import PROCESSED_DIR  # noqa: E402

TMP_POOL = Path(__file__).resolve().parent.parent / "eval" / "_tmp_feedback_pool.jsonl"


def _reset_feedback(path: Path | None) -> None:
    """把反馈池指向指定文件（None = 空池）并清掉模块缓存。"""
    feedback.FEEDBACK_PATH = path or TMP_POOL.parent / "_no_such_pool.jsonl"
    if feedback.FEEDBACK_PATH.exists():
        feedback.FEEDBACK_PATH.unlink()
    feedback._cache = {"mtime": None, "rows": [], "tokenized": [], "idf": {}}


def _build_pool(df: pd.DataFrame, p: int) -> int:
    """用训练集样本模拟「人工已审核并确认」的反馈池。"""
    TMP_POOL.parent.mkdir(exist_ok=True)
    rows = df.sample(min(p, len(df)), random_state=7)
    with TMP_POOL.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows.itertuples(), 1):
            f.write(
                json.dumps(
                    {
                        "ticket_id": f"SIM-{i:04d}",
                        "text": (r.text or "")[:2500],
                        "type": r.type,
                        "priority": r.priority,
                        "corrected": False,
                        "ts": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    # 关键：把反馈池指向刚写入的模拟池（否则分诊仍读空池，实验无效）
    feedback.FEEDBACK_PATH = TMP_POOL
    feedback._cache = {"mtime": None, "rows": [], "tokenized": [], "idf": {}}
    return len(feedback.load_all())


def _evaluate(samples: list[tuple[str, str]], workers: int) -> dict:
    def one(item):
        text, gold = item
        try:
            r = triage(text)
            return gold, r["type"], r.get("examples", {}).get("feedback", 0)
        except Exception:
            return gold, None, 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(one, samples))
    n = len(out)
    correct = sum(1 for g, p, _ in out if g == p)
    return {
        "n": n,
        "correct": correct,
        "type_accuracy": round(correct / n, 4) if n else 0.0,
        "avg_feedback_used": round(sum(f for _, _, f in out) / n, 3) if n else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="测试样本数")
    ap.add_argument("--pool", type=int, default=60, help="模拟人工确认的样本数")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="eval/feedback_gain_report.json")
    args = ap.parse_args()

    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    ev = pd.read_parquet(PROCESSED_DIR / "eval.parquet")
    ev = ev.sample(min(args.n, len(ev)), random_state=42).reset_index(drop=True)
    samples = [(r.text or "", r.type) for r in ev.itertuples()]

    print(f"[A 组] 反馈池为空，n={len(samples)} …")
    _reset_feedback(None)
    a = _evaluate(samples, args.workers)
    print(f"        type 准确率 {a['type_accuracy']:.1%}（{a['correct']}/{a['n']}）")

    print(f"[B 组] 反馈池 = {args.pool} 条人工确认样本 …")
    pool_size = _build_pool(train, args.pool)
    b = _evaluate(samples, args.workers)
    print(f"        type 准确率 {b['type_accuracy']:.1%}（{b['correct']}/{b['n']}）"
          f"，平均命中 {b['avg_feedback_used']} 条人工样本")

    delta = round(b["type_accuracy"] - a["type_accuracy"], 4)
    report = {
        "n": len(samples),
        "feedback_pool": pool_size,
        "baseline": a,
        "with_feedback": b,
        "delta": delta,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n净收益：{delta:+.1%}（{a['type_accuracy']:.1%} → {b['type_accuracy']:.1%}）")
    print(f"报告已写入 {out}")
    if TMP_POOL.exists():
        TMP_POOL.unlink()  # 模拟池不留在仓库里
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
