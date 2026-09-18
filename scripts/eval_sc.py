"""D5-1 分诊 self-consistency 评估：多次采样 + 多数投票，对比单次基线。

用法: python scripts/eval_sc.py [--n 200] [--k 5] [--seed 42] [--workers 8]
输出: eval/sc_report.json, eval/sc_predictions.csv
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.triage import SYSTEM_PROMPT, _normalize, load_fewshot
from app.config import PROCESSED_DIR
from app.llm import chat_json


def sc_triage(text: str, examples: list[dict], k: int = 5) -> dict:
    """k 次采样（temperature>0），type/priority 多数投票，confidence 用投票一致度。"""
    prompt = f"工单内容：\n{text[:2500]}"
    votes_t, votes_p = [], []
    for _ in range(k):
        try:
            raw = chat_json(
                prompt, system=SYSTEM_PROMPT, examples=examples,
                temperature=0.7, max_tokens=300,
            )
            norm = _normalize(raw)
            votes_t.append(norm["type"])
            votes_p.append(norm["priority"])
        except Exception:
            continue
    if not votes_t:
        return {"type": "Request", "priority": "medium", "confidence": 0.0, "agree": 0.0}
    t, tc = Counter(votes_t).most_common(1)[0]
    p, pc = Counter(votes_p).most_common(1)[0]
    return {
        "type": t, "priority": p,
        "confidence": round(tc / len(votes_t), 3),
        "agree": round(tc / len(votes_t), 3),
        "votes": "/".join(votes_t),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")
    n = min(args.n, len(df))
    _, idx = train_test_split(df.index, test_size=n, stratify=df["type"], random_state=args.seed)
    sampled = df.loc[idx].reset_index(drop=True)
    print(f"SC 评估: {len(sampled)} 条 × k={args.k} 采样 | workers={args.workers}")

    examples = load_fewshot()
    t0 = time.time()
    rows = [None] * len(sampled)

    def work(i_row):
        i, row = i_row
        pred = sc_triage(row.text, examples, k=args.k)
        return i, row, pred

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, pair) for pair in enumerate(sampled.itertuples())]
        done = 0
        for fut in as_completed(futures):
            i, row, pred = fut.result()
            rows[i] = {
                "text": row.text[:200],
                "true_type": row.type,
                "pred_type": pred["type"],
                "true_priority": row.priority,
                "pred_priority": pred["priority"],
                "agree": pred["agree"],
                "votes": pred["votes"],
            }
            done += 1
            if done % 20 == 0:
                print(f"  进度 {done}/{len(sampled)} | 耗时 {time.time()-t0:.0f}s")

    res = pd.DataFrame(rows)
    report = {
        "method": f"self-consistency k={args.k} temperature=0.7",
        "n": len(res),
        "cost_seconds": round(time.time() - t0, 1),
        "type": {
            "accuracy": round(accuracy_score(res["true_type"], res["pred_type"]), 4),
            "macro_f1": round(f1_score(res["true_type"], res["pred_type"], average="macro"), 4),
            "weighted_f1": round(f1_score(res["true_type"], res["pred_type"], average="weighted"), 4),
        },
        "priority": {
            "accuracy": round(accuracy_score(res["true_priority"], res["pred_priority"]), 4),
            "macro_f1": round(f1_score(res["true_priority"], res["pred_priority"], average="macro"), 4),
        },
        "type_report": classification_report(res["true_type"], res["pred_type"], output_dict=True),
        "baseline_type_accuracy": 0.742,
    }

    out = Path(__file__).resolve().parent.parent / "eval"
    out.mkdir(exist_ok=True)
    (out / "sc_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    res.to_csv(out / "sc_predictions.csv", index=False)

    print(f"\n=== Self-Consistency 结果 (n={report['n']}, k={args.k}) ===")
    print(f"type     准确率 {report['type']['accuracy']:.4f} | macro-F1 {report['type']['macro_f1']:.4f} | 基线 0.7420")
    print(f"priority 准确率 {report['priority']['accuracy']:.4f}")
    print(f"耗时 {report['cost_seconds']}s | 报告: eval/sc_report.json")


if __name__ == "__main__":
    main()
