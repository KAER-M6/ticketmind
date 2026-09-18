"""分诊 Agent 评估：对评估集抽样跑分，输出准确率/F1/混淆矩阵/错误样本。

用法: python scripts/eval_triage.py [--n 500] [--seed 42]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.triage import load_fewshot, triage
from app.config import PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="抽样评估条数")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")
    # 按 type 分层抽样，保证每类都覆盖
    n = min(args.n, len(df))
    _, sampled_idx = train_test_split(
        df.index, test_size=n, stratify=df["type"], random_state=args.seed
    )
    sampled = df.loc[sampled_idx].reset_index(drop=True)
    print(f"评估样本: {len(sampled)} 条 | type: {dict(sampled['type'].value_counts())}")

    rows, t0 = [], time.time()
    examples = load_fewshot()
    print(f"few-shot 示例: {len(examples)} 条")
    for i, row in enumerate(sampled.itertuples(), 1):
        try:
            pred = triage(row.text, examples)
        except Exception as e:
            print(f"  [{i}] 调用失败: {e}")
            continue
        rows.append(
            {
                "text": row.text[:200],
                "true_type": row.type,
                "pred_type": pred["type"],
                "true_priority": row.priority,
                "pred_priority": pred["priority"],
                "reason": pred["reason"],
                "confidence": pred["confidence"],
            }
        )
        if i % 100 == 0:
            print(f"  进度 {i}/{len(sampled)} | 耗时 {time.time()-t0:.0f}s")

    res = pd.DataFrame(rows)
    report = {
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
            "weighted_f1": round(f1_score(res["true_priority"], res["pred_priority"], average="weighted"), 4),
        },
        "type_report": classification_report(res["true_type"], res["pred_type"], output_dict=True),
        "priority_report": classification_report(res["true_priority"], res["pred_priority"], output_dict=True),
        "type_confusion": _confusion(res["true_type"], res["pred_type"]),
        "priority_confusion": _confusion(res["true_priority"], res["pred_priority"]),
    }

    out = Path(__file__).resolve().parent.parent / "eval"
    out.mkdir(exist_ok=True)
    (out / "triage_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    res.to_csv(out / "triage_predictions.csv", index=False)

    print(f"\n=== 分诊评估结果 (n={report['n']}) ===")
    print(f"type     准确率 {report['type']['accuracy']:.4f} | macro-F1 {report['type']['macro_f1']:.4f}")
    print(f"priority 准确率 {report['priority']['accuracy']:.4f} | macro-F1 {report['priority']['macro_f1']:.4f}")
    print(f"耗时 {report['cost_seconds']}s | 报告: eval/triage_report.json")


def _confusion(true, pred) -> dict:
    labels = sorted(set(true) | set(pred))
    m = {a: {b: 0 for b in labels} for a in labels}
    for t, p in zip(true, pred, strict=True):
        m[t][p] += 1
    return m


if __name__ == "__main__":
    main()
