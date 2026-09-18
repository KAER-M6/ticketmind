"""D5-1b 分诊动态 few-shot 评估：用混合检索取相似训练工单做 few-shot 示例。

对比基线（固定 few-shot, 74.2%）。检索本地完成，LLM 单次调用。

用法: python scripts/eval_dfs.py [--n 200] [--k 5] [--seed 42] [--workers 8]
输出: eval/dfs_report.json, eval/dfs_predictions.csv
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.retriever import HybridRetriever
from app.agents.triage import SYSTEM_PROMPT, _normalize
from app.config import PROCESSED_DIR
from app.llm import chat_json


def build_examples(hits: list[dict]) -> list[dict]:
    """检索到的相似工单 → few-shot 消息对（input=工单文本，output=标准 JSON）。"""
    examples = []
    for h in hits:
        out = {
            "type": h["type"],
            "priority": h["priority"],
            "reason": "与该历史工单场景相似，沿用同类判定",
            "confidence": 0.9,
        }
        examples.append({"input": f"工单内容：\n{h['text'][:800]}", "output": json.dumps(out, ensure_ascii=False)})
    return examples


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
    print(f"动态 few-shot 评估: {len(sampled)} 条 | k={args.k} 相似示例/条")

    r = HybridRetriever()
    if not r.load():
        print("索引不存在，先运行 scripts/build_index.py")
        sys.exit(1)

    t0 = time.time()
    rows = [None] * len(sampled)

    def work(pair):
        i, row = pair
        hits = r.retrieve(row.text, k=args.k)
        examples = build_examples(hits)
        prompt = f"工单内容：\n{row.text[:2500]}"
        pred = _normalize(chat_json(
            prompt, system=SYSTEM_PROMPT, examples=examples,
            temperature=0.0, max_tokens=300,
        ))
        return i, row, pred, hits

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, p) for p in enumerate(sampled.itertuples())]
        done = 0
        for fut in as_completed(futures):
            i, row, pred, hits = fut.result()
            rows[i] = {
                "text": row.text[:200],
                "true_type": row.type,
                "pred_type": pred["type"],
                "true_priority": row.priority,
                "pred_priority": pred["priority"],
                "top1_train_type": hits[0]["type"] if hits else None,
            }
            done += 1
            if done % 40 == 0:
                print(f"  进度 {done}/{len(sampled)} | 耗时 {time.time()-t0:.0f}s")

    res = pd.DataFrame(rows)
    report = {
        "method": f"dynamic few-shot (kNN hybrid retrieval, k={args.k}, temp=0)",
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
        "sc_k5_type_accuracy": 0.675,
    }

    out = Path(__file__).resolve().parent.parent / "eval"
    out.mkdir(exist_ok=True)
    (out / "dfs_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    res.to_csv(out / "dfs_predictions.csv", index=False)

    print(f"\n=== 动态 few-shot 结果 (n={report['n']}) ===")
    print(f"type     准确率 {report['type']['accuracy']:.4f} | macro-F1 {report['type']['macro_f1']:.4f} | 基线 0.7420 | SC 0.6750")
    print(f"priority 准确率 {report['priority']['accuracy']:.4f}")
    print(f"耗时 {report['cost_seconds']}s | 报告: eval/dfs_report.json")


if __name__ == "__main__":
    main()
