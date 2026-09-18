"""RAG 检索评估：从评估集抽样，量化检索质量。

指标：
- answer_token_f1@1/@5：检索出案例的 answer 与真实 answer 的 token F1
- 命中率@5：top-5 内存在相似度超阈值的答案
- type 一致率@1：检索 top-1 案例的 type 与真实 type 一致的比例

用法: python scripts/eval_rag.py [--n 200] [--mode hybrid|bm25|vec]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.retriever import HybridRetriever
from app.config import PROCESSED_DIR


def tok(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(pred: str, ref: str) -> float:
    p, r = tok(pred), tok(ref)
    if not p or not r:
        return 0.0
    inter = len(p & r)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(p), inter / len(r)
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "vec"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")
    df = df.sample(min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    r = HybridRetriever()
    if not r.load():
        print("索引不存在，先运行 scripts/build_index.py")
        sys.exit(1)

    f1_top1, f1_top5, hit5, type_hit1, t0 = [], [], [], [], time.time()
    for i, row in enumerate(df.itertuples(), 1):
        hits = r.retrieve(row.text, k=5, mode=args.mode)
        f1s = [token_f1(h["answer"], row.answer) for h in hits]
        f1_top5.append(max(f1s) if f1s else 0.0)
        f1_top1.append(f1s[0] if f1s else 0.0)
        hit5.append(1 if any(f >= 0.25 for f in f1s) else 0)
        type_hit1.append(1 if hits and hits[0]["type"] == row.type else 0)
        if i % 50 == 0:
            print(f"  {i}/{len(df)} | 耗时 {time.time()-t0:.0f}s")

    report = {
        "n": len(df),
        "mode": args.mode,
        "answer_token_f1@1": round(sum(f1_top1) / len(f1_top1), 4),
        "answer_token_f1@5": round(sum(f1_top5) / len(f1_top5), 4),
        "hit_rate@5": round(sum(hit5) / len(hit5), 4),
        "type_consistency@1": round(sum(type_hit1) / len(type_hit1), 4),
        "cost_seconds": round(time.time() - t0, 1),
    }
    out = Path(__file__).resolve().parent.parent / "eval"
    out.mkdir(exist_ok=True)
    (out / f"rag_report_{args.mode}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n=== RAG 检索评估 ({args.mode}, n={report['n']}) ===")
    for k, v in report.items():
        if k not in ("n", "mode", "cost_seconds"):
            print(f"  {k}: {v}")
    print(f"  耗时 {report['cost_seconds']}s")


if __name__ == "__main__":
    main()
