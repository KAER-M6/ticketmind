"""端到端演示：取 N 条真实评估集工单，跑完整链路并打印结果。

用法: python scripts/demo_pipeline.py [--n 3] [--save html]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.reply import generate_reply
from app.agents.retriever import HybridRetriever
from app.agents.triage import load_fewshot, triage
from app.config import PROCESSED_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_parquet(PROCESSED_DIR / "eval.parquet").sample(args.n, random_state=7)
    fewshot = load_fewshot()
    retriever = HybridRetriever()
    retriever.load()

    for i, row in enumerate(df.itertuples(), 1):
        print(f"\n{'='*70}\n工单 #{i} [{row.type}|{row.priority}] {row.subject[:60]}")
        print(f"内容: {row.body[:220].replace(chr(10),' ')}...")

        t = triage(row.text, fewshot)
        print(f"\n→ 分诊: {t['type']} / {t['priority']} (conf {t['confidence']:.2f}) | 真实: {row.type} / {row.priority}")
        print(f"  依据: {t['reason']}")

        hits = retriever.retrieve(row.text, k=3)
        print("→ 检索 Top-3: " + "; ".join(f"[{h['type']}|{h['queue']}]{h['text'][:40]}…" for h in hits))

        rep = generate_reply(row.text, hits)
        print(f"→ 草稿({len(rep['reply'])}字, 需人工={rep['needs_human']}):\n{rep['reply'][:400]}")


if __name__ == "__main__":
    main()
