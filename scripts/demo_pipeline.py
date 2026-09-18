"""端到端演示：跑完整链路（分诊 → 检索 → 回复草稿）并打印结果。

用法:
    python scripts/demo_pipeline.py --n 3                      # 抽 3 条真实评估集工单
    python scripts/demo_pipeline.py --text "共享网盘又中断了…"    # 跑指定工单（支持中文）
    python scripts/demo_pipeline.py --text "..." --subject "主题"
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.lang import has_cjk, to_retrieval_query  # noqa: E402
from app.agents.reply import generate_reply  # noqa: E402
from app.agents.retriever import HybridRetriever  # noqa: E402
from app.agents.triage import triage  # noqa: E402
from app.config import PROCESSED_DIR  # noqa: E402


def run_one(retriever: HybridRetriever, text: str, label: str, truth: str = "") -> None:
    print(f"\n{'=' * 70}\n{label}")
    print(f"内容: {text[:220].replace(chr(10), ' ')}...")

    if has_cjk(text):
        bridge = to_retrieval_query(text)
        print(f"\n→ 中文桥接检索查询: {bridge[:200]}")

    t = triage(text)  # 默认动态 few-shot（检索相似历史工单做示例）
    tail = f" | 真实: {truth}" if truth else ""
    print(f"\n→ 分诊: {t['type']} / {t['priority']} (conf {t['confidence']:.2f})"
          f" 示例来源={t.get('examples')}{tail}")
    print(f"  依据: {t['reason']}")

    hits = retriever.retrieve(text, k=3)
    print("→ 检索 Top-3: " + "; ".join(f"[{h['type']}|{h['queue']}]{h['text'][:40]}…" for h in hits))

    rep = generate_reply(text, hits)
    print(f"→ 草稿({len(rep['reply'])}字, 语言={rep['lang']}, 需人工={rep['needs_human']}):")
    print(rep["reply"][:500])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="从评估集抽样的条数")
    parser.add_argument("--text", default="", help="直接指定工单内容（优先级高于 --n）")
    parser.add_argument("--subject", default="", help="配合 --text 使用的工单主题")
    args = parser.parse_args()

    retriever = HybridRetriever()
    retriever.load()

    if args.text:
        text = f"{args.subject}\n{args.text}".strip() if args.subject else args.text
        run_one(retriever, text, "自定义工单")
        return

    df = pd.read_parquet(PROCESSED_DIR / "eval.parquet").sample(args.n, random_state=7)
    for i, row in enumerate(df.itertuples(), 1):
        run_one(retriever, row.text, f"工单 #{i} {row.subject[:60]}", f"{row.type} / {row.priority}")


if __name__ == "__main__":
    main()
