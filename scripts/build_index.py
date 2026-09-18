"""构建知识库索引（BM25 词表 + 向量），保存到 data/indexes/。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.retriever import HybridRetriever


def main() -> None:
    r = HybridRetriever()
    print("构建知识库索引（20k 工单 embedding，约 1-3 分钟）...")
    r.build()
    print(f"索引构建完成: 知识库 {len(r.df)} 条, embedding 维度 {r._embeddings.shape[1]}")
    print("验证检索:")
    hits = r.retrieve("I cannot log in to my account, invalid credentials error", k=3)
    for h in hits:
        print(f"  - [{h['type']}|{h['queue']}] score={h['score']}: {h['text'][:80]}...")


if __name__ == "__main__":
    main()
