"""RAG 检索：BM25 + 向量混合检索历史工单知识库（text → answer）。

- 知识库 = 训练集（带标准答案的历史工单）
- BM25：关键词精确匹配，对工单类密集文本效果好
- 向量：MiniLM 本地 embedding，numpy 暴力余弦（20k 级毫秒）
- 混合：RRF（Reciprocal Rank Fusion）合并两路排序
"""
import json
import os
import re
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像，避免模型加载直连超时

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import INDEXES_DIR, PROCESSED_DIR

STOPWORDS = set(
    """a an the and or but if then else for with without via into onto from by
    to of in on at is are was were be been being have has had do does did
    i you he she it we they them me my your our their this that these those
    not no so very can could would should will just really please need help
    about between after before during while when where why how what which who
    """.split()
)

_EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None
_model_lock = threading.Lock()


def _get_model():
    """懒加载 embedding 模型（加锁，多线程并发首访时只加载一次）。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(_EMB_MODEL_NAME)
    return _model


_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def tokenize(text: str) -> list[str]:
    """英文按词切分；中文按「单字 + 双字」切分兜底。

    中文没有空格，若不显式切分，rank_bm25 收到的词列表为空、BM25 一路直接归零。
    正常链路里中文查询会先经过 lang.to_retrieval_query 转写成英文，
    这里只是保底，确保转写失败时仍不做「静默空检索」。
    """
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    for seg in _CJK_RUN.findall(text):
        tokens.extend(seg)
        tokens.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return tokens


def _rrf(scores: np.ndarray, top: int = 100) -> dict[int, float]:
    order = np.argsort(-scores)[:top]
    return {int(i): 1.0 / (60 + r) for r, i in enumerate(order)}


class HybridRetriever:
    def __init__(self, kb_path: Path | None = None):
        self.kb_path = kb_path or (PROCESSED_DIR / "train.parquet")
        self.df: pd.DataFrame | None = None
        self._tokenized: list[list[str]] = []
        self._embeddings: np.ndarray | None = None
        self._bm25 = None
        self._cache: dict[tuple, list[dict]] = {}
        self._lock = threading.Lock()
        self.cache_hits = 0
        self.cache_misses = 0

    # ---- 构建 / 加载 ----
    def build(self) -> None:
        from rank_bm25 import BM25Okapi

        self.df = pd.read_parquet(self.kb_path)
        self._tokenized = [tokenize(t) for t in self.df["text"]]
        self._bm25 = BM25Okapi(self._tokenized)
        self._embeddings = _get_model().encode(
            [t[:2000] for t in self.df["text"]],
            batch_size=128,
            show_progress_bar=True,
        )
        self._embeddings = self._embeddings / np.linalg.norm(
            self._embeddings, axis=1, keepdims=True
        )
        INDEXES_DIR.mkdir(exist_ok=True)
        np.save(INDEXES_DIR / "embeddings.npy", self._embeddings)
        (INDEXES_DIR / "kb_meta.json").write_text(
            json.dumps({"n": len(self.df), "path": str(self.kb_path)}),
            encoding="utf-8",
        )

    def load(self) -> bool:
        from rank_bm25 import BM25Okapi

        if not (INDEXES_DIR / "embeddings.npy").exists():
            return False
        self.df = pd.read_parquet(self.kb_path)
        self._tokenized = [tokenize(t) for t in self.df["text"]]
        self._bm25 = BM25Okapi(self._tokenized)
        self._embeddings = np.load(INDEXES_DIR / "embeddings.npy")
        return True

    # ---- 检索 ----
    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        """混合检索 top-k。

        同一 (query, k, mode) 命中缓存直接返回，省掉一次编码 + 一次全库打分
        （约 30~80ms）。演示重复跑同一张工单、批量导入抽样重叠时都能命中。

        中文查询会先经 lang 桥接层转写成英文再检索（英文查询零开销）。
        """
        if self.df is None:
            raise RuntimeError("检索器未初始化，先调用 build() 或 load()")

        key = (mode, k, query[:2000])
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return [dict(h) for h in cached]
        self.cache_misses += 1

        search_text = query
        try:
            from app.agents.lang import has_cjk, to_retrieval_query

            if has_cjk(query):
                search_text = to_retrieval_query(query)
        except Exception as e:  # 桥接层异常不能拖垮检索
            print(f"[retriever] 查询桥接失败，直接检索原文：{type(e).__name__}: {e}")

        q_tokens = tokenize(search_text)
        q_emb = _get_model().encode(search_text[:2000])
        q_emb = q_emb / np.linalg.norm(q_emb)

        n = len(self.df)
        bm25_scores = (
            self._bm25.get_scores(q_tokens) if (mode in ("bm25", "hybrid") and q_tokens) else np.zeros(n)
        )
        vec_scores = (
            self._embeddings @ q_emb if mode in ("vec", "hybrid") else np.zeros(n)
        )

        fused: dict[int, float] = {}
        if mode in ("bm25", "hybrid"):
            for i, s in _rrf(bm25_scores).items():
                fused[i] = fused.get(i, 0) + s
        if mode in ("vec", "hybrid"):
            for i, s in _rrf(vec_scores).items():
                fused[i] = fused.get(i, 0) + s

        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        result = [
            {
                "text": self.df.iloc[i]["text"][:500],
                "answer": self.df.iloc[i]["answer"],
                "type": self.df.iloc[i]["type"],
                "priority": self.df.iloc[i]["priority"],
                "queue": self.df.iloc[i]["queue"],
                "score": round(float(s), 4),
            }
            for i, s in top
        ]

        with self._lock:
            if len(self._cache) >= 512:  # 简单容量控制，避免长跑吃内存
                self._cache.clear()
            self._cache[key] = result
        return [dict(h) for h in result]


_singleton: HybridRetriever | None = None
_singleton_lock = threading.Lock()


def get_retriever() -> HybridRetriever | None:
    """全进程共享的检索器单例（线程安全懒加载）。

    main.py 与 triage.py 共用同一份，避免模型 + 索引被加载两次（约 50MB 冗余）。
    索引不存在时返回 None，调用方自行降级。
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                r = HybridRetriever()
                _singleton = r if r.load() else None  # type: ignore[assignment]
    return _singleton

