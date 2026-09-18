"""人工反馈池：把审核台确认过的工单样本回流成分诊的 few-shot 语料。

数据闭环：
    审核台「通过」（可选纠正标签）→ 追加 data/feedback.jsonl
    → 分诊时用轻量 BM25 检索最相似的 k 条人工确认样本
    → 与历史知识库检索结果一起构成动态 few-shot

为什么单独存一份 jsonl 而不用 tickets 表：
- 反馈是「追加型事实」，jsonl 天然 append-only，审核动作不会被后续改状态影响
- 体量小（千级），用 BM25 现算即可，不需要重建 30MB 向量索引
- 与业务库解耦：清空演示数据不会丢反馈资产
"""
import json
import math
import threading
from collections import Counter
from datetime import datetime

from app.config import ROOT_DIR

FEEDBACK_PATH = ROOT_DIR / "data" / "feedback.jsonl"

_lock = threading.Lock()
_cache: dict = {"mtime": None, "rows": [], "tokenized": [], "idf": {}}


def _tokenize(text: str) -> list[str]:
    from app.agents.retriever import tokenize

    return tokenize(text)


def record(
    ticket_id: str,
    text: str,
    final_type: str,
    final_priority: str,
    corrected: bool,
) -> None:
    """追加一条人工确认样本（同一 ticket_id 只保留最新一条，读取时去重）。"""
    if not text.strip():
        return
    row = {
        "ticket_id": ticket_id,
        "text": text[:2500],
        "type": final_type,
        "priority": final_priority,
        "corrected": bool(corrected),
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _idf(tokenized: list[list[str]]) -> dict[str, float]:
    """平滑 IDF：log(1 + N/df)。恒为正，小样本池（几条样本）也不会退化成全零。"""
    n = len(tokenized) or 1
    df = Counter()
    for toks in tokenized:
        df.update(set(toks))
    return {t: math.log(1 + n / c) for t, c in df.items()}


def load_all() -> list[dict]:
    """读取全部反馈样本（按 ticket_id 去重取最新），文件变更时自动重建词权重。"""
    if not FEEDBACK_PATH.exists():
        return []
    mtime = FEEDBACK_PATH.stat().st_mtime
    if _cache["mtime"] == mtime:
        return _cache["rows"]

    rows: dict[str, dict] = {}
    with FEEDBACK_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r.get("ticket_id") or r.get("ts")] = r  # 后写的覆盖先写的
    data = list(rows.values())
    tokenized = [_tokenize(r["text"]) for r in data]

    with _lock:
        _cache["mtime"] = mtime
        _cache["rows"] = data
        _cache["tokenized"] = tokenized
        _cache["idf"] = _idf(tokenized)
    return data


def similar(text: str, k: int = 2) -> list[dict]:
    """检索最相似的 k 条人工确认样本（TF-IDF 词汇打分，池子小所以现算足够快）。"""
    data = load_all()
    if not data or k <= 0:
        return []
    q_tokens = set(_tokenize(text))
    if not q_tokens:
        return []
    idf = _cache["idf"]

    scored: list[tuple[float, int]] = []
    for i, d_tokens in enumerate(_cache["tokenized"]):
        if not d_tokens:
            continue
        cnt = Counter(d_tokens)
        norm = math.sqrt(len(d_tokens))
        s = sum(idf.get(t, 0.0) * (cnt[t] / norm) for t in q_tokens if t in cnt)
        if s > 0:  # 与查询没有任何词汇交集时不硬塞无关样本
            scored.append((s, i))
    scored.sort(reverse=True)
    return [data[i] for _, i in scored[:k]]


def stats() -> dict:
    data = load_all()
    return {"pool_size": len(data), "path": str(FEEDBACK_PATH)}
