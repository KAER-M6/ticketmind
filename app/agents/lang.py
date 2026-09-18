"""语言桥接层：让英文知识库支持中文工单输入。

问题背景（实测）：
    知识库是 19,782 条英文历史工单，检索由 BM25 + all-MiniLM-L6-v2（英文模型）组成。
    直接拿中文工单去检索，BM25 的 token 正则只吃 [a-z0-9] 于是命中 0 个词，
    向量侧英文模型对中文输出近似噪声 —— 实测召回 5 条全是不相关工单（乱答）。

解决：中文工单先转写成一段英文检索查询，再进原有检索流程。
    - 只在检测到中日韩字符时触发，英文工单零额外开销、行为完全不变；
    - 转写结果带缓存，同一张工单重复处理（演示重跑 / 批量重叠）不再调 LLM；
    - 转写失败一律回落到原文，不阻断主链路。

顺带提供回复语言判定：回复语言跟随工单语言（中文工单 → 中文草稿），
这样英文评估集（BLEU / ROUGE 对标英文标准答案）的指标仍然可比。
"""
import re
import threading

from app.llm import chat_json

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

TRANSLATE_SYSTEM = """你是检索查询转写器，服务对象是一个英文客服工单知识库。

把用户给出的工单（可能是中文）转写成一段英文检索查询，要求：
1. 只输出英文，保留原始诉求中的所有关键信息：故障现象、受影响范围（多少人/哪个团队）、
   时间与频次线索（第几次、持续多久、是否反复）、涉及的系统或设备、用户明确提出的动作
   （要求修复 / 索要资料 / 申请权限 / 申请变更）。
2. 使用客服工单领域常见英文表达（如 login failure、outage、cannot access、
   shared drive、recurring、root cause、upgrade、migrate），便于关键词匹配。
3. 不要翻译成通顺的商务邮件，也不要添加原文没有的信息，写成信息密集的查询即可。

必须严格输出 JSON：{"query": "英文检索查询"}"""

_cache: dict[str, str] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 512
_stats = {"hits": 0, "misses": 0, "translate_failed": 0}


def has_cjk(text: str) -> bool:
    """文本里是否含中日韩字符（用于判断是否要过桥接层）。"""
    return bool(_CJK_RE.search(text or ""))


def reply_language(text: str, override: str = "") -> str:
    """回复语言：override 为 zh/en 时强制，auto 时跟随工单语言。"""
    o = (override or "auto").strip().lower()
    if o in ("zh", "en"):
        return o
    return "zh" if has_cjk(text) else "en"


def to_retrieval_query(text: str, use_llm: bool = True) -> str:
    """中文工单 → 英文检索查询；英文工单原样返回（不调 LLM）。"""
    if not text:
        return text
    if not has_cjk(text):
        return text

    key = text[:2000]
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        _stats["hits"] += 1
        return hit

    _stats["misses"] += 1
    if not use_llm:
        return text

    query = text
    try:
        out = chat_json(
            f"工单内容：\n{key}",
            system=TRANSLATE_SYSTEM,
            temperature=0.0,
            max_tokens=400,
        )
        got = str(out.get("query", "")).strip()
        # 转写结果必须真的是英文，否则宁可退回原文（原文至少不会更差）
        if got and not has_cjk(got):
            query = got
    except Exception as e:
        _stats["translate_failed"] += 1
        print(f"[lang] 中文查询转写失败，回落原文检索：{type(e).__name__}: {e}")

    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = query
    return query


def stats() -> dict:
    total = _stats["hits"] + _stats["misses"]
    return {
        "bridge_cache_size": len(_cache),
        "bridge_hits": _stats["hits"],
        "bridge_misses": _stats["misses"],
        "bridge_hit_rate": round(_stats["hits"] / total, 4) if total else 0.0,
        "translate_failed": _stats["translate_failed"],
    }
