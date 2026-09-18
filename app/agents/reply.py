"""回复生成 Agent：基于 RAG 检索到的相似历史工单答案，生成建议回复草稿。

企业场景约束：回复必须可溯源（引用相似案例）、不编造事实、
需要补充信息时礼貌索取而非臆断。

语言策略：默认跟随工单语言（中文工单 → 中文草稿）。
历史案例答案都是英文，所以中文工单这一路必须在 prompt 里显式要求
「转写为中文」而不是照抄英文原文。可用 TICKETMIND_REPLY_LANG 强制 zh / en。
"""

from app.agents.lang import reply_language
from app.config import REPLY_LANG
from app.llm import chat_json

SYSTEM_PROMPT = """你是企业客户支持团队的资深客服 Agent。你的任务是：基于一张新工单和检索到的相似历史工单答案，生成一份可直接发送给客户的回复草稿。

要求：
1. 优先采用相似历史工单中已验证的解决方案，明确、直接地回应用户诉求。
2. 每条关键建议必须来自参考案例，不得编造事实、价格或政策；参考案例未覆盖的问题，礼貌说明并请用户提供补充信息。
3. 语气专业、简洁、有同理心；按需使用分段或列表提升可读性。
4. 回复开头先确认理解用户问题，结尾给出明确的下一步动作。
5. 如果参考案例无法解决用户问题，明确写出"需要人工升级处理"及原因。
6. 语言：回复语言必须与工单语言一致。工单是中文就写全中文，工单是英文就写英文。
   注意参考案例的历史答复是英文——若工单为中文，必须把解决方案完整转写成中文表达，
   不得直接粘贴英文原文，也不要中英混排（专有名词、产品名可保留原文）。

必须严格输出 JSON：
{"reply": "完整回复草稿（语言与工单一致）", "sources": [引用到的参考案例编号列表], "needs_human": true/false, "confidence": 0-1}"""

_LANG_INSTRUCTION = {
    "zh": "请用中文生成回复草稿。",
    "en": "Generate the reply draft in English.",
}


def generate_reply(ticket_text: str, hits: list[dict]) -> dict:
    """hits: HybridRetriever.retrieve() 返回的 top-k 结果。"""
    refs = "\n\n".join(
        f"[案例{i}] 工单主题: {h['text'][:150]}\n历史答复: {h['answer'][:800]}"
        for i, h in enumerate(hits)
    )
    lang = reply_language(ticket_text, REPLY_LANG)
    prompt = f"""新工单内容：
{ticket_text[:2500]}

【检索到的相似历史工单案例】
{refs}

{_LANG_INSTRUCTION[lang]}
请生成回复草稿。"""
    result = chat_json(prompt, system=SYSTEM_PROMPT, temperature=0.3, max_tokens=1500)
    return _normalize(result, lang)


def _normalize(raw: dict, lang: str = "auto") -> dict:
    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    reply = str(raw.get("reply", "")).strip()
    return {
        "reply": reply,
        "sources": raw.get("sources", []) if isinstance(raw.get("sources"), list) else [],
        # 模型返回空草稿时也必须转人工，不能往库里塞空回复
        "needs_human": bool(raw.get("needs_human", False)) or not reply,
        "confidence": max(0.0, min(1.0, conf)),
        "lang": lang,
    }
