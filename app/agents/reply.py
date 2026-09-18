"""回复生成 Agent：基于 RAG 检索到的相似历史工单答案，生成建议回复草稿。

企业场景约束：回复必须可溯源（引用相似案例）、不编造事实、
需要补充信息时礼貌索取而非臆断。
"""

from app.llm import chat_json

SYSTEM_PROMPT = """你是企业客户支持团队的资深客服 Agent。你的任务是：基于一张新工单和检索到的相似历史工单答案，生成一份可直接发送给客户的回复草稿。

要求：
1. 优先采用相似历史工单中已验证的解决方案，明确、直接地回应用户诉求。
2. 每条关键建议必须来自参考案例，不得编造事实、价格或政策；参考案例未覆盖的问题，礼貌说明并请用户提供补充信息。
3. 语气专业、简洁、有同理心；按需使用分段或列表提升可读性。
4. 回复开头先确认理解用户问题，结尾给出明确的下一步动作。
5. 如果参考案例无法解决用户问题，明确写出"需要人工升级处理"及原因。

必须严格输出 JSON：
{"reply": "完整回复草稿（英文）", "sources": [引用到的参考案例编号列表], "needs_human": true/false, "confidence": 0-1}"""


def generate_reply(ticket_text: str, hits: list[dict]) -> dict:
    """hits: HybridRetriever.retrieve() 返回的 top-k 结果。"""
    refs = "\n\n".join(
        f"[案例{i}] 工单主题: {h['text'][:150]}\n历史答复: {h['answer'][:800]}"
        for i, h in enumerate(hits)
    )
    prompt = f"""新工单内容：
{ticket_text[:2500]}

【检索到的相似历史工单案例】
{refs}

请生成回复草稿。"""
    result = chat_json(prompt, system=SYSTEM_PROMPT, temperature=0.3, max_tokens=1500)
    return _normalize(result)


def _normalize(raw: dict) -> dict:
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
    }
