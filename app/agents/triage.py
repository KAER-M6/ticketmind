"""分诊 Agent：工单分类（type）+ 定级（priority）+ 依据（reason）。

基于 ITIL 4 语义，type 四分类；priority 与 en 数据集分布对齐（3 级）。
支持 few-shot 示例提升分类准确率。
"""
import json
from pathlib import Path

from app.llm import chat_json

TYPE_DEFS = {
    "Incident": "服务中断或质量下降，用户报障，需要恢复服务（如：登录失败、系统宕机、报错）",
    "Request": "用户提出的服务请求或信息咨询，标准流程可满足（如：要资料、开通权限、询问流程、发票问题）",
    "Problem": "反复出现的同类故障或底层根因问题，需要定位根源（如：每周都出现的同一种报错、多个类似报障背后的共同原因）",
    "Change": "变更请求，需要修改现有配置、基础设施或系统（如：升级版本、调整配置、部署变更）",
}

# 与 en 子集真实分布对齐：low / medium / high 三级
PRIORITY_DEFS = {
    "low": "轻微影响，无关键业务受损，可常规排队处理",
    "medium": "部分用户或局部业务受影响，需尽快处理",
    "high": "多数用户或关键业务受影响/中断，需要优先响应",
}

SYSTEM_PROMPT = f"""你是企业 ITSM 工单分诊系统，负责对每张工单自动分类与定级。

【分类标准】type 只能是以下四选一：
{json.dumps(TYPE_DEFS, ensure_ascii=False, indent=2)}

【定级标准】priority 只能是以下三选一：
{json.dumps(PRIORITY_DEFS, ensure_ascii=False, indent=2)}

判断原则：
1. 先看用户实际诉求：报障（一次性故障）→ Incident；要东西/问流程 → Request；反复出现或找根因 → Problem；改系统配置 → Change。
2. Incident 与 Problem 的边界：
   - 出现 "down/not working/error/cannot/failed/broken/crash/outage" 等单一故障描述 → Incident
   - 出现 "repeated/recurring/root cause/keeps happening/again/frequent/pattern/multiple times" 等反复或根因描述 → Problem
3. Change 的信号：update/upgrade/migrate/deploy/configure/implement/integrate/install 等"要动系统、改配置"的请求 → Change；纯信息咨询 → Request。
4. priority：出现 data breach/security/outage/critical/urgent/大量用户受影响/数据丢失 → high；纯咨询、索要文档资料 → low；其余常规问题 → medium。
5. 信息不足时保守处理：Request + medium。

必须严格输出 JSON 对象，格式：
{{"type": "Incident|Request|Problem|Change", "priority": "low|medium|high", "reason": "不超过50字的判断依据", "confidence": 0-1}}"""

FEWSHOT_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "fewshot.json"


def load_fewshot(max_n: int = 20) -> list[dict]:
    if not FEWSHOT_PATH.exists():
        return []
    examples = json.loads(FEWSHOT_PATH.read_text(encoding="utf-8"))
    return examples[:max_n]


def _get_retriever():
    """取全进程共享的混合检索器（见 retriever.get_retriever），失败返回 None。"""
    try:
        from app.agents.retriever import get_retriever

        return get_retriever()
    except Exception:
        return None


FEEDBACK_K = 2  # 动态 few-shot 中优先给人工确认样本留 2 个位置
_TOTAL_K = 5    # few-shot 总条数，与基线一致（成本不变）


def _feedback_examples(text: str, k: int = FEEDBACK_K) -> list[dict]:
    """人工审核确认过的相似样本（最有价值的示例，优先级最高）。"""
    try:
        from app.agents import feedback

        rows = feedback.similar(text, k=k)
    except Exception:
        return []
    out = []
    for r in rows:
        out.append(
            {
                "input": f"工单内容：\n{r['text'][:800]}",
                "output": json.dumps(
                    {
                        "type": r["type"],
                        "priority": r["priority"],
                        "reason": "人工审核确认过的同类工单，沿用该判定",
                        "confidence": 0.95,
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return out


def build_examples_with_meta(text: str, k: int = _TOTAL_K) -> tuple[list[dict], dict]:
    """构造动态 few-shot 示例，返回 (示例列表, 来源统计)。

    组成：最多 2 条人工确认样本（反馈回流）+ 若干条检索到的历史工单，
    总数不超过 k，保证 token 成本与固定 few-shot 基线相当。
    """
    examples = _feedback_examples(text, k=FEEDBACK_K)
    n_fb = len(examples)

    hits: list[dict] = []
    r = _get_retriever()
    if r is not None:
        hits = r.retrieve(text, k=max(0, k - n_fb))

    for h in hits:
        examples.append(
            {
                "input": f"工单内容：\n{h['text'][:800]}",
                "output": json.dumps(
                    {
                        "type": h["type"],
                        "priority": h["priority"],
                        "reason": "与该历史工单场景相似，沿用同类判定",
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                ),
            }
        )
    return examples[:k], {"feedback": n_fb, "retrieved": len(hits)}


def build_dynamic_examples(text: str, k: int = _TOTAL_K) -> list[dict]:
    """检索相似工单（含人工反馈样本）构造 few-shot 消息对。"""
    return build_examples_with_meta(text, k=k)[0]


DECISION_SOURCE_DYNAMIC = "dynamic_fewshot"
DECISION_SOURCE_STATIC = "static_fewshot"
DECISION_SOURCE_FALLBACK = "fallback_manual_review"


def triage(text: str, examples: list[dict] | None = None) -> dict:
    """对单条工单做分类定级。

    默认动态 few-shot（检索相似历史工单做示例，评估准确率 90.8%），
    检索器不可用时回退静态 few-shot。
    LLM 输出无法解析时不再抛异常中断服务，而是返回保守判定并标记
    needs_human=True，交由人工审核介入。
    """
    source = DECISION_SOURCE_STATIC
    meta = {"feedback": 0, "retrieved": 0}
    if examples is None:
        examples, meta = build_examples_with_meta(text)
        if examples:
            source = DECISION_SOURCE_DYNAMIC
        else:
            examples = load_fewshot()

    prompt = f"工单内容：\n{text[:2500]}"
    try:
        result = chat_json(prompt, system=SYSTEM_PROMPT, examples=examples, temperature=0.0)
    except Exception as e:  # JSON 解析失败 / API 异常 → 降级，不让整条链路 500
        print(f"[triage] 判定失败，转人工复核：{type(e).__name__}: {e}")
        return {
            "type": "Request",
            "priority": "medium",
            "reason": "模型输出异常，需人工确认",
            "confidence": 0.0,
            "needs_human": True,
            "decision_source": DECISION_SOURCE_FALLBACK,
        }

    out = _normalize(result)
    out["needs_human"] = bool(out.get("confidence", 0.5) < 0.6)
    out["decision_source"] = source
    out["examples"] = meta  # 示例来源：人工反馈 n 条 + 历史检索 m 条
    return out


def _normalize(raw: dict) -> dict:
    t = str(raw.get("type", "")).strip()
    p = str(raw.get("priority", "")).strip().lower()
    try:
        conf = float(raw.get("confidence", 0.5))
    except (TypeError, ValueError):
        conf = 0.5
    if not t:
        t = "Request"
    # 只做首字母规范化，未知值保守回落到 Request/medium
    for k in TYPE_DEFS:
        if t.lower() == k.lower():
            t = k
            break
    return {
        "type": t if t in TYPE_DEFS else "Request",
        "priority": p if p in PRIORITY_DEFS else "medium",
        "reason": str(raw.get("reason", ""))[:100],
        "confidence": max(0.0, min(1.0, conf)),
    }
