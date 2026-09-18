"""工单智脑 Web API：FastAPI 入口。

端到端流程：POST /api/process 接收工单 → 分诊 → RAG 检索 → 回复生成 →
存库(drafted) → 人工审核 → 状态流转 → KPI 统计。
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import db
from app.agents import feedback
from app.agents.reply import generate_reply
from app.agents.retriever import get_retriever
from app.agents.triage import PRIORITY_DEFS, TYPE_DEFS, triage

VERSION = "0.2.0"

# 并发度：批量 AI 处理用；DeepSeek 侧并发过高易触发连接抖动，默认 6
IMPORT_WORKERS = int(os.getenv("TICKETMIND_IMPORT_WORKERS", "6"))
# 可选鉴权：设置该环境变量后，/api/* 需带 X-API-Key 头（本地演示默认关闭）
API_KEY = os.getenv("TICKETMIND_API_KEY", "").strip()

app = FastAPI(title="TicketMind 工单智脑", version=VERSION)

# CORS 默认只放行本机前端，不再 allow_origins=["*"]
_origins = os.getenv(
    "TICKETMIND_CORS_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent / "web" / "static"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_OPEN_PATHS = {"/", "/api/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """API Key 校验（仅当 TICKETMIND_API_KEY 已配置时生效）。"""
    path = request.url.path
    if API_KEY and path not in _OPEN_PATHS and not path.startswith("/static"):
        if request.headers.get("X-API-Key", "") != API_KEY:
            return JSONResponse({"detail": "未授权：缺少或错误的 X-API-Key"}, status_code=401)
    return await call_next(request)


def _kb():
    """取共享检索器；索引缺失时返回 503 而不是抛栈。"""
    r = get_retriever()
    if r is None or r.df is None:
        raise HTTPException(503, "检索索引未就绪，请先运行 scripts/build_index.py")
    return r


@app.get("/")
def index():
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>TicketMind</h1><p>index.html not found</p>", status_code=500)


class TicketIn(BaseModel):
    subject: str = ""
    body: str
    ticket_id: str | None = None


class ImportIn(BaseModel):
    n: int = 10
    process: bool = False  # True=逐条走 AI 全链路; False=用历史标准数据快速导入


class ReviewIn(BaseModel):
    """人工审核时可以顺带纠正标签（不传则沿用 AI 判定）。"""
    type: str | None = None
    priority: str | None = None


@app.get("/api/health")
def health():
    r = get_retriever()
    total_cache = (r.cache_hits + r.cache_misses) if r else 0
    return {
        "status": "ok",
        "version": VERSION,
        "kb_size": len(r.df) if (r is not None and r.df is not None) else 0,
        "retrieval_cache": {
            "hits": r.cache_hits if r else 0,
            "misses": r.cache_misses if r else 0,
            "hit_rate": round(r.cache_hits / total_cache, 4) if total_cache else 0.0,
        },
        "feedback_pool": feedback.stats()["pool_size"],
        "auth_required": bool(API_KEY),
    }


@app.post("/api/triage")
def api_triage(t: TicketIn):
    text = f"{t.subject}\n{t.body}".strip()
    return triage(text)


def _process_one(text: str, subject: str, tid: str) -> dict:
    """单条工单的完整 AI 链路（分诊 + 检索 + 草稿），供串行与并行共用。"""
    tri = triage(text)
    hits = _kb().retrieve(text, k=5, mode="hybrid")
    rep = generate_reply(text, hits)
    return {
        "ticket_id": tid,
        "subject": subject,
        "body": text,
        "tri": tri,
        "hits": hits,
        "rep": rep,
        "needs_human": bool(tri.get("needs_human") or rep.get("needs_human")),
    }


@app.post("/api/process")
def api_process(t: TicketIn):
    """端到端：分诊 → 检索 → 生成草稿 → 入库(drafted)。"""
    text = f"{t.subject}\n{t.body}".strip()
    if not text:
        raise HTTPException(400, "工单内容为空")

    tid = t.ticket_id or f"TK-{db.get_conn().execute('SELECT COALESCE(MAX(id),0)+1 FROM tickets').fetchone()[0]:04d}"
    item = _process_one(text, t.subject, tid)
    tri, rep, hits = item["tri"], item["rep"], item["hits"]

    row_id = db.insert_ticket(
        ticket_id=tid,
        subject=t.subject,
        body=t.body,
        pred_type=tri["type"],
        pred_priority=tri["priority"],
        triage_reason=tri["reason"],
        confidence=tri["confidence"],
        draft=rep["reply"],
        sources=rep["sources"],
        needs_human=item["needs_human"],
        decision_source=tri.get("decision_source"),
    )
    return {"id": row_id, "ticket_id": tid, "triage": tri, "reply": rep, "hits": hits}


@app.get("/api/tickets")
def api_tickets(status: str | None = None, limit: int = 50):
    return db.list_tickets(status, limit)


@app.post("/api/tickets/{tid}/approve")
def api_approve(tid: int, req: ReviewIn | None = None):
    """通过审核；可携带纠正后的 type/priority —— 人工标签会回流进 few-shot 语料池。"""
    row = db.get_ticket(tid)
    if row is None:
        raise HTTPException(404, "工单不存在")

    ft = (req.type if req and req.type else None) or row["pred_type"]
    fp = (req.priority if req and req.priority else None) or row["pred_priority"]
    if ft not in TYPE_DEFS:
        raise HTTPException(400, f"非法 type：{ft}（可选 {list(TYPE_DEFS)}）")
    if fp not in PRIORITY_DEFS:
        raise HTTPException(400, f"非法 priority：{fp}（可选 {list(PRIORITY_DEFS)}）")

    db.update_status(tid, "approved", ft, fp)
    corrected = ft != row["pred_type"] or fp != row["pred_priority"]
    feedback.record(
        row["ticket_id"], f"{row['subject']}\n{row['body']}".strip(), ft, fp, corrected
    )
    return {
        "ok": True,
        "corrected": corrected,
        "final": {"type": ft, "priority": fp},
        "feedback_pool": feedback.stats()["pool_size"],
    }


@app.post("/api/tickets/{tid}/reject")
def api_reject(tid: int):
    if db.update_status(tid, "rejected"):
        return {"ok": True}
    raise HTTPException(404, "工单不存在")


@app.post("/api/tickets/{tid}/resolve")
def api_resolve(tid: int):
    if db.update_status(tid, "resolved"):
        return {"ok": True}
    raise HTTPException(404, "工单不存在")


@app.get("/api/stats")
def api_stats():
    return db.stats()


@app.get("/api/feedback/stats")
def api_feedback_stats():
    """反馈池状态：人工确认样本量 + 线上准确率（人工复核口径）。"""
    return {**feedback.stats(), **db.stats()}


@app.post("/api/import")
def api_import(req: ImportIn):
    """批量导入工单：从评估集抽样 N 条入库（drafted）。

    process=False: 用历史标准标签/答复快速导入（模拟从旧工单系统迁移，秒级）。
    process=True:  走真实 AI 链路（分诊 + RAG + 草稿）。并行执行，
                   100 条从「串行约 30 分钟」压到「并行约 3 分钟」。
    """
    import pandas as pd

    from app.config import PROCESSED_DIR

    path = PROCESSED_DIR / "eval.parquet"
    if not path.exists():
        raise HTTPException(500, "评估集不存在，先运行 scripts/preprocess.py")
    df = pd.read_parquet(path)
    n = max(1, min(req.n, 500))
    sampled = df.sample(n, random_state=42).reset_index(drop=True)

    base = db.get_conn().execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    jobs = []
    for i, row in enumerate(sampled.itertuples(), 1):
        text = row.text or ""
        subject = (text.split("\n")[0][:80] if text else "无主题")
        jobs.append((text, subject, f"TK-IMP-{base + i:04d}"))

    imported, failed = 0, 0
    if not req.process:  # 迁移模式：纯写库，无需 AI
        for text, subject, tid in jobs:
            row = sampled.iloc[imported]
            db.insert_ticket(
                ticket_id=tid, subject=subject, body=text,
                pred_type=row["type"], pred_priority=row["priority"],
                triage_reason="历史工单迁移（源数据标注）", confidence=1.0,
                draft=row["answer"] or "（无历史答复）", sources=[],
                needs_human=False,
            )
            imported += 1
        return {"imported": imported, "failed": 0, "mode": "migrate"}

    # AI 模式：并行跑模型链路，写库仍串行（SQLite 单写者）
    with ThreadPoolExecutor(max_workers=min(IMPORT_WORKERS, n)) as pool:
        futures = [pool.submit(_process_one, text, subject, tid) for text, subject, tid in jobs]
        for fut in futures:
            try:
                it = fut.result()
            except Exception as e:
                failed += 1
                print(f"[import] 单条处理失败，已跳过：{type(e).__name__}: {e}")
                continue
            tri, rep = it["tri"], it["rep"]
            db.insert_ticket(
                ticket_id=it["ticket_id"], subject=it["subject"], body=it["body"],
                pred_type=tri["type"], pred_priority=tri["priority"],
                triage_reason=tri["reason"], confidence=tri["confidence"],
                draft=rep["reply"], sources=rep["sources"],
                needs_human=it["needs_human"],
                decision_source=tri.get("decision_source"),
            )
            imported += 1
    return {
        "imported": imported,
        "failed": failed,
        "mode": "ai",
        "workers": min(IMPORT_WORKERS, n),
    }
