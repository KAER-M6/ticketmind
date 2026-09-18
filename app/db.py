"""SQLite 工单存储：状态机 open → drafted → approved/rejected → resolved。

设计要点：数据访问全部收在这一层，换 PostgreSQL 只需替换 _connect() 与 SQL 方言，
业务代码（main.py / agents）零改动。
"""
import json
import sqlite3
from datetime import datetime

from app.config import ROOT_DIR

DB_PATH = ROOT_DIR / "data" / "tickets.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT UNIQUE,
    subject TEXT,
    body TEXT,
    pred_type TEXT,
    pred_priority TEXT,
    triage_reason TEXT,
    confidence REAL,
    draft TEXT,
    sources TEXT,
    needs_human INTEGER DEFAULT 0,
    status TEXT DEFAULT 'drafted',
    created_at TEXT,
    updated_at TEXT
);
"""

# 后加的列：启动时按需 ALTER TABLE（对已有库做轻量迁移，无需重建）
_MIGRATIONS = {
    "final_type": "ALTER TABLE tickets ADD COLUMN final_type TEXT",
    "final_priority": "ALTER TABLE tickets ADD COLUMN final_priority TEXT",
    "corrected": "ALTER TABLE tickets ADD COLUMN corrected INTEGER DEFAULT 0",
    "decision_source": "ALTER TABLE tickets ADD COLUMN decision_source TEXT",
}


def _connect() -> sqlite3.Connection:
    """唯一的连接入口（换库只需改这里）。"""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # 允许「读并发 + 单写」，缓解库级锁
    return conn


def _init(conn: sqlite3.Connection) -> None:
    conn.execute(_SCHEMA)
    have = {r["name"] for r in conn.execute("PRAGMA table_info(tickets)").fetchall()}
    for col, ddl in _MIGRATIONS.items():
        if col not in have:
            conn.execute(ddl)


def get_conn() -> sqlite3.Connection:
    conn = _connect()
    _init(conn)
    return conn


def insert_ticket(
    ticket_id: str,
    subject: str,
    body: str,
    pred_type: str,
    pred_priority: str,
    triage_reason: str,
    confidence: float,
    draft: str,
    sources: list,
    needs_human: bool,
    decision_source: str | None = None,
) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO tickets
            (ticket_id, subject, body, pred_type, pred_priority, triage_reason,
             confidence, draft, sources, needs_human, status, created_at, updated_at,
             decision_source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ticket_id, subject, body, pred_type, pred_priority, triage_reason,
                confidence, draft, json.dumps(sources, ensure_ascii=False),
                int(needs_human), "drafted", now, now, decision_source,
            ),
        )
        return cur.lastrowid


def list_tickets(status: str | None = None, limit: int = 50) -> list[dict]:
    sql = "SELECT * FROM tickets"
    args: list = []
    if status and status != "all":
        sql += " WHERE status = ?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with get_conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    result = [dict(r) for r in rows]
    for r in result:
        r["sources"] = json.loads(r["sources"]) if r["sources"] else []
    return result


def get_ticket(row_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (row_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["sources"] = json.loads(out["sources"]) if out["sources"] else []
    return out


def update_status(
    row_id: int,
    status: str,
    final_type: str | None = None,
    final_priority: str | None = None,
) -> bool:
    """状态流转；传入 final_type/final_priority 表示人工确认（或纠正）了标签。

    人工标签与 AI 判定不一致时 corrected=1，作为「模型错了」的显式信号，
    供反馈回流（few-shot 语料）与准确率统计使用。
    """
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        row = conn.execute(
            "SELECT pred_type, pred_priority FROM tickets WHERE id = ?", (row_id,)
        ).fetchone()
        if row is None:
            return False
        ft = final_type or row["pred_type"]
        fp = final_priority or row["pred_priority"]
        corrected = int(ft != row["pred_type"] or fp != row["pred_priority"])
        cur = conn.execute(
            """UPDATE tickets SET status = ?, updated_at = ?,
            final_type = ?, final_priority = ?, corrected = ? WHERE id = ?""",
            (status, now, ft, fp, corrected, row_id),
        )
        return cur.rowcount > 0


def feedback_rows(limit: int = 500) -> list[dict]:
    """人工已确认（approved/resolved）的样本，用于反馈回流 few-shot。

    只取人工最终确认的标签（final_*），没有则回落到 AI 判定。
    """
    sql = """
    SELECT ticket_id, body, COALESCE(final_type, pred_type) AS type,
           COALESCE(final_priority, pred_priority) AS priority,
           COALESCE(corrected, 0) AS corrected, updated_at
    FROM tickets
    WHERE status IN ('approved','resolved')
    ORDER BY updated_at DESC LIMIT ?
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def stats() -> dict:
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        by_status = {
            r["status"]: r["n"]
            for r in conn.execute(
                "SELECT status, COUNT(*) n FROM tickets GROUP BY status"
            ).fetchall()
        }
        by_type = {
            r["pred_type"]: r["n"]
            for r in conn.execute(
                "SELECT pred_type, COUNT(*) n FROM tickets GROUP BY pred_type"
            ).fetchall()
        }
        needs_human = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE needs_human = 1"
        ).fetchone()[0]
        approved = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE status IN ('approved','resolved')"
        ).fetchone()[0]
        reviewed = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE final_type IS NOT NULL"
        ).fetchone()[0]
        corrected = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE corrected = 1"
        ).fetchone()[0]
        ai_ok = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE final_type IS NOT NULL AND corrected = 0"
        ).fetchone()[0]
    return {
        "total": total,
        "by_status": by_status,
        "by_type": by_type,
        "needs_human": needs_human,
        "approved": approved,
        "automation_coverage": round(approved / total, 4) if total else 0.0,
        # 人工复核口径的线上准确率（只统计已被人工确认过的工单）
        "reviewed": reviewed,
        "corrected": corrected,
        "online_accuracy": round(ai_ok / reviewed, 4) if reviewed else None,
    }
