"""反馈回流闭环单测：审核纠正 → 落池 → 分诊优先使用人工样本 → 统计线上准确率。"""
import json

import pytest

from app import db
from app.agents import feedback
from app.agents import triage as triage_mod


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """反馈池与业务库都指到临时目录，测试不碰真实数据。"""
    monkeypatch.setattr(feedback, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(feedback, "_cache", {"mtime": None, "rows": [], "tokenized": [], "idf": {}})
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")


# ---------- 反馈池 ----------
class TestFeedbackPool:
    def test_record_and_load(self):
        feedback.record("TK-1", "printer not detected on macbook", "Incident", "medium", False)
        rows = feedback.load_all()
        assert len(rows) == 1
        assert rows[0]["type"] == "Incident"
        assert rows[0]["corrected"] is False

    def test_dedupe_keeps_latest(self):
        feedback.record("TK-1", "printer issue", "Incident", "medium", False)
        feedback.record("TK-1", "printer issue", "Request", "low", True)
        rows = feedback.load_all()
        assert len(rows) == 1
        assert rows[0]["type"] == "Request" and rows[0]["corrected"] is True

    def test_blank_text_ignored(self):
        feedback.record("TK-2", "   ", "Request", "low", False)
        assert feedback.load_all() == []

    def test_corrupt_line_skipped(self):
        feedback.record("TK-3", "vpn auth fails", "Incident", "high", False)
        with feedback.FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write("{not json}\n")
        assert len(feedback.load_all()) == 1

    def test_similar_ranks_relevant_first(self):
        feedback.record("TK-1", "printer not detected after windows update", "Incident", "medium", False)
        feedback.record("TK-2", "please reset my account password", "Request", "low", False)
        hit = feedback.similar("my printer is not detected after the update", k=1)
        assert len(hit) == 1
        assert hit[0]["ticket_id"] == "TK-1"

    def test_similar_without_lexical_overlap_returns_none(self):
        feedback.record("TK-1", "printer not detected", "Incident", "medium", False)
        assert feedback.similar("zzz qqq unrelated words xxx", k=2) == []

    def test_stats_pool_size(self):
        feedback.record("TK-1", "a printer issue on macbook", "Incident", "medium", False)
        assert feedback.stats()["pool_size"] == 1


# ---------- 分诊示例组装 ----------
class TestTriageExampleAssembly:
    def test_feedback_examples_take_priority(self, monkeypatch):
        class _StubRetriever:
            def retrieve(self, text, k=5, mode="hybrid"):
                return [{"text": "history ticket", "type": "Request", "priority": "low",
                         "queue": "q", "answer": "a", "score": 0.1}] * k

        monkeypatch.setattr(triage_mod, "_get_retriever", lambda: _StubRetriever())
        feedback.record("TK-1", "shared drive keeps failing repeatedly every week", "Problem", "high", True)
        examples, meta = triage_mod.build_examples_with_meta("shared drive failing again this week")
        assert meta["feedback"] == 1
        assert examples[0]["output"].find("Problem") >= 0  # 人工样本排在最前面
        assert len(examples) <= triage_mod._TOTAL_K

    def test_falls_back_to_retrieval_when_no_feedback(self, monkeypatch):
        class _StubRetriever:
            def retrieve(self, text, k=5, mode="hybrid"):
                return [{"text": "history ticket", "type": "Request", "priority": "low",
                         "queue": "q", "answer": "a", "score": 0.1}] * k

        monkeypatch.setattr(triage_mod, "_get_retriever", lambda: _StubRetriever())
        examples, meta = triage_mod.build_examples_with_meta("anything")
        assert meta["feedback"] == 0 and meta["retrieved"] == triage_mod._TOTAL_K
        assert len(examples) == triage_mod._TOTAL_K

    def test_no_retriever_and_no_feedback_returns_empty(self, monkeypatch):
        monkeypatch.setattr(triage_mod, "_get_retriever", lambda: None)
        examples, meta = triage_mod.build_examples_with_meta("anything")
        assert examples == [] and meta["feedback"] == 0

    def test_triage_reports_example_meta(self, monkeypatch):
        monkeypatch.setattr(triage_mod, "_get_retriever", lambda: None)
        monkeypatch.setattr(triage_mod, "load_fewshot", lambda *a, **k: [])
        monkeypatch.setattr(
            triage_mod, "chat_json",
            lambda *a, **k: {"type": "Incident", "priority": "high", "reason": "r", "confidence": 0.9},
        )
        out = triage_mod.triage("server down for all users")
        assert out["examples"] == {"feedback": 0, "retrieved": 0}
        assert out["decision_source"] == triage_mod.DECISION_SOURCE_STATIC


# ---------- 审核纠正 → 落库 + 落池 ----------
class TestReviewCorrection:
    def _ticket(self, pred_type="Incident", pred_priority="high"):
        return db.insert_ticket(
            ticket_id="TK-1", subject="printer", body="printer not detected",
            pred_type=pred_type, pred_priority=pred_priority, triage_reason="r",
            confidence=0.9, draft="d", sources=[], needs_human=False,
            decision_source="dynamic_fewshot",
        )

    def test_approve_without_correction(self):
        rid = self._ticket()
        assert db.update_status(rid, "approved") is True
        row = db.get_ticket(rid)
        assert row["final_type"] == "Incident" and row["corrected"] == 0
        assert db.stats()["online_accuracy"] == 1.0

    def test_approve_with_correction_marks_and_tracks(self):
        rid = self._ticket(pred_type="Incident")
        db.update_status(rid, "approved", "Request", "low")
        row = db.get_ticket(rid)
        assert row["final_type"] == "Request" and row["final_priority"] == "low"
        assert row["corrected"] == 1
        s = db.stats()
        assert s["reviewed"] == 1 and s["corrected"] == 1 and s["online_accuracy"] == 0.0

    def test_feedback_rows_use_final_labels(self):
        rid = self._ticket(pred_type="Incident", pred_priority="high")
        db.update_status(rid, "approved", "Problem", "medium")
        rows = db.feedback_rows()
        assert len(rows) == 1
        assert rows[0]["type"] == "Problem" and rows[0]["priority"] == "medium"
        assert rows[0]["corrected"] == 1

    def test_feedback_rows_only_reviewed(self):
        self._ticket()
        assert db.feedback_rows() == []

    def test_decision_source_persisted(self):
        rid = self._ticket()
        assert db.get_ticket(rid)["decision_source"] == "dynamic_fewshot"

    def test_jsonl_roundtrip_shape(self):
        """落盘格式必须能被 feedback.load_all 直接读回。"""
        feedback.record("TK-9", "body text", "Change", "medium", True)
        line = feedback.FEEDBACK_PATH.read_text(encoding="utf-8").strip()
        obj = json.loads(line)
        assert set(["ticket_id", "text", "type", "priority", "corrected", "ts"]) <= set(obj)
