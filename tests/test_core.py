"""核心逻辑单测：不依赖网络与 LLM，验证解析容错、RRF 融合、评估指标、状态机。

跑法：pytest -q
"""
import numpy as np
import pytest

from app import db
from app.agents import reply as reply_mod
from app.agents import triage as triage_mod
from app.agents.retriever import _rrf, tokenize
from scripts.eval_reply import bleu4, rouge_l, token_f1


# ---------- 分诊解析容错 ----------
class TestTriageNormalize:
    def test_valid(self):
        out = triage_mod._normalize(
            {"type": "Incident", "priority": "high", "reason": "服务中断", "confidence": 0.92}
        )
        assert out["type"] == "Incident"
        assert out["priority"] == "high"
        assert out["confidence"] == 0.92

    def test_case_insensitive_type(self):
        assert triage_mod._normalize({"type": "incident"})["type"] == "Incident"
        assert triage_mod._normalize({"type": "CHANGE"})["type"] == "Change"

    def test_unknown_values_fall_back(self):
        out = triage_mod._normalize({"type": "Bug", "priority": "urgent"})
        assert out["type"] == "Request"
        assert out["priority"] == "medium"

    def test_bad_confidence_does_not_raise(self):
        assert triage_mod._normalize({"type": "Request", "confidence": "N/A"})["confidence"] == 0.5
        assert triage_mod._normalize({"type": "Request", "confidence": None})["confidence"] == 0.5

    def test_confidence_clamped(self):
        assert triage_mod._normalize({"type": "Request", "confidence": 3})["confidence"] == 1.0
        assert triage_mod._normalize({"type": "Request", "confidence": -1})["confidence"] == 0.0

    def test_reason_truncated(self):
        out = triage_mod._normalize({"type": "Request", "reason": "啊" * 500})
        assert len(out["reason"]) <= 100

    def test_llm_failure_degrades_to_human_review(self, monkeypatch):
        """LLM 抛异常时不能把异常抛给调用方，必须降级并标记人工。"""
        monkeypatch.setattr(
            triage_mod, "chat_json", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad json"))
        )
        monkeypatch.setattr(
            triage_mod, "build_examples_with_meta", lambda *a, **k: ([], {"feedback": 0, "retrieved": 0})
        )
        monkeypatch.setattr(triage_mod, "load_fewshot", lambda *a, **k: [])
        out = triage_mod.triage("printer broken")
        assert out["needs_human"] is True
        assert out["confidence"] == 0.0
        assert out["decision_source"] == triage_mod.DECISION_SOURCE_FALLBACK


# ---------- 回复解析容错 ----------
class TestReplyNormalize:
    def test_empty_reply_forces_human(self):
        assert reply_mod._normalize({"reply": "  "})["needs_human"] is True

    def test_sources_type_guard(self):
        assert reply_mod._normalize({"reply": "hi", "sources": "1,2"})["sources"] == []

    def test_bad_confidence(self):
        assert reply_mod._normalize({"reply": "hi", "confidence": "high"})["confidence"] == 0.5


# ---------- 检索 ----------
class TestRetrieverHelpers:
    def test_tokenize_drops_stopwords_and_short(self):
        toks = tokenize("The printer is not working on my MacBook")
        assert "the" not in toks and "is" not in toks and "on" not in toks
        assert "printer" in toks and "macbook" in toks

    def test_rrf_top1_has_highest_score(self):
        scores = np.array([0.1, 0.9, 0.5, 0.3])
        fused = _rrf(scores, top=4)
        assert max(fused, key=fused.get) == 1
        assert fused[1] > fused[2] > fused[3]

    def test_rrf_limits_to_top(self):
        assert len(_rrf(np.arange(10.0), top=3)) == 3


# ---------- 评估指标 ----------
class TestMetrics:
    def test_identical_text_scores_high(self):
        s = "please reset my password for the account"
        assert token_f1(s, s) == pytest.approx(1.0)
        assert rouge_l(s, s) == pytest.approx(1.0)

    def test_disjoint_text_scores_zero(self):
        assert token_f1("aaa bbb", "ccc ddd") == 0.0

    def test_bleu_smoothed_not_all_zero_on_short_text(self):
        assert bleu4("reset my password now", "reset my password") > 0.0

    def test_bleu4_perfect_match(self):
        s = "reset my password please now"
        assert bleu4(s, s) == pytest.approx(1.0, abs=0.05)


# ---------- 数据库状态机 ----------
class TestStateMachine:
    @pytest.fixture(autouse=True)
    def _tmp_db(self, tmp_path, monkeypatch):
        monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")

    def _insert(self, tid="T-1"):
        return db.insert_ticket(
            ticket_id=tid, subject="s", body="b", pred_type="Incident",
            pred_priority="high", triage_reason="r", confidence=0.9,
            draft="d", sources=[{"n": 1}], needs_human=False,
        )

    def test_insert_and_list(self):
        rid = self._insert()
        rows = db.list_tickets()
        assert len(rows) == 1
        assert rows[0]["id"] == rid
        assert rows[0]["status"] == "drafted"
        assert rows[0]["sources"] == [{"n": 1}]

    def test_status_flow_and_coverage(self):
        rid = self._insert()
        assert db.update_status(rid, "approved") is True
        assert db.stats()["by_status"]["approved"] == 1
        assert db.stats()["automation_coverage"] == 1.0

    def test_update_missing_ticket_returns_false(self):
        assert db.update_status(999, "approved") is False

    def test_filter_by_status(self):
        self._insert("T-1")
        rid = self._insert("T-2")
        db.update_status(rid, "resolved")
        assert len(db.list_tickets("drafted")) == 1
        assert len(db.list_tickets("resolved")) == 1
        assert len(db.list_tickets("all")) == 2
