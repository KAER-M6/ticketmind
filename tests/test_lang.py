"""语言桥接层单测：不联网，全部用 monkeypatch 假 LLM。

覆盖：中文检测、回复语言决策、中文→英文检索查询桥接（含缓存与失败降级）、
中文 tokenize 兜底。
"""
import pytest

from app.agents import lang as lang_mod
from app.agents.retriever import tokenize


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(lang_mod, "_cache", {})
    monkeypatch.setattr(lang_mod, "_stats", {"hits": 0, "misses": 0, "translate_failed": 0})


class TestDetect:
    def test_english_is_not_cjk(self):
        assert lang_mod.has_cjk("Shared drive is down") is False

    def test_chinese_is_cjk(self):
        assert lang_mod.has_cjk("共享网盘又崩了") is True

    def test_mixed_text_counts_as_cjk(self):
        assert lang_mod.has_cjk("VPN 认证失败") is True

    def test_empty_is_safe(self):
        assert lang_mod.has_cjk("") is False


class TestReplyLanguage:
    def test_follows_ticket_language(self):
        assert lang_mod.reply_language("共享网盘又崩了") == "zh"
        assert lang_mod.reply_language("Shared drive is down") == "en"

    def test_override_wins(self):
        assert lang_mod.reply_language("Shared drive is down", "zh") == "zh"
        assert lang_mod.reply_language("共享网盘又崩了", "en") == "en"

    def test_unknown_override_falls_back_to_auto(self):
        assert lang_mod.reply_language("共享网盘又崩了", "fr") == "zh"


class TestRetrievalQueryBridge:
    def test_english_never_calls_llm(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("英文工单不应触发翻译 LLM 调用")

        monkeypatch.setattr(lang_mod, "chat_json", boom)
        text = "Shared drive is down again"
        assert lang_mod.to_retrieval_query(text) == text

    def test_chinese_is_translated(self, monkeypatch):
        monkeypatch.setattr(
            lang_mod, "chat_json", lambda *a, **k: {"query": "shared drive outage recurring"}
        )
        out = lang_mod.to_retrieval_query("共享网盘又中断了")
        assert out == "shared drive outage recurring"

    def test_translation_is_cached(self, monkeypatch):
        calls = {"n": 0}

        def fake(*a, **k):
            calls["n"] += 1
            return {"query": "shared drive outage"}

        monkeypatch.setattr(lang_mod, "chat_json", fake)
        lang_mod.to_retrieval_query("共享网盘又中断了")
        lang_mod.to_retrieval_query("共享网盘又中断了")
        assert calls["n"] == 1
        assert lang_mod.stats()["bridge_hits"] == 1

    def test_llm_failure_falls_back_to_original(self, monkeypatch):
        monkeypatch.setattr(
            lang_mod, "chat_json", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("timeout"))
        )
        text = "共享网盘又中断了"
        assert lang_mod.to_retrieval_query(text) == text
        assert lang_mod.stats()["translate_failed"] == 1

    def test_cjk_in_translation_result_is_rejected(self, monkeypatch):
        """模型没听话、返回了中文 → 宁可退回原文，也不要拿中文去查英文库。"""
        monkeypatch.setattr(lang_mod, "chat_json", lambda *a, **k: {"query": "共享网盘 中断"})
        text = "共享网盘又中断了"
        assert lang_mod.to_retrieval_query(text) == text

    def test_empty_query_returns_empty(self):
        assert lang_mod.to_retrieval_query("") == ""


class TestChineseTokenize:
    def test_chinese_yields_tokens(self):
        toks = tokenize("共享网盘中断")
        assert "共" in toks and "共享" in toks

    def test_english_still_works(self):
        toks = tokenize("The printer is not working")
        assert "printer" in toks and "the" not in toks

    def test_mixed_text_keeps_both_sides(self):
        toks = tokenize("VPN 认证失败")
        assert "vpn" in toks and "认证" in toks
