"""构建 few-shot 示例集：每 type 选 4 条（含判别性边界样本），覆盖 type×priority 组合。

判别词策略：
- Problem: repeated/recurring/root cause/again/keeps
- Incident: down/not working/error/cannot/failed/broken/outage
- Change: update/upgrade/migrate/deploy/configure/change/implement
- Request: 无故障信号的信息请求
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import DATA_DIR, PROCESSED_DIR

TYPES = ["Incident", "Request", "Problem", "Change"]
PRIORITIES = ["low", "medium", "high"]
SIGNAL_WORDS = {
    "Incident": ["down", "not working", "error", "cannot", "failed", "broken", "outage", "crash"],
    "Request": ["how can i", "information", "question", "inquire", "details", "guidance", "advice"],
    "Problem": ["repeated", "recurring", "root cause", "again", "keeps", "frequent", "pattern", "multiple times"],
    "Change": ["update", "upgrade", "migrate", "deploy", "configure", "implement", "integrate", "install", "change"],
}


def _has_signal(text: str, type_: str) -> bool:
    low = text.lower()
    return any(w in low for w in SIGNAL_WORDS.get(type_, []))


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    df = df[(df["answer"].str.len() > 30) & (df["text"].str.len() > 120) & (df["text"].str.len() < 900)]

    examples: list[dict] = []
    for t in TYPES:
        # 1) 每个 priority 各 1 条（普通样本）
        for p in PRIORITIES:
            sub = df[(df["type"] == t) & (df["priority"] == p)]
            if sub.empty:
                continue
            pick = sub.sort_values("text").head(1)
            for _, r in pick.iterrows():
                examples.append(_make(r))
        # 2) 1 条判别性样本（含该 type 信号词）
        sig = df[(df["type"] == t) & df["text"].apply(lambda x, t=t: _has_signal(x, t))]
        if not sig.empty:
            r = sig.sort_values("text").iloc[0]
            if not any(e["output"].startswith(f'{{"type": "{t}"') and r["text"][:40] in e["input"] for e in examples):
                examples.append(_make(r))

    out = DATA_DIR / "fewshot.json"
    out.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"few-shot 示例: {len(examples)} 条 → {out}")
    for e in examples:
        tag = json.loads(e["output"])
        print(f"  [{tag['type']:9s}|{tag['priority']:7s}] {e['input'][:70].replace(chr(10),' ')}...")


def _make(r) -> dict:
    reason = {
        "Incident": "用户报障，服务故障需尽快恢复",
        "Request": "信息咨询/标准服务请求",
        "Problem": "故障反复出现，需定位根因",
        "Change": "变更系统配置或实施调整",
    }.get(r["type"], "根据工单内容判断")
    return {
        "input": f"工单内容：\n{r['text'][:600]}",
        "output": json.dumps(
            {"type": r["type"], "priority": r["priority"], "reason": reason, "confidence": 0.9},
            ensure_ascii=False,
        ),
    }


if __name__ == "__main__":
    main()
