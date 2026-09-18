"""D5-2 回复生成质量批量评估：RAG 检索 + 生成草稿 vs 标准答案。

指标：token-F1、BLEU-4、ROUGE-L、needs_human 率、引用数。

用法: python scripts/eval_reply.py [--n 100] [--workers 6]
输出: eval/reply_report.json, eval/reply_predictions.csv
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import exp
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.reply import generate_reply
from app.agents.retriever import HybridRetriever
from app.config import PROCESSED_DIR


def tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(pred: str, ref: str) -> float:
    p, r = set(tok(pred)), set(tok(ref))
    if not p or not r:
        return 0.0
    inter = len(p & r)
    if inter == 0:
        return 0.0
    prec, rec = inter / len(p), inter / len(r)
    return 2 * prec * rec / (prec + rec)


def bleu4(pred: str, ref: str) -> float:
    """corpus-less 单句 BLEU-4，平滑处理。"""
    hyp, ref_t = tok(pred), tok(ref)
    if len(hyp) == 0 or len(ref_t) == 0:
        return 0.0
    n = min(4, len(hyp), len(ref_t))
    if n == 0:
        return 0.0
    precisions = []
    for i in range(1, n + 1):
        hyp_n = Counter(tuple(hyp[j : j + i]) for j in range(len(hyp) - i + 1))
        ref_n = Counter(tuple(ref_t[j : j + i]) for j in range(len(ref_t) - i + 1))
        overlap = sum(min(c, ref_n[g]) for g, c in hyp_n.items())
        total = sum(hyp_n.values())
        # Chen-Cherry 平滑：零重叠时 add-1，避免短句全零
        smooth = 1 if overlap == 0 else 0
        precisions.append((overlap + smooth) / (total + smooth))
    bp = 1.0 if len(hyp) >= len(ref_t) else exp(1 - len(ref_t) / max(len(hyp), 1))
    # n-gram 精度的几何平均
    prod = 1.0
    for p in precisions:
        prod *= p
    geo = prod ** (1 / len(precisions))
    return bp * geo


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F1（最长公共子序列）。"""
    hyp, ref_t = tok(pred), tok(ref)
    if not hyp or not ref_t:
        return 0.0
    m, n = len(hyp), len(ref_t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if hyp[i - 1] == ref_t[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    df = pd.read_parquet(PROCESSED_DIR / "eval.parquet")
    df = df.sample(min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    r = HybridRetriever()
    if not r.load():
        print("索引不存在，先运行 scripts/build_index.py")
        sys.exit(1)

    t0 = time.time()
    rows = [None] * len(df)

    def work(i_row):
        i, row = i_row
        hits = r.retrieve(row.text, k=5)
        out = generate_reply(row.text, hits)
        return i, row, out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, p) for p in enumerate(df.itertuples())]
        done = 0
        for fut in as_completed(futures):
            i, row, out = fut.result()
            reply = out["reply"]
            rows[i] = {
                "text": row.text[:150],
                "gold_answer": row.answer[:400],
                "reply": reply[:600],
                "token_f1": round(token_f1(reply, row.answer), 4),
                "bleu4": round(bleu4(reply, row.answer), 4),
                "rouge_l": round(rouge_l(reply, row.answer), 4),
                "needs_human": out["needs_human"],
                "n_sources": len(out["sources"]),
                "confidence": out["confidence"],
                "reply_len": len(tok(reply)),
            }
            done += 1
            if done % 20 == 0:
                print(f"  进度 {done}/{len(df)} | 耗时 {time.time()-t0:.0f}s")

    res = pd.DataFrame(rows)
    report = {
        "n": len(res),
        "token_f1": round(res["token_f1"].mean(), 4),
        "bleu4": round(res["bleu4"].mean(), 4),
        "rouge_l": round(res["rouge_l"].mean(), 4),
        "needs_human_rate": round(res["needs_human"].mean(), 4),
        "avg_sources": round(res["n_sources"].mean(), 2),
        "avg_confidence": round(res["confidence"].mean(), 3),
        "avg_reply_len_tokens": round(res["reply_len"].mean(), 1),
        "cost_seconds": round(time.time() - t0, 1),
    }

    out_dir = Path(__file__).resolve().parent.parent / "eval"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "reply_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    res.to_csv(out_dir / "reply_predictions.csv", index=False)

    print(f"\n=== 回复生成评估 (n={report['n']}) ===")
    for k, v in report.items():
        if k not in ("n", "cost_seconds"):
            print(f"  {k}: {v}")
    print(f"  耗时 {report['cost_seconds']}s | 报告: eval/reply_report.json")


if __name__ == "__main__":
    main()
