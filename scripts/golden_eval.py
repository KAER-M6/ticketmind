"""Golden 回归集：20 条固定工单，改 prompt / 换模型后跑一遍，防止优化 A 打坏 B。

用途：
    python scripts/golden_eval.py                # 跑全部 20 条，阈值 0.85
    python scripts/golden_eval.py --limit 5      # 只跑前 5 条（快速冒烟）
    python scripts/golden_eval.py --threshold 0.9

退出码：0 = 达标；1 = 低于阈值（可直接挂到 CI 或 pre-push）
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.triage import triage  # noqa: E402

# 20 条覆盖四类 + 边界（Incident vs Problem、Request vs Change）
CASES = [
    # --- Incident：一次性故障，报障求恢复 ---
    {"id": 1, "type": "Incident", "priority": "high", "text":
     "Subject: Login page returns 500 error for all users\nBody: Since 08:00 this morning nobody in the company can log in to the portal. The login page shows a 500 internal server error. This blocks all work."},
    {"id": 2, "type": "Incident", "priority": "medium", "text":
     "Subject: Outlook crashes when opening attachments\nBody: Outlook closes unexpectedly every time I open an .xlsx attachment. Restarting did not help. Only my account seems affected."},
    {"id": 3, "type": "Incident", "priority": "high", "text":
     "Subject: VPN authentication error\nBody: I cannot connect to the corporate VPN. It says 'authentication failed' although my password is correct and works for email. I am working from home and cannot reach internal systems."},
    {"id": 4, "type": "Incident", "priority": "high", "text":
     "Subject: Website is down, customers cannot complete checkout\nBody: Our storefront is returning a database connection error. Every checkout attempt fails right now. We are losing orders every minute."},
    {"id": 5, "type": "Incident", "priority": "medium", "text":
     "Subject: Printer not detected after Windows update\nBody: After the latest Windows update my laptop no longer detects the office printer. It worked yesterday. Please help me restore printing."},
    {"id": 6, "type": "Incident", "priority": "medium", "text":
     "Subject: Email attachments are not delivered to external recipients\nBody: Since this morning mails with attachments to external addresses bounce back, while internal mails work fine. Please fix this."},
    # --- Request：索要资料/权限/流程咨询，标准流程可满足 ---
    {"id": 7, "type": "Request", "priority": "low", "text":
     "Subject: How do I export my order history to Excel?\nBody: Hello, I would like to download a record of all my purchases from the past year for accounting purposes. Is there a way to export this as a spreadsheet? Thanks!"},
    {"id": 8, "type": "Request", "priority": "medium", "text":
     "Subject: Please grant access to the shared HR folder\nBody: I joined the HR team this week and need read access to the shared HR drive to do my job. My manager has approved this request."},
    {"id": 9, "type": "Request", "priority": "low", "text":
     "Subject: Need copies of last year's invoices\nBody: Could you send me the invoices for all orders placed in 2025? Our finance department needs them for the annual audit."},
    {"id": 10, "type": "Request", "priority": "low", "text":
     "Subject: What is the process to request a new laptop?\nBody: My current laptop is four years old. Could you tell me the approval process and lead time for ordering a replacement device?"},
    {"id": 11, "type": "Request", "priority": "medium", "text":
     "Subject: Please reset my password\nBody: I forgot my account password and I am locked out after several attempts. Please reset it and send me instructions."},
    # --- Problem：反复出现 / 找根因 ---
    {"id": 12, "type": "Problem", "priority": "high", "text":
     "Subject: Shared drive down again - third time this week\nBody: This is the third outage of our shared drive since Monday. The whole sales team cannot access files right now. Previous tickets were closed after a reboot but the problem keeps coming back. We need a permanent fix, not another restart."},
    {"id": 13, "type": "Problem", "priority": "medium", "text":
     "Subject: Recurring sync errors every Monday morning\nBody: The ERP sync has failed every Monday morning for the last three months with the same timeout message. We restart it manually each time. Nobody has looked at the root cause yet."},
    {"id": 14, "type": "Problem", "priority": "medium", "text":
     "Subject: Same invoice PDF corruption keeps happening\nBody: Multiple users report corrupted invoice PDFs again and again over the past weeks. The file is regenerated and works for a while, then breaks again. We suspect a deeper issue in the report service."},
    {"id": 15, "type": "Problem", "priority": "high", "text":
     "Subject: Repeated database timeouts every night\nBody: Our reporting database times out every night around 02:00. This has happened for weeks with growing frequency. We need a root cause analysis instead of daily manual restarts."},
    {"id": 16, "type": "Problem", "priority": "medium", "text":
     "Subject: Application keeps crashing intermittently for months\nBody: Several tickets about the same crash have been closed separately without a solution. It happens again and again across different users and machines. Please investigate the common pattern."},
    # --- Change：要动系统 / 改配置 ---
    {"id": 17, "type": "Change", "priority": "medium", "text":
     "Subject: Please upgrade our PostgreSQL from 13 to 16\nBody: We would like to schedule a database version upgrade next weekend during the maintenance window. Please assess the impact and prepare a rollback plan."},
    {"id": 18, "type": "Change", "priority": "medium", "text":
     "Subject: Migrate the on-prem file server to SharePoint\nBody: We plan to migrate about 2 TB of department files from the local file server to SharePoint Online. Please confirm the required steps and timeline."},
    {"id": 19, "type": "Change", "priority": "high", "text":
     "Subject: Deploy the new payment service to production\nBody: The new payment microservice is ready for production. We request deployment to the production cluster and a configuration change for the new endpoints."},
    {"id": 20, "type": "Change", "priority": "medium", "text":
     "Subject: Change SSO configuration to enforce MFA for all admins\nBody: Security asked us to update the SSO configuration so that multi-factor authentication is mandatory for all administrator accounts. Please implement this configuration change."},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--threshold", type=float, default=0.85, help="type 准确率阈值")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="eval/golden_report.json")
    args = ap.parse_args()

    cases = CASES[: args.limit] if args.limit else CASES

    def run(c):
        try:
            r = triage(c["text"])
            return {**c, "pred_type": r["type"], "pred_priority": r["priority"],
                    "confidence": r["confidence"], "source": r.get("decision_source"),
                    "ok": r["type"] == c["type"], "error": None}
        except Exception as e:  # 单条失败不影响整体报告
            return {**c, "pred_type": None, "pred_priority": None, "confidence": 0.0,
                    "source": None, "ok": False, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, cases))

    n = len(results)
    correct = sum(r["ok"] for r in results)
    acc = correct / n if n else 0.0
    pri_hit = sum(r["pred_priority"] == r["priority"] for r in results) / n if n else 0.0

    print(f"\n{'id':>3} {'期望':<9} {'预测':<9} {'优先级':<8} {'置信':<6} 结果")
    for r in results:
        flag = "OK " if r["ok"] else "MISS"
        pred = r["pred_type"] or f"ERR({r['error'][:20]})"
        print(f"{r['id']:>3} {r['type']:<9} {pred:<9} {str(r['pred_priority']):<8} "
              f"{r['confidence']:<6} {flag}")

    print(f"\ntype 准确率 {correct}/{n} = {acc:.1%}  (阈值 {args.threshold:.0%})")
    print(f"priority 命中 {pri_hit:.1%}（参考值，不作为门禁）")

    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {"n": n, "correct": correct, "type_accuracy": round(acc, 4),
             "priority_accuracy": round(pri_hit, 4), "threshold": args.threshold,
             "passed": acc >= args.threshold,
             "misses": [{"id": r["id"], "expect": r["type"], "pred": r["pred_type"],
                         "text": r["text"][:120]} for r in results if not r["ok"]]},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"报告已写入 {out}")

    if acc < args.threshold:
        print(f"\n❌ 回归门禁未通过：{acc:.1%} < {args.threshold:.0%}")
        return 1
    print("\n✅ 回归门禁通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
