"""Golden 回归集：固定工单，改 prompt / 换模型后跑一遍，防止优化 A 打坏 B。

用途：
    python scripts/golden_eval.py                  # 跑英文 20 条，阈值 0.85
    python scripts/golden_eval.py --set zh         # 跑中文 12 条（验证中文链路）
    python scripts/golden_eval.py --set all        # 中英一起跑
    python scripts/golden_eval.py --limit 5        # 只跑前 5 条（快速冒烟）
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
CASES_EN = [
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

# 中文回归集：与英文集同构（同四类、同边界、同优先级分布），
# 用来验证「中文工单 → 英文知识库」这条链路的判定质量不塌方。
CASES_ZH = [
    # --- Incident：一次性故障，报障求恢复 ---
    {"id": 101, "type": "Incident", "priority": "high", "text":
     "主题：门户登录页报 500 错误，全员无法登录\n正文：今天早上 8 点开始，公司所有人都登录不了门户，登录页显示 500 内部服务器错误。现在全公司的工作都卡住了。"},
    {"id": 102, "type": "Incident", "priority": "medium", "text":
     "主题：Outlook 一打开附件就崩溃\n正文：每次打开 .xlsx 附件 Outlook 就意外关闭，重启也没用。好像只有我这个账号有问题。"},
    {"id": 103, "type": "Incident", "priority": "high", "text":
     "主题：VPN 认证失败\n正文：我连不上公司 VPN，提示认证失败，但密码是对的，邮箱能正常登录。我在家办公，现在访问不了内网系统。"},
    {"id": 104, "type": "Incident", "priority": "medium", "text":
     "主题：Windows 更新后检测不到打印机\n正文：上周更新 Windows 之后，我的笔记本就找不到办公室的打印机了，前一天还是正常的。麻烦帮我恢复打印功能。"},
    {"id": 105, "type": "Incident", "priority": "medium", "text":
     "主题：带附件的邮件发不到外部收件人\n正文：今天早上开始，带附件发给外部地址的邮件都会被退信，发给公司内部同事是正常的。请帮忙修复。"},
    # --- Request：索要资料/权限/流程咨询 ---
    {"id": 106, "type": "Request", "priority": "low", "text":
     "主题：怎么把历史订单导出成 Excel？\n正文：你好，我想把过去一年的采购记录下载下来给财务对账用，请问可以导出成表格吗？谢谢！"},
    {"id": 107, "type": "Request", "priority": "medium", "text":
     "主题：申请开通 HR 共享文件夹权限\n正文：我这周刚加入 HR 团队，需要 HR 共享盘的只读权限才能开展工作，我的主管已经批准了。"},
    {"id": 108, "type": "Request", "priority": "low", "text":
     "主题：需要去年的发票\n正文：能不能把 2025 年所有订单的发票发给我？我们财务部门做年度审计要用。"},
    {"id": 109, "type": "Request", "priority": "low", "text":
     "主题：申请更换笔记本电脑的流程是什么？\n正文：我现在的笔记本已经用了四年，想了解申请更换设备的审批流程和大概需要多久。"},
    # --- Problem：反复出现 / 找根因 ---
    {"id": 110, "type": "Problem", "priority": "high", "text":
     "主题：共享网盘又中断了——本周第三次\n正文：这是本周一以来共享网盘第三次中断，整个销售团队现在都访问不了文件。之前的工单重启后就关闭了，但问题反复出现。我们需要彻底修复，而不是再重启一次。"},
    {"id": 111, "type": "Problem", "priority": "medium", "text":
     "主题：每周一早上都会同步报错\n正文：过去三个月里，ERP 同步每周一早上都会以同样的超时错误失败，每次都是我们手动重启。还没有人查过根本原因。"},
    {"id": 112, "type": "Problem", "priority": "medium", "text":
     "主题：发票 PDF 损坏的问题反复发生\n正文：过去几周多个用户反复反馈发票 PDF 文件损坏，重新生成后能用一阵子，然后又坏了。我们怀疑是报表服务里有更深层的问题。"},
    # --- Change：要动系统 / 改配置 ---
    {"id": 113, "type": "Change", "priority": "medium", "text":
     "主题：请把 PostgreSQL 从 13 升级到 16\n正文：我们想在下周的维护窗口安排一次数据库版本升级，请评估影响并准备回滚方案。"},
    {"id": 114, "type": "Change", "priority": "medium", "text":
     "主题：把本地文件服务器迁移到 SharePoint\n正文：我们计划把部门约 2TB 的文件从本地文件服务器迁移到 SharePoint Online，请确认需要的步骤和时间安排。"},
    {"id": 115, "type": "Change", "priority": "high", "text":
     "主题：把新的支付服务部署到生产环境\n正文：新的支付微服务已经可以上生产了，申请部署到生产集群，并调整新接口相关的配置。"},
    {"id": 116, "type": "Change", "priority": "medium", "text":
     "主题：改 SSO 配置，让所有管理员强制启用 MFA\n正文：安全部门要求我们更新 SSO 配置，让所有管理员账号必须使用多因素认证，请实施这项配置变更。"},
]

SETS = {"en": CASES_EN, "zh": CASES_ZH}
DEFAULT_OUT = {"en": "eval/golden_report.json", "zh": "eval/golden_report_zh.json",
               "all": "eval/golden_report_all.json"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="en", choices=["en", "zh", "all"], help="用例集：英文 / 中文 / 全部")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--threshold", type=float, default=0.85, help="type 准确率阈值")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="", help="报告路径（默认按用例集命名）")
    args = ap.parse_args()

    if args.set == "all":
        cases = CASES_EN + CASES_ZH
    else:
        cases = SETS[args.set]
    if args.limit:
        cases = cases[: args.limit]

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

    print(f"\n用例集：{args.set}（{n} 条）")
    print(f"{'id':>4} {'期望':<9} {'预测':<9} {'优先级':<8} {'置信':<6} 结果")
    for r in results:
        flag = "OK " if r["ok"] else "MISS"
        pred = r["pred_type"] or f"ERR({r['error'][:20]})"
        print(f"{r['id']:>4} {r['type']:<9} {pred:<9} {str(r['pred_priority']):<8} "
              f"{r['confidence']:<6} {flag}")

    print(f"\ntype 准确率 {correct}/{n} = {acc:.1%}  (阈值 {args.threshold:.0%})")
    print(f"priority 命中 {pri_hit:.1%}（参考值，不作为门禁）")

    out = Path(__file__).resolve().parent.parent / (args.out or DEFAULT_OUT[args.set])
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {"set": args.set, "n": n, "correct": correct, "type_accuracy": round(acc, 4),
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
