# 工单智脑 TicketMind

客服工单智能处理系统：工单进来自动**分类定级 → 检索相似历史案例 → 生成可发送回复草稿**，人工只做审核。

## 核心指标

| 指标 | 数值 | 说明 |
|---|---|---|
| 分诊 type 准确率 | **90.8%** | n=500，macro-F1 0.909（基线 74.2%） |
| 优先级判定 | **78.2%** | 基线 38.6% |
| 混合检索 hit@5 | **99.5%** | BM25 + MiniLM 向量，RRF 融合 |
| 反馈闭环净收益 | **+4.0pp** | 同批样本 A/B：87.0% → 91.0%（池内 60 条人工确认样本） |
| Golden 回归集 | 20/20 (100%) | 固定 20 条，改 prompt 后必跑 |
| 单测 | 38 passed / ~3s | 不联网 |

> 反馈闭环实验：`python scripts/eval_feedback_gain.py --n 100 --pool 60`，两组除反馈池外完全一致，差值为净收益。

## 技术要点

- **自建 Agent 流水线**（FastAPI + SQLite + 状态机），不套 LangChain，行为可预期、调试直接
- **动态 few-shot（核心亮点）**：每张工单先用混合检索取 5 条最相似历史工单，把其真实标签构造为 few-shot 示例——LLM 单次调用，准确率 74.2% → 90.8%，**token 成本零增加**
- **人工反馈闭环**：审核台通过/纠正标签 → 落 `data/feedback.jsonl` → 分诊时优先使用人工确认样本（占 few-shot 前 2 位）→ KPI 显示线上准确率。数据越用越准，且不重建向量索引
- **降级容错**：检索器不可用回退静态 few-shot；LLM 调用 60s 超时 + 2 次指数退避重试；输出无法解析时返回保守判定并标记 `needs_human`
- **并发批量处理**：`/api/import` AI 模式线程池并发（默认 6），配合检索缓存（同工单第二次检索 0ms）

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env      # 填 DEEPSEEK_API_KEY=sk-xxxx
start.bat                 # 或 python -m uvicorn app.main:app --port 8000
```

浏览器打开 <http://127.0.0.1:8000>。

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/triage` | 仅分诊（type + priority + reason + confidence + decision_source） |
| POST | `/api/process` | 端到端：分诊 → 检索 → 草稿 → 入库 |
| POST | `/api/import` | 批量导入，`process=true` 走 AI 全链路并并行 |
| GET | `/api/tickets` | 队列查询（可按 status 过滤） |
| POST | `/api/tickets/{id}/approve` | 通过审核，body 可带 `{type, priority}` 纠正标签（自动回流语料池） |
| POST | `/api/tickets/{id}/reject\|resolve` | 驳回 / 完成 |
| GET | `/api/stats` | KPI：总量 / 类型分布 / 自动化覆盖率 / 线上准确率 |
| GET | `/api/feedback/stats` | 反馈池规模 + 人工复核口径统计 |
| GET | `/api/health` | 健康 + 检索缓存命中率 + 反馈池规模 |

## 质量门禁

```bash
pip install -r requirements-dev.txt
pytest -q                                # 38 项单测，约 3s，不联网
ruff check .                             # 全绿（I/F/E/UP/B 规则集）
python scripts/golden_eval.py            # 20 条回归集，type 准确率 <85% 退出码 1
python scripts/golden_eval.py --limit 5  # 快速冒烟
python scripts/eval_feedback_gain.py     # 反馈闭环 A/B 增益
```

`.github/workflows/ci.yml`：单测 + lint 必过；golden 在配置了 `DEEPSEEK_API_KEY` 且数据可用时自动启用。

## Docker 部署

```bash
cp .env.example .env      # 填 DEEPSEEK_API_KEY
docker compose up -d      # 起服务；data/ 与模型缓存已挂载持久化
```

## 目录

```
app/                 FastAPI 服务
  agents/            triage（分诊）· retriever（混合检索）· reply（回复生成）· feedback（人工反馈池）
  web/static/        审核台前端（单页，支持审核时纠正标签）
  db.py              SQLite 状态机 + 标签纠正记录（open → drafted → approved/rejected → resolved）
scripts/             数据预处理、索引构建、评估（triage/rag/sc/dfs/reply/golden）
tests/               pytest 单测
data/                processed（parquet）· indexes（向量）· tickets.db · feedback.jsonl
```

## 已知限制

- priority 标注源数据仅 3 级粒度，UI 标为"建议值"，不作绝对真值
- SQLite 不支持高并发写；多用户场景需换 PostgreSQL（`db.py` 已封装，换连接层即可）
- 回复质量指标（token-F1 0.445 / ROUGE-L 0.297）为**转述口径**，与"检索直出"不可直接比较

详见 `部署说明.md`。
