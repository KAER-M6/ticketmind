"""数据预处理：提取 en 子集 → 清洗 → 分层划分训练/评估集。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROCESSED_DIR, RAW_PARQUET


def main() -> None:
    df = pd.read_parquet(RAW_PARQUET)
    print(f"原始: {len(df)} 行")

    # 1. en 子集（评估/演示统一语言）
    df = df[df["language"] == "en"].copy()
    print(f"en 子集: {len(df)} 行")

    # 2. 清洗
    df["subject"] = df["subject"].fillna("").astype(str).str.strip()
    df["body"] = df["body"].fillna("").astype(str).str.strip()
    df["answer"] = df["answer"].fillna("").astype(str).str.strip()
    df = df[(df["subject"] != "") | (df["body"] != "")].copy()
    print(f"去空后: {len(df)} 行")

    # 3. 合并文本（subject + body）
    df["text"] = df["subject"] + "\n" + df["body"]
    df["text"] = df["text"].str[:2500]

    # 4. 按 type+priority 组合分层划分
    df["_strat"] = df["type"] + "|" + df["priority"]
    from sklearn.model_selection import train_test_split

    train_df, eval_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["_strat"]
    )
    train_df = train_df.drop(columns=["_strat"])
    eval_df = eval_df.drop(columns=["_strat"])
    print(f"训练集: {len(train_df)} | 评估集: {len(eval_df)}")

    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    eval_df.to_parquet(PROCESSED_DIR / "eval.parquet", index=False)

    # 5. 输出分布确认
    for name, sub in (("训练", train_df), ("评估", eval_df)):
        print(f"\n=== {name}集 type 分布 ===")
        print(sub["type"].value_counts().to_string())
        print(f"=== {name}集 priority 分布 ===")
        print(sub["priority"].value_counts().to_string())


if __name__ == "__main__":
    main()
