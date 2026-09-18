"""下载 Tobi-Bueck/customer-support-tickets 数据集并落盘为 parquet。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 国内镜像，避免直连超时

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from app.config import DATASET_NAME, RAW_PARQUET


def main() -> None:
    print(f"[1/2] 下载数据集: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.to_pandas()
    df.to_parquet(RAW_PARQUET, index=False)
    print(f"[2/2] 已保存: {RAW_PARQUET} | 共 {len(df)} 行, {df.shape[1]} 列")

    print("\n=== 字段 ===")
    for c in df.columns:
        print(f"  - {c} ({df[c].dtype})")

    for col in ("type", "priority", "language"):
        if col in df.columns:
            print(f"\n=== {col} 分布 ===")
            print(df[col].value_counts().to_string())

    if "queue" in df.columns:
        print("\n=== queue Top 15 ===")
        print(df["queue"].value_counts().head(15).to_string())


if __name__ == "__main__":
    main()
