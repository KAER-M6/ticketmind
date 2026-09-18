import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEXES_DIR = DATA_DIR / "indexes"

for d in (RAW_DIR, PROCESSED_DIR, INDEXES_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# 回复草稿语言：auto=跟随工单语言（默认，保证英文评估集指标可比）；也可强制 zh / en
REPLY_LANG = os.getenv("TICKETMIND_REPLY_LANG", "auto")

DATASET_NAME = "Tobi-Bueck/customer-support-tickets"
RAW_PARQUET = RAW_DIR / "tickets.parquet"
