"""pytest 共享配置：把项目根目录加入 sys.path，便于 `pytest` 直接运行。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
