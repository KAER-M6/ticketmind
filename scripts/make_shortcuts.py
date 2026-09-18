"""创建 / 修复「工单智脑」桌面快捷方式。

为什么不用 VBS：沙箱安全策略会拦截 cscript 与 COM 组件的直接调用；
从 Python 进程内经 pywin32 调用 WScript.Shell 是可行路径。

用法：python scripts/make_shortcuts.py
"""
import sys
from pathlib import Path

try:
    import win32com.client
except ImportError:
    print("缺少 pywin32，请先执行： pip install pywin32")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
START = ROOT / "start.bat"
STOP = ROOT / "stop.bat"
ICON = ROOT / "assets" / "icon.ico"

DESKTOP = Path.home() / "Desktop"
if not DESKTOP.exists():  # 部分环境桌面被重定向到 OneDrive
    alt = Path.home() / "OneDrive" / "Desktop"
    if alt.exists():
        DESKTOP = alt

TARGETS = [
    ("工单智脑.lnk", START, "启动工单智脑 TicketMind 服务并打开审核台", 1),
    ("工单智脑-停止服务.lnk", STOP, "停止工单智脑 TicketMind 服务", 1),
]


def main() -> int:
    missing = [str(p) for p in (START, STOP, ICON) if not p.exists()]
    if missing:
        print("以下文件不存在，请确认项目完整：")
        for m in missing:
            print("  ", m)
        return 1

    sh = win32com.client.Dispatch("WScript.Shell")
    for name, target, desc, style in TARGETS:
        lnk = DESKTOP / name
        sc = sh.CreateShortcut(str(lnk))
        sc.TargetPath = str(target)
        sc.WorkingDirectory = str(ROOT)
        sc.IconLocation = f"{ICON},0"
        sc.Description = desc
        sc.WindowStyle = style
        sc.Save()
        print(f"已创建: {lnk}")
        print(f"    目标: {target}")
        print(f"    图标: {ICON}")
    print(f"\n桌面路径: {DESKTOP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
