"""生成工单智脑 TicketMind 图标：icon.png + icon.ico（多尺寸）。

设计语言：简洁 + 科技
- 深海军蓝圆角底（渐变）→ 科技稳重
- 白色工单卡片 + 三行内容线（工单字段），第三行青色高亮 = AI 处理中
- 卡片右下闪电徽章 = 智能 Agent
- 底部弧线流水线 + 节点 = 处理流程
"""
import numpy as np
from PIL import Image, ImageDraw

S = 4  # 超采样倍数（画 1024 再缩到 256，抗锯齿）
SIZE = 256 * S


def C(x, y):
    """逻辑坐标(256系) -> 画布坐标"""
    return x * S, y * S


def rrect(draw, box, radius, fill):
    """逻辑坐标圆角矩形"""
    x0, y0, x1, y1 = C(box[0], box[1]) + C(box[2], box[3])
    draw.rounded_rectangle((x0, y0, x1, y1), radius=radius * S, fill=fill)


def hline(draw, x0, x1, y, h, fill):
    """水平圆角条（行线）"""
    xa, ya = C(x0, y - h / 2)
    xb, yb = C(x1, y + h / 2)
    draw.rounded_rectangle((xa, ya, xb, yb), radius=h / 2 * S, fill=fill)


def circle(draw, cx, cy, r, fill):
    xa, ya = C(cx - r, cy - r)
    xb, yb = C(cx + r, cy + r)
    draw.ellipse((xa, ya, xb, yb), fill=fill)


# ---------- 渐变背景 ----------
top = np.array([24, 34, 76], dtype=np.float64)    # #18224C 深海军蓝
bot = np.array([9, 13, 28], dtype=np.float64)     # #090D1C 近黑
g = np.linspace(0, 1, SIZE)[:, None, None]
arr = (top[None, None, :] * (1 - g) + bot[None, None, :] * g).repeat(SIZE, axis=1)
img = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
draw = ImageDraw.Draw(img)

# 圆角透明蒙版
mask = Image.new("L", (SIZE, SIZE), 0)
ImageDraw.Draw(mask).rounded_rectangle((0, 0, SIZE - 1, SIZE - 1), radius=56 * S, fill=255)
img.putalpha(mask)

# ---------- 背景节点点缀（右上，电路感） ----------
circle(draw, 208, 56, 5, (148, 163, 184, 140))
circle(draw, 218, 82, 3, (148, 163, 184, 100))
circle(draw, 40, 210, 4, (148, 163, 184, 110))

# ---------- 工单卡片 ----------
rrect(draw, (72, 46, 184, 158), 18, (255, 255, 255, 255))

# 工单号：青色圆点 + 短线
circle(draw, 92, 74, 5, (34, 211, 238, 255))
hline(draw, 106, 158, 74, 7, (203, 213, 225, 255))

# 标题行（灰）
hline(draw, 92, 164, 100, 9, (148, 163, 184, 255))

# 处理行（青色高亮 = AI 处理中），右端让位给闪电徽章
hline(draw, 92, 150, 124, 9, (34, 211, 238, 255))

# ---------- 闪电徽章（智能 Agent） ----------
rrect(draw, (158, 120, 198, 160), 14, (14, 165, 233, 255))  # #0EA5E9
bolt = [(173, 127), (184, 127), (179, 136), (186, 136), (172, 154), (178, 142), (170, 142)]
draw.polygon([(x * S, y * S) for x, y in bolt], fill=(255, 255, 255, 255))

# ---------- 底部处理流水线弧线 ----------
pts = []
for t in np.linspace(0, 1, 60):
    x = (1 - t) ** 2 * 64 + 2 * (1 - t) * t * 128 + t ** 2 * 192
    y = (1 - t) ** 2 * 206 + 2 * (1 - t) * t * 188 + t ** 2 * 206
    pts.append((x * S, y * S))
draw.line(pts, fill=(56, 189, 248, 255), width=3 * S, joint="curve")

# 弧线两端灰色节点 + 中心青色节点（当前处理位置）
circle(draw, 64, 206, 6, (71, 85, 105, 255))
circle(draw, 192, 206, 6, (71, 85, 105, 255))
circle(draw, 128, 197, 8, (34, 211, 238, 255))

# ---------- 输出 ----------
import os

os.makedirs("assets", exist_ok=True)
icon = img.resize((256, 256), Image.LANCZOS)
icon.save("assets/icon.png")
icon.save("assets/icon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("图标已生成: assets/icon.png (256x256) + assets/icon.ico (6 尺寸)")
