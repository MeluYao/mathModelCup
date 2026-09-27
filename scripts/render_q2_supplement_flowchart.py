"""Render the Q2 urgent-priority supplement flowchart in the Q3 figure style."""

from __future__ import annotations

from math import atan2, cos, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "GMCM2026(3)"
    / "GMCM2026"
    / "2-7_问题二补充模型求解流程.png"
)

WIDTH, HEIGHT = 2103, 3199
NAVY = "#173754"
BLUE = "#387DBB"
BLUE_FILL = "#EAF3FA"
ORANGE = "#EB7A1A"
ORANGE_FILL = "#FFF1E4"
PURPLE_FILL = "#F3EEF8"
GREEN = "#2AA574"
GREEN_FILL = "#EAF7F2"
WHITE = "#FFFFFF"

FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_BOLD), size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    size: int = 48,
    fill: str = NAVY,
    spacing: int = 14,
) -> None:
    text_font = font(size)
    bounds = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=spacing, align="center")
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = (box[0] + box[2] - text_width) / 2
    y = (box[1] + box[3] - text_height) / 2 - bounds[1]
    draw.multiline_text(
        (x, y),
        text,
        font=text_font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    outline: str = NAVY,
    fill: str = BLUE_FILL,
    size: int = 48,
) -> None:
    draw.rectangle(box, fill=fill, outline=outline, width=6)
    centered_text(draw, box, text, size=size)


def parallelogram(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    outline: str,
    fill: str,
    size: int = 46,
    slant: int = 90,
) -> None:
    x1, y1, x2, y2 = box
    points = [(x1 + slant, y1), (x2, y1), (x2 - slant, y2), (x1, y2)]
    draw.polygon(points, fill=fill)
    draw.line(points + [points[0]], fill=outline, width=6, joint="curve")
    centered_text(draw, box, text, size=size)


def diamond(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    size: int = 43,
) -> None:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    points = [(cx, y1), (x2, cy), (cx, y2), (x1, cy)]
    draw.polygon(points, fill=PURPLE_FILL)
    draw.line(points + [points[0]], fill=NAVY, width=6, joint="curve")
    centered_text(draw, (x1 + 115, y1 + 35, x2 - 115, y2 - 35), text, size=size)


def arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    color: str = NAVY,
    width: int = 7,
    head_length: int = 25,
    head_width: int = 18,
) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    (x0, y0), (x1, y1) = points[-2], points[-1]
    angle = atan2(y1 - y0, x1 - x0)
    left = (
        x1 - head_length * cos(angle) + head_width * sin(angle),
        y1 - head_length * sin(angle) - head_width * cos(angle),
    )
    right = (
        x1 - head_length * cos(angle) - head_width * sin(angle),
        y1 - head_length * sin(angle) + head_width * cos(angle),
    )
    draw.polygon([(x1, y1), left, right], fill=color)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    center_x = WIDTH // 2

    start = (755, 85, 1348, 220)
    draw.rounded_rectangle(start, radius=58, fill=WHITE, outline=NAVY, width=7)
    centered_text(draw, start, "开始", size=53)

    input_box = (315, 280, 1788, 450)
    parallelogram(
        draw,
        input_box,
        "输入物资、服务区、无人机、电池与 30 m DEM",
        outline=NAVY,
        fill="#F5F9FC",
        size=45,
    )

    gis_box = (315, 515, 1788, 680)
    rectangle(
        draw,
        gis_box,
        "构建有向 GIS 航段矩阵\n大地距离、最高地形、爬升与下降",
        size=45,
    )

    classify_box = (315, 745, 1788, 925)
    rectangle(
        draw,
        classify_box,
        "划分 31 箱急送物资与 49 箱普通物资\n为全部 80 箱设置强制截止时间",
        size=44,
    )

    candidate_box = (315, 990, 1788, 1175)
    rectangle(
        draw,
        candidate_box,
        "生成物理可行候选架次\n容量、时间、等效航程能耗、SOC 与 20% 安全余量",
        size=42,
    )

    seed_box = (315, 1240, 1788, 1410)
    rectangle(
        draw,
        seed_box,
        "构造全箱按时初始解\n并完成无人机—电池双资源解码",
        size=45,
    )

    cpsat_box = (315, 1475, 1788, 1685)
    rectangle(
        draw,
        cpsat_box,
        "受限一体化 CP-SAT\n联合选择架次、开始时刻、实体无人机与电池",
        outline=ORANGE,
        fill=ORANGE_FILL,
        size=44,
    )

    lex_box = (315, 1750, 1788, 1965)
    rectangle(
        draw,
        lex_box,
        "分阶段词典序求解\n急送指标 → 总能耗 → 架次数 → 完工时间",
        outline=ORANGE,
        fill=ORANGE_FILL,
        size=44,
    )

    decision_box = (410, 2040, 1693, 2315)
    diamond(
        draw,
        decision_box,
        "80 箱是否全部按时，且容量、能耗、SOC、\n无人机与电池互斥约束全部满足？",
        size=41,
    )

    adjust_box = (1670, 2075, 2040, 2260)
    rectangle(
        draw,
        adjust_box,
        "调整候选架次\n或初始分组",
        outline=ORANGE,
        fill=ORANGE_FILL,
        size=37,
    )

    validate_box = (315, 2390, 1788, 2570)
    rectangle(
        draw,
        validate_box,
        "独立复算与校验\n逐箱送达、物理量、资源时间线与目标值",
        size=43,
    )

    output_box = (255, 2640, 1848, 2885)
    parallelogram(
        draw,
        output_box,
        "输出急送优先可行方案\n21 架次  |  急送指标 0.5883230237  |  能耗 66.3312 kWh\n80 箱零迟到  |  最低返航 SOC 20.9493%",
        outline=GREEN,
        fill=GREEN_FILL,
        size=39,
        slant=95,
    )

    end = (755, 2970, 1348, 3105)
    draw.rounded_rectangle(end, radius=58, fill=WHITE, outline=NAVY, width=7)
    centered_text(draw, end, "结束", size=53)

    arrow(draw, [(center_x, 220), (center_x, 280)])
    arrow(draw, [(center_x, 450), (center_x, 515)])
    arrow(draw, [(center_x, 680), (center_x, 745)])
    arrow(draw, [(center_x, 925), (center_x, 990)])
    arrow(draw, [(center_x, 1175), (center_x, 1240)])
    arrow(draw, [(center_x, 1410), (center_x, 1475)])
    arrow(draw, [(center_x, 1685), (center_x, 1750)])
    arrow(draw, [(center_x, 1965), (center_x, 2040)])
    arrow(draw, [(center_x, 2315), (center_x, 2390)], color=GREEN)
    centered_text(draw, (1070, 2300, 1190, 2380), "是", size=38, fill=GREEN)

    decision_right = (1693, (2040 + 2315) // 2)
    arrow(draw, [decision_right, (1670, decision_right[1])], color=ORANGE)
    centered_text(draw, (1635, 2020, 1760, 2090), "否", size=38, fill=ORANGE)
    arrow(
        draw,
        [(2040, 2168), (2070, 2168), (2070, 1580), (1788, 1580)],
        color=ORANGE,
    )
    centered_text(
        draw,
        (1955, 1640, 2075, 2040),
        "调\n整\n后\n重\n新\n求\n解",
        size=29,
        fill=ORANGE,
        spacing=0,
    )

    arrow(draw, [(center_x, 2570), (center_x, 2640)], color=GREEN)
    arrow(draw, [(center_x, 2885), (center_x, 2970)], color=GREEN)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
