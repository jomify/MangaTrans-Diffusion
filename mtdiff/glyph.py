"""Glyph condition images: Chinese text typeset in a standard font at given line boxes.

Shared by dataset building (boxes from OCR) and inference (boxes supplied by the user).
Lines are dicts {"box": [[x, y] * 4 polygon], "text": str}.
"""
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_ENV = "MTDIFF_FONT"  # "path[:face_index]" overrides the search below
FONT_CANDIDATES = (  # (path, face index of Simplified Chinese)
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 2),
    ("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf", 0),
    ("/System/Library/Fonts/PingFang.ttc", 0),
    ("C:/Windows/Fonts/msyh.ttc", 0),
    ("C:/Windows/Fonts/simhei.ttf", 0),
)
VERTICAL_RATIO = 1.3  # boxes taller than this multiple of their width are set vertically
GLYPH_FILL = 0.92
MIN_FONT_SIZE = 6

FontForSize = Callable[[int], ImageFont.FreeTypeFont]


def find_font() -> tuple[str, int]:
    """Locate a Simplified Chinese font; set MTDIFF_FONT=path[:index] to choose one explicitly."""
    override = os.environ.get(FONT_ENV)
    if override:
        path, separator, index = override.rpartition(":")
        if separator and index.isdecimal():
            return path, int(index)
        return override, 0
    for path, index in FONT_CANDIDATES:
        if Path(path).exists():
            return path, index
    raise FileNotFoundError(f"no Chinese font found; install Noto Sans CJK or set {FONT_ENV}=path[:index]")


def font_loader(path: str | None = None, index: int = 0) -> FontForSize:
    """Cached font factory; defaults to the font found by `find_font`."""
    if path is None:
        path, index = find_font()
    cache: dict[int, ImageFont.FreeTypeFont] = {}

    def at_size(size: int) -> ImageFont.FreeTypeFont:
        if size not in cache:
            cache[size] = ImageFont.truetype(path, size, index=index)
        return cache[size]

    return at_size


def line_rect(line: dict, height: int, width: int) -> tuple[int, int, int, int]:
    """Axis-aligned integer rectangle (x0, y0, x1, y1) of a line, clipped to the page."""
    box = np.array(line["box"])
    x0, y0 = np.floor(box.min(axis=0)).astype(int)
    x1, y1 = np.ceil(box.max(axis=0)).astype(int)
    return max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)


def layout_line(rect: tuple[int, int, int, int], text: str) -> tuple[int, list[tuple[tuple[float, float], str]]]:
    """Font size and per-character centres for a line box; tall boxes are set vertically, one character per cell."""
    x0, y0, x1, y1 = rect
    box_w, box_h = x1 - x0, y1 - y0
    if not text or box_w <= 0 or box_h <= 0:
        return 0, []
    vertical = box_h > VERTICAL_RATIO * box_w
    pitch = (box_h if vertical else box_w) / len(text)
    size = max(MIN_FONT_SIZE, int(min(box_w if vertical else box_h, pitch) * GLYPH_FILL))
    cells = [
        (((x0 + x1) / 2, y0 + (index + 0.5) * pitch) if vertical else (x0 + (index + 0.5) * pitch, (y0 + y1) / 2), char)
        for index, char in enumerate(text)
    ]
    return size, cells


def render_glyphs(lines: list[dict], height: int, width: int, font_for_size: FontForSize | None = None) -> np.ndarray:
    """White-on-black uint8 rendering of every line."""
    font_for_size = font_for_size or font_loader()
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for line in lines:
        size, cells = layout_line(line_rect(line, height, width), line["text"])
        for centre, char in cells:
            draw.text(centre, char, fill=255, font=font_for_size(size), anchor="mm")
    return np.asarray(canvas)
