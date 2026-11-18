"""Synthetic text variants of real training pages, to teach the glyph -> rendered-character mapping.

For every real training page and every variant:
  * each text line that sits on a plain light background is erased (filled with the local paper colour);
  * a new string of the same length is typeset in the same box with a random font;
  * the glyph condition is re-rendered with the new strings in the standard font.
The source (Japanese) page is untouched, so layout, bubbles and artwork stay real; only the
characters and the lettering font change.

Usage: python build_synth.py --proc /workspace/data/proc --ocr /workspace/data/ocr --variants 4
Writes {proc}/syn/{tgt2,mask2,glyph}/NAME_sK.png and {proc}/syn/manifest_syn.jsonl
"""
import argparse
import json
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from build_v2 import layout_line, line_rect, load_lines, render_glyphs, touches_region

NOTO = "/usr/share/fonts/opentype/noto"
# (path, face index, sampling weight); bold sans faces dominate, as in real scanlation lettering
FONTS = (
    (f"{NOTO}/NotoSansCJK-Black.ttc", 2, 2.0),
    (f"{NOTO}/NotoSansCJK-Bold.ttc", 2, 3.0),
    (f"{NOTO}/NotoSansCJK-Medium.ttc", 2, 2.0),
    (f"{NOTO}/NotoSansCJK-Regular.ttc", 2, 1.0),
    (f"{NOTO}/NotoSansCJK-DemiLight.ttc", 2, 0.5),
    (f"{NOTO}/NotoSerifCJK-Bold.ttc", 2, 1.0),
    (f"{NOTO}/NotoSerifCJK-SemiBold.ttc", 2, 0.5),
    (f"{NOTO}/NotoSerifCJK-Black.ttc", 2, 0.5),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0, 1.0),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0, 1.0),
    ("/usr/share/fonts/truetype/arphic/ukai.ttc", 0, 0.5),
    ("/usr/share/fonts/truetype/arphic/uming.ttc", 0, 0.5),
)
RING = 4  # width of the band around a line box used to judge the background
RING_GAP = 1
MIN_PAPER_LEVEL = 190  # the band's median grey must be at least this light
PAPER_TOLERANCE = 30
MIN_PAPER_SHARE = 0.65  # share of band pixels close to the median
INK_LEVEL = 90
MAX_MIDTONE_SHARE = 0.3  # share of in-box pixels that are neither ink nor paper
FILL_GROW = 2
CORPUS_PROB = 0.5
END_MARKS = "。！？…，～"
END_MARK_PROB = 0.4
COMMA_PROB = 0.08
MAX_INK = 50
MAX_BLUR_SIGMA = 0.8
SIZE_RANGE = (0.9, 1.08)
MIN_FONT_SIZE = 6
_FONT_CACHE: dict[tuple[str, int, int], ImageFont.FreeTypeFont] = {}


def common_characters() -> str:
    """The 3755 level-1 GB2312 characters (the most frequent simplified Chinese characters)."""
    chars = []
    for high in range(0xB0, 0xD8):
        for low in range(0xA1, 0xFF):
            try:
                chars.append(bytes([high, low]).decode("gb2312"))
            except UnicodeDecodeError:
                continue
    return "".join(chars)


def make_text(length: int, rng: random.Random, corpus: list[str], charset: str) -> str:
    """A replacement string of exactly `length` characters: real dialogue or random common characters."""
    if length <= 0:
        return ""
    if corpus and rng.random() < CORPUS_PROB:
        text = ""
        while len(text) < length:
            text += rng.choice(corpus)
        start = rng.randrange(len(text) - length + 1)
        return text[start : start + length]
    chars = [rng.choice("，、") if rng.random() < COMMA_PROB else rng.choice(charset) for _ in range(length)]
    if length > 1 and rng.random() < END_MARK_PROB:
        chars[-1] = rng.choice(END_MARKS)
    return "".join(chars)


def paper_colour(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray | None:
    """Median colour of the band around `rect` if that band is plain and light, else None."""
    height, width = image.shape[:2]
    x0, y0, x1, y1 = rect
    outer = (max(x0 - RING - RING_GAP, 0), max(y0 - RING - RING_GAP, 0),
             min(x1 + RING + RING_GAP, width), min(y1 + RING + RING_GAP, height))
    band = np.ones((outer[3] - outer[1], outer[2] - outer[0]), bool)
    inner_y0, inner_x0 = max(y0 - RING_GAP, 0) - outer[1], max(x0 - RING_GAP, 0) - outer[0]
    inner_y1, inner_x1 = min(y1 + RING_GAP, height) - outer[1], min(x1 + RING_GAP, width) - outer[0]
    band[inner_y0:inner_y1, inner_x0:inner_x1] = False
    pixels = image[outer[1] : outer[3], outer[0] : outer[2]][band]
    if len(pixels) == 0:
        return None
    grey = pixels.mean(axis=1)
    median = float(np.median(grey))
    share = float((np.abs(grey - median) <= PAPER_TOLERANCE).mean())
    if median < MIN_PAPER_LEVEL or share < MIN_PAPER_SHARE:
        return None
    # Inside the box only ink and paper are expected; many mid-tones mean the text sits on artwork.
    inside = image[y0:y1, x0:x1].mean(axis=2)
    midtones = float(((inside > INK_LEVEL) & (inside < median - PAPER_TOLERANCE)).mean()) if inside.size else 1.0
    if midtones > MAX_MIDTONE_SHARE:
        return None
    return np.median(pixels, axis=0)


def grow(rect: tuple[int, int, int, int], amount: int, height: int, width: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    return max(x0 - amount, 0), max(y0 - amount, 0), min(x1 + amount, width), min(y1 + amount, height)


def font_at(path: str, index: int, size: int) -> ImageFont.FreeTypeFont:
    key = (path, index, size)
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, size, index=index)
    return _FONT_CACHE[key]


def typeset(page: np.ndarray, lines: list[dict], rng: random.Random) -> np.ndarray:
    """Draw `lines` on a copy of `page` with one random font, ink level and softness."""
    height, width = page.shape[:2]
    path, index, _ = rng.choices(FONTS, weights=[f[2] for f in FONTS])[0]
    scale = rng.uniform(*SIZE_RANGE)
    layer = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(layer)
    for line in lines:
        size, cells = layout_line(line_rect(line, height, width), line["text"])
        font = font_at(path, index, max(MIN_FONT_SIZE, int(size * scale)))
        for centre, char in cells:
            draw.text(centre, char, fill=255, font=font, anchor="mm")
    alpha = np.asarray(layer, np.float32) / 255.0
    sigma = rng.uniform(0.0, MAX_BLUR_SIGMA)
    if sigma > 0.2:
        alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    ink = float(rng.randint(0, MAX_INK))
    blended = page.astype(np.float32) * (1 - alpha[..., None]) + ink * alpha[..., None]
    return blended.round().astype(np.uint8)


def build_variant(page: np.ndarray, region: np.ndarray, lines: list[dict], rng: random.Random,
                  corpus: list[str], charset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Returns (target, mask, glyph, replaced line count), or None when no line can be replaced."""
    height, width = region.shape
    erased = page.copy()
    mask = region.copy()
    replaced, kept = [], []
    for line in lines:
        rect = line_rect(line, height, width)
        colour = paper_colour(page, rect)
        if colour is None:
            kept.append(line)
            continue
        x0, y0, x1, y1 = grow(rect, FILL_GROW, height, width)
        erased[y0:y1, x0:x1] = colour
        mask[y0:y1, x0:x1] = True
        replaced.append({**line, "text": make_text(len(line["text"]), rng, corpus, charset)})
    if not replaced:
        return None
    target = typeset(erased, replaced, rng)
    glyph = render_glyphs(replaced + kept, height, width)
    return target, mask, glyph, len(replaced)


def build_page(proc: Path, ocr_dir: Path, variants: int, corpus: list[str], charset: str, row: dict) -> list[dict]:
    name = row["name"]
    page = cv2.imread(str(proc / "tgt2" / f"{name}.png"), cv2.IMREAD_COLOR)
    region = cv2.imread(str(proc / "mask2" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if page is None or region is None:
        return []
    region = region > 0
    height, width = region.shape
    lines = [l for l in load_lines(ocr_dir / f"{name}.json") if touches_region(region, line_rect(l, height, width))]
    out = []
    for k in range(1, variants + 1):
        variant_name = f"{name}_s{k}"
        result = build_variant(page, region, lines, random.Random(variant_name), corpus, charset)
        if result is None:
            break
        target, mask, glyph, replaced = result
        cv2.imwrite(str(proc / "syn" / "tgt2" / f"{variant_name}.png"), target)
        cv2.imwrite(str(proc / "syn" / "mask2" / f"{variant_name}.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(proc / "syn" / "glyph" / f"{variant_name}.png"), glyph)
        out.append({
            **row, "name": variant_name, "base": name, "synthetic": True,
            "mask_ratio": round(float(mask.mean()), 5), "lines_replaced": replaced, "ok": True,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.proc / "manifest_v2.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["ok"] and r["split"] == "train" and r["domain"] == "manga"]
    corpus = [l["text"] for r in rows for l in load_lines(args.ocr / f"{r['name']}.json")]
    charset = common_characters()
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    for sub in ("tgt2", "mask2", "glyph"):
        (args.proc / "syn" / sub).mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        work = partial(build_page, args.proc, args.ocr, args.variants, corpus, charset)
        out = [variant for page in pool.map(work, rows, chunksize=4) for variant in page]

    suffix = "_sample" if args.limit else ""
    (args.proc / "syn" / f"manifest_syn{suffix}.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    )
    summary = {
        "pages_in": len(rows),
        "pages_with_variants": len({r["base"] for r in out}),
        "variants": len(out),
        "lines_replaced": sum(r["lines_replaced"] for r in out),
        "lines_total": sum(r["lines"] for r in out),
        "corpus_lines": len(corpus),
        "charset": len(charset),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
