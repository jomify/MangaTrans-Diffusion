"""Build v2 training targets: composite targets, OCR-filtered masks, and glyph condition images.

For every usable pair:
  * keep only the changed regions that contain recognised Chinese text (drops watermarks, scan noise);
  * composite target = Chinese page inside those regions, Japanese (source) page everywhere else;
  * glyph image = the recognised text rendered in a standard font at the OCR line boxes.

Usage: python build_v2.py --proc /workspace/data/proc --ocr /workspace/data/ocr
Writes {proc}/tgt2, {proc}/mask2, {proc}/glyph and {proc}/manifest_v2.jsonl
"""
import argparse
import json
import re
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtdiff.glyph import layout_line, line_rect, render_glyphs as render_with  # noqa: E402,F401

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_INDEX_SC = 2  # Simplified Chinese face inside the Noto CJK collection
MIN_SCORE = 0.6
CJK = re.compile(r"[一-鿿]")
FEATHER = 9
BOX_DILATE = 5
_FONTS: dict[int, ImageFont.FreeTypeFont] = {}


def font_of(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONTS:
        _FONTS[size] = ImageFont.truetype(FONT_PATH, size, index=FONT_INDEX_SC)
    return _FONTS[size]


def load_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = json.loads(path.read_text())
    return [l for l in lines if l["score"] >= MIN_SCORE and CJK.search(l["text"])]


def boxes_mask(lines: list[dict], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    for line in lines:
        x0, y0, x1, y1 = line_rect(line, height, width)
        mask[y0:y1, x0:x1] = 1
    kernel = np.ones((BOX_DILATE, BOX_DILATE), np.uint8)
    return cv2.dilate(mask, kernel)


def select_regions(diff_mask: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Keep the connected changed regions that touch recognised text."""
    count, labels = cv2.connectedComponents((diff_mask > 0).astype(np.uint8))
    touched = np.unique(labels[text_mask > 0])
    keep = np.zeros(count, dtype=bool)
    keep[touched[touched > 0]] = True
    return keep[labels]


def touches_region(region: np.ndarray, rect: tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = rect
    return bool(region[y0:y1, x0:x1].any())


def render_glyphs(lines: list[dict], height: int, width: int) -> np.ndarray:
    """White-on-black rendering of every line in the standard font."""
    return render_with(lines, height, width, font_of)


def composite(src: np.ndarray, tgt: np.ndarray, region: np.ndarray) -> np.ndarray:
    alpha = cv2.GaussianBlur(region.astype(np.float32), (FEATHER, FEATHER), 0)[..., None]
    return (alpha * tgt.astype(np.float32) + (1 - alpha) * src.astype(np.float32)).round().astype(np.uint8)


def build_one(proc: Path, ocr_dir: Path, row: dict) -> dict:
    name = row["name"]
    try:
        src = cv2.imread(str(proc / "src" / f"{name}.png"), cv2.IMREAD_COLOR)
        tgt = cv2.imread(str(proc / "tgt" / f"{name}.png"), cv2.IMREAD_COLOR)
        diff = cv2.imread(str(proc / "mask" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if src is None or tgt is None or diff is None:
            raise FileNotFoundError(name)
        height, width = diff.shape
        lines = load_lines(ocr_dir / f"{name}.json")
        region = select_regions(diff, boxes_mask(lines, height, width))
        used = [l for l in lines if touches_region(region, line_rect(l, height, width))]
        cv2.imwrite(str(proc / "tgt2" / f"{name}.png"), composite(src, tgt, region))
        cv2.imwrite(str(proc / "mask2" / f"{name}.png"), region.astype(np.uint8) * 255)
        cv2.imwrite(str(proc / "glyph" / f"{name}.png"), render_glyphs(used, height, width))
    except Exception as exc:  # keep going; the row is marked unusable
        return {**row, "ok": False, "error": repr(exc)}
    return {
        **row,
        "mask_ratio": round(float(region.mean()), 5),
        "regions": int(cv2.connectedComponents(region.astype(np.uint8))[0] - 1),
        "lines": len(used),
        "chars": sum(len(l["text"]) for l in used),
        "ok": bool(used),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in (args.proc / "manifest.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r["ok"]]
    if args.limit:
        rows = rows[:: max(1, len(rows) // args.limit)][: args.limit]
    for sub in ("tgt2", "mask2", "glyph"):
        (args.proc / sub).mkdir(exist_ok=True)

    with Pool(args.workers) as pool:
        out = pool.map(partial(build_one, args.proc, args.ocr), rows, chunksize=8)

    suffix = "_sample" if args.limit else ""
    (args.proc / f"manifest_v2{suffix}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    good = [r for r in out if r["ok"]]
    summary = {
        "pages_in": len(out),
        "pages_ok": len(good),
        "errors": sum(bool(r.get("error")) for r in out),
        "lines": sum(r["lines"] for r in good),
        "chars": sum(r["chars"] for r in good),
        "mask_ratio_median": float(np.median([r["mask_ratio"] for r in good])) if good else None,
        "by_domain_split": {
            f"{d}/{s}": sum(r["domain"] == d and r["split"] == s for r in good)
            for d in ("manga", "landscape")
            for s in ("train", "val", "test")
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
