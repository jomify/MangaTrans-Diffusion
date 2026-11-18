"""Render Chinese text into a Japanese manga page.

Usage:
  python translate_page.py page.png regions.json out.png --weights weights/mangatrans-v3

regions.json: [{"box": [x0, y0, x1, y1], "text": "中文", "erase": [x0, y0, x1, y1]}, ...]
  box   - where the Chinese line is typeset (tall boxes are set vertically, wide boxes horizontally)
  erase - optional larger area to repaint, e.g. the whole original Japanese balloon text
Coordinates are in pixels of the input image.
"""
import argparse
import json
from pathlib import Path

import cv2

from mtdiff.infer import DEFAULT_MARGIN, DEFAULT_WIDTH, PageTranslator, parse_regions, scale_regions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("regions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", required=True, help="directory with model.safetensors and config.json")
    parser.add_argument("--vae", help="local VAE directory (default: download the one named in config.json)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="working width; 0 keeps the input size")
    parser.add_argument("--restore-size", action="store_true", help="resize the result back to the input size")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=int, default=DEFAULT_MARGIN, help="pixels around each region that may change")
    parser.add_argument("--full-page", action="store_true", help="use the model output for the whole page")
    args = parser.parse_args()

    page = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if page is None:
        raise SystemExit(f"cannot read image: {args.image}")
    original_h, original_w = page.shape[:2]
    regions = parse_regions(json.loads(args.regions.read_text(encoding="utf-8")))
    factor = args.width / original_w if args.width else 1.0
    if factor != 1.0:
        page = cv2.resize(page, (round(original_w * factor), round(original_h * factor)), interpolation=cv2.INTER_AREA)
        regions = scale_regions(regions, factor)

    translator = PageTranslator(args.weights, vae=args.vae)
    result = translator.translate(
        cv2.cvtColor(page, cv2.COLOR_BGR2RGB), regions, args.steps, args.seed, args.margin, args.full_page
    )
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    if args.restore_size and factor != 1.0:
        result = cv2.resize(result, (original_w, original_h), interpolation=cv2.INTER_CUBIC)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise SystemExit(f"cannot write image: {args.output}")


if __name__ == "__main__":
    main()
