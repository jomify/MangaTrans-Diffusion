"""Score rendered pages by reading every reference text line back with OCR.

For each reference line (the OCR line of the real Chinese page that was used as glyph condition),
the same box is cropped from the image under test and passed through text recognition only.
Runs in the tools environment.

Usage: python scripts/ocr_eval.py --images DIR --proc /workspace/data/proc --ocr /workspace/data/ocr --out result.json
"""
import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from build_v2 import line_rect, load_lines, touches_region
from text_metrics import summarize

PAD = 3
VERTICAL_RATIO = 1.5  # same rule PP-OCR uses to rotate tall crops before recognition
THREADS_PER_WORKER = 3
LOG = logging.getLogger("ocr_eval")
_ENGINE = None


def get_engine():
    global _ENGINE
    if _ENGINE is None:
        from rapidocr import RapidOCR

        _ENGINE = RapidOCR(
            params={
                "Global.log_level": "error",
                "EngineConfig.onnxruntime.intra_op_num_threads": THREADS_PER_WORKER,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
    return _ENGINE


def crop_line(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    height, width = image.shape[:2]
    x0, y0, x1, y1 = rect
    crop = image[max(y0 - PAD, 0) : min(y1 + PAD, height), max(x0 - PAD, 0) : min(x1 + PAD, width)]
    return np.rot90(crop).copy() if crop.shape[0] >= VERTICAL_RATIO * crop.shape[1] else crop


def read_line(crop: np.ndarray) -> str:
    result = get_engine()(crop, use_det=False, use_cls=True, use_rec=True)
    texts = getattr(result, "txts", None)
    return "".join(texts) if texts else ""


def score_page(job: tuple[str, str, str]) -> list[tuple[str, str, tuple[int, int, int, int], str]]:
    image_path, mask_path, ocr_path = job
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    region = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or region is None:
        raise FileNotFoundError(image_path)
    height, width = region.shape
    pairs = []
    for line in load_lines(Path(ocr_path)):
        rect = line_rect(line, height, width)
        if touches_region(region > 0, rect):
            pairs.append((line["text"], read_line(crop_line(image, rect)), rect, Path(image_path).stem))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--proc", type=Path, required=True)
    parser.add_argument("--ocr", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    jobs = [
        (str(path), str(args.proc / "mask2" / path.name), str(args.ocr / f"{path.stem}.json"))
        for path in sorted(args.images.glob("*.png"))
    ]
    with Pool(args.workers) as pool:
        per_page = pool.map(score_page, jobs, chunksize=2)
    records = [record for page in per_page for record in page]
    pairs = [(reference, hypothesis) for reference, hypothesis, _, _ in records]
    summary = {"pages": len(jobs), **summarize(pairs)}
    args.out.write_text(
        json.dumps({"summary": summary, "pairs": pairs[:400]}, ensure_ascii=False, indent=1)
    )
    args.out.with_suffix(".records.json").write_text(
        json.dumps([[r, h, list(rect), page] for r, h, rect, page in records], ensure_ascii=False)
    )
    LOG.info("%s %s", args.images, json.dumps(summary))


if __name__ == "__main__":
    main()
