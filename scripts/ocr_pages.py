"""OCR the Chinese (target) pages to obtain text lines for glyph conditioning.

Runs in the tools environment (RapidOCR / onnxruntime on CPU).
Usage: python ocr_pages.py --proc /workspace/data/proc --out /workspace/data/ocr
Writes one JSON per page: [{"box": [[x, y] * 4], "text": str, "score": float}, ...]
"""
import argparse
import json
import logging
from multiprocessing import Pool
from pathlib import Path

THREADS_PER_WORKER = 3
LOG = logging.getLogger("ocr_pages")
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


def ocr_one(job: tuple[str, str]) -> tuple[str, int, str]:
    image_path, out_path = job
    try:
        result = get_engine()(image_path)
        lines = []
        if result.txts is not None:
            for box, text, score in zip(result.boxes, result.txts, result.scores):
                lines.append({"box": [[float(x), float(y)] for x, y in box], "text": text, "score": float(score)})
        Path(out_path).write_text(json.dumps(lines, ensure_ascii=False))
        return image_path, len(lines), ""
    except Exception as exc:  # record and continue; a missing JSON is treated as "no text" downstream
        return image_path, 0, repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proc", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = [json.loads(line) for line in (args.proc / "manifest.jsonl").read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(args.proc / "tgt" / f"{r['name']}.png"), str(args.out / f"{r['name']}.json"))
        for r in rows
        if r["ok"] and not (args.out / f"{r['name']}.json").exists()
    ]
    LOG.info("pages to OCR: %d", len(jobs))

    total_lines, failures = 0, 0
    with Pool(args.workers) as pool:
        for index, (path, count, error) in enumerate(pool.imap_unordered(ocr_one, jobs, chunksize=4)):
            total_lines += count
            if error:
                failures += 1
                LOG.error("failed on %s: %s", path, error)
            if index % 250 == 0:
                LOG.info("%d / %d", index, len(jobs))
    LOG.info("done: %d pages, %d lines, %d failures", len(jobs), total_lines, failures)


if __name__ == "__main__":
    main()
