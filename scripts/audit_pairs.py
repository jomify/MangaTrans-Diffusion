"""Audit JA/CH page pairs: size, colour, alignment, and text-change mask statistics.

Usage: python audit_pairs.py --root /workspace/data/COMIC --out /workspace/out/audit
"""
import argparse
import csv
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from pair_utils import list_pair_names, load_pair, match_size, psnr, text_change_mask, to_gray

GRAY_TOLERANCE = 12
VIS_HEIGHT = 900
ALIGN_SIZE = 512


def is_grayscale(rgb: np.ndarray) -> bool:
    sample = rgb[::8, ::8].astype(np.int16)
    spread = sample.max(axis=2) - sample.min(axis=2)
    return float(np.percentile(spread, 99)) < GRAY_TOLERANCE


def estimate_shift(src_gray: np.ndarray, tgt_gray: np.ndarray) -> tuple[float, float]:
    """Global translation (in full-res pixels) via phase correlation on downsized pages."""
    height, width = tgt_gray.shape
    small_src = cv2.resize(src_gray, (ALIGN_SIZE, ALIGN_SIZE)).astype(np.float32)
    small_tgt = cv2.resize(tgt_gray, (ALIGN_SIZE, ALIGN_SIZE)).astype(np.float32)
    (dx, dy), _ = cv2.phaseCorrelate(small_src, small_tgt)
    return dx * width / ALIGN_SIZE, dy * height / ALIGN_SIZE


def audit_one(root: Path, name: str) -> dict:
    try:
        src, tgt = load_pair(root, name)
    except Exception as exc:  # corrupt or unreadable file: record and move on
        return {"name": name, "error": repr(exc)}
    src_resized = match_size(src, tgt)
    src_gray, tgt_gray = to_gray(src_resized), to_gray(tgt)
    mask = text_change_mask(src_gray, tgt_gray)
    region_count = cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1
    dx, dy = estimate_shift(src_gray, tgt_gray)
    return {
        "name": name,
        "error": "",
        "src_w": src.shape[1], "src_h": src.shape[0],
        "tgt_w": tgt.shape[1], "tgt_h": tgt.shape[0],
        "same_size": src.shape == tgt.shape,
        "src_gray": is_grayscale(src), "tgt_gray": is_grayscale(tgt),
        "shift_x": round(dx, 2), "shift_y": round(dy, 2),
        "mask_ratio": round(float((mask > 0).mean()), 5),
        "regions": region_count,
        "bg_psnr": round(psnr(src_gray, tgt_gray, mask == 0), 2),
    }


def save_visual(root: Path, out_dir: Path, name: str) -> None:
    src, tgt = load_pair(root, name)
    src = match_size(src, tgt)
    mask = text_change_mask(to_gray(src), to_gray(tgt))
    overlay = tgt.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    panel = np.concatenate([src, tgt, overlay], axis=1)
    scale = VIS_HEIGHT / panel.shape[0]
    panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_dir / f"{Path(name).stem}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    def col(key):
        return np.array([r[key] for r in ok], dtype=np.float64)
    shift = np.hypot(col("shift_x"), col("shift_y"))
    quantiles = [5, 25, 50, 75, 95]
    return {
        "pairs": len(rows),
        "errors": len(rows) - len(ok),
        "same_size_frac": float(col("same_size").mean()),
        "src_gray_frac": float(col("src_gray").mean()),
        "tgt_gray_frac": float(col("tgt_gray").mean()),
        "tgt_w_pct": np.percentile(col("tgt_w"), quantiles).tolist(),
        "tgt_h_pct": np.percentile(col("tgt_h"), quantiles).tolist(),
        "shift_px_pct": np.percentile(shift, quantiles).tolist(),
        "shift_gt_2px_frac": float((shift > 2).mean()),
        "mask_ratio_pct": np.percentile(col("mask_ratio"), quantiles).tolist(),
        "regions_pct": np.percentile(col("regions"), quantiles).tolist(),
        "regions_total": int(col("regions").sum()),
        "no_change_frac": float((col("regions") == 0).mean()),
        "bg_psnr_pct": np.nanpercentile(col("bg_psnr"), quantiles).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--n-vis", type=int, default=40)
    args = parser.parse_args()

    names = list_pair_names(args.root)
    if not names:
        raise SystemExit(f"no pairs found under {args.root}")
    vis_dir = args.out / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        rows = pool.map(partial(audit_one, args.root), names, chunksize=16)

    all_fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "name", k))
    with open(args.out / "pairs.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    step = max(1, len(names) // args.n_vis)
    for name in names[::step][: args.n_vis]:
        save_visual(args.root, vis_dir, name)


if __name__ == "__main__":
    main()
