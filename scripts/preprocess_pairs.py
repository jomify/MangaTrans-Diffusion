"""Register JA pages onto CH pages, tone-match, rebuild masks, and write a split manifest.

Usage: python preprocess_pairs.py --root /workspace/data/COMIC --out /workspace/data/proc
Outputs: {out}/src/*.png (aligned JA), {out}/tgt/*.png (CH), {out}/mask/*.png, {out}/manifest.jsonl
"""
import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from make_split import assign_splits
from pair_utils import list_pair_names, load_pair, match_size, psnr, text_change_mask, to_gray

FEATURE_SIDE = 1000
ORB_FEATURES = 4000
RANSAC_THRESHOLD = 3.0
MIN_INLIERS = 40
SKIP_WARP_BELOW_PX = 0.5
MAX_CORNER_SHIFT_FRAC = 0.08
MAX_MASK_RATIO = 0.25
GRAY_TOLERANCE = 12
TONE_PERCENTILES = (2, 98)
RESAMPLED_BLUR = 7
MIN_WORK_PAGES = 6


def page_kind(rgb: np.ndarray) -> str:
    height, width = rgb.shape[:2]
    if width > height:
        return "landscape"
    sample = rgb[::8, ::8].astype(np.int16)
    spread = sample.max(axis=2) - sample.min(axis=2)
    return "page_gray" if float(np.percentile(spread, 99)) < GRAY_TOLERANCE else "page_color"


def estimate_affine(src_gray: np.ndarray, tgt_gray: np.ndarray) -> tuple[np.ndarray | None, int]:
    """Affine transform mapping src onto tgt (both already the same size), plus inlier count."""
    scale = FEATURE_SIDE / max(tgt_gray.shape)
    small_src = cv2.resize(src_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small_tgt = cv2.resize(tgt_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    orb = cv2.ORB_create(ORB_FEATURES)
    kp_src, des_src = orb.detectAndCompute(small_src, None)
    kp_tgt, des_tgt = orb.detectAndCompute(small_tgt, None)
    if des_src is None or des_tgt is None:
        return None, 0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des_src, des_tgt)
    if len(matches) < MIN_INLIERS:
        return None, len(matches)
    pts_src = np.float32([kp_src[m.queryIdx].pt for m in matches]) / scale
    pts_tgt = np.float32([kp_tgt[m.trainIdx].pt for m in matches]) / scale
    # Full affine: the initial resize may stretch pages whose aspect ratios differ.
    matrix, inliers = cv2.estimateAffine2D(
        pts_src, pts_tgt, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_THRESHOLD
    )
    count = 0 if inliers is None else int(inliers.sum())
    return (matrix if count >= MIN_INLIERS else None), count


def corner_shift(matrix: np.ndarray, height: int, width: int) -> float:
    corners = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    moved = corners @ matrix[:, :2].T + matrix[:, 2]
    return float(np.linalg.norm(moved - corners, axis=1).max())


def match_tone(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Per-channel levels adjustment (black/white point) of src towards tgt; returns a new array."""
    channels = []
    for c in range(3):
        src_lo, src_hi = np.percentile(src[::4, ::4, c], TONE_PERCENTILES)
        tgt_lo, tgt_hi = np.percentile(tgt[::4, ::4, c], TONE_PERCENTILES)
        gain = (tgt_hi - tgt_lo) / max(src_hi - src_lo, 1.0)
        channels.append(((src[..., c].astype(np.float32) - src_lo) * gain + tgt_lo).clip(0, 255).astype(np.uint8))
    return np.stack(channels, axis=2)


def align_source(src: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, int, float]:
    height, width = tgt.shape[:2]
    resized = match_size(src, tgt)
    matrix, inliers = estimate_affine(to_gray(resized), to_gray(tgt))
    if matrix is None:
        return resized, inliers, -1.0
    shift = corner_shift(matrix, height, width)
    if shift < SKIP_WARP_BELOW_PX or shift > MAX_CORNER_SHIFT_FRAC * max(height, width):
        return resized, inliers, shift
    warped = cv2.warpAffine(resized, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, inliers, shift


def process_one(root: Path, out: Path, name: str) -> dict:
    stem = Path(name).stem
    try:
        src, tgt = load_pair(root, name)
        aligned, inliers, shift = align_source(src, tgt)
        aligned = match_tone(aligned, tgt)
        src_gray, tgt_gray = to_gray(aligned), to_gray(tgt)
        blur = RESAMPLED_BLUR if src.shape != tgt.shape or shift >= SKIP_WARP_BELOW_PX else 3
        mask = text_change_mask(src_gray, tgt_gray, blur)
        for sub, image in (("src", aligned), ("tgt", tgt)):
            cv2.imwrite(str(out / sub / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / "mask" / f"{stem}.png"), mask)
    except Exception as exc:  # keep the batch going; the manifest records the failure
        return {"name": stem, "ok": False, "error": repr(exc)}
    mask_ratio = float((mask > 0).mean())
    regions = int(cv2.connectedComponents((mask > 0).astype(np.uint8))[0] - 1)
    return {
        "name": stem,
        "kind": page_kind(tgt),
        "w": int(tgt.shape[1]), "h": int(tgt.shape[0]),
        "src_w": int(src.shape[1]), "src_h": int(src.shape[0]),
        "inliers": inliers,
        "corner_shift": round(shift, 2),
        "mask_ratio": round(mask_ratio, 5),
        "regions": regions,
        "bg_psnr": round(psnr(src_gray, tgt_gray, mask == 0), 2),
        "ok": bool(regions >= 1 and mask_ratio <= MAX_MASK_RATIO),
        "error": "",
    }


def assign_works(rows: list[dict]) -> list[dict]:
    """Group consecutive pages with the same layout into works; short runs join the previous work."""
    runs: list[list[dict]] = []
    previous_key = None
    for row in rows:
        key = (row.get("kind") == "landscape", row.get("w"), round(row.get("h", 0) / 40))
        if key != previous_key or not runs:
            runs.append([])
            previous_key = key
        runs[-1].append(row)
    works: list[list[dict]] = []
    for run in runs:
        if works and len(run) < MIN_WORK_PAGES:
            works[-1] = works[-1] + run
        else:
            works.append(run)
    return [{**row, "work": index} for index, work in enumerate(works) for row in work]


def report(rows: list[dict]) -> dict:
    summary: dict = {"works": len({r["work"] for r in rows})}
    for kind in ("page_gray", "page_color", "landscape"):
        group = [r for r in rows if r.get("kind") == kind]
        good = [r for r in group if r["ok"]]
        summary[kind] = {
            "pages": len(group),
            "ok": len(good),
            "regions_ok": sum(r["regions"] for r in good),
            "bg_psnr_median_ok": round(float(np.nanmedian([r["bg_psnr"] for r in good])), 2) if good else None,
            "by_split_ok": {s: sum(r["split"] == s for r in good) for s in ("train", "val", "test")},
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    names = list_pair_names(args.root)
    if args.limit:
        names = names[:: max(1, len(names) // args.limit)][: args.limit]
    if not names:
        raise SystemExit(f"no pairs found under {args.root}")
    for sub in ("src", "tgt", "mask"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        rows = pool.map(partial(process_one, args.root, args.out), names, chunksize=8)

    rows = assign_splits(assign_works(rows))
    with open(args.out / "manifest.jsonl", "w") as handle:
        handle.writelines(json.dumps(row) + "\n" for row in rows)
    summary = report(rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
