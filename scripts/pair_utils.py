"""Shared helpers for loading JA/CH page pairs and deriving text-change masks."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

SRC_DIR = "2JA"
TGT_DIR = "1CH"
DIFF_THRESHOLD = 40
MIN_COMPONENT_AREA = 150
CLOSE_KERNEL = 9
DILATE_KERNEL = 15


def list_pair_names(root: Path) -> list[str]:
    src = {p.name for p in (root / SRC_DIR).glob("*.webp")}
    tgt = {p.name for p in (root / TGT_DIR).glob("*.webp")}
    return sorted(src & tgt, key=lambda n: int(Path(n).stem))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def load_pair(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    return load_rgb(root / SRC_DIR / name), load_rgb(root / TGT_DIR / name)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def match_size(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Return src resized to tgt's size (a new array; inputs untouched)."""
    if src.shape[:2] == tgt.shape[:2]:
        return src
    height, width = tgt.shape[:2]
    return cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)


def text_change_mask(src_gray: np.ndarray, tgt_gray: np.ndarray, blur: int = 3) -> np.ndarray:
    """Binary uint8 mask (0/255) of regions that differ between source and target.

    Use a larger odd `blur` when one side was resampled, so both share a similar bandwidth.
    """
    diff = cv2.absdiff(cv2.GaussianBlur(src_gray, (blur, blur), 0), cv2.GaussianBlur(tgt_gray, (blur, blur), 0))
    raw = (diff > DIFF_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA
    cleaned = keep[labels].astype(np.uint8)
    dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_KERNEL, DILATE_KERNEL))
    return cv2.dilate(cleaned, dilate) * 255


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    err = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None:
        if not mask.any():
            return float("nan")
        err = err[mask]
    mse = float(err.mean())
    return 99.0 if mse < 1e-8 else float(10 * np.log10(255.0**2 / mse))
