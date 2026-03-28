"""
Offline assist maps for the labeling UI (foreground + edge barriers + superpixel boundaries).

Writes sibling PNGs next to each screenshot::

    {stem}_assist_fg.png   — uint8, 255 ≈ likely train / changed vs background
    {stem}_assist_edge.png — uint8, high = strong edge / superpixel border

Run::

    python -m vision.label_assist_preprocess --help
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError("label_assist requires OpenCV (pip install opencv-python-headless)") from e

ASSIST_FG_SUFFIX = "_assist_fg.png"
ASSIST_EDGE_SUFFIX = "_assist_edge.png"


def _resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def build_median_background(
    paths: list[Path],
    *,
    max_side: int = 960,
    max_images: int = 80,
    seed: int = 42,
) -> np.ndarray | None:
    """Median BGR image from a sample of PNGs (empty-ish scenes help)."""
    pngs = [p for p in paths if p.suffix.lower() == ".png" and "_assist_" not in p.name]
    if len(pngs) < 3:
        return None
    rng = random.Random(seed)
    sample = pngs if len(pngs) <= max_images else rng.sample(pngs, max_images)
    stack: list[np.ndarray] = []
    target_shape: tuple[int, int] | None = None
    for p in sample:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        im = _resize_max_side(im, max_side)
        if target_shape is None:
            target_shape = (im.shape[1], im.shape[0])
        elif (im.shape[1], im.shape[0]) != target_shape:
            im = cv2.resize(im, target_shape, interpolation=cv2.INTER_AREA)
        stack.append(im.astype(np.float32))
    if len(stack) < 3:
        return None
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def _sobel_mag_u8(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if mag.max() > 1e-6:
        mag = mag / mag.max()
    return np.clip(mag * 255.0, 0, 255).astype(np.uint8)


def _morph_gradient_u8(gray: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    g = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, k)
    if g.max() > 0:
        g = (g.astype(np.float32) / float(g.max()) * 255.0).astype(np.uint8)
    return g


def _superpixel_boundary_mask(bgr: np.ndarray, region_size: int, ruler: float) -> np.ndarray | None:
    """Return uint8 mask (255 on superpixel contours) or None if SLIC unavailable."""
    try:
        ximg = getattr(cv2, "ximgproc", None)
        if ximg is None:
            return None
        slic_type = getattr(ximg, "SLIC", 100)
        slic = ximg.createSuperpixelSLIC(bgr, slic_type, region_size=region_size, ruler=ruler)
        slic.iterate(10)
        m = slic.getLabelContourMask()
        return (m > 0).astype(np.uint8) * 255
    except Exception:
        return None


def compute_assist_maps(
    bgr: np.ndarray,
    background_bgr: np.ndarray | None,
    *,
    fg_absdiff_thresh: int = 18,
    fg_dilate: int = 3,
    spx_region_size: int = 22,
    spx_ruler: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (fg_u8, edge_u8) same shape as ``bgr``.

    ``fg_u8``: 255 where foreground-ish; if no background, all 255.
    ``edge_u8``: combined Sobel + morphology + optional superpixel borders.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edge = _sobel_mag_u8(gray)
    edge = np.maximum(edge, _morph_gradient_u8(gray))

    spx = _superpixel_boundary_mask(bgr, spx_region_size, spx_ruler)
    if spx is not None:
        edge = np.maximum(edge, spx)

    if background_bgr is None:
        fg = np.full(gray.shape, 255, dtype=np.uint8)
    else:
        if background_bgr.shape != bgr.shape:
            background_bgr = cv2.resize(
                background_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA
            )
        diff = cv2.absdiff(bgr, background_bgr)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, fg = cv2.threshold(diff_gray, fg_absdiff_thresh, 255, cv2.THRESH_BINARY)
        if fg_dilate > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fg_dilate, fg_dilate))
            fg = cv2.dilate(fg, k, iterations=1)
        fg = cv2.medianBlur(fg, 5)

    return fg, edge


def write_assist_for_png(
    png_path: Path,
    background_bgr: np.ndarray | None,
    *,
    fg_absdiff_thresh: int = 18,
    overwrite: bool = False,
) -> tuple[bool, str]:
    """Write ``*_assist_fg.png`` and ``*_assist_edge.png`` next to ``png_path``."""
    stem = png_path.stem
    if "_assist_" in stem:
        return False, "skip assist source"
    out_fg = png_path.parent / f"{stem}{ASSIST_FG_SUFFIX}"
    out_edge = png_path.parent / f"{stem}{ASSIST_EDGE_SUFFIX}"
    if not overwrite and out_fg.is_file() and out_edge.is_file():
        return False, "exists"
    bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return False, "read fail"
    fg, edge = compute_assist_maps(
        bgr,
        background_bgr,
        fg_absdiff_thresh=fg_absdiff_thresh,
    )
    cv2.imwrite(str(out_fg), fg)
    cv2.imwrite(str(out_edge), edge)
    return True, "ok"


def preprocess_directory(
    directory: Path,
    *,
    bg_image: Path | None,
    median_max_images: int,
    median_max_side: int,
    fg_thresh: int,
    max_files: int | None,
    overwrite: bool,
    seed: int,
) -> tuple[int, int, int]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))

    pngs = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".png" and "_assist_" not in p.name)
    if max_files is not None:
        pngs = pngs[: max(0, max_files)]

    bg_model: np.ndarray | None = None
    if bg_image is not None:
        bg_path = bg_image.expanduser().resolve()
        if not bg_path.is_file():
            raise FileNotFoundError(str(bg_path))
        bg_model = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
        if bg_model is None:
            raise ValueError(f"Could not read background image: {bg_path}")
        bg_model = _resize_max_side(bg_model, median_max_side)
    else:
        bg_model = build_median_background(
            pngs,
            max_side=median_max_side,
            max_images=median_max_images,
            seed=seed,
        )

    written = 0
    skipped = 0
    failed = 0
    for p in pngs:
        ok, reason = write_assist_for_png(
            p,
            bg_model,
            fg_absdiff_thresh=fg_thresh,
            overwrite=overwrite,
        )
        if ok:
            written += 1
        elif reason in ("exists", "skip assist source"):
            skipped += 1
        else:
            failed += 1
    return written, skipped, failed


def main() -> None:
    ap = argparse.ArgumentParser(description="Build assist FG/edge PNGs for label UI region grow.")
    ap.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Directory of screenshot PNGs (default: VISION_SCREENSHOT_DIR or vision config)",
    )
    ap.add_argument("--bg-image", type=Path, default=None, help="Reference empty-scene PNG (optional).")
    ap.add_argument(
        "--median-max",
        type=int,
        default=80,
        help="Max PNGs to use for median background when --bg-image not set (default 80).",
    )
    ap.add_argument(
        "--median-side",
        type=int,
        default=960,
        help="Longest side when building median / resizing bg image (default 960).",
    )
    ap.add_argument(
        "--fg-thresh",
        type=int,
        default=18,
        help="Absdiff threshold vs background for foreground mask (default 18).",
    )
    ap.add_argument("--max-files", type=int, default=None, help="Process at most N PNGs (debug).")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate even if assist PNGs exist.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for median sampling.")
    args = ap.parse_args()

    shot_dir = args.dir
    if shot_dir is None:
        from vision.config import load_vision_config

        shot_dir = load_vision_config().screenshot_dir

    w, sk, fl = preprocess_directory(
        shot_dir,
        bg_image=args.bg_image,
        median_max_images=args.median_max,
        median_max_side=args.median_side,
        fg_thresh=args.fg_thresh,
        max_files=args.max_files,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    bg_note = f"bg={args.bg_image}" if args.bg_image else "bg=median(sample)"
    print(f"[label_assist] dir={shot_dir} {bg_note} written={w} skipped={sk} failed={fl}")


if __name__ == "__main__":
    main()
