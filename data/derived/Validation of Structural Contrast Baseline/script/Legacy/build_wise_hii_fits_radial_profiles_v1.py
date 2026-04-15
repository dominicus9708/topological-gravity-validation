#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
FITS-based radial profile builder v1

Author: Kwon Dominicus

Purpose
-------
This script reads the final input file from:

    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/wise_hii_final_input.csv

and builds radial profile CSV files from FITS images stored in the raw layer:

    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/

Recommended placement
---------------------
    data/derived/Validation of Structural Contrast Baseline/script/

Outputs
-------
Derived radial profiles:
    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/radial_profiles/

Run logs and previews:
    data/derived/Validation of Structural Contrast Baseline/input/logs/

Expected FITS naming
--------------------
The script searches in the raw image layer for files like:
    <wise_name>.fits
    <wise_name>.fit
    <wise_name>.fts
    <wise_name>_retry_*.fits
and also safe-name equivalents.

Pixel scale
-----------
Priority:
1) image_metadata.csv override if present
2) FITS header pixel scale from CDELT / CD matrix
3) fallback = 1.0 arcsec/pixel

Practical note
--------------
This v1 still assumes the target center is the image center.
That is acceptable for the current first-pass consistency-oriented workflow.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except Exception as e:
    raise SystemExit(
        "astropy is required for FITS-based profile building. "
        "Install it with: pip install astropy\n"
        f"Import error: {e}"
    )

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

INPUT_FILE = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "wise_hii_catalog"
    / "wise_hii_final_input.csv"
)

IMAGE_METADATA_FILE = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "wise_hii_catalog"
    / "image_metadata.csv"
)

RAW_IMAGE_DIR = (
    Path("data")
    / "raw"
    / "Validation of Structural Contrast Baseline"
    / "wise_hii_catalog"
    / "images"
)

PROFILE_OUTPUT_DIR = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "wise_hii_catalog"
    / "radial_profiles"
)

LOG_OUTPUT_DIR = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "logs"
)


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def sanitize_wise_name(name: str) -> str:
    text = str(name).strip()
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_final_input(project_root: Path) -> pd.DataFrame:
    path = project_root / INPUT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Final input file not found: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["wise_name", "radius_arcsec", "target_status", "input_ready"]
    ensure_required_columns(df, required)

    df["wise_name"] = df["wise_name"].astype(str).str.strip()
    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")

    bad = df[df["radius_arcsec_num"].isna()]
    if not bad.empty:
        raise ValueError("Invalid radius_arcsec values for: " + ", ".join(bad["wise_name"].tolist()))

    not_ready = df.loc[df["input_ready"].astype(str).str.lower().ne("yes"), "wise_name"].tolist()
    if not_ready:
        raise ValueError("Rows not marked input_ready=yes: " + ", ".join(not_ready))

    not_fixed = df.loc[df["target_status"].astype(str).ne("fixed_final_input"), "wise_name"].tolist()
    if not_fixed:
        raise ValueError("Rows not fixed_final_input: " + ", ".join(not_fixed))

    return df.reset_index(drop=True)


def load_image_metadata(project_root: Path) -> pd.DataFrame:
    path = project_root / IMAGE_METADATA_FILE
    if not path.exists():
        return pd.DataFrame(columns=["wise_name", "pixel_scale_arcsec_per_pixel"])
    df = pd.read_csv(path, low_memory=False)
    if "wise_name" not in df.columns or "pixel_scale_arcsec_per_pixel" not in df.columns:
        return pd.DataFrame(columns=["wise_name", "pixel_scale_arcsec_per_pixel"])
    df["wise_name"] = df["wise_name"].astype(str).str.strip()
    df["pixel_scale_arcsec_per_pixel"] = pd.to_numeric(df["pixel_scale_arcsec_per_pixel"], errors="coerce")
    return df.dropna(subset=["pixel_scale_arcsec_per_pixel"])


def get_pixel_scale_override(wise_name: str, metadata: pd.DataFrame) -> Optional[float]:
    row = metadata.loc[metadata["wise_name"] == wise_name]
    if row.empty:
        return None
    return float(row.iloc[0]["pixel_scale_arcsec_per_pixel"])


def find_best_fits_file(project_root: Path, wise_name: str) -> Optional[Path]:
    base = project_root / RAW_IMAGE_DIR
    safe = sanitize_wise_name(wise_name)

    direct_candidates = [
        base / f"{wise_name}.fits",
        base / f"{wise_name}.fit",
        base / f"{wise_name}.fts",
        base / f"{safe}.fits",
        base / f"{safe}.fit",
        base / f"{safe}.fts",
    ]
    for p in direct_candidates:
        if p.exists():
            return p

    retry_candidates = []
    for pattern in [
        f"{wise_name}_retry_*.fits",
        f"{wise_name}_retry_*.fit",
        f"{wise_name}_retry_*.fts",
        f"{safe}_retry_*.fits",
        f"{safe}_retry_*.fit",
        f"{safe}_retry_*.fts",
    ]:
        retry_candidates.extend(base.glob(pattern))

    if not retry_candidates:
        return None

    # Prefer the largest file as a pragmatic first-pass proxy for richer content
    retry_candidates = sorted(retry_candidates, key=lambda p: p.stat().st_size, reverse=True)
    return retry_candidates[0]


def extract_2d_image_and_header(path: Path) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        header = None
        data_2d = None

        for hdu in hdul:
            if getattr(hdu, "data", None) is None:
                continue
            data = np.asarray(hdu.data)
            if data.size == 0:
                continue

            # Reduce to 2D in a safe practical way
            if data.ndim == 2:
                data_2d = data
                header = hdu.header
                break
            if data.ndim > 2:
                reduced = np.squeeze(data)
                if reduced.ndim == 2:
                    data_2d = reduced
                    header = hdu.header
                    break
                while reduced.ndim > 2:
                    reduced = reduced[0]
                if reduced.ndim == 2:
                    data_2d = reduced
                    header = hdu.header
                    break

        if data_2d is None or header is None:
            raise ValueError(f"No usable 2D image plane found in FITS file: {path}")

        arr = np.array(data_2d, dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        return arr, header


def estimate_pixel_scale_from_header(header: fits.Header) -> Optional[float]:
    # 1) direct CDELT
    for key in ("CDELT1", "CDELT2"):
        if key in header:
            try:
                deg_per_pix = abs(float(header[key]))
                if deg_per_pix > 0:
                    return deg_per_pix * 3600.0
            except Exception:
                pass

    # 2) CD matrix
    if "CD1_1" in header and "CD1_2" in header:
        try:
            cd11 = float(header.get("CD1_1", 0.0))
            cd12 = float(header.get("CD1_2", 0.0))
            deg_per_pix = math.sqrt(cd11 * cd11 + cd12 * cd12)
            if deg_per_pix > 0:
                return abs(deg_per_pix) * 3600.0
        except Exception:
            pass

    if "CD2_1" in header and "CD2_2" in header:
        try:
            cd21 = float(header.get("CD2_1", 0.0))
            cd22 = float(header.get("CD2_2", 0.0))
            deg_per_pix = math.sqrt(cd21 * cd21 + cd22 * cd22)
            if deg_per_pix > 0:
                return abs(deg_per_pix) * 3600.0
        except Exception:
            pass

    return None


def radial_profile_from_fits_image(
    image: np.ndarray,
    pixel_scale_arcsec_per_pixel: float,
    max_radius_arcsec: float,
) -> pd.DataFrame:
    h, w = image.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    y, x = np.indices(image.shape)
    r_pix = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_arcsec = r_pix * pixel_scale_arcsec_per_pixel

    max_bin = int(np.floor(max_radius_arcsec))
    rows = []
    for i in range(max_bin + 1):
        r_in = float(i)
        r_out = float(i + 1)
        mask = (r_arcsec >= r_in) & (r_arcsec < r_out)
        if not np.any(mask):
            continue

        vals = image[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue

        rows.append(
            {
                "radius_arcsec": 0.5 * (r_in + r_out),
                "intensity": float(np.nanmean(vals)),
                "median_intensity": float(np.nanmedian(vals)),
                "std_intensity": float(np.nanstd(vals)),
                "n_pixels": int(vals.size),
            }
        )

    return pd.DataFrame(rows)


def save_profile_plot(wise_name: str, profile_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(profile_df["radius_arcsec"], profile_df["intensity"], label="mean intensity")
    if "median_intensity" in profile_df.columns:
        ax.plot(profile_df["radius_arcsec"], profile_df["median_intensity"], label="median intensity")
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{wise_name} FITS radial profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    profile_out_dir = project_root / PROFILE_OUTPUT_DIR
    profile_out_dir.mkdir(parents=True, exist_ok=True)

    log_dir = project_root / LOG_OUTPUT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)
    metadata = load_image_metadata(project_root)

    summary_rows = []

    for _, row in df.iterrows():
        wise_name = str(row["wise_name"])
        safe_name = sanitize_wise_name(wise_name)
        radius_arcsec = float(row["radius_arcsec_num"])

        fits_path = find_best_fits_file(project_root, wise_name)
        profile_path = profile_out_dir / f"{safe_name}_radial_profile.csv"
        preview_path = log_dir / f"{safe_name}_fits_radial_profile_preview_{timestamp}.png"

        if fits_path is None:
            summary_rows.append(
                {
                    "wise_name": wise_name,
                    "radius_arcsec": radius_arcsec,
                    "fits_file": "",
                    "pixel_scale_arcsec_per_pixel": "",
                    "profile_file": "",
                    "status": "missing_fits_file",
                    "detail": "",
                }
            )
            continue

        status = "unknown"
        detail = ""
        pixel_scale = None

        try:
            image, header = extract_2d_image_and_header(fits_path)

            pixel_scale = get_pixel_scale_override(wise_name, metadata)
            if pixel_scale is None:
                pixel_scale = estimate_pixel_scale_from_header(header)
            if pixel_scale is None:
                pixel_scale = 1.0
                detail = "pixel_scale_fallback_1.0_arcsec_per_pixel"
            else:
                detail = "pixel_scale_from_metadata_or_header"

            profile_df = radial_profile_from_fits_image(
                image=image,
                pixel_scale_arcsec_per_pixel=float(pixel_scale),
                max_radius_arcsec=1.8 * radius_arcsec,
            )

            if profile_df.empty:
                status = "fits_loaded_but_profile_empty"
            else:
                profile_df.to_csv(profile_path, index=False, encoding="utf-8-sig")
                save_profile_plot(wise_name, profile_df, preview_path)
                status = "profile_built_from_fits"

        except Exception as e:
            status = "fits_profile_build_failed"
            detail = str(e)

        summary_rows.append(
            {
                "wise_name": wise_name,
                "radius_arcsec": radius_arcsec,
                "fits_file": str(fits_path) if fits_path else "",
                "pixel_scale_arcsec_per_pixel": pixel_scale if pixel_scale is not None else "",
                "profile_file": str(profile_path) if profile_path.exists() else "",
                "status": status,
                "detail": detail,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = log_dir / f"wise_hii_fits_radial_profile_summary_{timestamp}.csv"
    latest_summary_path = log_dir / "wise_hii_fits_radial_profile_summary_latest.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(latest_summary_path, index=False, encoding="utf-8-sig")

    manifest_lines = [
        "Validation of Structural Contrast Baseline",
        "FITS-based radial profile builder manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"input_file: {project_root / INPUT_FILE}",
        f"raw_image_dir: {project_root / RAW_IMAGE_DIR}",
        f"profile_output_dir: {profile_out_dir}",
        f"log_output_dir: {log_dir}",
        "",
        "Notes",
        "-" * 20,
        "This script builds radial profiles from FITS files in the raw image layer.",
        "Direct files and retry files are both searched.",
        "Pixel scale priority: metadata override > FITS header > fallback 1.0 arcsec/pixel.",
        "Current v1 still assumes the target center is the image center.",
    ]
    manifest_text = "\n".join(manifest_lines)

    manifest_path = log_dir / f"wise_hii_fits_radial_profile_manifest_{timestamp}.txt"
    latest_manifest_path = log_dir / "wise_hii_fits_radial_profile_manifest_latest.txt"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    latest_manifest_path.write_text(manifest_text, encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - FITS-based radial profile builder v1")
    print("=" * 72)
    print(f"Project root      : {project_root}")
    print(f"Input file        : {project_root / INPUT_FILE}")
    print(f"Raw image dir     : {project_root / RAW_IMAGE_DIR}")
    print(f"Profile output dir: {profile_out_dir}")
    print(f"Log output dir    : {log_dir}")
    print("")
    print("[OK] Created:")
    print(summary_path)
    print(latest_summary_path)
    print(manifest_path)
    print(latest_manifest_path)
    print(profile_out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
