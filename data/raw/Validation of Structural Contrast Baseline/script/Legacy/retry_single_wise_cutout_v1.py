#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
WISE single-target retry downloader v1

Author: Kwon Dominicus

Purpose
-------
This script is for retrying a target whose WISE PNG cutout came back visually blank.

Recommended placement:
    data/raw/Validation of Structural Contrast Baseline/script/

Default target:
    G028.983-00.604

Input:
    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/wise_hii_final_input.csv

Outputs:
    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/
    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/logs/

What this script does
---------------------
1) Reads the target coordinates and radius from the final input layer
2) Retries WISE cutout downloads with multiple subset sizes
3) Downloads both PNG and FITS when possible
4) Checks whether PNG is visually blank
5) Saves a retry summary CSV and manifest into the raw log layer
"""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib.image as mpimg

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")
DEFAULT_TARGET = "G028.983-00.604"

INPUT_FILE = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "wise_hii_catalog"
    / "wise_hii_final_input.csv"
)

IMAGE_OUTPUT_DIR = (
    Path("data")
    / "raw"
    / "Validation of Structural Contrast Baseline"
    / "wise_hii_catalog"
    / "images"
)

LOG_OUTPUT_DIR = (
    Path("data")
    / "raw"
    / "Validation of Structural Contrast Baseline"
    / "wise_hii_catalog"
    / "logs"
)

FINDERCHART_API = "https://irsa.ipac.caltech.edu/applications/finderchart/servlet/api"
REQUEST_TIMEOUT_SEC = 180


def sanitize_wise_name(name: str) -> str:
    text = str(name).strip()
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def load_final_input(project_root: Path) -> pd.DataFrame:
    path = project_root / INPUT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Final input file not found: {path}")
    return pd.read_csv(path, dtype=str, low_memory=False)


def normalize_input(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = ["wise_name", "ra", "dec", "radius_arcsec", "target_status", "input_ready"]
    ensure_required_columns(df, required)

    df["wise_name"] = df["wise_name"].astype(str).str.strip()
    df["ra_num"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec_num"] = pd.to_numeric(df["dec"], errors="coerce")
    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")

    bad = df[df[["ra_num", "dec_num", "radius_arcsec_num"]].isna().any(axis=1)]
    if not bad.empty:
        raise ValueError(
            "Some rows have invalid numeric coordinates or radius: "
            + ", ".join(bad["wise_name"].astype(str).tolist())
        )

    not_ready = df.loc[df["input_ready"].astype(str).str.lower().ne("yes"), "wise_name"].tolist()
    if not_ready:
        raise ValueError("Rows not marked input_ready=yes: " + ", ".join(not_ready))

    not_fixed = df.loc[df["target_status"].astype(str).ne("fixed_final_input"), "wise_name"].tolist()
    if not_fixed:
        raise ValueError("Rows not fixed_final_input: " + ", ".join(not_fixed))

    return df.reset_index(drop=True)


def get_target_row(df: pd.DataFrame, wise_name: str) -> pd.Series:
    row = df.loc[df["wise_name"] == wise_name]
    if row.empty:
        raise ValueError(f"Target not found in final input: {wise_name}")
    return row.iloc[0]


def compute_base_subsetsize_arcmin(radius_arcsec: float) -> float:
    size_arcsec = max(2.0 * 1.8 * radius_arcsec + 30.0, 60.0)
    return float(min(max(size_arcsec / 60.0, 0.1), 60.0))


def build_subsetsize_candidates(radius_arcsec: float) -> List[float]:
    base = compute_base_subsetsize_arcmin(radius_arcsec)
    candidates = [
        base,
        min(base * 1.5, 60.0),
        min(base * 2.0, 60.0),
        min(max(10.0, base * 2.5), 60.0),
    ]
    out = []
    for x in candidates:
        x = round(float(x), 4)
        if x not in out:
            out.append(x)
    return out


def build_request_params(ra: float, dec: float, subsetsize_arcmin: float, file_type: str) -> dict:
    locstr = f"{ra:.8f} {dec:.8f}"
    return {
        "mode": "getImage",
        "survey": "WISE",
        "file_type": file_type,
        "locstr": locstr,
        "subsetsize": f"{subsetsize_arcmin:.4f}",
        "marker": "false",
        "grid": "false",
    }


def download_payload(params: dict) -> bytes:
    response = requests.get(FINDERCHART_API, params=params, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return response.content


def extract_matching_file(zip_bytes: bytes, suffixes: Tuple[str, ...]) -> Optional[Tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        matches = [n for n in names if n.lower().endswith(suffixes)]
        if not matches:
            return None
        matches.sort()
        chosen = matches[0]
        return chosen, zf.read(chosen)


def png_is_visually_blank(path: Path) -> Tuple[bool, str]:
    try:
        arr = mpimg.imread(path)
        arr = np.asarray(arr)
        if arr.size == 0:
            return True, "empty_array"
        arr = arr.astype(float)

        min_val = float(np.nanmin(arr))
        max_val = float(np.nanmax(arr))
        mean_val = float(np.nanmean(arr))
        std_val = float(np.nanstd(arr))

        # Strong blank criterion for this workflow:
        # completely black or virtually no dynamic range
        if max_val <= 0.0:
            return True, f"max={max_val:.6f}, std={std_val:.6f}"
        if std_val < 1e-6:
            return True, f"std={std_val:.6f}, mean={mean_val:.6f}, min={min_val:.6f}, max={max_val:.6f}"

        return False, f"std={std_val:.6f}, mean={mean_val:.6f}, min={min_val:.6f}, max={max_val:.6f}"
    except Exception as e:
        return True, f"png_check_failed: {e}"


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    target_name = DEFAULT_TARGET

    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) >= 3:
        target_name = str(sys.argv[2]).strip()

    image_dir = project_root / IMAGE_OUTPUT_DIR
    image_dir.mkdir(parents=True, exist_ok=True)

    log_dir = project_root / LOG_OUTPUT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)
    row = get_target_row(df, target_name)

    wise_name = str(row["wise_name"])
    ra = float(row["ra_num"])
    dec = float(row["dec_num"])
    radius_arcsec = float(row["radius_arcsec_num"])

    subset_candidates = build_subsetsize_candidates(radius_arcsec)

    summary_rows = []
    safe_name = sanitize_wise_name(wise_name)

    best_png_path = None
    best_fits_path = None

    for idx, subset_arcmin in enumerate(subset_candidates, start=1):
        # PNG retry
        png_status = "not_attempted"
        png_detail = ""
        png_out = image_dir / f"{safe_name}_retry_s{subset_arcmin:.4f}arcmin.png"

        try:
            png_params = build_request_params(ra, dec, subset_arcmin, "png")
            png_payload = download_payload(png_params)
            extracted = extract_matching_file(png_payload, (".png",))
            if extracted is None:
                png_status = "png_not_found_in_zip"
                png_detail = "API response zip contained no PNG file"
            else:
                zip_name, file_bytes = extracted
                png_out.write_bytes(file_bytes)
                is_blank, blank_detail = png_is_visually_blank(png_out)
                if is_blank:
                    png_status = "png_downloaded_but_blank"
                    png_detail = f"{zip_name} | {blank_detail}"
                else:
                    png_status = "png_downloaded_usable"
                    png_detail = f"{zip_name} | {blank_detail}"
                    if best_png_path is None:
                        best_png_path = png_out
        except Exception as e:
            png_status = "png_download_failed"
            png_detail = str(e)

        # FITS retry
        fits_status = "not_attempted"
        fits_detail = ""
        fits_out = image_dir / f"{safe_name}_retry_s{subset_arcmin:.4f}arcmin.fits"

        try:
            fits_params = build_request_params(ra, dec, subset_arcmin, "fits")
            fits_payload = download_payload(fits_params)
            extracted = extract_matching_file(fits_payload, (".fits", ".fit", ".fts"))
            if extracted is None:
                fits_status = "fits_not_found_in_zip"
                fits_detail = "API response zip contained no FITS file"
            else:
                zip_name, file_bytes = extracted
                fits_out.write_bytes(file_bytes)
                fits_status = "fits_downloaded"
                fits_detail = zip_name
                if best_fits_path is None:
                    best_fits_path = fits_out
        except Exception as e:
            fits_status = "fits_download_failed"
            fits_detail = str(e)

        summary_rows.append(
            {
                "wise_name": wise_name,
                "ra": ra,
                "dec": dec,
                "radius_arcsec": radius_arcsec,
                "subsetsize_arcmin": subset_arcmin,
                "png_output": str(png_out) if png_out.exists() else "",
                "png_status": png_status,
                "png_detail": png_detail,
                "fits_output": str(fits_out) if fits_out.exists() else "",
                "fits_status": fits_status,
                "fits_detail": fits_detail,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = log_dir / f"{safe_name}_wise_retry_summary_{timestamp}.csv"
    latest_summary_path = log_dir / f"{safe_name}_wise_retry_summary_latest.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(latest_summary_path, index=False, encoding="utf-8-sig")

    manifest_lines = [
        "Validation of Structural Contrast Baseline",
        "WISE single-target retry downloader manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"target_name: {wise_name}",
        f"input_file: {project_root / INPUT_FILE}",
        f"image_output_dir: {image_dir}",
        f"log_output_dir: {log_dir}",
        f"finderchart_api: {FINDERCHART_API}",
        f"subsetsize_candidates_arcmin: {subset_candidates}",
        "",
        "Notes",
        "-" * 20,
        "This retry script is intended for targets whose PNG cutout appears visually blank.",
        "It retries multiple subset sizes.",
        "It requests both PNG and FITS.",
        "PNG is checked for visual blankness.",
        f"best_png_path: {best_png_path if best_png_path else 'none'}",
        f"best_fits_path: {best_fits_path if best_fits_path else 'none'}",
    ]
    manifest_text = "\n".join(manifest_lines)

    manifest_path = log_dir / f"{safe_name}_wise_retry_manifest_{timestamp}.txt"
    latest_manifest_path = log_dir / f"{safe_name}_wise_retry_manifest_latest.txt"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    latest_manifest_path.write_text(manifest_text, encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - WISE single-target retry downloader v1")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Target       : {wise_name}")
    print(f"Input file   : {project_root / INPUT_FILE}")
    print(f"Image dir    : {image_dir}")
    print(f"Log dir      : {log_dir}")
    print("")
    print("[OK] Created:")
    print(summary_path)
    print(latest_summary_path)
    print(manifest_path)
    print(latest_manifest_path)
    if best_png_path:
        print(best_png_path)
    if best_fits_path:
        print(best_fits_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
