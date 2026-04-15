#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
WISE raw FITS-only downloader v1

Author: Kwon Dominicus

Purpose
-------
This script belongs to the raw-data stage and is intended to be placed under:

    data/raw/Validation of Structural Contrast Baseline/script/

It reads the final input file from:

    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/wise_hii_final_input.csv

and tries to download FITS cutouts into:

    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/

Summary and manifest files are saved into:

    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/logs/

Practical note
--------------
This script is deliberately FITS-only.
It does not attempt PNG fallback.
"""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import requests

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

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


def compute_subsetsize_arcmin(radius_arcsec: float) -> float:
    size_arcsec = max(2.0 * 1.8 * radius_arcsec + 30.0, 60.0)
    size_arcmin = size_arcsec / 60.0
    return float(min(max(size_arcmin, 0.1), 60.0))


def build_request_params(ra: float, dec: float, radius_arcsec: float) -> dict:
    locstr = f"{ra:.8f} {dec:.8f}"
    subsetsize = compute_subsetsize_arcmin(radius_arcsec)
    return {
        "mode": "getImage",
        "survey": "WISE",
        "file_type": "fits",
        "locstr": locstr,
        "subsetsize": f"{subsetsize:.4f}",
        "marker": "false",
        "grid": "false",
    }


def download_payload(params: dict) -> bytes:
    response = requests.get(FINDERCHART_API, params=params, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return response.content


def try_extract_fits_from_zip(payload: bytes) -> Tuple[str, Optional[bytes], str]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            names = zf.namelist()
            fits_names = [n for n in names if n.lower().endswith((".fits", ".fit", ".fts"))]
            if not fits_names:
                return "zip_no_fits_found", None, "; ".join(names[:20])
            fits_names.sort()
            chosen = fits_names[0]
            return "zip_fits_found", zf.read(chosen), chosen
    except zipfile.BadZipFile:
        return "not_a_zip_response", None, ""


def save_nonzip_payload_if_fits_like(payload: bytes, output_path: Path) -> Tuple[bool, str]:
    head = payload[:64]
    # FITS files usually begin with ASCII SIMPLE  =
    if b"SIMPLE" in head:
        output_path.write_bytes(payload)
        return True, "raw_payload_looks_like_fits"
    return False, f"nonzip_nonfits_payload_prefix={head[:24]!r}"


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    image_dir = project_root / IMAGE_OUTPUT_DIR
    image_dir.mkdir(parents=True, exist_ok=True)

    log_dir = project_root / LOG_OUTPUT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)

    summary_rows = []

    for _, row in df.iterrows():
        wise_name = str(row["wise_name"])
        ra = float(row["ra_num"])
        dec = float(row["dec_num"])
        radius_arcsec = float(row["radius_arcsec_num"])

        params = build_request_params(ra, dec, radius_arcsec)
        output_fits = image_dir / f"{sanitize_wise_name(wise_name)}.fits"

        status = "unknown"
        detail = ""
        response_kind = ""

        try:
            payload = download_payload(params)
            zip_status, extracted_bytes, zip_detail = try_extract_fits_from_zip(payload)
            response_kind = zip_status

            if zip_status == "zip_fits_found" and extracted_bytes is not None:
                output_fits.write_bytes(extracted_bytes)
                status = "downloaded_fits"
                detail = zip_detail
            elif zip_status == "zip_no_fits_found":
                status = "download_failed_no_fits_in_zip"
                detail = zip_detail
            else:
                saved, save_detail = save_nonzip_payload_if_fits_like(payload, output_fits)
                if saved:
                    status = "downloaded_fits_raw_response"
                    detail = save_detail
                else:
                    status = "download_failed_nonfits_response"
                    detail = save_detail

        except Exception as e:
            status = "download_failed_exception"
            detail = str(e)

        summary_rows.append(
            {
                "wise_name": wise_name,
                "ra": ra,
                "dec": dec,
                "radius_arcsec": radius_arcsec,
                "subsetsize_arcmin": compute_subsetsize_arcmin(radius_arcsec),
                "output_fits": str(output_fits) if output_fits.exists() else "",
                "status": status,
                "response_kind": response_kind,
                "detail": detail,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = log_dir / f"wise_fits_download_summary_{timestamp}.csv"
    latest_summary_path = log_dir / "wise_fits_download_summary_latest.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(latest_summary_path, index=False, encoding="utf-8-sig")

    manifest_lines = [
        "Validation of Structural Contrast Baseline",
        "WISE raw FITS-only downloader manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"input_file: {project_root / INPUT_FILE}",
        f"image_output_dir: {image_dir}",
        f"log_output_dir: {log_dir}",
        f"finderchart_api: {FINDERCHART_API}",
        "",
        "Notes",
        "-" * 20,
        "This script belongs to the raw-data stage.",
        "It requests FITS cutouts only.",
        "It first tries to interpret the response as a zip archive containing FITS.",
        "If the response is not a zip, it checks whether the raw payload itself looks like FITS.",
        "subsetsize is computed to cover approximately out to 1.8R with added margin.",
    ]
    manifest_text = "\n".join(manifest_lines)

    manifest_path = log_dir / f"wise_fits_download_manifest_{timestamp}.txt"
    latest_manifest_path = log_dir / "wise_fits_download_manifest_latest.txt"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    latest_manifest_path.write_text(manifest_text, encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - WISE raw FITS-only downloader v1")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Input file   : {project_root / INPUT_FILE}")
    print(f"Image dir    : {image_dir}")
    print(f"Log dir      : {log_dir}")
    print("")
    print("[OK] Created:")
    print(summary_path)
    print(latest_summary_path)
    print(manifest_path)
    print(latest_manifest_path)
    print(image_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
