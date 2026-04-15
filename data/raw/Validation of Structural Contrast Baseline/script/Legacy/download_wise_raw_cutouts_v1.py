#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
WISE raw cutout downloader v1

Author: Kwon Dominicus

Purpose
-------
This script belongs to the raw-data stage and is intended to be placed under:

    data/raw/Validation of Structural Contrast Baseline/script/

It reads the final input file from:

    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/wise_hii_final_input.csv

and downloads WISE PNG cutouts into the raw-data image layer:

    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/

Notes
-----
- This is a raw-data acquisition script, not a derived-data script.
- It requests PNG cutouts from the official IRSA Finder Chart API.
- PNG is suitable for first-pass geometric/profile work, but is not a full
  scientific replacement for FITS products.
- subsetsize is chosen to cover about 1.8R with additional margin.
"""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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

RESULTS_BASE = (
    Path("results")
    / "Validation of Structural Contrast Baseline"
    / "output"
    / "wise_cutout_downloader"
)

FINDERCHART_API = "https://irsa.ipac.caltech.edu/applications/finderchart/servlet/api"
REQUEST_TIMEOUT_SEC = 120


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def sanitize_wise_name(name: str) -> str:
    text = str(name).strip()
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def create_timestamped_output_dir(project_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = project_root / RESULTS_BASE / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


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
        "file_type": "png",
        "locstr": locstr,
        "subsetsize": f"{subsetsize:.4f}",
        "marker": "false",
        "grid": "false",
    }


def find_png_in_zip(zip_bytes: bytes) -> Optional[tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        png_names = [n for n in zf.namelist() if n.lower().endswith(".png")]
        if not png_names:
            return None
        png_names.sort()
        chosen = png_names[0]
        return chosen, zf.read(chosen)


def download_cutout(params: dict) -> bytes:
    response = requests.get(FINDERCHART_API, params=params, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return response.content


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = create_timestamped_output_dir(project_root)
    image_dir = project_root / IMAGE_OUTPUT_DIR
    image_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)

    summary_rows = []

    for _, row in df.iterrows():
        wise_name = str(row["wise_name"])
        ra = float(row["ra_num"])
        dec = float(row["dec_num"])
        radius_arcsec = float(row["radius_arcsec_num"])

        params = build_request_params(ra, dec, radius_arcsec)
        output_png = image_dir / f"{sanitize_wise_name(wise_name)}.png"

        status = "unknown"
        detail = ""
        try:
            content = download_cutout(params)
            found = find_png_in_zip(content)
            if found is None:
                status = "downloaded_but_no_png_found"
                detail = "API response zip contained no PNG file"
            else:
                zip_name, png_bytes = found
                output_png.write_bytes(png_bytes)
                status = "downloaded_png"
                detail = zip_name
        except Exception as e:
            status = "download_failed"
            detail = str(e)

        summary_rows.append(
            {
                "wise_name": wise_name,
                "ra": ra,
                "dec": dec,
                "radius_arcsec": radius_arcsec,
                "subsetsize_arcmin": compute_subsetsize_arcmin(radius_arcsec),
                "output_png": str(output_png) if output_png.exists() else "",
                "status": status,
                "detail": detail,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "wise_cutout_download_summary.csv", index=False, encoding="utf-8-sig")

    manifest_lines = [
        "Validation of Structural Contrast Baseline",
        "WISE raw cutout downloader manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"input_file: {project_root / INPUT_FILE}",
        f"image_output_dir: {image_dir}",
        f"results_output_dir: {out_dir}",
        f"finderchart_api: {FINDERCHART_API}",
        "",
        "Notes",
        "-" * 20,
        "This script belongs to the raw-data stage.",
        "It downloads WISE PNG cutouts from the IRSA Finder Chart API into the raw image layer.",
        "subsetsize is computed to cover approximately out to 1.8R with added margin.",
    ]
    (out_dir / "run_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - WISE raw cutout downloader v1")
    print("=" * 72)
    print(f"Project root      : {project_root}")
    print(f"Input file        : {project_root / INPUT_FILE}")
    print(f"Image output dir  : {image_dir}")
    print(f"Results output dir: {out_dir}")
    print("")
    print("[OK] Created:")
    print(out_dir / "wise_cutout_download_summary.csv")
    print(out_dir / "run_manifest.txt")
    print(image_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
