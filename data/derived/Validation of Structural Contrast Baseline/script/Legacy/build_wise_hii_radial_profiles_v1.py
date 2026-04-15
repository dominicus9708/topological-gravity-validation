#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Radial profile builder v1

Author: Kwon Dominicus

Purpose
-------
This pipeline reads the final input file directly from:

    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/wise_hii_final_input.csv

and generates per-target radial profile CSV files for later use by the
topological pipeline.

Expected image locations
------------------------
For each target name W, place an observational image file at one of:

    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/W.png
    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/W.jpg
    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/W.jpeg
    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/W.tif
    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/W.tiff
    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/images/W.npy

Safe practical assumptions in v1
--------------------------------
- The image center is assumed to be the center of the target.
- radius_arcsec from the input file is used only as the target scale.
- If no reliable pixel scale file exists, this v1 assumes:
      1 pixel = 1 arcsec
- This is a practical first pass and should be documented as such.

Optional pixel scale override
-----------------------------
You may provide:

    data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/image_metadata.csv

with columns:
    wise_name,pixel_scale_arcsec_per_pixel

If present, the per-target pixel scale will be used.

Outputs
-------
data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/radial_profiles/
    W_radial_profile.csv

results/Validation of Structural Contrast Baseline/output/radial_profile_builder/YYYYMMDD_HHMMSS/
    radial_profile_build_summary.csv
    run_manifest.txt

Execution
---------
cd /d C:\Users\mincu\Desktop\topological_gravity_project
python "src\Validation of Structural Contrast Baseline\topological\build_wise_hii_radial_profiles_v1.py"
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

IMAGE_DIR = (
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

RESULTS_BASE = (
    Path("results")
    / "Validation of Structural Contrast Baseline"
    / "output"
    / "radial_profile_builder"
)


@dataclass
class TargetInfo:
    wise_name: str
    radius_arcsec: float


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
    required = ["wise_name", "radius_arcsec", "target_status", "input_ready"]
    ensure_required_columns(df, required)

    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    if df["radius_arcsec_num"].isna().any():
        bad = df.loc[df["radius_arcsec_num"].isna(), "wise_name"].tolist()
        raise ValueError("Invalid radius_arcsec values for: " + ", ".join(map(str, bad)))

    df["wise_name"] = df["wise_name"].astype(str).str.strip()

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


def get_pixel_scale_arcsec_per_pixel(wise_name: str, metadata: pd.DataFrame) -> float:
    row = metadata.loc[metadata["wise_name"] == wise_name]
    if row.empty:
        return 1.0
    return float(row.iloc[0]["pixel_scale_arcsec_per_pixel"])


def find_image_file(project_root: Path, wise_name: str) -> Optional[Path]:
    base = project_root / IMAGE_DIR
    stems = [wise_name, sanitize_wise_name(wise_name)]
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"]
    for stem in stems:
        for ext in exts:
            p = base / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def load_image_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
    else:
        arr = mpimg.imread(path)

    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"Unsupported image shape {arr.shape} for {path}")
    return arr.astype(float)


def radial_profile_from_image(
    image: np.ndarray,
    pixel_scale_arcsec_per_pixel: float,
    max_radius_arcsec: Optional[float] = None,
) -> pd.DataFrame:
    h, w = image.shape
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0

    y, x = np.indices(image.shape)
    r_pix = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_arcsec = r_pix * pixel_scale_arcsec_per_pixel

    if max_radius_arcsec is None:
        max_radius_arcsec = float(r_arcsec.max())

    max_bin = int(np.floor(max_radius_arcsec))
    rows = []
    for i in range(max_bin + 1):
        r_in = float(i)
        r_out = float(i + 1)
        mask = (r_arcsec >= r_in) & (r_arcsec < r_out)
        if not np.any(mask):
            continue
        intensity = float(np.nanmean(image[mask]))
        rows.append(
            {
                "radius_arcsec": 0.5 * (r_in + r_out),
                "intensity": intensity,
                "n_pixels": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def save_profile_plot(
    wise_name: str,
    profile_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(profile_df["radius_arcsec"], profile_df["intensity"])
    ax.set_xlabel("Radius (arcsec)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{wise_name} radial profile")
    fig.tight_layout()
    fig.savefig(out_dir / f"{sanitize_wise_name(wise_name)}_radial_profile_preview.png", dpi=180)
    plt.close(fig)


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    results_out = create_timestamped_output_dir(project_root)
    profile_out_dir = project_root / PROFILE_OUTPUT_DIR
    profile_out_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_final_input(project_root)
    df = normalize_input(raw_df)
    metadata = load_image_metadata(project_root)

    summary_rows = []

    for _, row in df.iterrows():
        wise_name = str(row["wise_name"])
        radius_arcsec = float(row["radius_arcsec_num"])

        image_path = find_image_file(project_root, wise_name)
        target_preview_dir = results_out / sanitize_wise_name(wise_name)
        target_preview_dir.mkdir(parents=True, exist_ok=True)

        if image_path is None:
            summary_rows.append(
                {
                    "wise_name": wise_name,
                    "radius_arcsec": radius_arcsec,
                    "image_file": "",
                    "pixel_scale_arcsec_per_pixel": "",
                    "profile_file": "",
                    "status": "missing_image_file",
                }
            )
            continue

        pixel_scale = get_pixel_scale_arcsec_per_pixel(wise_name, metadata)
        image = load_image_array(image_path)

        # Build profile out to 1.8R so that topological zones are covered
        profile_df = radial_profile_from_image(
            image=image,
            pixel_scale_arcsec_per_pixel=pixel_scale,
            max_radius_arcsec=1.8 * radius_arcsec,
        )

        profile_path = profile_out_dir / f"{sanitize_wise_name(wise_name)}_radial_profile.csv"
        profile_df.to_csv(profile_path, index=False, encoding="utf-8-sig")
        save_profile_plot(wise_name, profile_df, target_preview_dir)

        summary_rows.append(
            {
                "wise_name": wise_name,
                "radius_arcsec": radius_arcsec,
                "image_file": str(image_path),
                "pixel_scale_arcsec_per_pixel": pixel_scale,
                "profile_file": str(profile_path),
                "status": "profile_built",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_out / "radial_profile_build_summary.csv", index=False, encoding="utf-8-sig")

    manifest_lines = [
        "Validation of Structural Contrast Baseline",
        "Radial profile builder manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"input_file: {project_root / INPUT_FILE}",
        f"image_dir: {project_root / IMAGE_DIR}",
        f"image_metadata_file: {project_root / IMAGE_METADATA_FILE}",
        f"profile_output_dir: {profile_out_dir}",
        f"results_output_dir: {results_out}",
        "",
        "Notes",
        "-" * 20,
        "This v1 builder assumes the target center is the image center.",
        "If image_metadata.csv is absent, 1 pixel = 1 arcsec is assumed.",
        "Profiles are built out to 1.8R so that topological inner/shell/background zones are covered.",
    ]
    (results_out / "run_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - radial profile builder v1")
    print("=" * 72)
    print(f"Project root      : {project_root}")
    print(f"Input file        : {project_root / INPUT_FILE}")
    print(f"Image dir         : {project_root / IMAGE_DIR}")
    print(f"Profile output dir: {profile_out_dir}")
    print(f"Results output dir: {results_out}")
    print("")
    print("[OK] Created:")
    print(results_out / "radial_profile_build_summary.csv")
    print(results_out / "run_manifest.txt")
    print(profile_out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
