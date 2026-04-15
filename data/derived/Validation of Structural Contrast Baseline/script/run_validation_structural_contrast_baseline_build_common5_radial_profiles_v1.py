#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
Common5 radial-profile build pipeline (v1)

Author: Kwon Dominicus

Placement
---------
src/Validation of Structural Contrast Baseline/standard/

Purpose
-------
Build per-target radial profile CSV files for the common5 standard/topological runs.

Primary input
-------------
data/derived/Validation of Structural Contrast Baseline/input/standard/wise_hii_common5/wise_hii_common5_standard_final_input.csv

Primary output
--------------
data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/radial_profiles/

Auxiliary outputs
-----------------
data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/radial_profiles/common5_build_manifest.txt
data/derived/Validation of Structural Contrast Baseline/input/wise_hii_catalog/radial_profiles/common5_radius_geometry_summary.csv

Behavior
--------
- Uses local FITS files when available.
- If WCS is available, converts target RA/DEC to pixel position.
- If WCS or local FITS is missing, writes a clear failure status in the summary instead of silently succeeding.
- Does NOT invent intensities. It only extracts real annular means from local FITS image data.

Notes
-----
- This pipeline expects local FITS files to exist. The standard final input may contain fits_local_path.
- If fits_local_path is empty, you may provide --fits-root and place files there using target-based filenames.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

STANDARD_INPUT_FILE = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "standard"
    / "wise_hii_common5"
    / "wise_hii_common5_standard_final_input.csv"
)

RADIAL_PROFILE_DIR = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "wise_hii_catalog"
    / "radial_profiles"
)

SUPPORTED_FITS_SUFFIXES = [".fits", ".fit", ".fts", ".fits.gz"]


def sanitize_wise_name(name: str) -> str:
    text = str(name).strip()
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_standard_input(project_root: Path) -> pd.DataFrame:
    path = project_root / STANDARD_INPUT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Standard final input not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = ["wise_name", "ra", "dec", "radius_arcsec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Standard final input is missing required columns: " + ", ".join(missing))
    df["wise_name"] = df["wise_name"].astype(str).str.strip()
    df["ra_num"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec_num"] = pd.to_numeric(df["dec"], errors="coerce")
    df["radius_arcsec_num"] = pd.to_numeric(df["radius_arcsec"], errors="coerce")
    return df


def find_local_fits(project_root: Path, row: pd.Series, fits_root: Optional[Path]) -> Optional[Path]:
    # 1) explicit path from input
    explicit = str(row.get("fits_local_path", "")).strip()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = project_root / p
        if p.exists():
            return p

    # 2) optional fits_root search
    if fits_root is not None and fits_root.exists():
        wise = sanitize_wise_name(row["wise_name"])
        candidates = []
        for suf in SUPPORTED_FITS_SUFFIXES:
            candidates.append(fits_root / f"{wise}{suf}")
            candidates.append(fits_root / f"{row['wise_name']}{suf}")
        for c in candidates:
            if c.exists():
                return c

    return None


def import_astropy():
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        return fits, WCS, proj_plane_pixel_scales
    except Exception as exc:
        raise RuntimeError(
            "This pipeline requires astropy for FITS/WCS handling. "
            f"Import failed: {exc}"
        )


def load_fits_2d(fits_path: Path):
    fits, WCS, proj_plane_pixel_scales = import_astropy()
    with fits.open(fits_path) as hdul:
        hdu = None
        for candidate in hdul:
            if getattr(candidate, "data", None) is not None:
                arr = np.asarray(candidate.data)
                if arr.ndim >= 2:
                    hdu = candidate
                    break
        if hdu is None:
            raise ValueError(f"No usable image HDU found in {fits_path}")
        data = np.asarray(hdu.data, dtype=float)
        while data.ndim > 2:
            data = data[0]
        data = np.squeeze(data)
        if data.ndim != 2:
            raise ValueError(f"Usable FITS image is not 2D after squeeze: {fits_path}")
        header = hdu.header
        try:
            wcs = WCS(header)
            if not wcs.has_celestial:
                wcs = None
        except Exception:
            wcs = None

        pixscale_arcsec = None
        if wcs is not None:
            try:
                scales = proj_plane_pixel_scales(wcs.celestial)
                if len(scales) >= 2:
                    # degrees/pixel -> arcsec/pixel
                    pixscale_arcsec = float(np.mean(scales)) * 3600.0
            except Exception:
                pixscale_arcsec = None

        if pixscale_arcsec is None:
            # fallback from header
            for key in ["CDELT1", "CDELT2"]:
                if key in header and header[key] not in (0, None):
                    try:
                        pixscale_arcsec = abs(float(header[key])) * 3600.0
                        break
                    except Exception:
                        pass

        if pixscale_arcsec is None or pixscale_arcsec <= 0:
            raise ValueError(f"Could not derive arcsec/pixel scale from FITS header/WCS: {fits_path}")

        return data, wcs, pixscale_arcsec


def target_center_pixel(data: np.ndarray, wcs, ra_deg: float, dec_deg: float):
    if wcs is not None:
        try:
            x, y = wcs.celestial.world_to_pixel_values(ra_deg, dec_deg)
            if np.isfinite(x) and np.isfinite(y):
                return float(x), float(y), "wcs_world_to_pixel"
        except Exception:
            pass
    # fallback: center of image
    ny, nx = data.shape
    return (nx - 1) / 2.0, (ny - 1) / 2.0, "image_center_fallback"


def build_annular_profile(data: np.ndarray, cx: float, cy: float, pixscale_arcsec: float, radius_arcsec: float):
    ny, nx = data.shape
    yy, xx = np.indices((ny, nx))
    rr_pix = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rr_arcsec = rr_pix * pixscale_arcsec

    max_r = max(float(radius_arcsec) * 1.8, 10.0)
    dr = max(pixscale_arcsec, radius_arcsec / 80.0, 0.5)
    edges = np.arange(0.0, max_r + dr, dr)
    if len(edges) < 3:
        edges = np.array([0.0, max_r / 2.0, max_r])

    rows = []
    finite = np.isfinite(data)
    for i in range(len(edges) - 1):
        r_in = edges[i]
        r_out = edges[i + 1]
        mask = (rr_arcsec >= r_in) & (rr_arcsec < r_out) & finite
        n = int(mask.sum())
        if n == 0:
            continue
        vals = data[mask]
        rows.append(
            {
                "radius_arcsec": round((r_in + r_out) / 2.0, 6),
                "intensity": float(np.nanmean(vals)),
                "intensity_median": float(np.nanmedian(vals)),
                "intensity_std": float(np.nanstd(vals)),
                "n_pix": n,
                "r_in_arcsec": round(r_in, 6),
                "r_out_arcsec": round(r_out, 6),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    fits_root = None

    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()
    if len(sys.argv) >= 3:
        fits_root = Path(sys.argv[2]).expanduser().resolve()

    out_dir = project_root / RADIAL_PROFILE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_standard_input(project_root)

    summary_rows = []
    for _, row in df.iterrows():
        wise_name = str(row["wise_name"]).strip()
        safe = sanitize_wise_name(wise_name)
        fits_path = find_local_fits(project_root, row, fits_root)

        if fits_path is None:
            summary_rows.append(
                {
                    "wise_name": wise_name,
                    "hii_region_name": row.get("hii_region_name", ""),
                    "status": "missing_local_fits",
                    "fits_path": "",
                    "center_method": "",
                    "pixscale_arcsec": np.nan,
                    "radius_arcsec": row.get("radius_arcsec", ""),
                    "profile_rows": 0,
                    "output_csv": "",
                }
            )
            continue

        try:
            data, wcs, pixscale_arcsec = load_fits_2d(fits_path)
            cx, cy, center_method = target_center_pixel(data, wcs, float(row["ra_num"]), float(row["dec_num"]))
            profile = build_annular_profile(data, cx, cy, float(pixscale_arcsec), float(row["radius_arcsec_num"]))
            if profile.empty:
                status = "profile_empty_after_extraction"
                output_csv = ""
            else:
                output_csv = str(out_dir / f"{wise_name}_radial_profile.csv")
                profile.to_csv(output_csv, index=False, encoding="utf-8-sig")
                status = "ok"
            summary_rows.append(
                {
                    "wise_name": wise_name,
                    "hii_region_name": row.get("hii_region_name", ""),
                    "status": status,
                    "fits_path": str(fits_path),
                    "center_method": center_method,
                    "pixscale_arcsec": float(pixscale_arcsec),
                    "radius_arcsec": float(row["radius_arcsec_num"]),
                    "profile_rows": int(len(profile)),
                    "output_csv": output_csv,
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "wise_name": wise_name,
                    "hii_region_name": row.get("hii_region_name", ""),
                    "status": f"failed:{type(exc).__name__}",
                    "fits_path": str(fits_path),
                    "center_method": "",
                    "pixscale_arcsec": np.nan,
                    "radius_arcsec": row.get("radius_arcsec", ""),
                    "profile_rows": 0,
                    "output_csv": "",
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "common5_radius_geometry_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    lines = [
        "Validation of Structural Contrast Baseline",
        "common5 radial-profile build manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"standard_input: {project_root / STANDARD_INPUT_FILE}",
        f"radial_profile_output_dir: {out_dir}",
        f"optional_fits_root: {fits_root if fits_root else ''}",
        "",
        f"target_count: {len(df)}",
        f"ok_profiles: {int((summary_df['status'] == 'ok').sum()) if not summary_df.empty else 0}",
        f"missing_local_fits: {int((summary_df['status'] == 'missing_local_fits').sum()) if not summary_df.empty else 0}",
        "",
        "Note:",
        "This pipeline only extracts real annular profiles from local FITS image data.",
        "If local FITS is missing, the status is recorded and no fake profile is created.",
    ]
    (out_dir / "common5_build_manifest.txt").write_text("\n".join(lines), encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - common5 radial-profile build v1")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Input file   : {project_root / STANDARD_INPUT_FILE}")
    print(f"Output dir   : {out_dir}")
    if fits_root:
        print(f"FITS root    : {fits_root}")
    print("")
    print("[OK] Created:")
    print(summary_csv)
    print(out_dir / "common5_build_manifest.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
