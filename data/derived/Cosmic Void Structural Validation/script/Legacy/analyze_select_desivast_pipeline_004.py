#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except Exception as exc:
    raise SystemExit(
        "astropy is required for this pipeline. Please install it with:\n"
        "pip install astropy"
    ) from exc


RAW_DEFAULT = Path("data") / "raw" / "Cosmic Void Structural Validation" / "DESIVAST"
DERIVED_DEFAULT = Path("data") / "derived" / "Cosmic Void Structural Validation" / "derived"

SUPPORTED_FILES = {
    "VoidFinder_NGC": "DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits",
    "VoidFinder_SGC": "DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits",
    "ZOBOV_NGC": "DESIVAST_BGS_VOLLIM_V2_ZOBOV_NGC.fits",
    "ZOBOV_SGC": "DESIVAST_BGS_VOLLIM_V2_ZOBOV_SGC.fits",
    "VIDE_NGC": "DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits",
    "VIDE_SGC": "DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits",
    "REVOLVER_NGC": "DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits",
    "REVOLVER_SGC": "DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits",
}
SHA_FILE = "dr1_vac_dr1_desivast_v1.0.sha256sum"

# Important revision in 004:
# - Do NOT treat bare "Z" as physical redshift.
# - Treat it as coordinate-like axis unless an explicit redshift-named column exists.

CANONICAL_SYNONYMS = {
    "ra": ["RA", "ra", "RACEN", "RA_CEN", "ra_cen", "ra_center", "RA_CENTER"],
    "dec": ["DEC", "dec", "DECCEN", "DEC_CEN", "dec_cen", "dec_center", "DEC_CENTER"],

    # Explicit redshift names only
    "redshift": [
        "REDSHIFT", "redshift", "ZRED", "zred", "Z_RED", "z_red",
        "Z_COSMO", "z_cosmo", "REDSHIFT_CEN", "redshift_cen", "ZSPEC", "zspec"
    ],

    # Cartesian / coordinate-like axes
    "coord_x": ["X", "x", "XCEN", "X_CEN", "x_cen", "X_CENTER", "x_center"],
    "coord_y": ["Y", "y", "YCEN", "Y_CEN", "y_cen", "Y_CENTER", "y_center"],
    "coord_z": ["Z", "z", "ZCEN", "Z_CEN", "z_cen", "Z_CENTER", "z_center"],

    "radius_eff": [
        "EFFECTIVE_RADIUS", "effective_radius", "R_EFF", "r_eff", "REFF", "RADIUS",
        "RADIUS_EFF", "radius_eff", "rv", "RV"
    ],
    "radius_eff_unc": [
        "EFFECTIVE_RADIUS_UNCERT", "effective_radius_uncert", "R_EFF_ERR", "REFF_ERR", "radius_eff_err"
    ],
    "edge_area": ["EDGE_AREA", "edge_area", "AREA_EDGE", "FRAC_EDGE", "frac_edge"],
    "tot_area": ["TOT_AREA", "tot_area", "TOTAL_AREA", "AREA_TOT"],
    "void_id": ["VOID_ID", "void_id", "ID", "id", "VID", "voidid", "VOID"],
    "n_members": ["N_MEMBERS", "N_GAL", "NGAL", "n_gal", "NPARTICLES", "N_PARTICLES", "n_members"],
}


@dataclass
class CatalogContext:
    key: str
    path: Path
    algorithm: str
    sky_region: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--raw-root", default=str(RAW_DEFAULT), help="DESIVAST raw input folder")
    parser.add_argument("--output-root", default=str(DERIVED_DEFAULT), help="Derived output folder")
    parser.add_argument("--representative-count", type=int, default=6, help="Number of representative cases")
    parser.add_argument("--stacked-pool-count", type=int, default=36, help="Stacked-pool target size")
    parser.add_argument("--verify-sha256", action="store_true", help="Verify files against sha256sum if present")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite standardized CSV files")
    parser.add_argument("--dedupe-ra-dec-threshold-deg", type=float, default=0.05, help="Duplicate filtering threshold in degrees")
    parser.add_argument("--dedupe-radius-rel-tol", type=float, default=0.03, help="Relative radius tolerance for duplicate filtering")
    return parser.parse_args()


def resolve_path(project_root: Path, raw_value: str) -> Path:
    p = Path(raw_value)
    return (project_root / p).resolve() if not p.is_absolute() else p.resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_algorithm_and_region(filename: str) -> Tuple[str, str]:
    parts = filename.replace(".fits", "").split("_")
    region = parts[-1]
    if "VoidFinder" in filename:
        return "VoidFinder", region
    if "ZOBOV" in filename:
        return "ZOBOV", region
    if "VIDE" in filename:
        return "VIDE", region
    if "REVOLVER" in filename:
        return "REVOLVER", region
    return "Unknown", region


def parse_sha256_file(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            checksum = parts[0]
            filename = parts[-1].lstrip("*")
            mapping[filename] = checksum
    return mapping


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(raw_root: Path, output_root: Path) -> pd.DataFrame:
    sum_path = raw_root / SHA_FILE
    expected = parse_sha256_file(sum_path)
    rows = []
    for filename in [SHA_FILE, *SUPPORTED_FILES.values()]:
        file_path = raw_root / filename
        row = {
            "filename": filename,
            "exists": file_path.exists(),
            "expected_sha256": expected.get(filename, ""),
            "actual_sha256": "",
            "match": "",
        }
        if file_path.exists() and filename in expected and filename != SHA_FILE:
            try:
                actual = compute_sha256(file_path)
                row["actual_sha256"] = actual
                row["match"] = actual.lower() == expected[filename].lower()
            except Exception as exc:
                row["actual_sha256"] = f"ERROR: {exc}"
                row["match"] = False
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "desivast_sha256_verification.csv", index=False, encoding="utf-8-sig")
    return df


def read_fits_table(path: Path) -> pd.DataFrame:
    with fits.open(path, memmap=False) as hdul:
        table_hdu = None
        for hdu in hdul:
            if getattr(hdu, "data", None) is not None and hasattr(hdu.data, "dtype"):
                table_hdu = hdu
                break
        if table_hdu is None:
            raise ValueError(f"No table HDU found in {path}")
        data = table_hdu.data
        cols = table_hdu.columns.names
        records = {}
        for col in cols:
            arr = data[col]
            if getattr(arr, "ndim", 1) > 1:
                if arr.shape[1] == 1:
                    arr = arr[:, 0]
                else:
                    continue
            if getattr(arr, "dtype", None) is not None and arr.dtype.kind == "S":
                arr = arr.astype(str)
            records[col] = np.array(arr)
        return pd.DataFrame(records)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    return out


def to_native_endian_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    arr = s.to_numpy(copy=True)
    dtype = getattr(arr, "dtype", None)
    if dtype is not None and hasattr(dtype, "byteorder") and dtype.byteorder not in ("=", "|"):
        native_dtype = dtype.newbyteorder("=")
        arr = arr.byteswap().view(native_dtype)
    return pd.Series(arr, index=s.index, name=s.name)


def first_matching_column(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        lc = candidate.lower()
        if lc in cols_lower:
            return cols_lower[lc]
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    return to_native_endian_series(pd.to_numeric(series, errors="coerce"))


def standardize_catalog(df: pd.DataFrame, ctx: CatalogContext) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = normalize_columns(df)
    matched: Dict[str, str] = {}
    standardized = pd.DataFrame(index=df.index)

    for canonical, candidate_cols in CANONICAL_SYNONYMS.items():
        match = first_matching_column(df.columns, candidate_cols)
        matched[canonical] = match if match is not None else ""
        standardized[canonical] = df[match] if match is not None else np.nan

    standardized["source_catalog_key"] = ctx.key
    standardized["source_filename"] = ctx.path.name
    standardized["algorithm"] = ctx.algorithm
    standardized["sky_region"] = ctx.sky_region
    standardized["source_row_index"] = np.arange(len(standardized), dtype=np.int64)

    numeric_cols = [
        "ra", "dec", "redshift", "coord_x", "coord_y", "coord_z",
        "radius_eff", "radius_eff_unc", "edge_area", "tot_area", "n_members"
    ]
    for col in numeric_cols:
        standardized[col] = safe_numeric(standardized[col])

    void_id_series = standardized["void_id"].astype(str).fillna("")
    standardized["void_id"] = np.where(
        (void_id_series == "nan") | (void_id_series == "") | (void_id_series == "None"),
        standardized["source_catalog_key"].astype(str) + "_" + standardized["source_row_index"].astype(str),
        void_id_series
    )

    standardized["edge_fraction"] = np.where(
        standardized["tot_area"].notna() & (standardized["tot_area"] > 0),
        standardized["edge_area"] / standardized["tot_area"],
        np.nan
    )

    standardized["coord_r3d"] = np.where(
        standardized["coord_x"].notna() & standardized["coord_y"].notna() & standardized["coord_z"].notna(),
        np.sqrt(standardized["coord_x"] ** 2 + standardized["coord_y"] ** 2 + standardized["coord_z"] ** 2),
        np.nan
    )

    standardized["distance_proxy"] = standardized["redshift"].where(
        standardized["redshift"].notna(),
        standardized["coord_r3d"]
    )

    standardized["distance_kind"] = np.where(
        standardized["redshift"].notna(),
        "explicit_redshift",
        np.where(standardized["coord_r3d"].notna(), "coord_r3d_proxy", "missing")
    )

    standardized["has_basic_geometry"] = (
        standardized["ra"].notna() & standardized["dec"].notna() & standardized["radius_eff"].notna()
    )
    standardized["has_distance_proxy"] = standardized["distance_proxy"].notna()

    standardized["radius_rank_pct"] = safe_numeric(standardized["radius_eff"]).rank(pct=True)
    standardized["distance_rank_pct"] = safe_numeric(standardized["distance_proxy"]).rank(pct=True)

    return standardized, matched


def safe_min(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.min()) if len(s) else float("nan")


def safe_median(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.median()) if len(s) else float("nan")


def safe_max(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.max()) if len(s) else float("nan")


def safe_missing_frac(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return float(series.isna().mean())


def summarize_catalog(df_raw: pd.DataFrame, df_std: pd.DataFrame, ctx: CatalogContext, matched: Dict[str, str]) -> Dict[str, object]:
    return {
        "catalog_key": ctx.key,
        "filename": ctx.path.name,
        "algorithm": ctx.algorithm,
        "sky_region": ctx.sky_region,
        "rows": int(len(df_raw)),
        "columns_raw": int(len(df_raw.columns)),
        "basic_geometry_rows": int(df_std["has_basic_geometry"].sum()),
        "distance_proxy_rows": int(df_std["has_distance_proxy"].sum()),
        "ra_col": matched.get("ra", ""),
        "dec_col": matched.get("dec", ""),
        "redshift_col": matched.get("redshift", ""),
        "coord_x_col": matched.get("coord_x", ""),
        "coord_y_col": matched.get("coord_y", ""),
        "coord_z_col": matched.get("coord_z", ""),
        "radius_eff_col": matched.get("radius_eff", ""),
        "edge_area_col": matched.get("edge_area", ""),
        "tot_area_col": matched.get("tot_area", ""),
        "void_id_col": matched.get("void_id", ""),
        "n_members_col": matched.get("n_members", ""),
        "distance_kind_counts": str(df_std["distance_kind"].value_counts(dropna=False).to_dict()),
        "distance_proxy_min": safe_min(df_std["distance_proxy"]),
        "distance_proxy_max": safe_max(df_std["distance_proxy"]),
        "radius_eff_min": safe_min(df_std["radius_eff"]),
        "radius_eff_med": safe_median(df_std["radius_eff"]),
        "radius_eff_max": safe_max(df_std["radius_eff"]),
        "edge_fraction_med": safe_median(df_std["edge_fraction"]),
        "missing_ra_frac": safe_missing_frac(df_std["ra"]),
        "missing_dec_frac": safe_missing_frac(df_std["dec"]),
        "missing_distance_proxy_frac": safe_missing_frac(df_std["distance_proxy"]),
        "missing_radius_frac": safe_missing_frac(df_std["radius_eff"]),
    }


def add_selection_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    r = safe_numeric(out["radius_eff"])
    if r.notna().sum() > 0:
        r_pct = r.rank(pct=True)
        out["score_radius"] = 1.0 - (r_pct - 0.72).abs() / 0.72
        out["score_radius"] = out["score_radius"].clip(lower=0, upper=1)
    else:
        out["score_radius"] = 0.0

    ef = pd.to_numeric(out["edge_fraction"], errors="coerce")
    out["score_edge"] = np.where(ef.notna(), 1.0 - np.clip(ef, 0, 1), 0.5)

    out["score_geometry"] = out["has_basic_geometry"].astype(float)

    d = safe_numeric(out["distance_proxy"])
    if d.notna().sum() > 0:
        d_pct = d.rank(pct=True)
        out["score_distance"] = 1.0 - (d_pct - 0.50).abs() / 0.50
        out["score_distance"] = out["score_distance"].clip(lower=0, upper=1)
    else:
        out["score_distance"] = 0.5

    nm = safe_numeric(out["n_members"])
    out["score_members"] = nm.rank(pct=True) if nm.notna().sum() > 0 else 0.5

    out["selection_score"] = (
        0.35 * out["score_radius"]
        + 0.30 * out["score_edge"]
        + 0.15 * out["score_geometry"]
        + 0.10 * out["score_distance"]
        + 0.10 * out["score_members"]
    )

    if r.notna().sum() > 0:
        q1, q2 = r.quantile([0.33, 0.66]).tolist()
        conditions = [r <= q1, (r > q1) & (r <= q2), r > q2]
        choices = ["small", "medium", "large"]
        out["radius_band"] = np.select(conditions, choices, default="unknown")
    else:
        out["radius_band"] = "unknown"

    return out


def angular_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    # small robust spherical separation
    ra1r, dec1r = np.deg2rad([ra1, dec1])
    ra2r, dec2r = np.deg2rad([ra2, dec2])
    sin_d1, cos_d1 = np.sin(dec1r), np.cos(dec1r)
    sin_d2, cos_d2 = np.sin(dec2r), np.cos(dec2r)
    cosang = sin_d1 * sin_d2 + cos_d1 * cos_d2 * np.cos(ra1r - ra2r)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosang)))


def is_duplicate_candidate(
    row: pd.Series,
    chosen: List[pd.Series],
    ra_dec_threshold_deg: float,
    radius_rel_tol: float,
) -> bool:
    for prev in chosen:
        if row["sky_region"] != prev["sky_region"]:
            continue
        if pd.isna(row["ra"]) or pd.isna(row["dec"]) or pd.isna(prev["ra"]) or pd.isna(prev["dec"]):
            continue
        sep = angular_sep_deg(float(row["ra"]), float(row["dec"]), float(prev["ra"]), float(prev["dec"]))
        if sep > ra_dec_threshold_deg:
            continue

        r1 = float(row["radius_eff"]) if pd.notna(row["radius_eff"]) else np.nan
        r2 = float(prev["radius_eff"]) if pd.notna(prev["radius_eff"]) else np.nan
        if np.isnan(r1) or np.isnan(r2):
            return True

        rel = abs(r1 - r2) / max(r1, r2, 1e-12)
        if rel <= radius_rel_tol:
            return True
    return False


def choose_representatives(
    df: pd.DataFrame,
    representative_count: int,
    ra_dec_threshold_deg: float,
    radius_rel_tol: float,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    pool = df.sort_values(["selection_score", "radius_eff"], ascending=[False, False]).copy()
    chosen_rows: List[pd.Series] = []

    desired_by_band = {"small": 2, "medium": 2, "large": 2}
    if representative_count != 6:
        bands = ["small", "medium", "large"]
        base = representative_count // 3
        rem = representative_count % 3
        desired_by_band = {b: base for b in bands}
        for b in bands[:rem]:
            desired_by_band[b] += 1

    used_alg_region = set()

    for band, target_n in desired_by_band.items():
        subset = pool[pool["radius_band"] == band].copy()
        selected_for_band = 0
        for _, row in subset.iterrows():
            key = (row["algorithm"], row["sky_region"])
            if key in used_alg_region and len(subset) > target_n:
                continue
            if is_duplicate_candidate(row, chosen_rows, ra_dec_threshold_deg, radius_rel_tol):
                continue
            chosen_rows.append(row)
            used_alg_region.add(key)
            selected_for_band += 1
            if selected_for_band >= target_n:
                break

    if len(chosen_rows) < representative_count:
        for _, row in pool.iterrows():
            if is_duplicate_candidate(row, chosen_rows, ra_dec_threshold_deg, radius_rel_tol):
                continue
            chosen_rows.append(row)
            if len(chosen_rows) >= representative_count:
                break

    chosen = pd.DataFrame(chosen_rows).drop_duplicates(subset=["source_catalog_key", "void_id"]).reset_index(drop=True)
    chosen["representative_rank"] = np.arange(1, len(chosen) + 1)
    return chosen


def choose_stacked_pool(df: pd.DataFrame, stacked_pool_count: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    pool = df[df["has_basic_geometry"]].copy()
    pool = pool.sort_values(["selection_score", "radius_eff"], ascending=[False, False]).reset_index(drop=True)
    out = pool.head(stacked_pool_count).copy()
    out["stacked_pool_rank"] = np.arange(1, len(out) + 1)
    return out


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    raw_root = resolve_path(project_root, args.raw_root)
    output_root = resolve_path(project_root, args.output_root)

    ensure_dir(output_root)
    ensure_dir(output_root / "standardized_catalogs")

    inventory_rows = []
    contexts: List[CatalogContext] = []

    for key, filename in SUPPORTED_FILES.items():
        p = raw_root / filename
        algorithm, sky_region = detect_algorithm_and_region(filename)
        inventory_rows.append({
            "catalog_key": key,
            "filename": filename,
            "exists": p.exists(),
            "algorithm": algorithm,
            "sky_region": sky_region,
            "path": str(p),
            "size_bytes": p.stat().st_size if p.exists() else np.nan,
        })
        if p.exists():
            contexts.append(CatalogContext(key=key, path=p, algorithm=algorithm, sky_region=sky_region))

    sha_path = raw_root / SHA_FILE
    inventory_rows.append({
        "catalog_key": "sha256sum",
        "filename": SHA_FILE,
        "exists": sha_path.exists(),
        "algorithm": "",
        "sky_region": "",
        "path": str(sha_path),
        "size_bytes": sha_path.stat().st_size if sha_path.exists() else np.nan,
    })

    inventory_df = pd.DataFrame(inventory_rows)
    inventory_df.to_csv(output_root / "desivast_file_inventory.csv", index=False, encoding="utf-8-sig")

    if args.verify_sha256 and sha_path.exists():
        verify_sha256(raw_root, output_root)

    standardized_dfs = []
    summary_rows = []
    column_map_rows = []

    for ctx in contexts:
        raw_df = read_fits_table(ctx.path)
        std_df, matched = standardize_catalog(raw_df, ctx)
        standardized_dfs.append(std_df)
        summary_rows.append(summarize_catalog(raw_df, std_df, ctx, matched))

        for canonical, matched_col in matched.items():
            column_map_rows.append({
                "catalog_key": ctx.key,
                "filename": ctx.path.name,
                "algorithm": ctx.algorithm,
                "sky_region": ctx.sky_region,
                "canonical_field": canonical,
                "matched_source_column": matched_col,
            })

        std_path = output_root / "standardized_catalogs" / f"{ctx.key}_standardized.csv"
        if args.overwrite or not std_path.exists():
            std_df.to_csv(std_path, index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "desivast_catalog_summary.csv", index=False, encoding="utf-8-sig")

    column_map_df = pd.DataFrame(column_map_rows)
    column_map_df.to_csv(output_root / "desivast_column_mapping.csv", index=False, encoding="utf-8-sig")

    combined = pd.concat(standardized_dfs, ignore_index=True) if standardized_dfs else pd.DataFrame()
    representatives = pd.DataFrame()
    stacked_pool = pd.DataFrame()

    if not combined.empty:
        combined = add_selection_scores(combined)
        combined.to_csv(output_root / "desivast_all_candidates_scored.csv", index=False, encoding="utf-8-sig")

        representatives = choose_representatives(
            combined,
            args.representative_count,
            args.dedupe_ra_dec_threshold_deg,
            args.dedupe_radius_rel_tol,
        )
        representatives.to_csv(output_root / "desivast_representative_voids_selected.csv", index=False, encoding="utf-8-sig")

        stacked_pool = choose_stacked_pool(combined, args.stacked_pool_count)
        stacked_pool.to_csv(output_root / "desivast_stacked_pool_candidates.csv", index=False, encoding="utf-8-sig")

    summary_lines = []
    summary_lines.append("DESIVAST analysis and selection pipeline summary")
    summary_lines.append("================================================")
    summary_lines.append(f"Project root: {project_root}")
    summary_lines.append(f"Raw root: {raw_root}")
    summary_lines.append(f"Output root: {output_root}")
    summary_lines.append("")
    summary_lines.append("Detected input catalogs")
    summary_lines.append("-----------------------")
    for _, row in inventory_df.iterrows():
        if row["catalog_key"] == "sha256sum":
            continue
        summary_lines.append(f"- {row['filename']} :: exists={row['exists']} :: algorithm={row['algorithm']} :: sky_region={row['sky_region']}")
    summary_lines.append("")

    if not summary_df.empty:
        summary_lines.append("Catalog summary")
        summary_lines.append("---------------")
        for _, row in summary_df.iterrows():
            summary_lines.append(
                f"- {row['catalog_key']}: rows={row['rows']}, basic_geometry_rows={row['basic_geometry_rows']}, "
                f"distance_proxy_rows={row['distance_proxy_rows']}, radius_eff_med={row['radius_eff_med']:.6g}, "
                f"distance_proxy_range=({row['distance_proxy_min']:.6g}, {row['distance_proxy_max']:.6g})"
            )
        summary_lines.append("")

    if not representatives.empty:
        summary_lines.append("Recommended representative cases")
        summary_lines.append("-------------------------------")
        for _, row in representatives.iterrows():
            summary_lines.append(
                f"- rank={int(row['representative_rank'])} :: {row['source_catalog_key']} :: "
                f"void_id={row['void_id']} :: algorithm={row['algorithm']} :: sky_region={row['sky_region']} :: "
                f"radius_band={row['radius_band']} :: radius_eff={row['radius_eff']:.6g} :: "
                f"distance_proxy={row['distance_proxy']:.6g} :: distance_kind={row['distance_kind']} :: "
                f"selection_score={row['selection_score']:.4f}"
            )
        summary_lines.append("")

    if not stacked_pool.empty:
        summary_lines.append("Stacked pool")
        summary_lines.append("------------")
        summary_lines.append(f"- selected rows: {len(stacked_pool)}")
        summary_lines.append(f"- radius median: {pd.to_numeric(stacked_pool['radius_eff'], errors='coerce').median():.6g}")
        summary_lines.append(f"- distance_proxy median: {pd.to_numeric(stacked_pool['distance_proxy'], errors='coerce').median():.6g}")
        summary_lines.append("")

    summary_lines.append("Generated files")
    summary_lines.append("---------------")
    generated = [
        "desivast_file_inventory.csv",
        "desivast_catalog_summary.csv",
        "desivast_column_mapping.csv",
        "desivast_all_candidates_scored.csv",
        "desivast_representative_voids_selected.csv",
        "desivast_stacked_pool_candidates.csv",
        "standardized_catalogs/*.csv",
    ]
    if args.verify_sha256 and sha_path.exists():
        generated.insert(1, "desivast_sha256_verification.csv")
    for g in generated:
        summary_lines.append(f"- {g}")
    summary_lines.append("")
    summary_lines.append("Interpretation note")
    summary_lines.append("-------------------")
    summary_lines.append(
        "004 treats explicit redshift and coordinate-like Z separately. "
        "Bare Z is no longer interpreted as physical redshift. "
        "Representative selection also applies cross-algorithm duplicate filtering."
    )

    write_text(output_root / "desivast_pipeline_summary.txt", "\n".join(summary_lines))
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
