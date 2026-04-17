#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
except Exception as exc:
    raise SystemExit(
        "astropy is required for this pipeline. Please install it with:\n"
        "pip install astropy"
    ) from exc


BASE_DERIVED_DEFAULT = Path("data") / "derived" / "Cosmic Void Structural Validation" / "derived"
INPUT_ROOT_DEFAULT = Path("data") / "derived" / "Cosmic Void Structural Validation" / "input"
STELLAR_ROOT_DEFAULT = Path("data") / "raw" / "Cosmic Void Structural Validation" / "DESI_Stellar_Mass_Emission"
STELLAR_FITS_NAME = "dr1_galaxy_stellarmass_lineinfo_v1.0.fits"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument(
        "--analysis-root",
        default=str(BASE_DERIVED_DEFAULT),
        help=(
            "Base derived folder. If a direct run folder is not given, the script will auto-detect "
            "the latest folder under analysis-root/derived/."
        ),
    )
    parser.add_argument(
        "--input-root",
        default=str(INPUT_ROOT_DEFAULT),
        help="Output input root. Final CSVs will be written directly here.",
    )
    parser.add_argument(
        "--stellar-root",
        default=str(STELLAR_ROOT_DEFAULT),
        help="Folder containing the DESI stellar mass FITS file.",
    )
    parser.add_argument(
        "--stellar-fits-name",
        default=STELLAR_FITS_NAME,
        help="DESI stellar mass FITS filename.",
    )
    parser.add_argument(
        "--prefer-core-algorithms",
        default="VIDE,REVOLVER",
        help="Comma-separated algorithms preferred for core inputs.",
    )
    parser.add_argument("--core-representative-count", type=int, default=4)
    parser.add_argument("--support-representative-count", type=int, default=2)
    parser.add_argument("--stacked-core-size", type=int, default=24)
    parser.add_argument("--stacked-support-size", type=int, default=24)
    parser.add_argument(
        "--match-radius-deg",
        type=float,
        default=0.2,
        help="Angular aperture in degrees for attaching stellar-mass summaries to each void row. Default tightened from 1.0 to 0.2.",
    )
    parser.add_argument(
        "--max-nearest-sep-deg",
        type=float,
        default=0.03,
        help="Require nearest stellar match to be within this separation to count as eligible.",
    )
    parser.add_argument(
        "--min-match-count",
        type=int,
        default=10,
        help="Require at least this many matched stellar rows inside the aperture to count as eligible.",
    )
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument("--candidate-pool-size", type=int, default=300)
    parser.add_argument("--run-tag", default="")
    return parser.parse_args()


def resolve_path(project_root: Path, raw_value: str) -> Path:
    p = Path(raw_value)
    return (project_root / p).resolve() if not p.is_absolute() else p.resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_latest_run_folder(base_analysis_root: Path) -> Path:
    direct_files = [
        "desivast_representative_voids_selected.csv",
        "desivast_stacked_pool_candidates.csv",
        "desivast_all_candidates_scored.csv",
    ]
    if all((base_analysis_root / f).exists() for f in direct_files):
        return base_analysis_root

    nested_root = base_analysis_root / "derived"
    if not nested_root.exists():
        raise FileNotFoundError(
            f"Could not find a direct run folder or nested derived folder under: {base_analysis_root}"
        )

    candidates = [p for p in nested_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No timestamped run folders found under: {nested_root}")

    candidates = sorted(candidates, key=lambda p: p.name, reverse=True)
    for candidate in candidates:
        if all((candidate / f).exists() for f in direct_files):
            return candidate

    raise FileNotFoundError(
        f"No valid run folder found under: {nested_root}. "
        "Expected representative, stacked, and scored CSVs."
    )


def read_required_csvs(run_root: Path) -> Dict[str, pd.DataFrame]:
    files = {
        "representatives": run_root / "desivast_representative_voids_selected.csv",
        "stacked": run_root / "desivast_stacked_pool_candidates.csv",
        "scored": run_root / "desivast_all_candidates_scored.csv",
        "summary": run_root / "desivast_catalog_summary.csv",
        "mapping": run_root / "desivast_column_mapping.csv",
    }
    dfs = {}
    for key, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Required CSV not found: {path}")
        dfs[key] = pd.read_csv(path)
    return dfs


def read_standardized_catalogs(run_root: Path) -> Dict[str, pd.DataFrame]:
    std_dir = run_root / "standardized_catalogs"
    if not std_dir.exists():
        raise FileNotFoundError(f"standardized_catalogs folder not found: {std_dir}")

    out: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(std_dir.glob("*_standardized.csv")):
        key = csv_path.stem.replace("_standardized", "")
        out[key] = pd.read_csv(csv_path)
    if not out:
        raise FileNotFoundError(f"No standardized catalog CSVs found in: {std_dir}")
    return out


def clean_algorithm_list(raw_text: str) -> List[str]:
    return [x.strip() for x in raw_text.split(",") if x.strip()]


def add_input_metadata(df: pd.DataFrame, input_role: str, selection_basis: str) -> pd.DataFrame:
    out = df.copy()
    out["input_role"] = input_role
    out["selection_basis"] = selection_basis
    return out


def find_table_hdu(hdul) -> object:
    for hdu in hdul:
        if hasattr(hdu, "columns") and getattr(hdu.columns, "names", None):
            return hdu
    raise ValueError("No table HDU with named columns found in FITS file.")


def get_stellar_columns(stellar_path: Path) -> Tuple[Dict[str, str], int]:
    with fits.open(stellar_path, memmap=True) as hdul:
        table_hdu = find_table_hdu(hdul)
        cols = list(table_hdu.columns.names)

        def pick(candidates: List[str]) -> Optional[str]:
            cols_lower = {c.lower(): c for c in cols}
            for cand in candidates:
                if cand in cols:
                    return cand
                lc = cand.lower()
                if lc in cols_lower:
                    return cols_lower[lc]
            return None

        ra_col = pick(["TARGET_RA", "RA", "ra"])
        dec_col = pick(["TARGET_DEC", "DEC", "dec"])
        direct_mass_col = pick(["MASS", "LOGMASS", "LOG_MASS", "TOTAL_MASS", "MSTAR_TOTAL"])
        proxy_mass_col = pick(["MASS_CG", "MSTAR", "STELLAR_MASS", "LOGMSTAR", "LOG_MSTAR"])
        redshift_col = pick(["REDSHIFT", "redshift", "Z", "z"])

        if ra_col is None or dec_col is None:
            raise ValueError("Could not identify RA/DEC columns in stellar mass FITS.")

        meta = {
            "ra_col": ra_col or "",
            "dec_col": dec_col or "",
            "redshift_col": redshift_col or "",
            "direct_mass_col": direct_mass_col or "",
            "proxy_mass_col": proxy_mass_col or "",
        }
        nrows = len(table_hdu.data)
        return meta, nrows


def read_column_chunk(table_data, colname: Optional[str], start: int, end: int) -> np.ndarray:
    if not colname:
        return np.full(end - start, np.nan, dtype=np.float64)
    arr = table_data[colname][start:end]
    if getattr(arr, "ndim", 1) > 1:
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            return np.full(end - start, np.nan, dtype=np.float64)
    try:
        return np.asarray(arr, dtype=np.float64)
    except Exception:
        s = pd.to_numeric(pd.Series(arr), errors="coerce")
        return s.to_numpy(dtype=np.float64)


def angular_sep_deg_vec(ra1_deg: float, dec1_deg: float, ra2_deg: np.ndarray, dec2_deg: np.ndarray) -> np.ndarray:
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)

    sin_d1 = np.sin(dec1)
    cos_d1 = np.cos(dec1)
    sin_d2 = np.sin(dec2)
    cos_d2 = np.cos(dec2)

    cosang = sin_d1 * sin_d2 + cos_d1 * cos_d2 * np.cos(ra1 - ra2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))


def delta_ra_wrap(ra_chunk: np.ndarray, ra0: float) -> np.ndarray:
    return np.abs(((ra_chunk - ra0 + 180.0) % 360.0) - 180.0)


def prepare_candidate_pool(scored: pd.DataFrame, preferred_algorithms: List[str], candidate_pool_size: int) -> pd.DataFrame:
    df = scored.copy()
    needed = ["void_id", "algorithm", "sky_region", "ra", "dec", "radius_eff", "selection_score"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Required column missing in scored CSV: {col}")

    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df["radius_eff"] = pd.to_numeric(df["radius_eff"], errors="coerce")
    df["selection_score"] = pd.to_numeric(df["selection_score"], errors="coerce")

    if "has_basic_geometry" in df.columns:
        df = df[df["has_basic_geometry"].fillna(False)].copy()
    else:
        df = df[df["ra"].notna() & df["dec"].notna() & df["radius_eff"].notna()].copy()

    df["_pref_alg"] = df["algorithm"].isin(preferred_algorithms).astype(int)
    df = df.sort_values(["_pref_alg", "selection_score", "radius_eff"], ascending=[False, False, False]).copy()
    if candidate_pool_size > 0:
        df = df.head(candidate_pool_size).copy()
    df = df.drop(columns=["_pref_alg"])
    return df.reset_index(drop=True)


def init_match_accumulators(n: int, match_radius_deg: float) -> Dict[str, np.ndarray]:
    return {
        "mass_match_radius_deg": np.full(n, match_radius_deg, dtype=np.float64),
        "mass_match_count": np.zeros(n, dtype=np.int64),
        "mass_direct_count": np.zeros(n, dtype=np.int64),
        "mass_proxy_count": np.zeros(n, dtype=np.int64),
        "mass_direct_sum": np.zeros(n, dtype=np.float64),
        "mass_proxy_sum": np.zeros(n, dtype=np.float64),
        "mass_preferred_sum": np.zeros(n, dtype=np.float64),
        "mass_preferred_max": np.full(n, np.nan, dtype=np.float64),
        "mass_nearest_sep_deg": np.full(n, np.inf, dtype=np.float64),
        "mass_nearest_value": np.full(n, np.nan, dtype=np.float64),
        "mass_nearest_type": np.array(["missing"] * n, dtype=object),
    }


def update_accumulators_for_candidate(
    idx: int,
    ra0: float,
    dec0: float,
    radius_deg: float,
    chunk_ra: np.ndarray,
    chunk_dec: np.ndarray,
    direct_mass: np.ndarray,
    proxy_mass: np.ndarray,
    preferred_mass: np.ndarray,
    preferred_type: np.ndarray,
    acc: Dict[str, np.ndarray],
) -> None:
    dra = delta_ra_wrap(chunk_ra, ra0)
    ddec = np.abs(chunk_dec - dec0)
    box = (dra <= radius_deg) & (ddec <= radius_deg)
    if not np.any(box):
        return

    local_ra = chunk_ra[box]
    local_dec = chunk_dec[box]
    local_direct = direct_mass[box]
    local_proxy = proxy_mass[box]
    local_pref = preferred_mass[box]
    local_type = preferred_type[box]

    sep = angular_sep_deg_vec(ra0, dec0, local_ra, local_dec)
    keep = sep <= radius_deg
    if not np.any(keep):
        return

    local_sep = sep[keep]
    local_direct = local_direct[keep]
    local_proxy = local_proxy[keep]
    local_pref = local_pref[keep]
    local_type = local_type[keep]

    acc["mass_match_count"][idx] += len(local_sep)

    direct_mask = np.isfinite(local_direct)
    proxy_mask = np.isfinite(local_proxy)
    pref_mask = np.isfinite(local_pref)

    acc["mass_direct_count"][idx] += int(np.sum(direct_mask))
    acc["mass_proxy_count"][idx] += int(np.sum(proxy_mask))
    if np.any(direct_mask):
        acc["mass_direct_sum"][idx] += float(np.nansum(local_direct[direct_mask]))
    if np.any(proxy_mask):
        acc["mass_proxy_sum"][idx] += float(np.nansum(local_proxy[proxy_mask]))
    if np.any(pref_mask):
        acc["mass_preferred_sum"][idx] += float(np.nansum(local_pref[pref_mask]))
        local_max = float(np.nanmax(local_pref[pref_mask]))
        prev = acc["mass_preferred_max"][idx]
        if np.isnan(prev) or local_max > prev:
            acc["mass_preferred_max"][idx] = local_max

    nearest_local = int(np.argmin(local_sep))
    nearest_sep = float(local_sep[nearest_local])
    if nearest_sep < acc["mass_nearest_sep_deg"][idx]:
        acc["mass_nearest_sep_deg"][idx] = nearest_sep
        val = local_pref[nearest_local]
        acc["mass_nearest_value"][idx] = float(val) if np.isfinite(val) else np.nan
        acc["mass_nearest_type"][idx] = str(local_type[nearest_local])


def build_preferred_mass_arrays(direct_mass: np.ndarray, proxy_mass: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pref = np.where(np.isfinite(direct_mass), direct_mass, proxy_mass)
    pref_type = np.where(
        np.isfinite(direct_mass), "direct_mass",
        np.where(np.isfinite(proxy_mass), "mass_proxy", "missing")
    )
    return pref, pref_type


def scan_stellar_matches(
    stellar_path: Path,
    stellar_meta: Dict[str, str],
    candidates: pd.DataFrame,
    match_radius_deg: float,
    chunk_size: int,
) -> pd.DataFrame:
    n_candidates = len(candidates)
    acc = init_match_accumulators(n_candidates, match_radius_deg)

    cand_ra = pd.to_numeric(candidates["ra"], errors="coerce").to_numpy(dtype=np.float64)
    cand_dec = pd.to_numeric(candidates["dec"], errors="coerce").to_numpy(dtype=np.float64)

    with fits.open(stellar_path, memmap=True) as hdul:
        table_hdu = find_table_hdu(hdul)
        data = table_hdu.data
        total_rows = len(data)

        for start in range(0, total_rows, chunk_size):
            end = min(start + chunk_size, total_rows)

            chunk_ra = read_column_chunk(data, stellar_meta["ra_col"], start, end)
            chunk_dec = read_column_chunk(data, stellar_meta["dec_col"], start, end)
            direct_mass = read_column_chunk(data, stellar_meta["direct_mass_col"], start, end)
            proxy_mass = read_column_chunk(data, stellar_meta["proxy_mass_col"], start, end)
            preferred_mass, preferred_type = build_preferred_mass_arrays(direct_mass, proxy_mass)

            valid_pos = np.isfinite(chunk_ra) & np.isfinite(chunk_dec)
            if not np.any(valid_pos):
                continue

            chunk_ra = chunk_ra[valid_pos]
            chunk_dec = chunk_dec[valid_pos]
            direct_mass = direct_mass[valid_pos]
            proxy_mass = proxy_mass[valid_pos]
            preferred_mass = preferred_mass[valid_pos]
            preferred_type = preferred_type[valid_pos]

            for i in range(n_candidates):
                if not np.isfinite(cand_ra[i]) or not np.isfinite(cand_dec[i]):
                    continue
                update_accumulators_for_candidate(
                    i,
                    float(cand_ra[i]),
                    float(cand_dec[i]),
                    match_radius_deg,
                    chunk_ra,
                    chunk_dec,
                    direct_mass,
                    proxy_mass,
                    preferred_mass,
                    preferred_type,
                    acc,
                )

    out = candidates.copy()
    out["mass_match_radius_deg"] = acc["mass_match_radius_deg"]
    out["mass_match_count"] = acc["mass_match_count"]
    out["mass_direct_count"] = acc["mass_direct_count"]
    out["mass_proxy_count"] = acc["mass_proxy_count"]

    out["mass_direct_sum"] = acc["mass_direct_sum"]
    out["mass_proxy_sum"] = acc["mass_proxy_sum"]
    out["mass_preferred_sum"] = acc["mass_preferred_sum"]
    out["mass_preferred_max"] = acc["mass_preferred_max"]

    out["mass_direct_mean"] = np.where(out["mass_direct_count"] > 0, out["mass_direct_sum"] / out["mass_direct_count"], np.nan)
    out["mass_proxy_mean"] = np.where(out["mass_proxy_count"] > 0, out["mass_proxy_sum"] / out["mass_proxy_count"], np.nan)
    out["mass_preferred_mean"] = np.where(out["mass_match_count"] > 0, out["mass_preferred_sum"] / out["mass_match_count"], np.nan)
    out["mass_preferred_median"] = np.nan

    out["mass_nearest_sep_deg"] = np.where(np.isfinite(acc["mass_nearest_sep_deg"]), acc["mass_nearest_sep_deg"], np.nan)
    out["mass_nearest_value"] = acc["mass_nearest_value"]
    out["mass_nearest_type"] = acc["mass_nearest_type"]

    out["mass_match_status"] = np.where(out["mass_match_count"] > 0, "matched", "no_match")
    out["mass_match_best_type"] = np.where(
        out["mass_direct_count"] > 0, "direct_mass",
        np.where(out["mass_proxy_count"] > 0, "mass_proxy", "missing")
    )
    out["mass_best_value"] = np.where(
        out["mass_direct_count"] > 0, out["mass_direct_mean"],
        np.where(out["mass_proxy_count"] > 0, out["mass_proxy_mean"], np.nan)
    )

    return out


def apply_strict_mass_filters(
    audited: pd.DataFrame,
    min_match_count: int,
    max_nearest_sep_deg: float,
) -> pd.DataFrame:
    df = audited.copy()
    df["mass_strict_eligible"] = (
        df["mass_match_status"].eq("matched")
        & df["mass_match_best_type"].isin(["direct_mass", "mass_proxy"])
        & pd.to_numeric(df["mass_match_count"], errors="coerce").fillna(0).ge(min_match_count)
        & pd.to_numeric(df["mass_nearest_sep_deg"], errors="coerce").fillna(np.inf).le(max_nearest_sep_deg)
    )
    return df


def assign_radius_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    r = pd.to_numeric(out["radius_eff"], errors="coerce")
    if r.notna().sum() > 0:
        q1, q2 = r.quantile([0.33, 0.66]).tolist()
        conditions = [r <= q1, (r > q1) & (r <= q2), r > q2]
        choices = ["small", "medium", "large"]
        out["radius_band"] = np.select(conditions, choices, default="unknown")
    else:
        out["radius_band"] = "unknown"
    return out


def diversify_select(df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    df = assign_radius_band(df.copy())
    bands = ["small", "medium", "large"]
    base = target_count // 3
    rem = target_count % 3
    desired = {b: base for b in bands}
    for b in bands[:rem]:
        desired[b] += 1

    pool = df.sort_values(
        ["selection_score", "mass_match_count", "mass_nearest_sep_deg", "radius_eff"],
        ascending=[False, False, True, False]
    ).copy()
    chosen = []

    for band in bands:
        sub = pool[pool["radius_band"] == band]
        want = desired[band]
        for _, row in sub.iterrows():
            chosen.append(row)
            if len([x for x in chosen if x["radius_band"] == band]) >= want:
                break

    if len(chosen) < target_count:
        used = {(str(r["source_catalog_key"]), str(r["void_id"])) for r in chosen}
        for _, row in pool.iterrows():
            key = (str(row["source_catalog_key"]), str(row["void_id"]))
            if key in used:
                continue
            chosen.append(row)
            used.add(key)
            if len(chosen) >= target_count:
                break

    return pd.DataFrame(chosen).reset_index(drop=True)


def build_core_representatives(candidates: pd.DataFrame, preferred_algorithms: List[str], count: int) -> pd.DataFrame:
    core = candidates[
        candidates["algorithm"].isin(preferred_algorithms)
        & candidates["mass_strict_eligible"].fillna(False)
    ].copy()
    core = diversify_select(core, count)
    core = add_input_metadata(core, "core_representative", "mass_augmented_candidate_selection_strict")
    core["representative_rank"] = np.arange(1, len(core) + 1)
    return core


def build_support_representatives(candidates: pd.DataFrame, preferred_algorithms: List[str], count: int) -> pd.DataFrame:
    support = candidates[
        (~candidates["algorithm"].isin(preferred_algorithms))
        & candidates["mass_strict_eligible"].fillna(False)
    ].copy()
    support = diversify_select(support, count)
    support = add_input_metadata(support, "support_representative", "mass_augmented_candidate_selection_strict")
    support["representative_rank"] = np.arange(1, len(support) + 1)
    return support


def build_core_stacked(candidates: pd.DataFrame, preferred_algorithms: List[str], size: int) -> pd.DataFrame:
    core = candidates[
        candidates["algorithm"].isin(preferred_algorithms)
        & candidates["mass_strict_eligible"].fillna(False)
    ].copy()
    core = core.sort_values(
        ["selection_score", "mass_match_count", "mass_nearest_sep_deg", "radius_eff"],
        ascending=[False, False, True, False]
    ).head(size).copy()
    core = add_input_metadata(core, "core_stacked", "mass_augmented_candidate_selection_strict")
    core["stacked_pool_rank"] = np.arange(1, len(core) + 1)
    return core


def build_support_stacked(candidates: pd.DataFrame, preferred_algorithms: List[str], size: int) -> pd.DataFrame:
    support = candidates[
        (~candidates["algorithm"].isin(preferred_algorithms))
        & candidates["mass_strict_eligible"].fillna(False)
    ].copy()
    support = support.sort_values(
        ["selection_score", "mass_match_count", "mass_nearest_sep_deg", "radius_eff"],
        ascending=[False, False, True, False]
    ).head(size).copy()
    support = add_input_metadata(support, "support_stacked", "mass_augmented_candidate_selection_strict")
    support["stacked_pool_rank"] = np.arange(1, len(support) + 1)
    return support


def build_master_input(core_rep: pd.DataFrame, support_rep: pd.DataFrame, core_stacked: pd.DataFrame, support_stacked: pd.DataFrame) -> pd.DataFrame:
    frames = [f for f in [core_rep, support_rep, core_stacked, support_stacked] if not f.empty]
    if not frames:
        return pd.DataFrame()
    master = pd.concat(frames, ignore_index=True)
    role_priority = {
        "core_representative": 0,
        "support_representative": 1,
        "core_stacked": 2,
        "support_stacked": 3,
    }
    master["_role_priority"] = master["input_role"].map(role_priority).fillna(9)
    master = (
        master.sort_values(["_role_priority", "selection_score", "radius_eff"], ascending=[True, False, False])
        .drop(columns=["_role_priority"])
        .reset_index(drop=True)
    )
    return master


def build_source_manifest(std_catalogs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for catalog_key, df in std_catalogs.items():
        algo = str(df["algorithm"].iloc[0]) if "algorithm" in df.columns and len(df) else ""
        sky = str(df["sky_region"].iloc[0]) if "sky_region" in df.columns and len(df) else ""
        rows.append({
            "source_catalog_key": catalog_key,
            "algorithm": algo,
            "sky_region": sky,
            "rows": len(df),
            "usable_distance_rows": int(df["distance_status"].isin(["usable"]).sum()) if "distance_status" in df.columns else np.nan,
            "proxy_distance_rows": int(df["distance_status"].isin(["proxy_only"]).sum()) if "distance_status" in df.columns else np.nan,
            "missing_or_invalid_rows": int(df["distance_status"].isin(["missing", "missing_or_invalid", "invalid_redshift_like"]).sum()) if "distance_status" in df.columns else np.nan,
        })
    return pd.DataFrame(rows)


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    desired_cols = [
        "void_id", "source_catalog_key", "source_filename", "algorithm", "sky_region",
        "ra", "dec", "radius_eff", "radius_eff_unc", "edge_area", "tot_area",
        "edge_fraction", "n_members", "redshift", "coord_x", "coord_y", "coord_z",
        "coord_r3d", "distance_proxy", "distance_kind", "distance_status",
        "has_basic_geometry", "has_distance_proxy", "radius_rank_pct", "distance_rank_pct",
        "score_radius", "score_edge", "score_geometry", "score_distance",
        "score_distance_status", "score_members", "selection_score", "radius_band",
        "representative_rank", "stacked_pool_rank", "input_role", "selection_basis",
        "mass_match_radius_deg", "mass_match_count", "mass_direct_count", "mass_proxy_count",
        "mass_match_status", "mass_match_best_type", "mass_best_value", "mass_direct_sum",
        "mass_direct_mean", "mass_proxy_sum", "mass_proxy_mean", "mass_preferred_sum",
        "mass_preferred_mean", "mass_preferred_median", "mass_preferred_max",
        "mass_nearest_sep_deg", "mass_nearest_value", "mass_nearest_type",
        "mass_strict_eligible",
    ]
    for col in desired_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[desired_cols]


def write_summary(
    path: Path,
    run_root: Path,
    input_root: Path,
    stellar_path: Path,
    stellar_meta: Dict[str, str],
    preferred_algorithms: List[str],
    candidate_pool_size: int,
    audited_candidates: int,
    matched_candidates: int,
    strict_candidates: int,
    core_rep: pd.DataFrame,
    support_rep: pd.DataFrame,
    core_stacked: pd.DataFrame,
    support_stacked: pd.DataFrame,
    master: pd.DataFrame,
    match_radius_deg: float,
    max_nearest_sep_deg: float,
    min_match_count: int,
    chunk_size: int,
    run_tag: str,
) -> None:
    lines = []
    lines.append("Cosmic Void final input builder summary (004 stricter mass-eligible reselection)")
    lines.append("==========================================================================")
    lines.append(f"Source run root: {run_root}")
    lines.append(f"Input root: {input_root}")
    lines.append(f"Stellar FITS: {stellar_path}")
    lines.append(f"Preferred core algorithms: {', '.join(preferred_algorithms)}")
    lines.append(f"Run tag: {run_tag}")
    lines.append(f"Candidate pool size requested: {candidate_pool_size}")
    lines.append(f"Audited candidates: {audited_candidates}")
    lines.append(f"Broad matched candidates: {matched_candidates}")
    lines.append(f"Strict eligible candidates: {strict_candidates}")
    lines.append(f"Match radius (deg): {match_radius_deg}")
    lines.append(f"Max nearest separation (deg): {max_nearest_sep_deg}")
    lines.append(f"Minimum match count: {min_match_count}")
    lines.append(f"Chunk size: {chunk_size}")
    lines.append("")
    lines.append("Stellar mass field mapping")
    lines.append("--------------------------")
    for k, v in stellar_meta.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Generated input files")
    lines.append("---------------------")
    lines.append("- cosmic_void_core_representatives_input.csv")
    lines.append("- cosmic_void_support_representatives_input.csv")
    lines.append("- cosmic_void_core_stacked_input.csv")
    lines.append("- cosmic_void_support_stacked_input.csv")
    lines.append("- cosmic_void_master_input.csv")
    lines.append("- cosmic_void_input_source_manifest.csv")
    lines.append("- cosmic_void_input_build_summary.txt")
    lines.append("- cosmic_void_mass_audited_candidates.csv")
    lines.append("")
    lines.append("Row counts")
    lines.append("----------")
    lines.append(f"- core representatives: {len(core_rep)}")
    lines.append(f"- support representatives: {len(support_rep)}")
    lines.append(f"- core stacked: {len(core_stacked)}")
    lines.append(f"- support stacked: {len(support_stacked)}")
    lines.append(f"- master input total: {len(master)}")
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append(
        "This 004 stage keeps the streaming audit from 003 but applies stricter eligibility rules. "
        "A void is retained only if it is broadly matched, has enough matched stellar rows, and has a sufficiently small nearest separation."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    base_analysis_root = resolve_path(project_root, args.analysis_root)
    input_root = resolve_path(project_root, args.input_root)
    stellar_root = resolve_path(project_root, args.stellar_root)
    stellar_path = stellar_root / args.stellar_fits_name

    if not stellar_path.exists():
        raise FileNotFoundError(f"Stellar mass FITS not found: {stellar_path}")

    run_root = detect_latest_run_folder(base_analysis_root)
    dfs = read_required_csvs(run_root)
    std_catalogs = read_standardized_catalogs(run_root)

    preferred_algorithms = clean_algorithm_list(args.prefer_core_algorithms)
    ensure_dir(input_root)

    scored = dfs["scored"]
    candidate_pool = prepare_candidate_pool(scored, preferred_algorithms, args.candidate_pool_size)

    stellar_meta, _ = get_stellar_columns(stellar_path)
    audited = scan_stellar_matches(
        stellar_path=stellar_path,
        stellar_meta=stellar_meta,
        candidates=candidate_pool,
        match_radius_deg=args.match_radius_deg,
        chunk_size=args.chunk_size,
    )
    audited = apply_strict_mass_filters(
        audited,
        min_match_count=args.min_match_count,
        max_nearest_sep_deg=args.max_nearest_sep_deg,
    )

    broad_matched = audited[
        audited["mass_match_status"].eq("matched")
        & audited["mass_match_best_type"].isin(["direct_mass", "mass_proxy"])
    ].copy()

    strict_matched = audited[audited["mass_strict_eligible"].fillna(False)].copy()

    core_rep = harmonize_columns(build_core_representatives(strict_matched, preferred_algorithms, args.core_representative_count))
    support_rep = harmonize_columns(build_support_representatives(strict_matched, preferred_algorithms, args.support_representative_count))
    core_stacked = harmonize_columns(build_core_stacked(strict_matched, preferred_algorithms, args.stacked_core_size))
    support_stacked = harmonize_columns(build_support_stacked(strict_matched, preferred_algorithms, args.stacked_support_size))
    master = harmonize_columns(build_master_input(core_rep, support_rep, core_stacked, support_stacked))
    manifest = build_source_manifest(std_catalogs)

    core_rep.to_csv(input_root / "cosmic_void_core_representatives_input.csv", index=False, encoding="utf-8-sig")
    support_rep.to_csv(input_root / "cosmic_void_support_representatives_input.csv", index=False, encoding="utf-8-sig")
    core_stacked.to_csv(input_root / "cosmic_void_core_stacked_input.csv", index=False, encoding="utf-8-sig")
    support_stacked.to_csv(input_root / "cosmic_void_support_stacked_input.csv", index=False, encoding="utf-8-sig")
    master.to_csv(input_root / "cosmic_void_master_input.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(input_root / "cosmic_void_input_source_manifest.csv", index=False, encoding="utf-8-sig")
    audited.to_csv(input_root / "cosmic_void_mass_audited_candidates.csv", index=False, encoding="utf-8-sig")

    write_summary(
        input_root / "cosmic_void_input_build_summary.txt",
        run_root=run_root,
        input_root=input_root,
        stellar_path=stellar_path,
        stellar_meta=stellar_meta,
        preferred_algorithms=preferred_algorithms,
        candidate_pool_size=args.candidate_pool_size,
        audited_candidates=len(audited),
        matched_candidates=len(broad_matched),
        strict_candidates=len(strict_matched),
        core_rep=core_rep,
        support_rep=support_rep,
        core_stacked=core_stacked,
        support_stacked=support_stacked,
        master=master,
        match_radius_deg=args.match_radius_deg,
        max_nearest_sep_deg=args.max_nearest_sep_deg,
        min_match_count=args.min_match_count,
        chunk_size=args.chunk_size,
        run_tag=args.run_tag,
    )

    print(f"Source run root: {run_root}")
    print(f"Input root: {input_root}")
    print(f"Stellar FITS: {stellar_path}")
    print(f"Candidate pool audited: {len(audited)}")
    print(f"Broad matched candidates: {len(broad_matched)}")
    print(f"Strict eligible candidates: {len(strict_matched)}")
    print(f"Core representatives: {len(core_rep)}")
    print(f"Support representatives: {len(support_rep)}")
    print(f"Core stacked: {len(core_stacked)}")
    print(f"Support stacked: {len(support_stacked)}")
    print(f"Master input total: {len(master)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
