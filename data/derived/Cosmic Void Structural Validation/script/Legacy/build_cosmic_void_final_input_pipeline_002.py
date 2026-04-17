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
    parser.add_argument(
        "--stacked-core-size",
        type=int,
        default=24,
        help="Number of rows to keep for the core stacked input.",
    )
    parser.add_argument(
        "--stacked-support-size",
        type=int,
        default=24,
        help="Number of rows to keep for the support stacked input.",
    )
    parser.add_argument(
        "--match-radius-deg",
        type=float,
        default=1.0,
        help="Angular aperture in degrees for attaching stellar-mass summaries to each void row.",
    )
    parser.add_argument(
        "--max-stellar-rows",
        type=int,
        default=0,
        help="Optional row cap for development/testing. 0 means read all rows.",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional tag recorded in summary files.",
    )
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


def read_fits_table(path: Path, max_rows: int = 0) -> pd.DataFrame:
    with fits.open(path, memmap=True) as hdul:
        table_hdu = None
        for hdu in hdul:
            # Only accept real table HDUs that expose columns
            if getattr(hdu, "data", None) is None:
                continue
            if not hasattr(hdu, "columns") or hdu.columns is None:
                continue
            if not hasattr(hdu.columns, "names") or hdu.columns.names is None:
                continue
            table_hdu = hdu
            break

        if table_hdu is None:
            raise ValueError(f"No table HDU with columns found in {path}")

        data = table_hdu.data
        cols = list(table_hdu.columns.names)
        records = {}
        n = len(data) if max_rows <= 0 else min(len(data), max_rows)

        for col in cols:
            arr = data[col][:n]
            if getattr(arr, "ndim", 1) > 1:
                if arr.shape[1] == 1:
                    arr = arr[:, 0]
                else:
                    continue
            if getattr(arr, "dtype", None) is not None and arr.dtype.kind == "S":
                arr = arr.astype(str)
            records[col] = np.array(arr)

        return pd.DataFrame(records)


def find_first_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in columns:
            return cand
        lc = cand.lower()
        if lc in cols_lower:
            return cols_lower[lc]
    return None


def prepare_stellar_mass_catalog(stellar_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cols = list(stellar_df.columns)

    ra_col = find_first_column(cols, ["TARGET_RA", "RA", "ra"])
    dec_col = find_first_column(cols, ["TARGET_DEC", "DEC", "dec"])
    direct_mass_col = find_first_column(
        cols,
        ["MASS", "LOGMASS", "LOG_MASS", "TOTAL_MASS", "MSTAR_TOTAL"]
    )
    proxy_mass_col = find_first_column(
        cols,
        ["MASS_CG", "MSTAR", "STELLAR_MASS", "LOGMSTAR", "LOG_MSTAR"]
    )
    redshift_col = find_first_column(cols, ["REDSHIFT", "redshift", "Z", "z"])

    if ra_col is None or dec_col is None:
        raise ValueError("Could not identify RA/DEC columns in stellar mass FITS.")

    out = pd.DataFrame()
    out["stellar_ra"] = pd.to_numeric(stellar_df[ra_col], errors="coerce")
    out["stellar_dec"] = pd.to_numeric(stellar_df[dec_col], errors="coerce")
    out["stellar_redshift"] = pd.to_numeric(stellar_df[redshift_col], errors="coerce") if redshift_col else np.nan

    if direct_mass_col:
        out["direct_mass_value"] = pd.to_numeric(stellar_df[direct_mass_col], errors="coerce")
    else:
        out["direct_mass_value"] = np.nan

    if proxy_mass_col:
        out["mass_proxy_value"] = pd.to_numeric(stellar_df[proxy_mass_col], errors="coerce")
    else:
        out["mass_proxy_value"] = np.nan

    out["preferred_mass_value"] = out["direct_mass_value"].where(
        out["direct_mass_value"].notna(),
        out["mass_proxy_value"]
    )
    out["preferred_mass_type"] = np.where(
        out["direct_mass_value"].notna(), "direct_mass",
        np.where(out["mass_proxy_value"].notna(), "mass_proxy", "missing")
    )

    out = out.dropna(subset=["stellar_ra", "stellar_dec"]).reset_index(drop=True)

    meta = {
        "ra_col": ra_col or "",
        "dec_col": dec_col or "",
        "redshift_col": redshift_col or "",
        "direct_mass_col": direct_mass_col or "",
        "proxy_mass_col": proxy_mass_col or "",
    }
    return out, meta


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


def summarize_mass_match(
    row: pd.Series,
    stellar: pd.DataFrame,
    match_radius_deg: float,
) -> Dict[str, object]:
    result = {
        "mass_match_radius_deg": match_radius_deg,
        "mass_match_count": 0,
        "mass_direct_count": 0,
        "mass_proxy_count": 0,
        "mass_match_status": "no_geometry",
        "mass_match_best_type": "missing",
        "mass_best_value": np.nan,
        "mass_direct_sum": np.nan,
        "mass_direct_mean": np.nan,
        "mass_proxy_sum": np.nan,
        "mass_proxy_mean": np.nan,
        "mass_preferred_sum": np.nan,
        "mass_preferred_mean": np.nan,
        "mass_preferred_median": np.nan,
        "mass_preferred_max": np.nan,
        "mass_nearest_sep_deg": np.nan,
        "mass_nearest_value": np.nan,
        "mass_nearest_type": "missing",
    }

    ra = row.get("ra", np.nan)
    dec = row.get("dec", np.nan)
    if pd.isna(ra) or pd.isna(dec):
        return result

    sep = angular_sep_deg_vec(
        float(ra),
        float(dec),
        stellar["stellar_ra"].to_numpy(),
        stellar["stellar_dec"].to_numpy(),
    )
    mask = sep <= match_radius_deg
    local = stellar.loc[mask].copy()

    if local.empty:
        result["mass_match_status"] = "no_match"
        return result

    result["mass_match_status"] = "matched"
    result["mass_match_count"] = int(len(local))
    result["mass_direct_count"] = int(local["direct_mass_value"].notna().sum())
    result["mass_proxy_count"] = int(local["mass_proxy_value"].notna().sum())

    result["mass_direct_sum"] = float(local["direct_mass_value"].sum()) if local["direct_mass_value"].notna().any() else np.nan
    result["mass_direct_mean"] = float(local["direct_mass_value"].mean()) if local["direct_mass_value"].notna().any() else np.nan

    result["mass_proxy_sum"] = float(local["mass_proxy_value"].sum()) if local["mass_proxy_value"].notna().any() else np.nan
    result["mass_proxy_mean"] = float(local["mass_proxy_value"].mean()) if local["mass_proxy_value"].notna().any() else np.nan

    if local["preferred_mass_value"].notna().any():
        result["mass_preferred_sum"] = float(local["preferred_mass_value"].sum())
        result["mass_preferred_mean"] = float(local["preferred_mass_value"].mean())
        result["mass_preferred_median"] = float(local["preferred_mass_value"].median())
        result["mass_preferred_max"] = float(local["preferred_mass_value"].max())

    local_sep = sep[mask]
    nearest_local_pos = int(np.argmin(local_sep))
    nearest_row = local.iloc[nearest_local_pos]
    result["mass_nearest_sep_deg"] = float(local_sep[nearest_local_pos])
    result["mass_nearest_value"] = float(nearest_row["preferred_mass_value"]) if pd.notna(nearest_row["preferred_mass_value"]) else np.nan
    result["mass_nearest_type"] = str(nearest_row["preferred_mass_type"])

    if result["mass_direct_count"] > 0:
        result["mass_match_best_type"] = "direct_mass"
        result["mass_best_value"] = result["mass_direct_mean"]
    elif result["mass_proxy_count"] > 0:
        result["mass_match_best_type"] = "mass_proxy"
        result["mass_best_value"] = result["mass_proxy_mean"]
    else:
        result["mass_match_best_type"] = "missing"
        result["mass_best_value"] = np.nan

    return result


def augment_with_mass_proxy(
    df: pd.DataFrame,
    stellar: pd.DataFrame,
    match_radius_deg: float,
) -> pd.DataFrame:
    out = df.copy()
    match_rows = []
    for _, row in out.iterrows():
        match_rows.append(summarize_mass_match(row, stellar, match_radius_deg))
    match_df = pd.DataFrame(match_rows)
    return pd.concat([out.reset_index(drop=True), match_df.reset_index(drop=True)], axis=1)


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    desired_cols = [
        "void_id",
        "source_catalog_key",
        "source_filename",
        "algorithm",
        "sky_region",
        "ra",
        "dec",
        "radius_eff",
        "radius_eff_unc",
        "edge_area",
        "tot_area",
        "edge_fraction",
        "n_members",
        "redshift",
        "coord_x",
        "coord_y",
        "coord_z",
        "coord_r3d",
        "distance_proxy",
        "distance_kind",
        "distance_status",
        "has_basic_geometry",
        "has_distance_proxy",
        "radius_rank_pct",
        "distance_rank_pct",
        "score_radius",
        "score_edge",
        "score_geometry",
        "score_distance",
        "score_distance_status",
        "score_members",
        "selection_score",
        "radius_band",
        "representative_rank",
        "stacked_pool_rank",
        "input_role",
        "selection_basis",
        "mass_match_radius_deg",
        "mass_match_count",
        "mass_direct_count",
        "mass_proxy_count",
        "mass_match_status",
        "mass_match_best_type",
        "mass_best_value",
        "mass_direct_sum",
        "mass_direct_mean",
        "mass_proxy_sum",
        "mass_proxy_mean",
        "mass_preferred_sum",
        "mass_preferred_mean",
        "mass_preferred_median",
        "mass_preferred_max",
        "mass_nearest_sep_deg",
        "mass_nearest_value",
        "mass_nearest_type",
    ]
    for col in desired_cols:
        if col not in out.columns:
            out[col] = np.nan
    return out[desired_cols]


def build_core_representatives(
    representatives: pd.DataFrame,
    preferred_algorithms: List[str],
) -> pd.DataFrame:
    df = representatives.copy()
    core = df[
        df["algorithm"].isin(preferred_algorithms)
        & df["distance_status"].isin(["usable", "proxy_only"])
    ].copy()
    if core.empty:
        core = df[df["distance_status"].isin(["usable", "proxy_only"])].copy()
    core = add_input_metadata(core, "core_representative", "representative_selection")
    return core


def build_support_representatives(
    representatives: pd.DataFrame,
    preferred_algorithms: List[str],
) -> pd.DataFrame:
    df = representatives.copy()
    support = df[~df["algorithm"].isin(preferred_algorithms)].copy()
    if support.empty:
        support = df[df["distance_status"].isin(["proxy_only", "missing", "invalid_redshift_like"])].copy()
    support = add_input_metadata(support, "support_representative", "representative_selection")
    return support


def build_core_stacked(
    stacked: pd.DataFrame,
    preferred_algorithms: List[str],
    size: int,
) -> pd.DataFrame:
    df = stacked.copy()
    core = df[
        df["algorithm"].isin(preferred_algorithms)
        & df["distance_status"].isin(["usable", "proxy_only"])
    ].copy()
    if core.empty:
        core = df[df["distance_status"].isin(["usable", "proxy_only"])].copy()
    core = core.sort_values(["selection_score", "radius_eff"], ascending=[False, False]).head(size).copy()
    core = add_input_metadata(core, "core_stacked", "stacked_pool_selection")
    return core


def build_support_stacked(
    stacked: pd.DataFrame,
    preferred_algorithms: List[str],
    size: int,
) -> pd.DataFrame:
    df = stacked.copy()
    support = df[~df["algorithm"].isin(preferred_algorithms)].copy()
    support = support.sort_values(["selection_score", "radius_eff"], ascending=[False, False]).head(size).copy()
    if support.empty:
        support = df[df["distance_status"].isin(["proxy_only", "missing", "invalid_redshift_like"])].copy()
        support = support.sort_values(["selection_score", "radius_eff"], ascending=[False, False]).head(size).copy()
    support = add_input_metadata(support, "support_stacked", "stacked_pool_selection")
    return support


def build_master_input(
    core_rep: pd.DataFrame,
    support_rep: pd.DataFrame,
    core_stacked: pd.DataFrame,
    support_stacked: pd.DataFrame,
) -> pd.DataFrame:
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


def write_summary(
    path: Path,
    run_root: Path,
    input_root: Path,
    stellar_path: Path,
    stellar_meta: Dict[str, str],
    preferred_algorithms: List[str],
    core_rep: pd.DataFrame,
    support_rep: pd.DataFrame,
    core_stacked: pd.DataFrame,
    support_stacked: pd.DataFrame,
    master: pd.DataFrame,
    match_radius_deg: float,
    run_tag: str,
) -> None:
    def matched_count(df: pd.DataFrame) -> int:
        return int(df["mass_match_status"].eq("matched").sum()) if ("mass_match_status" in df.columns and not df.empty) else 0

    lines = []
    lines.append("Cosmic Void final input builder summary (002 mass-augmented)")
    lines.append("===========================================================")
    lines.append(f"Source run root: {run_root}")
    lines.append(f"Input root: {input_root}")
    lines.append(f"Stellar FITS: {stellar_path}")
    lines.append(f"Preferred core algorithms: {', '.join(preferred_algorithms)}")
    lines.append(f"Run tag: {run_tag}")
    lines.append(f"Match radius (deg): {match_radius_deg}")
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
    lines.append("")
    lines.append("Row counts")
    lines.append("----------")
    lines.append(f"- core representatives: {len(core_rep)} | matched: {matched_count(core_rep)}")
    lines.append(f"- support representatives: {len(support_rep)} | matched: {matched_count(support_rep)}")
    lines.append(f"- core stacked: {len(core_stacked)} | matched: {matched_count(core_stacked)}")
    lines.append(f"- support stacked: {len(support_stacked)} | matched: {matched_count(support_stacked)}")
    lines.append(f"- master input total: {len(master)} | matched: {matched_count(master)}")
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append(
        "This 002 stage keeps the same placement and output role as 001, but augments the selected void inputs "
        "with locally matched stellar-mass fields from the DESI stellar-mass catalog. "
        "Direct mass-like fields are preferred when available; otherwise mass-proxy fields are used."
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

    representatives = dfs["representatives"]
    stacked = dfs["stacked"]
    preferred_algorithms = clean_algorithm_list(args.prefer_core_algorithms)

    ensure_dir(input_root)

    stellar_raw = read_fits_table(stellar_path, max_rows=args.max_stellar_rows)
    stellar_prepared, stellar_meta = prepare_stellar_mass_catalog(stellar_raw)

    core_rep = build_core_representatives(representatives, preferred_algorithms)
    support_rep = build_support_representatives(representatives, preferred_algorithms)
    core_stacked = build_core_stacked(stacked, preferred_algorithms, args.stacked_core_size)
    support_stacked = build_support_stacked(stacked, preferred_algorithms, args.stacked_support_size)

    core_rep = harmonize_columns(augment_with_mass_proxy(core_rep, stellar_prepared, args.match_radius_deg))
    support_rep = harmonize_columns(augment_with_mass_proxy(support_rep, stellar_prepared, args.match_radius_deg))
    core_stacked = harmonize_columns(augment_with_mass_proxy(core_stacked, stellar_prepared, args.match_radius_deg))
    support_stacked = harmonize_columns(augment_with_mass_proxy(support_stacked, stellar_prepared, args.match_radius_deg))

    master = harmonize_columns(build_master_input(core_rep, support_rep, core_stacked, support_stacked))
    manifest = build_source_manifest(std_catalogs)

    core_rep.to_csv(input_root / "cosmic_void_core_representatives_input.csv", index=False, encoding="utf-8-sig")
    support_rep.to_csv(input_root / "cosmic_void_support_representatives_input.csv", index=False, encoding="utf-8-sig")
    core_stacked.to_csv(input_root / "cosmic_void_core_stacked_input.csv", index=False, encoding="utf-8-sig")
    support_stacked.to_csv(input_root / "cosmic_void_support_stacked_input.csv", index=False, encoding="utf-8-sig")
    master.to_csv(input_root / "cosmic_void_master_input.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(input_root / "cosmic_void_input_source_manifest.csv", index=False, encoding="utf-8-sig")

    write_summary(
        input_root / "cosmic_void_input_build_summary.txt",
        run_root=run_root,
        input_root=input_root,
        stellar_path=stellar_path,
        stellar_meta=stellar_meta,
        preferred_algorithms=preferred_algorithms,
        core_rep=core_rep,
        support_rep=support_rep,
        core_stacked=core_stacked,
        support_stacked=support_stacked,
        master=master,
        match_radius_deg=args.match_radius_deg,
        run_tag=args.run_tag,
    )

    print(f"Source run root: {run_root}")
    print(f"Input root: {input_root}")
    print(f"Stellar FITS: {stellar_path}")
    print(f"Prepared stellar rows: {len(stellar_prepared)}")
    print(f"Core representatives: {len(core_rep)}")
    print(f"Support representatives: {len(support_rep)}")
    print(f"Core stacked: {len(core_stacked)}")
    print(f"Support stacked: {len(support_stacked)}")
    print(f"Master input total: {len(master)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
