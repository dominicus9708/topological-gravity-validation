#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


BASE_DERIVED_DEFAULT = Path("data") / "derived" / "Cosmic Void Structural Validation" / "derived"
INPUT_ROOT_DEFAULT = Path("data") / "derived" / "Cosmic Void Structural Validation" / "input"


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
    return harmonize_columns(core)


def build_support_representatives(
    representatives: pd.DataFrame,
    preferred_algorithms: List[str],
) -> pd.DataFrame:
    df = representatives.copy()
    support = df[~df["algorithm"].isin(preferred_algorithms)].copy()
    if support.empty:
        support = df[df["distance_status"].isin(["proxy_only", "missing", "invalid_redshift_like"])].copy()
    support = add_input_metadata(support, "support_representative", "representative_selection")
    return harmonize_columns(support)


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
    return harmonize_columns(core)


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
    return harmonize_columns(support)


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
    master = harmonize_columns(master)

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
    preferred_algorithms: List[str],
    core_rep: pd.DataFrame,
    support_rep: pd.DataFrame,
    core_stacked: pd.DataFrame,
    support_stacked: pd.DataFrame,
    master: pd.DataFrame,
    run_tag: str,
) -> None:
    lines = []
    lines.append("Cosmic Void final input builder summary")
    lines.append("======================================")
    lines.append(f"Source run root: {run_root}")
    lines.append(f"Input root: {input_root}")
    lines.append(f"Preferred core algorithms: {', '.join(preferred_algorithms)}")
    lines.append(f"Run tag: {run_tag}")
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
    lines.append(f"- core representatives: {len(core_rep)}")
    lines.append(f"- support representatives: {len(support_rep)}")
    lines.append(f"- core stacked: {len(core_stacked)}")
    lines.append(f"- support stacked: {len(support_stacked)}")
    lines.append(f"- master input total: {len(master)}")
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append(
        "This stage produces official input CSV files for the next standard/topological pipelines. "
        "It does not reprocess raw DESIVAST FITS. It consumes the latest derived selection outputs."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    base_analysis_root = resolve_path(project_root, args.analysis_root)
    input_root = resolve_path(project_root, args.input_root)

    run_root = detect_latest_run_folder(base_analysis_root)
    dfs = read_required_csvs(run_root)
    std_catalogs = read_standardized_catalogs(run_root)

    representatives = dfs["representatives"]
    stacked = dfs["stacked"]
    preferred_algorithms = clean_algorithm_list(args.prefer_core_algorithms)

    ensure_dir(input_root)

    core_rep = build_core_representatives(representatives, preferred_algorithms)
    support_rep = build_support_representatives(representatives, preferred_algorithms)
    core_stacked = build_core_stacked(stacked, preferred_algorithms, args.stacked_core_size)
    support_stacked = build_support_stacked(stacked, preferred_algorithms, args.stacked_support_size)
    master = build_master_input(core_rep, support_rep, core_stacked, support_stacked)
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
        preferred_algorithms=preferred_algorithms,
        core_rep=core_rep,
        support_rep=support_rep,
        core_stacked=core_stacked,
        support_stacked=support_stacked,
        master=master,
        run_tag=args.run_tag,
    )

    print(f"Source run root: {run_root}")
    print(f"Input root: {input_root}")
    print(f"Core representatives: {len(core_rep)}")
    print(f"Support representatives: {len(support_rep)}")
    print(f"Core stacked: {len(core_stacked)}")
    print(f"Support stacked: {len(support_stacked)}")
    print(f"Master input total: {len(master)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
