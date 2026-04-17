#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_ROOT_DEFAULT = Path("data") / "derived" / "Cosmic Void Structural Validation" / "input"
OUTPUT_ROOT_DEFAULT = Path("results") / "Cosmic Void Structural Validation" / "output" / "standard"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--input-root", default=str(INPUT_ROOT_DEFAULT), help="Official input folder")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT_DEFAULT), help="Standard output root")
    parser.add_argument("--run-tag", default="", help="Optional suffix appended to timestamp folder")
    return parser.parse_args()


def resolve_path(project_root: Path, raw_value: str) -> Path:
    p = Path(raw_value)
    return (project_root / p).resolve() if not p.is_absolute() else p.resolve()


def make_run_dir(output_root: Path, run_tag: str) -> Path:
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    folder_name = f"{timestamp}_{run_tag}" if run_tag else timestamp
    run_dir = output_root / folder_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_standard_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["radius_eff"] = safe_num(out.get("radius_eff", pd.Series(dtype=float)))
    out["edge_fraction"] = safe_num(out.get("edge_fraction", pd.Series(dtype=float)))
    out["mass_preferred_sum"] = safe_num(out.get("mass_preferred_sum", pd.Series(dtype=float)))
    out["mass_preferred_mean"] = safe_num(out.get("mass_preferred_mean", pd.Series(dtype=float)))
    out["mass_match_count"] = safe_num(out.get("mass_match_count", pd.Series(dtype=float)))
    out["selection_score"] = safe_num(out.get("selection_score", pd.Series(dtype=float)))
    out["mass_quality_score"] = safe_num(out.get("mass_quality_score", pd.Series(dtype=float)))
    out["mass_nearest_sep_deg"] = safe_num(out.get("mass_nearest_sep_deg", pd.Series(dtype=float)))

    out["effective_volume_proxy"] = (4.0 / 3.0) * np.pi * (out["radius_eff"] ** 3)

    area_proxy = np.pi * (out["radius_eff"] ** 2)
    out["projected_area_proxy"] = area_proxy

    out["mass_surface_proxy"] = np.where(
        area_proxy > 0,
        out["mass_preferred_sum"] / area_proxy,
        np.nan,
    )
    out["mass_volume_proxy_density"] = np.where(
        out["effective_volume_proxy"] > 0,
        out["mass_preferred_sum"] / out["effective_volume_proxy"],
        np.nan,
    )

    out["edge_corrected_radius_proxy"] = np.where(
        out["edge_fraction"].notna(),
        out["radius_eff"] * (1.0 - out["edge_fraction"].clip(lower=0)),
        out["radius_eff"],
    )

    out["standard_quality_flag"] = np.where(
        out["mass_strict_eligible"].fillna(False)
        & out["radius_eff"].notna()
        & out["mass_preferred_sum"].notna(),
        "usable",
        "limited",
    )

    return out


def summarize_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    rows.append({"subset": label, "metric": "rows", "value": len(df)})
    if len(df) == 0:
        return pd.DataFrame(rows)

    numeric_cols = [
        "radius_eff",
        "effective_volume_proxy",
        "mass_preferred_sum",
        "mass_surface_proxy",
        "mass_volume_proxy_density",
        "mass_match_count",
        "mass_nearest_sep_deg",
        "selection_score",
        "mass_quality_score",
    ]

    for col in numeric_cols:
        s = safe_num(df[col]).dropna() if col in df.columns else pd.Series(dtype=float)
        if len(s):
            rows.extend([
                {"subset": label, "metric": f"{col}_min", "value": float(s.min())},
                {"subset": label, "metric": f"{col}_median", "value": float(s.median())},
                {"subset": label, "metric": f"{col}_max", "value": float(s.max())},
            ])

    if "algorithm" in df.columns:
        rows.append({"subset": label, "metric": "algorithm_counts", "value": str(df["algorithm"].value_counts(dropna=False).to_dict())})
    if "sky_region" in df.columns:
        rows.append({"subset": label, "metric": "sky_region_counts", "value": str(df["sky_region"].value_counts(dropna=False).to_dict())})
    if "radius_band" in df.columns:
        rows.append({"subset": label, "metric": "radius_band_counts", "value": str(df["radius_band"].value_counts(dropna=False).to_dict())})
    if "standard_quality_flag" in df.columns:
        rows.append({"subset": label, "metric": "standard_quality_counts", "value": str(df["standard_quality_flag"].value_counts(dropna=False).to_dict())})

    return pd.DataFrame(rows)


def save_scatter(df: pd.DataFrame, xcol: str, ycol: str, title: str, xlabel: str, ylabel: str, out_path: Path) -> None:
    data = df[[xcol, ycol]].copy()
    data[xcol] = safe_num(data[xcol])
    data[ycol] = safe_num(data[ycol])
    data = data.dropna()
    plt.figure(figsize=(8, 6))
    plt.scatter(data[xcol], data[ycol], s=25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_rank_plot(df: pd.DataFrame, ycol: str, title: str, ylabel: str, out_path: Path) -> None:
    work = df.copy().reset_index(drop=True)
    work[ycol] = safe_num(work[ycol])
    plt.figure(figsize=(9, 6))
    plt.scatter(work.index, work[ycol], s=25)
    plt.xlabel("Row index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_bar_counts(df: pd.DataFrame, col: str, title: str, out_path: Path) -> None:
    counts = df[col].astype(str).value_counts()
    plt.figure(figsize=(8, 6))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    input_root = resolve_path(project_root, args.input_root)
    output_root = resolve_path(project_root, args.output_root)
    run_dir = make_run_dir(output_root, args.run_tag)

    master = read_csv(input_root / "cosmic_void_master_input.csv")
    core_rep = read_csv(input_root / "cosmic_void_core_representatives_input.csv")
    support_rep = read_csv(input_root / "cosmic_void_support_representatives_input.csv")
    core_stacked = read_csv(input_root / "cosmic_void_core_stacked_input.csv")
    support_stacked = read_csv(input_root / "cosmic_void_support_stacked_input.csv")

    master_std = add_standard_metrics(master)
    core_rep_std = add_standard_metrics(core_rep)
    support_rep_std = add_standard_metrics(support_rep)
    core_stacked_std = add_standard_metrics(core_stacked)
    support_stacked_std = add_standard_metrics(support_stacked)

    master_std.to_csv(run_dir / "standard_master_working.csv", index=False, encoding="utf-8-sig")
    core_rep_std.to_csv(run_dir / "standard_core_representatives_working.csv", index=False, encoding="utf-8-sig")
    support_rep_std.to_csv(run_dir / "standard_support_representatives_working.csv", index=False, encoding="utf-8-sig")
    core_stacked_std.to_csv(run_dir / "standard_core_stacked_working.csv", index=False, encoding="utf-8-sig")
    support_stacked_std.to_csv(run_dir / "standard_support_stacked_working.csv", index=False, encoding="utf-8-sig")

    summary = pd.concat([
        summarize_frame(master_std, "master"),
        summarize_frame(core_rep_std, "core_representatives"),
        summarize_frame(support_rep_std, "support_representatives"),
        summarize_frame(core_stacked_std, "core_stacked"),
        summarize_frame(support_stacked_std, "support_stacked"),
    ], ignore_index=True)
    summary.to_csv(run_dir / "standard_summary.csv", index=False, encoding="utf-8-sig")

    if len(master_std):
        if master_std["radius_eff"].notna().any() and master_std["mass_preferred_sum"].notna().any():
            save_scatter(
                master_std,
                "radius_eff",
                "mass_preferred_sum",
                "Standard baseline: radius vs mass proxy sum",
                "radius_eff",
                "mass_preferred_sum",
                run_dir / "standard_radius_vs_mass_proxy_sum.png",
            )
        if master_std["radius_eff"].notna().any() and master_std["mass_volume_proxy_density"].notna().any():
            save_scatter(
                master_std,
                "radius_eff",
                "mass_volume_proxy_density",
                "Standard baseline: radius vs mass-volume proxy density",
                "radius_eff",
                "mass_volume_proxy_density",
                run_dir / "standard_radius_vs_mass_density_proxy.png",
            )
        if "algorithm" in master_std.columns:
            save_bar_counts(master_std, "algorithm", "Standard baseline: algorithm counts", run_dir / "standard_algorithm_counts.png")
        if core_rep_std["radius_eff"].notna().any():
            save_rank_plot(
                core_rep_std,
                "radius_eff",
                "Core representatives: radius_eff",
                "radius_eff",
                run_dir / "standard_core_representatives_radius.png",
            )
        if core_stacked_std["mass_quality_score"].notna().any():
            save_rank_plot(
                core_stacked_std,
                "mass_quality_score",
                "Core stacked: mass quality score",
                "mass_quality_score",
                run_dir / "standard_core_stacked_mass_quality.png",
            )

    lines = []
    lines.append("Cosmic Void Structural Validation standard summary")
    lines.append("=================================================")
    lines.append(f"Project root: {project_root}")
    lines.append(f"Input root: {input_root}")
    lines.append(f"Run output: {run_dir}")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This standard-stage pipeline constructs a geometry-first and proxy-mass baseline from the finalized 006 input. "
        "It does not claim direct void mass measurement. Instead it summarizes radius, effective-volume proxy, "
        "projected area proxy, stellar mass-proxy totals, and simple density-like ratios."
    )
    lines.append("")
    lines.append("Subsets")
    lines.append("-------")
    lines.append(f"- master rows: {len(master_std)}")
    lines.append(f"- core representatives: {len(core_rep_std)}")
    lines.append(f"- support representatives: {len(support_rep_std)}")
    lines.append(f"- core stacked: {len(core_stacked_std)}")
    lines.append(f"- support stacked: {len(support_stacked_std)}")
    lines.append("")
    lines.append("Generated files")
    lines.append("---------------")
    lines.append("- standard_master_working.csv")
    lines.append("- standard_core_representatives_working.csv")
    lines.append("- standard_support_representatives_working.csv")
    lines.append("- standard_core_stacked_working.csv")
    lines.append("- standard_support_stacked_working.csv")
    lines.append("- standard_summary.csv")
    lines.append("- standard_radius_vs_mass_proxy_sum.png")
    lines.append("- standard_radius_vs_mass_density_proxy.png")
    lines.append("- standard_algorithm_counts.png")
    lines.append("- standard_core_representatives_radius.png")
    lines.append("- standard_core_stacked_mass_quality.png")
    lines.append("- standard_summary.txt")
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append(
        "This is a standard observational baseline for Section 9.3. "
        "The quantities labeled as mass or density here are proxy-level summaries based on MASS_CG-derived matches, "
        "not direct total void masses."
    )
    write_text(run_dir / "standard_summary.txt", "\n".join(lines))

    print(f"Input root: {input_root}")
    print(f"Run output: {run_dir}")
    print(f"Master rows: {len(master_std)}")
    print(f"Core representatives: {len(core_rep_std)}")
    print(f"Support representatives: {len(support_rep_std)}")
    print(f"Core stacked: {len(core_stacked_std)}")
    print(f"Support stacked: {len(support_stacked_std)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
