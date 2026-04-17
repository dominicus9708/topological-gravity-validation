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
OUTPUT_ROOT_DEFAULT = Path("results") / "Cosmic Void Structural Validation" / "output" / "topological"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--input-root", default=str(INPUT_ROOT_DEFAULT), help="Official input folder from finalized selection")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT_DEFAULT), help="Topological output root")
    parser.add_argument("--gamma", type=float, default=1.0, help="Constant coefficient in D_void = 3 + gamma * sigma_void")
    parser.add_argument("--background-subset", choices=["core_stacked", "master"], default="core_stacked", help="Subset used to define rho_bg")
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


def add_topological_metrics(df: pd.DataFrame, rho_bg: float, gamma: float) -> pd.DataFrame:
    out = df.copy()

    out["radius_eff"] = safe_num(out.get("radius_eff", pd.Series(dtype=float)))
    out["edge_fraction"] = safe_num(out.get("edge_fraction", pd.Series(dtype=float)))
    out["mass_preferred_sum"] = safe_num(out.get("mass_preferred_sum", pd.Series(dtype=float)))
    out["mass_match_count"] = safe_num(out.get("mass_match_count", pd.Series(dtype=float)))
    out["selection_score"] = safe_num(out.get("selection_score", pd.Series(dtype=float)))
    out["mass_quality_score"] = safe_num(out.get("mass_quality_score", pd.Series(dtype=float)))
    out["mass_nearest_sep_deg"] = safe_num(out.get("mass_nearest_sep_deg", pd.Series(dtype=float)))

    # IMPORTANT:
    # Recompute all baseline-like quantities directly from official input.
    # Do not reuse standard pipeline outputs.
    out["effective_volume_proxy"] = (4.0 / 3.0) * np.pi * (out["radius_eff"] ** 3)
    out["projected_area_proxy"] = np.pi * (out["radius_eff"] ** 2)

    out["rho_proxy"] = np.where(
        out["effective_volume_proxy"] > 0,
        out["mass_preferred_sum"] / out["effective_volume_proxy"],
        np.nan,
    )

    out["rho_bg"] = rho_bg
    out["sigma_void"] = np.where(
        np.isfinite(out["rho_proxy"]) & np.isfinite(rho_bg) & (rho_bg != 0),
        (out["rho_proxy"] - rho_bg) / rho_bg,
        np.nan,
    )

    out["D_void"] = 3.0 + gamma * out["sigma_void"]

    # Optional edge-aware variant kept internal to this pipeline only
    out["sigma_void_edge_adjusted"] = np.where(
        out["edge_fraction"].notna(),
        out["sigma_void"] * (1.0 - out["edge_fraction"].clip(lower=0)),
        out["sigma_void"],
    )
    out["D_void_edge_adjusted"] = 3.0 + gamma * out["sigma_void_edge_adjusted"]

    out["topological_quality_flag"] = np.where(
        out["mass_strict_eligible"].fillna(False)
        & out["rho_proxy"].notna()
        & out["sigma_void"].notna(),
        "usable",
        "limited",
    )

    return out


def compute_background_density(core_stacked: pd.DataFrame, master: pd.DataFrame, subset_name: str) -> float:
    def density(df: pd.DataFrame) -> pd.Series:
        r = safe_num(df["radius_eff"])
        m = safe_num(df["mass_preferred_sum"])
        v = (4.0 / 3.0) * np.pi * (r ** 3)
        rho = np.where(v > 0, m / v, np.nan)
        return pd.Series(rho)

    if subset_name == "core_stacked":
        series = density(core_stacked).dropna()
    else:
        series = density(master).dropna()

    if len(series) == 0:
        return np.nan
    return float(series.median())


def summarize_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    rows.append({"subset": label, "metric": "rows", "value": len(df)})
    if len(df) == 0:
        return pd.DataFrame(rows)

    numeric_cols = [
        "radius_eff",
        "effective_volume_proxy",
        "mass_preferred_sum",
        "rho_proxy",
        "rho_bg",
        "sigma_void",
        "D_void",
        "sigma_void_edge_adjusted",
        "D_void_edge_adjusted",
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
    if "topological_quality_flag" in df.columns:
        rows.append({"subset": label, "metric": "topological_quality_counts", "value": str(df["topological_quality_flag"].value_counts(dropna=False).to_dict())})

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

    # Read official input only
    master = read_csv(input_root / "cosmic_void_master_input.csv")
    core_rep = read_csv(input_root / "cosmic_void_core_representatives_input.csv")
    support_rep = read_csv(input_root / "cosmic_void_support_representatives_input.csv")
    core_stacked = read_csv(input_root / "cosmic_void_core_stacked_input.csv")
    support_stacked = read_csv(input_root / "cosmic_void_support_stacked_input.csv")

    rho_bg = compute_background_density(core_stacked=core_stacked, master=master, subset_name=args.background_subset)

    master_topo = add_topological_metrics(master, rho_bg=rho_bg, gamma=args.gamma)
    core_rep_topo = add_topological_metrics(core_rep, rho_bg=rho_bg, gamma=args.gamma)
    support_rep_topo = add_topological_metrics(support_rep, rho_bg=rho_bg, gamma=args.gamma)
    core_stacked_topo = add_topological_metrics(core_stacked, rho_bg=rho_bg, gamma=args.gamma)
    support_stacked_topo = add_topological_metrics(support_stacked, rho_bg=rho_bg, gamma=args.gamma)

    master_topo.to_csv(run_dir / "topological_master_working.csv", index=False, encoding="utf-8-sig")
    core_rep_topo.to_csv(run_dir / "topological_core_representatives_working.csv", index=False, encoding="utf-8-sig")
    support_rep_topo.to_csv(run_dir / "topological_support_representatives_working.csv", index=False, encoding="utf-8-sig")
    core_stacked_topo.to_csv(run_dir / "topological_core_stacked_working.csv", index=False, encoding="utf-8-sig")
    support_stacked_topo.to_csv(run_dir / "topological_support_stacked_working.csv", index=False, encoding="utf-8-sig")

    summary = pd.concat([
        summarize_frame(master_topo, "master"),
        summarize_frame(core_rep_topo, "core_representatives"),
        summarize_frame(support_rep_topo, "support_representatives"),
        summarize_frame(core_stacked_topo, "core_stacked"),
        summarize_frame(support_stacked_topo, "support_stacked"),
    ], ignore_index=True)
    summary.to_csv(run_dir / "topological_summary.csv", index=False, encoding="utf-8-sig")

    if len(master_topo):
        if master_topo["radius_eff"].notna().any() and master_topo["sigma_void"].notna().any():
            save_scatter(
                master_topo,
                "radius_eff",
                "sigma_void",
                "Topological baseline: radius vs sigma_void",
                "radius_eff",
                "sigma_void",
                run_dir / "topological_radius_vs_sigma_void.png",
            )
        if master_topo["radius_eff"].notna().any() and master_topo["D_void"].notna().any():
            save_scatter(
                master_topo,
                "radius_eff",
                "D_void",
                "Topological baseline: radius vs D_void",
                "radius_eff",
                "D_void",
                run_dir / "topological_radius_vs_D_void.png",
            )
        if "algorithm" in master_topo.columns:
            save_bar_counts(master_topo, "algorithm", "Topological baseline: algorithm counts", run_dir / "topological_algorithm_counts.png")
        if core_rep_topo["D_void"].notna().any():
            save_rank_plot(
                core_rep_topo,
                "D_void",
                "Core representatives: D_void",
                "D_void",
                run_dir / "topological_core_representatives_D_void.png",
            )
        if core_stacked_topo["sigma_void"].notna().any():
            save_rank_plot(
                core_stacked_topo,
                "sigma_void",
                "Core stacked: sigma_void",
                "sigma_void",
                run_dir / "topological_core_stacked_sigma_void.png",
            )

    lines = []
    lines.append("Cosmic Void Structural Validation topological summary")
    lines.append("====================================================")
    lines.append(f"Project root: {project_root}")
    lines.append(f"Input root: {input_root}")
    lines.append(f"Run output: {run_dir}")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This topological-stage pipeline reads the finalized official input directly and computes "
        "void structural contrast and effective structural dimension without reusing any standard-stage outputs."
    )
    lines.append("")
    lines.append("Definitions")
    lines.append("-----------")
    lines.append("V_eff = (4/3) * pi * r_eff^3")
    lines.append("rho_proxy = M_proxy / V_eff")
    lines.append("sigma_void = (rho_proxy - rho_bg) / rho_bg")
    lines.append(f"D_void = 3 + gamma * sigma_void, gamma = {args.gamma}")
    lines.append(f"rho_bg subset = {args.background_subset}")
    lines.append(f"rho_bg value = {rho_bg}")
    lines.append("")
    lines.append("Subsets")
    lines.append("-------")
    lines.append(f"- master rows: {len(master_topo)}")
    lines.append(f"- core representatives: {len(core_rep_topo)}")
    lines.append(f"- support representatives: {len(support_rep_topo)}")
    lines.append(f"- core stacked: {len(core_stacked_topo)}")
    lines.append(f"- support stacked: {len(support_stacked_topo)}")
    lines.append("")
    lines.append("Generated files")
    lines.append("---------------")
    lines.append("- topological_master_working.csv")
    lines.append("- topological_core_representatives_working.csv")
    lines.append("- topological_support_representatives_working.csv")
    lines.append("- topological_core_stacked_working.csv")
    lines.append("- topological_support_stacked_working.csv")
    lines.append("- topological_summary.csv")
    lines.append("- topological_radius_vs_sigma_void.png")
    lines.append("- topological_radius_vs_D_void.png")
    lines.append("- topological_algorithm_counts.png")
    lines.append("- topological_core_representatives_D_void.png")
    lines.append("- topological_core_stacked_sigma_void.png")
    lines.append("- topological_summary.txt")
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append(
        "This stage uses the same official input source as the standard stage, but recomputes its own "
        "effective-volume and proxy-density quantities internally. No standard pipeline result file is used as input."
    )
    write_text(run_dir / "topological_summary.txt", "\n".join(lines))

    print(f"Input root: {input_root}")
    print(f"Run output: {run_dir}")
    print(f"rho_bg: {rho_bg}")
    print(f"Master rows: {len(master_topo)}")
    print(f"Core representatives: {len(core_rep_topo)}")
    print(f"Support representatives: {len(support_rep_topo)}")
    print(f"Core stacked: {len(core_stacked_topo)}")
    print(f"Support stacked: {len(support_stacked_topo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
