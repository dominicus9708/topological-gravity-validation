#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_RESULTS_DEFAULT = Path("results") / "Cosmic Void Structural Validation"
STANDARD_ROOT_DEFAULT = BASE_RESULTS_DEFAULT / "output" / "standard"
TOPOLOGICAL_ROOT_DEFAULT = BASE_RESULTS_DEFAULT / "output" / "topological"
INTEGRATION_ROOT_DEFAULT = BASE_RESULTS_DEFAULT / "output" / "Integration"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--standard-root", default=str(STANDARD_ROOT_DEFAULT), help="Root folder containing standard timestamp runs")
    parser.add_argument("--topological-root", default=str(TOPOLOGICAL_ROOT_DEFAULT), help="Root folder containing topological timestamp runs")
    parser.add_argument("--output-root", default=str(INTEGRATION_ROOT_DEFAULT), help="Integration output root")
    parser.add_argument("--standard-run", default="", help="Optional explicit standard run folder name")
    parser.add_argument("--topological-run", default="", help="Optional explicit topological run folder name")
    parser.add_argument("--run-tag", default="", help="Optional suffix appended to integration timestamp folder")
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
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path)


def safe_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def looks_like_standard_run(path: Path) -> bool:
    return (path / "standard_master_working.csv").exists()


def looks_like_topological_run(path: Path) -> bool:
    return (path / "topological_master_working.csv").exists()


def maybe_case_variant(path: Path) -> Path:
    if path.exists():
        return path
    parent = path.parent
    target = path.name.lower()
    if parent.exists():
        for child in parent.iterdir():
            if child.name.lower() == target:
                return child
    return path


def detect_latest_run(root: Path, mode: str) -> Path:
    root = maybe_case_variant(root)
    if not root.exists():
        raise FileNotFoundError(f"Run root not found: {root}")

    # Case 1: root itself already contains the required files
    if mode == "standard" and looks_like_standard_run(root):
        return root
    if mode == "topological" and looks_like_topological_run(root):
        return root

    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"No run folders found under: {root}\n"
            f"Expected either timestamp subfolders or the required working CSVs directly inside the folder."
        )

    candidates = sorted(candidates, key=lambda p: p.name, reverse=True)
    for cand in candidates:
        if mode == "standard" and looks_like_standard_run(cand):
            return cand
        if mode == "topological" and looks_like_topological_run(cand):
            return cand

    raise FileNotFoundError(
        f"No valid {mode} run folder found under: {root}\n"
        f"Expected a folder containing the required working CSVs."
    )


def select_run(root: Path, explicit_name: str, mode: str) -> Path:
    root = maybe_case_variant(root)
    if explicit_name:
        path = root / explicit_name
        path = maybe_case_variant(path)
        if not path.exists():
            raise FileNotFoundError(f"Explicit run folder not found: {path}")
        if mode == "standard" and not looks_like_standard_run(path):
            raise FileNotFoundError(f"Explicit standard run folder does not contain standard_master_working.csv: {path}")
        if mode == "topological" and not looks_like_topological_run(path):
            raise FileNotFoundError(f"Explicit topological run folder does not contain topological_master_working.csv: {path}")
        return path
    return detect_latest_run(root, mode)


def standard_density_column(df: pd.DataFrame) -> str:
    preferred = [
        "mass_volume_proxy_density",
        "rho_proxy",
        "rho_proxy_standard_like",
    ]
    for col in preferred:
        if col in df.columns:
            return col
    raise ValueError("Could not find a standard-side density column in standard master working CSV.")


def topological_response_column(df: pd.DataFrame) -> str:
    if "D_void" in df.columns:
        return "D_void"
    raise ValueError("Could not find D_void in topological master working CSV.")


def merge_master_frames(std_df: pd.DataFrame, topo_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["void_id", "source_catalog_key", "algorithm", "sky_region"]
    shared = [c for c in key_cols if c in std_df.columns and c in topo_df.columns]
    if not shared:
        raise ValueError("Could not find shared key columns between standard and topological master tables.")

    std = std_df.copy()
    topo = topo_df.copy()

    std_nonkeys = [c for c in std.columns if c not in shared]
    topo_nonkeys = [c for c in topo.columns if c not in shared]

    merged = std[shared + std_nonkeys].merge(
        topo[shared + topo_nonkeys],
        on=shared,
        how="inner",
        suffixes=("_std", "_topo"),
    )
    return merged


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_comparison_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()

    rho_std_col = standard_density_column(out)
    out["rho_std"] = safe_num(out[rho_std_col])

    if "rho_bg" in out.columns:
        out["rho_bg_topo"] = safe_num(out["rho_bg"])
    elif "rho_bg_topo" in out.columns:
        out["rho_bg_topo"] = safe_num(out["rho_bg_topo"])
    else:
        out["rho_bg_topo"] = np.nan

    d_col = topological_response_column(out)
    out["D_void"] = safe_num(out[d_col])
    out["sigma_void"] = safe_num(out["sigma_void"]) if "sigma_void" in out.columns else np.nan

    if "radius_eff" in out.columns:
        out["radius_eff"] = safe_num(out["radius_eff"])
    elif "radius_eff_std" in out.columns:
        out["radius_eff"] = safe_num(out["radius_eff_std"])
    else:
        out["radius_eff"] = np.nan

    out["r_topo"] = safe_num(out["r_topo"]) if "r_topo" in out.columns else np.nan

    out["standard_log_density_ratio"] = np.where(
        out["rho_std"].notna() & out["rho_bg_topo"].notna() & (out["rho_std"] > 0) & (out["rho_bg_topo"] > 0),
        np.log10(out["rho_std"] / out["rho_bg_topo"]),
        np.nan,
    )
    out["topological_dimension_shift"] = out["D_void"] - 3.0

    return out


def save_comparison_map(df: pd.DataFrame, out_path: Path) -> None:
    data = df[["standard_log_density_ratio", "topological_dimension_shift"]].copy()
    data["standard_log_density_ratio"] = safe_num(data["standard_log_density_ratio"])
    data["topological_dimension_shift"] = safe_num(data["topological_dimension_shift"])
    data = data.dropna()

    plt.figure(figsize=(8, 7))
    plt.scatter(data["standard_log_density_ratio"], data["topological_dimension_shift"], s=30)
    plt.axhline(0.0, linewidth=1.0)
    plt.axvline(0.0, linewidth=1.0)
    plt.xlabel("log10(rho_proxy_std / rho_bg)")
    plt.ylabel("D_void - 3")
    plt.title("Standard vs Topological comparison map")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_radius_transform(df: pd.DataFrame, out_path: Path) -> None:
    data = df[["radius_eff", "r_topo"]].copy()
    data["radius_eff"] = safe_num(data["radius_eff"])
    data["r_topo"] = safe_num(data["r_topo"])
    data = data.dropna()

    plt.figure(figsize=(8, 7))
    plt.scatter(data["radius_eff"], data["r_topo"], s=30)
    if len(data):
        lo = float(min(data["radius_eff"].min(), data["r_topo"].min()))
        hi = float(max(data["radius_eff"].max(), data["r_topo"].max()))
        plt.plot([lo, hi], [lo, hi], linewidth=1.0)
    plt.xlabel("radius_eff")
    plt.ylabel("r_topo")
    plt.title("Radius transformation map")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_representative_vs_stacked(core_rep_topo: pd.DataFrame, core_stacked_topo: pd.DataFrame, out_path: Path) -> None:
    rep = core_rep_topo.copy().reset_index(drop=True)
    st = core_stacked_topo.copy().reset_index(drop=True)

    rep["D_void"] = safe_num(rep["D_void"]) if "D_void" in rep.columns else np.nan
    st["sigma_void"] = safe_num(st["sigma_void"]) if "sigma_void" in st.columns else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(rep.index, rep["D_void"], s=40)
    axes[0].axhline(3.0, linewidth=1.0)
    axes[0].set_xlabel("Representative index")
    axes[0].set_ylabel("D_void")
    axes[0].set_title("Representatives: D_void")

    axes[1].scatter(st.index, st["sigma_void"], s=25)
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_xlabel("Stacked index")
    axes[1].set_ylabel("sigma_void")
    axes[1].set_title("Core stacked: sigma_void")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_three_panel_figure(comparison_df: pd.DataFrame, core_rep_topo: pd.DataFrame, core_stacked_topo: pd.DataFrame, out_path: Path) -> None:
    comp = comparison_df.copy()
    comp["standard_log_density_ratio"] = safe_num(comp["standard_log_density_ratio"])
    comp["topological_dimension_shift"] = safe_num(comp["topological_dimension_shift"])
    comp["radius_eff"] = safe_num(comp["radius_eff"])
    comp["r_topo"] = safe_num(comp["r_topo"])

    rep = core_rep_topo.copy().reset_index(drop=True)
    rep["D_void"] = safe_num(rep["D_void"]) if "D_void" in rep.columns else np.nan

    st = core_stacked_topo.copy().reset_index(drop=True)
    st["sigma_void"] = safe_num(st["sigma_void"]) if "sigma_void" in st.columns else np.nan

    fig = plt.figure(figsize=(12, 10))

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    comp1 = comp[["standard_log_density_ratio", "topological_dimension_shift"]].dropna()
    ax1.scatter(comp1["standard_log_density_ratio"], comp1["topological_dimension_shift"], s=30)
    ax1.axhline(0.0, linewidth=1.0)
    ax1.axvline(0.0, linewidth=1.0)
    ax1.set_xlabel("log10(rho_proxy_std / rho_bg)")
    ax1.set_ylabel("D_void - 3")
    ax1.set_title("A. Standard vs Topological comparison map")

    ax2 = plt.subplot2grid((2, 2), (1, 0))
    comp2 = comp[["radius_eff", "r_topo"]].dropna()
    ax2.scatter(comp2["radius_eff"], comp2["r_topo"], s=30)
    if len(comp2):
        lo = float(min(comp2["radius_eff"].min(), comp2["r_topo"].min()))
        hi = float(max(comp2["radius_eff"].max(), comp2["r_topo"].max()))
        ax2.plot([lo, hi], [lo, hi], linewidth=1.0)
    ax2.set_xlabel("radius_eff")
    ax2.set_ylabel("r_topo")
    ax2.set_title("B. Radius transformation")

    ax3 = plt.subplot2grid((2, 2), (1, 1))
    ax3.scatter(rep.index, rep["D_void"], s=40)
    ax3.axhline(3.0, linewidth=1.0)
    ax3.set_xlabel("Representative index")
    ax3.set_ylabel("D_void")
    ax3.set_title("C. Representative response")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    standard_root = resolve_path(project_root, args.standard_root)
    topological_root = resolve_path(project_root, args.topological_root)
    output_root = resolve_path(project_root, args.output_root)

    standard_run = select_run(standard_root, args.standard_run, "standard")
    topological_run = select_run(topological_root, args.topological_run, "topological")
    run_dir = make_run_dir(output_root, args.run_tag)

    std_master = read_csv(standard_run / "standard_master_working.csv")
    topo_master = read_csv(topological_run / "topological_master_working.csv")
    topo_core_rep = read_csv(topological_run / "topological_core_representatives_working.csv")
    topo_core_stacked = read_csv(topological_run / "topological_core_stacked_working.csv")

    merged = merge_master_frames(std_master, topo_master)
    comparison = build_comparison_metrics(merged)

    comparison.to_csv(run_dir / "integration_master_comparison.csv", index=False, encoding="utf-8-sig")

    save_comparison_map(comparison, run_dir / "integration_standard_vs_topological_map.png")
    save_radius_transform(comparison, run_dir / "integration_radius_transformation_map.png")
    save_representative_vs_stacked(topo_core_rep, topo_core_stacked, run_dir / "integration_representative_vs_stacked.png")
    save_three_panel_figure(comparison, topo_core_rep, topo_core_stacked, run_dir / "integration_three_panel_figure.png")

    rows = []
    rows.append({"metric": "standard_run", "value": str(standard_run)})
    rows.append({"metric": "topological_run", "value": str(topological_run)})
    rows.append({"metric": "merged_rows", "value": len(comparison)})

    for col in ["standard_log_density_ratio", "topological_dimension_shift", "radius_eff", "r_topo"]:
        s = safe_num(comparison[col]).dropna()
        if len(s):
            rows.extend([
                {"metric": f"{col}_min", "value": float(s.min())},
                {"metric": f"{col}_median", "value": float(s.median())},
                {"metric": f"{col}_max", "value": float(s.max())},
            ])

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(run_dir / "integration_summary.csv", index=False, encoding="utf-8-sig")

    text = []
    text.append("Cosmic Void Structural Validation integration summary")
    text.append("====================================================")
    text.append(f"Project root: {project_root}")
    text.append(f"Standard root: {standard_root}")
    text.append(f"Topological root: {topological_root}")
    text.append(f"Selected standard run: {standard_run}")
    text.append(f"Selected topological run: {topological_run}")
    text.append(f"Integration output: {run_dir}")
    text.append("")
    text.append("Purpose")
    text.append("-------")
    text.append(
        "This integration stage reads the latest standard and topological run folders, "
        "merges the master comparison table on shared void keys, and exports paper-style comparison figures."
    )
    text.append("")
    text.append("Generated files")
    text.append("---------------")
    text.append("- integration_master_comparison.csv")
    text.append("- integration_summary.csv")
    text.append("- integration_standard_vs_topological_map.png")
    text.append("- integration_radius_transformation_map.png")
    text.append("- integration_representative_vs_stacked.png")
    text.append("- integration_three_panel_figure.png")
    text.append("- integration_summary.txt")
    write_text(run_dir / "integration_summary.txt", "\n".join(text))

    print(f"Standard run: {standard_run}")
    print(f"Topological run: {topological_run}")
    print(f"Integration output: {run_dir}")
    print(f"Merged rows: {len(comparison)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
