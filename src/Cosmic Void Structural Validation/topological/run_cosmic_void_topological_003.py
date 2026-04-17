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
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument("--input-root", default=str(INPUT_ROOT_DEFAULT), help="Official input folder from finalized selection")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT_DEFAULT), help="Topological output root")
    parser.add_argument(
        "--lambda-mode",
        choices=["self_normalized", "core_stacked_median", "global_max"],
        default="self_normalized",
        help="How to derive lambda_edge from edge_fraction",
    )
    parser.add_argument(
        "--gamma-mode",
        choices=["std", "mad"],
        default="std",
        help="How to derive gamma from sigma_void distribution in core_stacked",
    )
    parser.add_argument(
        "--background-subset",
        choices=["core_stacked", "master"],
        default="core_stacked",
        help="Subset used to define rho_bg after r_topo is derived",
    )
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


def derive_lambda_series(
    edge_fraction: pd.Series,
    mode: str,
    reference_series: pd.Series | None = None,
) -> pd.Series:
    edge = safe_num(edge_fraction).fillna(0.0).clip(lower=0.0)
    if mode == "self_normalized":
        lam = edge / (1.0 + edge)
    elif mode == "core_stacked_median":
        if reference_series is None:
            raise ValueError("reference_series is required for core_stacked_median mode")
        ref_med = float(safe_num(reference_series).dropna().median()) if len(safe_num(reference_series).dropna()) else 0.0
        lam = edge / (ref_med + edge + EPS)
    elif mode == "global_max":
        ref = reference_series if reference_series is not None else edge
        ref_max = float(safe_num(ref).dropna().max()) if len(safe_num(ref).dropna()) else 1.0
        lam = edge / (ref_max + EPS)
    else:
        raise ValueError(f"Unknown lambda mode: {mode}")
    return lam.clip(lower=0.0, upper=1.0)


def compute_r_topo(radius_eff: pd.Series, lambda_edge: pd.Series, edge_fraction: pd.Series) -> pd.Series:
    r = safe_num(radius_eff)
    edge = safe_num(edge_fraction).fillna(0.0).clip(lower=0.0, upper=1.0)
    r_topo = r * (1.0 - lambda_edge * edge)
    return r_topo.clip(lower=1e-9)


def compute_rho_proxy(mass_preferred_sum: pd.Series, radius_like: pd.Series) -> pd.Series:
    m = safe_num(mass_preferred_sum)
    r = safe_num(radius_like)
    v = (4.0 / 3.0) * np.pi * (r ** 3)
    rho = np.where(v > 0, m / v, np.nan)
    return pd.Series(rho)


def derive_background_density(
    master: pd.DataFrame,
    core_stacked: pd.DataFrame,
    background_subset: str,
    lambda_mode: str,
    ref_edge_all: pd.Series,
    ref_edge_core: pd.Series,
) -> float:
    target = core_stacked.copy() if background_subset == "core_stacked" else master.copy()
    ref_series = ref_edge_core if lambda_mode == "core_stacked_median" else ref_edge_all
    lam = derive_lambda_series(target["edge_fraction"], lambda_mode, ref_series)
    r_topo = compute_r_topo(target["radius_eff"], lam, target["edge_fraction"])
    rho = compute_rho_proxy(target["mass_preferred_sum"], r_topo).dropna()
    if len(rho) == 0:
        return np.nan
    return float(rho.median())


def derive_gamma_from_core(
    core_stacked_df: pd.DataFrame,
    rho_bg: float,
    lambda_mode: str,
    gamma_mode: str,
    ref_edge_all: pd.Series,
    ref_edge_core: pd.Series,
) -> tuple[float, pd.Series]:
    ref_series = ref_edge_core if lambda_mode == "core_stacked_median" else ref_edge_all
    lam = derive_lambda_series(core_stacked_df["edge_fraction"], lambda_mode, ref_series)
    r_topo = compute_r_topo(core_stacked_df["radius_eff"], lam, core_stacked_df["edge_fraction"])
    rho_topo = compute_rho_proxy(core_stacked_df["mass_preferred_sum"], r_topo)
    sigma = ((rho_topo - rho_bg) / rho_bg).replace([np.inf, -np.inf], np.nan)
    sigma_clean = sigma.dropna()

    if len(sigma_clean) == 0:
        return 1.0, sigma

    if gamma_mode == "std":
        scale = float(sigma_clean.std(ddof=0))
    elif gamma_mode == "mad":
        med = float(sigma_clean.median())
        scale = float((sigma_clean - med).abs().median())
    else:
        raise ValueError(f"Unknown gamma mode: {gamma_mode}")

    gamma = 1.0 / (scale + EPS)
    return float(gamma), sigma


def add_topological_metrics(
    df: pd.DataFrame,
    rho_bg: float,
    gamma: float,
    lambda_mode: str,
    ref_edge_all: pd.Series,
    ref_edge_core: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    out["radius_eff"] = safe_num(out.get("radius_eff", pd.Series(dtype=float)))
    out["edge_fraction"] = safe_num(out.get("edge_fraction", pd.Series(dtype=float)))
    out["mass_preferred_sum"] = safe_num(out.get("mass_preferred_sum", pd.Series(dtype=float)))
    out["mass_match_count"] = safe_num(out.get("mass_match_count", pd.Series(dtype=float)))
    out["selection_score"] = safe_num(out.get("selection_score", pd.Series(dtype=float)))
    out["mass_quality_score"] = safe_num(out.get("mass_quality_score", pd.Series(dtype=float)))
    out["mass_nearest_sep_deg"] = safe_num(out.get("mass_nearest_sep_deg", pd.Series(dtype=float)))

    ref_series = ref_edge_core if lambda_mode == "core_stacked_median" else ref_edge_all
    out["lambda_edge"] = derive_lambda_series(out["edge_fraction"], lambda_mode, ref_series)

    out["r_topo"] = compute_r_topo(out["radius_eff"], out["lambda_edge"], out["edge_fraction"])
    out["V_topo"] = (4.0 / 3.0) * np.pi * (out["r_topo"] ** 3)

    out["rho_proxy_standard_like"] = compute_rho_proxy(out["mass_preferred_sum"], out["radius_eff"])
    out["rho_proxy_topo"] = compute_rho_proxy(out["mass_preferred_sum"], out["r_topo"])

    out["rho_bg"] = rho_bg
    out["sigma_void"] = np.where(
        np.isfinite(out["rho_proxy_topo"]) & np.isfinite(rho_bg) & (rho_bg != 0),
        (out["rho_proxy_topo"] - rho_bg) / rho_bg,
        np.nan,
    )
    out["D_void"] = 3.0 + gamma * out["sigma_void"]

    out["radius_shrink_fraction"] = np.where(
        out["radius_eff"] > 0,
        1.0 - (out["r_topo"] / out["radius_eff"]),
        np.nan,
    )

    out["topological_quality_flag"] = np.where(
        out["mass_strict_eligible"].fillna(False)
        & out["rho_proxy_topo"].notna()
        & out["sigma_void"].notna(),
        "usable",
        "limited",
    )
    return out


def summarize_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = [{"subset": label, "metric": "rows", "value": len(df)}]
    if len(df) == 0:
        return pd.DataFrame(rows)

    numeric_cols = [
        "radius_eff",
        "edge_fraction",
        "lambda_edge",
        "r_topo",
        "radius_shrink_fraction",
        "V_topo",
        "mass_preferred_sum",
        "rho_bg",
        "rho_proxy_standard_like",
        "rho_proxy_topo",
        "sigma_void",
        "D_void",
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

    for col in ["algorithm", "sky_region", "radius_band", "topological_quality_flag"]:
        if col in df.columns:
            rows.append({"subset": label, "metric": f"{col}_counts", "value": str(df[col].value_counts(dropna=False).to_dict())})

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

    ref_edge_all = safe_num(master["edge_fraction"])
    ref_edge_core = safe_num(core_stacked["edge_fraction"])

    rho_bg = derive_background_density(
        master=master,
        core_stacked=core_stacked,
        background_subset=args.background_subset,
        lambda_mode=args.lambda_mode,
        ref_edge_all=ref_edge_all,
        ref_edge_core=ref_edge_core,
    )

    gamma, core_sigma_for_gamma = derive_gamma_from_core(
        core_stacked_df=core_stacked,
        rho_bg=rho_bg,
        lambda_mode=args.lambda_mode,
        gamma_mode=args.gamma_mode,
        ref_edge_all=ref_edge_all,
        ref_edge_core=ref_edge_core,
    )

    master_topo = add_topological_metrics(master, rho_bg, gamma, args.lambda_mode, ref_edge_all, ref_edge_core)
    core_rep_topo = add_topological_metrics(core_rep, rho_bg, gamma, args.lambda_mode, ref_edge_all, ref_edge_core)
    support_rep_topo = add_topological_metrics(support_rep, rho_bg, gamma, args.lambda_mode, ref_edge_all, ref_edge_core)
    core_stacked_topo = add_topological_metrics(core_stacked, rho_bg, gamma, args.lambda_mode, ref_edge_all, ref_edge_core)
    support_stacked_topo = add_topological_metrics(support_stacked, rho_bg, gamma, args.lambda_mode, ref_edge_all, ref_edge_core)

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
                master_topo, "radius_eff", "sigma_void",
                "Topological baseline 003: radius vs sigma_void",
                "radius_eff", "sigma_void",
                run_dir / "topological_radius_vs_sigma_void.png",
            )
        if master_topo["radius_eff"].notna().any() and master_topo["D_void"].notna().any():
            save_scatter(
                master_topo, "radius_eff", "D_void",
                "Topological baseline 003: radius vs D_void",
                "radius_eff", "D_void",
                run_dir / "topological_radius_vs_D_void.png",
            )
        if master_topo["radius_eff"].notna().any() and master_topo["r_topo"].notna().any():
            save_scatter(
                master_topo, "radius_eff", "r_topo",
                "Topological baseline 003: radius_eff vs r_topo",
                "radius_eff", "r_topo",
                run_dir / "topological_radius_eff_vs_r_topo.png",
            )
        if "algorithm" in master_topo.columns:
            save_bar_counts(master_topo, "algorithm", "Topological baseline 003: algorithm counts", run_dir / "topological_algorithm_counts.png")
        if core_rep_topo["D_void"].notna().any():
            save_rank_plot(
                core_rep_topo, "D_void",
                "Core representatives 003: D_void",
                "D_void",
                run_dir / "topological_core_representatives_D_void.png",
            )
        if core_stacked_topo["sigma_void"].notna().any():
            save_rank_plot(
                core_stacked_topo, "sigma_void",
                "Core stacked 003: sigma_void",
                "sigma_void",
                run_dir / "topological_core_stacked_sigma_void.png",
            )

    lines = []
    lines.append("Cosmic Void Structural Validation topological summary (003 fully derived)")
    lines.append("====================================================================")
    lines.append(f"Project root: {project_root}")
    lines.append(f"Input root: {input_root}")
    lines.append(f"Run output: {run_dir}")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This topological-stage pipeline reads the finalized official input directly and derives lambda_edge, "
        "r_topo, rho_bg, gamma, sigma_void, and D_void from the official input itself. "
        "No standard-stage output file is used as input."
    )
    lines.append("")
    lines.append("Definitions")
    lines.append("-----------")
    lines.append(f"lambda mode = {args.lambda_mode}")
    lines.append(f"gamma mode = {args.gamma_mode}")
    lines.append("r_topo = r_eff * (1 - lambda_edge * edge_fraction)")
    lines.append("V_topo = (4/3) * pi * r_topo^3")
    lines.append("rho_proxy_topo = M_proxy / V_topo")
    lines.append("sigma_void = (rho_proxy_topo - rho_bg) / rho_bg")
    lines.append("D_void = 3 + gamma * sigma_void")
    lines.append(f"background subset = {args.background_subset}")
    lines.append(f"derived gamma = {gamma}")
    lines.append(f"derived rho_bg = {rho_bg}")
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
    lines.append("- topological_radius_eff_vs_r_topo.png")
    lines.append("- topological_algorithm_counts.png")
    lines.append("- topological_core_representatives_D_void.png")
    lines.append("- topological_core_stacked_sigma_void.png")
    lines.append("- topological_summary.txt")
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append(
        "This stage keeps the official input fixed, but replaces fixed lambda and gamma with internally derived quantities. "
        "The resulting D_void is therefore a data-closed topological quantity rather than a fixed-coefficient trial quantity."
    )
    write_text(run_dir / "topological_summary.txt", "\n".join(lines))

    print(f"Input root: {input_root}")
    print(f"Run output: {run_dir}")
    print(f"lambda mode: {args.lambda_mode}")
    print(f"gamma mode: {args.gamma_mode}")
    print(f"derived rho_bg: {rho_bg}")
    print(f"derived gamma: {gamma}")
    print(f"Master rows: {len(master_topo)}")
    print(f"Core representatives: {len(core_rep_topo)}")
    print(f"Support representatives: {len(support_rep_topo)}")
    print(f"Core stacked: {len(core_stacked_topo)}")
    print(f"Support stacked: {len(support_stacked_topo)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
