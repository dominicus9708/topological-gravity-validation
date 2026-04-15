from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_STANDARD_DIR = Path(
    r"results/Validation of Structural Contrast Baseline/output/standard/20260412_123131"
)
DEFAULT_TOPOLOGICAL_DIR = Path(
    r"results/Validation of Structural Contrast Baseline/output/topological/20260412_123920"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"results/Validation of Structural Contrast Baseline/output/Integration"
)

STANDARD_SUMMARY_CANDIDATES = [
    "standard_summary_common5_v3.csv",
    "standard_summary.csv",
]
TOPOLOGICAL_SUMMARY_CANDIDATES = [
    "topological_summary_common5_v11.csv",
    "topological_summary.csv",
]
OPTIONAL_AUX_CANDIDATES = [
    "standard_baseline_comparison_table.csv",
    "target_standard_zones.csv",
    "run_summary.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrate standard and topological common5 CSV outputs into a single comparison bundle."
        )
    )
    parser.add_argument(
        "--standard-dir",
        type=Path,
        default=DEFAULT_STANDARD_DIR,
        help="Directory containing standard output CSV files.",
    )
    parser.add_argument(
        "--topological-dir",
        type=Path,
        default=DEFAULT_TOPOLOGICAL_DIR,
        help="Directory containing topological output CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory under which a timestamped Integration output folder will be created.",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional fixed timestamp folder name, e.g. 20260415_134500.",
    )
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_first_existing(directory: Path, candidates: Iterable[str]) -> Path | None:
    for name in candidates:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def find_summary_file(directory: Path, preferred_candidates: Iterable[str], keyword: str) -> Path:
    direct = find_first_existing(directory, preferred_candidates)
    if direct is not None:
        return direct

    matches = sorted(
        [
            p
            for p in directory.glob("*.csv")
            if keyword.lower() in p.name.lower() and "summary" in p.name.lower()
        ]
    )
    if not matches:
        raise FileNotFoundError(
            f"No suitable {keyword} summary CSV found in {directory}."
        )
    return matches[0]


def classify_relationship(delta_dw: float, delta_mv: float) -> str:
    if pd.isna(delta_dw) or pd.isna(delta_mv):
        return "insufficient_data"
    if delta_dw >= 0 and delta_mv >= 0:
        if delta_mv > max(abs(delta_dw), 1e-12) * 1.5:
            return "bridge_amplified_shell"
        return "aligned_shell_supported"
    if delta_dw < 0 and delta_mv >= 0:
        return "branching_or_inversion"
    if delta_dw >= 0 and delta_mv < 0:
        return "shell_to_background_reversal"
    return "aligned_background_supported"


def conceptual_alignment_score(delta_dw: float, delta_mv: float) -> float:
    """
    Heuristic score in [0, 1].
    1 means directionally aligned and similar sign.
    This is not a statistical chi-square and should be documented as conceptual only.
    """
    if pd.isna(delta_dw) or pd.isna(delta_mv):
        return float("nan")
    sign_match = 1.0 if (delta_dw == 0 or delta_mv == 0 or math.copysign(1, delta_dw) == math.copysign(1, delta_mv)) else 0.0
    magnitude_ratio = min(abs(delta_dw), abs(delta_mv)) / max(max(abs(delta_dw), abs(delta_mv)), 1e-12)
    return round(0.65 * sign_match + 0.35 * magnitude_ratio, 6)


def conceptual_divergence_index(delta_dw: float, delta_mv: float) -> float:
    """
    Non-statistical divergence proxy for internal reading only.
    Larger means more divergence between non-mass structural contrast and mass-volume contrast.
    """
    if pd.isna(delta_dw) or pd.isna(delta_mv):
        return float("nan")
    return round(abs(delta_dw - delta_mv), 6)


def load_and_prepare(standard_path: Path, topological_path: Path) -> pd.DataFrame:
    standard = pd.read_csv(standard_path).copy()
    topological = pd.read_csv(topological_path).copy()

    standard = standard.rename(
        columns={
            "Dw_obs_inner": "std_Dw_obs_inner",
            "Dw_obs_shell": "std_Dw_obs_shell",
            "Dw_obs_background": "std_Dw_obs_background",
            "radius_arcsec": "std_radius_arcsec",
            "profile_file": "std_profile_file",
        }
    )
    topological = topological.rename(
        columns={
            "Dw_obs_inner": "topo_Dw_obs_inner",
            "Dw_obs_shell": "topo_Dw_obs_shell",
            "Dw_obs_background": "topo_Dw_obs_background",
            "radius_arcsec": "topo_radius_arcsec",
            "profile_file": "topo_profile_file",
        }
    )

    merged = pd.merge(standard, topological, on="wise_name", how="outer", validate="one_to_one")

    merged["std_delta_dw_shell_minus_bg"] = (
        merged["std_Dw_obs_shell"] - merged["std_Dw_obs_background"]
    )
    merged["topo_delta_dw_shell_minus_bg"] = (
        merged["topo_Dw_obs_shell"] - merged["topo_Dw_obs_background"]
    )
    merged["delta_mv_shell_minus_bg"] = (
        merged["shell_mass_volume_density"] - merged["background_mass_volume_density"]
    )

    # In current common5, structural base is shared. Keep both for audit but use standard delta for comparison axis.
    merged["comparison_delta_dw"] = merged["std_delta_dw_shell_minus_bg"]

    merged["relationship_class"] = merged.apply(
        lambda row: classify_relationship(
            row["comparison_delta_dw"], row["delta_mv_shell_minus_bg"]
        ),
        axis=1,
    )
    merged["conceptual_alignment_score"] = merged.apply(
        lambda row: conceptual_alignment_score(
            row["comparison_delta_dw"], row["delta_mv_shell_minus_bg"]
        ),
        axis=1,
    )
    merged["conceptual_divergence_index"] = merged.apply(
        lambda row: conceptual_divergence_index(
            row["comparison_delta_dw"], row["delta_mv_shell_minus_bg"]
        ),
        axis=1,
    )

    merged = merged.sort_values(by="wise_name").reset_index(drop=True)
    return merged


def maybe_merge_auxiliary(merged: pd.DataFrame, standard_dir: Path) -> pd.DataFrame:
    baseline_table = standard_dir / "standard_baseline_comparison_table.csv"
    zones_table = standard_dir / "target_standard_zones.csv"

    out = merged.copy()
    if baseline_table.exists():
        baseline_df = pd.read_csv(baseline_table)
        keep = [c for c in [
            "wise_name", "hii_region_name", "catalog_class", "region_bucket",
            "priority_rank", "score_total", "radius_arcsec"
        ] if c in baseline_df.columns]
        out = out.merge(baseline_df[keep], on="wise_name", how="left", suffixes=("", "_baseline_meta"))

    if zones_table.exists():
        zones_df = pd.read_csv(zones_table)
        zone_cols = [c for c in zones_df.columns if c != "radius_arcsec"]
        out = out.merge(zones_df[zone_cols], on="wise_name", how="left")

    return out


def save_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    ordered_columns = [
        "wise_name",
        "hii_region_name",
        "catalog_class",
        "priority_rank",
        "region_bucket",
        "std_radius_arcsec",
        "topo_radius_arcsec",
        "std_Dw_obs_inner",
        "std_Dw_obs_shell",
        "std_Dw_obs_background",
        "comparison_delta_dw",
        "topo_Dw_obs_inner",
        "topo_Dw_obs_shell",
        "topo_Dw_obs_background",
        "topo_delta_dw_shell_minus_bg",
        "shell_mass_volume_density",
        "background_mass_volume_density",
        "delta_mv_shell_minus_bg",
        "sigma_mass_volume",
        "bridge_status",
        "interpretation_flag",
        "relationship_class",
        "conceptual_alignment_score",
        "conceptual_divergence_index",
        "mass_proxy",
        "mass_proxy_kind",
        "radius_scale_pc",
        "radius_scale_kind",
        "score_total",
        "std_profile_file",
        "topo_profile_file",
    ]
    final_cols = [c for c in ordered_columns if c in df.columns]
    df[final_cols].to_csv(output_dir / "integration_common5_comparison_summary.csv", index=False)

    plot_cols = [
        "wise_name",
        "comparison_delta_dw",
        "delta_mv_shell_minus_bg",
        "relationship_class",
        "conceptual_alignment_score",
        "conceptual_divergence_index",
    ]
    df[plot_cols].to_csv(output_dir / "integration_common5_plot_values.csv", index=False)


def make_bar_plot(df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = df[["wise_name", "comparison_delta_dw", "delta_mv_shell_minus_bg"]].copy()
    x = range(len(plot_df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar([i - width / 2 for i in x], plot_df["comparison_delta_dw"], width=width, label="Non-mass contrast: ΔDw (shell - bg)")
    ax.bar([i + width / 2 for i in x], plot_df["delta_mv_shell_minus_bg"], width=width, label="Mass-volume contrast: ΔMVdens (shell - bg)")
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["wise_name"], rotation=25, ha="right")
    ax.set_ylabel("Contrast difference")
    ax.set_title("Common5 comparison: non-mass structural contrast vs mass-volume contrast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "integration_common5_dual_delta_bar.png", dpi=200)
    plt.close(fig)


def make_scatter_plot(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axhline(0.0, linewidth=1)
    ax.axvline(0.0, linewidth=1)

    for _, row in df.iterrows():
        x = row["comparison_delta_dw"]
        y = row["delta_mv_shell_minus_bg"]
        ax.scatter([x], [y], s=70)
        ax.annotate(row["wise_name"], (x, y), xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Non-mass contrast: ΔDw (shell - bg)")
    ax.set_ylabel("Mass-volume contrast: ΔMVdens (shell - bg)")
    ax.set_title("Common5 comparison map: aligned, branching, and bridge-amplified cases")
    fig.tight_layout()
    fig.savefig(output_dir / "integration_common5_quadrant_scatter.png", dpi=200)
    plt.close(fig)


def write_summary(df: pd.DataFrame, output_dir: Path, standard_path: Path, topological_path: Path) -> None:
    n_total = len(df)
    class_counts = df["relationship_class"].value_counts(dropna=False).to_dict()

    lowest_div = df.sort_values("conceptual_divergence_index").head(1)
    highest_div = df.sort_values("conceptual_divergence_index", ascending=False).head(1)

    lines = []
    lines.append("Integration summary for common5 standard-topological comparison")
    lines.append("=" * 72)
    lines.append(f"standard summary input: {standard_path}")
    lines.append(f"topological summary input: {topological_path}")
    lines.append(f"target count: {n_total}")
    lines.append("")
    lines.append("Relationship class counts")
    lines.append("-" * 72)
    for key, value in class_counts.items():
        lines.append(f"{key}: {value}")
    lines.append("")

    if not lowest_div.empty:
        row = lowest_div.iloc[0]
        lines.append("Most aligned case under the conceptual comparison index")
        lines.append("-" * 72)
        lines.append(
            f"{row['wise_name']}: ΔDw={row['comparison_delta_dw']:.6f}, "
            f"ΔMVdens={row['delta_mv_shell_minus_bg']:.6f}, "
            f"class={row['relationship_class']}"
        )
        lines.append("")

    if not highest_div.empty:
        row = highest_div.iloc[0]
        lines.append("Most divergent case under the conceptual comparison index")
        lines.append("-" * 72)
        lines.append(
            f"{row['wise_name']}: ΔDw={row['comparison_delta_dw']:.6f}, "
            f"ΔMVdens={row['delta_mv_shell_minus_bg']:.6f}, "
            f"class={row['relationship_class']}"
        )
        lines.append("")

    lines.append("Important note")
    lines.append("-" * 72)
    lines.append(
        "The conceptual_alignment_score and conceptual_divergence_index written by this pipeline "
        "are internal descriptive quantities only. They are not statistical chi-square values and "
        "should not be presented as formal fit statistics in the paper."
    )
    lines.append("")

    (output_dir / "integration_common5_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    ensure_exists(args.standard_dir, "standard directory")
    ensure_exists(args.topological_dir, "topological directory")
    ensure_exists(args.output_root, "output root")

    timestamp = args.timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_summary = find_summary_file(args.standard_dir, STANDARD_SUMMARY_CANDIDATES, "standard")
    topological_summary = find_summary_file(args.topological_dir, TOPOLOGICAL_SUMMARY_CANDIDATES, "topological")

    merged = load_and_prepare(standard_summary, topological_summary)
    merged = maybe_merge_auxiliary(merged, args.standard_dir)

    save_csvs(merged, output_dir)
    make_bar_plot(merged, output_dir)
    make_scatter_plot(merged, output_dir)
    write_summary(merged, output_dir, standard_summary, topological_summary)

    print(f"[OK] Integration output written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
