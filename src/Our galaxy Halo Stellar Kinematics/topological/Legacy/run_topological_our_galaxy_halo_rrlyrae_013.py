from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SHELL_LABELS = ["5-10", "10-20", "20-40", "40-80", "80+"]
SHELL_CENTERS = {"5-10": 7.5, "10-20": 15.0, "20-40": 30.0, "40-80": 60.0, "80+": 100.0}


def find_project_root(start_file: Path) -> Path:
    current = start_file.resolve()
    for parent in current.parents:
        if (parent / "data").exists() and (parent / "results").exists():
            return parent
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return start_file.resolve().parents[4]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def safe_numeric(series):
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def timestamp_folder_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def shell_x_and_labels(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    shells = df["shell"].astype(str).tolist()
    x = np.array([SHELL_CENTERS.get(s, np.nan) for s in shells], dtype=float)
    return x, shells


def latest_topological_output_dir(project_root: Path) -> Path:
    topological_root = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "topological"
    if not topological_root.exists():
        raise FileNotFoundError(f"Topological output root not found: {topological_root}")
    subdirs = [p for p in topological_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No timestamped topological output folders found under: {topological_root}")
    subdirs_sorted = sorted(subdirs, key=lambda p: p.name)
    return subdirs_sorted[-1]


def plot_variance_comparison(shell_df: pd.DataFrame, dataset_name: str, outpath: Path) -> None:
    x, labels = shell_x_and_labels(shell_df)
    obs = safe_numeric(shell_df.get("observed_variance_shell"))
    std = safe_numeric(shell_df.get("standard_variance_shell"))
    resid = safe_numeric(shell_df.get("residual_observed_minus_standard"))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(x, obs, marker="o", label="Observed variance")
    axes[0].plot(x, std, marker="s", label="Standard variance")
    axes[0].set_ylabel("Variance")
    axes[0].set_title(f"{dataset_name.upper()} shell-wise observed vs standard variance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, resid, marker="o", label="Residual (observed - standard)")
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Shell center [kpc]")
    axes[1].set_title(f"{dataset_name.upper()} shell-wise residual")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_structural_response(shell_df: pd.DataFrame, dataset_name: str, outpath: Path) -> None:
    x, labels = shell_x_and_labels(shell_df)
    d_halo = safe_numeric(shell_df.get("D_halo_shell"))
    sigma_bg = safe_numeric(shell_df.get("Sigma_bg_shell"))
    sigma_spread = safe_numeric(shell_df.get("sigma_local_spread_shell"))
    sigma_eff = safe_numeric(shell_df.get("Sigma_shell_effective"))
    grad_eff = safe_numeric(shell_df.get("topological_gradient_term_effective"))

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    axes[0].plot(x, d_halo, marker="o", label="D_halo_shell")
    axes[0].plot(x, sigma_bg, marker="s", label="Sigma_bg_shell")
    axes[0].set_ylabel("Structure level")
    axes[0].set_title(f"{dataset_name.upper()} structural background response")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, sigma_spread, marker="o", label="sigma_local_spread_shell")
    axes[1].plot(x, sigma_eff, marker="s", label="Sigma_shell_effective")
    axes[1].set_ylabel("Spread / effective contrast")
    axes[1].set_title(f"{dataset_name.upper()} shell-local spread and effective contrast")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, grad_eff, marker="o", label="topological_gradient_term_effective")
    axes[2].set_ylabel("Effective gradient")
    axes[2].set_xlabel("Shell center [kpc]")
    axes[2].set_title(f"{dataset_name.upper()} shell-wise effective topological gradient")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_halo_complexity(shell_df: pd.DataFrame, dataset_name: str, outpath: Path) -> None:
    x, labels = shell_x_and_labels(shell_df)
    n_points = safe_numeric(shell_df.get("n_stream_points_overlap")).fillna(0)
    n_streams = safe_numeric(shell_df.get("n_unique_streams_overlap")).fillna(0)
    richness = shell_df.get("stream_richness_flag")
    caution = shell_df.get("shell_caution_label")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].bar(x - 1.0, n_points, width=2.0, label="n_stream_points_overlap")
    axes[0].bar(x + 1.0, n_streams, width=2.0, label="n_unique_streams_overlap")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{dataset_name.upper()} shell-wise halo complexity diagnostics")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    richness_num = []
    caution_num = []
    richness_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
    caution_map = {"sparse_shell": 0, "nominal_shell": 1, "stream_moderate_shell": 2, "stream_rich_shell": 3}

    for r in richness.astype(str).tolist():
        richness_num.append(richness_map.get(r, np.nan))
    for c in caution.astype(str).tolist():
        caution_num.append(caution_map.get(c, np.nan))

    axes[1].plot(x, richness_num, marker="o", label="stream_richness_flag")
    axes[1].plot(x, caution_num, marker="s", label="shell_caution_label")
    axes[1].set_ylabel("Diagnostic level")
    axes[1].set_xlabel("Shell center [kpc]")
    axes[1].set_title(f"{dataset_name.upper()} shell diagnostic labels")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(["0", "1", "2", "3"])

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readme(path: Path, source_dir: Path, plot_dir: Path) -> None:
    text = (
        "Our galaxy Halo Stellar Kinematics - topological stage 013 plots\n\n"
        f"Source topological result folder\n- {source_dir}\n\n"
        f"Plot folder\n- {plot_dir}\n\n"
        "Generated figures\n"
        "- 5d_variance_comparison.png\n"
        "- 5d_structural_response.png\n"
        "- 5d_halo_complexity.png\n"
        "- 6d_variance_comparison.png\n"
        "- 6d_structural_response.png\n"
        "- 6d_halo_complexity.png\n\n"
        "Interpretation note\n"
        "- Variance comparison plots show observed vs standard baseline and residual.\n"
        "- Structural response plots show D_halo_shell, Sigma_bg_shell, sigma_local_spread_shell, "
        "Sigma_shell_effective, and effective topological gradient.\n"
        "- Halo complexity plots show stream overlap counts and shell diagnostic labels.\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    project_root = find_project_root(Path(__file__))
    source_dir = latest_topological_output_dir(project_root)
    plot_dir = source_dir / "plot"
    ensure_dir(plot_dir)

    path_5d = source_dir / "gaia_rrlyrae_5d_topological_shells.csv"
    path_6d = source_dir / "gaia_rrlyrae_6d_topological_shells.csv"

    if not path_5d.exists():
        raise FileNotFoundError(f"5D topological shell CSV not found: {path_5d}")
    if not path_6d.exists():
        raise FileNotFoundError(f"6D topological shell CSV not found: {path_6d}")

    shell_5d = read_csv(path_5d)
    shell_6d = read_csv(path_6d)

    plot_variance_comparison(shell_5d, "5d", plot_dir / "5d_variance_comparison.png")
    plot_structural_response(shell_5d, "5d", plot_dir / "5d_structural_response.png")
    plot_halo_complexity(shell_5d, "5d", plot_dir / "5d_halo_complexity.png")

    plot_variance_comparison(shell_6d, "6d", plot_dir / "6d_variance_comparison.png")
    plot_structural_response(shell_6d, "6d", plot_dir / "6d_structural_response.png")
    plot_halo_complexity(shell_6d, "6d", plot_dir / "6d_halo_complexity.png")

    write_readme(source_dir / "README_plots.txt", source_dir, plot_dir)

    print("[DONE] Saved:")
    print(f" - {plot_dir / '5d_variance_comparison.png'}")
    print(f" - {plot_dir / '5d_structural_response.png'}")
    print(f" - {plot_dir / '5d_halo_complexity.png'}")
    print(f" - {plot_dir / '6d_variance_comparison.png'}")
    print(f" - {plot_dir / '6d_structural_response.png'}")
    print(f" - {plot_dir / '6d_halo_complexity.png'}")
    print(f" - {source_dir / 'README_plots.txt'}")


if __name__ == "__main__":
    main()
