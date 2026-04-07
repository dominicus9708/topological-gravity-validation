from __future__ import annotations

from pathlib import Path
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


def latest_topological_output_dir(project_root: Path) -> Path:
    root = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "topological"
    if not root.exists():
        raise FileNotFoundError(f"Topological output root not found: {root}")
    subdirs = [p for p in root.iterdir() if p.is_dir() and p.name.lower() != "legacy"]
    if not subdirs:
        raise FileNotFoundError(f"No timestamped topological output folders found under: {root}")
    return sorted(subdirs, key=lambda p: p.name)[-1]


def shell_positions(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    shells = df["shell"].astype(str).tolist()
    x = np.array([SHELL_CENTERS.get(s, np.nan) for s in shells], dtype=float)
    return x, shells


def lollipop(ax, x: np.ndarray, y: pd.Series, label: str, marker: str = "o") -> None:
    ys = safe_numeric(y).to_numpy()
    m = np.isfinite(x) & np.isfinite(ys)
    if not m.any():
        return
    ax.vlines(x[m], 0, ys[m], linewidth=1.0, alpha=0.7)
    ax.scatter(x[m], ys[m], s=45, marker=marker, label=label)


def point_series(ax, x: np.ndarray, y: pd.Series, label: str, marker: str = "o") -> None:
    ys = safe_numeric(y).to_numpy()
    m = np.isfinite(x) & np.isfinite(ys)
    if not m.any():
        return
    ax.scatter(x[m], ys[m], s=45, marker=marker, label=label)


def bar_series(ax, x: np.ndarray, y1: pd.Series, y2: pd.Series, label1: str, label2: str) -> None:
    v1 = safe_numeric(y1).fillna(0).to_numpy()
    v2 = safe_numeric(y2).fillna(0).to_numpy()
    m = np.isfinite(x)
    x = x[m]
    v1 = v1[m]
    v2 = v2[m]
    if len(x) == 0:
        return
    ax.bar(x - 1.2, v1, width=2.2, label=label1)
    ax.bar(x + 1.2, v2, width=2.2, label=label2)


def category_tile_plot(ax, x: np.ndarray, shells: list[str], richness: pd.Series, caution: pd.Series) -> None:
    richness_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
    caution_map = {"sparse_shell": 0, "nominal_shell": 1, "stream_moderate_shell": 2, "stream_rich_shell": 3}

    y_rich = [richness_map.get(str(v), np.nan) for v in richness.tolist()]
    y_caution = [caution_map.get(str(v), np.nan) for v in caution.tolist()]

    mr = np.isfinite(x) & np.isfinite(np.array(y_rich, dtype=float))
    mc = np.isfinite(x) & np.isfinite(np.array(y_caution, dtype=float))

    if mr.any():
        ax.scatter(x[mr], np.array(y_rich, dtype=float)[mr], s=70, marker="o", label="stream_richness_flag")
    if mc.any():
        ax.scatter(x[mc], np.array(y_caution, dtype=float)[mc], s=70, marker="s", label="shell_caution_label")

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3"])
    ax.set_xticks(x)
    ax.set_xticklabels(shells)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_variance_comparison(shell_df: pd.DataFrame, dataset_name: str, outpath: Path) -> None:
    x, shells = shell_positions(shell_df)
    obs = safe_numeric(shell_df.get("observed_variance_shell"))
    std = safe_numeric(shell_df.get("standard_variance_shell"))
    resid = safe_numeric(shell_df.get("residual_observed_minus_standard"))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    point_series(axes[0], x, obs, "Observed variance", marker="o")
    point_series(axes[0], x, std, "Standard variance", marker="s")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Variance (log)")
    axes[0].set_title(f"{dataset_name.upper()} shell-wise observed vs standard variance")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    point_series(axes[1], x, resid, "Residual (observed - standard)", marker="o")
    axes[1].axhline(0.0, linewidth=1)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Shell center [kpc]")
    axes[1].set_title(f"{dataset_name.upper()} shell-wise residual")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(shells)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_structural_response(shell_df: pd.DataFrame, dataset_name: str, outpath: Path) -> None:
    x, shells = shell_positions(shell_df)
    d_halo = safe_numeric(shell_df.get("D_halo_shell"))
    sigma_bg = safe_numeric(shell_df.get("Sigma_bg_shell"))
    sigma_spread = safe_numeric(shell_df.get("sigma_local_spread_shell"))
    sigma_eff = safe_numeric(shell_df.get("Sigma_shell_effective"))
    grad_eff = safe_numeric(shell_df.get("topological_gradient_term_effective"))

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    lollipop(axes[0], x, d_halo, "D_halo_shell", marker="o")
    lollipop(axes[0], x, sigma_bg, "Sigma_bg_shell", marker="s")
    axes[0].set_ylabel("Structure level")
    axes[0].set_title(f"{dataset_name.upper()} structural background response")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    lollipop(axes[1], x, sigma_spread, "sigma_local_spread_shell", marker="o")
    lollipop(axes[1], x, sigma_eff, "Sigma_shell_effective", marker="s")
    axes[1].set_ylabel("Spread / effective contrast")
    axes[1].set_title(f"{dataset_name.upper()} shell-local spread and effective contrast")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    lollipop(axes[2], x, grad_eff, "topological_gradient_term_effective", marker="o")
    axes[2].set_ylabel("Effective gradient")
    axes[2].set_xlabel("Shell center [kpc]")
    axes[2].set_title(f"{dataset_name.upper()} shell-wise effective topological gradient")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(shells)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_halo_complexity(shell_df: pd.DataFrame, dataset_name: str, outpath: Path) -> None:
    x, shells = shell_positions(shell_df)
    n_points = safe_numeric(shell_df.get("n_stream_points_overlap"))
    n_streams = safe_numeric(shell_df.get("n_unique_streams_overlap"))
    richness = shell_df.get("stream_richness_flag", pd.Series(["none"] * len(shell_df)))
    caution = shell_df.get("shell_caution_label", pd.Series(["nominal_shell"] * len(shell_df)))

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    bar_series(
        axes[0], x, n_points, n_streams,
        "n_stream_points_overlap", "n_unique_streams_overlap"
    )
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"{dataset_name.upper()} shell-wise halo complexity diagnostics")
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend()

    category_tile_plot(axes[1], x, shells, richness, caution)
    axes[1].set_ylabel("Diagnostic level")
    axes[1].set_xlabel("Shell center [kpc]")
    axes[1].set_title(f"{dataset_name.upper()} shell diagnostic labels")

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_readme(path: Path, source_dir: Path, plot_dir: Path) -> None:
    text = (
        "Our galaxy Halo Stellar Kinematics - topological stage 014 plot refresh\n\n"
        f"Source folder\n- {source_dir}\n\n"
        f"Plot folder\n- {plot_dir}\n\n"
        "Plot design changes\n"
        "- Variance comparison uses point plots with log scale.\n"
        "- Structural response uses lollipop plots.\n"
        "- Halo complexity uses bar plots plus categorical marker plots.\n"
        "- Continuous line connections were intentionally removed to avoid misleading continuity across sparse shells.\n"
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

    plot_variance_comparison(shell_5d, "5d", plot_dir / "5d_variance_comparison_v2.png")
    plot_structural_response(shell_5d, "5d", plot_dir / "5d_structural_response_v2.png")
    plot_halo_complexity(shell_5d, "5d", plot_dir / "5d_halo_complexity_v2.png")

    plot_variance_comparison(shell_6d, "6d", plot_dir / "6d_variance_comparison_v2.png")
    plot_structural_response(shell_6d, "6d", plot_dir / "6d_structural_response_v2.png")
    plot_halo_complexity(shell_6d, "6d", plot_dir / "6d_halo_complexity_v2.png")

    write_readme(source_dir / "README_plots_v2.txt", source_dir, plot_dir)

    print("[DONE] Saved:")
    print(f" - {plot_dir / '5d_variance_comparison_v2.png'}")
    print(f" - {plot_dir / '5d_structural_response_v2.png'}")
    print(f" - {plot_dir / '5d_halo_complexity_v2.png'}")
    print(f" - {plot_dir / '6d_variance_comparison_v2.png'}")
    print(f" - {plot_dir / '6d_structural_response_v2.png'}")
    print(f" - {plot_dir / '6d_halo_complexity_v2.png'}")
    print(f" - {source_dir / 'README_plots_v2.txt'}")


if __name__ == "__main__":
    main()
