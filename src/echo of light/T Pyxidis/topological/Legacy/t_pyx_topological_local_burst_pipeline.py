from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANGLE_DEG_MAP = {
    "north": 90,
    "north #2": 75,
    "south": 270,
    "south #2": 255,
    "northwest": 135,
    "west": 180,
}

FAMILY_MAP = {
    "North family": ["N1", "N2"],
    "South family": ["S1", "S2"],
    "West family": ["NW", "W"],
}


def build_output_dir(root: str | Path, object_name: str, mode_name: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(root) / object_name / mode_name / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - s_min) / (s_max - s_min)


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def run_pipeline(
    patch_csv: str | Path,
    output_root: str | Path,
    object_name: str,
    mode_name: str,
    w_obs: float,
    w_burst: float,
    burst_strength: float,
) -> Path:
    df = pd.read_csv(patch_csv, encoding="utf-8-sig")
    validate_columns(
        df,
        [
            "echo_patch",
            "location_label",
            "delay_time_days",
            "theta_arcsec",
            "z_1e17_cm",
        ],
        "patch CSV",
    )

    for col in ["delay_time_days", "theta_arcsec", "z_1e17_cm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ----- Standard geometric observables -----
    df["z_abs_1e17_cm"] = df["z_1e17_cm"].abs()
    df["theta_norm"] = normalize_series(df["theta_arcsec"])
    df["z_abs_norm"] = normalize_series(df["z_abs_1e17_cm"])
    df["delay_norm"] = normalize_series(df["delay_time_days"])

    # ----- Observer-side structural distortion term -----
    # Reuses the existing observational geometry term.
    df["sigma_obs"] = 0.60 * df["z_abs_norm"] + 0.40 * (1.0 - df["theta_norm"])

    # ----- Local burst-induced low-effective-dimension term -----
    # Physical interpretation:
    # earlier / more directly disturbed echo regions are assumed to carry stronger
    # local structural degradation from the nova event itself.
    # This does NOT model the entire source-observer path.
    df["sigma_burst_local"] = burst_strength * (1.0 - df["delay_norm"])

    # ----- Total topological structural deviation -----
    df["sigma_total"] = w_obs * df["sigma_obs"] + w_burst * df["sigma_burst_local"]
    df["Dw_trial_topological"] = 3.0 - df["sigma_total"]

    # Delay proxy rescaled to the standard delay range
    delay_min = df["delay_time_days"].min()
    delay_max = df["delay_time_days"].max()
    df["delay_topological_proxy_days"] = delay_min + df["sigma_total"] * (delay_max - delay_min)

    # Visual placement
    df["angle_deg"] = df["location_label"].map(ANGLE_DEG_MAP)
    df["angle_rad"] = np.deg2rad(df["angle_deg"])

    out_dir = build_output_dir(output_root, object_name, mode_name)
    df.to_csv(out_dir / "t_pyx_topological_local_burst_patch_comparison.csv", index=False, encoding="utf-8-sig")

    # ----- Graph 1: Circular layout with local-burst Dw -----
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    sc = ax.scatter(df["angle_rad"], df["theta_arcsec"], c=df["Dw_trial_topological"], s=95)
    for _, row in df.iterrows():
        ax.text(
            row["angle_rad"],
            row["theta_arcsec"] + 0.18,
            f"{row['echo_patch']}\nDw={row['Dw_trial_topological']:.2f}",
            ha="center",
            va="center",
            fontsize=9,
        )

    for _, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].sort_values("delay_time_days")
        ax.plot(sub["angle_rad"], sub["theta_arcsec"], linewidth=1.2)

    plt.colorbar(sc, pad=0.1, label="Dw_trial_topological")
    ax.set_title("T Pyxidis topological: circular patch layout (local burst term included)")
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_topological_local_burst_circular_layout.png", dpi=180)
    plt.close()

    # ----- Graph 2: delay-vs-theta family-separated standard vs topological -----
    fig, ax = plt.subplots(figsize=(10, 6))
    for family_name, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].sort_values("delay_time_days")
        ax.plot(sub["delay_time_days"], sub["theta_arcsec"], marker="o", label=f"{family_name} standard")
        ax.plot(sub["delay_topological_proxy_days"], sub["theta_arcsec"], marker="o", linestyle="--", label=f"{family_name} topological")
    for _, row in df.iterrows():
        ax.text(row["delay_time_days"], row["theta_arcsec"], f" {row['echo_patch']}", fontsize=9)
    ax.set_title("T Pyxidis: delay vs angular distance, standard vs topological (local burst term)")
    ax.set_xlabel("Delay time (days)")
    ax.set_ylabel("Angular distance θ (arcsec)")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_topological_local_burst_delay_theta_comparison.png", dpi=180)
    plt.close()

    # ----- Graph 3: component comparison -----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    ax.plot(x, df["sigma_obs"], marker="o", label="sigma_obs")
    ax.plot(x, df["sigma_burst_local"], marker="o", label="sigma_burst_local")
    ax.plot(x, df["sigma_total"], marker="o", label="sigma_total")
    ax.set_xticks(x)
    ax.set_xticklabels(df["echo_patch"])
    ax.set_title("T Pyxidis: observer-side vs local-burst structural terms")
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_topological_local_burst_components.png", dpi=180)
    plt.close()

    summary = [
        "T Pyxidis topological local-burst run",
        f"patch_csv={Path(patch_csv)}",
        f"output_dir={out_dir}",
        "",
        "Interpretive structure:",
        "- sigma_obs captures the observer-side geometric distortion term from theta and |z|",
        "- sigma_burst_local captures the local burst-induced low-effective-dimension term",
        "- sigma_total = w_obs * sigma_obs + w_burst * sigma_burst_local",
        "",
        f"Parameter values:",
        f"- w_obs = {w_obs}",
        f"- w_burst = {w_burst}",
        f"- burst_strength = {burst_strength}",
        "",
        "Important note:",
        "- This model does NOT include the entire source-observer path.",
        "- It only adds a local burst-induced low-effective-dimension term near T Pyxidis.",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8-sig")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T Pyxidis topological model with local burst term.")
    parser.add_argument(
        "--patch-csv",
        default=r"data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv",
        help="Path to the T Pyx standard patch CSV.",
    )
    parser.add_argument(
        "--output-root",
        default=r"results\Echo of Light\output",
        help="Root output directory.",
    )
    parser.add_argument(
        "--object-name",
        default="T Pyxidis",
        help="Object folder name under output root.",
    )
    parser.add_argument(
        "--mode-name",
        default="topological_local_burst",
        help="Mode folder name under object folder.",
    )
    parser.add_argument(
        "--w-obs",
        type=float,
        default=0.65,
        help="Weight for observer-side geometric structural term.",
    )
    parser.add_argument(
        "--w-burst",
        type=float,
        default=0.35,
        help="Weight for local burst-induced low-effective-dimension term.",
    )
    parser.add_argument(
        "--burst-strength",
        type=float,
        default=1.0,
        help="Strength multiplier for local burst term.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = run_pipeline(
        patch_csv=args.patch_csv,
        output_root=args.output_root,
        object_name=args.object_name,
        mode_name=args.mode_name,
        w_obs=args.w_obs,
        w_burst=args.w_burst,
        burst_strength=args.burst_strength,
    )
    print(f"[DONE] Output written to: {out_dir}")
