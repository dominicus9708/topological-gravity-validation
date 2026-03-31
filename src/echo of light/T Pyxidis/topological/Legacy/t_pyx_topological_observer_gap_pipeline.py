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
    w_local: float,
    w_gap: float,
    local_strength: float,
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

    # ----- Standard observables -----
    df["theta_norm"] = normalize_series(df["theta_arcsec"])
    df["z_abs_1e17_cm"] = df["z_1e17_cm"].abs()
    df["z_abs_norm"] = normalize_series(df["z_abs_1e17_cm"])
    df["delay_norm"] = normalize_series(df["delay_time_days"])

    # ----- Theory-aligned reinterpretation -----
    # sigma_local_burst:
    # local low-effective-dimension term caused by the eruptive disturbance itself.
    # Earlier / more directly affected patches are assigned stronger local degradation.
    df["sigma_local_burst"] = local_strength * (1.0 - df["delay_norm"])

    # sigma_observer_gap:
    # inspired by Paper 1: mismatch between the disturbed local structure and
    # the external observer's describability. We approximate this as a mixed term
    # between line-of-sight displacement and projection mismatch.
    df["sigma_observer_gap"] = 0.50 * df["z_abs_norm"] + 0.50 * (df["delay_norm"] - df["theta_norm"]).abs()

    # sigma_total:
    # weighted combination of local disturbance and describability gap
    df["sigma_total"] = w_local * df["sigma_local_burst"] + w_gap * df["sigma_observer_gap"]
    df["Dw_trial_topological"] = 3.0 - df["sigma_total"]

    # delay driver and proxy:
    # not a direct light-echo geometry derivation, but a theory-aligned comparison layer
    df["delay_driver"] = 0.50 * df["sigma_local_burst"] + 0.50 * df["sigma_observer_gap"]
    dmin = df["delay_time_days"].min()
    dmax = df["delay_time_days"].max()
    df["delay_topological_proxy_days"] = dmin + df["delay_driver"] * (dmax - dmin)

    df["angle_deg"] = df["location_label"].map(ANGLE_DEG_MAP)
    df["angle_rad"] = np.deg2rad(df["angle_deg"])

    out_dir = build_output_dir(output_root, object_name, mode_name)
    df.to_csv(out_dir / "t_pyx_topological_observer_gap_patch_comparison.csv", index=False, encoding="utf-8-sig")

    # ----- Graph 1: circular layout -----
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
    ax.set_title("T Pyxidis topological: circular patch layout (observer-gap model)")
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_topological_observer_gap_circular_layout.png", dpi=180)
    plt.close()

    # ----- Graph 2: family-separated standard vs topological -----
    fig, ax = plt.subplots(figsize=(10, 6))
    for family_name, patches in FAMILY_MAP.items():
        sub = df[df["echo_patch"].isin(patches)].sort_values("delay_time_days")
        ax.plot(sub["delay_time_days"], sub["theta_arcsec"], marker="o", label=f"{family_name} standard")
        ax.plot(sub["delay_topological_proxy_days"], sub["theta_arcsec"], marker="o", linestyle="--", label=f"{family_name} topological")
    for _, row in df.iterrows():
        ax.text(row["delay_time_days"], row["theta_arcsec"], f" {row['echo_patch']}", fontsize=9)
    ax.set_title("T Pyxidis: delay vs angular distance, standard vs topological (observer-gap model)")
    ax.set_xlabel("Delay time (days)")
    ax.set_ylabel("Angular distance θ (arcsec)")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_topological_observer_gap_delay_theta_comparison.png", dpi=180)
    plt.close()

    # ----- Graph 3: theory components -----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    ax.plot(x, df["sigma_local_burst"], marker="o", label="sigma_local_burst")
    ax.plot(x, df["sigma_observer_gap"], marker="o", label="sigma_observer_gap")
    ax.plot(x, df["sigma_total"], marker="o", label="sigma_total")
    ax.set_xticks(x)
    ax.set_xticklabels(df["echo_patch"])
    ax.set_title("T Pyxidis: local-burst term vs observer-describability-gap term")
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_topological_observer_gap_components.png", dpi=180)
    plt.close()

    # ----- Graph 4: delay driver breakdown -----
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    ax.plot(x, df["delay_norm"], marker="o", label="standard_delay_norm")
    ax.plot(x, df["delay_driver"], marker="o", label="delay_driver")
    ax.set_xticks(x)
    ax.set_xticklabels(df["echo_patch"])
    ax.set_title("T Pyxidis: standard delay norm vs topological delay driver")
    ax.set_xlabel("Echo patch")
    ax.set_ylabel("Normalized value")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_topological_observer_gap_delay_driver.png", dpi=180)
    plt.close()

    summary = [
        "T Pyxidis topological observer-gap run",
        f"patch_csv={Path(patch_csv)}",
        f"output_dir={out_dir}",
        "",
        "Interpretive structure:",
        "- sigma_local_burst: local low-effective-dimension term generated by the eruption",
        "- sigma_observer_gap: external-observer describability-gap term inspired by Paper 1",
        "- sigma_total = w_local * sigma_local_burst + w_gap * sigma_observer_gap",
        "",
        "Delay proxy structure:",
        "- delay_driver = 0.50 * sigma_local_burst + 0.50 * sigma_observer_gap",
        "- delay_topological_proxy_days rescales delay_driver into the standard delay range",
        "",
        f"Parameter values:",
        f"- w_local = {w_local}",
        f"- w_gap = {w_gap}",
        f"- local_strength = {local_strength}",
        "",
        "Important note:",
        "- This is a theory-aligned reconstruction trial, not a claim of direct derivation from the standard paper.",
        "- It models the local disturbed region near T Pyxidis rather than the entire source-observer path.",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8-sig")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T Pyxidis topological observer-gap model.")
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
        default="topological_observer_gap",
        help="Mode folder name under object folder.",
    )
    parser.add_argument(
        "--w-local",
        type=float,
        default=0.60,
        help="Weight for local burst-induced low-effective-dimension term.",
    )
    parser.add_argument(
        "--w-gap",
        type=float,
        default=0.40,
        help="Weight for observer describability-gap term.",
    )
    parser.add_argument(
        "--local-strength",
        type=float,
        default=1.0,
        help="Strength multiplier for the local burst term.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = run_pipeline(
        patch_csv=args.patch_csv,
        output_root=args.output_root,
        object_name=args.object_name,
        mode_name=args.mode_name,
        w_local=args.w_local,
        w_gap=args.w_gap,
        local_strength=args.local_strength,
    )
    print(f"[DONE] Output written to: {out_dir}")
