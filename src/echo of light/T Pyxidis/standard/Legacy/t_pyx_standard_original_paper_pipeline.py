from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_output_dir(root: str | Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_pipeline(observation_csv: str | Path, echo_patch_csv: str | Path, output_dir: str | Path) -> Path:
    obs = pd.read_csv(observation_csv, encoding="utf-8-sig")
    patches = pd.read_csv(echo_patch_csv, encoding="utf-8-sig")

    validate_columns(
        obs,
        ["epoch", "instrument", "date", "day_since_outburst", "filters_or_gratings", "exposure_summary_s", "source_table", "notes"],
        "observation CSV",
    )
    validate_columns(
        patches,
        [
            "echo_patch",
            "location_label",
            "delay_time_days",
            "delay_time_err_plus_days",
            "delay_time_err_minus_days",
            "theta_arcsec",
            "z_1e17_cm",
            "z_err_1e17_cm",
            "hemisphere_group",
            "source_table",
            "notes",
        ],
        "echo patch CSV",
    )

    out_dir = build_output_dir(output_dir)
    obs.to_csv(out_dir / "t_pyx_standard_observation_epochs_processed.csv", index=False, encoding="utf-8-sig")
    patches.to_csv(out_dir / "t_pyx_standard_echo_patches_processed.csv", index=False, encoding="utf-8-sig")

    plot_df = patches.sort_values("delay_time_days").copy()

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        plot_df["delay_time_days"],
        plot_df["theta_arcsec"],
        xerr=[plot_df["delay_time_err_minus_days"], plot_df["delay_time_err_plus_days"]],
        fmt="o",
    )
    for _, row in plot_df.iterrows():
        plt.text(row["delay_time_days"], row["theta_arcsec"], f" {row['echo_patch']}")
    plt.title("T Pyxidis standard: echo patch angle vs delay time")
    plt.xlabel("Delay time (days)")
    plt.ylabel("Angular distance θ (arcsec)")
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_standard_theta_vs_delay.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        plot_df["theta_arcsec"],
        plot_df["z_1e17_cm"],
        yerr=plot_df["z_err_1e17_cm"],
        fmt="o",
    )
    for _, row in plot_df.iterrows():
        plt.text(row["theta_arcsec"], row["z_1e17_cm"], f" {row['echo_patch']}")
    plt.axhline(0.0, linewidth=1)
    plt.title("T Pyxidis standard: foreground distance z vs angular distance θ")
    plt.xlabel("Angular distance θ (arcsec)")
    plt.ylabel("Foreground distance z (10^17 cm)")
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_standard_z_vs_theta.png", dpi=150)
    plt.close()

    obs_plot = obs.sort_values("day_since_outburst").copy()
    obs_plot["y"] = 1
    plt.figure(figsize=(10, 3.8))
    plt.scatter(obs_plot["day_since_outburst"], obs_plot["y"])
    for _, row in obs_plot.iterrows():
        plt.text(row["day_since_outburst"], row["y"], f" E{int(row['epoch'])}", va="bottom")
    plt.yticks([])
    plt.title("T Pyxidis standard: HST observation timeline")
    plt.xlabel("Days since outburst")
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_standard_observation_timeline.png", dpi=150)
    plt.close()

    summary_lines = [
        "T Pyx standard pipeline run (original-paper extracted values)",
        f"observation_csv={Path(observation_csv)}",
        f"echo_patch_csv={Path(echo_patch_csv)}",
        f"processed_observation_rows={len(obs)}",
        f"processed_echo_patch_rows={len(patches)}",
        "",
        "This standard run is based on directly extracted values from Sokoloski et al. 2013:",
        "- Table 1 for WFC3 observation epochs",
        "- Table 2 for echo patch delay times, angular distances, and foreground distances",
        "",
        "This is distinct from the earlier representative-scale skeleton/baseline attempt.",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8-sig")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T Pyx standard pipeline from original-paper extracted values.")
    parser.add_argument(
        "--observation-csv",
        default=r"data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_observation_epochs_original_paper.csv",
        help="Path to observation epochs CSV.",
    )
    parser.add_argument(
        "--echo-patch-csv",
        default=r"data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv",
        help="Path to extracted echo patch CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"results\Echo of Light\output\T Pyxidis\standard",
        help="Root directory for output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = run_pipeline(args.observation_csv, args.echo_patch_csv, args.output_dir)
    print(f"[DONE] Output written to: {out_dir}")
