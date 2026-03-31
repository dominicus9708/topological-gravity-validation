from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_output_dir(root: str | Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(root) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_pipeline(observation_csv: str | Path, geometry_csv: str | Path, output_dir: str | Path) -> Path:
    obs = pd.read_csv(observation_csv, encoding="utf-8-sig")
    geom = pd.read_csv(geometry_csv, encoding="utf-8-sig")

    validate_columns(
        obs,
        ["date", "day_since_outburst", "obs_type", "instrument_or_context", "echo_visibility_note", "source"],
        "observation CSV",
    )
    validate_columns(
        geom,
        [
            "feature_id",
            "geometry_type",
            "adopted_ring_radius_arcsec",
            "ring_radius_status",
            "inclination_deg_min",
            "inclination_deg_max",
            "foreground_distance_pc_status",
            "delay_time_days",
            "position_note",
            "directness",
            "source",
            "notes",
        ],
        "geometry CSV",
    )

    out_dir = build_output_dir(output_dir)

    obs.to_csv(out_dir / "t_pyx_standard_observation_epochs_processed.csv", index=False, encoding="utf-8-sig")
    geom.to_csv(out_dir / "t_pyx_standard_geometry_reference_processed.csv", index=False, encoding="utf-8-sig")

    plot_df = geom.copy()
    plot_df = plot_df[pd.to_numeric(plot_df["delay_time_days"], errors="coerce").notna()].copy()
    plot_df["day_since_outburst"] = pd.to_numeric(plot_df["delay_time_days"], errors="coerce")
    plot_df["radius_arcsec"] = pd.to_numeric(plot_df["adopted_ring_radius_arcsec"], errors="coerce")
    plot_df = plot_df.sort_values("day_since_outburst")
    plot_df.to_csv(out_dir / "t_pyx_standard_plot_input.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["day_since_outburst"], plot_df["radius_arcsec"], marker="o")
    for _, row in plot_df.iterrows():
        plt.text(row["day_since_outburst"], row["radius_arcsec"], f"{int(row['day_since_outburst'])} d")
    plt.title("T Pyxidis standard baseline (original-paper geometry reference)")
    plt.xlabel("Days since outburst")
    plt.ylabel("Representative ring radius (arcsec)")
    plt.tight_layout()
    plt.savefig(out_dir / "t_pyx_standard_baseline_radius_vs_time.png", dpi=150)
    plt.close()

    summary_lines = [
        "T Pyx standard baseline pipeline run",
        f"observation_csv={Path(observation_csv)}",
        f"geometry_csv={Path(geometry_csv)}",
        f"processed_observation_rows={len(obs)}",
        f"processed_geometry_rows={len(geom)}",
        f"plot_rows={len(plot_df)}",
        "",
        "Caution:",
        "- This is an original-paper baseline using global geometry references attached to public epochs.",
        "- The adopted 5 arcsec ring radius is a representative literature scale, not a strict epoch-by-epoch fitted table.",
        "- Keep this file distinct from the earlier skeleton/seed baseline.",
    ]
    (out_dir / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8-sig")

    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run T Pyx standard baseline pipeline.")
    parser.add_argument(
        "--observation-csv",
        default=r"data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_observation_epochs.csv",
        help="Path to observation epochs CSV.",
    )
    parser.add_argument(
        "--geometry-csv",
        default=r"data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_geometry_reference.csv",
        help="Path to geometry reference CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"results\Echo of Light\output\T Pyxidis\standard\baseline",
        help="Root directory for output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = run_pipeline(args.observation_csv, args.geometry_csv, args.output_dir)
    print(f"[DONE] Output written to: {out_dir}")
