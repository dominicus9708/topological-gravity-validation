#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
T Pyxidis standard baseline pipeline
-----------------------------------
Reads:
- t_pyx_standard_input_seed.csv

Writes:
- processed input copy
- filtered plotting CSV
- summary TXT
- standard baseline PNG

Designed for placement under:
src/echo of light/T Pyxidis/standard/t_pyx_standard_pipeline.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class PipelinePaths:
    input_csv: Path
    output_dir: Path


REQUIRED_COLUMNS = [
    "date",
    "day_since_outburst",
    "dataset",
    "structure_stage",
    "radius_arcsec",
    "radius_err_arcsec",
    "radius_value_status",
    "event_flag",
    "notes",
    "source_group",
    "ready_for_standard_plot",
    "measured_radius_flag",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a first-pass standard baseline plot for T Pyxidis."
    )
    parser.add_argument(
        "--input-csv",
        required=True,
        help="Path to t_pyx_standard_input_seed.csv",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory. A timestamped folder will be created under it.",
    )
    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["day_since_outburst"] = pd.to_numeric(out["day_since_outburst"], errors="coerce")
    out["radius_arcsec"] = pd.to_numeric(out["radius_arcsec"], errors="coerce")
    out["radius_err_arcsec"] = pd.to_numeric(out["radius_err_arcsec"], errors="coerce")
    out["ready_for_standard_plot"] = pd.to_numeric(
        out["ready_for_standard_plot"], errors="coerce"
    ).fillna(0).astype(int)
    out["measured_radius_flag"] = pd.to_numeric(
        out["measured_radius_flag"], errors="coerce"
    ).fillna(0).astype(int)
    return out


def make_timestamped_output_dir(base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = base_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def prepare_plotting_table(df: pd.DataFrame) -> pd.DataFrame:
    plot_df = df.loc[df["ready_for_standard_plot"] == 1].copy()
    plot_df = plot_df.sort_values(["day_since_outburst", "date"], na_position="last")
    return plot_df


def write_summary(df: pd.DataFrame, plot_df: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "T Pyxidis standard baseline pipeline summary",
        "===========================================",
        "",
        f"Total input rows: {len(df)}",
        f"Plot-ready rows: {len(plot_df)}",
        "",
        "Notes",
        "-----",
        "- This is a first-pass standard baseline based on seed input.",
        "- Current radius values are representative structural scales for public image epochs.",
        "- They should not yet be interpreted as strict per-epoch measured radius values.",
        "",
    ]
    if not plot_df.empty:
        lines.append("Plot-ready epochs")
        lines.append("----------------")
        for _, row in plot_df.iterrows():
            lines.append(
                f"- day={row['day_since_outburst']}, date={row['date'].date() if pd.notna(row['date']) else 'NA'}, "
                f"radius_arcsec={row['radius_arcsec']}, stage={row['structure_stage']}"
            )

    summary_path = output_dir / "run_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def build_plot(plot_df: pd.DataFrame, output_dir: Path) -> Path:
    fig_path = output_dir / "t_pyx_standard_radius_vs_time.png"

    plt.figure(figsize=(8, 5))
    if not plot_df.empty:
        yerr = None
        if plot_df["radius_err_arcsec"].notna().any():
            yerr = plot_df["radius_err_arcsec"]
        plt.errorbar(
            plot_df["day_since_outburst"],
            plot_df["radius_arcsec"],
            yerr=yerr,
            marker="o",
            linestyle="-",
        )

        for _, row in plot_df.iterrows():
            label = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "NA"
            plt.annotate(
                label,
                (row["day_since_outburst"], row["radius_arcsec"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel("Days since outburst")
    plt.ylabel("Radius (arcsec)")
    plt.title("T Pyxidis standard baseline (seed input)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    return fig_path


def main() -> None:
    args = parse_args()
    paths = PipelinePaths(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
    )

    if not paths.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {paths.input_csv}")

    df = pd.read_csv(paths.input_csv, encoding="utf-8-sig")
    validate_input(df)
    df = coerce_types(df)

    output_dir = make_timestamped_output_dir(paths.output_dir)

    processed_input_path = output_dir / "t_pyx_standard_input_processed.csv"
    df.to_csv(processed_input_path, index=False, encoding="utf-8-sig")

    plot_df = prepare_plotting_table(df)
    plot_input_path = output_dir / "t_pyx_standard_plot_input.csv"
    plot_df.to_csv(plot_input_path, index=False, encoding="utf-8-sig")

    build_plot(plot_df, output_dir)
    write_summary(df, plot_df, output_dir)

    print(f"[DONE] Output written to: {output_dir}")


if __name__ == "__main__":
    main()
