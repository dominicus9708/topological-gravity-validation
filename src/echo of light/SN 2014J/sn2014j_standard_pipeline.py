#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SN 2014J standard pipeline
--------------------------
Input folder:
    data/derived/Echo of Light/input/SN 2014J

Output root:
    results/Echo of Light/output/SN 2014J/YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_DIR = Path("data/derived/Echo of Light/input/SN 2014J")
DEFAULT_OUTPUT_ROOT = Path("results/Echo of Light/output/SN 2014J")


def ensure_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def make_output_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_radius_plot(df: pd.DataFrame, out_path: Path) -> None:
    p = df.copy()
    p["time_value"] = pd.to_numeric(p["time_value"], errors="coerce")
    p["radius_arcsec"] = pd.to_numeric(p["radius_arcsec"], errors="coerce")
    p = p.dropna(subset=["time_value", "radius_arcsec"]).sort_values("time_value")

    plt.figure(figsize=(8, 5))
    for label, g in p.groupby("component_label", dropna=False):
        lab = str(label) if pd.notna(label) else "unlabeled_component"
        plt.plot(g["time_value"], g["radius_arcsec"], marker="o", label=lab)
    plt.xlabel("time_value")
    plt.ylabel("radius_arcsec")
    plt.title("SN 2014J Radius-like Measurements vs Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="SN 2014J standard pipeline")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--epochs-csv", default="SN2014J_epochs_processed.csv")
    parser.add_argument("--measurements-csv", default="SN2014J_measurements_processed.csv")
    parser.add_argument("--model-input-csv", default="SN2014J_model_input_radius.csv")
    parser.add_argument("--auxiliary-csv", default="SN2014J_auxiliary_processed.csv")
    parser.add_argument("--sources-csv", default="SN2014J_sources_combined.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    epochs_df = load_csv(ensure_file(input_dir / args.epochs_csv))
    measurements_df = load_csv(ensure_file(input_dir / args.measurements_csv))
    model_input_df = load_csv(ensure_file(input_dir / args.model_input_csv))
    auxiliary_df = load_csv(ensure_file(input_dir / args.auxiliary_csv))
    sources_df = load_csv(ensure_file(input_dir / args.sources_csv))

    out_dir = make_output_dir(output_root)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_csv(epochs_df, out_dir / "epochs_copy.csv")
    write_csv(measurements_df, out_dir / "measurements_copy.csv")
    write_csv(model_input_df, out_dir / "model_input_radius_copy.csv")
    write_csv(auxiliary_df, out_dir / "auxiliary_copy.csv")
    write_csv(sources_df, out_dir / "sources_copy.csv")

    save_radius_plot(model_input_df, plots_dir / "sn2014j_radius_vs_time.png")

    summary = {
        "pipeline_name": "SN 2014J standard pipeline",
        "created_at": datetime.now().isoformat(),
        "input_dir": str(input_dir),
        "results_dir": str(out_dir),
        "row_counts": {
            "epochs": int(len(epochs_df)),
            "measurements": int(len(measurements_df)),
            "model_input_radius": int(len(model_input_df)),
            "auxiliary": int(len(auxiliary_df)),
            "sources": int(len(sources_df)),
        },
        "notes": [
            "This is a first standard-reference pipeline for SN 2014J.",
            "The model_input_radius table contains arcsec radius-like rows extracted from accessible sources.",
            "The radius plot groups rows by component_label.",
        ],
        "generated_plots": [
            "plots/sn2014j_radius_vs_time.png"
        ],
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    (out_dir / "README_results.txt").write_text(
        "SN 2014J standard pipeline results folder.\n"
        "Contains copies of processed input CSVs and a first radius-vs-time plot.\n",
        encoding="utf-8",
    )

    print(f"[DONE] Results written to: {out_dir}")


if __name__ == "__main__":
    main()
