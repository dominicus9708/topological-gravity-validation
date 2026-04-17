#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_ROOT_DEFAULT = Path("data") / "derived" / "Cosmic Void Structural Validation" / "input"
OUTPUT_ROOT_DEFAULT = Path("results") / "Cosmic Void Structural Validation" / "output" / "skeleton"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root")
    parser.add_argument(
        "--input-root",
        default=str(INPUT_ROOT_DEFAULT),
        help="Input root containing cosmic_void_*_input.csv files",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT_DEFAULT),
        help="Skeleton output root. Results go into a timestamped subfolder.",
    )
    parser.add_argument(
        "--input-file",
        default="cosmic_void_master_input.csv",
        help="Primary input CSV filename inside input root",
    )
    parser.add_argument(
        "--run-tag",
        default="",
        help="Optional suffix appended to timestamp folder name",
    )
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


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def save_basic_plot(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    if "input_role" in df.columns:
        roles = [x for x in df["input_role"].dropna().unique().tolist()]
        if roles:
            for role in roles:
                sub = df[df["input_role"] == role].copy()
                plt.scatter(
                    sub.index,
                    pd.to_numeric(sub["radius_eff"], errors="coerce"),
                    s=20,
                    label=str(role),
                )
        else:
            plt.scatter(df.index, pd.to_numeric(df["radius_eff"], errors="coerce"), s=20, label="all_rows")
    else:
        plt.scatter(df.index, pd.to_numeric(df["radius_eff"], errors="coerce"), s=20, label="all_rows")

    plt.xlabel("Row index")
    plt.ylabel("radius_eff")
    plt.title("Cosmic Void Skeleton Check: radius_eff by input row")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    input_root = resolve_path(project_root, args.input_root)
    output_root = resolve_path(project_root, args.output_root)

    input_path = input_root / args.input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Primary input CSV not found: {input_path}")

    run_dir = make_run_dir(output_root, args.run_tag)

    df = pd.read_csv(input_path)

    required_cols = ["void_id", "algorithm", "sky_region", "ra", "dec", "radius_eff"]
    missing_required = [c for c in required_cols if c not in df.columns]

    summary_rows = []
    summary_rows.append({"metric": "input_path", "value": str(input_path)})
    summary_rows.append({"metric": "rows", "value": int(len(df))})
    summary_rows.append({"metric": "columns", "value": int(len(df.columns))})
    summary_rows.append({"metric": "missing_required_count", "value": int(len(missing_required))})
    summary_rows.append({"metric": "missing_required_columns", "value": ", ".join(missing_required) if missing_required else ""})

    for col in required_cols:
        if col in df.columns:
            summary_rows.append({"metric": f"{col}_missing_frac", "value": float(df[col].isna().mean())})

    if "algorithm" in df.columns:
        alg_counts = df["algorithm"].value_counts(dropna=False).to_dict()
        summary_rows.append({"metric": "algorithm_counts", "value": str(alg_counts)})

    if "input_role" in df.columns:
        role_counts = df["input_role"].value_counts(dropna=False).to_dict()
        summary_rows.append({"metric": "input_role_counts", "value": str(role_counts)})

    if "radius_eff" in df.columns:
        radius = pd.to_numeric(df["radius_eff"], errors="coerce")
        summary_rows.append({"metric": "radius_eff_min", "value": float(radius.min()) if radius.notna().any() else ""})
        summary_rows.append({"metric": "radius_eff_median", "value": float(radius.median()) if radius.notna().any() else ""})
        summary_rows.append({"metric": "radius_eff_max", "value": float(radius.max()) if radius.notna().any() else ""})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "skeleton_summary.csv", index=False, encoding="utf-8-sig")

    working_df = df.copy()
    working_df["skeleton_row_index"] = range(len(working_df))
    working_df["skeleton_has_basic_geometry"] = (
        working_df["ra"].notna()
        & working_df["dec"].notna()
        & working_df["radius_eff"].notna()
    ) if all(c in working_df.columns for c in ["ra", "dec", "radius_eff"]) else False
    working_df.to_csv(run_dir / "skeleton_working_copy.csv", index=False, encoding="utf-8-sig")

    if "radius_eff" in working_df.columns:
        save_basic_plot(working_df, run_dir / "skeleton_radius_eff_check.png")

    lines = []
    lines.append("Cosmic Void Structural Validation skeleton summary")
    lines.append("==================================================")
    lines.append(f"Project root: {project_root}")
    lines.append(f"Input root: {input_root}")
    lines.append(f"Primary input: {input_path}")
    lines.append(f"Run output: {run_dir}")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append("This is a skeleton-stage pipeline. It checks path connectivity, input loading,")
    lines.append("basic column presence, working-copy creation, and minimal plot generation.")
    lines.append("")
    lines.append("Row / column status")
    lines.append("-------------------")
    lines.append(f"- rows: {len(df)}")
    lines.append(f"- columns: {len(df.columns)}")
    lines.append(f"- missing required columns: {len(missing_required)}")
    if missing_required:
        lines.append(f"- missing list: {', '.join(missing_required)}")
    else:
        lines.append("- missing list: none")
    lines.append("")
    lines.append("Generated files")
    lines.append("---------------")
    lines.append("- skeleton_summary.csv")
    lines.append("- skeleton_working_copy.csv")
    if "radius_eff" in working_df.columns:
        lines.append("- skeleton_radius_eff_check.png")
    lines.append("- skeleton_summary.txt")
    lines.append("")
    lines.append("Interpretation note")
    lines.append("-------------------")
    lines.append(
        "This stage does not test the physical validity of the cosmic void model. "
        "It only confirms that the official input can be read and that the pipeline can create "
        "basic derived outputs in the expected results structure."
    )

    write_text(run_dir / "skeleton_summary.txt", "\n".join(lines))

    print(f"Primary input: {input_path}")
    print(f"Run output: {run_dir}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Missing required columns: {len(missing_required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
