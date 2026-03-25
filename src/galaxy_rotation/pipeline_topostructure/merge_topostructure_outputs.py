from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd


# -----------------------------------------------------------------------------
# Merge helper for topostructure outputs
#
# Modes
#   pointwise : merge *_pointwise_topostructure.csv into one master CSV
#   segment   : merge *_segment_structure.csv into one master CSV
#   summary   : pass through topostructure_summary.csv if present, or merge *_summary.csv
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge topostructure pipeline outputs into a single CSV.")
    parser.add_argument("--input-dir", required=True, help="Directory containing topostructure output CSV files.")
    parser.add_argument("--output-file", required=True, help="Output merged CSV path.")
    parser.add_argument(
        "--mode",
        choices=["pointwise", "segment", "summary"],
        default="pointwise",
        help="Which output family to merge. Default: pointwise",
    )
    return parser.parse_args()



def infer_glob(mode: str) -> str:
    if mode == "pointwise":
        return "*_pointwise_topostructure.csv"
    if mode == "segment":
        return "*_segment_structure.csv"
    return "*_summary.csv"



def infer_galaxy_name(df: pd.DataFrame, file_path: Path, mode: str) -> str:
    if "galaxy" in df.columns and df["galaxy"].notna().any():
        return str(df["galaxy"].dropna().iloc[0]).strip()
    stem = file_path.stem
    suffix_map = {
        "pointwise": "_pointwise_topostructure",
        "segment": "_segment_structure",
        "summary": "_summary",
    }
    suffix = suffix_map[mode]
    return stem[:-len(suffix)] if stem.endswith(suffix) else stem



def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "summary":
        summary_path = input_dir / "topostructure_summary.csv"
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df.to_csv(output_file, index=False)
            print(f"[OK] Copied existing summary: {summary_path} -> {output_file}")
            return

    glob_pattern = infer_glob(args.mode)
    files: List[Path] = sorted(input_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pattern!r} in {input_dir}")

    merged_frames: List[pd.DataFrame] = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if "source_file" not in df.columns:
            df["source_file"] = file_path.name
        if "galaxy" not in df.columns or df["galaxy"].isna().all():
            df["galaxy"] = infer_galaxy_name(df, file_path, args.mode)
        merged_frames.append(df)
        print(f"[OK] {file_path.name}")

    merged = pd.concat(merged_frames, ignore_index=True)
    merged.to_csv(output_file, index=False)
    print("\nMerge complete.")
    print(f"Rows: {len(merged)}")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
