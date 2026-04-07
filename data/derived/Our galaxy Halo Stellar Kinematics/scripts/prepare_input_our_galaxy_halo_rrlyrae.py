from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd


# ============================================================
# Our Galaxy Halo Stellar Kinematics
# Final input-preparation script
# ------------------------------------------------------------
# Recommended location:
#   data/derived/Our galaxy Halo Stellar Kinematics/scripts/
#
# Inputs:
#   data/derived/Our galaxy Halo Stellar Kinematics/first_processed_v1/
#
# Outputs:
#   data/derived/Our galaxy Halo Stellar Kinematics/input/
# ============================================================


PROCESSED_FOLDER_NAME = "first_processed_v1"


def find_project_root(start_file: Path) -> Path:
    current = start_file.resolve()
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return start_file.resolve().parents[4]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def add_external_body_flag_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "external_body_flag" not in out.columns:
        out["external_body_flag"] = False

    if "external_body_reason" not in out.columns:
        out["external_body_reason"] = ""

    return out


def add_input_selection_flags(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()

    if "quality_basic_pass" not in out.columns:
        out["quality_basic_pass"] = False

    if "external_body_flag" not in out.columns:
        out["external_body_flag"] = False

    if "halo_candidate_flag" not in out.columns:
        out["halo_candidate_flag"] = out["quality_basic_pass"].fillna(False)

    if "processing_note" not in out.columns:
        out["processing_note"] = ""

    if "has_full_6d" not in out.columns:
        out["has_full_6d"] = False

    if is_6d:
        out["input_ready_flag"] = (
            out["quality_basic_pass"].fillna(False)
            & (~out["external_body_flag"].fillna(False))
            & out["has_full_6d"].fillna(False)
        )
    else:
        out["input_ready_flag"] = (
            out["quality_basic_pass"].fillna(False)
            & (~out["external_body_flag"].fillna(False))
        )

    out["input_exclusion_reason"] = ""

    quality_fail_mask = ~out["quality_basic_pass"].fillna(False)
    out.loc[quality_fail_mask, "input_exclusion_reason"] = "quality_basic_fail"

    external_mask = out["external_body_flag"].fillna(False)
    out.loc[external_mask, "input_exclusion_reason"] = "external_body_flag"

    if is_6d:
        full6d_fail_mask = ~out["has_full_6d"].fillna(False)
        out.loc[full6d_fail_mask, "input_exclusion_reason"] = "missing_full_6d"

    return out


def select_input_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    common_cols: List[str] = [
        "source_id",
        "ra_deg",
        "dec_deg",
        "parallax_mas",
        "parallax_error_mas",
        "pmra_masyr",
        "pmra_error_masyr",
        "pmdec_masyr",
        "pmdec_error_masyr",
        "phot_g_mean_mag",
        "bp_rp",
        "int_average_g",
        "metallicity",
        "quality_basic_pass",
        "external_body_flag",
        "external_body_reason",
        "halo_candidate_flag",
        "has_full_6d",
        "input_ready_flag",
        "input_exclusion_reason",
        "processing_note",
    ]

    extra_6d = [
        "radial_velocity_kms",
        "radial_velocity_error_kms",
    ]

    selected = common_cols + (extra_6d if is_6d else [])
    existing = [c for c in selected if c in df.columns]
    return df[existing].copy()


def summarize(df_full: pd.DataFrame, df_input: pd.DataFrame, dataset_name: str) -> dict:
    return {
        "dataset": dataset_name,
        "rows_processed_input_source": len(df_full),
        "rows_input_ready": int(df_full["input_ready_flag"].sum()) if "input_ready_flag" in df_full.columns else len(df_input),
        "rows_saved_input_csv": len(df_input),
        "quality_basic_pass_count": int(df_full["quality_basic_pass"].sum()) if "quality_basic_pass" in df_full.columns else 0,
        "external_body_flag_count": int(df_full["external_body_flag"].sum()) if "external_body_flag" in df_full.columns else 0,
        "has_full_6d_count": int(df_full["has_full_6d"].sum()) if "has_full_6d" in df_full.columns else 0,
    }


def write_readme(path: Path, processed_dir: Path, input_dir: Path) -> None:
    text = f"""Our galaxy Halo Stellar Kinematics - final input preparation

Processed source directory
- {processed_dir}

Output directory
- {input_dir}

Produced files
- gaia_rrlyrae_5d_input.csv
- gaia_rrlyrae_6d_input.csv
- input_preparation_summary.csv
- README_input_preparation.txt

Current input policy
- keep only input_ready_flag == True
- quality_basic_pass required
- external_body_flag must be False
- 6D requires has_full_6d == True
- no final physical halo cut yet

Operational rule
- no versioned subfolder is created under input
- files are written directly into the input directory
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    project_root = find_project_root(Path(__file__))

    derived_root = project_root / "data" / "derived" / "Our galaxy Halo Stellar Kinematics"
    processed_dir = derived_root / PROCESSED_FOLDER_NAME
    input_dir = derived_root / "input"

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed folder not found: {processed_dir}")

    processed_5d = processed_dir / "gaia_rrlyrae_5d_first_processed.csv"
    processed_6d = processed_dir / "gaia_rrlyrae_6d_first_processed.csv"

    if not processed_5d.exists():
        raise FileNotFoundError(f"5D processed CSV not found: {processed_5d}")
    if not processed_6d.exists():
        raise FileNotFoundError(f"6D processed CSV not found: {processed_6d}")

    ensure_dir(input_dir)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Processed dir: {processed_dir}")
    print(f"[INFO] Input output dir: {input_dir}")

    df_5d = read_csv_safe(processed_5d)
    df_6d = read_csv_safe(processed_6d)

    df_5d = add_external_body_flag_placeholders(df_5d)
    df_6d = add_external_body_flag_placeholders(df_6d)

    df_5d = add_input_selection_flags(df_5d, is_6d=False)
    df_6d = add_input_selection_flags(df_6d, is_6d=True)

    df_5d_input = select_input_columns(df_5d[df_5d["input_ready_flag"]].copy(), is_6d=False)
    df_6d_input = select_input_columns(df_6d[df_6d["input_ready_flag"]].copy(), is_6d=True)

    out_5d = input_dir / "gaia_rrlyrae_5d_input.csv"
    out_6d = input_dir / "gaia_rrlyrae_6d_input.csv"
    out_summary = input_dir / "input_preparation_summary.csv"
    out_readme = input_dir / "README_input_preparation.txt"

    df_5d_input.to_csv(out_5d, index=False)
    df_6d_input.to_csv(out_6d, index=False)

    summary_df = pd.DataFrame(
        [
            summarize(df_5d, df_5d_input, "5d"),
            summarize(df_6d, df_6d_input, "6d"),
        ]
    )
    summary_df.to_csv(out_summary, index=False)

    write_readme(out_readme, processed_dir, input_dir)

    print("[DONE] Saved:")
    print(f" - {out_5d}")
    print(f" - {out_6d}")
    print(f" - {out_summary}")
    print(f" - {out_readme}")


if __name__ == "__main__":
    main()