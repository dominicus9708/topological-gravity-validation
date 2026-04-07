from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# Our Galaxy Halo Stellar Kinematics
# Final input-preparation script with real selection flags
# ------------------------------------------------------------
# Recommended location:
#   data/derived/Our galaxy Halo Stellar Kinematics/scripts/
#
# Inputs:
#   data/derived/Our galaxy Halo Stellar Kinematics/first_processed_v1/
#
# Outputs:
#   data/derived/Our galaxy Halo Stellar Kinematics/input/
#
# Purpose:
# - Build the final input from first-processed CSVs.
# - external_body_flag and halo_candidate_flag are filled by actual rules,
#   not placeholders.
# ============================================================


PROCESSED_FOLDER_NAME = "first_processed_v1"

# ------------------------------------------------------------
# Selection policy (v1 practical finalization)
# ------------------------------------------------------------
# 1) external_body_flag
#    - Mark obvious Magellanic-Cloud main-body regions by sky position.
#    - This is intentionally conservative: it flags clear main-body regions only.
#
# 2) halo_candidate_flag
#    - Require quality_basic_pass == True
#    - Reject external_body_flag == True
#    - Require |b| >= 20 deg  (reduce Galactic disk contamination)
#    - Require distance_proxy_kpc >= 5 (favor halo-oriented population)
#    - Require metallicity <= -0.5 when metallicity exists
#
# 3) final input saving
#    - 5D final input: keep halo_candidate_flag == True
#    - 6D final input: keep halo_candidate_flag == True and has_full_6d == True
#
# Notes
# - This script finalizes the input. Downstream skeleton / standard /
#   topological stages should not re-filter rows.
# - The distance proxy is a technical proxy from parallax only:
#       distance_proxy_kpc = 1 / parallax_mas   if parallax_mas > 0
# ------------------------------------------------------------


KPC_PER_MAS_PARALLAX = 1.0


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


def add_distance_proxy_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "distance_proxy_kpc" not in out.columns:
        par = pd.to_numeric(out.get("parallax_mas"), errors="coerce")
        out["distance_proxy_kpc"] = np.where(par > 0, KPC_PER_MAS_PARALLAX / par, np.nan)
    return out


def add_galactic_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate equatorial(J2000) -> galactic conversion.

    Uses the standard IAU/J2000 constants:
      alpha_ngp = 192.85948 deg
      delta_ngp = 27.12825 deg
      l_omega   = 32.93192 deg
    """
    out = df.copy()

    ra_deg = pd.to_numeric(out.get("ra_deg"), errors="coerce")
    dec_deg = pd.to_numeric(out.get("dec_deg"), errors="coerce")

    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)

    alpha_ngp = np.deg2rad(192.85948)
    delta_ngp = np.deg2rad(27.12825)
    l_omega = np.deg2rad(32.93192)

    sin_b = (
        np.sin(dec) * np.sin(delta_ngp)
        + np.cos(dec) * np.cos(delta_ngp) * np.cos(ra - alpha_ngp)
    )
    b = np.arcsin(np.clip(sin_b, -1.0, 1.0))

    y = np.cos(dec) * np.sin(ra - alpha_ngp)
    x = (
        np.sin(dec) * np.cos(delta_ngp)
        - np.cos(dec) * np.sin(delta_ngp) * np.cos(ra - alpha_ngp)
    )
    l = np.arctan2(y, x) + l_omega
    l = np.mod(l, 2.0 * np.pi)

    out["gal_l_deg"] = np.rad2deg(l)
    out["gal_b_deg"] = np.rad2deg(b)
    return out


def compute_angular_separation_deg(
    ra_deg: pd.Series,
    dec_deg: pd.Series,
    ra0_deg: float,
    dec0_deg: float,
) -> pd.Series:
    ra1 = np.deg2rad(pd.to_numeric(ra_deg, errors="coerce"))
    dec1 = np.deg2rad(pd.to_numeric(dec_deg, errors="coerce"))
    ra2 = np.deg2rad(ra0_deg)
    dec2 = np.deg2rad(dec0_deg)

    cos_sep = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    sep = np.arccos(np.clip(cos_sep, -1.0, 1.0))
    return pd.Series(np.rad2deg(sep), index=ra_deg.index)


def add_external_body_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conservative external-body masking:
    - Flag only obvious Magellanic Cloud main-body regions.
    - Keep this strict and simple rather than over-flagging the sky.

    Approximate centers:
      LMC ~ (RA, Dec) = (80.894, -69.756)
      SMC ~ (RA, Dec) = (13.187, -72.829)

    Practical main-body radii:
      LMC main-body radius: 8 deg
      SMC main-body radius: 5 deg
    """
    out = df.copy()

    lmc_sep = compute_angular_separation_deg(out["ra_deg"], out["dec_deg"], 80.894, -69.756)
    smc_sep = compute_angular_separation_deg(out["ra_deg"], out["dec_deg"], 13.187, -72.829)

    in_lmc_main = lmc_sep <= 8.0
    in_smc_main = smc_sep <= 5.0

    out["external_body_flag"] = in_lmc_main | in_smc_main
    out["external_body_reason"] = ""

    out.loc[in_lmc_main, "external_body_reason"] = "LMC_main_body_region"
    out.loc[in_smc_main, "external_body_reason"] = "SMC_main_body_region"

    return out


def add_halo_candidate_flag(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()

    # Ensure needed columns exist
    if "quality_basic_pass" not in out.columns:
        out["quality_basic_pass"] = False
    if "external_body_flag" not in out.columns:
        out["external_body_flag"] = False
    if "has_full_6d" not in out.columns:
        out["has_full_6d"] = False

    gal_b = pd.to_numeric(out.get("gal_b_deg"), errors="coerce")
    dist = pd.to_numeric(out.get("distance_proxy_kpc"), errors="coerce")
    metallicity = pd.to_numeric(out.get("metallicity"), errors="coerce")

    latitude_pass = gal_b.abs() >= 20.0
    distance_pass = dist >= 5.0

    # Metallicity condition:
    # - If metallicity exists: require <= -0.5
    # - If metallicity missing: fail conservatively at final-input stage
    metallicity_pass = metallicity.notna() & (metallicity <= -0.5)

    halo_base = (
        out["quality_basic_pass"].fillna(False)
        & (~out["external_body_flag"].fillna(False))
        & latitude_pass.fillna(False)
        & distance_pass.fillna(False)
        & metallicity_pass.fillna(False)
    )

    if is_6d:
        halo_final = halo_base & out["has_full_6d"].fillna(False)
    else:
        halo_final = halo_base

    out["halo_candidate_flag"] = halo_final

    out["halo_candidate_reason"] = ""
    out.loc[~out["quality_basic_pass"].fillna(False), "halo_candidate_reason"] = "quality_basic_fail"
    out.loc[out["external_body_flag"].fillna(False), "halo_candidate_reason"] = "external_body_region"
    out.loc[gal_b.abs().fillna(-999) < 20.0, "halo_candidate_reason"] = "low_galactic_latitude"
    out.loc[dist.fillna(-999) < 5.0, "halo_candidate_reason"] = "distance_proxy_lt_5kpc"
    out.loc[~metallicity_pass.fillna(False), "halo_candidate_reason"] = "metallicity_not_halo_like"

    if is_6d:
        out.loc[~out["has_full_6d"].fillna(False), "halo_candidate_reason"] = "missing_full_6d"

    return out


def finalize_input_flags(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()

    if "halo_candidate_flag" not in out.columns:
        out["halo_candidate_flag"] = False

    out["input_ready_flag"] = out["halo_candidate_flag"].fillna(False)
    out["input_exclusion_reason"] = ""

    exclusion_mask = ~out["input_ready_flag"].fillna(False)
    if "halo_candidate_reason" in out.columns:
        out.loc[exclusion_mask, "input_exclusion_reason"] = out.loc[exclusion_mask, "halo_candidate_reason"]
    else:
        out.loc[exclusion_mask, "input_exclusion_reason"] = "not_input_ready"

    if is_6d:
        if "has_full_6d" not in out.columns:
            out["has_full_6d"] = False

    return out


def select_input_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    common_cols = [
        "source_id",
        "ra_deg",
        "dec_deg",
        "gal_l_deg",
        "gal_b_deg",
        "parallax_mas",
        "parallax_error_mas",
        "distance_proxy_kpc",
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
        "halo_candidate_reason",
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
        "rows_processed_input_source": int(len(df_full)),
        "rows_external_body_flag_true": int(df_full["external_body_flag"].sum()) if "external_body_flag" in df_full.columns else 0,
        "rows_halo_candidate_true": int(df_full["halo_candidate_flag"].sum()) if "halo_candidate_flag" in df_full.columns else 0,
        "rows_input_ready": int(df_full["input_ready_flag"].sum()) if "input_ready_flag" in df_full.columns else int(len(df_input)),
        "rows_saved_input_csv": int(len(df_input)),
        "quality_basic_pass_count": int(df_full["quality_basic_pass"].sum()) if "quality_basic_pass" in df_full.columns else 0,
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

This version uses actual finalization rules:
1) external_body_flag
   - conservative LMC/SMC main-body masking by sky position
2) halo_candidate_flag
   - quality_basic_pass == True
   - external_body_flag == False
   - |gal_b_deg| >= 20
   - distance_proxy_kpc >= 5
   - metallicity <= -0.5
   - and for 6D, has_full_6d == True

Operational rule
- The saved input files are treated as final fixed input for downstream stages.
- Downstream stages should not re-filter rows.
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
    print(f"[INFO] Final input dir: {input_dir}")

    df_5d = read_csv_safe(processed_5d)
    df_6d = read_csv_safe(processed_6d)

    df_5d = add_distance_proxy_if_needed(df_5d)
    df_6d = add_distance_proxy_if_needed(df_6d)

    df_5d = add_galactic_coordinates(df_5d)
    df_6d = add_galactic_coordinates(df_6d)

    df_5d = add_external_body_flag(df_5d)
    df_6d = add_external_body_flag(df_6d)

    df_5d = add_halo_candidate_flag(df_5d, is_6d=False)
    df_6d = add_halo_candidate_flag(df_6d, is_6d=True)

    df_5d = finalize_input_flags(df_5d, is_6d=False)
    df_6d = finalize_input_flags(df_6d, is_6d=True)

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
