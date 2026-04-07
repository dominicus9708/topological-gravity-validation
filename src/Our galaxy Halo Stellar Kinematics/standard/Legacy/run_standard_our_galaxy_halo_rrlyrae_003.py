from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# ============================================================
# Our Galaxy Halo Stellar Kinematics
# Standard pipeline 003
# ------------------------------------------------------------
# Placement:
#   src/Our galaxy Halo Stellar Kinematics/standard/
#     run_standard_our_galaxy_halo_rrlyrae_003.py
#
# Input:
#   results/Our galaxy Halo Stellar Kinematics/output/skeleton/
#
# Output:
#   results/Our galaxy Halo Stellar Kinematics/output/standard/YYYYMMDD_HHMMSS/
#
# Rules:
# - Skeleton CSVs are treated as fixed sample-preserving inputs.
# - No row deletion or new sample selection is allowed here.
# - This revision keeps descriptive columns and additionally
#   produces shell-level independent baseline variance models
#   for later topological comparison.
# ============================================================

SHELL_BINS_KPC = [5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
SHELL_LABELS = ["5-10", "10-20", "20-40", "40-80", "80+"]
SHELL_CENTERS = {"5-10": 7.5, "10-20": 15.0, "20-40": 30.0, "40-80": 60.0, "80+": 100.0}
MIN_SHELL_N_FOR_MODEL = 20
EPS = 1e-12


def find_project_root(start_file: Path) -> Path:
    current = start_file.resolve()
    for parent in current.parents:
        if (parent / "data").exists() and (parent / "results").exists():
            return parent
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return start_file.resolve().parents[4]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def safe_numeric(series):
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def timestamp_folder_name() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def assign_shells(dist: pd.Series) -> pd.Series:
    return pd.cut(dist, bins=SHELL_BINS_KPC, labels=SHELL_LABELS, right=False)


def add_standard_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()
    out["standard_preserved_input"] = True

    dist = safe_numeric(out.get("distance_proxy_kpc"))
    vt = safe_numeric(out.get("vt_total_proxy_kms"))

    out["standard_distance_proxy_kpc"] = dist
    out["standard_vt_total_proxy_kms"] = vt
    out["standard_vt_proxy_centered_kms"] = vt - np.nanmedian(vt) if vt.notna().any() else np.nan
    out["standard_distance_shell_kpc"] = assign_shells(dist)

    if is_6d:
        rv = safe_numeric(out.get("radial_velocity_kms"))
        sp = safe_numeric(out.get("speed_total_proxy_kms"))
        out["standard_radial_velocity_kms"] = rv
        out["standard_speed_total_proxy_kms"] = sp
        out["standard_rv_centered_kms"] = rv - np.nanmedian(rv) if rv.notna().any() else np.nan
        out["standard_speed_proxy_centered_kms"] = sp - np.nanmedian(sp) if sp.notna().any() else np.nan
    else:
        out["standard_radial_velocity_kms"] = np.nan
        out["standard_speed_total_proxy_kms"] = np.nan
        out["standard_rv_centered_kms"] = np.nan
        out["standard_speed_proxy_centered_kms"] = np.nan

    return out


def shell_stats(values: pd.Series) -> tuple[float, float, float]:
    x = safe_numeric(values)
    if x.notna().sum() == 0:
        return np.nan, np.nan, np.nan
    median = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - median)))
    var = float(np.nanvar(x, ddof=1)) if x.notna().sum() > 1 else np.nan
    return median, mad, var


def fit_log_variance_model(shell_df: pd.DataFrame, obs_col: str, model_col: str) -> pd.DataFrame:
    out = shell_df.copy()

    valid = (
        pd.to_numeric(out["n_shell"], errors="coerce") >= MIN_SHELL_N_FOR_MODEL
    ) & pd.to_numeric(out[obs_col], errors="coerce").notna() & (pd.to_numeric(out[obs_col], errors="coerce") > 0)

    x = np.log(pd.to_numeric(out.loc[valid, "shell_center_kpc"], errors="coerce").to_numpy())
    y = np.log(pd.to_numeric(out.loc[valid, obs_col], errors="coerce").to_numpy())

    if valid.sum() >= 2:
        coeffs = np.polyfit(x, y, deg=1)
        x_all = np.log(pd.to_numeric(out["shell_center_kpc"], errors="coerce").to_numpy())
        y_hat = coeffs[0] * x_all + coeffs[1]
        out[model_col] = np.exp(y_hat)
        out[f"{model_col}_source"] = "log_linear_shell_variance_fit"
    elif valid.sum() == 1:
        only_val = float(pd.to_numeric(out.loc[valid, obs_col], errors="coerce").iloc[0])
        out[model_col] = only_val
        out[f"{model_col}_source"] = "single_shell_fallback"
    else:
        out[model_col] = np.nan
        out[f"{model_col}_source"] = "no_valid_shell"

    return out


def build_shell_summary(df: pd.DataFrame, dataset_name: str, is_6d: bool) -> pd.DataFrame:
    rows = []

    for shell in SHELL_LABELS:
        sub = df[df["standard_distance_shell_kpc"].astype(str) == shell].copy()
        n = len(sub)

        dist = safe_numeric(sub.get("standard_distance_proxy_kpc"))
        vt = safe_numeric(sub.get("standard_vt_total_proxy_kms"))
        vt_centered = safe_numeric(sub.get("standard_vt_proxy_centered_kms"))

        vt_med, vt_mad, vt_var = shell_stats(vt)
        _, _, vt_centered_var = shell_stats(vt_centered)
        dist_med, _, _ = shell_stats(dist)

        row = {
            "dataset": dataset_name,
            "shell": shell,
            "shell_center_kpc": SHELL_CENTERS[shell],
            "n_shell": int(n),
            "distance_median_shell_kpc": dist_med,
            "standard_vt_median_shell_kms": vt_med,
            "standard_vt_mad_shell_kms": vt_mad,
            "standard_vt_variance_shell": vt_var,
            "standard_vt_centered_variance_shell": vt_centered_var,
        }

        if is_6d:
            rv = safe_numeric(sub.get("standard_radial_velocity_kms"))
            sp = safe_numeric(sub.get("standard_speed_total_proxy_kms"))
            rv_centered = safe_numeric(sub.get("standard_rv_centered_kms"))
            sp_centered = safe_numeric(sub.get("standard_speed_proxy_centered_kms"))

            rv_med, rv_mad, rv_var = shell_stats(rv)
            sp_med, sp_mad, sp_var = shell_stats(sp)
            _, _, rv_centered_var = shell_stats(rv_centered)
            _, _, sp_centered_var = shell_stats(sp_centered)

            row.update({
                "standard_rv_median_shell_kms": rv_med,
                "standard_rv_mad_shell_kms": rv_mad,
                "standard_rv_variance_shell": rv_var,
                "standard_rv_centered_variance_shell": rv_centered_var,
                "standard_speed_median_shell_kms": sp_med,
                "standard_speed_mad_shell_kms": sp_mad,
                "standard_speed_variance_shell": sp_var,
                "standard_speed_centered_variance_shell": sp_centered_var,
            })
        else:
            row.update({
                "standard_rv_median_shell_kms": np.nan,
                "standard_rv_mad_shell_kms": np.nan,
                "standard_rv_variance_shell": np.nan,
                "standard_rv_centered_variance_shell": np.nan,
                "standard_speed_median_shell_kms": np.nan,
                "standard_speed_mad_shell_kms": np.nan,
                "standard_speed_variance_shell": np.nan,
                "standard_speed_centered_variance_shell": np.nan,
            })

        rows.append(row)

    shell_df = pd.DataFrame(rows)
    shell_df = fit_log_variance_model(shell_df, "standard_vt_variance_shell", "standard_vt_variance_shell_model")

    if is_6d:
        shell_df = fit_log_variance_model(shell_df, "standard_speed_variance_shell", "standard_speed_variance_shell_model")
        shell_df = fit_log_variance_model(shell_df, "standard_rv_variance_shell", "standard_rv_variance_shell_model")
    else:
        shell_df["standard_speed_variance_shell_model"] = np.nan
        shell_df["standard_speed_variance_shell_model_source"] = "not_applicable"
        shell_df["standard_rv_variance_shell_model"] = np.nan
        shell_df["standard_rv_variance_shell_model_source"] = "not_applicable"

    return shell_df


def summarize_standard(df: pd.DataFrame, shell_df: pd.DataFrame, dataset_name: str, is_6d: bool) -> dict:
    summary = {
        "dataset": dataset_name,
        "rows_read": int(len(df)),
        "rows_written": int(len(df)),
        "shells_nonempty": int((pd.to_numeric(shell_df["n_shell"], errors="coerce") > 0).sum()),
        "median_distance_proxy_kpc": float(np.nanmedian(safe_numeric(df.get("standard_distance_proxy_kpc")))) if "standard_distance_proxy_kpc" in df.columns else np.nan,
        "median_vt_total_proxy_kms": float(np.nanmedian(safe_numeric(df.get("standard_vt_total_proxy_kms")))) if "standard_vt_total_proxy_kms" in df.columns else np.nan,
        "median_vt_variance_model": float(np.nanmedian(safe_numeric(shell_df.get("standard_vt_variance_shell_model")))) if "standard_vt_variance_shell_model" in shell_df.columns else np.nan,
    }
    if is_6d:
        summary["median_speed_total_proxy_kms"] = float(np.nanmedian(safe_numeric(df.get("standard_speed_total_proxy_kms")))) if "standard_speed_total_proxy_kms" in df.columns else np.nan
        summary["median_speed_variance_model"] = float(np.nanmedian(safe_numeric(shell_df.get("standard_speed_variance_shell_model")))) if "standard_speed_variance_shell_model" in shell_df.columns else np.nan
    else:
        summary["median_speed_total_proxy_kms"] = np.nan
        summary["median_speed_variance_model"] = np.nan
    return summary


def main() -> None:
    project_root = find_project_root(Path(__file__))
    skeleton_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "skeleton"
    output_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "standard" / timestamp_folder_name()

    sk_5d_path = skeleton_dir / "gaia_rrlyrae_5d_skeleton.csv"
    sk_6d_path = skeleton_dir / "gaia_rrlyrae_6d_skeleton.csv"

    if not sk_5d_path.exists():
        raise FileNotFoundError(f"5D skeleton CSV not found: {sk_5d_path}")
    if not sk_6d_path.exists():
        raise FileNotFoundError(f"6D skeleton CSV not found: {sk_6d_path}")

    ensure_dir(output_dir)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Skeleton dir: {skeleton_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    sk_5d = read_csv(sk_5d_path)
    sk_6d = read_csv(sk_6d_path)

    std_5d = add_standard_columns(sk_5d, is_6d=False)
    std_6d = add_standard_columns(sk_6d, is_6d=True)

    shell_5d = build_shell_summary(std_5d, dataset_name="5d", is_6d=False)
    shell_6d = build_shell_summary(std_6d, dataset_name="6d", is_6d=True)

    out_5d = output_dir / "gaia_rrlyrae_5d_standard.csv"
    out_6d = output_dir / "gaia_rrlyrae_6d_standard.csv"
    out_5d_shell = output_dir / "gaia_rrlyrae_5d_standard_shells.csv"
    out_6d_shell = output_dir / "gaia_rrlyrae_6d_standard_shells.csv"
    out_summary = output_dir / "standard_summary.csv"
    out_readme = output_dir / "README_standard.txt"

    std_5d.to_csv(out_5d, index=False)
    std_6d.to_csv(out_6d, index=False)
    shell_5d.to_csv(out_5d_shell, index=False)
    shell_6d.to_csv(out_6d_shell, index=False)

    pd.DataFrame([
        summarize_standard(std_5d, shell_5d, "5d", is_6d=False),
        summarize_standard(std_6d, shell_6d, "6d", is_6d=True),
    ]).to_csv(out_summary, index=False)

    out_readme.write_text(
        "Our galaxy Halo Stellar Kinematics - standard stage 003\n\n"
        "Changes in this revision\n"
        "- Output is written into a timestamped subfolder.\n"
        "- Detail CSVs and shell summary CSVs are saved.\n"
        "- Non-centered proxy columns are preserved for later comparison.\n"
        "- Centered columns are retained for descriptive baseline use only.\n"
        "- Shell-level independent variance baseline models are additionally constructed.\n"
        "- No graph output is generated in this revision.\n\n"
        "Important interpretation note\n"
        "- Raw non-centered proxy columns are descriptive baselines.\n"
        "- *_variance_shell_model columns are the intended comparison targets for later topological shell-level analysis.\n",
        encoding="utf-8",
    )

    print("[DONE] Saved:")
    print(f" - {out_5d}")
    print(f" - {out_6d}")
    print(f" - {out_5d_shell}")
    print(f" - {out_6d_shell}")
    print(f" - {out_summary}")
    print(f" - {out_readme}")


if __name__ == "__main__":
    main()
