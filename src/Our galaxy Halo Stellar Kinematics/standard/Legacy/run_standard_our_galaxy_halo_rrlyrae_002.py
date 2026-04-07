from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

SHELL_BINS_KPC = [5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
SHELL_LABELS = ["5-10", "10-20", "20-40", "40-80", "80+"]

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

def add_standard_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()
    out["standard_preserved_input"] = True

    dist = safe_numeric(out.get("distance_proxy_kpc"))
    out["standard_distance_proxy_kpc"] = dist

    vt = safe_numeric(out.get("vt_total_proxy_kms"))
    out["standard_vt_total_proxy_kms"] = vt
    out["standard_vt_proxy_centered_kms"] = vt - np.nanmedian(vt) if vt.notna().any() else np.nan

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

    out["standard_distance_shell_kpc"] = pd.cut(
        dist, bins=SHELL_BINS_KPC, labels=SHELL_LABELS, right=False
    )
    return out

def build_shell_summary(df: pd.DataFrame, dataset_name: str, is_6d: bool) -> pd.DataFrame:
    centers_map = {"5-10": 7.5, "10-20": 15.0, "20-40": 30.0, "40-80": 60.0, "80+": 100.0}
    rows = []

    for shell in SHELL_LABELS:
        sub = df[df["standard_distance_shell_kpc"].astype(str) == shell].copy()
        n = len(sub)

        vt = safe_numeric(sub.get("standard_vt_total_proxy_kms"))
        vt_centered = safe_numeric(sub.get("standard_vt_proxy_centered_kms"))
        dist = safe_numeric(sub.get("standard_distance_proxy_kpc"))

        row = {
            "dataset": dataset_name,
            "shell": shell,
            "shell_center_kpc": centers_map[shell],
            "n_shell": int(n),
            "distance_median_shell_kpc": float(np.nanmedian(dist)) if dist.notna().any() else np.nan,
            "standard_vt_median_shell_kms": float(np.nanmedian(vt)) if vt.notna().any() else np.nan,
            "standard_vt_variance_shell": float(np.nanvar(vt, ddof=1)) if vt.notna().sum() > 1 else np.nan,
            "standard_vt_centered_variance_shell": float(np.nanvar(vt_centered, ddof=1)) if vt_centered.notna().sum() > 1 else np.nan,
        }

        if is_6d:
            rv = safe_numeric(sub.get("standard_radial_velocity_kms"))
            sp = safe_numeric(sub.get("standard_speed_total_proxy_kms"))
            rv_centered = safe_numeric(sub.get("standard_rv_centered_kms"))
            sp_centered = safe_numeric(sub.get("standard_speed_proxy_centered_kms"))

            row.update({
                "standard_rv_median_shell_kms": float(np.nanmedian(rv)) if rv.notna().any() else np.nan,
                "standard_rv_variance_shell": float(np.nanvar(rv, ddof=1)) if rv.notna().sum() > 1 else np.nan,
                "standard_rv_centered_variance_shell": float(np.nanvar(rv_centered, ddof=1)) if rv_centered.notna().sum() > 1 else np.nan,
                "standard_speed_median_shell_kms": float(np.nanmedian(sp)) if sp.notna().any() else np.nan,
                "standard_speed_variance_shell": float(np.nanvar(sp, ddof=1)) if sp.notna().sum() > 1 else np.nan,
                "standard_speed_centered_variance_shell": float(np.nanvar(sp_centered, ddof=1)) if sp_centered.notna().sum() > 1 else np.nan,
            })
        else:
            row.update({
                "standard_rv_median_shell_kms": np.nan,
                "standard_rv_variance_shell": np.nan,
                "standard_rv_centered_variance_shell": np.nan,
                "standard_speed_median_shell_kms": np.nan,
                "standard_speed_variance_shell": np.nan,
                "standard_speed_centered_variance_shell": np.nan,
            })

        rows.append(row)

    return pd.DataFrame(rows)

def summarize_standard(df: pd.DataFrame, shell_df: pd.DataFrame, dataset_name: str, is_6d: bool) -> dict:
    summary = {
        "dataset": dataset_name,
        "rows_read": int(len(df)),
        "rows_written": int(len(df)),
        "shells_nonempty": int((pd.to_numeric(shell_df["n_shell"], errors="coerce") > 0).sum()),
        "median_distance_proxy_kpc": float(np.nanmedian(safe_numeric(df.get("standard_distance_proxy_kpc")))) if "standard_distance_proxy_kpc" in df.columns else np.nan,
        "median_vt_total_proxy_kms": float(np.nanmedian(safe_numeric(df.get("standard_vt_total_proxy_kms")))) if "standard_vt_total_proxy_kms" in df.columns else np.nan,
    }
    if is_6d:
        summary["median_speed_total_proxy_kms"] = float(np.nanmedian(safe_numeric(df.get("standard_speed_total_proxy_kms")))) if "standard_speed_total_proxy_kms" in df.columns else np.nan
    else:
        summary["median_speed_total_proxy_kms"] = np.nan
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
        "Our galaxy Halo Stellar Kinematics - standard stage 002\n\n"
        "Changes in this revision\n"
        "- Output is written into a timestamped subfolder.\n"
        "- Standard detail CSVs and shell summary CSVs are both saved.\n"
        "- Non-centered standard proxy columns are preserved for later comparison.\n"
        "- Centered columns are retained for descriptive baseline use only.\n"
        "- No graph output is generated in this revision.\n\n"
        "Rules\n"
        "- Skeleton CSVs are read as fixed sample-preserving inputs.\n"
        "- No row deletion or new sample selection is allowed here.\n"
        "- Only baseline descriptive/statistical columns are added.\n",
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
