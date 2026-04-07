from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

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

def add_standard_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()
    out["standard_preserved_input"] = True

    if "vt_total_proxy_kms" in out.columns:
        vt = pd.to_numeric(out["vt_total_proxy_kms"], errors="coerce")
        out["standard_vt_proxy_centered_kms"] = vt - np.nanmedian(vt)
    else:
        out["standard_vt_proxy_centered_kms"] = np.nan

    if is_6d and "radial_velocity_kms" in out.columns:
        rv = pd.to_numeric(out["radial_velocity_kms"], errors="coerce")
        out["standard_rv_centered_kms"] = rv - np.nanmedian(rv)
    else:
        out["standard_rv_centered_kms"] = np.nan

    if is_6d and "speed_total_proxy_kms" in out.columns:
        sp = pd.to_numeric(out["speed_total_proxy_kms"], errors="coerce")
        out["standard_speed_proxy_centered_kms"] = sp - np.nanmedian(sp)
    else:
        out["standard_speed_proxy_centered_kms"] = np.nan

    if "distance_proxy_kpc" in out.columns:
        dist = pd.to_numeric(out["distance_proxy_kpc"], errors="coerce")
        bins = [-np.inf, 5, 10, 20, 40, 80, np.inf]
        labels = ["<=5", "5-10", "10-20", "20-40", "40-80", "80+"]
        out["standard_distance_shell_kpc"] = pd.cut(dist, bins=bins, labels=labels)
    else:
        out["standard_distance_shell_kpc"] = pd.NA

    return out

def summarize_standard(df: pd.DataFrame, dataset_name: str) -> dict:
    return {
        "dataset": dataset_name,
        "rows_read": int(len(df)),
        "rows_written": int(len(df)),
        "median_distance_proxy_kpc": float(np.nanmedian(pd.to_numeric(df["distance_proxy_kpc"], errors="coerce"))) if "distance_proxy_kpc" in df.columns else np.nan,
        "median_vt_total_proxy_kms": float(np.nanmedian(pd.to_numeric(df["vt_total_proxy_kms"], errors="coerce"))) if "vt_total_proxy_kms" in df.columns else np.nan,
        "median_speed_total_proxy_kms": float(np.nanmedian(pd.to_numeric(df["speed_total_proxy_kms"], errors="coerce"))) if "speed_total_proxy_kms" in df.columns else np.nan,
    }

def main() -> None:
    project_root = find_project_root(Path(__file__))
    skeleton_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "skeleton"
    output_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "standard"

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

    out_5d = output_dir / "gaia_rrlyrae_5d_standard.csv"
    out_6d = output_dir / "gaia_rrlyrae_6d_standard.csv"
    out_summary = output_dir / "standard_summary.csv"
    out_readme = output_dir / "README_standard.txt"

    std_5d.to_csv(out_5d, index=False)
    std_6d.to_csv(out_6d, index=False)

    pd.DataFrame([
        summarize_standard(std_5d, "5d"),
        summarize_standard(std_6d, "6d"),
    ]).to_csv(out_summary, index=False)

    out_readme.write_text(
        "Our galaxy Halo Stellar Kinematics - standard stage\n\n"
        "Rule\n"
        "- Skeleton CSVs are read as fixed sample-preserving inputs.\n"
        "- No row deletion or new sample selection is allowed here.\n"
        "- Only baseline descriptive/statistical columns are added.\n",
        encoding="utf-8",
    )

    print("[DONE] Saved:")
    print(f" - {out_5d}")
    print(f" - {out_6d}")
    print(f" - {out_summary}")
    print(f" - {out_readme}")

if __name__ == "__main__":
    main()
