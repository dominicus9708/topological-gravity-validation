from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

KPC_PER_MAS_PARALLAX = 1.0
KM_S_PER_MASYR_KPC = 4.74047

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

def add_skeleton_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()
    out["input_row_index"] = np.arange(len(out), dtype=np.int64)

    if "ra_deg" in out.columns:
        out["ra_rad"] = np.deg2rad(pd.to_numeric(out["ra_deg"], errors="coerce"))
    if "dec_deg" in out.columns:
        out["dec_rad"] = np.deg2rad(pd.to_numeric(out["dec_deg"], errors="coerce"))

    if "parallax_mas" in out.columns:
        par = pd.to_numeric(out["parallax_mas"], errors="coerce")
        out["distance_proxy_kpc"] = np.where(par > 0, KPC_PER_MAS_PARALLAX / par, np.nan)
    else:
        out["distance_proxy_kpc"] = np.nan

    pmra = pd.to_numeric(out["pmra_masyr"], errors="coerce") if "pmra_masyr" in out.columns else np.nan
    pmdec = pd.to_numeric(out["pmdec_masyr"], errors="coerce") if "pmdec_masyr" in out.columns else np.nan
    dist = pd.to_numeric(out["distance_proxy_kpc"], errors="coerce")

    out["vt_ra_proxy_kms"] = KM_S_PER_MASYR_KPC * pmra * dist
    out["vt_dec_proxy_kms"] = KM_S_PER_MASYR_KPC * pmdec * dist
    out["vt_total_proxy_kms"] = np.sqrt(
        np.square(pd.to_numeric(out["vt_ra_proxy_kms"], errors="coerce")) +
        np.square(pd.to_numeric(out["vt_dec_proxy_kms"], errors="coerce"))
    )

    if is_6d and "radial_velocity_kms" in out.columns:
        rv = pd.to_numeric(out["radial_velocity_kms"], errors="coerce")
        out["speed_total_proxy_kms"] = np.sqrt(
            np.square(pd.to_numeric(out["vt_total_proxy_kms"], errors="coerce")) +
            np.square(rv)
        )
    else:
        out["speed_total_proxy_kms"] = np.nan

    out["skeleton_preserved_input"] = True
    return out

def summarize_skeleton(df: pd.DataFrame, dataset_name: str) -> dict:
    return {
        "dataset": dataset_name,
        "rows_read": int(len(df)),
        "rows_written": int(len(df)),
        "source_id_unique": int(df["source_id"].nunique()) if "source_id" in df.columns else 0,
        "distance_proxy_nonnull": int(df["distance_proxy_kpc"].notna().sum()) if "distance_proxy_kpc" in df.columns else 0,
        "vt_total_proxy_nonnull": int(df["vt_total_proxy_kms"].notna().sum()) if "vt_total_proxy_kms" in df.columns else 0,
        "speed_total_proxy_nonnull": int(df["speed_total_proxy_kms"].notna().sum()) if "speed_total_proxy_kms" in df.columns else 0,
    }

def main() -> None:
    project_root = find_project_root(Path(__file__))
    input_dir = project_root / "data" / "derived" / "Our galaxy Halo Stellar Kinematics" / "input"
    output_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "skeleton"

    input_5d = input_dir / "gaia_rrlyrae_5d_input.csv"
    input_6d = input_dir / "gaia_rrlyrae_6d_input.csv"

    if not input_5d.exists():
        raise FileNotFoundError(f"5D input CSV not found: {input_5d}")
    if not input_6d.exists():
        raise FileNotFoundError(f"6D input CSV not found: {input_6d}")

    ensure_dir(output_dir)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Input dir: {input_dir}")
    print(f"[INFO] Output dir: {output_dir}")

    df_5d = read_csv(input_5d)
    df_6d = read_csv(input_6d)

    sk_5d = add_skeleton_columns(df_5d, is_6d=False)
    sk_6d = add_skeleton_columns(df_6d, is_6d=True)

    out_5d = output_dir / "gaia_rrlyrae_5d_skeleton.csv"
    out_6d = output_dir / "gaia_rrlyrae_6d_skeleton.csv"
    out_summary = output_dir / "skeleton_summary.csv"
    out_readme = output_dir / "README_skeleton.txt"

    sk_5d.to_csv(out_5d, index=False)
    sk_6d.to_csv(out_6d, index=False)

    pd.DataFrame([
        summarize_skeleton(sk_5d, "5d"),
        summarize_skeleton(sk_6d, "6d"),
    ]).to_csv(out_summary, index=False)

    out_readme.write_text(
        "Our galaxy Halo Stellar Kinematics - skeleton stage\n\n"
        "Rule\n"
        "- Input CSVs are treated as fixed final raw input.\n"
        "- No row deletion or new sample selection is allowed here.\n"
        "- Only technical diagnostic/coordinate/proxy columns are added.\n",
        encoding="utf-8",
    )

    print("[DONE] Saved:")
    print(f" - {out_5d}")
    print(f" - {out_6d}")
    print(f" - {out_summary}")
    print(f" - {out_readme}")

if __name__ == "__main__":
    main()
