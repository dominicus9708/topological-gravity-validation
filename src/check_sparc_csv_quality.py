# src/check_sparc_csv_quality.py

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


INPUT_DIR = Path("data/raw/sparc_csv")
SUMMARY_PATH = Path("results/summaries/sparc_csv_quality_check.csv")

REQUIRED_COLUMNS = [
    "r",
    "v_obs",
    "v_err",
    "v_gas",
    "v_disk",
    "v_bul",
]


def ensure_directories() -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def check_one_file(file_path: Path) -> dict:
    result = {
        "galaxy": file_path.stem,
        "source_file": str(file_path),
        "status": "ok",
        "n_rows": 0,
        "has_nan": False,
        "missing_columns": "",
        "non_positive_r": 0,
        "non_positive_v_err": 0,
        "negative_v_obs": 0,
        "not_monotonic_r": False,
        "duplicate_r_count": 0,
        "r_min": np.nan,
        "r_max": np.nan,
        "v_obs_min": np.nan,
        "v_obs_max": np.nan,
        "notes": "",
    }

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        result["status"] = "failed_to_read"
        result["notes"] = str(exc)
        return result

    result["n_rows"] = int(len(df))

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        result["status"] = "missing_columns"
        result["missing_columns"] = ",".join(missing)
        result["notes"] = f"Missing required columns: {missing}"
        return result

    # 숫자 강제 변환
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[REQUIRED_COLUMNS].isnull().any().any():
        result["has_nan"] = True
        result["status"] = "has_nan"

    result["non_positive_r"] = int((df["r"] <= 0).sum())
    result["non_positive_v_err"] = int((df["v_err"] <= 0).sum())
    result["negative_v_obs"] = int((df["v_obs"] < 0).sum())

    # 반지름 증가 여부
    r = df["r"].to_numpy(dtype=float)
    if len(r) >= 2:
        diffs = np.diff(r)
        # strictly increasing not required, but non-decreasing is expected
        if np.any(diffs < 0):
            result["not_monotonic_r"] = True
            result["status"] = "bad_radius_order"

    # 중복 반지름 개수
    result["duplicate_r_count"] = int(df["r"].duplicated().sum())

    # 범위 요약
    result["r_min"] = float(df["r"].min())
    result["r_max"] = float(df["r"].max())
    result["v_obs_min"] = float(df["v_obs"].min())
    result["v_obs_max"] = float(df["v_obs"].max())

    # 추가 판정
    notes = []

    if result["n_rows"] < 5:
        notes.append("too_few_points")

    if result["non_positive_r"] > 0:
        notes.append("non_positive_r")

    if result["non_positive_v_err"] > 0:
        notes.append("non_positive_v_err")

    if result["negative_v_obs"] > 0:
        notes.append("negative_v_obs")

    if result["duplicate_r_count"] > 0:
        notes.append("duplicate_r_present")

    if result["has_nan"]:
        notes.append("nan_present")

    if result["not_monotonic_r"]:
        notes.append("radius_not_monotonic")

    # status 보강
    if result["status"] == "ok" and notes:
        result["status"] = "warning"

    result["notes"] = ",".join(notes)

    return result


def check_all_files() -> pd.DataFrame:
    ensure_directories()

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {INPUT_DIR}")

    results = []

    for file_path in csv_files:
        info = check_one_file(file_path)
        results.append(info)

        if info["status"] == "ok":
            print(f"[OK] {file_path.name}")
        else:
            print(f"[{info['status'].upper()}] {file_path.name} -> {info['notes']}")

    df_summary = pd.DataFrame(results)
    df_summary.to_csv(SUMMARY_PATH, index=False)

    return df_summary


if __name__ == "__main__":
    summary = check_all_files()
    print("\nQuality check complete.")
    print(summary.head())
    print(f"\nTotal files checked: {len(summary)}")