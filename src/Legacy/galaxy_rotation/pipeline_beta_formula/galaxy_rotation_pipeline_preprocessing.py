from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "galaxy",
    "r_kpc",
    "v_obs_kmps",
    "v_err_kmps",
    "v_gas_kmps",
    "v_disk_kmps",
    "v_bul_kmps",
]


NUMERIC_COLUMNS = [
    "r_kpc",
    "v_obs_kmps",
    "v_err_kmps",
    "v_gas_kmps",
    "v_disk_kmps",
    "v_bul_kmps",
]


def load_processed_sparc_table(path: str | Path) -> pd.DataFrame:
    """
    정규화된 SPARC CSV를 읽고,
    베타 포뮬러 파이프라인에서 바로 사용할 수 있도록 정리해서 반환합니다.

    처리 내용:
    - 파일 존재 여부 확인
    - 필수 열 확인
    - 주요 수치 열 numeric 변환
    - NaN / inf 제거
    - r_kpc > 0 조건 유지
    - 반지름 기준 정렬
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed SPARC table not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Loaded SPARC table is empty: {path}")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in SPARC table: {missing}")

    df = df.copy()

    # 숫자형 강제 변환
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # inf -> NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # 필수 수치열 기준 결측 제거
    df = df.dropna(subset=NUMERIC_COLUMNS)

    # 반지름은 반드시 양수
    df = df[df["r_kpc"] > 0].copy()

    if df.empty:
        raise ValueError(f"No valid rows remain after cleaning: {path}")

    # 반지름 기준 정렬
    df = df.sort_values("r_kpc").reset_index(drop=True)

    return df