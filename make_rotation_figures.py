from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 0. 경로 설정
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

FIG_DIR = BASE_DIR / "paper" / "figure"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def resolve_file(filename: str) -> Path:
    direct = BASE_DIR / filename
    if direct.exists():
        return direct

    matches = list(BASE_DIR.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"파일을 찾지 못했습니다: {filename}\n"
            f"검색 시작 위치: {BASE_DIR}"
        )

    if len(matches) > 1:
        print(f"[WARNING] 같은 이름의 파일이 여러 개 발견되었습니다: {filename}")
        for m in matches:
            print(" -", m)
        print(f"[USE] 첫 번째 파일 사용: {matches[0]}")

    return matches[0]


SUMMARY_ABAR = resolve_file("rotation_batch_summary_beta_abar.csv")
SUMMARY_FORMULA = resolve_file("rotation_batch_summary_beta_formula.csv")
SPARC_DIR = BASE_DIR / "data" / "raw" / "sparc_csv"

print("SUMMARY_ABAR   =", SUMMARY_ABAR)
print("SUMMARY_FORMULA=", SUMMARY_FORMULA)
print("SPARC_DIR      =", SPARC_DIR)
print("FIG_DIR        =", FIG_DIR)


# =========================================================
# 1. sigma_model import
# =========================================================
import importlib.util

SIGMA_MODEL_PATH = SRC_DIR / "sigma_model.py"
if not SIGMA_MODEL_PATH.exists():
    raise FileNotFoundError(f"src/sigma_model.py 파일을 찾지 못했습니다: {SIGMA_MODEL_PATH}")

spec = importlib.util.spec_from_file_location("sigma_model", SIGMA_MODEL_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"sigma_model 모듈 스펙 생성 실패: {SIGMA_MODEL_PATH}")

sigma_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sigma_model)

print("sigma_model.__file__ =", getattr(sigma_model, "__file__", "UNKNOWN"))


# =========================================================
# 2. 사용자 설정
# =========================================================
REPRESENTATIVE_GALAXIES = [
    "UGC00128",  # 강한 개선
    "DDO154",    # 대표적 개선
    "NGC6674",   # 악화 사례
]


# =========================================================
# 3. 유틸
# =========================================================
def pick_first_existing_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for col in aliases:
        if col in df.columns:
            return col
    return None


def as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# =========================================================
# 4. 요약 CSV 병합
# =========================================================
def load_and_merge_summaries() -> pd.DataFrame:
    df_abar = pd.read_csv(SUMMARY_ABAR)
    df_formula = pd.read_csv(SUMMARY_FORMULA)

    required = {"galaxy", "chi2", "beta_value", "status"}
    missing_abar = required - set(df_abar.columns)
    missing_formula = required - set(df_formula.columns)

    if missing_abar:
        raise ValueError(f"beta_abar CSV 필수 컬럼 누락: {missing_abar}")
    if missing_formula:
        raise ValueError(f"beta_formula CSV 필수 컬럼 누락: {missing_formula}")

    df_abar = df_abar.rename(columns={c: f"{c}_abar" for c in df_abar.columns if c != "galaxy"})
    df_formula = df_formula.rename(columns={c: f"{c}_formula" for c in df_formula.columns if c != "galaxy"})

    merged = pd.merge(df_abar, df_formula, on="galaxy", how="inner")
    return merged


# =========================================================
# 5. SPARC 원본 CSV 찾기 / 정규화
# =========================================================
def find_sparc_file(galaxy: str, row: pd.Series) -> Path:
    candidates: list[Path] = []

    for key in ["source_file_formula", "source_file_abar"]:
        if key in row and pd.notna(row[key]):
            raw_path = str(row[key]).strip()
            if raw_path:
                candidates.append(Path(raw_path))

    for cand in candidates:
        if cand.is_absolute() and cand.exists():
            return cand

        p1 = BASE_DIR / cand
        if p1.exists():
            return p1

        p2 = BASE_DIR / cand.name
        if p2.exists():
            return p2

        matches = list(BASE_DIR.rglob(cand.name))
        if matches:
            return matches[0]

    matches = list(BASE_DIR.rglob(f"{galaxy}.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"{galaxy}의 SPARC 원본 CSV를 찾지 못했습니다.")


def load_and_normalize_sparc_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)

    col_r = pick_first_existing_column(raw, ["r_kpc", "R", "r", "radius", "rad", "Rad"])
    col_vobs = pick_first_existing_column(raw, ["v_obs_kmps", "Vobs", "v_obs", "V", "V_obs"])
    col_verr = pick_first_existing_column(raw, ["v_err_kmps", "e_Vobs", "v_err", "Verr", "eVobs"])
    col_vgas = pick_first_existing_column(raw, ["v_gas_kmps", "Vgas", "v_gas"])
    col_vdisk = pick_first_existing_column(raw, ["v_disk_kmps", "Vdisk", "v_disk"])
    col_vbul = pick_first_existing_column(raw, ["v_bul_kmps", "Vbul", "v_bul", "Vbulge"])

    if col_r is None or col_vobs is None or col_vgas is None or col_vdisk is None or col_vbul is None:
        raise ValueError(
            "SPARC CSV의 필수 컬럼을 찾지 못했습니다.\n"
            f"현재 컬럼: {list(raw.columns)}"
        )

    df = pd.DataFrame()
    df["r_kpc"] = as_float_array(raw[col_r])
    df["v_obs_kmps"] = as_float_array(raw[col_vobs])
    df["v_gas_kmps"] = as_float_array(raw[col_vgas])
    df["v_disk_kmps"] = as_float_array(raw[col_vdisk])
    df["v_bul_kmps"] = as_float_array(raw[col_vbul])

    if col_verr is not None:
        df["v_err_kmps"] = as_float_array(raw[col_verr])
    else:
        df["v_err_kmps"] = np.nan

    eps = 1.0e-12
    df["a_obs_kmps2_per_kpc"] = (
        df["v_obs_kmps"].to_numpy() ** 2
    ) / np.maximum(df["r_kpc"].to_numpy(), eps)

    return df


# =========================================================
# 6. 모델 재구성
# =========================================================
def reconstruct_model_curves(df_norm: pd.DataFrame, row: pd.Series) -> dict[str, np.ndarray]:
    beta_abar = float(row["beta_value_abar"]) if pd.notna(row["beta_value_abar"]) else np.nan
    beta_formula = float(row["beta_value_formula"]) if pd.notna(row["beta_value_formula"]) else np.nan

    rs_kpc = None
    if "rs_kpc_formula" in row and pd.notna(row["rs_kpc_formula"]):
        rs_kpc = float(row["rs_kpc_formula"])
    elif "rs_kpc_abar" in row and pd.notna(row["rs_kpc_abar"]):
        rs_kpc = float(row["rs_kpc_abar"])

    radius = df_norm["r_kpc"].to_numpy(dtype=float)
    v_obs = df_norm["v_obs_kmps"].to_numpy(dtype=float)
    v_err = df_norm["v_err_kmps"].to_numpy(dtype=float)

    v_bar = sigma_model.compute_baryonic_velocity(
        df_norm["v_gas_kmps"].values,
        df_norm["v_disk_kmps"].values,
        df_norm["v_bul_kmps"].values,
    )

    v_model_abar = sigma_model.compute_model_velocity_curve(
        df_norm,
        beta=beta_abar,
        use_positive_only=True,
        rs_kpc=rs_kpc,
    )
    v_model_formula = sigma_model.compute_model_velocity_curve(
        df_norm,
        beta=beta_formula,
        use_positive_only=True,
        rs_kpc=rs_kpc,
    )

    return {
        "radius": radius,
        "v_obs": v_obs,
        "v_err": v_err,
        "v_bar": v_bar,
        "v_model_abar": v_model_abar,
        "v_model_formula": v_model_formula,
    }


# =========================================================
# 7. 그림 생성
# =========================================================
def plot_rotation_curve(galaxy: str, curves: dict[str, np.ndarray]) -> None:
    r = curves["radius"]
    v_obs = curves["v_obs"]
    v_err = curves["v_err"]
    v_bar = curves["v_bar"]
    v_model_abar = curves["v_model_abar"]
    v_model_formula = curves["v_model_formula"]

    plt.figure(figsize=(6, 4.2))

    if np.isfinite(v_err).any():
        plt.errorbar(
            r, v_obs, yerr=v_err,
            fmt="o", ms=3.5, capsize=2,
            label="Observed"
        )
    else:
        plt.plot(r, v_obs, "o", ms=3.5, label="Observed")

    plt.plot(r, v_bar, linestyle=":", label="Baryonic baseline")
    plt.plot(r, v_model_abar, linestyle="--", label=r"Model ($\beta=200$)")
    plt.plot(r, v_model_formula, linewidth=2, label=r"Model ($\beta_{\mathrm{struct}}$)")

    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.title(galaxy)
    plt.legend(fontsize=8)
    plt.tight_layout()

    outfile = FIG_DIR / f"rotation_curve_{galaxy}.png"
    plt.savefig(outfile, dpi=300)
    plt.close()

    print("saved:", outfile)


# =========================================================
# 8. 메인
# =========================================================
def main():
    merged = load_and_merge_summaries()
    selected = merged[merged["galaxy"].isin(REPRESENTATIVE_GALAXIES)].copy()

    if selected.empty:
        raise ValueError("대표 은하가 merged summary에서 선택되지 않았습니다.")

    for _, row in selected.iterrows():
        galaxy = row["galaxy"]
        try:
            sparc_path = find_sparc_file(galaxy, row)
            print(f"[SPARC] {galaxy} -> {sparc_path}")

            df_norm = load_and_normalize_sparc_csv(sparc_path)
            curves = reconstruct_model_curves(df_norm, row)
            plot_rotation_curve(galaxy, curves)

        except Exception as e:
            warnings.warn(f"{galaxy} 처리 실패: {e}")

    print("대표 은하 그래프 생성 완료")


if __name__ == "__main__":
    main()