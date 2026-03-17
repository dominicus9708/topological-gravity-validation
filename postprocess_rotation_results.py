from __future__ import annotations

import sys
import warnings
from pathlib import Path

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


def resolve_file(filename: str) -> Path:
    """
    1) 프로젝트 루트에서 먼저 찾고
    2) 없으면 하위 폴더 전체를 재귀 탐색하여
    3) 가장 먼저 찾은 파일 경로를 반환
    """
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

OUTPUT_DIR = BASE_DIR / "postprocess_outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR       =", BASE_DIR)
print("SRC_DIR        =", SRC_DIR)
print("SUMMARY_ABAR   =", SUMMARY_ABAR)
print("SUMMARY_FORMULA=", SUMMARY_FORMULA)
print("OUTPUT_DIR     =", OUTPUT_DIR)
print("FIG_DIR        =", FIG_DIR)
print("TABLE_DIR      =", TABLE_DIR)


# =========================================================
# 1. 대표 6개 은하
# =========================================================
REPRESENTATIVE_GALAXIES = [
    "UGC00128",
    "DDO154",
    "F574-2",
    "NGC3953",
    "UGC02953",
    "NGC6674",
]

SIMILAR_EPSILON = 1e-12
TOP_N_OUTLIERS = 10


# =========================================================
# 2. sigma_model import
# =========================================================
import importlib.util

SIGMA_MODEL_PATH = SRC_DIR / "sigma_model.py"
if not SIGMA_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"src/sigma_model.py 파일을 찾지 못했습니다: {SIGMA_MODEL_PATH}"
    )

spec = importlib.util.spec_from_file_location("sigma_model", SIGMA_MODEL_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"sigma_model 모듈 스펙 생성 실패: {SIGMA_MODEL_PATH}")

sigma_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sigma_model)

print("SIGMA_MODEL_PATH =", SIGMA_MODEL_PATH)
print("sigma_model.__file__ =", getattr(sigma_model, "__file__", "UNKNOWN"))
print(
    "has compute_baryonic_velocity =",
    hasattr(sigma_model, "compute_baryonic_velocity"),
)
print(
    "has compute_galactic_sigma_proxy =",
    hasattr(sigma_model, "compute_galactic_sigma_proxy"),
)
print(
    "has compute_galactic_sigma_cumulative_effect =",
    hasattr(sigma_model, "compute_galactic_sigma_cumulative_effect"),
)
print(
    "has compute_model_velocity_curve =",
    hasattr(sigma_model, "compute_model_velocity_curve"),
)


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
# 4. 요약 CSV 로드 / 병합
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

    merged["delta_chi2"] = merged["chi2_abar"] - merged["chi2_formula"]
    merged["improvement_ratio"] = np.where(
        merged["chi2_abar"] != 0,
        merged["delta_chi2"] / merged["chi2_abar"],
        np.nan,
    )

    def classify(delta: float) -> str:
        if abs(delta) <= SIMILAR_EPSILON:
            return "similar"
        return "improved" if delta > 0 else "worsened"

    merged["classification"] = merged["delta_chi2"].apply(classify)
    return merged


# =========================================================
# 5. Population 분석 저장
# =========================================================
def save_population_tables(merged: pd.DataFrame) -> None:
    merged.to_csv(TABLE_DIR / "merged_rotation_summary.csv", index=False, encoding="utf-8-sig")

    stats = pd.DataFrame([
        {
            "model": "beta_abar",
            "n": len(merged),
            "mean_chi2": merged["chi2_abar"].mean(),
            "median_chi2": merged["chi2_abar"].median(),
            "min_chi2": merged["chi2_abar"].min(),
            "max_chi2": merged["chi2_abar"].max(),
        },
        {
            "model": "beta_formula",
            "n": len(merged),
            "mean_chi2": merged["chi2_formula"].mean(),
            "median_chi2": merged["chi2_formula"].median(),
            "min_chi2": merged["chi2_formula"].min(),
            "max_chi2": merged["chi2_formula"].max(),
        },
        {
            "model": "delta_chi2",
            "n": len(merged),
            "mean_chi2": merged["delta_chi2"].mean(),
            "median_chi2": merged["delta_chi2"].median(),
            "min_chi2": merged["delta_chi2"].min(),
            "max_chi2": merged["delta_chi2"].max(),
        },
    ])
    stats.to_csv(TABLE_DIR / "population_basic_stats.csv", index=False, encoding="utf-8-sig")

    counts = (
        merged["classification"]
        .value_counts(dropna=False)
        .rename_axis("classification")
        .reset_index(name="count")
    )
    counts.to_csv(TABLE_DIR / "classification_counts.csv", index=False, encoding="utf-8-sig")

    merged.sort_values("delta_chi2", ascending=False).head(TOP_N_OUTLIERS).to_csv(
        TABLE_DIR / "top_improved.csv", index=False, encoding="utf-8-sig"
    )
    merged.sort_values("delta_chi2", ascending=True).head(TOP_N_OUTLIERS).to_csv(
        TABLE_DIR / "top_worsened.csv", index=False, encoding="utf-8-sig"
    )
    merged.sort_values("chi2_formula", ascending=False).head(TOP_N_OUTLIERS).to_csv(
        TABLE_DIR / "top_high_chi2_formula.csv", index=False, encoding="utf-8-sig"
    )


def plot_population_figures(merged: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(merged["delta_chi2"].dropna(), bins=30)
    plt.xlabel("Delta chi2 = chi2_abar - chi2_formula")
    plt.ylabel("Number of galaxies")
    plt.title("Distribution of fit improvement across the galaxy sample")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "delta_chi2_histogram.png", dpi=200)
    plt.close()

    if "beta_value_formula" in merged.columns:
        plt.figure(figsize=(8, 5))
        plt.hist(merged["beta_value_formula"].dropna(), bins=30)
        plt.xlabel("beta (formula)")
        plt.ylabel("Number of galaxies")
        plt.title("Distribution of structural coefficient beta")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "beta_formula_histogram.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter(merged["beta_value_formula"], merged["delta_chi2"], s=16)
        plt.xlabel("beta (formula)")
        plt.ylabel("Delta chi2")
        plt.title("Structural coefficient beta vs fit improvement")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "beta_vs_delta_chi2.png", dpi=200)
        plt.close()


# =========================================================
# 6. 대표 은하 테이블
# =========================================================
def save_representative_table(merged: pd.DataFrame) -> pd.DataFrame:
    rep = merged[merged["galaxy"].isin(REPRESENTATIVE_GALAXIES)].copy()
    rep["order"] = rep["galaxy"].apply(REPRESENTATIVE_GALAXIES.index)
    rep = rep.sort_values("order").drop(columns=["order"])
    rep.to_csv(TABLE_DIR / "representative_6_galaxies.csv", index=False, encoding="utf-8-sig")
    return rep


# =========================================================
# 7. SPARC 원본 CSV 찾기 / 읽기
# =========================================================
def find_sparc_file(galaxy: str, row: pd.Series) -> Path:
    """
    요약 CSV의 source_file 우선 사용.
    없거나 경로가 깨졌으면 프로젝트 하위 폴더 전체에서 galaxy.csv 탐색.
    """
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

    raise FileNotFoundError(
        f"{galaxy}의 SPARC 원본 CSV를 찾지 못했습니다.\n"
        f"source_file 값 또는 프로젝트 하위 폴더 구조를 확인해 주세요."
    )


def load_and_normalize_sparc_csv(path: Path) -> pd.DataFrame:
    """
    sigma_model.py가 요구하는 df_norm 형식으로 컬럼명을 맞춘다.
    필요한 핵심 컬럼:
      r_kpc
      v_obs_kmps
      v_gas_kmps
      v_disk_kmps
      v_bul_kmps
      a_obs_kmps2_per_kpc
    """
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
# 8. 모델 재구성
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

    sigma_profile = sigma_model.compute_galactic_sigma_proxy(df_norm, D_bg=0.0)

    delta_v2_abar = sigma_model.compute_galactic_sigma_cumulative_effect(
        df_norm,
        beta=beta_abar,
        use_positive_only=True,
        rs_kpc=rs_kpc,
    )
    delta_v2_formula = sigma_model.compute_galactic_sigma_cumulative_effect(
        df_norm,
        beta=beta_formula,
        use_positive_only=True,
        rs_kpc=rs_kpc,
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
        "sigma_profile": sigma_profile,
        "delta_v2_abar": delta_v2_abar,
        "delta_v2_formula": delta_v2_formula,
        "v_model_abar": v_model_abar,
        "v_model_formula": v_model_formula,
    }


# =========================================================
# 9. 그림 생성
# =========================================================
def plot_rotation_curve(galaxy: str, curves: dict[str, np.ndarray]) -> None:
    r = curves["radius"]
    v_obs = curves["v_obs"]
    v_err = curves["v_err"]
    v_model_abar = curves["v_model_abar"]
    v_model_formula = curves["v_model_formula"]

    plt.figure(figsize=(8, 5))
    if np.isfinite(v_err).any():
        plt.errorbar(r, v_obs, yerr=v_err, fmt="o", ms=4, capsize=2, label="Observed")
    else:
        plt.plot(r, v_obs, "o", ms=4, label="Observed")

    plt.plot(r, v_model_abar, label="Model (beta_abar)")
    plt.plot(r, v_model_formula, label="Model (beta_formula)")

    plt.xlabel("Radius [kpc]")
    plt.ylabel("Rotation velocity [km/s]")
    plt.title(f"Rotation curve: {galaxy}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"rotation_curve_{galaxy}.png", dpi=200)
    plt.close()


def plot_term_decomposition(galaxy: str, curves: dict[str, np.ndarray]) -> None:
    r = curves["radius"]
    v_bar = curves["v_bar"]
    delta_v2_formula = curves["delta_v2_formula"]
    v_model_formula = curves["v_model_formula"]

    structural_velocity_equiv = np.sqrt(np.clip(delta_v2_formula, 0.0, None))

    plt.figure(figsize=(8, 5))
    plt.plot(r, v_bar, label="Baryonic baseline")
    plt.plot(r, structural_velocity_equiv, label="Structural correction (sqrt(Delta_v2))")
    plt.plot(r, v_model_formula, label="Total model (formula)")
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity scale [km/s]")
    plt.title(f"Term decomposition: {galaxy}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"term_decomposition_{galaxy}.png", dpi=200)
    plt.close()


def plot_sigma_profile(galaxy: str, curves: dict[str, np.ndarray]) -> None:
    r = curves["radius"]
    sigma_profile = curves["sigma_profile"]

    plt.figure(figsize=(8, 5))
    plt.plot(r, sigma_profile)
    plt.xlabel("Radius [kpc]")
    plt.ylabel("sigma_gal")
    plt.title(f"Structural sigma profile: {galaxy}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"sigma_profile_{galaxy}.png", dpi=200)
    plt.close()


def plot_residuals(galaxy: str, curves: dict[str, np.ndarray]) -> None:
    r = curves["radius"]
    v_obs = curves["v_obs"]
    res_abar = curves["v_model_abar"] - v_obs
    res_formula = curves["v_model_formula"] - v_obs

    plt.figure(figsize=(8, 5))
    plt.axhline(0.0)
    plt.plot(r, res_abar, label="Residual (beta_abar)")
    plt.plot(r, res_formula, label="Residual (beta_formula)")
    plt.xlabel("Radius [kpc]")
    plt.ylabel("v_model - v_obs [km/s]")
    plt.title(f"Residuals: {galaxy}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"residuals_{galaxy}.png", dpi=200)
    plt.close()


# =========================================================
# 10. 대표 6개 처리
# =========================================================
def process_representative_galaxies(merged: pd.DataFrame) -> None:
    rep = save_representative_table(merged)

    for _, row in rep.iterrows():
        galaxy = row["galaxy"]
        try:
            sparc_path = find_sparc_file(galaxy, row)
            print(f"[SPARC] {galaxy} -> {sparc_path}")

            df_norm = load_and_normalize_sparc_csv(sparc_path)
            curves = reconstruct_model_curves(df_norm, row)

            plot_rotation_curve(galaxy, curves)
            plot_term_decomposition(galaxy, curves)
            plot_sigma_profile(galaxy, curves)
            plot_residuals(galaxy, curves)

            print(f"[OK] {galaxy} 완료")
        except Exception as e:
            warnings.warn(f"{galaxy} 처리 실패: {e}")


# =========================================================
# 11. 메인
# =========================================================
def main() -> None:
    merged = load_and_merge_summaries()
    save_population_tables(merged)
    plot_population_figures(merged)
    process_representative_galaxies(merged)
    print("후처리 완료")


if __name__ == "__main__":
    main()