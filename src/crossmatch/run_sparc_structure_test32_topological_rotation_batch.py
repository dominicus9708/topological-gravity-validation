#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPARC structure test32 batch runner with graph output
-----------------------------------------------------
목적
- data/derived/crossmatch/sparc_structure_test32/per_galaxy 아래의 은하 CSV들을 일괄 계산
- structure_enriched 형식 CSV(2014 / 2002 계열 혼합)에도 맞도록 컬럼 자동 해석
- 각 은하별 결과 CSV + 그래프 PNG 저장
- 출력은 data/derived/crossmatched/<timestamp>/ 아래에 저장

지원 입력 컬럼 예시
-------------------
1) 직접 입력형
- r_kpc / r
- v_obs_kms / v_obs / v_obs_kmps
- v_vis_kms / v_vis
- h_kpc / h

2) structure_enriched 형
- r_kpc
- v_obs_kmps
- v_gas_kmps
- v_disk_kmps
- v_bul_kmps
- structure_2014_z0
- structure_2014_h
- structure_2002_Z0
- structure_2002_Re
- structure_2002_z0_over_re

v_vis 처리 규칙
---------------
v_vis_kms 가 없으면 아래로 자동 구성:
v_vis = sqrt(v_gas^2 + v_disk^2 + v_bul^2)

h_kpc 처리 규칙
---------------
우선순위:
1) h_kpc / h / thickness_kpc
2) structure_2014_z0
3) structure_2002_Z0
4) structure_2014_h
5) structure_2002_z0_over_re * structure_2002_Re
6) structure_2014_z0_over_h * structure_2014_h

그래프 출력
-----------
각 은하마다 아래 곡선을 PNG로 저장:
- v_obs
- v_vis
- v_struct
- v_topo

주의
----
- rho_struct, M_struct는 아직 정규화 이전의 구조 누적량으로 취급합니다.
- 따라서 절대 질량 해석보다 정합지향 비교용 프로토타입 배치 실행본으로 쓰는 것이 적절합니다.
"""

from __future__ import annotations

# -----
# Section 1. imports
# -----

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----
# Section 2. constants
# -----

G_KPC_KMS2_PER_MSUN = 4.3e-6


# -----
# Section 3. path helpers
# -----

def project_root_from_script(script_path: Path) -> Path:
    """
    src/crossmatch/<this_file>.py 에 있다고 가정하고
    상위 2단계를 프로젝트 루트로 사용.
    """
    return script_path.resolve().parents[2]


def default_input_dir(project_root: Path) -> Path:
    return project_root / "data" / "derived" / "crossmatch" / "sparc_structure_test32" / "per_galaxy"


def default_output_base(project_root: Path) -> Path:
    return project_root / "data" / "derived" / "crossmatched"


def make_timestamp_dir(base_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# -----
# Section 4. generic helpers
# -----

def choose_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_to_original = {c.strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_to_original:
            return lower_to_original[key]
    return None


def numeric_series(df: pd.DataFrame, column_name: str) -> pd.Series:
    return pd.to_numeric(df[column_name], errors="coerce")


def build_v_vis_from_components(df: pd.DataFrame) -> Optional[pd.Series]:
    gas_col = choose_first_existing_column(df, ["v_gas_kms", "v_gas_kmps", "v_gas"])
    disk_col = choose_first_existing_column(df, ["v_disk_kms", "v_disk_kmps", "v_disk"])
    bul_col = choose_first_existing_column(df, ["v_bul_kms", "v_bul_kmps", "v_bul", "v_bulge"])

    if gas_col is None and disk_col is None and bul_col is None:
        return None

    gas = numeric_series(df, gas_col) if gas_col else pd.Series(0.0, index=df.index)
    disk = numeric_series(df, disk_col) if disk_col else pd.Series(0.0, index=df.index)
    bul = numeric_series(df, bul_col) if bul_col else pd.Series(0.0, index=df.index)

    return np.sqrt((gas.fillna(0.0) ** 2) + (disk.fillna(0.0) ** 2) + (bul.fillna(0.0) ** 2))


def build_h_kpc(df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    수직 두께 h_kpc를 자동 구성.
    우선순위:
    1) h_kpc / h / thickness_kpc
    2) structure_2014_z0
    3) structure_2002_Z0
    4) structure_2014_h
    5) structure_2002_z0_over_re * structure_2002_Re
    6) structure_2014_z0_over_h * structure_2014_h
    """
    direct_col = choose_first_existing_column(
        df,
        ["h_kpc", "h", "thickness_kpc", "disk_thickness_kpc", "z0_kpc"],
    )
    if direct_col is not None:
        return numeric_series(df, direct_col), direct_col

    z0_2014_col = choose_first_existing_column(df, ["structure_2014_z0"])
    if z0_2014_col is not None:
        return numeric_series(df, z0_2014_col), z0_2014_col

    z0_2002_col = choose_first_existing_column(df, ["structure_2002_Z0"])
    if z0_2002_col is not None:
        return numeric_series(df, z0_2002_col), z0_2002_col

    h2014_col = choose_first_existing_column(df, ["structure_2014_h"])
    if h2014_col is not None:
        return numeric_series(df, h2014_col), h2014_col

    ratio_2002_col = choose_first_existing_column(df, ["structure_2002_z0_over_re"])
    re_2002_col = choose_first_existing_column(df, ["structure_2002_Re"])
    if ratio_2002_col is not None and re_2002_col is not None:
        return (
            numeric_series(df, ratio_2002_col) * numeric_series(df, re_2002_col),
            "structure_2002_z0_over_re * structure_2002_Re",
        )

    ratio_2014_col = choose_first_existing_column(df, ["structure_2014_z0_over_h"])
    if ratio_2014_col is not None and h2014_col is not None:
        return (
            numeric_series(df, ratio_2014_col) * numeric_series(df, h2014_col),
            "structure_2014_z0_over_h * structure_2014_h",
        )

    return None, None


def first_nonempty_value(df: pd.DataFrame, candidates: List[str], default: str = "") -> str:
    for col in candidates:
        if col in df.columns:
            ser = df[col].dropna()
            if len(ser) > 0:
                val = str(ser.iloc[0]).strip()
                if val:
                    return val
    return default


# -----
# Section 5. column normalization
# -----

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # r_kpc
    r_col = choose_first_existing_column(out, ["r_kpc", "r", "radius_kpc", "rad_kpc"])
    if r_col is not None and r_col != "r_kpc":
        out = out.rename(columns={r_col: "r_kpc"})

    # v_obs_kms
    vobs_col = choose_first_existing_column(
        out,
        ["v_obs_kms", "v_obs", "vobs", "v_obs_km_s", "vobs_kms", "v_obs_kmps"],
    )
    if vobs_col is not None and vobs_col != "v_obs_kms":
        out = out.rename(columns={vobs_col: "v_obs_kms"})

    # v_vis_kms
    vvis_col = choose_first_existing_column(
        out,
        ["v_vis_kms", "v_vis", "vbar", "v_baryon", "v_baryonic", "v_vis_km_s", "v_vis_kmps"],
    )
    if vvis_col is not None:
        if vvis_col != "v_vis_kms":
            out = out.rename(columns={vvis_col: "v_vis_kms"})
    else:
        built_vvis = build_v_vis_from_components(out)
        if built_vvis is not None:
            out["v_vis_kms"] = built_vvis
            out["v_vis_source"] = "sqrt(v_gas^2 + v_disk^2 + v_bul^2)"

    # h_kpc
    if "h_kpc" not in out.columns:
        built_h, h_source = build_h_kpc(out)
        if built_h is not None:
            out["h_kpc"] = built_h
            out["h_kpc_source"] = h_source

    required = ["r_kpc", "v_obs_kms", "v_vis_kms", "h_kpc"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {missing}. 현재 컬럼: {list(df.columns)}")

    return out


# -----
# Section 6. validation
# -----

def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    keep_cols = ["r_kpc", "v_obs_kms", "v_vis_kms", "h_kpc"]
    optional_keep = [
        c for c in [
            "galaxy",
            "matched_sparc_name",
            "h_kpc_source",
            "v_vis_source",
        ] if c in df.columns
    ]
    out = df[optional_keep + keep_cols].copy()

    for col in keep_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out[keep_cols].isna().any().any():
        nan_counts = out[keep_cols].isna().sum()
        raise ValueError(f"수치 변환 후 NaN이 발생했습니다:\n{nan_counts}")

    out = out.sort_values("r_kpc").reset_index(drop=True)

    if (out["r_kpc"] <= 0).any():
        bad_rows = out.index[out["r_kpc"] <= 0].tolist()
        raise ValueError(f"r_kpc는 모두 양수여야 합니다. 문제 행: {bad_rows}")

    if (out["h_kpc"] < 0).any():
        bad_rows = out.index[out["h_kpc"] < 0].tolist()
        raise ValueError(f"h_kpc는 음수가 될 수 없습니다. 문제 행: {bad_rows}")

    if out["r_kpc"].nunique() < 2:
        raise ValueError("서로 다른 반경 포인트가 최소 2개 필요합니다.")

    if np.any(np.diff(out["r_kpc"].to_numpy(dtype=float)) <= 0):
        raise ValueError("r_kpc는 중복 없이 증가해야 합니다.")

    return out


# -----
# Section 7. core model
# -----

def compute_delta_r(r: np.ndarray) -> np.ndarray:
    dr = np.empty_like(r, dtype=float)
    dr[0] = r[1] - r[0]
    dr[1:] = np.diff(r)

    if np.any(dr <= 0):
        raise ValueError("반경 간격 Δr는 모두 양수여야 합니다.")
    return dr


def compute_topological_rotation(df: pd.DataFrame, structural_mass_scale: float = 1.0) -> pd.DataFrame:
    df = validate_and_prepare(df)

    r = df["r_kpc"].to_numpy(dtype=float)
    v_obs = df["v_obs_kms"].to_numpy(dtype=float)
    v_vis = df["v_vis_kms"].to_numpy(dtype=float)
    h = df["h_kpc"].to_numpy(dtype=float)

    r_min = float(np.min(r))
    r_max = float(np.max(r))
    if not (r_max > r_min):
        raise ValueError("r_max는 r_min보다 커야 합니다.")

    denom_dw = np.log(r / (r_min / 10.0))
    if np.any(np.isclose(denom_dw, 0.0)):
        raise ValueError("Dw 계산 분모가 0에 가깝습니다.")

    dw = 2.0 + np.log(1.0 + (h / r)) / denom_dw

    sigma = np.zeros_like(dw)
    sigma[0] = 0.0
    sigma[1:] = np.abs(np.diff(dw))

    ln_span = np.log(r_max / r_min)
    if np.isclose(ln_span, 0.0):
        raise ValueError("ln(r_max / r_min)이 0에 가깝습니다.")

    sigma_scale = 1.0 / ln_span
    beta = np.exp(-sigma / sigma_scale)

    rho_struct = (dw ** 2) * beta
    dr = compute_delta_r(r)
    m_struct = np.cumsum(rho_struct * dr) * structural_mass_scale

    g_topo = G_KPC_KMS2_PER_MSUN * m_struct / (r ** 2)
    g_topo = np.clip(g_topo, a_min=0.0, a_max=None)

    v_struct = np.sqrt(r * g_topo)
    v_topo = np.sqrt(v_vis ** 2 + v_struct ** 2)

    residual = v_obs - v_topo
    abs_residual = np.abs(residual)
    sq_error = residual ** 2

    out = df.copy()
    out["r_min_kpc"] = r_min
    out["r_max_kpc"] = r_max
    out["delta_r_kpc"] = dr
    out["Dw"] = dw
    out["sigma"] = sigma
    out["sigma_scale"] = sigma_scale
    out["beta"] = beta
    out["rho_struct"] = rho_struct
    out["M_struct"] = m_struct
    out["g_topo_kms2_per_kpc"] = g_topo
    out["v_struct_kms"] = v_struct
    out["v_topo_kms"] = v_topo
    out["residual_kms"] = residual
    out["abs_residual_kms"] = abs_residual
    out["sq_error"] = sq_error
    return out


def summarize_results(result_df: pd.DataFrame) -> Dict[str, float]:
    residual = result_df["residual_kms"].to_numpy(dtype=float)
    abs_residual = result_df["abs_residual_kms"].to_numpy(dtype=float)
    sq_error = result_df["sq_error"].to_numpy(dtype=float)
    v_obs = result_df["v_obs_kms"].to_numpy(dtype=float)
    v_topo = result_df["v_topo_kms"].to_numpy(dtype=float)

    rmse = float(np.sqrt(np.mean(sq_error)))
    mae = float(np.mean(abs_residual))

    ss_res = float(np.sum((v_obs - v_topo) ** 2))
    ss_tot = float(np.sum((v_obs - np.mean(v_obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    summary = {
        "n_points": int(len(result_df)),
        "r_min_kpc": float(result_df["r_kpc"].min()),
        "r_max_kpc": float(result_df["r_kpc"].max()),
        "Dw_min": float(result_df["Dw"].min()),
        "Dw_max": float(result_df["Dw"].max()),
        "beta_min": float(result_df["beta"].min()),
        "beta_max": float(result_df["beta"].max()),
        "v_struct_max_kms": float(result_df["v_struct_kms"].max()),
        "rmse_kms": rmse,
        "mae_kms": mae,
        "r2": r2,
    }

    if "h_kpc_source" in result_df.columns and len(result_df["h_kpc_source"].dropna()) > 0:
        summary["h_kpc_source"] = str(result_df["h_kpc_source"].dropna().iloc[0])

    if "v_vis_source" in result_df.columns and len(result_df["v_vis_source"].dropna()) > 0:
        summary["v_vis_source"] = str(result_df["v_vis_source"].dropna().iloc[0])

    return summary


# -----
# Section 8. plotting
# -----

def save_rotation_plot(result_df: pd.DataFrame, png_path: Path, title: str) -> None:
    r = result_df["r_kpc"].to_numpy(dtype=float)
    v_obs = result_df["v_obs_kms"].to_numpy(dtype=float)
    v_vis = result_df["v_vis_kms"].to_numpy(dtype=float)
    v_struct = result_df["v_struct_kms"].to_numpy(dtype=float)
    v_topo = result_df["v_topo_kms"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 5))
    plt.plot(r, v_obs, marker="o", linewidth=1.8, label="v_obs")
    plt.plot(r, v_vis, marker="s", linewidth=1.6, label="v_vis")
    plt.plot(r, v_struct, marker="^", linewidth=1.6, label="v_struct")
    plt.plot(r, v_topo, marker="d", linewidth=2.0, label="v_topo")

    plt.xlabel("r [kpc]")
    plt.ylabel("velocity [km/s]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()


def save_residual_plot(result_df: pd.DataFrame, png_path: Path, title: str) -> None:
    r = result_df["r_kpc"].to_numpy(dtype=float)
    residual = result_df["residual_kms"].to_numpy(dtype=float)

    plt.figure(figsize=(8, 4.5))
    plt.axhline(0.0, linewidth=1.2)
    plt.plot(r, residual, marker="o", linewidth=1.8)
    plt.xlabel("r [kpc]")
    plt.ylabel("v_obs - v_topo [km/s]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160, bbox_inches="tight")
    plt.close()


# -----
# Section 9. batch runner
# -----

@dataclass
class BatchResult:
    galaxy_name: str
    status: str
    message: str
    output_csv: Optional[str]
    output_plot_png: Optional[str] = None
    output_residual_png: Optional[str] = None
    n_points: Optional[int] = None
    rmse_kms: Optional[float] = None
    mae_kms: Optional[float] = None
    r2: Optional[float] = None
    Dw_min: Optional[float] = None
    Dw_max: Optional[float] = None
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    h_kpc_source: Optional[str] = None
    v_vis_source: Optional[str] = None


def iter_csv_files(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.glob("*.csv")):
        if path.is_file():
            yield path


def process_one_file(
    csv_path: Path,
    output_csv_dir: Path,
    output_plot_dir: Path,
    output_residual_dir: Path,
    structural_mass_scale: float,
) -> BatchResult:
    galaxy_name = csv_path.stem
    try:
        raw_df = pd.read_csv(csv_path)
        result_df = compute_topological_rotation(raw_df, structural_mass_scale=structural_mass_scale)
        summary = summarize_results(result_df)

        label_name = first_nonempty_value(
            result_df,
            ["galaxy", "matched_sparc_name"],
            default=galaxy_name,
        )

        out_csv = output_csv_dir / f"{galaxy_name}_topological_rotation.csv"
        out_plot = output_plot_dir / f"{galaxy_name}_rotation_curve.png"
        out_residual = output_residual_dir / f"{galaxy_name}_residual.png"

        result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        save_rotation_plot(result_df, out_plot, f"{label_name} rotation curve")
        save_residual_plot(result_df, out_residual, f"{label_name} residual")

        return BatchResult(
            galaxy_name=galaxy_name,
            status="OK",
            message="success",
            output_csv=str(out_csv),
            output_plot_png=str(out_plot),
            output_residual_png=str(out_residual),
            n_points=int(summary["n_points"]),
            rmse_kms=float(summary["rmse_kms"]),
            mae_kms=float(summary["mae_kms"]),
            r2=float(summary["r2"]) if not pd.isna(summary["r2"]) else np.nan,
            Dw_min=float(summary["Dw_min"]),
            Dw_max=float(summary["Dw_max"]),
            beta_min=float(summary["beta_min"]),
            beta_max=float(summary["beta_max"]),
            h_kpc_source=str(summary.get("h_kpc_source", "")) if summary.get("h_kpc_source") is not None else None,
            v_vis_source=str(summary.get("v_vis_source", "")) if summary.get("v_vis_source") is not None else None,
        )
    except Exception as e:
        return BatchResult(
            galaxy_name=galaxy_name,
            status="FAILED",
            message=str(e),
            output_csv=None,
            output_plot_png=None,
            output_residual_png=None,
        )


def run_batch(
    input_dir: Path,
    output_base: Path,
    structural_mass_scale: float = 1.0,
) -> Tuple[Path, pd.DataFrame]:
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")

    csv_files = list(iter_csv_files(input_dir))
    if not csv_files:
        raise FileNotFoundError(f"입력 폴더에 CSV 파일이 없습니다: {input_dir}")

    run_dir = make_timestamp_dir(output_base)

    per_galaxy_csv_dir = run_dir / "per_galaxy_results"
    rotation_plot_dir = run_dir / "per_galaxy_plots"
    residual_plot_dir = run_dir / "per_galaxy_residual_plots"

    per_galaxy_csv_dir.mkdir(parents=True, exist_ok=True)
    rotation_plot_dir.mkdir(parents=True, exist_ok=True)
    residual_plot_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for csv_path in csv_files:
        result = process_one_file(
            csv_path=csv_path,
            output_csv_dir=per_galaxy_csv_dir,
            output_plot_dir=rotation_plot_dir,
            output_residual_dir=residual_plot_dir,
            structural_mass_scale=structural_mass_scale,
        )
        records.append(result.__dict__)

    summary_df = pd.DataFrame(records)
    summary_csv = run_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    ok_count = int((summary_df["status"] == "OK").sum())
    fail_count = int((summary_df["status"] == "FAILED").sum())

    metadata_path = run_dir / "run_metadata.txt"
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("SPARC Structure Test32 Batch Run\n")
        f.write("=" * 40 + "\n")
        f.write(f"input_dir: {input_dir}\n")
        f.write(f"output_dir: {run_dir}\n")
        f.write(f"structural_mass_scale: {structural_mass_scale}\n")
        f.write(f"total_files: {len(csv_files)}\n")
        f.write(f"ok_count: {ok_count}\n")
        f.write(f"fail_count: {fail_count}\n")

    return run_dir, summary_df


# -----
# Section 10. CLI
# -----

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="sparc_structure_test32/per_galaxy 배치 계산 스크립트"
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="입력 폴더. 기본값: data/derived/crossmatch/sparc_structure_test32/per_galaxy",
    )
    parser.add_argument(
        "--output-base",
        default=None,
        help="출력 상위 폴더. 기본값: data/derived/crossmatched",
    )
    parser.add_argument(
        "--structural-mass-scale",
        type=float,
        default=1.0,
        help="구조질량 누적 스케일. 원식 그대로는 1.0",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    script_path = Path(__file__)
    project_root = project_root_from_script(script_path)

    input_dir = Path(args.input_dir) if args.input_dir else default_input_dir(project_root)
    output_base = Path(args.output_base) if args.output_base else default_output_base(project_root)

    run_dir, summary_df = run_batch(
        input_dir=input_dir,
        output_base=output_base,
        structural_mass_scale=args.structural_mass_scale,
    )

    print("[완료] 배치 계산이 끝났습니다.")
    print(f"[입력] {input_dir}")
    print(f"[출력] {run_dir}")

    ok_df = summary_df[summary_df["status"] == "OK"]
    fail_df = summary_df[summary_df["status"] == "FAILED"]

    print(f"[성공] {len(ok_df)}개")
    print(f"[실패] {len(fail_df)}개")

    if len(ok_df) > 0:
        cols = ["galaxy_name", "n_points", "rmse_kms", "mae_kms", "r2"]
        for opt in ["h_kpc_source", "v_vis_source"]:
            if opt in ok_df.columns:
                cols.append(opt)
        print("\n[성공 파일 예시]")
        print(ok_df[cols].head(10).to_string(index=False))

    if len(fail_df) > 0:
        print("\n[실패 파일]")
        print(fail_df[["galaxy_name", "message"]].to_string(index=False))


if __name__ == "__main__":
    main()
