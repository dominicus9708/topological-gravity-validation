from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt


def _sanitize_filename(name: str) -> str:
    """
    파일명에 부적절한 문자를 안전하게 치환합니다.
    """
    name = str(name).strip()
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    return name


def _unique_output_path(output_dir: str | Path, filename: str) -> Path:
    """
    같은 파일명이 이미 존재하면 _001, _002 ... suffix를 붙여
    덮어쓰지 않는 경로를 반환합니다.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(filename)
    candidate = output_dir / filename

    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix

    for i in range(1, 10000):
        alt = output_dir / f"{stem}_{i:03d}{suffix}"
        if not alt.exists():
            return alt

    raise RuntimeError(f"Could not allocate unique output path for: {candidate}")


def save_rotation_curve_plot(
    galaxy_name: str,
    r_kpc,
    v_obs_kmps,
    v_err_kmps,
    v_model_kmps,
    output_dir: str | Path,
    v_bar_kmps=None,
    v_struct_kmps=None,
    beta: float | None = None,
    rmse_value: float | None = None,
    chi2_value: float | None = None,
    reduced_chi2_value: float | None = None,
    figsize=(8, 5),
    dpi: int = 150,
) -> str:
    """
    은하 회전곡선 비교 플롯을 저장합니다.

    필수 입력:
    - galaxy_name
    - r_kpc
    - v_obs_kmps
    - v_err_kmps
    - v_model_kmps
    - output_dir

    선택 입력:
    - v_bar_kmps
    - v_struct_kmps
    - beta
    - rmse_value
    - chi2_value
    - reduced_chi2_value
    """
    r = np.asarray(r_kpc, dtype=float)
    v_obs = np.asarray(v_obs_kmps, dtype=float)
    v_err = np.asarray(v_err_kmps, dtype=float)
    v_model = np.asarray(v_model_kmps, dtype=float)

    if not (r.shape == v_obs.shape == v_err.shape == v_model.shape):
        raise ValueError("r_kpc, v_obs_kmps, v_err_kmps, and v_model_kmps must have the same shape.")

    if v_bar_kmps is not None:
        v_bar = np.asarray(v_bar_kmps, dtype=float)
        if v_bar.shape != r.shape:
            raise ValueError("v_bar_kmps must have the same shape as r_kpc.")
    else:
        v_bar = None

    if v_struct_kmps is not None:
        v_struct = np.asarray(v_struct_kmps, dtype=float)
        if v_struct.shape != r.shape:
            raise ValueError("v_struct_kmps must have the same shape as r_kpc.")
    else:
        v_struct = None

    fig, ax = plt.subplots(figsize=figsize)

    # 관측값
    ax.errorbar(
        r,
        v_obs,
        yerr=v_err,
        fmt="o",
        markersize=4,
        capsize=2,
        label="Observed",
    )

    # 모델
    ax.plot(
        r,
        v_model,
        linewidth=2,
        label="Model",
    )

    # 바리온 기준선
    if v_bar is not None:
        ax.plot(
            r,
            v_bar,
            linestyle="--",
            linewidth=1.5,
            label="Baryonic",
        )

    # 구조항의 속도 등가량(참고용)
    if v_struct is not None:
        ax.plot(
            r,
            v_struct,
            linestyle=":",
            linewidth=1.5,
            label="Structural component",
        )

    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Velocity [km/s]")
    ax.set_title(f"{galaxy_name} Rotation Curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    info_parts = []
    if beta is not None and np.isfinite(beta):
        info_parts.append(f"beta={beta:.4g}")
    if rmse_value is not None and np.isfinite(rmse_value):
        info_parts.append(f"RMSE={rmse_value:.4g}")
    if chi2_value is not None and np.isfinite(chi2_value):
        info_parts.append(f"chi2={chi2_value:.4g}")
    if reduced_chi2_value is not None and np.isfinite(reduced_chi2_value):
        info_parts.append(f"red_chi2={reduced_chi2_value:.4g}")

    if info_parts:
        info_text = " | ".join(info_parts)
        fig.text(
            0.01,
            0.01,
            info_text,
            ha="left",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    filename = f"{_sanitize_filename(galaxy_name)}_rotation_curve.png"
    outpath = _unique_output_path(output_dir, filename)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(outpath)


def save_beta_histogram(
    beta_values,
    output_dir: str | Path,
    filename: str = "beta_distribution.png",
    title: str = "Derived Beta Distribution",
    figsize=(8, 5),
    dpi: int = 150,
) -> str:
    """
    beta 분포 히스토그램 저장.
    """
    beta_arr = np.asarray(beta_values, dtype=float)
    beta_arr = beta_arr[np.isfinite(beta_arr)]

    fig, ax = plt.subplots(figsize=figsize)

    if beta_arr.size > 0:
        ax.hist(beta_arr, bins=30)
    else:
        ax.text(0.5, 0.5, "No valid beta values", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel("Beta")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    outpath = _unique_output_path(output_dir, filename)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(outpath)


def save_metric_histogram(
    metric_values,
    metric_name: str,
    output_dir: str | Path,
    filename: str | None = None,
    title: str | None = None,
    figsize=(8, 5),
    dpi: int = 150,
) -> str:
    """
    임의의 단일 metric 분포 히스토그램 저장.
    """
    arr = np.asarray(metric_values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if filename is None:
        filename = f"{_sanitize_filename(metric_name)}_distribution.png"
    if title is None:
        title = f"{metric_name} Distribution"

    fig, ax = plt.subplots(figsize=figsize)

    if arr.size > 0:
        ax.hist(arr, bins=30)
    else:
        ax.text(0.5, 0.5, "No valid metric values", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlabel(metric_name)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    outpath = _unique_output_path(output_dir, filename)
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(outpath)