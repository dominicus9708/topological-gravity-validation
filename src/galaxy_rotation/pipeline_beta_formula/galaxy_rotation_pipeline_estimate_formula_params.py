from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 0. project paths
# ---------------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parents[2]

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "galaxy_rotation" / "pipeline_beta_formula"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_ROOT / "estimated_formula_params.json"


# ---------------------------------------------------------
# 1. utilities
# ---------------------------------------------------------
def _safe_positive(x: np.ndarray, floor: float = 1.0e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    return np.maximum(x, floor)


def _find_latest_structural_summary(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    output_root = Path(output_root)

    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    candidates: list[Path] = []
    for run_dir in output_root.glob("*_structural"):
        summary_path = run_dir / "summaries" / "rotation_pipeline_summary_structural.csv"
        if summary_path.exists():
            candidates.append(summary_path)

    if not candidates:
        raise FileNotFoundError(
            "No structural summary file found under "
            f"{output_root}/*_structural/summaries/rotation_pipeline_summary_structural.csv"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_structural_summary(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structural summary not found: {path}")

    df = pd.read_csv(path)

    required = [
        "beta_raw",
        "gbar_char_logmean",
        "sigma_char_rms",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "This estimator requires an expanded structural summary with columns "
            f"{required}. Missing: {missing}"
        )

    df = df.copy()

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required)

    df = df[
        (df["beta_raw"] > 0) &
        (df["gbar_char_logmean"] > 0) &
        (df["sigma_char_rms"] > 0)
    ].copy()

    if df.empty:
        raise ValueError("No valid positive rows remain in structural summary.")

    return df.reset_index(drop=True)


# ---------------------------------------------------------
# 2. derive beta-side parameters
# ---------------------------------------------------------
def _derive_beta_params_from_structural_summary(
    df: pd.DataFrame,
    beta_max: float = 2000.0,
) -> dict[str, float]:
    """
    log(beta_raw / beta_max)
    ~= log(lambda_beta) + p log(g/g0) + q log(sigma/sigma0)
    """
    g = _safe_positive(df["gbar_char_logmean"].to_numpy(dtype=float))
    s = _safe_positive(df["sigma_char_rms"].to_numpy(dtype=float))
    beta_raw = _safe_positive(df["beta_raw"].to_numpy(dtype=float))

    g0 = float(np.median(g))
    sigma0 = float(np.median(s))

    x1 = np.log(g / g0)
    x2 = np.log(s / sigma0)

    z_target = beta_raw / float(beta_max)
    y = np.log(_safe_positive(z_target))

    X = np.column_stack([
        np.ones_like(x1),
        x1,
        x2,
    ])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, p_raw, q_raw = coef

    lambda_beta_raw = float(np.exp(intercept))

    p = float(max(p_raw, 0.0))
    q = float(max(q_raw, 0.0))
    lambda_beta = float(max(lambda_beta_raw, 1.0e-6))

    if q < p:
        q = p

    y_hat = X @ np.array([np.log(lambda_beta), p, q])
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "g0": g0,
        "sigma0": sigma0,
        "p": p,
        "q": q,
        "lambda_beta": lambda_beta,
        "beta_regression_r2": r2,
        "n_used_galaxies": int(len(df)),
    }


# ---------------------------------------------------------
# 3. softened cap hyperparameters
# ---------------------------------------------------------
def _derive_dynamic_cap_params_from_beta_params(
    p: float,
    q: float,
) -> dict[str, float]:
    """
    지나치게 억제적인 cap을 피하기 위한 완화형 전역 하이퍼파라미터.

    원칙:
    - sigma 쪽 지수는 beta식의 q보다 완만해야 함
    - alpha_mix는 0과 1 사이의 중간값으로 둬서
      sigma가 커져도 상한이 무너지지 않게 함
    - floor는 너무 낮지 않게 두어 바리온 회귀를 방지
    """
    cap_eta_g = 1.0
    cap_eta_sigma = float(np.clip(0.5 * q, 0.55, 0.95))
    cap_mix_alpha = 0.60
    coherence_floor = 0.35

    return {
        "cap_eta_g": float(cap_eta_g),
        "cap_eta_sigma": float(cap_eta_sigma),
        "cap_mix_alpha": float(cap_mix_alpha),
        "coherence_floor": float(coherence_floor),
    }


# ---------------------------------------------------------
# 4. sigma weight params
# ---------------------------------------------------------
def _derive_sigma_weight_params_from_beta_params(
    p: float,
    q: float,
    lambda_beta: float,
) -> dict[str, float]:
    u = max(0.25, 0.5 * float(p))
    v = max(0.25, 0.5 * float(q))
    if v < u:
        v = u

    lambda_w = max(0.1, float(np.sqrt(max(lambda_beta, 1.0e-12))))

    return {
        "u": float(u),
        "v": float(v),
        "lambda_w": float(lambda_w),
    }


# ---------------------------------------------------------
# 5. public API
# ---------------------------------------------------------
def estimate_formula_params(
    structural_summary_path: str | Path | None = None,
    output_json_path: str | Path | None = DEFAULT_OUTPUT_JSON,
    beta_max: float = 2000.0,
) -> dict[str, float]:
    if structural_summary_path is None:
        structural_summary_path = _find_latest_structural_summary(DEFAULT_OUTPUT_ROOT)
    else:
        structural_summary_path = Path(structural_summary_path)

    df = _load_structural_summary(structural_summary_path)

    beta_params = _derive_beta_params_from_structural_summary(
        df=df,
        beta_max=beta_max,
    )
    cap_params = _derive_dynamic_cap_params_from_beta_params(
        p=beta_params["p"],
        q=beta_params["q"],
    )
    sigma_params = _derive_sigma_weight_params_from_beta_params(
        p=beta_params["p"],
        q=beta_params["q"],
        lambda_beta=beta_params["lambda_beta"],
    )

    result = {
        "structural_summary_path": str(structural_summary_path),
        "formula_g0": float(beta_params["g0"]),
        "formula_sigma0": float(beta_params["sigma0"]),
        "formula_p": float(beta_params["p"]),
        "formula_q": float(beta_params["q"]),
        "formula_lambda_beta": float(beta_params["lambda_beta"]),
        "formula_cap_eta_g": float(cap_params["cap_eta_g"]),
        "formula_cap_eta_sigma": float(cap_params["cap_eta_sigma"]),
        "formula_cap_mix_alpha": float(cap_params["cap_mix_alpha"]),
        "formula_coherence_floor": float(cap_params["coherence_floor"]),
        "sigma_weight_u": float(sigma_params["u"]),
        "sigma_weight_v": float(sigma_params["v"]),
        "sigma_weight_lambda": float(sigma_params["lambda_w"]),
        "formula_g_choice": "gbar_char_logmean",
        "formula_sigma_choice": "sigma_char_rms",
        "sigma_weight_g_choice": "gbar_char_logmean",
        "sigma_weight_sigma_choice": "sigma_char_rms",
        "beta_regression_r2": float(beta_params["beta_regression_r2"]),
        "n_used_galaxies": int(beta_params["n_used_galaxies"]),
        "derivation_note": (
            "The beta driver is estimated from the structural summary through "
            "log(beta_raw/beta_max) ~= log(lambda_beta) + p log(g/g0) + q log(sigma/sigma0). "
            "The dynamic cap is softened to avoid over-suppressing the structural term: "
            "it uses a mixed coherence ratio (g_hat + alpha*s_hat)/(g_hat + s_hat) "
            "instead of the stronger g_hat/(g_hat + s_hat), with a non-negligible floor."
        ),
    }

    if output_json_path is not None:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# ---------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    estimated = estimate_formula_params(
        structural_summary_path=None,
        output_json_path=DEFAULT_OUTPUT_JSON,
        beta_max=2000.0,
    )
    print(json.dumps(estimated, indent=2, ensure_ascii=False))