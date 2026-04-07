from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# ============================================================
# Our Galaxy Halo Stellar Kinematics
# Topological pipeline 006
# ------------------------------------------------------------
# Placement:
#   src/Our galaxy Halo Stellar Kinematics/topological/
#     run_topological_our_galaxy_halo_rrlyrae_006.py
#
# Input:
#   data/derived/Our galaxy Halo Stellar Kinematics/input/
#
# Optional comparison input:
#   results/Our galaxy Halo Stellar Kinematics/output/standard/
#
# Output:
#   results/Our galaxy Halo Stellar Kinematics/output/topological/YYYYMMDD_HHMMSS/
#
# Revision intent:
# - Final input remains fixed and is not re-filtered here.
# - Background reference is written as D_bg = 3.0.
# - Hard clipping of proxy values is reduced.
# - Stabilization is moved toward quality-weighted robust aggregation.
# - Pointwise local sigma is kept for diagnostics only.
# - Coupling-related quantities are retained only as diagnostics.
# ============================================================

KPC_PER_MAS_PARALLAX = 1.0
KM_S_PER_MASYR_KPC = 4.74047
SHELL_BINS_KPC = [5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
SHELL_LABELS = ["5-10", "10-20", "20-40", "40-80", "80+"]
K1 = 8
K2 = 32
EPS = 1e-12
D_BG = 3.0

# Stabilization settings are now focused on robust weighting rather than aggressive clipping.
MIN_SHELL_N_FOR_GRADIENT = 20
MIN_SHELL_N_FOR_COUPLING_DIAG = 20
WINSOR_Q_LOW = 0.10
WINSOR_Q_HIGH = 0.90
SOFT_WEIGHT_POWER = 2.0


@dataclass
class DatasetConfig:
    name: str
    is_6d: bool
    input_filename: str
    standard_filename: str
    shell_value_column: str


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


def finite_quantile(arr: np.ndarray, q: float, fallback: float = 1.0) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return fallback
    val = float(np.nanquantile(arr, q))
    if not np.isfinite(val) or val <= EPS:
        return fallback
    return val


def signed_asinh_scale(values: np.ndarray, scale: float) -> np.ndarray:
    scale = max(float(scale), EPS)
    out = np.full_like(values, np.nan, dtype=float)
    m = np.isfinite(values)
    out[m] = np.arcsinh(values[m] / scale)
    return out


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return np.nan
    v = values[m]
    w = weights[m]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    return float(np.interp(q, cdf, v))


def weighted_winsorized_mean(values: np.ndarray, weights: np.ndarray, q_low: float = WINSOR_Q_LOW, q_high: float = WINSOR_Q_HIGH) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(m):
        return np.nan
    v = values[m]
    w = weights[m]
    lo = weighted_quantile(v, w, q_low)
    hi = weighted_quantile(v, w, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.nan
    vw = np.clip(v, lo, hi)
    return float(np.average(vw, weights=w))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return weighted_quantile(values, weights, 0.5)


def add_proxy_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()

    par = safe_numeric(out.get("parallax_mas"))
    out["parallax_mas_for_proxy"] = np.where(par > 0, par, np.nan)
    out["distance_proxy_kpc_stable"] = np.where(
        np.isfinite(out["parallax_mas_for_proxy"]),
        KPC_PER_MAS_PARALLAX / out["parallax_mas_for_proxy"],
        np.nan,
    )

    if "distance_proxy_kpc" not in out.columns:
        out["distance_proxy_kpc"] = out["distance_proxy_kpc_stable"]

    if "gal_l_deg" not in out.columns or "gal_b_deg" not in out.columns:
        ra_deg = safe_numeric(out.get("ra_deg"))
        dec_deg = safe_numeric(out.get("dec_deg"))
        ra = np.deg2rad(ra_deg)
        dec = np.deg2rad(dec_deg)

        alpha_ngp = np.deg2rad(192.85948)
        delta_ngp = np.deg2rad(27.12825)
        l_omega = np.deg2rad(32.93192)

        sin_b = (
            np.sin(dec) * np.sin(delta_ngp)
            + np.cos(dec) * np.cos(delta_ngp) * np.cos(ra - alpha_ngp)
        )
        b = np.arcsin(np.clip(sin_b, -1.0, 1.0))
        y = np.cos(dec) * np.sin(ra - alpha_ngp)
        x = (
            np.sin(dec) * np.cos(delta_ngp)
            - np.cos(dec) * np.sin(delta_ngp) * np.cos(ra - alpha_ngp)
        )
        l = np.arctan2(y, x) + l_omega
        l = np.mod(l, 2.0 * np.pi)

        out["gal_l_deg"] = np.rad2deg(l)
        out["gal_b_deg"] = np.rad2deg(b)

    dist = safe_numeric(out.get("distance_proxy_kpc_stable"))
    pmra = safe_numeric(out.get("pmra_masyr"))
    pmdec = safe_numeric(out.get("pmdec_masyr"))

    out["vt_ra_proxy_kms_stable"] = KM_S_PER_MASYR_KPC * pmra * dist
    out["vt_dec_proxy_kms_stable"] = KM_S_PER_MASYR_KPC * pmdec * dist
    out["vt_total_proxy_kms_stable"] = np.sqrt(
        np.square(safe_numeric(out.get("vt_ra_proxy_kms_stable")))
        + np.square(safe_numeric(out.get("vt_dec_proxy_kms_stable")))
    )

    if is_6d:
        rv = safe_numeric(out.get("radial_velocity_kms"))
        out["speed_total_proxy_kms_stable"] = np.sqrt(
            np.square(safe_numeric(out.get("vt_total_proxy_kms_stable")))
            + np.square(rv)
        )
    else:
        out["speed_total_proxy_kms_stable"] = np.nan

    return out


def add_shells(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dist = safe_numeric(out.get("distance_proxy_kpc_stable"))
    out["topological_shell"] = pd.cut(
        dist, bins=SHELL_BINS_KPC, labels=SHELL_LABELS, right=False
    )
    return out


def cartesian_from_galactic(df: pd.DataFrame) -> np.ndarray:
    l = np.deg2rad(safe_numeric(df.get("gal_l_deg")).to_numpy())
    b = np.deg2rad(safe_numeric(df.get("gal_b_deg")).to_numpy())
    r = safe_numeric(df.get("distance_proxy_kpc_stable")).to_numpy()

    x = r * np.cos(b) * np.cos(l)
    y = r * np.cos(b) * np.sin(l)
    z = r * np.sin(b)
    return np.column_stack([x, y, z])


def robust_scale_matrix(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    med = np.nanmedian(out, axis=0)
    q25 = np.nanpercentile(out, 25, axis=0)
    q75 = np.nanpercentile(out, 75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > EPS), scale, 1.0)
    out = (out - med) / scale
    out[~np.isfinite(out)] = np.nan
    return out


def pairwise_knn_dimension(features: np.ndarray) -> np.ndarray:
    n = features.shape[0]
    result = np.full(n, np.nan, dtype=float)

    valid_rows = np.all(np.isfinite(features), axis=1)
    idx_valid = np.where(valid_rows)[0]
    if len(idx_valid) <= K2:
        return result

    f = features[idx_valid]
    diff = f[:, None, :] - f[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist_sorted = np.sort(dist, axis=1)

    r1 = dist_sorted[:, K1 - 1]
    r2 = dist_sorted[:, K2 - 1]

    good = (r1 > EPS) & (r2 > r1)
    dim = np.full(len(idx_valid), np.nan, dtype=float)
    dim[good] = (np.log(K2) - np.log(K1)) / (np.log(r2[good]) - np.log(r1[good]))
    result[idx_valid] = dim
    return result


def build_position_features(df: pd.DataFrame) -> np.ndarray:
    xyz = cartesian_from_galactic(df)
    r = safe_numeric(df.get("distance_proxy_kpc_stable")).to_numpy()
    r_scale = finite_quantile(r, 0.75, fallback=10.0)
    xyz_soft = signed_asinh_scale(xyz, r_scale)
    return robust_scale_matrix(xyz_soft)


def build_kinematic_features(df: pd.DataFrame, is_6d: bool) -> np.ndarray:
    vt_ra = safe_numeric(df.get("vt_ra_proxy_kms_stable")).to_numpy()
    vt_dec = safe_numeric(df.get("vt_dec_proxy_kms_stable")).to_numpy()
    v_scale = finite_quantile(np.abs(np.concatenate([vt_ra[np.isfinite(vt_ra)], vt_dec[np.isfinite(vt_dec)]])), 0.75, fallback=100.0)

    if is_6d:
        rv = safe_numeric(df.get("radial_velocity_kms")).to_numpy()
        arr = np.column_stack([
            signed_asinh_scale(vt_ra, v_scale),
            signed_asinh_scale(vt_dec, v_scale),
            signed_asinh_scale(rv, v_scale),
        ])
    else:
        arr = np.column_stack([
            signed_asinh_scale(vt_ra, v_scale),
            signed_asinh_scale(vt_dec, v_scale),
        ])
    return robust_scale_matrix(arr)


def add_quality_weights(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()

    par = safe_numeric(out.get("parallax_mas"))
    par_err = safe_numeric(out.get("parallax_error_mas"))
    pmra = safe_numeric(out.get("pmra_masyr"))
    pmra_err = safe_numeric(out.get("pmra_error_masyr"))
    pmdec = safe_numeric(out.get("pmdec_masyr"))
    pmdec_err = safe_numeric(out.get("pmdec_error_masyr"))

    eta_pos = np.abs(par_err) / (np.abs(par) + EPS)
    eta_pm = np.sqrt(
        np.square(pmra_err / (np.abs(pmra) + EPS))
        + np.square(pmdec_err / (np.abs(pmdec) + EPS))
    )

    q_pos = 1.0 / (1.0 + np.power(eta_pos, SOFT_WEIGHT_POWER))
    q_kin = 1.0 / (1.0 + np.power(eta_pm, SOFT_WEIGHT_POWER))

    if is_6d:
        rv = safe_numeric(out.get("radial_velocity_kms"))
        rv_err = safe_numeric(out.get("radial_velocity_error_kms"))
        eta_rv = np.abs(rv_err) / (np.abs(rv) + EPS)
        q_rv = 1.0 / (1.0 + np.power(eta_rv, SOFT_WEIGHT_POWER))
        q_kin = q_kin * q_rv

    dist = safe_numeric(out.get("distance_proxy_kpc_stable")).to_numpy()
    vt = safe_numeric(out.get("vt_total_proxy_kms_stable")).to_numpy()
    dist_scale = finite_quantile(dist, 0.90, fallback=20.0)
    vt_scale = finite_quantile(np.abs(vt), 0.90, fallback=300.0)

    q_dist_soft = 1.0 / (1.0 + np.power(np.where(np.isfinite(dist), dist / dist_scale, np.nan), SOFT_WEIGHT_POWER))
    q_vt_soft = 1.0 / (1.0 + np.power(np.where(np.isfinite(vt), np.abs(vt) / vt_scale, np.nan), SOFT_WEIGHT_POWER))

    q_pos = q_pos * q_dist_soft
    q_kin = q_kin * q_vt_soft

    lambda_pos = q_pos / (q_pos + q_kin + EPS)

    out["q_pos"] = q_pos
    out["q_kin"] = q_kin
    out["lambda_pos"] = lambda_pos
    out["quality_weight"] = np.sqrt(q_pos * q_kin)
    return out


def add_local_dimensions(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()

    pos_features = build_position_features(out)
    kin_features = build_kinematic_features(out, is_6d=is_6d)

    d_pos = pairwise_knn_dimension(pos_features)
    d_kin = pairwise_knn_dimension(kin_features)

    out["D_loc_pos"] = d_pos
    out["D_loc_kin"] = d_kin

    lam = safe_numeric(out.get("lambda_pos")).to_numpy()
    out["D_loc"] = lam * d_pos + (1.0 - lam) * d_kin
    return out


def attach_standard_columns(df: pd.DataFrame, standard_path: Path) -> pd.DataFrame:
    out = df.copy()
    if not standard_path.exists():
        out["standard_available"] = False
        return out

    std = read_csv(standard_path)
    keep = ["source_id"]
    for col in [
        "distance_proxy_kpc",
        "vt_total_proxy_kms",
        "speed_total_proxy_kms",
        "standard_vt_proxy_centered_kms",
        "standard_rv_centered_kms",
        "standard_speed_proxy_centered_kms",
        "standard_distance_shell_kpc",
    ]:
        if col in std.columns:
            keep.append(col)

    std = std[keep].copy()
    rename = {c: f"std_{c}" for c in std.columns if c != "source_id"}
    std = std.rename(columns=rename)

    out = out.merge(std, on="source_id", how="left")
    out["standard_available"] = True
    return out



def choose_observed_series(sub: pd.DataFrame, is_6d: bool) -> tuple[pd.Series, str]:
    if is_6d:
        candidates = [
            "speed_total_proxy_kms_stable",
            "speed_total_proxy_kms",
        ]
    else:
        candidates = [
            "vt_total_proxy_kms_stable",
            "vt_total_proxy_kms",
        ]
    for col in candidates:
        if col in sub.columns:
            s = safe_numeric(sub.get(col))
            if s.notna().sum() > 1:
                return s, col
    return pd.Series(dtype=float), "missing"


def _series_effectively_same(a: pd.Series, b: pd.Series, atol: float = 1e-9, rtol: float = 1e-7) -> bool:
    ax = pd.to_numeric(a, errors="coerce").to_numpy(dtype=float)
    bx = pd.to_numeric(b, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(ax) & np.isfinite(bx)
    if m.sum() < 2:
        return False
    return bool(np.allclose(ax[m], bx[m], atol=atol, rtol=rtol))


def choose_standard_series(sub: pd.DataFrame, is_6d: bool, observed_series: pd.Series | None = None) -> tuple[pd.Series, str]:
    if is_6d:
        primary = [
            "std_standard_speed_proxy_centered_kms",
            "std_speed_total_proxy_kms",
            "std_standard_rv_centered_kms",
        ]
        fallback = [
            "std_speed_total_proxy_kms",
            "std_standard_speed_proxy_centered_kms",
            "std_standard_rv_centered_kms",
        ]
    else:
        primary = [
            "std_standard_vt_proxy_centered_kms",
            "std_vt_total_proxy_kms",
        ]
        fallback = [
            "std_vt_total_proxy_kms",
            "std_standard_vt_proxy_centered_kms",
        ]

    chosen = None
    chosen_name = "missing"
    for col in primary:
        if col in sub.columns:
            s = safe_numeric(sub.get(col))
            if s.notna().sum() > 1:
                chosen = s
                chosen_name = col
                break

    if chosen is None:
        for col in fallback:
            if col in sub.columns:
                s = safe_numeric(sub.get(col))
                if s.notna().sum() > 1:
                    chosen = s
                    chosen_name = col
                    break

    if chosen is None:
        return pd.Series(dtype=float), "missing"

    if observed_series is not None and _series_effectively_same(chosen, observed_series):
        for col in fallback:
            if col == chosen_name or col not in sub.columns:
                continue
            alt = safe_numeric(sub.get(col))
            if alt.notna().sum() > 1 and not _series_effectively_same(alt, observed_series):
                return alt, col

    return chosen, chosen_name


def shell_centers_from_labels():
    return {"5-10": 7.5, "10-20": 15.0, "20-40": 30.0, "40-80": 60.0, "80+": 100.0}


def build_shell_summary(df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
    rows = []
    centers_map = shell_centers_from_labels()

    for shell in SHELL_LABELS:
        sub = df[df["topological_shell"].astype(str) == shell].copy()
        n = len(sub)

        if n == 0:
            rows.append({
                "dataset": config.name,
                "shell": shell,
                "shell_center_kpc": centers_map[shell],
                "n_shell": 0,
                "D_bg_reference": D_BG,
                "D_halo_shell": np.nan,
                "Sigma_bg_shell": np.nan,
                "sigma_local_mean_shell": np.nan,
                "sigma_local_median_shell": np.nan,
                "observed_series_source": "missing",
                "standard_series_source": "missing",
                "observed_variance_shell": np.nan,
                "standard_variance_shell": np.nan,
            })
            continue

        q = safe_numeric(sub.get("quality_weight")).to_numpy()
        w = np.where(np.isfinite(q) & (q > 0), q, 1.0)

        dloc = safe_numeric(sub.get("D_loc")).to_numpy()
        d_halo = weighted_winsorized_mean(dloc, w)
        sigma_bg_shell = d_halo - D_BG if np.isfinite(d_halo) else np.nan

        sigma_local = dloc - d_halo
        sigma_local_mean = weighted_winsorized_mean(sigma_local, w)
        sigma_local_median = weighted_median(sigma_local, w)

        observed_series, observed_source = choose_observed_series(sub, is_6d=config.is_6d)
        observed_var = float(np.nanvar(observed_series, ddof=1)) if observed_series.notna().sum() > 1 else np.nan

        std_series, standard_source = choose_standard_series(
            sub,
            is_6d=config.is_6d,
            observed_series=observed_series,
        )
        standard_var = float(np.nanvar(std_series, ddof=1)) if std_series.notna().sum() > 1 else np.nan

        rows.append({
            "dataset": config.name,
            "shell": shell,
            "shell_center_kpc": centers_map[shell],
            "n_shell": int(n),
            "D_bg_reference": D_BG,
            "D_halo_shell": d_halo,
            "Sigma_bg_shell": sigma_bg_shell,
            "sigma_local_mean_shell": sigma_local_mean,
            "sigma_local_median_shell": sigma_local_median,
            "observed_series_source": observed_source,
            "standard_series_source": standard_source,
            "observed_variance_shell": observed_var,
            "standard_variance_shell": standard_var,
        })

    shell_df = pd.DataFrame(rows)

    grad_bg = np.full(len(shell_df), np.nan, dtype=float)
    coupling_diag = np.full(len(shell_df), np.nan, dtype=float)
    coupling_valid = np.full(len(shell_df), False, dtype=bool)

    r = pd.to_numeric(shell_df["shell_center_kpc"], errors="coerce").to_numpy()
    sigma_bg = pd.to_numeric(shell_df["Sigma_bg_shell"], errors="coerce").to_numpy()
    std_var = pd.to_numeric(shell_df["standard_variance_shell"], errors="coerce").to_numpy()
    obs_var = pd.to_numeric(shell_df["observed_variance_shell"], errors="coerce").to_numpy()
    n_shell = pd.to_numeric(shell_df["n_shell"], errors="coerce").to_numpy()

    for i in range(len(shell_df) - 1):
        enough_n = (n_shell[i] >= MIN_SHELL_N_FOR_GRADIENT) and (n_shell[i + 1] >= MIN_SHELL_N_FOR_GRADIENT)
        if enough_n and np.isfinite(r[i]) and np.isfinite(r[i + 1]) and (r[i + 1] > r[i]):
            if np.isfinite(sigma_bg[i]) and np.isfinite(sigma_bg[i + 1]):
                grad_bg[i] = abs((sigma_bg[i + 1] - sigma_bg[i]) / (r[i + 1] - r[i]))

    finite_grad = np.where(np.isfinite(grad_bg))[0]
    if len(finite_grad) > 0 and not np.isfinite(grad_bg[-1]):
        grad_bg[-1] = grad_bg[finite_grad[-1]]

    for i in range(len(shell_df)):
        enough_n = n_shell[i] >= MIN_SHELL_N_FOR_COUPLING_DIAG
        has_neighbor = False
        if i < len(shell_df) - 1 and np.isfinite(sigma_bg[i]) and np.isfinite(sigma_bg[i + 1]) and n_shell[i + 1] >= MIN_SHELL_N_FOR_COUPLING_DIAG:
            has_neighbor = True
        if enough_n and has_neighbor and np.isfinite(grad_bg[i]) and grad_bg[i] > EPS and np.isfinite(obs_var[i]) and np.isfinite(std_var[i]):
            coupling_diag[i] = (obs_var[i] - std_var[i]) / grad_bg[i]
            coupling_valid[i] = True

    shell_df["topological_gradient_term_bg"] = grad_bg
    shell_df["coupling_shell_diagnostic"] = coupling_diag
    shell_df["coupling_shell_diagnostic_valid"] = coupling_valid
    shell_df["residual_observed_minus_standard"] = shell_df["observed_variance_shell"] - shell_df["standard_variance_shell"]
    return shell_df


def summarize_dataset(df: pd.DataFrame, shell_df: pd.DataFrame, config: DatasetConfig) -> dict:
    sigma_bg = pd.to_numeric(shell_df.get("Sigma_bg_shell"), errors="coerce")
    grad_bg = pd.to_numeric(shell_df.get("topological_gradient_term_bg"), errors="coerce")
    coupling = pd.to_numeric(shell_df.get("coupling_shell_diagnostic"), errors="coerce")
    valid = shell_df.get("coupling_shell_diagnostic_valid")
    return {
        "dataset": config.name,
        "rows_read": int(len(df)),
        "rows_written": int(len(df)),
        "D_loc_pos_nonnull": int(pd.to_numeric(df["D_loc_pos"], errors="coerce").notna().sum()),
        "D_loc_kin_nonnull": int(pd.to_numeric(df["D_loc_kin"], errors="coerce").notna().sum()),
        "D_loc_nonnull": int(pd.to_numeric(df["D_loc"], errors="coerce").notna().sum()),
        "shells_nonempty": int((pd.to_numeric(shell_df["n_shell"], errors="coerce") > 0).sum()),
        "median_D_loc": float(np.nanmedian(pd.to_numeric(df["D_loc"], errors="coerce"))) if "D_loc" in df.columns else np.nan,
        "median_lambda_pos": float(np.nanmedian(pd.to_numeric(df["lambda_pos"], errors="coerce"))) if "lambda_pos" in df.columns else np.nan,
        "median_Sigma_bg_shell": float(np.nanmedian(sigma_bg)) if sigma_bg.notna().any() else np.nan,
        "median_topological_gradient_term_bg": float(np.nanmedian(grad_bg)) if grad_bg.notna().any() else np.nan,
        "median_coupling_shell_diagnostic": float(np.nanmedian(coupling)) if coupling.notna().any() else np.nan,
        "coupling_shell_diagnostic_valid_n": int(np.sum(valid.astype(bool))) if valid is not None else 0,
    }


def run_one_dataset(project_root: Path, config: DatasetConfig, output_dir: Path) -> dict:
    input_dir = project_root / "data" / "derived" / "Our galaxy Halo Stellar Kinematics" / "input"
    standard_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "standard"

    input_path = input_dir / config.input_filename
    standard_path = standard_dir / config.standard_filename

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = read_csv(input_path)
    df = add_proxy_columns(df, is_6d=config.is_6d)
    df = add_shells(df)
    df = add_quality_weights(df, is_6d=config.is_6d)
    df = add_local_dimensions(df, is_6d=config.is_6d)
    df = attach_standard_columns(df, standard_path)

    shell_df = build_shell_summary(df, config=config)

    shell_to_d_halo = dict(zip(shell_df["shell"], shell_df["D_halo_shell"]))
    shell_to_sigma_bg = dict(zip(shell_df["shell"], shell_df["Sigma_bg_shell"]))
    df["D_halo_shell"] = df["topological_shell"].astype(str).map(shell_to_d_halo)
    df["Sigma_bg_shell"] = df["topological_shell"].astype(str).map(shell_to_sigma_bg)
    df["sigma_local"] = pd.to_numeric(df["D_loc"], errors="coerce") - pd.to_numeric(df["D_halo_shell"], errors="coerce")
    df["D_bg_reference"] = D_BG
    df["topological_preserved_input"] = True

    out_detail = output_dir / f"gaia_rrlyrae_{config.name}_topological.csv"
    out_shell = output_dir / f"gaia_rrlyrae_{config.name}_topological_shells.csv"

    df.to_csv(out_detail, index=False)
    shell_df.to_csv(out_shell, index=False)

    return summarize_dataset(df, shell_df, config=config)


def main() -> None:
    project_root = find_project_root(Path(__file__))
    timestamp = timestamp_folder_name()
    output_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "topological" / timestamp
    ensure_dir(output_dir)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Output dir: {output_dir}")

    configs = [
        DatasetConfig("5d", False, "gaia_rrlyrae_5d_input.csv", "gaia_rrlyrae_5d_standard.csv", "vt_total_proxy_kms_stable"),
        DatasetConfig("6d", True, "gaia_rrlyrae_6d_input.csv", "gaia_rrlyrae_6d_standard.csv", "speed_total_proxy_kms_stable"),
    ]

    summaries = []
    for cfg in configs:
        summaries.append(run_one_dataset(project_root, cfg, output_dir))

    pd.DataFrame(summaries).to_csv(output_dir / "topological_summary.csv", index=False)

    readme = """Our galaxy Halo Stellar Kinematics - topological stage 006

Changes in this revision
- Output is written into a timestamped subfolder.
- Final input remains fixed and is not re-filtered.
- Background reference is written as D_bg = 3.0.
- Hard clipping of proxy values is reduced.
- Stabilization is moved toward quality-weighted robust aggregation.
- Local pointwise sigma is retained for diagnostics only.
- Coupling-related quantities are retained only as diagnostics.

Stabilization logic in 005
- Positive-parallax proxy use is retained, but hard max clipping is reduced.
- Position and velocity features use soft asinh compression before robust scaling.
- Shell structural estimates use weighted winsorized aggregation.
- Coupling diagnostics are reported only when shell support is sufficient.

Parameters
- MIN_SHELL_N_FOR_GRADIENT = {MIN_SHELL_N_FOR_GRADIENT}
- MIN_SHELL_N_FOR_COUPLING_DIAG = {MIN_SHELL_N_FOR_COUPLING_DIAG}
- WINSOR_Q_LOW = {WINSOR_Q_LOW}
- WINSOR_Q_HIGH = {WINSOR_Q_HIGH}
- SOFT_WEIGHT_POWER = {SOFT_WEIGHT_POWER}
""".format(
        MIN_SHELL_N_FOR_GRADIENT=MIN_SHELL_N_FOR_GRADIENT,
        MIN_SHELL_N_FOR_COUPLING_DIAG=MIN_SHELL_N_FOR_COUPLING_DIAG,
        WINSOR_Q_LOW=WINSOR_Q_LOW,
        WINSOR_Q_HIGH=WINSOR_Q_HIGH,
        SOFT_WEIGHT_POWER=SOFT_WEIGHT_POWER,
    )
    (output_dir / "README_topological.txt").write_text(readme, encoding="utf-8")

    print("[DONE] Saved:")
    print(f" - {output_dir / 'gaia_rrlyrae_5d_topological.csv'}")
    print(f" - {output_dir / 'gaia_rrlyrae_5d_topological_shells.csv'}")
    print(f" - {output_dir / 'gaia_rrlyrae_6d_topological.csv'}")
    print(f" - {output_dir / 'gaia_rrlyrae_6d_topological_shells.csv'}")
    print(f" - {output_dir / 'topological_summary.csv'}")
    print(f" - {output_dir / 'README_topological.txt'}")


if __name__ == "__main__":
    main()
