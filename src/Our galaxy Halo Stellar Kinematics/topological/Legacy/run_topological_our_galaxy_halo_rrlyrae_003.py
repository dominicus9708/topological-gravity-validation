from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd

# ============================================================
# Our Galaxy Halo Stellar Kinematics
# Topological pipeline 003
# ------------------------------------------------------------
# Placement:
#   src/Our galaxy Halo Stellar Kinematics/topological/
#     run_topological_our_galaxy_halo_rrlyrae_003.py
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
# - Background reference is written as D_bg = 3.0 rather than
#   an ontological "outside galaxy = 3" claim.
# - The topological driver is coarse-grained shell structure:
#       Sigma_bg_shell(r) = D_halo_shell(r) - D_bg
#   and the main gradient term is d(Sigma_bg_shell)/dr.
# - Pointwise local sigma is kept for diagnostics only:
#       sigma_local = D_loc - D_halo_shell
# - shell-by-shell xi*c_info^2 back-solving is retained only
#   as a diagnostic; the main prediction uses a single global
#   A = xi*c_info^2 estimated from stable 6D shells and then
#   projected to both 6D and 5D summaries.
# ============================================================

KPC_PER_MAS_PARALLAX = 1.0
KM_S_PER_MASYR_KPC = 4.74047
SHELL_BINS_KPC = [5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
SHELL_LABELS = ["5-10", "10-20", "20-40", "40-80", "80+"]
K1 = 8
K2 = 32
EPS = 1e-12

# Structural background reference adopted in the current
# observation-frame pipeline.
D_BG = 3.0

# Stabilization settings for derived quantities only.
MIN_PARALLAX_MAS = 0.02
MAX_DISTANCE_PROXY_KPC = 50.0
MAX_VT_PROXY_KMS = 1000.0
MIN_SHELL_N_FOR_GRADIENT = 20
MIN_SHELL_N_FOR_GLOBAL_A = 20


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


def add_proxy_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()

    par = safe_numeric(out.get("parallax_mas"))
    out["parallax_mas_for_proxy"] = np.where(par >= MIN_PARALLAX_MAS, par, np.nan)
    out["distance_proxy_kpc_stable"] = np.where(
        np.isfinite(out["parallax_mas_for_proxy"]),
        KPC_PER_MAS_PARALLAX / out["parallax_mas_for_proxy"],
        np.nan,
    )
    out["distance_proxy_kpc_stable"] = np.clip(
        out["distance_proxy_kpc_stable"],
        a_min=None,
        a_max=MAX_DISTANCE_PROXY_KPC,
    )

    if "distance_proxy_kpc" not in out.columns:
        out["distance_proxy_kpc"] = np.where(par > 0, KPC_PER_MAS_PARALLAX / par, np.nan)

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

    dist_stable = safe_numeric(out.get("distance_proxy_kpc_stable"))
    pmra = safe_numeric(out.get("pmra_masyr"))
    pmdec = safe_numeric(out.get("pmdec_masyr"))

    out["vt_ra_proxy_kms_stable"] = KM_S_PER_MASYR_KPC * pmra * dist_stable
    out["vt_dec_proxy_kms_stable"] = KM_S_PER_MASYR_KPC * pmdec * dist_stable
    out["vt_total_proxy_kms_stable"] = np.sqrt(
        np.square(safe_numeric(out.get("vt_ra_proxy_kms_stable")))
        + np.square(safe_numeric(out.get("vt_dec_proxy_kms_stable")))
    )
    out["vt_total_proxy_kms_stable"] = np.clip(
        out["vt_total_proxy_kms_stable"],
        a_min=None,
        a_max=MAX_VT_PROXY_KMS,
    )

    if is_6d:
        rv = safe_numeric(out.get("radial_velocity_kms"))
        out["speed_total_proxy_kms_stable"] = np.sqrt(
            np.square(safe_numeric(out.get("vt_total_proxy_kms_stable")))
            + np.square(rv)
        )
        out["speed_total_proxy_kms_stable"] = np.clip(
            out["speed_total_proxy_kms_stable"],
            a_min=None,
            a_max=MAX_VT_PROXY_KMS,
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
    return robust_scale_matrix(xyz)


def build_kinematic_features(df: pd.DataFrame, is_6d: bool) -> np.ndarray:
    if is_6d:
        arr = np.column_stack([
            safe_numeric(df.get("vt_ra_proxy_kms_stable")).to_numpy(),
            safe_numeric(df.get("vt_dec_proxy_kms_stable")).to_numpy(),
            safe_numeric(df.get("radial_velocity_kms")).to_numpy(),
        ])
    else:
        arr = np.column_stack([
            safe_numeric(df.get("vt_ra_proxy_kms_stable")).to_numpy(),
            safe_numeric(df.get("vt_dec_proxy_kms_stable")).to_numpy(),
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

    eta_pos = np.sqrt(np.square(par_err / (np.abs(par) + EPS)))
    eta_pm = np.sqrt(
        np.square(pmra_err / (np.abs(pmra) + EPS))
        + np.square(pmdec_err / (np.abs(pmdec) + EPS))
    )

    q_pos = 1.0 / (1.0 + np.square(eta_pos))
    q_kin = 1.0 / (1.0 + np.square(eta_pm))

    if is_6d:
        rv = safe_numeric(out.get("radial_velocity_kms"))
        rv_err = safe_numeric(out.get("radial_velocity_error_kms"))
        eta_rv = rv_err / (np.abs(rv) + EPS)
        q_rv = 1.0 / (1.0 + np.square(eta_rv))
        q_kin = q_kin * q_rv

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


def shell_centers_from_labels():
    return {"5-10": 7.5, "10-20": 15.0, "20-40": 30.0, "40-80": 60.0, "80+": 100.0}


def choose_standard_series(sub: pd.DataFrame, is_6d: bool) -> pd.Series:
    if is_6d and "std_speed_total_proxy_kms" in sub.columns:
        return safe_numeric(sub.get("std_speed_total_proxy_kms"))
    if (not is_6d) and "std_vt_total_proxy_kms" in sub.columns:
        return safe_numeric(sub.get("std_vt_total_proxy_kms"))
    return pd.Series(dtype=float)


def estimate_global_A(shell_df: pd.DataFrame) -> float:
    n_shell = pd.to_numeric(shell_df.get("n_shell"), errors="coerce")
    grad = pd.to_numeric(shell_df.get("topological_gradient_term_bg"), errors="coerce")
    obs_var = pd.to_numeric(shell_df.get("observed_variance_shell"), errors="coerce")
    std_var = pd.to_numeric(shell_df.get("standard_variance_shell"), errors="coerce")
    shell_est = pd.to_numeric(shell_df.get("xi_cinfo2_shell_estimate_diag"), errors="coerce")

    mask = (
        np.isfinite(shell_est)
        & np.isfinite(grad)
        & np.isfinite(obs_var)
        & np.isfinite(std_var)
        & (grad > EPS)
        & (n_shell >= MIN_SHELL_N_FOR_GLOBAL_A)
    )
    if mask.sum() == 0:
        return np.nan

    return float(np.nanmedian(shell_est[mask]))


def build_shell_summary(df: pd.DataFrame, config: DatasetConfig, coupling_A: float | None = None) -> tuple[pd.DataFrame, float]:
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
                "observed_variance_shell": np.nan,
                "standard_variance_shell": np.nan,
            })
            continue

        q = safe_numeric(sub.get("quality_weight"))
        w = np.where(np.isfinite(q) & (q > 0), q, 1.0)

        dloc = safe_numeric(sub.get("D_loc"))
        finite_mask = np.isfinite(dloc)
        d_halo = np.average(dloc[finite_mask], weights=w[finite_mask]) if finite_mask.any() else np.nan

        sigma_bg_shell = d_halo - D_BG if np.isfinite(d_halo) else np.nan
        sigma_local = dloc - d_halo

        sigma_local_mean = np.average(
            sigma_local[np.isfinite(sigma_local)],
            weights=w[np.isfinite(sigma_local)],
        ) if np.isfinite(sigma_local).any() else np.nan
        sigma_local_median = float(np.nanmedian(sigma_local)) if np.isfinite(sigma_local).any() else np.nan

        observed_series = safe_numeric(sub.get(config.shell_value_column))
        observed_var = float(np.nanvar(observed_series, ddof=1)) if observed_series.notna().sum() > 1 else np.nan

        std_series = choose_standard_series(sub, is_6d=config.is_6d)
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
            "observed_variance_shell": observed_var,
            "standard_variance_shell": standard_var,
        })

    shell_df = pd.DataFrame(rows)

    r = pd.to_numeric(shell_df["shell_center_kpc"], errors="coerce").to_numpy()
    sigma_bg = pd.to_numeric(shell_df["Sigma_bg_shell"], errors="coerce").to_numpy()
    obs_var = pd.to_numeric(shell_df["observed_variance_shell"], errors="coerce").to_numpy()
    std_var = pd.to_numeric(shell_df["standard_variance_shell"], errors="coerce").to_numpy()
    n_shell = pd.to_numeric(shell_df["n_shell"], errors="coerce").to_numpy()

    grad_bg = np.full(len(shell_df), np.nan, dtype=float)

    for i in range(len(shell_df)):
        if not np.isfinite(sigma_bg[i]) or n_shell[i] < MIN_SHELL_N_FOR_GRADIENT:
            continue

        if i == 0:
            j_left, j_right = i, i + 1
        elif i == len(shell_df) - 1:
            j_left, j_right = i - 1, i
        else:
            left_ok = np.isfinite(sigma_bg[i - 1]) and (n_shell[i - 1] >= MIN_SHELL_N_FOR_GRADIENT)
            right_ok = np.isfinite(sigma_bg[i + 1]) and (n_shell[i + 1] >= MIN_SHELL_N_FOR_GRADIENT)
            if left_ok and right_ok:
                j_left, j_right = i - 1, i + 1
            elif left_ok:
                j_left, j_right = i - 1, i
            elif right_ok:
                j_left, j_right = i, i + 1
            else:
                continue

        dr = r[j_right] - r[j_left]
        if np.isfinite(dr) and dr > 0 and np.isfinite(sigma_bg[j_left]) and np.isfinite(sigma_bg[j_right]):
            grad_bg[i] = abs((sigma_bg[j_right] - sigma_bg[j_left]) / dr)

    xi_diag = np.full(len(shell_df), np.nan, dtype=float)
    for i in range(len(shell_df)):
        if np.isfinite(grad_bg[i]) and grad_bg[i] > EPS and np.isfinite(obs_var[i]) and np.isfinite(std_var[i]):
            xi_diag[i] = (obs_var[i] - std_var[i]) / grad_bg[i]

    shell_df["topological_gradient_term_bg"] = grad_bg
    shell_df["xi_cinfo2_shell_estimate_diag"] = xi_diag

    if coupling_A is None:
        coupling_A = estimate_global_A(shell_df)

    topological_prediction = np.full(len(shell_df), np.nan, dtype=float)
    residual_std = np.full(len(shell_df), np.nan, dtype=float)
    residual_topo = np.full(len(shell_df), np.nan, dtype=float)

    for i in range(len(shell_df)):
        if np.isfinite(std_var[i]) and np.isfinite(grad_bg[i]) and np.isfinite(coupling_A):
            topological_prediction[i] = std_var[i] + coupling_A * grad_bg[i]
        if np.isfinite(obs_var[i]) and np.isfinite(std_var[i]):
            residual_std[i] = obs_var[i] - std_var[i]
        if np.isfinite(obs_var[i]) and np.isfinite(topological_prediction[i]):
            residual_topo[i] = obs_var[i] - topological_prediction[i]

    shell_df["A_global_adopted"] = coupling_A
    shell_df["topological_prediction_from_Aglobal"] = topological_prediction
    shell_df["residual_observed_minus_standard"] = residual_std
    shell_df["residual_observed_minus_topological"] = residual_topo

    return shell_df, coupling_A


def summarize_dataset(df: pd.DataFrame, shell_df: pd.DataFrame, config: DatasetConfig, global_A: float) -> dict:
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
        "median_Sigma_bg_shell": float(np.nanmedian(pd.to_numeric(shell_df["Sigma_bg_shell"], errors="coerce"))) if "Sigma_bg_shell" in shell_df.columns else np.nan,
        "median_topological_gradient_term_bg": float(np.nanmedian(pd.to_numeric(shell_df["topological_gradient_term_bg"], errors="coerce"))) if "topological_gradient_term_bg" in shell_df.columns else np.nan,
        "median_xi_cinfo2_shell_estimate_diag": float(np.nanmedian(pd.to_numeric(shell_df["xi_cinfo2_shell_estimate_diag"], errors="coerce"))) if "xi_cinfo2_shell_estimate_diag" in shell_df.columns else np.nan,
        "A_global_adopted": float(global_A) if np.isfinite(global_A) else np.nan,
    }


def run_one_dataset(
    project_root: Path,
    config: DatasetConfig,
    output_dir: Path,
    coupling_A: float | None = None,
) -> tuple[dict, float]:
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

    shell_df, used_A = build_shell_summary(df, config=config, coupling_A=coupling_A)

    shell_to_d_halo = dict(zip(shell_df["shell"], shell_df["D_halo_shell"]))
    shell_to_sigma_bg = dict(zip(shell_df["shell"], shell_df["Sigma_bg_shell"]))
    shell_to_grad = dict(zip(shell_df["shell"], shell_df["topological_gradient_term_bg"]))

    df["D_bg_reference"] = D_BG
    df["D_halo_shell"] = df["topological_shell"].astype(str).map(shell_to_d_halo)
    df["Sigma_bg_shell"] = df["topological_shell"].astype(str).map(shell_to_sigma_bg)
    df["sigma_local"] = pd.to_numeric(df["D_loc"], errors="coerce") - pd.to_numeric(df["D_halo_shell"], errors="coerce")
    df["topological_gradient_term_bg"] = df["topological_shell"].astype(str).map(shell_to_grad)
    df["A_global_adopted"] = used_A
    df["topological_preserved_input"] = True

    out_detail = output_dir / f"gaia_rrlyrae_{config.name}_topological.csv"
    out_shell = output_dir / f"gaia_rrlyrae_{config.name}_topological_shells.csv"

    df.to_csv(out_detail, index=False)
    shell_df.to_csv(out_shell, index=False)

    return summarize_dataset(df, shell_df, config=config, global_A=used_A), used_A


def main() -> None:
    project_root = find_project_root(Path(__file__))
    timestamp = timestamp_folder_name()
    output_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "topological" / timestamp
    ensure_dir(output_dir)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Output dir: {output_dir}")

    config_6d = DatasetConfig(
        "6d",
        True,
        "gaia_rrlyrae_6d_input.csv",
        "gaia_rrlyrae_6d_standard.csv",
        "speed_total_proxy_kms_stable",
    )
    config_5d = DatasetConfig(
        "5d",
        False,
        "gaia_rrlyrae_5d_input.csv",
        "gaia_rrlyrae_5d_standard.csv",
        "vt_total_proxy_kms_stable",
    )

    summaries = []

    summary_6d, global_A_6d = run_one_dataset(project_root, config_6d, output_dir, coupling_A=None)
    summaries.append(summary_6d)

    summary_5d, _ = run_one_dataset(project_root, config_5d, output_dir, coupling_A=global_A_6d)
    summaries.append(summary_5d)

    pd.DataFrame(summaries).to_csv(output_dir / "topological_summary.csv", index=False)

    readme = (
        "Our galaxy Halo Stellar Kinematics - topological stage 003\n\n"
        "Changes in this revision\n"
        "- Output is written into a timestamped subfolder.\n"
        "- Background reference is written as D_bg = 3.0.\n"
        "- Main topological driver uses shell-averaged background contrast:\n"
        "    Sigma_bg_shell = D_halo_shell - D_bg\n"
        "- Main gradient term uses d(Sigma_bg_shell)/dr only.\n"
        "- pointwise sigma_local = D_loc - D_halo_shell is retained only as a diagnostic.\n"
        "- shell-by-shell xi*c_info^2 inverse estimates are retained only as diagnostics.\n"
        "- the main prediction adopts a single global A = xi*c_info^2 estimated from stable 6D shells,\n"
        "  then projects that A into both 6D and 5D shell summaries.\n"
        "- Input rows are not re-filtered or re-selected.\n\n"
        "Structural reference\n"
        f"- D_bg = {D_BG}\n\n"
        "Stabilization\n"
        f"- MIN_PARALLAX_MAS = {MIN_PARALLAX_MAS}\n"
        f"- MAX_DISTANCE_PROXY_KPC = {MAX_DISTANCE_PROXY_KPC}\n"
        f"- MAX_VT_PROXY_KMS = {MAX_VT_PROXY_KMS}\n"
        f"- MIN_SHELL_N_FOR_GRADIENT = {MIN_SHELL_N_FOR_GRADIENT}\n"
        f"- MIN_SHELL_N_FOR_GLOBAL_A = {MIN_SHELL_N_FOR_GLOBAL_A}\n"
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
