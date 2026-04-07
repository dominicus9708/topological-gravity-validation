from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd

KPC_PER_MAS_PARALLAX = 1.0
KM_S_PER_MASYR_KPC = 4.74047
SHELL_BINS_KPC = [5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
SHELL_LABELS = ["5-10", "10-20", "20-40", "40-80", "80+"]
K1 = 8
K2 = 32
EPS = 1e-12


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


def add_proxy_columns(df: pd.DataFrame, is_6d: bool) -> pd.DataFrame:
    out = df.copy()

    if "distance_proxy_kpc" not in out.columns:
        par = safe_numeric(out.get("parallax_mas"))
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

    dist = safe_numeric(out.get("distance_proxy_kpc"))
    pmra = safe_numeric(out.get("pmra_masyr"))
    pmdec = safe_numeric(out.get("pmdec_masyr"))

    out["vt_ra_proxy_kms"] = KM_S_PER_MASYR_KPC * pmra * dist
    out["vt_dec_proxy_kms"] = KM_S_PER_MASYR_KPC * pmdec * dist
    out["vt_total_proxy_kms"] = np.sqrt(
        np.square(safe_numeric(out.get("vt_ra_proxy_kms")))
        + np.square(safe_numeric(out.get("vt_dec_proxy_kms")))
    )

    if is_6d:
        rv = safe_numeric(out.get("radial_velocity_kms"))
        out["speed_total_proxy_kms"] = np.sqrt(
            np.square(safe_numeric(out.get("vt_total_proxy_kms")))
            + np.square(rv)
        )
    else:
        out["speed_total_proxy_kms"] = np.nan

    return out


def add_shells(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dist = safe_numeric(out.get("distance_proxy_kpc"))
    out["topological_shell"] = pd.cut(
        dist, bins=SHELL_BINS_KPC, labels=SHELL_LABELS, right=False
    )
    return out


def cartesian_from_galactic(df: pd.DataFrame) -> np.ndarray:
    l = np.deg2rad(safe_numeric(df.get("gal_l_deg")).to_numpy())
    b = np.deg2rad(safe_numeric(df.get("gal_b_deg")).to_numpy())
    r = safe_numeric(df.get("distance_proxy_kpc")).to_numpy()

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
    """
    D ~ (ln k2 - ln k1) / (ln r_k2 - ln r_k1)
    """
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
            safe_numeric(df.get("vt_ra_proxy_kms")).to_numpy(),
            safe_numeric(df.get("vt_dec_proxy_kms")).to_numpy(),
            safe_numeric(df.get("radial_velocity_kms")).to_numpy(),
        ])
    else:
        arr = np.column_stack([
            safe_numeric(df.get("vt_ra_proxy_kms")).to_numpy(),
            safe_numeric(df.get("vt_dec_proxy_kms")).to_numpy(),
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
                "D_halo_shell": np.nan,
                "Delta_halo_shell": np.nan,
                "sigma_bar_shell": np.nan,
                "observed_variance_shell": np.nan,
                "standard_variance_shell": np.nan,
                "topological_gradient_term": np.nan,
                "topological_prediction_norm": np.nan,
            })
            continue

        q = safe_numeric(sub.get("quality_weight"))
        w = np.where(np.isfinite(q) & (q > 0), q, 1.0)

        dloc = safe_numeric(sub.get("D_loc"))
        finite_mask = np.isfinite(dloc)
        d_halo = np.average(dloc[finite_mask], weights=w[finite_mask]) if finite_mask.any() else np.nan
        delta_halo = d_halo - 3.0 if np.isfinite(d_halo) else np.nan

        sigma_local = dloc - d_halo
        sigma_mask = np.isfinite(sigma_local)
        sigma_bar = np.average(sigma_local[sigma_mask], weights=w[sigma_mask]) if sigma_mask.any() else np.nan

        observed_series = safe_numeric(sub.get(config.shell_value_column))
        observed_var = float(np.nanvar(observed_series, ddof=1)) if observed_series.notna().sum() > 1 else np.nan

        if config.is_6d and "std_speed_total_proxy_kms" in sub.columns:
            std_series = safe_numeric(sub.get("std_speed_total_proxy_kms"))
        elif (not config.is_6d) and "std_vt_total_proxy_kms" in sub.columns:
            std_series = safe_numeric(sub.get("std_vt_total_proxy_kms"))
        else:
            std_series = pd.Series(dtype=float)

        standard_var = float(np.nanvar(std_series, ddof=1)) if std_series.notna().sum() > 1 else np.nan

        rows.append({
            "dataset": config.name,
            "shell": shell,
            "shell_center_kpc": centers_map[shell],
            "n_shell": int(n),
            "D_halo_shell": d_halo,
            "Delta_halo_shell": delta_halo,
            "sigma_bar_shell": sigma_bar,
            "observed_variance_shell": observed_var,
            "standard_variance_shell": standard_var,
        })

    shell_df = pd.DataFrame(rows)

    grad = np.full(len(shell_df), np.nan, dtype=float)
    pred_norm = np.full(len(shell_df), np.nan, dtype=float)

    r = pd.to_numeric(shell_df["shell_center_kpc"], errors="coerce").to_numpy()
    delta_h = pd.to_numeric(shell_df["Delta_halo_shell"], errors="coerce").to_numpy()
    sigma_bar = pd.to_numeric(shell_df["sigma_bar_shell"], errors="coerce").to_numpy()
    std_var = pd.to_numeric(shell_df["standard_variance_shell"], errors="coerce").to_numpy()

    for i in range(len(shell_df) - 1):
        if np.isfinite(r[i]) and np.isfinite(r[i + 1]) and (r[i + 1] > r[i]):
            d1 = delta_h[i + 1] - delta_h[i] if np.isfinite(delta_h[i + 1]) and np.isfinite(delta_h[i]) else np.nan
            d2 = sigma_bar[i + 1] - sigma_bar[i] if np.isfinite(sigma_bar[i + 1]) and np.isfinite(sigma_bar[i]) else np.nan
            if np.isfinite(d1) or np.isfinite(d2):
                val = 0.0
                if np.isfinite(d1):
                    val += d1
                if np.isfinite(d2):
                    val += d2
                grad[i] = abs(val / (r[i + 1] - r[i]))

    finite_grad = np.where(np.isfinite(grad))[0]
    if len(finite_grad) > 0 and not np.isfinite(grad[-1]):
        grad[-1] = grad[finite_grad[-1]]

    finite_std = std_var[np.isfinite(std_var)]
    std_scale = np.nanmedian(finite_std) if len(finite_std) > 0 else 1.0
    if not np.isfinite(std_scale) or std_scale <= EPS:
        std_scale = 1.0

    for i in range(len(shell_df)):
        if np.isfinite(grad[i]) and np.isfinite(std_var[i]):
            pred_norm[i] = std_var[i] + std_scale * grad[i]

    shell_df["topological_gradient_term"] = grad
    shell_df["topological_prediction_norm"] = pred_norm
    return shell_df


def summarize_dataset(df: pd.DataFrame, shell_df: pd.DataFrame, config: DatasetConfig) -> dict:
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
    df["D_halo_shell"] = df["topological_shell"].astype(str).map(shell_to_d_halo)
    df["sigma_local"] = pd.to_numeric(df["D_loc"], errors="coerce") - pd.to_numeric(df["D_halo_shell"], errors="coerce")
    df["topological_preserved_input"] = True

    out_detail = output_dir / f"gaia_rrlyrae_{config.name}_topological.csv"
    out_shell = output_dir / f"gaia_rrlyrae_{config.name}_topological_shells.csv"

    df.to_csv(out_detail, index=False)
    shell_df.to_csv(out_shell, index=False)

    return summarize_dataset(df, shell_df, config=config)


def main() -> None:
    project_root = find_project_root(Path(__file__))
    output_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "topological"
    ensure_dir(output_dir)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Output dir: {output_dir}")

    configs = [
        DatasetConfig("5d", False, "gaia_rrlyrae_5d_input.csv", "gaia_rrlyrae_5d_standard.csv", "vt_total_proxy_kms"),
        DatasetConfig("6d", True, "gaia_rrlyrae_6d_input.csv", "gaia_rrlyrae_6d_standard.csv", "speed_total_proxy_kms"),
    ]

    summaries = []
    for cfg in configs:
        summaries.append(run_one_dataset(project_root, cfg, output_dir))

    pd.DataFrame(summaries).to_csv(output_dir / "topological_summary.csv", index=False)

    readme = (
        "Our galaxy Halo Stellar Kinematics - topological stage v1\n\n"
        "What this stage does\n"
        "- Reads the fixed final input used by skeleton/standard.\n"
        "- Computes local position-based structural dimension D_loc_pos.\n"
        "- Computes local kinematic structural dimension D_loc_kin.\n"
        "- Combines them into D_loc using quality-derived lambda_pos.\n"
        "- Computes shell-based halo background dimension D_halo_shell.\n"
        "- Computes sigma_local = D_loc - D_halo_shell.\n"
        "- Computes shell-based topological_gradient_term and topological_prediction_norm.\n\n"
        "What this stage does not do\n"
        "- It does not re-filter or re-select the final input.\n"
        "- It does not insert an arbitrary coupling constant.\n"
        "- topological_prediction_norm is a comparison-oriented normalized form.\n"
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