from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

SHELL_BINS_KPC = [5.0, 10.0, 20.0, 40.0, 80.0, np.inf]
SHELL_LABELS = ["5-10", "10-20", "20-40", "40-80", "80+"]
SHELL_CENTERS = {"5-10": 7.5, "10-20": 15.0, "20-40": 30.0, "40-80": 60.0, "80+": 100.0}
K1 = 8
K2 = 32
EPS = 1e-12

MIN_PARALLAX_MAS = 0.0
MIN_SHELL_N_FOR_GRADIENT = 20
MIN_SHELL_N_FOR_COUPLING_DIAG = 20

D_BG_REFERENCE = 3.0
WINSOR_LOWER_Q = 0.10
WINSOR_UPPER_Q = 0.90


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
    pmra = safe_numeric(out.get("pmra_masyr"))
    pmdec = safe_numeric(out.get("pmdec_masyr"))

    out["parallax_mas_for_proxy"] = np.where(par > MIN_PARALLAX_MAS, par, np.nan)
    out["distance_proxy_kpc_soft"] = np.where(
        np.isfinite(out["parallax_mas_for_proxy"]),
        1.0 / out["parallax_mas_for_proxy"],
        np.nan,
    )

    if "distance_proxy_kpc" not in out.columns:
        out["distance_proxy_kpc"] = out["distance_proxy_kpc_soft"]

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

    dist = safe_numeric(out.get("distance_proxy_kpc_soft"))
    out["vt_ra_proxy_kms_soft"] = 4.74047 * pmra * dist
    out["vt_dec_proxy_kms_soft"] = 4.74047 * pmdec * dist
    out["vt_total_proxy_kms_soft"] = np.sqrt(
        np.square(safe_numeric(out.get("vt_ra_proxy_kms_soft")))
        + np.square(safe_numeric(out.get("vt_dec_proxy_kms_soft")))
    )

    if is_6d:
        rv = safe_numeric(out.get("radial_velocity_kms"))
        out["speed_total_proxy_kms_soft"] = np.sqrt(
            np.square(safe_numeric(out.get("vt_total_proxy_kms_soft")))
            + np.square(rv)
        )
    else:
        out["speed_total_proxy_kms_soft"] = np.nan

    return out


def add_shells(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dist = safe_numeric(out.get("distance_proxy_kpc_soft"))
    out["topological_shell"] = pd.cut(dist, bins=SHELL_BINS_KPC, labels=SHELL_LABELS, right=False)
    return out


def cartesian_from_galactic(df: pd.DataFrame) -> np.ndarray:
    l = np.deg2rad(safe_numeric(df.get("gal_l_deg")).to_numpy())
    b = np.deg2rad(safe_numeric(df.get("gal_b_deg")).to_numpy())
    r = safe_numeric(df.get("distance_proxy_kpc_soft")).to_numpy()

    x = r * np.cos(b) * np.cos(l)
    y = r * np.cos(b) * np.sin(l)
    z = r * np.sin(b)
    return np.column_stack([x, y, z])


def robust_scale_matrix(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(float).copy()
    out = np.arcsinh(out)

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
    return robust_scale_matrix(cartesian_from_galactic(df))


def build_kinematic_features(df: pd.DataFrame, is_6d: bool) -> np.ndarray:
    if is_6d:
        arr = np.column_stack([
            safe_numeric(df.get("vt_ra_proxy_kms_soft")).to_numpy(),
            safe_numeric(df.get("vt_dec_proxy_kms_soft")).to_numpy(),
            safe_numeric(df.get("radial_velocity_kms")).to_numpy(),
        ])
    else:
        arr = np.column_stack([
            safe_numeric(df.get("vt_ra_proxy_kms_soft")).to_numpy(),
            safe_numeric(df.get("vt_dec_proxy_kms_soft")).to_numpy(),
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
    d_pos = pairwise_knn_dimension(build_position_features(out))
    d_kin = pairwise_knn_dimension(build_kinematic_features(out, is_6d))

    out["D_loc_pos"] = d_pos
    out["D_loc_kin"] = d_kin

    lam = safe_numeric(out.get("lambda_pos")).to_numpy()
    out["D_loc"] = lam * d_pos + (1.0 - lam) * d_kin
    return out


def weighted_winsorized_mean(values: np.ndarray, weights: np.ndarray, lower_q: float, upper_q: float) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    x = values[mask].astype(float)
    w = weights[mask].astype(float)
    lo = np.nanquantile(x, lower_q)
    hi = np.nanquantile(x, upper_q)
    xw = np.clip(x, lo, hi)
    return float(np.average(xw, weights=w))


def attach_standard_shell_data(dataset_name: str, standard_dir: Path, is_6d: bool) -> tuple[pd.DataFrame | None, str]:
    shell_path = standard_dir / f"gaia_rrlyrae_{dataset_name}_standard_shells.csv"
    if not shell_path.exists():
        return None, ""
    sdf = read_csv(shell_path)
    return sdf, str(shell_path)


def choose_observed_series(sub: pd.DataFrame, is_6d: bool) -> tuple[pd.Series, str]:
    candidates = ["speed_total_proxy_kms_soft", "speed_total_proxy_kms"] if is_6d else ["vt_total_proxy_kms_soft", "vt_total_proxy_kms"]
    for col in candidates:
        if col in sub.columns:
            s = safe_numeric(sub.get(col))
            if s.notna().sum() > 1:
                return s, col
    return pd.Series(dtype=float), ""


def get_standard_maps(standard_shell_df: pd.DataFrame | None, is_6d: bool) -> tuple[dict, str, pd.DataFrame | None]:
    if standard_shell_df is None:
        return {}, "", None

    model_col = "standard_speed_variance_shell_model" if is_6d else "standard_vt_variance_shell_model"
    if model_col not in standard_shell_df.columns:
        return {}, "", standard_shell_df

    variance_map = dict(zip(standard_shell_df["shell"].astype(str), safe_numeric(standard_shell_df[model_col])))
    return variance_map, model_col, standard_shell_df


def classify_shell_caution(n_shell: int, stream_richness_flag: object) -> str:
    if n_shell < MIN_SHELL_N_FOR_COUPLING_DIAG:
        return "sparse_shell"
    if isinstance(stream_richness_flag, str) and stream_richness_flag == "high":
        return "stream_rich_shell"
    if isinstance(stream_richness_flag, str) and stream_richness_flag == "medium":
        return "stream_moderate_shell"
    return "nominal_shell"


def build_shell_summary(df: pd.DataFrame, dataset_name: str, is_6d: bool, standard_shell_df: pd.DataFrame | None, standard_source_path: str) -> pd.DataFrame:
    rows = []
    standard_map, standard_model_source, standard_shell_df = get_standard_maps(standard_shell_df, is_6d=is_6d)

    standard_diag_cols = [
        "n_stream_points_overlap",
        "n_unique_tracks_overlap",
        "n_unique_streams_overlap",
        "dominant_halo_usefulness_label",
        "dominant_InfoFlags_str",
        "stream_richness_flag",
    ]

    standard_diag_df = None
    if standard_shell_df is not None:
        diag_cols = ["shell"] + [c for c in standard_diag_cols if c in standard_shell_df.columns]
        standard_diag_df = standard_shell_df[diag_cols].copy()

    for shell in SHELL_LABELS:
        sub = df[df["topological_shell"].astype(str) == shell].copy()
        n = len(sub)

        if n == 0:
            rows.append({
                "dataset": dataset_name,
                "shell": shell,
                "shell_center_kpc": SHELL_CENTERS[shell],
                "n_shell": 0,
                "D_bg_reference": D_BG_REFERENCE,
                "D_halo_shell": np.nan,
                "Sigma_bg_shell": np.nan,
                "sigma_local_mean_shell": np.nan,
                "Sigma_shell_total": np.nan,
                "observed_series_source": "",
                "standard_series_source": standard_model_source,
                "standard_source_path": standard_source_path,
                "observed_variance_shell": np.nan,
                "standard_variance_shell": np.nan,
                "residual_observed_minus_standard": np.nan,
            })
            continue

        q = safe_numeric(sub.get("quality_weight")).to_numpy()
        q = np.where(np.isfinite(q) & (q > 0), q, 1.0)

        dloc = safe_numeric(sub.get("D_loc")).to_numpy()
        d_halo = weighted_winsorized_mean(dloc, q, WINSOR_LOWER_Q, WINSOR_UPPER_Q)
        sigma_bg = d_halo - D_BG_REFERENCE if np.isfinite(d_halo) else np.nan

        sigma_local = dloc - d_halo
        sigma_local_mean = weighted_winsorized_mean(sigma_local, q, WINSOR_LOWER_Q, WINSOR_UPPER_Q)

        sigma_shell_total = sigma_local_mean + sigma_bg if np.isfinite(sigma_local_mean) and np.isfinite(sigma_bg) else np.nan

        observed_series, observed_source = choose_observed_series(sub, is_6d=is_6d)
        observed_var = float(np.nanvar(observed_series, ddof=1)) if observed_series.notna().sum() > 1 else np.nan

        standard_var = standard_map.get(shell, np.nan)
        standard_var = float(standard_var) if pd.notna(standard_var) else np.nan
        residual = observed_var - standard_var if np.isfinite(observed_var) and np.isfinite(standard_var) else np.nan

        rows.append({
            "dataset": dataset_name,
            "shell": shell,
            "shell_center_kpc": SHELL_CENTERS[shell],
            "n_shell": int(n),
            "D_bg_reference": D_BG_REFERENCE,
            "D_halo_shell": d_halo,
            "Sigma_bg_shell": sigma_bg,
            "sigma_local_mean_shell": sigma_local_mean,
            "Sigma_shell_total": sigma_shell_total,
            "observed_series_source": observed_source,
            "standard_series_source": standard_model_source,
            "standard_source_path": standard_source_path,
            "observed_variance_shell": observed_var,
            "standard_variance_shell": standard_var,
            "residual_observed_minus_standard": residual,
        })

    shell_df = pd.DataFrame(rows)

    if standard_diag_df is not None:
        shell_df = shell_df.merge(standard_diag_df, on="shell", how="left")
    else:
        shell_df["n_stream_points_overlap"] = 0
        shell_df["n_unique_tracks_overlap"] = 0
        shell_df["n_unique_streams_overlap"] = 0
        shell_df["dominant_halo_usefulness_label"] = pd.NA
        shell_df["dominant_InfoFlags_str"] = pd.NA
        shell_df["stream_richness_flag"] = "none"

    grad_bg = np.full(len(shell_df), np.nan, dtype=float)
    grad_total = np.full(len(shell_df), np.nan, dtype=float)
    coupling_total = np.full(len(shell_df), np.nan, dtype=float)
    valid = np.full(len(shell_df), False, dtype=bool)

    r = pd.to_numeric(shell_df["shell_center_kpc"], errors="coerce").to_numpy()
    sigma_bg = pd.to_numeric(shell_df["Sigma_bg_shell"], errors="coerce").to_numpy()
    sigma_total = pd.to_numeric(shell_df["Sigma_shell_total"], errors="coerce").to_numpy()
    residual = pd.to_numeric(shell_df["residual_observed_minus_standard"], errors="coerce").to_numpy()
    n_shell = pd.to_numeric(shell_df["n_shell"], errors="coerce").to_numpy()

    for i in range(len(shell_df) - 1):
        enough_n = (n_shell[i] >= MIN_SHELL_N_FOR_GRADIENT) and (n_shell[i + 1] >= MIN_SHELL_N_FOR_GRADIENT)
        if enough_n and np.isfinite(r[i]) and np.isfinite(r[i + 1]) and (r[i + 1] > r[i]):
            if np.isfinite(sigma_bg[i + 1]) and np.isfinite(sigma_bg[i]):
                grad_bg[i] = abs((sigma_bg[i + 1] - sigma_bg[i]) / (r[i + 1] - r[i]))
            if np.isfinite(sigma_total[i + 1]) and np.isfinite(sigma_total[i]):
                grad_total[i] = abs((sigma_total[i + 1] - sigma_total[i]) / (r[i + 1] - r[i]))

    finite_grad_bg = np.where(np.isfinite(grad_bg))[0]
    if len(finite_grad_bg) > 0 and not np.isfinite(grad_bg[-1]):
        grad_bg[-1] = grad_bg[finite_grad_bg[-1]]

    finite_grad_total = np.where(np.isfinite(grad_total))[0]
    if len(finite_grad_total) > 0 and not np.isfinite(grad_total[-1]):
        grad_total[-1] = grad_total[finite_grad_total[-1]]

    for i in range(len(shell_df)):
        if (
            n_shell[i] >= MIN_SHELL_N_FOR_COUPLING_DIAG
            and np.isfinite(grad_total[i]) and grad_total[i] > EPS
            and np.isfinite(residual[i])
        ):
            coupling_total[i] = residual[i] / grad_total[i]
            valid[i] = True

    shell_df["topological_gradient_term_bg"] = grad_bg
    shell_df["topological_gradient_term_shell"] = grad_total
    shell_df["coupling_shell_diagnostic"] = coupling_total
    shell_df["coupling_shell_diagnostic_valid"] = valid
    shell_df["shell_caution_label"] = [
        classify_shell_caution(int(n) if pd.notna(n) else 0, s)
        for n, s in zip(shell_df["n_shell"], shell_df["stream_richness_flag"])
    ]

    return shell_df


def summarize_dataset(df: pd.DataFrame, shell_df: pd.DataFrame, dataset_name: str) -> dict:
    dloc = safe_numeric(df.get("D_loc"))
    lam = safe_numeric(df.get("lambda_pos"))
    sigma_bg = safe_numeric(shell_df.get("Sigma_bg_shell"))
    sigma_total = safe_numeric(shell_df.get("Sigma_shell_total"))
    grad_shell = safe_numeric(shell_df.get("topological_gradient_term_shell"))
    coupling = safe_numeric(shell_df.get("coupling_shell_diagnostic"))
    valid = shell_df.get("coupling_shell_diagnostic_valid")
    valid_n = int(pd.Series(valid).fillna(False).astype(bool).sum()) if valid is not None else 0

    return {
        "dataset": dataset_name,
        "rows_read": int(len(df)),
        "rows_written": int(len(df)),
        "shells_nonempty": int((pd.to_numeric(shell_df["n_shell"], errors="coerce") > 0).sum()),
        "median_D_loc": float(np.nanmedian(dloc)) if dloc.notna().any() else np.nan,
        "median_lambda_pos": float(np.nanmedian(lam)) if lam.notna().any() else np.nan,
        "median_Sigma_bg_shell": float(np.nanmedian(sigma_bg)) if sigma_bg.notna().any() else np.nan,
        "median_Sigma_shell_total": float(np.nanmedian(sigma_total)) if sigma_total.notna().any() else np.nan,
        "median_topological_gradient_term_shell": float(np.nanmedian(grad_shell)) if grad_shell.notna().any() else np.nan,
        "median_coupling_shell_diagnostic": float(np.nanmedian(coupling)) if coupling.notna().any() else np.nan,
        "coupling_shell_diagnostic_valid_n": valid_n,
    }


def run_one_dataset(project_root: Path, dataset_name: str, is_6d: bool, output_dir: Path) -> dict:
    input_dir = project_root / "data" / "derived" / "Our galaxy Halo Stellar Kinematics" / "input"
    standard_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "standard"

    input_path = input_dir / f"gaia_rrlyrae_{dataset_name}_input.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = read_csv(input_path)
    df = add_proxy_columns(df, is_6d=is_6d)
    df = add_shells(df)
    df = add_quality_weights(df, is_6d=is_6d)
    df = add_local_dimensions(df, is_6d=is_6d)

    standard_shell_df, standard_source_path = attach_standard_shell_data(dataset_name, standard_dir, is_6d=is_6d)

    shell_df = build_shell_summary(
        df=df,
        dataset_name=dataset_name,
        is_6d=is_6d,
        standard_shell_df=standard_shell_df,
        standard_source_path=standard_source_path,
    )

    shell_to_d_halo = dict(zip(shell_df["shell"], shell_df["D_halo_shell"]))
    df["D_bg_reference"] = D_BG_REFERENCE
    df["D_halo_shell"] = df["topological_shell"].astype(str).map(shell_to_d_halo)
    df["sigma_local"] = safe_numeric(df.get("D_loc")) - safe_numeric(df.get("D_halo_shell"))
    df["topological_preserved_input"] = True

    out_detail = output_dir / f"gaia_rrlyrae_{dataset_name}_topological.csv"
    out_shell = output_dir / f"gaia_rrlyrae_{dataset_name}_topological_shells.csv"
    df.to_csv(out_detail, index=False)
    shell_df.to_csv(out_shell, index=False)

    return summarize_dataset(df, shell_df, dataset_name)


def main() -> None:
    project_root = find_project_root(Path(__file__))
    output_dir = project_root / "results" / "Our galaxy Halo Stellar Kinematics" / "output" / "topological" / timestamp_folder_name()
    ensure_dir(output_dir)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Output dir: {output_dir}")

    summaries = [
        run_one_dataset(project_root, "5d", False, output_dir),
        run_one_dataset(project_root, "6d", True, output_dir),
    ]
    pd.DataFrame(summaries).to_csv(output_dir / "topological_summary.csv", index=False)

    readme = (
        "Our galaxy Halo Stellar Kinematics - topological stage 011\n\n"
        "Changes in this revision\n"
        "- Robust structural aggregation is retained.\n"
        "- Standard comparison is read from standard shell summary model files.\n"
        "- Shell-level standard stream-overlap diagnostics are also ingested from standard stage 004.\n"
        "- A shell-total structural contrast is defined as Sigma_shell_total = sigma_local_mean_shell + Sigma_bg_shell.\n"
        "- The main topological gradient diagnostic is now computed from Sigma_shell_total, not only Sigma_bg_shell.\n"
        "- shell_caution_label is added to distinguish sparse / stream-rich / nominal shells.\n"
        "- Coupling remains diagnostic only and is not applied to prediction.\n"
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
