from __future__ import annotations

import argparse
import math
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
import numpy as np
import pandas as pd

TARGET_FOLDER_CANDIDATES = ["η Carinae", "Eta Carinae"]


def find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "data").exists() and (candidate / "results").exists():
            return candidate
    return Path.cwd()


def find_target_input_dir(repo_root: Path, override: Optional[str] = None) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else (repo_root / p)
    base = repo_root / "data" / "derived" / "Echo of Light" / "input"
    for name in TARGET_FOLDER_CANDIDATES:
        candidate = base / name
        if candidate.exists():
            return candidate
    return base / TARGET_FOLDER_CANDIDATES[0]


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_numeric_like(value) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    text = text.replace(",", "")
    text = text.replace("−", "-").replace("–", "-")
    text = text.replace("~", "").replace("≈", "")
    text = text.replace("approaching", "").replace(">", "").replace("<", "").strip()

    if "/" in text:
        parts = []
        for part in text.split("/"):
            num = parse_numeric_like(part)
            if not np.isnan(num):
                parts.append(abs(num))
        if parts:
            return float(np.mean(parts))
        return np.nan

    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return np.nan
    return np.nan


def normalize_series_numeric(series: pd.Series) -> pd.Series:
    return series.apply(parse_numeric_like)


def normalize_0_1(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    vmin = valid.min()
    vmax = valid.max()
    if math.isclose(vmin, vmax):
        out = pd.Series(0.0, index=values.index, dtype=float)
        out[values.isna()] = np.nan
        return out
    return (numeric - vmin) / (vmax - vmin)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def copy_processed_inputs(input_files: Iterable[Path], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for path in input_files:
        if path and path.exists():
            shutil.copy2(path, processed_dir / path.name)


def prepare_observation_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "effective_temperature_k",
        "spectral_velocity_kmps",
        "bulk_velocity_kmps",
        "fast_component_velocity_kmps",
        "mass_context_msun_lower_bound",
    ]:
        if col in out.columns:
            out[col] = normalize_series_numeric(out[col])

    if "observation_label" not in out.columns:
        out["observation_label"] = [f"row_{i+1}" for i in range(len(out))]
    return out


def plot_standard_velocity(df: pd.DataFrame, output_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    labels = df["observation_label"].astype(str).tolist()

    offsets = [-0.2, 0.0, 0.2]
    width = 0.18
    specs = [
        ("spectral_velocity_kmps", "spectral_velocity_kmps"),
        ("bulk_velocity_kmps", "bulk_velocity_kmps"),
        ("fast_component_velocity_kmps", "fast_component_velocity_kmps"),
    ]

    plotted = False
    for (col, label), offset in zip(specs, offsets):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                plt.bar(x + offset, vals.fillna(0.0), width=width, label=label, alpha=0.8)
                plotted = True

    if not plotted:
        plt.text(0.5, 0.5, "No velocity columns available", ha="center", va="center", transform=plt.gca().transAxes)

    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Velocity (km/s)")
    plt.title("Eta Carinae light echo velocity overview")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def plot_standard_temperature_mass(df: pd.DataFrame, output_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    x = np.arange(len(df))
    labels = df["observation_label"].astype(str).tolist()

    temp = pd.to_numeric(df.get("effective_temperature_k", pd.Series([np.nan] * len(df))), errors="coerce")
    mass = pd.to_numeric(df.get("mass_context_msun_lower_bound", pd.Series([np.nan] * len(df))), errors="coerce")

    ax1 = plt.gca()
    if temp.notna().any():
        ax1.plot(x, temp, marker="o", label="effective_temperature_k")
        ax1.set_ylabel("Temperature (K)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_title("Eta Carinae observation context")

    if mass.notna().any():
        ax2 = ax1.twinx()
        ax2.bar(x, mass.fillna(0.0), alpha=0.3, label="mass_context_msun_lower_bound")
        ax2.set_ylabel("Mass context lower bound (Msun)")

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()


def _first_row(df: pd.DataFrame, component_label: str):
    subset = df[df["component_label"].astype(str) == component_label]
    if subset.empty:
        return None
    return subset.iloc[0]


def plot_standard_circular_context_tpyx_style(mass_df: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")

    ring_radii = [2, 4, 6, 8, 10]
    for r in ring_radii:
        ax.add_patch(Circle((0, 0), r, fill=False, linewidth=0.8, linestyle=":"))
        ax.text(r, 0.18, f'{r}"', fontsize=9, ha="center", va="bottom")

    star = Circle((0, 0), radius=0.22, fill=False, linewidth=2.0)
    ax.add_patch(star)
    ax.plot(0, 0, marker="+", markersize=10, markeredgewidth=1.8)
    ax.text(0.0, -0.55, "Eta Car", ha="center", va="top", fontsize=10)

    legend_handles = [
        Line2D([0], [0], marker='o', markersize=8, markerfacecolor='none', markeredgecolor='black', linestyle='None', label='Central star'),
        Line2D([0], [0], marker='+', markersize=10, color='black', linestyle='None', label='COM / center'),
    ]

    hom = _first_row(mass_df, "homunculus_main_lobes")
    if hom is not None:
        major = parse_numeric_like(hom.get("angular_major_axis_arcsec"))
        minor = parse_numeric_like(hom.get("angular_minor_axis_arcsec"))
        if np.isnan(major):
            major = 20.0
        if np.isnan(minor):
            minor = 10.0
        hom_patch = Ellipse((0, 0), width=minor, height=major, angle=0, fill=False, linewidth=2.0)
        ax.add_patch(hom_patch)
        ax.text(0, major / 2 + 0.7, "Homunculus", ha="center", va="bottom", fontsize=10)
        legend_handles.append(Line2D([0], [0], color='black', linewidth=2.0, label='Homunculus lobes'))

    tor = _first_row(mass_df, "dust_torus")
    if tor is not None:
        torus_r = 2.3
        ax.add_patch(Circle((0, 0), torus_r, fill=False, linewidth=1.5, linestyle="--"))
        ax.text(torus_r + 0.4, -0.2, "dust torus", ha="left", va="center", fontsize=9)
        legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Dust torus'))

    out = _first_row(mass_df, "outer_ejecta")
    if out is None:
        out = _first_row(mass_df, "outer_ejecta_context")
    if out is not None:
        outer_r = 6.2
        ax.add_patch(Circle((0, 0), outer_r, fill=False, linewidth=1.2, linestyle="-."))
        clumps = [(4.8, 1.4), (-4.1, 2.6), (2.4, -4.9), (-5.0, -1.3)]
        for x, y in clumps:
            ax.add_patch(Circle((x, y), 0.14, fill=True, alpha=0.7))
        ax.text(outer_r + 0.35, 0.0, "outer ejecta", ha="left", va="center", fontsize=9)
        legend_handles.append(Line2D([0], [0], color='black', linestyle='-.', linewidth=1.2, label='Outer ejecta shell'))
        legend_handles.append(Line2D([0], [0], marker='o', markersize=5, color='black', linestyle='None', label='Representative knots'))

    leg1 = ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    ax.add_artist(leg1)

    info_lines = [
        'Angular guide rings',
        '2", 4", 6", 8", 10"',
        '',
        'Drawing mode:',
        'standard context schematic',
        'not epoch-by-epoch echo map',
    ]
    ax.text(0.98, 0.04, "\n".join(info_lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"))

    ax.set_xlim(-11.5, 11.5)
    ax.set_ylim(-11.5, 11.5)
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_title("Eta Carinae standard circular context")
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def run_standard_pipeline(observation_csv: Path, mass_csv: Path | None, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed_input"
    files = [observation_csv]
    if mass_csv:
        files.append(mass_csv)
    copy_processed_inputs(files, processed_dir)

    df = read_csv_required(observation_csv)
    prepared = prepare_observation_table(df)
    prepared.to_csv(output_dir / "eta_carinae_light_echo_observations_processed.csv", index=False, encoding="utf-8-sig")

    plot_standard_velocity(prepared, output_dir / "eta_carinae_velocity_overview.png")
    plot_standard_temperature_mass(prepared, output_dir / "eta_carinae_observation_context.png")

    circular_generated = False
    if mass_csv and mass_csv.exists():
        mass_df = read_csv_required(mass_csv)
        plot_standard_circular_context_tpyx_style(mass_df, output_dir / "eta_carinae_standard_circular_context.png")
        circular_generated = True

    summary = pd.DataFrame({
        "row_count": [len(prepared)],
        "spectral_velocity_rows": [int(prepared["spectral_velocity_kmps"].notna().sum()) if "spectral_velocity_kmps" in prepared else 0],
        "bulk_velocity_rows": [int(prepared["bulk_velocity_kmps"].notna().sum()) if "bulk_velocity_kmps" in prepared else 0],
        "fast_component_rows": [int(prepared["fast_component_velocity_kmps"].notna().sum()) if "fast_component_velocity_kmps" in prepared else 0],
        "temperature_rows": [int(prepared["effective_temperature_k"].notna().sum()) if "effective_temperature_k" in prepared else 0],
        "circular_context_generated": [circular_generated],
    })
    summary.to_csv(output_dir / "eta_carinae_standard_summary.csv", index=False, encoding="utf-8-sig")
    return {"processed_rows": len(prepared), "output_dir": str(output_dir), "circular_generated": circular_generated}


def estimate_volume_proxy(row: pd.Series) -> float:
    geometry = str(row.get("geometry_class", "")).strip()
    major = parse_numeric_like(row.get("angular_major_axis_arcsec"))
    minor = parse_numeric_like(row.get("angular_minor_axis_arcsec"))
    depth = parse_numeric_like(row.get("depth_assumption_arcsec"))
    if np.isnan(depth):
        if not np.isnan(minor):
            depth = minor
        elif not np.isnan(major):
            depth = major / 2.0

    if np.isnan(major) and np.isnan(minor) and np.isnan(depth):
        return np.nan

    a = major / 2.0 if not np.isnan(major) else minor / 2.0 if not np.isnan(minor) else depth / 2.0
    b = minor / 2.0 if not np.isnan(minor) else a
    c = depth / 2.0 if not np.isnan(depth) else b

    if geometry == "torus":
        R = max(a, b)
        r = min(b, c)
        return float(2.0 * math.pi**2 * max(R, 0.0) * max(r, 0.0) ** 2)

    if geometry in ("two_ellipsoids_sum", "bipolar_lobes"):
        ellipsoid = (4.0 / 3.0) * math.pi * max(a, 0.0) * max(b, 0.0) * max(c, 0.0)
        return float(2.0 * ellipsoid)

    if geometry in ("bounding_ellipsoid_or_shell_factor", "irregular_shell"):
        return float((4.0 / 3.0) * math.pi * max(a, 0.0) * max(b, 0.0) * max(c, 0.0) * 0.5)

    return float((4.0 / 3.0) * math.pi * max(a, 0.0) * max(b, 0.0) * max(c, 0.0))


def select_mass_proxy(row: pd.Series) -> float:
    mass_value = parse_numeric_like(row.get("mass_value_msun"))
    dust_mass = parse_numeric_like(row.get("dust_mass_msun"))
    gas_mass = parse_numeric_like(row.get("gas_mass_msun"))
    if not np.isnan(mass_value):
        return mass_value
    if not np.isnan(gas_mass):
        return gas_mass
    if not np.isnan(dust_mass):
        return dust_mass
    return np.nan


def is_real_component_row(row: pd.Series) -> bool:
    label = str(row.get("component_label", "")).strip().lower()
    comp_type = str(row.get("component_type", "")).strip().lower()
    verification = str(row.get("verification_status", "")).strip().lower()
    if label.endswith("_context"):
        return False
    if "context" in comp_type:
        return False
    if verification == "context_only":
        return False
    return True


def prepare_mass_table(df: pd.DataFrame, d_bg: float = 3.0, lam: float = 0.8) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "mass_value_msun", "mass_err_msun", "dust_mass_msun", "gas_mass_msun",
        "angular_major_axis_arcsec", "angular_minor_axis_arcsec", "depth_assumption_arcsec",
        "bulk_velocity_kmps", "fast_component_velocity_kmps",
    ]:
        if col in out.columns:
            out[col] = normalize_series_numeric(out[col])

    out["is_real_component"] = out.apply(is_real_component_row, axis=1)
    out["mass_proxy_raw"] = out.apply(select_mass_proxy, axis=1)
    out["volume_proxy_raw"] = out.apply(estimate_volume_proxy, axis=1)
    out["density_proxy_raw"] = out["mass_proxy_raw"] / out["volume_proxy_raw"]

    eps = 1.0e-9
    out["log_mass_volume_proxy"] = np.where(
        out["is_real_component"] &
        out["mass_proxy_raw"].notna() &
        out["volume_proxy_raw"].notna() &
        (out["mass_proxy_raw"] > 0) &
        (out["volume_proxy_raw"] > 0),
        np.log((out["mass_proxy_raw"] + eps) / (out["volume_proxy_raw"] + eps)),
        np.nan,
    )

    out["sigma_mass_volume"] = normalize_0_1(out["log_mass_volume_proxy"])
    out["sigma_velocity"] = normalize_0_1(out["bulk_velocity_kmps"].where(out["is_real_component"], np.nan))
    anisotropy_map = {"low": 0.2, "medium": 0.5, "high": 0.9}
    out["anisotropy_proxy_raw"] = out["anisotropy_class"].astype(str).str.lower().map(anisotropy_map).where(out["is_real_component"], np.nan)
    out["sigma_anisotropy"] = normalize_0_1(out["anisotropy_proxy_raw"])

    def combine_sigma(row: pd.Series) -> float:
        if not bool(row.get("is_real_component", False)):
            return np.nan
        vals = []
        weights = []
        for key, w in [("sigma_mass_volume", 0.5), ("sigma_velocity", 0.25), ("sigma_anisotropy", 0.25)]:
            val = row.get(key)
            if not pd.isna(val):
                vals.append(float(val))
                weights.append(w)
        if not vals:
            return np.nan
        return float(sum(v * w for v, w in zip(vals, weights)) / sum(weights))

    out["sigma_eta"] = out.apply(combine_sigma, axis=1)
    out["Dw_trial_mass_volume"] = np.where(out["sigma_eta"].notna(), d_bg - lam * out["sigma_eta"], np.nan)
    return out


def plot_mass_profile(df: pd.DataFrame, output_png: Path) -> None:
    plot_df = df[df["is_real_component"]].copy()
    plt.figure(figsize=(10, 6))
    labels = plot_df["component_label"].astype(str).tolist()
    x = np.arange(len(plot_df))
    sigma = pd.to_numeric(plot_df["sigma_eta"], errors="coerce")
    dw = pd.to_numeric(plot_df["Dw_trial_mass_volume"], errors="coerce")

    ax1 = plt.gca()
    ax1.bar(x, sigma.fillna(0.0), alpha=0.7, label="sigma_eta")
    ax1.set_ylabel("Sigma")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_title("Eta Carinae component sigma / Dw profile")

    ax2 = ax1.twinx()
    ax2.plot(x, dw, marker="o", label="Dw_trial_mass_volume")
    ax2.set_ylabel("Dw trial")

    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def plot_mass_velocity_geometry(df: pd.DataFrame, output_png: Path) -> None:
    plot_df = df[df["is_real_component"]].copy()
    plt.figure(figsize=(10, 6))
    labels = plot_df["component_label"].astype(str).tolist()
    x = np.arange(len(plot_df))
    vel = pd.to_numeric(plot_df.get("bulk_velocity_kmps", pd.Series([np.nan] * len(plot_df))), errors="coerce")
    major = pd.to_numeric(plot_df.get("angular_major_axis_arcsec", pd.Series([np.nan] * len(plot_df))), errors="coerce")
    dust = pd.to_numeric(plot_df.get("dust_mass_msun", pd.Series([np.nan] * len(plot_df))), errors="coerce")

    ax1 = plt.gca()
    if vel.notna().any():
        ax1.plot(x, vel, marker="o", label="bulk_velocity_kmps")
    if major.notna().any():
        ax1.plot(x, major, marker="s", label="angular_major_axis_arcsec")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("Velocity / angular extent proxy")
    ax1.set_title("Eta Carinae component geometry context")
    ax1.legend(loc="upper left")

    if dust.notna().any():
        ax2 = ax1.twinx()
        ax2.bar(x, dust.fillna(0.0), alpha=0.25, label="dust_mass_msun")
        ax2.set_ylabel("Dust mass (Msun)")

    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def run_mass_pipeline(mass_csv: Path, source_registry_csv: Path | None, output_dir: Path, d_bg: float = 3.0, lam: float = 0.8) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed_input"
    files = [mass_csv]
    if source_registry_csv:
        files.append(source_registry_csv)
    copy_processed_inputs(files, processed_dir)

    df = read_csv_required(mass_csv)
    prepared = prepare_mass_table(df, d_bg=d_bg, lam=lam)
    prepared.to_csv(output_dir / "eta_carinae_mass_volume_processed.csv", index=False, encoding="utf-8-sig")
    plot_mass_profile(prepared, output_dir / "eta_carinae_mass_profile.png")
    plot_mass_velocity_geometry(prepared, output_dir / "eta_carinae_mass_geometry_context.png")

    calc_df = prepared[prepared["is_real_component"]].copy()
    summary = pd.DataFrame({
        "row_count_all": [len(prepared)],
        "row_count_real_components": [len(calc_df)],
        "mass_proxy_rows": [int(calc_df["mass_proxy_raw"].notna().sum())],
        "volume_proxy_rows": [int(calc_df["volume_proxy_raw"].notna().sum())],
        "sigma_rows": [int(calc_df["sigma_eta"].notna().sum())],
        "dw_rows": [int(calc_df["Dw_trial_mass_volume"].notna().sum())],
    })
    summary.to_csv(output_dir / "eta_carinae_mass_summary.csv", index=False, encoding="utf-8-sig")
    return {"processed_rows": len(prepared), "real_component_rows": len(calc_df), "output_dir": str(output_dir)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eta Carinae light echo / topological pipeline")
    parser.add_argument("--repo-root", default="", help="Project repository root")
    parser.add_argument("--input-dir", default="", help="Override input directory")
    parser.add_argument("--run-standard", action="store_true", help="Run standard/light echo pipeline only")
    parser.add_argument("--run-mass", action="store_true", help="Run topological mass pipeline only")
    parser.add_argument("--d-bg", type=float, default=3.0, help="Background effective dimension")
    parser.add_argument("--lambda-mass", type=float, default=0.8, help="Lambda for Dw trial")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(script_path)
    input_dir = find_target_input_dir(repo_root, args.input_dir if args.input_dir else None)
    target_folder_name = input_dir.name

    obs_csv = input_dir / "eta_carinae_light_echo_observations_input_v1.csv"
    mass_csv = input_dir / "eta_carinae_mass_volume_auxiliary_input_v1.csv"
    source_registry_csv = input_dir / "eta_carinae_source_registry_input_v1.csv"

    if not args.run_standard and not args.run_mass:
        run_standard = True
        run_mass = True
    else:
        run_standard = args.run_standard
        run_mass = args.run_mass

    stamp = timestamp_label()
    standard_result = None
    mass_result = None

    if run_standard:
        standard_out = repo_root / "results" / "Echo of Light" / "output" / target_folder_name / "standard" / stamp
        standard_result = run_standard_pipeline(obs_csv, mass_csv, standard_out)

    if run_mass:
        mass_out = repo_root / "results" / "Echo of Light" / "output" / target_folder_name / "topological" / "mass" / stamp
        mass_result = run_mass_pipeline(mass_csv, source_registry_csv, mass_out, d_bg=args.d_bg, lam=args.lambda_mass)

    print("Eta Carinae pipeline completed.")
    print(f"repo_root = {repo_root}")
    print(f"input_dir = {input_dir}")
    if standard_result:
        print(f"standard_output = {standard_result['output_dir']}")
        print(f"standard_rows = {standard_result['processed_rows']}")
        print(f"standard_circular_context_generated = {standard_result['circular_generated']}")
    if mass_result:
        print(f"mass_output = {mass_result['output_dir']}")
        print(f"mass_rows_all = {mass_result['processed_rows']}")
        print(f"mass_rows_real_components = {mass_result['real_component_rows']}")


if __name__ == "__main__":
    main()
