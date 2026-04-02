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


def _first_row(df: pd.DataFrame, component_label: str):
    subset = df[df["component_label"].astype(str) == component_label]
    if subset.empty:
        return None
    return subset.iloc[0]


def get_component_draw_specs(mass_df: pd.DataFrame) -> list[dict]:
    specs = []

    hom = _first_row(mass_df, "homunculus_main_lobes")
    if hom is not None:
        major = parse_numeric_like(hom.get("angular_major_axis_arcsec"))
        minor = parse_numeric_like(hom.get("angular_minor_axis_arcsec"))
        if np.isnan(major):
            major = 20.0
        if np.isnan(minor):
            minor = 10.0
        specs.append({
            "component_label": "homunculus_main_lobes",
            "shape": "ellipse",
            "width": minor,
            "height": major,
            "angle": 0.0,
            "center": (0.0, 0.0),
            "annotation_xy": (0.0, major / 2 + 0.7),
            "annotation_text": "Homunculus",
        })

    tor = _first_row(mass_df, "dust_torus")
    if tor is not None:
        specs.append({
            "component_label": "dust_torus",
            "shape": "circle",
            "radius": 2.3,
            "center": (0.0, 0.0),
            "annotation_xy": (2.7, -0.15),
            "annotation_text": "dust torus",
        })

    out = _first_row(mass_df, "outer_ejecta")
    if out is None:
        out = _first_row(mass_df, "outer_ejecta_context")
    if out is not None:
        specs.append({
            "component_label": "outer_ejecta",
            "shape": "circle",
            "radius": 6.2,
            "center": (0.0, 0.0),
            "annotation_xy": (6.6, 0.0),
            "annotation_text": "outer ejecta",
            "knots": [(4.8, 1.4), (-4.1, 2.6), (2.4, -4.9), (-5.0, -1.3)],
        })

    return specs


def add_ring_guides(ax) -> None:
    for r in [2, 4, 6, 8, 10]:
        ax.add_patch(Circle((0, 0), r, fill=False, linewidth=0.8, linestyle=":"))
        ax.text(r, 0.18, f'{r}"', fontsize=9, ha="center", va="bottom")


def add_center_system(ax) -> list:
    star = Circle((0, 0), radius=0.22, fill=False, linewidth=2.0)
    ax.add_patch(star)
    ax.plot(0, 0, marker="+", markersize=10, markeredgewidth=1.8)
    ax.text(0.0, -0.55, "Eta Car", ha="center", va="top", fontsize=10)
    return [
        Line2D([0], [0], marker='o', markersize=8, markerfacecolor='none', markeredgecolor='black', linestyle='None', label='Central star'),
        Line2D([0], [0], marker='+', markersize=10, color='black', linestyle='None', label='COM / center'),
    ]


def finalize_circular_axes(ax, title: str, info_lines: list[str]) -> None:
    ax.text(
        0.98, 0.04, "\n".join(info_lines),
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black")
    )
    ax.set_xlim(-11.5, 11.5)
    ax.set_ylim(-11.5, 11.5)
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_title(title)


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


def derive_phase_rank(label: str, phase_reference: str) -> float:
    text = f"{label} {phase_reference}".lower()
    if "discovery" in text:
        return 0.0
    if "early" in text:
        return 1.0
    if "bulk" in text or "plateau" in text:
        return 2.0
    if "fast" in text or "tail" in text:
        return 3.0
    return 0.0


def derive_phase_year(row: pd.Series) -> float:
    event_text = str(row.get("event_time_span_text", "")).strip()
    label = str(row.get("observation_label", "")).strip()
    phase_reference = str(row.get("phase_reference", "")).strip().lower()

    numbers = [int(x) for x in re.findall(r"\d{4}", event_text)]
    if len(numbers) >= 2:
        if "discovery" in label.lower():
            return float((numbers[0] + numbers[1]) / 2.0)
        return float(numbers[0] + 4.0)

    if "1850" in event_text or "1850" in phase_reference:
        if "early" in label.lower():
            return 1851.0
        if "bulk" in label.lower():
            return 1854.0
        if "fast" in label.lower():
            return 1858.0
        return 1854.0

    return 1848.0 + derive_phase_rank(label, phase_reference) * 2.0


def reconstruct_standard_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["phase_rank"] = [
        derive_phase_rank(str(label), str(phase))
        for label, phase in zip(out["observation_label"], out["phase_reference"])
    ]
    out["phase_year_reconstructed"] = out.apply(derive_phase_year, axis=1)

    velocity_candidates = []
    for _, row in out.iterrows():
        vals = [
            parse_numeric_like(row.get("spectral_velocity_kmps")),
            parse_numeric_like(row.get("bulk_velocity_kmps")),
            parse_numeric_like(row.get("fast_component_velocity_kmps")),
        ]
        vals = [abs(v) for v in vals if not np.isnan(v)]
        velocity_candidates.append(float(np.mean(vals)) if vals else np.nan)
    out["velocity_magnitude_proxy_kmps"] = velocity_candidates

    return out.sort_values(["phase_year_reconstructed", "phase_rank"]).reset_index(drop=True)


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
    plt.savefig(output_png, dpi=220)
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
    plt.savefig(output_png, dpi=220)
    plt.close()


def plot_standard_timeseries_reconstructed(df: pd.DataFrame, output_png: Path) -> None:
    plt.figure(figsize=(10, 6))
    years = pd.to_numeric(df["phase_year_reconstructed"], errors="coerce")
    vel = pd.to_numeric(df["velocity_magnitude_proxy_kmps"], errors="coerce")
    plt.plot(years, vel, marker="o")
    for _, row in df.iterrows():
        plt.annotate(
            row["observation_label"],
            (row["phase_year_reconstructed"], row["velocity_magnitude_proxy_kmps"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    plt.xlabel("Reconstructed event-phase year")
    plt.ylabel("Velocity magnitude proxy (km/s)")
    plt.title("Eta Carinae reconstructed light echo phase timeline")
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def plot_standard_circular_context(mass_df: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    add_ring_guides(ax)
    legend_handles = add_center_system(ax)

    for spec in get_component_draw_specs(mass_df):
        if spec["shape"] == "ellipse":
            patch = Ellipse(spec["center"], width=spec["width"], height=spec["height"], angle=spec["angle"], fill=False, linewidth=2.0)
            ax.add_patch(patch)
            ax.text(*spec["annotation_xy"], spec["annotation_text"], ha="center", va="bottom", fontsize=10)
            legend_handles.append(Line2D([0], [0], color='black', linewidth=2.0, label='Homunculus lobes'))
        elif spec["shape"] == "circle" and spec["component_label"] == "dust_torus":
            patch = Circle(spec["center"], spec["radius"], fill=False, linewidth=1.5, linestyle="--")
            ax.add_patch(patch)
            ax.text(*spec["annotation_xy"], spec["annotation_text"], ha="left", va="center", fontsize=9)
            legend_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Dust torus'))
        elif spec["shape"] == "circle" and spec["component_label"] == "outer_ejecta":
            patch = Circle(spec["center"], spec["radius"], fill=False, linewidth=1.2, linestyle="-.")
            ax.add_patch(patch)
            for x, y in spec.get("knots", []):
                ax.add_patch(Circle((x, y), 0.14, fill=True, alpha=0.7))
            ax.text(*spec["annotation_xy"], spec["annotation_text"], ha="left", va="center", fontsize=9)
            legend_handles.append(Line2D([0], [0], color='black', linestyle='-.', linewidth=1.2, label='Outer ejecta shell'))
            legend_handles.append(Line2D([0], [0], marker='o', markersize=5, color='black', linestyle='None', label='Representative knots'))

    dedup = {}
    for h in legend_handles:
        dedup[h.get_label()] = h
    leg1 = ax.legend(handles=list(dedup.values()), loc="upper right", frameon=True)
    ax.add_artist(leg1)

    info_lines = ['Angular guide rings', '2", 4", 6", 8", 10"', '', 'Drawing mode:', 'standard context schematic', 'not epoch-by-epoch echo map']
    finalize_circular_axes(ax, "Eta Carinae standard circular context", info_lines)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def run_standard_pipeline(observation_csv: Path, mass_csv: Optional[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed_input"
    files = [observation_csv]
    if mass_csv:
        files.append(mass_csv)
    copy_processed_inputs(files, processed_dir)

    df = read_csv_required(observation_csv)
    prepared = prepare_observation_table(df)
    reconstructed = reconstruct_standard_timeseries(prepared)
    prepared.to_csv(output_dir / "eta_carinae_light_echo_observations_processed.csv", index=False, encoding="utf-8-sig")
    reconstructed.to_csv(output_dir / "eta_carinae_standard_timeseries_reconstructed.csv", index=False, encoding="utf-8-sig")

    plot_standard_velocity(prepared, output_dir / "eta_carinae_velocity_overview.png")
    plot_standard_temperature_mass(prepared, output_dir / "eta_carinae_observation_context.png")
    plot_standard_timeseries_reconstructed(reconstructed, output_dir / "eta_carinae_reconstructed_phase_timeline.png")

    circular_generated = False
    if mass_csv and mass_csv.exists():
        mass_df = read_csv_required(mass_csv)
        plot_standard_circular_context(mass_df, output_dir / "eta_carinae_standard_circular_context.png")
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
        if not np.isnan(major) and not np.isnan(minor):
            depth = math.sqrt(max(major * minor, 0.0))
        elif not np.isnan(minor):
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


def derive_component_embedding_dim(row: pd.Series) -> float:
    geometry = str(row.get("geometry_class", "")).strip().lower()
    major = parse_numeric_like(row.get("angular_major_axis_arcsec"))
    minor = parse_numeric_like(row.get("angular_minor_axis_arcsec"))
    depth = parse_numeric_like(row.get("depth_assumption_arcsec"))
    observed_axes = sum([not np.isnan(v) and v > 0 for v in [major, minor, depth]])
    base_dim = float(max(1, observed_axes))
    if any(key in geometry for key in ["bipolar", "ellipsoid", "shell", "torus"]):
        base_dim = max(base_dim, 3.0)
    return float(base_dim)


def derive_d_bg(real_df: pd.DataFrame) -> float:
    dims = real_df.apply(derive_component_embedding_dim, axis=1)
    dims = pd.to_numeric(dims, errors="coerce").dropna()
    if dims.empty:
        return 3.0
    return float(dims.median())


def derive_anisotropy_proxy(row: pd.Series) -> float:
    major = parse_numeric_like(row.get("angular_major_axis_arcsec"))
    minor = parse_numeric_like(row.get("angular_minor_axis_arcsec"))
    if not np.isnan(major) and not np.isnan(minor) and major > 0 and minor > 0:
        ratio = max(major, minor) / min(major, minor)
        return float(np.log(ratio))
    return np.nan


def derive_weight(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    completeness = float(numeric.notna().mean())
    variability = float(numeric.dropna().std()) if numeric.notna().sum() >= 2 else 0.0
    return completeness * (1.0 + variability)


def derive_lambda_from_sigma(raw_sigma: pd.Series, d_bg: float) -> float:
    valid = pd.to_numeric(raw_sigma, errors="coerce").dropna()
    if valid.empty:
        return 1.0
    char_sigma = float(valid.abs().median())
    if char_sigma <= 0:
        return 1.0
    return float(min(1.0, 0.4 / char_sigma))


def prepare_mass_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
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
    out["anisotropy_proxy_raw"] = out.apply(derive_anisotropy_proxy, axis=1).where(out["is_real_component"], np.nan)
    out["sigma_anisotropy"] = normalize_0_1(out["anisotropy_proxy_raw"])

    real_df = out[out["is_real_component"]].copy()
    d_bg = derive_d_bg(real_df)

    weight_raw = {
        "sigma_mass_volume": derive_weight(real_df["sigma_mass_volume"]),
        "sigma_velocity": derive_weight(real_df["sigma_velocity"]),
        "sigma_anisotropy": derive_weight(real_df["sigma_anisotropy"]),
    }
    weight_sum = sum(weight_raw.values()) if sum(weight_raw.values()) > 0 else 1.0
    weights = {k: v / weight_sum for k, v in weight_raw.items()}

    def combine_sigma(row: pd.Series) -> float:
        if not bool(row.get("is_real_component", False)):
            return np.nan
        vals = []
        wts = []
        for key in ["sigma_mass_volume", "sigma_velocity", "sigma_anisotropy"]:
            val = row.get(key)
            wt = weights.get(key, 0.0)
            if not pd.isna(val) and wt > 0:
                vals.append(float(val))
                wts.append(float(wt))
        if not vals:
            return np.nan
        return float(sum(v * w for v, w in zip(vals, wts)) / sum(wts))

    out["sigma_eta"] = out.apply(combine_sigma, axis=1)
    lam = derive_lambda_from_sigma(out["sigma_eta"].where(out["is_real_component"], np.nan), d_bg)
    out["Dw_trial_mass_volume"] = np.where(out["sigma_eta"].notna(), d_bg - lam * out["sigma_eta"], np.nan)

    derivation = {
        "d_bg_derived": d_bg,
        "lambda_derived": lam,
        "weight_sigma_mass_volume": weights["sigma_mass_volume"],
        "weight_sigma_velocity": weights["sigma_velocity"],
        "weight_sigma_anisotropy": weights["sigma_anisotropy"],
    }
    return out, derivation


def plot_mass_profile(df: pd.DataFrame, output_png: Path) -> None:
    plot_df = df[df["is_real_component"]].copy()
    plt.figure(figsize=(10, 6))
    if plot_df.empty:
        plt.text(0.5, 0.5, "No real component rows for sigma/Dw profile", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(output_png, dpi=220)
        plt.close()
        return

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


def plot_component_sigma_breakdown(df: pd.DataFrame, output_png: Path) -> None:
    plot_df = df[df["is_real_component"]].copy()
    plt.figure(figsize=(10, 6))
    if plot_df.empty:
        plt.text(0.5, 0.5, "No real component rows for sigma breakdown", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(output_png, dpi=220)
        plt.close()
        return

    x = np.arange(len(plot_df))
    labels = plot_df["component_label"].astype(str).tolist()
    width = 0.22
    for offset, col in zip([-width, 0.0, width], ["sigma_mass_volume", "sigma_velocity", "sigma_anisotropy"]):
        vals = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)
        plt.bar(x + offset, vals, width=width, label=col, alpha=0.8)

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Sigma component")
    plt.title("Eta Carinae sigma component breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def plot_mass_velocity_geometry(df: pd.DataFrame, output_png: Path) -> None:
    plot_df = df[df["is_real_component"]].copy()
    plt.figure(figsize=(10, 6))
    if plot_df.empty:
        plt.text(0.5, 0.5, "No real component rows for geometry context", ha="center", va="center", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(output_png, dpi=220)
        plt.close()
        return

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


def component_linewidth(style_value: float) -> float:
    if np.isnan(style_value):
        return 1.2
    return 1.4 + 4.0 * float(style_value)


def plot_topological_circular_context(df: pd.DataFrame, output_png: Path) -> None:
    plot_df = df[df["is_real_component"]].copy()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    add_ring_guides(ax)
    legend_handles = add_center_system(ax)

    lookup = {}
    for _, row in plot_df.iterrows():
        lookup[str(row["component_label"])] = row

    for spec in get_component_draw_specs(df):
        row = lookup.get(spec["component_label"])
        sigma_mv = parse_numeric_like(row.get("sigma_mass_volume")) if row is not None else np.nan
        sigma_v = parse_numeric_like(row.get("sigma_velocity")) if row is not None else np.nan
        sigma_a = parse_numeric_like(row.get("sigma_anisotropy")) if row is not None else np.nan
        sigma_eta = parse_numeric_like(row.get("sigma_eta")) if row is not None else np.nan
        dw = parse_numeric_like(row.get("Dw_trial_mass_volume")) if row is not None else np.nan

        if spec["component_label"] == "homunculus_main_lobes":
            driver = np.nanmax([sigma_mv, sigma_a]) if not (np.isnan(sigma_mv) and np.isnan(sigma_a)) else np.nan
            lw = component_linewidth(driver)
            linestyle = "-" if np.isnan(sigma_a) or sigma_a < 0.35 else "-."
            patch = Ellipse(spec["center"], width=spec["width"], height=spec["height"], angle=spec["angle"], fill=False, linewidth=lw, linestyle=linestyle)
            ax.add_patch(patch)
            ann = spec["annotation_text"]
            if row is not None:
                ann += f"\nσmv={sigma_mv:.2f} σa={sigma_a:.2f}\nσ={sigma_eta:.2f} / Dw={dw:.2f}"
            ax.text(spec["annotation_xy"][0], spec["annotation_xy"][1], ann, ha="center", va="bottom", fontsize=9)
            legend_handles.append(Line2D([0], [0], color='black', linewidth=lw, linestyle=linestyle, label='Topological Homunculus'))

        elif spec["component_label"] == "dust_torus":
            driver = sigma_mv
            lw = component_linewidth(driver)
            linestyle = "--" if np.isnan(driver) or driver < 0.35 else ":"
            patch = Circle(spec["center"], spec["radius"], fill=False, linewidth=lw, linestyle=linestyle)
            ax.add_patch(patch)
            ann = spec["annotation_text"]
            if row is not None:
                ann += f"\nσmv={sigma_mv:.2f}\nσ={sigma_eta:.2f} / Dw={dw:.2f}"
            ax.text(spec["annotation_xy"][0], spec["annotation_xy"][1], ann, ha="left", va="center", fontsize=9)
            legend_handles.append(Line2D([0], [0], color='black', linewidth=lw, linestyle=linestyle, label='Topological dust torus'))

        elif spec["component_label"] == "outer_ejecta":
            driver = np.nanmax([sigma_v, sigma_a]) if not (np.isnan(sigma_v) and np.isnan(sigma_a)) else np.nan
            lw = component_linewidth(driver)
            linestyle = "-." if np.isnan(driver) or driver < 0.35 else (0, (5, 2))
            patch = Circle(spec["center"], spec["radius"], fill=False, linewidth=lw, linestyle=linestyle)
            ax.add_patch(patch)
            knot_radius = 0.14 if np.isnan(driver) else 0.14 + 0.14 * float(driver)
            for x, y in spec.get("knots", []):
                ax.add_patch(Circle((x, y), knot_radius, fill=True, alpha=0.7))
            ann = spec["annotation_text"]
            if row is not None:
                ann += f"\nσv={sigma_v:.2f} σa={sigma_a:.2f}\nσ={sigma_eta:.2f} / Dw={dw:.2f}"
            ax.text(spec["annotation_xy"][0], spec["annotation_xy"][1], ann, ha="left", va="center", fontsize=9)
            legend_handles.append(Line2D([0], [0], color='black', linewidth=lw, linestyle=linestyle, label='Topological outer ejecta'))

    dedup = {}
    for h in legend_handles:
        dedup[h.get_label()] = h
    leg1 = ax.legend(handles=list(dedup.values()), loc="upper right", frameon=True)
    ax.add_artist(leg1)

    info_lines = ['Same circular frame as standard', 'style reflects component sigma', '', 'Comparison mode:', 'component-wise topological context', 'shared geometry / shared scale']
    finalize_circular_axes(ax, "Eta Carinae topological circular context", info_lines)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def run_mass_pipeline(mass_csv: Path, source_registry_csv: Optional[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = output_dir / "processed_input"
    files = [mass_csv]
    if source_registry_csv:
        files.append(source_registry_csv)
    copy_processed_inputs(files, processed_dir)

    df = read_csv_required(mass_csv)
    prepared, derivation = prepare_mass_table(df)
    prepared.to_csv(output_dir / "eta_carinae_mass_volume_processed.csv", index=False, encoding="utf-8-sig")
    plot_mass_profile(prepared, output_dir / "eta_carinae_mass_profile.png")
    plot_component_sigma_breakdown(prepared, output_dir / "eta_carinae_sigma_component_breakdown.png")
    plot_mass_velocity_geometry(prepared, output_dir / "eta_carinae_mass_geometry_context.png")
    plot_topological_circular_context(prepared, output_dir / "eta_carinae_topological_circular_context.png")

    calc_df = prepared[prepared["is_real_component"]].copy()
    summary = pd.DataFrame({
        "row_count_all": [len(prepared)],
        "row_count_real_components": [len(calc_df)],
        "mass_proxy_rows": [int(calc_df["mass_proxy_raw"].notna().sum())],
        "volume_proxy_rows": [int(calc_df["volume_proxy_raw"].notna().sum())],
        "sigma_mass_volume_rows": [int(calc_df["sigma_mass_volume"].notna().sum())],
        "sigma_velocity_rows": [int(calc_df["sigma_velocity"].notna().sum())],
        "sigma_anisotropy_rows": [int(calc_df["sigma_anisotropy"].notna().sum())],
        "sigma_rows": [int(calc_df["sigma_eta"].notna().sum())],
        "dw_rows": [int(calc_df["Dw_trial_mass_volume"].notna().sum())],
        "d_bg_derived": [derivation["d_bg_derived"]],
        "lambda_derived": [derivation["lambda_derived"]],
        "weight_sigma_mass_volume": [derivation["weight_sigma_mass_volume"]],
        "weight_sigma_velocity": [derivation["weight_sigma_velocity"]],
        "weight_sigma_anisotropy": [derivation["weight_sigma_anisotropy"]],
    })
    summary.to_csv(output_dir / "eta_carinae_mass_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([derivation]).to_csv(output_dir / "eta_carinae_topological_derivation_terms.csv", index=False, encoding="utf-8-sig")

    return {"processed_rows": len(prepared), "real_component_rows": len(calc_df), "output_dir": str(output_dir), "d_bg_derived": derivation["d_bg_derived"], "lambda_derived": derivation["lambda_derived"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Eta Carinae light echo / topological pipeline")
    parser.add_argument("--repo-root", default="", help="Project repository root")
    parser.add_argument("--input-dir", default="", help="Override input directory")
    parser.add_argument("--run-standard", action="store_true", help="Run standard/light echo pipeline only")
    parser.add_argument("--run-mass", action="store_true", help="Run topological mass pipeline only")
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
        mass_result = run_mass_pipeline(mass_csv, source_registry_csv, mass_out)

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
        print(f"d_bg_derived = {mass_result['d_bg_derived']}")
        print(f"lambda_derived = {mass_result['lambda_derived']}")


if __name__ == "__main__":
    main()
