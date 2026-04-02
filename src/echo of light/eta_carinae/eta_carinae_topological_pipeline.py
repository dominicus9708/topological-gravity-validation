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
    text = text.replace(",", "").replace("−", "-").replace("–", "-")
    text = text.replace("~", "").replace("≈", "").replace(">", "").replace("<", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(m.group()) if m else np.nan


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
    specs: list[dict] = []

    hom = _first_row(mass_df, "homunculus_main_lobes")
    if hom is not None:
        major = parse_numeric_like(hom.get("angular_major_axis_arcsec"))
        minor = parse_numeric_like(hom.get("angular_minor_axis_arcsec"))
        if np.isnan(major):
            major = 20.0
        if np.isnan(minor):
            minor = 10.0
        specs.append(
            {
                "component_label": "homunculus_main_lobes",
                "shape": "ellipse",
                "width": minor,
                "height": major,
                "angle": 0.0,
                "center": (0.0, 0.0),
                "anchor_xy": (-2.8, 8.2),
                "text_xy": (-8.4, 7.2),
                "annotation_text": "Homunculus",
                "ha": "left",
            }
        )

    tor = _first_row(mass_df, "dust_torus")
    if tor is not None:
        specs.append(
            {
                "component_label": "dust_torus",
                "shape": "circle",
                "radius": 2.3,
                "center": (0.0, 0.0),
                "anchor_xy": (2.2, -2.1),
                "text_xy": (5.0, -6.0),
                "annotation_text": "Dust torus",
                "ha": "left",
            }
        )

    out = _first_row(mass_df, "outer_ejecta_component")
    if out is None:
        out = _first_row(mass_df, "outer_ejecta")
    if out is None:
        out = _first_row(mass_df, "outer_ejecta_context")
    if out is not None:
        specs.append(
            {
                "component_label": "outer_ejecta_component",
                "fallback_labels": ["outer_ejecta", "outer_ejecta_context"],
                "shape": "circle",
                "radius": 6.2,
                "center": (0.0, 0.0),
                "anchor_xy": (5.8, -1.8),
                "text_xy": (6.0, -1.2),
                "annotation_text": "Out ejecta",
                "knots": [(4.8, 1.4), (-4.1, 2.6), (2.4, -4.9), (-5.0, -1.3)],
                "ha": "left",
            }
        )

    return specs
def add_box_annotation(ax, text: str, text_xy: tuple[float, float], anchor_xy: tuple[float, float], ha: str) -> None:
    ax.annotate(
        text,
        xy=anchor_xy,
        xytext=text_xy,
        textcoords="data",
        ha=ha,
        va="center",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            alpha=0.92,
            edgecolor="black",
        ),
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            lw=1.0,
            shrinkA=0,
            shrinkB=0,
            connectionstyle="angle3,angleA=0,angleB=90",
        ),
        zorder=8,
    )

def add_ring_guides(ax) -> None:
    for r in [2, 4, 6, 8, 10]:
        ax.add_patch(Circle((0, 0), r, fill=False, linewidth=0.8, linestyle=":", edgecolor="0.55"))
        ax.text(r, 0.18, f'{r}"', fontsize=9, ha="center", va="bottom")


def add_center_system(ax) -> list[Line2D]:
    ax.add_patch(Circle((0, 0), radius=0.26, fill=False, linewidth=2.2, edgecolor="black"))
    ax.plot(0, 0, marker="+", markersize=10, markeredgewidth=1.8, color="black")
    ax.text(0.0, -0.7, "Eta Car", ha="center", va="top", fontsize=10)
    return [
        Line2D([0], [0], marker="o", markersize=8, markerfacecolor="none", markeredgecolor="black", linestyle="None", label="Central star"),
        Line2D([0], [0], marker="+", markersize=10, color="black", linestyle="None", label="COM / center"),
    ]


def finalize_circular_axes(ax, title: str, info_lines: list[str]) -> None:
    ax.text(
        0.98,
        0.04,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="black"),
    )
    ax.set_xlim(-11.5, 11.5)
    ax.set_ylim(-11.8, 11.8)
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_title(title)


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


def derive_outer_ejecta_component(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    row = None
    for label in ["outer_ejecta_component", "outer_ejecta", "outer_ejecta_context"]:
        subset = out[out["component_label"].astype(str) == label]
        if not subset.empty:
            row = subset.iloc[0].copy()
            break
    if row is None:
        return out

    promoted = row.copy()
    promoted["component_label"] = "outer_ejecta_component"
    promoted["component_type"] = "irregular_ejecta"
    promoted["verification_status"] = "input_ready_with_context"

    if np.isnan(parse_numeric_like(promoted.get("angular_major_axis_arcsec"))):
        promoted["angular_major_axis_arcsec"] = 12.4
    if np.isnan(parse_numeric_like(promoted.get("angular_minor_axis_arcsec"))):
        promoted["angular_minor_axis_arcsec"] = 8.6
    if np.isnan(parse_numeric_like(promoted.get("depth_assumption_arcsec"))):
        promoted["depth_assumption_arcsec"] = 7.5
    if np.isnan(parse_numeric_like(promoted.get("mass_value_msun"))):
        promoted["mass_value_msun"] = 1.0
    if np.isnan(parse_numeric_like(promoted.get("bulk_velocity_kmps"))):
        promoted["bulk_velocity_kmps"] = 1500.0
    if not str(promoted.get("geometry_class", "")).strip():
        promoted["geometry_class"] = "irregular_shell"

    promoted["proxy_rule_applied"] = "outer_ejecta_context_promoted_to_calculable_component"

    out = out[out["component_label"].astype(str) != "outer_ejecta_component"].copy()
    out = pd.concat([out, pd.DataFrame([promoted])], ignore_index=True)
    return out


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


def build_calculable_component_table(df: pd.DataFrame) -> pd.DataFrame:
    out = derive_outer_ejecta_component(df).copy()
    out["proxy_rule_applied"] = out.get("proxy_rule_applied", "")

    hom_idx = out.index[out["component_label"].astype(str) == "homunculus_main_lobes"]
    eruption_rows = out[out["component_label"].astype(str) == "great_eruption_bulk_ejecta_context"]
    eruption_mass = np.nan
    if not eruption_rows.empty:
        eruption_mass = parse_numeric_like(eruption_rows.iloc[0].get("mass_value_msun"))
    if np.isnan(eruption_mass):
        eruption_mass = 10.0

    for idx in hom_idx:
        if np.isnan(parse_numeric_like(out.at[idx, "mass_value_msun"])):
            out.at[idx, "mass_value_msun"] = eruption_mass
            out.at[idx, "proxy_rule_applied"] = "homunculus_mass_linked_to_great_eruption_lower_bound"

    tor_idx = out.index[out["component_label"].astype(str) == "dust_torus"]
    for idx in tor_idx:
        applied: list[str] = []
        if np.isnan(parse_numeric_like(out.at[idx, "angular_major_axis_arcsec"])):
            out.at[idx, "angular_major_axis_arcsec"] = 4.6
            applied.append("major_axis_proxy")
        if np.isnan(parse_numeric_like(out.at[idx, "angular_minor_axis_arcsec"])):
            out.at[idx, "angular_minor_axis_arcsec"] = 3.1
            applied.append("minor_axis_proxy")
        if np.isnan(parse_numeric_like(out.at[idx, "depth_assumption_arcsec"])):
            out.at[idx, "depth_assumption_arcsec"] = 2.2
            applied.append("depth_proxy")
        if np.isnan(parse_numeric_like(out.at[idx, "bulk_velocity_kmps"])):
            out.at[idx, "bulk_velocity_kmps"] = 120.0
            applied.append("bulk_velocity_proxy")
        if not str(out.at[idx, "geometry_class"]).strip():
            out.at[idx, "geometry_class"] = "torus"
            applied.append("torus_geometry_proxy")
        if applied:
            out.at[idx, "proxy_rule_applied"] = "dust_torus_" + "|".join(applied)

    return out


def prepare_mass_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = build_calculable_component_table(df)

    for col in [
        "mass_value_msun",
        "mass_err_msun",
        "dust_mass_msun",
        "gas_mass_msun",
        "angular_major_axis_arcsec",
        "angular_minor_axis_arcsec",
        "depth_assumption_arcsec",
        "bulk_velocity_kmps",
        "fast_component_velocity_kmps",
    ]:
        if col in out.columns:
            out[col] = normalize_series_numeric(out[col])

    out["is_real_component"] = out.apply(is_real_component_row, axis=1)
    out["mass_proxy_raw"] = out.apply(select_mass_proxy, axis=1)
    out["volume_proxy_raw"] = out.apply(estimate_volume_proxy, axis=1)
    out["density_proxy_raw"] = out["mass_proxy_raw"] / out["volume_proxy_raw"]

    eps = 1.0e-9
    out["log_mass_volume_proxy"] = np.where(
        out["is_real_component"]
        & out["mass_proxy_raw"].notna()
        & out["volume_proxy_raw"].notna()
        & (out["mass_proxy_raw"] > 0)
        & (out["volume_proxy_raw"] > 0),
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
        vals: list[float] = []
        wts: list[float] = []
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
    out["interpretation_role"] = np.where(
        ~out["is_real_component"],
        "excluded_context",
        np.where(
            out["sigma_eta"] >= 0.70,
            "strong_core_or_major_structural_distortion",
            np.where(out["sigma_eta"] >= 0.35, "mid_strength_structural_distortion", "weak_background_like_component"),
        ),
    )

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
    ax1.bar(x, sigma.fillna(0.0), alpha=0.75, label="sigma_eta")
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
        plt.bar(x + offset, vals, width=width, label=col, alpha=0.82)

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
        return 1.4
    return 1.4


def find_lookup_row(lookup: dict, spec: dict):
    if spec["component_label"] in lookup:
        return lookup[spec["component_label"]]
    for alt in spec.get("fallback_labels", []):
        if alt in lookup:
            return lookup[alt]
    return None


def style_values_text(name, sigma_mv, sigma_v, sigma_a, sigma_eta, dw) -> str:
    parts = [name]
    if not np.isnan(sigma_mv):
        parts.append(f"σmv={sigma_mv:.2f}")
    if not np.isnan(sigma_v):
        parts.append(f"σv={sigma_v:.2f}")
    if not np.isnan(sigma_a):
        parts.append(f"σa={sigma_a:.2f}")
    if not np.isnan(sigma_eta):
        parts.append(f"σ={sigma_eta:.2f}")
    if not np.isnan(dw):
        parts.append(f"Dw={dw:.2f}")
    return "\n".join(parts)

def plot_topological_circular_context(df: pd.DataFrame, output_png: Path) -> None:
    plot_df = df[df["is_real_component"]].copy()
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal", adjustable="box")
    add_ring_guides(ax)
    legend_handles = add_center_system(ax)
    lookup = {str(row["component_label"]): row for _, row in plot_df.iterrows()}

    for spec in get_component_draw_specs(df):
        row = find_lookup_row(lookup, spec)
        sigma_mv = parse_numeric_like(row.get("sigma_mass_volume")) if row is not None else np.nan
        sigma_v = parse_numeric_like(row.get("sigma_velocity")) if row is not None else np.nan
        sigma_a = parse_numeric_like(row.get("sigma_anisotropy")) if row is not None else np.nan
        sigma_eta = parse_numeric_like(row.get("sigma_eta")) if row is not None else np.nan
        dw = parse_numeric_like(row.get("Dw_trial_mass_volume")) if row is not None else np.nan

        if spec["component_label"] == "homunculus_main_lobes":
            lw = 1.4
            linestyle = "-." if not (np.isnan(sigma_a) or sigma_a < 0.35) else "-"
            patch = Ellipse(
                spec["center"],
                width=spec["width"],
                height=spec["height"],
                angle=spec["angle"],
                fill=False,
                linewidth=lw,
                linestyle=linestyle,
            )
            ax.add_patch(patch)

            ann = style_values_text(spec["annotation_text"], sigma_mv, np.nan, sigma_a, sigma_eta, dw)
            add_box_annotation(ax, ann, spec["text_xy"], spec["anchor_xy"], spec["ha"])

            legend_handles.append(
                Line2D([0], [0], color="black", linewidth=lw, linestyle=linestyle, label="Topological Homunculus")
            )

        elif spec["component_label"] == "dust_torus":
            lw = 1.7
            linestyle = "--" if np.isnan(sigma_mv) or sigma_mv < 0.35 else ":"
            patch = Circle(
                spec["center"],
                spec["radius"],
                fill=False,
                linewidth=lw,
                linestyle=linestyle,
            )
            ax.add_patch(patch)

            ann = style_values_text(spec["annotation_text"], sigma_mv, np.nan, np.nan, sigma_eta, dw)
            add_box_annotation(ax, ann, spec["text_xy"], spec["anchor_xy"], spec["ha"])

            legend_handles.append(
                Line2D([0], [0], color="black", linewidth=lw, linestyle=linestyle, label="Topological Dust torus")
            )

        elif spec["component_label"] == "outer_ejecta_component":
            lw = 1.4
            linestyle = (0, (5, 2)) if not (np.isnan(sigma_v) or sigma_v < 0.35) else "-."
            patch = Circle(
                spec["center"],
                spec["radius"],
                fill=False,
                linewidth=lw,
                linestyle=linestyle,
            )
            ax.add_patch(patch)

            knot_radius = 0.14
            for x, y in spec.get("knots", []):
                ax.add_patch(Circle((x, y), knot_radius, fill=True, alpha=0.7, color="tab:blue"))

            ann = style_values_text(spec["annotation_text"], sigma_mv, sigma_v, sigma_a, sigma_eta, dw)
            add_box_annotation(ax, ann, spec["text_xy"], spec["anchor_xy"], spec["ha"])

            legend_handles.append(
                Line2D([0], [0], color="black", linewidth=lw, linestyle=linestyle, label="Topological Out ejecta")
            )

    dedup: dict[str, Line2D] = {}
    for h in legend_handles:
        dedup[h.get_label()] = h

    primary_labels = [
        "Central star",
        "COM / center",
        "Topological Homunculus",
        "Topological Dust torus",
        "Topological Out ejecta",
    ]
    primary_handles = [dedup[k] for k in primary_labels if k in dedup]
    leg1 = ax.legend(handles=primary_handles, loc="upper right", frameon=True)
    ax.add_artist(leg1)

    blue_handle = Line2D(
        [0], [0],
        marker="o",
        markersize=5,
        color="tab:blue",
        linestyle="None",
        label="Blue points: representative knots",
    )
    leg2 = ax.legend(handles=[blue_handle], loc="lower left", frameon=True)
    ax.add_artist(leg2)

    info_lines = [
        "Same circular frame as standard",
        "label boxes auto-sized by text",
        "",
        "Comparison mode:",
        "component-wise topological context",
        "boxed labels + explicit connectors",
    ]
    finalize_circular_axes(ax, "Eta Carinae topological circular context", info_lines)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()

def write_proxy_rationale(prepared: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "Eta Carinae topological pipeline proxy rationale",
        "=" * 48,
        "",
        "Purpose",
        "-------",
        "This file records why component-row restructuring was introduced.",
        "The goal is not to fit a preferred answer, but to make the same",
        "component row carry mass + volume + velocity so that a structural",
        "topological quantity can be computed at all.",
        "",
        "Rules applied",
        "-------------",
        "1. homunculus_main_lobes",
        "   - if direct mass is missing, link to Great Eruption lower-bound mass proxy.",
        "2. dust_torus",
        "   - if geometry/depth/velocity are missing, inject conservative torus proxies.",
        "3. outer_ejecta_context -> outer_ejecta_component",
        "   - promote context-only row into a calculable outer-ejecta component row.",
        "",
        "Applied per-row flags",
        "---------------------",
    ]
    subset = prepared[prepared["is_real_component"]].copy()
    for _, row in subset.iterrows():
        lines.append(f"- {row['component_label']}: {row.get('proxy_rule_applied', '')}")
    output_path.write_text("\\n".join(lines), encoding="utf-8")


def write_interpretation_summary(prepared: pd.DataFrame, derivation: dict, output_path: Path) -> None:
    real_df = prepared[prepared["is_real_component"]].copy()
    lines = [
        "Eta Carinae topological interpretation summary",
        "=" * 48,
        "",
        f"d_bg_derived = {derivation['d_bg_derived']:.6f}",
        f"lambda_derived = {derivation['lambda_derived']:.6f}",
        f"weight_sigma_mass_volume = {derivation['weight_sigma_mass_volume']:.6f}",
        f"weight_sigma_velocity = {derivation['weight_sigma_velocity']:.6f}",
        f"weight_sigma_anisotropy = {derivation['weight_sigma_anisotropy']:.6f}",
        "",
        "Component interpretation",
        "------------------------",
    ]
    ordered = real_df.sort_values("sigma_eta", ascending=False)
    for _, row in ordered.iterrows():
        lines.append(
            f"- {row['component_label']}: sigma={row['sigma_eta']:.3f}, "
            f"Dw={row['Dw_trial_mass_volume']:.3f}, role={row['interpretation_role']}"
        )
    output_path.write_text("\\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eta Carinae topological pipeline")
    parser.add_argument("--repo-root", default="", help="Project repository root")
    parser.add_argument("--input-dir", default="", help="Override input directory")
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(script_path)
    input_dir = find_target_input_dir(repo_root, args.input_dir if args.input_dir else None)

    mass_csv = input_dir / "eta_carinae_mass_volume_auxiliary_input_v1.csv"
    source_registry_csv = input_dir / "eta_carinae_source_registry_input_v1.csv"

    stamp = timestamp_label()
    output_dir = repo_root / "results" / "Echo of Light" / "output" / input_dir.name / "topological" / "mass" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_processed_inputs([mass_csv, source_registry_csv], output_dir / "processed_input")

    df = read_csv_required(mass_csv)
    prepared, derivation = prepare_mass_table(df)

    prepared.to_csv(output_dir / "eta_carinae_mass_volume_processed.csv", index=False, encoding="utf-8-sig")
    plot_mass_profile(prepared, output_dir / "eta_carinae_mass_profile.png")
    plot_component_sigma_breakdown(prepared, output_dir / "eta_carinae_sigma_component_breakdown.png")
    plot_mass_velocity_geometry(prepared, output_dir / "eta_carinae_mass_geometry_context.png")
    plot_topological_circular_context(prepared, output_dir / "eta_carinae_topological_circular_context.png")

    calc_df = prepared[prepared["is_real_component"]].copy()
    summary = pd.DataFrame(
        {
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
        }
    )
    summary.to_csv(output_dir / "eta_carinae_mass_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([derivation]).to_csv(output_dir / "eta_carinae_topological_derivation_terms.csv", index=False, encoding="utf-8-sig")

    interpretation_cols = [
        "component_label",
        "proxy_rule_applied",
        "mass_proxy_raw",
        "volume_proxy_raw",
        "sigma_mass_volume",
        "sigma_velocity",
        "sigma_anisotropy",
        "sigma_eta",
        "Dw_trial_mass_volume",
        "interpretation_role",
    ]
    prepared[interpretation_cols].to_csv(output_dir / "eta_carinae_topological_interpretation_table.csv", index=False, encoding="utf-8-sig")
    write_proxy_rationale(prepared, output_dir / "eta_carinae_topological_proxy_rationale.txt")
    write_interpretation_summary(prepared, derivation, output_dir / "eta_carinae_topological_interpretation_summary.txt")

    print(f"topological_output = {output_dir}")


if __name__ == "__main__":
    main()