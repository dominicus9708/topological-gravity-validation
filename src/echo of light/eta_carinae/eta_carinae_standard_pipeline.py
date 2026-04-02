
from __future__ import annotations

import argparse
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
        specs.append({"component_label":"homunculus_main_lobes","shape":"ellipse","width":minor,"height":major,"angle":0.0,"center":(0.0,0.0),"label_anchor":(0.0,10.9),"label_text":"Homunculus"})
    tor = _first_row(mass_df, "dust_torus")
    if tor is not None:
        specs.append({"component_label":"dust_torus","shape":"circle","radius":2.3,"center":(0.0,0.0),"label_anchor":(4.0,-0.4),"label_text":"dust torus"})
    out = _first_row(mass_df, "outer_ejecta")
    if out is None:
        out = _first_row(mass_df, "outer_ejecta_context")
    if out is not None:
        specs.append({"component_label":"outer_ejecta","shape":"circle","radius":6.2,"center":(0.0,0.0),"label_anchor":(7.6,0.2),"label_text":"outer ejecta","knots":[(4.8,1.4),(-4.1,2.6),(2.4,-4.9),(-5.0,-1.3)]})
    return specs


def add_ring_guides(ax) -> None:
    for r in [2,4,6,8,10]:
        ax.add_patch(Circle((0,0), r, fill=False, linewidth=0.8, linestyle=":", edgecolor="0.55"))
        ax.text(r, 0.18, f'{r}"', fontsize=9, ha="center", va="bottom")


def add_center_system(ax) -> list:
    ax.add_patch(Circle((0,0), radius=0.26, fill=False, linewidth=2.2, edgecolor="black"))
    ax.plot(0,0, marker="+", markersize=10, markeredgewidth=1.8, color="black")
    ax.text(0.0, -0.7, "Eta Car", ha="center", va="top", fontsize=10)
    return [
        Line2D([0],[0], marker='o', markersize=8, markerfacecolor='none', markeredgecolor='black', linestyle='None', label='Central star'),
        Line2D([0],[0], marker='+', markersize=10, color='black', linestyle='None', label='COM / center'),
    ]


def finalize_circular_axes(ax, title: str, info_lines: list[str]) -> None:
    ax.text(
        0.985, 0.035, "\n".join(info_lines),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92, edgecolor="black"),
    )
    ax.set_xlim(-11.5, 11.5)
    ax.set_ylim(-11.5, 11.5)
    ax.set_xlabel("arcsec")
    ax.set_ylabel("arcsec")
    ax.set_title(title)


def draw_label_with_connector(ax, target_xy, text_xy, text, ha="left"):
    ax.annotate(
        text,
        xy=target_xy,
        xytext=text_xy,
        textcoords="data",
        ha=ha,
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="0.4"),
        arrowprops=dict(arrowstyle="-", lw=0.8, color="0.35"),
    )


def prepare_observation_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["effective_temperature_k","spectral_velocity_kmps","bulk_velocity_kmps","fast_component_velocity_kmps","mass_context_msun_lower_bound"]:
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
    out["phase_rank"] = [derive_phase_rank(str(label), str(phase)) for label, phase in zip(out["observation_label"], out["phase_reference"])]
    out["phase_year_reconstructed"] = out.apply(derive_phase_year, axis=1)
    velocity_candidates = []
    for _, row in out.iterrows():
        vals = [parse_numeric_like(row.get("spectral_velocity_kmps")), parse_numeric_like(row.get("bulk_velocity_kmps")), parse_numeric_like(row.get("fast_component_velocity_kmps"))]
        vals = [abs(v) for v in vals if not np.isnan(v)]
        velocity_candidates.append(float(np.mean(vals)) if vals else np.nan)
    out["velocity_magnitude_proxy_kmps"] = velocity_candidates
    return out.sort_values(["phase_year_reconstructed", "phase_rank"]).reset_index(drop=True)


def plot_standard_velocity(df: pd.DataFrame, output_png: Path) -> None:
    plt.figure(figsize=(10,6))
    x = np.arange(len(df))
    labels = df["observation_label"].astype(str).tolist()
    offsets = [-0.2,0.0,0.2]
    width = 0.18
    specs = [("spectral_velocity_kmps","spectral_velocity_kmps"),("bulk_velocity_kmps","bulk_velocity_kmps"),("fast_component_velocity_kmps","fast_component_velocity_kmps")]
    plotted = False
    for (col,label), offset in zip(specs, offsets):
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                plt.bar(x+offset, vals.fillna(0.0), width=width, label=label, alpha=0.8)
                plotted = True
    if not plotted:
        plt.text(0.5,0.5,"No velocity columns available",ha="center",va="center",transform=plt.gca().transAxes)
    plt.axhline(0.0, linewidth=1)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Velocity (km/s)")
    plt.title("Eta Carinae light echo velocity overview")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def plot_standard_temperature_mass(df: pd.DataFrame, output_png: Path) -> None:
    plt.figure(figsize=(10,6))
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
    plt.figure(figsize=(10,6))
    years = pd.to_numeric(df["phase_year_reconstructed"], errors="coerce")
    vel = pd.to_numeric(df["velocity_magnitude_proxy_kmps"], errors="coerce")
    plt.plot(years, vel, marker="o")
    for _, row in df.iterrows():
        plt.annotate(row["observation_label"], (row["phase_year_reconstructed"], row["velocity_magnitude_proxy_kmps"]), textcoords="offset points", xytext=(0,8), ha="center", fontsize=9)
    plt.xlabel("Reconstructed event-phase year")
    plt.ylabel("Velocity magnitude proxy (km/s)")
    plt.title("Eta Carinae reconstructed light echo phase timeline")
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def plot_standard_circular_context(mass_df: pd.DataFrame, output_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9,9))
    ax.set_aspect("equal", adjustable="box")
    add_ring_guides(ax)
    legend_handles = add_center_system(ax)
    specs = get_component_draw_specs(mass_df)
    for spec in specs:
        if spec["shape"] == "ellipse":
            ax.add_patch(Ellipse(spec["center"], width=spec["width"], height=spec["height"], angle=spec["angle"], fill=False, linewidth=2.1, edgecolor="black"))
            draw_label_with_connector(ax, (0.0, spec["height"]/2), spec["label_anchor"], spec["label_text"], ha="center")
            legend_handles.append(Line2D([0],[0], color='black', linewidth=2.1, label='Homunculus lobes'))
        elif spec["shape"] == "circle" and spec["component_label"] == "dust_torus":
            ax.add_patch(Circle(spec["center"], spec["radius"], fill=False, linewidth=1.7, linestyle="--", edgecolor="black"))
            draw_label_with_connector(ax, (spec["radius"], 0.0), spec["label_anchor"], spec["label_text"], ha="left")
            legend_handles.append(Line2D([0],[0], color='black', linestyle='--', linewidth=1.7, label='Dust torus'))
        elif spec["shape"] == "circle" and spec["component_label"] == "outer_ejecta":
            ax.add_patch(Circle(spec["center"], spec["radius"], fill=False, linewidth=1.4, linestyle="-.", edgecolor="black"))
            for x,y in spec.get("knots", []):
                ax.add_patch(Circle((x,y), 0.14, fill=True, alpha=0.75, color="tab:blue"))
            draw_label_with_connector(ax, (spec["radius"], 0.0), spec["label_anchor"], spec["label_text"], ha="left")
            legend_handles.append(Line2D([0],[0], color='black', linestyle='-.', linewidth=1.4, label='Outer ejecta shell'))
            legend_handles.append(Line2D([0],[0], marker='o', markersize=5, color='tab:blue', linestyle='None', label='Representative knots'))
    dedup = {}
    for h in legend_handles:
        dedup[h.get_label()] = h
    ax.legend(handles=list(dedup.values()), loc="upper right", frameon=True)
    info_lines = ['Angular guide rings','2", 4", 6", 8", 10"','','Drawing mode:','standard context schematic','labels moved outside']
    finalize_circular_axes(ax, "Eta Carinae standard circular context", info_lines)
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eta Carinae standard pipeline")
    parser.add_argument("--repo-root", default="", help="Project repository root")
    parser.add_argument("--input-dir", default="", help="Override input directory")
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(script_path)
    input_dir = find_target_input_dir(repo_root, args.input_dir if args.input_dir else None)
    obs_csv = input_dir / "eta_carinae_light_echo_observations_input_v1.csv"
    mass_csv = input_dir / "eta_carinae_mass_volume_auxiliary_input_v1.csv"
    stamp = timestamp_label()
    output_dir = repo_root / "results" / "Echo of Light" / "output" / input_dir.name / "standard" / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_processed_inputs([obs_csv, mass_csv], output_dir / "processed_input")
    df = read_csv_required(obs_csv)
    prepared = prepare_observation_table(df)
    reconstructed = reconstruct_standard_timeseries(prepared)
    prepared.to_csv(output_dir / "eta_carinae_light_echo_observations_processed.csv", index=False, encoding="utf-8-sig")
    reconstructed.to_csv(output_dir / "eta_carinae_standard_timeseries_reconstructed.csv", index=False, encoding="utf-8-sig")
    plot_standard_velocity(prepared, output_dir / "eta_carinae_velocity_overview.png")
    plot_standard_temperature_mass(prepared, output_dir / "eta_carinae_observation_context.png")
    plot_standard_timeseries_reconstructed(reconstructed, output_dir / "eta_carinae_reconstructed_phase_timeline.png")
    if mass_csv.exists():
        plot_standard_circular_context(read_csv_required(mass_csv), output_dir / "eta_carinae_standard_circular_context.png")
    print(f"standard_output = {output_dir}")


if __name__ == "__main__":
    main()
