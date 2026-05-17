#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exoplanetary System Structural Distribution Test
Internal discussion test for topological degrees of freedom / structural describability.
This is not an observational proof and not a topological-gravity validation claim.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1.0e-12
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
DOMAIN = "Exoplanetary System Structural Distribution"
QUERY = """
SELECT hostname, pl_name, sy_pnum, pl_orbsmax, pl_bmasse, pl_rade, pl_orbper,
       st_mass, discoverymethod, disc_year
FROM pscomppars
WHERE sy_pnum > 1
  AND pl_orbsmax IS NOT NULL
  AND (pl_bmasse IS NOT NULL OR pl_rade IS NOT NULL)
""".strip()


def ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def make_paths(root: Path) -> dict[str, Path]:
    paths = {
        "raw": root / "data" / "raw" / DOMAIN,
        "input": root / "data" / "derived" / DOMAIN / "input",
        "results": root / "results" / DOMAIN / "output" / ts(),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def download_table(raw_csv: Path) -> pd.DataFrame:
    url = TAP_URL + "?" + urllib.parse.urlencode({"query": QUERY, "format": "csv"})
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    raw_csv.write_bytes(data)
    return pd.read_csv(io.BytesIO(data))


def prepare(df: pd.DataFrame, paths: dict[str, Path]) -> pd.DataFrame:
    work = df.copy()
    for col in ["sy_pnum", "pl_orbsmax", "pl_bmasse", "pl_rade", "pl_orbper", "st_mass", "disc_year"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["mass_proxy"] = work["pl_bmasse"]
    work["mass_proxy_source"] = np.where(work["pl_bmasse"].notna(), "pl_bmasse", "pl_rade_cubic_proxy")
    missing = work["mass_proxy"].isna() & work["pl_rade"].notna()
    work.loc[missing, "mass_proxy"] = np.power(work.loc[missing, "pl_rade"].astype(float), 3.0)
    work = work[(work["hostname"].notna()) & (work["pl_name"].notna())]
    work = work[(work["pl_orbsmax"] > 0) & (work["mass_proxy"] > 0)].copy()
    counts = work.groupby("hostname")["pl_name"].nunique().rename("usable_planets")
    work = work.merge(counts, left_on="hostname", right_index=True, how="left")
    work = work[work["usable_planets"] >= 2].sort_values(["hostname", "pl_orbsmax", "pl_name"])
    work.to_csv(paths["input"] / "exoplanet_structural_input.csv", index=False)
    return work


def alpha_cumulative(a: np.ndarray, m: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    aa, mm = a[order], m[order]
    x = np.log(aa + EPS)
    y = np.log(np.cumsum(mm) + EPS)
    if len(aa) == 2:
        slope = (y[-1] - y[0]) / (x[-1] - x[0] + EPS)
        out_sorted = np.array([slope, slope], dtype=float)
    else:
        out_sorted = np.gradient(y, x, edge_order=1)
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def compute(prepared: pd.DataFrame, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    planet_parts = []
    systems = []
    for host, group in prepared.groupby("hostname", sort=True):
        group = group.copy().sort_values("pl_orbsmax")
        a = group["pl_orbsmax"].to_numpy(float)
        m = group["mass_proxy"].to_numpy(float)
        w = m / (m.sum() + EPS)
        alpha = alpha_cumulative(a, m)
        dw = float(np.sum(w * alpha))
        group["structural_weight"] = w
        group["alpha_obs_cumulative_mass_slope"] = alpha
        group["host_Dw_standard"] = dw
        planet_parts.append(group)
        inner = np.median(a)
        systems.append({
            "hostname": host,
            "n_usable_planets": int(group["pl_name"].nunique()),
            "archive_sy_pnum_max": int(np.nanmax(group["sy_pnum"])),
            "a_min_AU": float(np.min(a)),
            "a_max_AU": float(np.max(a)),
            "log10_a_span": float(np.log10((np.max(a) + EPS) / (np.min(a) + EPS))),
            "mass_proxy_sum_Earth_or_proxy": float(np.sum(m)),
            "mass_concentration_max_weight": float(np.max(w)),
            "inner_mass_fraction_median_split": float(np.sum(m[a <= inner]) / (np.sum(m) + EPS)),
            "S_dist_mass_weight": float(-np.sum(w * np.log(w + EPS))),
            "D_w_standard": dw,
            "direct_mass_fraction": float(np.mean(group["mass_proxy_source"].eq("pl_bmasse"))),
        })
    planets = pd.concat(planet_parts, ignore_index=True) if planet_parts else pd.DataFrame()
    sysdf = pd.DataFrame(systems)
    sysdf["D_bg_by_n"] = sysdf.groupby("n_usable_planets")["D_w_standard"].transform("median")
    sysdf["sigma_system"] = sysdf["D_w_standard"] - sysdf["D_bg_by_n"]
    sysdf["abs_sigma_system"] = sysdf["sigma_system"].abs()
    sysdf["identity_overlap_proxy"] = np.exp(-sysdf["abs_sigma_system"])
    q1, q3 = sysdf["sigma_system"].quantile([0.25, 0.75]).to_list()
    sysdf["structural_class"] = np.where(
        sysdf["sigma_system"] <= q1,
        "compressed / inner-concentrated relative to n-background",
        np.where(sysdf["sigma_system"] >= q3,
                 "extended / outer-distributed relative to n-background",
                 "background-like structural regime")
    )
    planets.to_csv(paths["results"] / "standard_planet_level_working.csv", index=False)
    sysdf.to_csv(paths["results"] / "topological_system_level_working.csv", index=False)
    return planets, sysdf


def plot(sysdf: pd.DataFrame, paths: dict[str, Path]) -> None:
    plt.figure(figsize=(9, 6))
    plt.scatter(sysdf["log10_a_span"], sysdf["mass_concentration_max_weight"], s=30)
    plt.xlabel("log10(a_max / a_min) [AU span]")
    plt.ylabel("maximum structural weight in system")
    plt.title("Standard baseline: orbital span vs mass concentration")
    plt.tight_layout()
    plt.savefig(paths["results"] / "standard_orbital_span_vs_mass_concentration.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    sizes = 20 + 15 * sysdf["n_usable_planets"].clip(upper=10)
    plt.scatter(sysdf["log10_a_span"], sysdf["sigma_system"], s=sizes)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("standard log10 orbital span")
    plt.ylabel("sigma_system = D_w - D_bg(n)")
    plt.title("Internal topological structural map")
    plt.tight_layout()
    plt.savefig(paths["results"] / "topological_structural_map.png", dpi=200)
    plt.close()

    top = sysdf.sort_values("abs_sigma_system", ascending=False).head(20)
    plt.figure(figsize=(10, 7))
    y = np.arange(len(top))
    plt.barh(y, top["sigma_system"])
    plt.yticks(y, top["hostname"])
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("sigma_system")
    plt.title("Representative high-contrast systems for discussion")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(paths["results"] / "representative_high_contrast_systems.png", dpi=200)
    plt.close()


def summary(sysdf: pd.DataFrame, prepared: pd.DataFrame, paths: dict[str, Path], source: str) -> None:
    lines = [
        "Exoplanetary System Structural Distribution Test",
        "================================================",
        "",
        "Status: internal discussion test, not observational proof.",
        f"Data source: {source}",
        f"Usable planet rows: {len(prepared)}",
        f"Usable host systems: {len(sysdf)}",
        "",
        "Core chain:",
        "mass proxy + semi-major axis -> cumulative mass slope alpha_obs -> D_w_standard",
        "D_w_standard - same-n median background -> sigma_system",
        "",
        f"D_w median: {sysdf['D_w_standard'].median():.6g}",
        f"D_w min/max: {sysdf['D_w_standard'].min():.6g} / {sysdf['D_w_standard'].max():.6g}",
        f"sigma min/max: {sysdf['sigma_system'].min():.6g} / {sysdf['sigma_system'].max():.6g}",
        f"median identity_overlap_proxy = exp(-|sigma|): {sysdf['identity_overlap_proxy'].median():.6g}",
        "",
        "Class counts:",
    ]
    for cls, count in sysdf["structural_class"].value_counts().items():
        lines.append(f"- {cls}: {count}")
    lines += ["", "Top representative high-|sigma| systems:"]
    for _, r in sysdf.sort_values("abs_sigma_system", ascending=False).head(10).iterrows():
        lines.append(f"- {r['hostname']}: n={int(r['n_usable_planets'])}, D_w={r['D_w_standard']:.4g}, D_bg={r['D_bg_by_n']:.4g}, sigma={r['sigma_system']:.4g}, class={r['structural_class']}")
    lines += [
        "",
        "Interpretation limits:",
        "- pl_bmasse is preferred, but missing masses are replaced only by a radius-cubic size proxy for internal testing.",
        "- D_w here is a structural descriptor of mass-distance organization, not a spatial dimension.",
        "- sigma_system is a background-relative discussion variable, not a gravitational measurement.",
        "- This test is suitable for internal notes, not for a claim of topological gravity validation.",
    ]
    (paths["results"] / "exoplanet_structural_test_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--local-csv", default=None)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    paths = make_paths(root)
    try:
        if args.local_csv:
            raw = pd.read_csv(args.local_csv)
            source = args.local_csv
        else:
            raw = download_table(paths["raw"] / "nasa_exoplanet_archive_pscomppars_multiplanet.csv")
            source = "NASA Exoplanet Archive PSCompPars via TAP"
        prepared = prepare(raw, paths)
        _, sysdf = compute(prepared, paths)
        plot(sysdf, paths)
        summary(sysdf, prepared, paths, source)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Completed Exoplanetary System Structural Distribution Test")
    print(f"Results: {paths['results']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
