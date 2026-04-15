#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply dual mass source registry into dual mass candidates working table.

Placement
---------
data/raw/Validation of Structural Contrast Baseline/script/

Inputs
------
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_dual_candidates_initial.csv
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_dual_source_registry.csv

Outputs
-------
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_dual_candidates_working.csv
- data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/wise_hii_mass_dual_working_manifest.txt

Purpose
-------
- Read the dual source registry
- Choose the best available source row for each wise_name
- Update dual candidates working table
- Preserve initial file unchanged

Selection preference
--------------------
1. direct mass with source key
2. proxy with numeric proxy_value and source key
3. log_nly source
4. spectral_type source
5. radio_continuum source
6. ionizing_source reference only

Windows example
---------------
python "data\raw\Validation of Structural Contrast Baseline\script\apply_wise_hii_mass_dual_updates_001.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


TEXTLIKE_COLUMNS = [
    "wise_name",
    "mass_source_key",
    "mass_value_type",
    "mass_value_notes",
    "proxy_kind",
    "proxy_value_unit",
    "log_nly",
    "spectral_type",
    "radio_proxy_available",
    "ionizing_source_reference",
    "mass_search_status",
    "notes",
]

NUMERIC_OPTIONAL_COLUMNS = [
    "mass_value_msun",
    "proxy_value",
]

REQUIRED_SOURCE_COLUMNS = [
    "source_key",
    "wise_name",
    "matched_object_name",
    "paper_title",
    "authors",
    "year",
    "journal",
    "doi",
    "ads_url",
    "arxiv_url",
    "source_type",
    "mass_field_description",
    "mass_value_msun",
    "mass_range_lower_msun",
    "mass_range_upper_msun",
    "log_nly",
    "spectral_type",
    "distance_kpc",
    "matching_notes",
    "mass_source_match_method",
    "radio_proxy_available",
    "proxy_kind",
    "proxy_value",
    "proxy_value_unit",
    "ionizing_source_reference",
]


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required input: {path}")
    return pd.read_csv(path)


def ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def cast_textlike(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("object")
    return df


def has_text(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def has_numeric(v) -> bool:
    try:
        return pd.notna(v) and str(v).strip() != ""
    except Exception:
        return False


def classify_source_row(row) -> str:
    source_type = str(row.get("source_type", "")).strip().lower()
    has_direct_mass = has_numeric(row.get("mass_value_msun"))
    has_proxy_value = has_numeric(row.get("proxy_value"))
    has_log_nly = has_text(row.get("log_nly"))
    has_spt = has_text(row.get("spectral_type"))
    has_radio = has_text(row.get("radio_proxy_available"))
    has_ionizing = has_text(row.get("ionizing_source_reference"))
    has_source = has_text(row.get("source_key"))

    if has_direct_mass and has_source:
        return "direct_mass"
    if has_proxy_value and has_source:
        return "proxy_numeric"
    if ("log_nly" in source_type or has_log_nly) and has_source:
        return "proxy_log_nly"
    if ("spectral" in source_type or has_spt) and has_source:
        return "proxy_spectral_type"
    if ("radio" in source_type or has_radio) and has_source:
        return "proxy_radio"
    if has_ionizing and has_source:
        return "proxy_ionizing_source"
    if has_source:
        return "source_only"
    return "unclassified"


def score_source_row(row) -> tuple:
    category = classify_source_row(row)
    order = {
        "direct_mass": 6,
        "proxy_numeric": 5,
        "proxy_log_nly": 4,
        "proxy_spectral_type": 3,
        "proxy_radio": 2,
        "proxy_ionizing_source": 1,
        "source_only": 0,
        "unclassified": -1,
    }
    source_type = str(row.get("source_type", "")).strip().lower()
    direct_hint = 1 if "direct" in source_type else 0
    proxy_hint = 1 if "proxy" in source_type else 0
    has_doi = 1 if has_text(row.get("doi")) else 0
    has_ads = 1 if has_text(row.get("ads_url")) else 0
    return (
        order.get(category, -1),
        direct_hint,
        proxy_hint,
        has_doi,
        has_ads,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    mass_dir = root / "data" / "raw" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "mass"

    candidates_path = mass_dir / "wise_hii_mass_dual_candidates_initial.csv"
    source_registry_path = mass_dir / "wise_hii_mass_dual_source_registry.csv"

    candidates = read_csv_required(candidates_path)
    source_registry = read_csv_required(source_registry_path)

    candidates = ensure_columns(candidates, TEXTLIKE_COLUMNS + NUMERIC_OPTIONAL_COLUMNS)
    candidates = cast_textlike(candidates, TEXTLIKE_COLUMNS)
    source_registry = ensure_columns(source_registry, REQUIRED_SOURCE_COLUMNS)

    candidates["wise_name"] = candidates["wise_name"].astype(str).str.strip()
    source_registry["wise_name"] = source_registry["wise_name"].astype(str).str.strip()

    working = candidates.copy()

    updates = 0
    chosen_categories = []

    usable_sources = source_registry[source_registry["wise_name"].astype(str).str.strip() != ""].copy()

    if not usable_sources.empty:
        best_rows = []
        for wise_name, group in usable_sources.groupby("wise_name", dropna=False):
            best = sorted(group.to_dict("records"), key=score_source_row, reverse=True)[0]
            best_rows.append(best)

        best_df = pd.DataFrame(best_rows)
        best_df["wise_name"] = best_df["wise_name"].astype(str).str.strip()

        for _, row in best_df.iterrows():
            mask = working["wise_name"].astype(str).str.strip() == str(row["wise_name"]).strip()
            if not mask.any():
                continue

            category = classify_source_row(row)
            chosen_categories.append(category)

            if has_text(row.get("source_key")):
                working.loc[mask, "mass_source_key"] = str(row.get("source_key"))

            if has_numeric(row.get("mass_value_msun")):
                working.loc[mask, "mass_value_msun"] = row.get("mass_value_msun")
                working.loc[mask, "mass_value_type"] = "direct_or_literature_value"

            if has_text(row.get("mass_field_description")):
                working.loc[mask, "mass_value_notes"] = str(row.get("mass_field_description"))

            if has_text(row.get("proxy_kind")):
                working.loc[mask, "proxy_kind"] = str(row.get("proxy_kind"))
            else:
                working.loc[mask, "proxy_kind"] = category

            if has_numeric(row.get("proxy_value")):
                working.loc[mask, "proxy_value"] = row.get("proxy_value")
            if has_text(row.get("proxy_value_unit")):
                working.loc[mask, "proxy_value_unit"] = str(row.get("proxy_value_unit"))

            if has_text(row.get("log_nly")):
                working.loc[mask, "log_nly"] = str(row.get("log_nly"))
            if has_text(row.get("spectral_type")):
                working.loc[mask, "spectral_type"] = str(row.get("spectral_type"))
            if has_text(row.get("radio_proxy_available")):
                working.loc[mask, "radio_proxy_available"] = str(row.get("radio_proxy_available"))
            if has_text(row.get("ionizing_source_reference")):
                working.loc[mask, "ionizing_source_reference"] = str(row.get("ionizing_source_reference"))

            working.loc[mask, "mass_search_status"] = "dual_source_row_applied"
            working.loc[mask, "notes"] = f"updated_from_dual_source_registry:{category}"
            updates += int(mask.sum())

    out_candidates = mass_dir / "wise_hii_mass_dual_candidates_working.csv"
    out_manifest = mass_dir / "wise_hii_mass_dual_working_manifest.txt"

    working.to_csv(out_candidates, index=False)

    cat_counts = pd.Series(chosen_categories).value_counts().to_dict() if chosen_categories else {}

    lines = [
        "WISE H II dual mass working update",
        "",
        f"Project root: {root}",
        f"Mass dir: {mass_dir}",
        f"Dual source rows read: {len(usable_sources)}",
        f"Candidate rows updated: {updates}",
        f"Output working file: {out_candidates}",
        "",
        "Chosen source categories:",
    ]
    if cat_counts:
        for k, v in cat_counts.items():
            lines.append(f"- {k}: {v}")
    else:
        lines.append("- none")
    lines += [
        "",
        "Note:",
        "This script does not overwrite the initial candidates file.",
        "Best-row selection prefers direct mass, then numeric proxy, then log_nly/spectral/radio/ionizing references.",
    ]

    out_manifest.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
