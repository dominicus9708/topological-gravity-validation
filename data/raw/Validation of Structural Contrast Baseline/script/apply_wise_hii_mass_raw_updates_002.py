#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


SOURCE_COLUMNS = [
    "source_key", "wise_name", "matched_object_name", "paper_title", "authors", "year", "journal",
    "doi", "ads_url", "arxiv_url", "source_type", "mass_field_description", "mass_value_msun",
    "mass_range_lower_msun", "mass_range_upper_msun", "log_nly", "spectral_type", "distance_kpc",
    "matching_notes",
]

EXTRA_COLUMNS = [
    "mass_source_match_method", "radio_proxy_available",
]

TEXTLIKE_CANDIDATE_COLUMNS = [
    "wise_name",
    "mass_search_status",
    "mass_source_key",
    "mass_source_match_method",
    "mass_value_type",
    "mass_value_notes",
    "log_nly",
    "spectral_type",
    "radio_proxy_available",
    "notes",
]

NUMERIC_OPTIONAL_COLUMNS = [
    "mass_value_msun",
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


def cast_textlike_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("object")
    return df


def has_text(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def score_row(row) -> tuple:
    direct_mass = has_text(row.get("mass_value_msun"))
    has_source = has_text(row.get("source_key"))
    has_nly = has_text(row.get("log_nly"))
    has_sp = has_text(row.get("spectral_type"))
    source_type = str(row.get("source_type", "")).strip().lower()
    is_direct = "direct" in source_type
    is_proxy = "proxy" in source_type
    return (
        1 if direct_mass else 0,
        1 if is_direct else 0,
        1 if has_nly else 0,
        1 if has_sp else 0,
        1 if is_proxy else 0,
        1 if has_source else 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    mass_dir = root / "data" / "raw" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "mass"
    evidence_dir = mass_dir / "raw"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = mass_dir / "wise_hii_mass_candidates_initial.csv"
    source_registry_path = mass_dir / "wise_hii_mass_source_registry.csv"

    candidates = read_csv_required(candidates_path)
    source_registry = read_csv_required(source_registry_path)

    candidates["wise_name"] = candidates["wise_name"].astype(str).str.strip()
    source_registry = ensure_columns(source_registry, SOURCE_COLUMNS)

    candidates = ensure_columns(candidates, TEXTLIKE_CANDIDATE_COLUMNS + NUMERIC_OPTIONAL_COLUMNS)
    candidates = cast_textlike_columns(candidates, TEXTLIKE_CANDIDATE_COLUMNS)

    evidence_files = sorted(evidence_dir.glob("*.csv"))
    evidence_tables = []
    for p in evidence_files:
        try:
            df = pd.read_csv(p)
            df = ensure_columns(df, SOURCE_COLUMNS + EXTRA_COLUMNS)
            df["_source_file"] = p.name
            df["wise_name"] = df["wise_name"].astype(str).str.strip()
            evidence_tables.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to read {p.name}: {exc}")

    if evidence_tables:
        evidence = pd.concat(evidence_tables, ignore_index=True, sort=False)
        evidence = evidence[evidence["wise_name"].astype(str).str.strip() != ""].copy()
    else:
        evidence = pd.DataFrame(columns=SOURCE_COLUMNS + EXTRA_COLUMNS + ["_source_file"])

    combined_sources = pd.concat(
        [ensure_columns(source_registry.copy(), SOURCE_COLUMNS), evidence[SOURCE_COLUMNS]],
        ignore_index=True,
        sort=False,
    )
    combined_sources = combined_sources.drop_duplicates(
        subset=["source_key", "wise_name", "paper_title", "doi"], keep="last"
    )
    combined_sources = combined_sources.sort_values(["wise_name", "source_key", "paper_title"], na_position="last")

    working_candidates = candidates.copy()
    working_candidates = ensure_columns(
        working_candidates,
        ["mass_source_key", "mass_source_match_method", "mass_value_msun", "mass_value_type",
         "mass_value_notes", "log_nly", "spectral_type", "radio_proxy_available", "notes"]
    )
    working_candidates = cast_textlike_columns(working_candidates, TEXTLIKE_CANDIDATE_COLUMNS)

    updates = 0
    if not evidence.empty:
        best_rows = []
        for wise_name, group in evidence.groupby("wise_name", dropna=False):
            best = sorted(group.to_dict("records"), key=score_row, reverse=True)[0]
            best_rows.append(best)

        best_df = pd.DataFrame(best_rows)
        best_df["wise_name"] = best_df["wise_name"].astype(str).str.strip()

        for _, row in best_df.iterrows():
            mask = working_candidates["wise_name"].astype(str).str.strip() == str(row["wise_name"]).strip()
            if not mask.any():
                continue
            if has_text(row.get("source_key")):
                working_candidates.loc[mask, "mass_source_key"] = str(row.get("source_key"))
            if has_text(row.get("mass_source_match_method")):
                working_candidates.loc[mask, "mass_source_match_method"] = str(row.get("mass_source_match_method"))
            if has_text(row.get("mass_value_msun")):
                working_candidates.loc[mask, "mass_value_msun"] = row.get("mass_value_msun")
                working_candidates.loc[mask, "mass_value_type"] = "direct_or_literature_value"
            elif has_text(row.get("log_nly")) or has_text(row.get("spectral_type")):
                working_candidates.loc[mask, "mass_value_type"] = "proxy_or_bridge_only"
            if has_text(row.get("mass_field_description")):
                working_candidates.loc[mask, "mass_value_notes"] = str(row.get("mass_field_description"))
            if has_text(row.get("log_nly")):
                working_candidates.loc[mask, "log_nly"] = str(row.get("log_nly"))
            if has_text(row.get("spectral_type")):
                working_candidates.loc[mask, "spectral_type"] = str(row.get("spectral_type"))
            if has_text(row.get("radio_proxy_available")):
                working_candidates.loc[mask, "radio_proxy_available"] = str(row.get("radio_proxy_available"))
            working_candidates.loc[mask, "mass_search_status"] = "source_row_applied"
            working_candidates.loc[mask, "notes"] = "updated_from_mass_raw_evidence"
            updates += int(mask.sum())

    out_source = mass_dir / "wise_hii_mass_source_registry_working.csv"
    out_candidates = mass_dir / "wise_hii_mass_candidates_working.csv"
    out_manifest = mass_dir / "wise_hii_mass_working_manifest.txt"

    combined_sources.to_csv(out_source, index=False)
    working_candidates.to_csv(out_candidates, index=False)

    lines = [
        "WISE H II mass working update",
        "",
        f"Project root: {root}",
        f"Mass dir: {mass_dir}",
        f"Evidence dir: {evidence_dir}",
        f"Evidence files read: {len(evidence_files)}",
        f"Evidence rows read: {len(evidence)}",
        f"Working source rows: {len(combined_sources)}",
        f"Candidate rows updated: {updates}",
        "",
        "Note:",
        "This script does not overwrite initial files.",
        "String-like update columns are explicitly cast to object dtype before assignment.",
        "Use the *_working.csv outputs for parallel progress tracking.",
    ]
    out_manifest.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
