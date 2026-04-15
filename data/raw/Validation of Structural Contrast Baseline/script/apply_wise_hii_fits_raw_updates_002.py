#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


SOURCE_COLUMNS = [
    "source_key", "wise_name", "source_service", "survey_name", "band_name", "ra", "dec",
    "cutout_radius_arcmin", "product_url", "access_notes", "download_status", "local_path",
    "fits_verified", "image_plane_verified", "matching_notes",
]

TEXTLIKE_CANDIDATE_COLUMNS = [
    "wise_name",
    "fits_source_service",
    "fits_band",
    "fits_url",
    "fits_local_path",
    "fits_local_verified",
    "fits_image_plane_verified",
    "fits_check_status",
    "fits_notes",
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


def truthy(v) -> bool:
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "verified", "downloaded", "ok", "ready"}


def score_row(row) -> tuple:
    verified = truthy(row.get("image_plane_verified")) or truthy(row.get("fits_verified"))
    downloaded = str(row.get("download_status", "")).strip().lower() in {"downloaded", "ok", "ready"}
    has_local = has_text(row.get("local_path"))
    has_url = has_text(row.get("product_url"))
    return (
        1 if verified else 0,
        1 if downloaded else 0,
        1 if has_local else 0,
        1 if has_url else 0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    fits_dir = root / "data" / "raw" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "fits"
    evidence_dir = fits_dir / "raw"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = fits_dir / "wise_hii_fits_candidates_initial.csv"
    source_registry_path = fits_dir / "wise_hii_fits_source_registry.csv"

    candidates = read_csv_required(candidates_path)
    source_registry = read_csv_required(source_registry_path)

    candidates["wise_name"] = candidates["wise_name"].astype(str).str.strip()
    source_registry = ensure_columns(source_registry, SOURCE_COLUMNS)
    candidates = ensure_columns(candidates, TEXTLIKE_CANDIDATE_COLUMNS + ["fits_downloadable"])
    candidates = cast_textlike_columns(candidates, TEXTLIKE_CANDIDATE_COLUMNS)

    evidence_files = sorted(evidence_dir.glob("*.csv"))
    evidence_tables = []
    for p in evidence_files:
        try:
            df = pd.read_csv(p)
            df = ensure_columns(df, SOURCE_COLUMNS)
            df["_source_file"] = p.name
            df["wise_name"] = df["wise_name"].astype(str).str.strip()
            evidence_tables.append(df)
        except Exception as exc:
            print(f"[WARN] Failed to read {p.name}: {exc}")

    if evidence_tables:
        evidence = pd.concat(evidence_tables, ignore_index=True, sort=False)
        evidence = evidence[evidence["wise_name"].astype(str).str.strip() != ""].copy()
    else:
        evidence = pd.DataFrame(columns=SOURCE_COLUMNS + ["_source_file"])

    combined_sources = pd.concat(
        [ensure_columns(source_registry.copy(), SOURCE_COLUMNS), evidence[SOURCE_COLUMNS]],
        ignore_index=True,
        sort=False,
    )
    combined_sources = combined_sources.drop_duplicates(
        subset=["source_key", "wise_name", "product_url", "local_path"], keep="last"
    )
    combined_sources = combined_sources.sort_values(["wise_name", "source_key", "product_url"], na_position="last")

    working_candidates = candidates.copy()
    working_candidates = ensure_columns(
        working_candidates,
        ["fits_source_service", "fits_band", "fits_url", "fits_downloadable",
         "fits_local_path", "fits_local_verified", "fits_image_plane_verified", "fits_notes"]
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
            if has_text(row.get("source_service")):
                working_candidates.loc[mask, "fits_source_service"] = str(row.get("source_service"))
            if has_text(row.get("band_name")):
                working_candidates.loc[mask, "fits_band"] = str(row.get("band_name"))
            if has_text(row.get("product_url")):
                working_candidates.loc[mask, "fits_url"] = str(row.get("product_url"))
                working_candidates.loc[mask, "fits_downloadable"] = True
            if has_text(row.get("local_path")):
                working_candidates.loc[mask, "fits_local_path"] = str(row.get("local_path"))
            if has_text(row.get("fits_verified")):
                working_candidates.loc[mask, "fits_local_verified"] = str(row.get("fits_verified"))
            if has_text(row.get("image_plane_verified")):
                working_candidates.loc[mask, "fits_image_plane_verified"] = str(row.get("image_plane_verified"))
            working_candidates.loc[mask, "fits_check_status"] = "source_row_applied"
            working_candidates.loc[mask, "fits_notes"] = "updated_from_fits_raw_evidence"
            updates += int(mask.sum())

    out_source = fits_dir / "wise_hii_fits_source_registry_working.csv"
    out_candidates = fits_dir / "wise_hii_fits_candidates_working.csv"
    out_manifest = fits_dir / "wise_hii_fits_working_manifest.txt"

    combined_sources.to_csv(out_source, index=False)
    working_candidates.to_csv(out_candidates, index=False)

    lines = [
        "WISE H II FITS working update",
        "",
        f"Project root: {root}",
        f"FITS dir: {fits_dir}",
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
