#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SPARC x Kautsch 2006 crossmatch (uses SPARC index WITH coordinates)

Purpose
-------
Re-run the Kautsch 2006 structural crossmatch now that
`sparc_galaxy_index_with_coords.csv` exists and includes RA/DEC.

Inputs
------
- data/derived/structure/edge_on_candidates/Kautsch_2006/kautsch_2006_prepared.csv
- data/derived/crossmatch/sparc_diskmass_S4G/sparc_galaxy_index_with_coords.csv

Outputs
-------
- data/derived/crossmatch/sparc_kautsch2006/working/sparc_kautsch2006_crossmatch_all.csv
- data/derived/crossmatch/sparc_kautsch2006/working/sparc_kautsch2006_crossmatch_exact.csv
- data/derived/crossmatch/sparc_kautsch2006/working/sparc_kautsch2006_crossmatch_coord_review.csv
- data/derived/crossmatch/sparc_kautsch2006/working/sparc_kautsch2006_crossmatch_matched.csv
- data/derived/crossmatch/sparc_kautsch2006/working/sparc_kautsch2006_crossmatch_unmatched.csv
- data/derived/crossmatch/sparc_kautsch2006/working/sparc_kautsch2006_nearest_neighbors.csv
- data/derived/crossmatch/sparc_kautsch2006/working/sparc_kautsch2006_crossmatch_summary.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
KAUTSCH_FILE = PROJECT_ROOT / "data" / "derived" / "structure" / "edge_on_candidates" / "Kautsch_2006" / "kautsch_2006_prepared.csv"
SPARC_INDEX_FILE = PROJECT_ROOT / "data" / "derived" / "crossmatch" / "sparc_diskmass_S4G" / "sparc_galaxy_index_with_coords.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived" / "crossmatch" / "sparc_kautsch2006" / "working"

STRICT_COORD_SEP_DEG = 0.05
EXACT_COORD_SEP_DEG = 0.15
REVIEW_COORD_SEP_DEG = 0.50


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("_", "")
    return s


def extract_catalog_token(name: object) -> str:
    s = normalize_name(name)
    if not s:
        return ""
    if re.fullmatch(r"J\d{6}(\.\d+)?[+-]\d{6}(\.\d+)?", s):
        return s
    m = re.match(r"^(NGC|UGC|IC|ESO|PGC|DDO|UGCA|F|KK|D)([-A-Z0-9]+)$", s)
    if m:
        prefix, rest = m.groups()
        if rest.isdigit():
            rest = str(int(rest))
        return f"{prefix}{rest}"
    return ""


def to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def angular_sep_deg(ra1_deg: float, dec1_deg: float, ra2_deg: np.ndarray, dec2_deg: np.ndarray) -> np.ndarray:
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)

    sin_d1 = np.sin(dec1)
    cos_d1 = np.cos(dec1)
    sin_d2 = np.sin(dec2)
    cos_d2 = np.cos(dec2)

    cosang = sin_d1 * sin_d2 + cos_d1 * cos_d2 * np.cos(ra1 - ra2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))


def collect_aliases(row: pd.Series, alias_cols: list[str]) -> list[str]:
    vals = []
    for col in alias_cols:
        if col in row.index:
            v = row[col]
            if pd.notna(v) and str(v).strip():
                vals.append(str(v))
    return vals


def main() -> int:
    ensure_dir(OUTPUT_DIR)

    if not KAUTSCH_FILE.exists():
        raise SystemExit(f"Missing Kautsch file: {KAUTSCH_FILE}")
    if not SPARC_INDEX_FILE.exists():
        raise SystemExit(f"Missing SPARC coordinate index: {SPARC_INDEX_FILE}")

    kautsch = pd.read_csv(KAUTSCH_FILE)
    sparc = pd.read_csv(SPARC_INDEX_FILE)

    k_name_col = find_first_existing_column(kautsch, ["preferred_name", "Name", "name", "SDSS", "objID"])
    k_ra_col = find_first_existing_column(kautsch, ["RAJ2000_deg", "RAJ2000", "ra_deg", "RAdeg", "ra"])
    k_dec_col = find_first_existing_column(kautsch, ["DEJ2000_deg", "DEJ2000", "dec_deg", "DEdeg", "dec"])

    if not all([k_name_col, k_ra_col, k_dec_col]):
        raise SystemExit(f"Missing required Kautsch columns. name={k_name_col}, ra={k_ra_col}, dec={k_dec_col}")

    s_name_col = find_first_existing_column(sparc, ["galaxy_id", "Name", "name", "Galaxy", "filename", "id"])
    s_ra_col = find_first_existing_column(sparc, ["RA_deg", "RAJ2000", "_RAJ2000", "ra_deg", "_RA", "RA"])
    s_dec_col = find_first_existing_column(sparc, ["DEC_deg", "DEJ2000", "_DEJ2000", "dec_deg", "_DE", "DE"])

    if not all([s_name_col, s_ra_col, s_dec_col]):
        raise SystemExit(f"Missing required SPARC columns. name={s_name_col}, ra={s_ra_col}, dec={s_dec_col}")

    sparc_alias_cols = []
    for cand in ["Name", "galaxy_id", "name_normalized", "SimbadName", "NEDname", "S4G", "simbad_normalized", "ned_normalized", "s4g_name_normalized"]:
        c = find_first_existing_column(sparc, [cand])
        if c and c not in sparc_alias_cols:
            sparc_alias_cols.append(c)

    kautsch = kautsch.copy()
    kautsch["kautsch_name"] = kautsch[k_name_col].astype(str)
    kautsch["kautsch_name_norm"] = kautsch["kautsch_name"].map(normalize_name)
    kautsch["kautsch_catalog_token"] = kautsch["kautsch_name"].map(extract_catalog_token)
    kautsch["RA_match_deg"] = to_float_series(kautsch[k_ra_col])
    kautsch["DE_match_deg"] = to_float_series(kautsch[k_dec_col])

    sdss_col = find_first_existing_column(kautsch, ["SDSS"])
    if sdss_col:
        kautsch["kautsch_sdss_norm"] = kautsch[sdss_col].astype(str).map(normalize_name)
        kautsch["kautsch_sdss_token"] = kautsch[sdss_col].astype(str).map(extract_catalog_token)
    else:
        kautsch["kautsch_sdss_norm"] = ""
        kautsch["kautsch_sdss_token"] = ""

    sparc = sparc.copy()
    sparc["sparc_name"] = sparc[s_name_col].astype(str)
    sparc["sparc_name_norm"] = sparc["sparc_name"].map(normalize_name)
    sparc["RA_match_deg"] = to_float_series(sparc[s_ra_col])
    sparc["DE_match_deg"] = to_float_series(sparc[s_dec_col])

    sparc_alias_map: dict[str, int] = {}
    sparc_token_map: dict[str, int] = {}

    for idx, row in sparc.iterrows():
        aliases = collect_aliases(row, sparc_alias_cols)
        aliases.append(row["sparc_name"])
        seen = set()
        for val in aliases:
            n = normalize_name(val)
            if n and n not in seen:
                seen.add(n)
                sparc_alias_map.setdefault(n, idx)
            token = extract_catalog_token(val)
            if token:
                sparc_token_map.setdefault(token, idx)

    sparc_ra = sparc["RA_match_deg"].to_numpy(dtype=float)
    sparc_dec = sparc["DE_match_deg"].to_numpy(dtype=float)

    records = []
    nearest_records = []

    for _, row in kautsch.iterrows():
        out = row.to_dict()
        out["matched"] = False
        out["match_reason"] = "unmatched"
        out["matched_sparc_name"] = None
        out["matched_sparc_index"] = None
        out["matched_sep_deg"] = None

        matched_idx = None

        for candidate in [
            row.get("kautsch_name_norm", ""),
            row.get("kautsch_sdss_norm", ""),
            row.get("kautsch_catalog_token", ""),
            row.get("kautsch_sdss_token", ""),
        ]:
            if candidate:
                if candidate in sparc_alias_map:
                    matched_idx = sparc_alias_map[candidate]
                    out["matched"] = True
                    out["match_reason"] = "name_exact"
                    out["matched_sep_deg"] = 0.0
                    break
                if candidate in sparc_token_map:
                    matched_idx = sparc_token_map[candidate]
                    out["matched"] = True
                    out["match_reason"] = "catalog_exact"
                    out["matched_sep_deg"] = 0.0
                    break

        ra = row.get("RA_match_deg")
        dec = row.get("DE_match_deg")
        if pd.notna(ra) and pd.notna(dec):
            valid = np.isfinite(sparc_ra) & np.isfinite(sparc_dec)
            if valid.any():
                seps = angular_sep_deg(float(ra), float(dec), sparc_ra[valid], sparc_dec[valid])
                valid_indices = np.where(valid)[0]
                min_pos = int(np.argmin(seps))
                nearest_sep = float(seps[min_pos])
                nearest_idx = int(valid_indices[min_pos])

                nearest_records.append({
                    "kautsch_name": row.get("kautsch_name"),
                    "kautsch_sdss": row.get("SDSS", None),
                    "RA_match_deg": ra,
                    "DE_match_deg": dec,
                    "nearest_sparc_name": sparc.at[nearest_idx, "sparc_name"],
                    "nearest_sep_deg": nearest_sep,
                })

                if matched_idx is None:
                    if nearest_sep <= STRICT_COORD_SEP_DEG:
                        matched_idx = nearest_idx
                        out["matched"] = True
                        out["match_reason"] = "coord_strict"
                        out["matched_sep_deg"] = nearest_sep
                    elif nearest_sep <= EXACT_COORD_SEP_DEG:
                        matched_idx = nearest_idx
                        out["matched"] = True
                        out["match_reason"] = "coord_exact"
                        out["matched_sep_deg"] = nearest_sep
                    elif nearest_sep <= REVIEW_COORD_SEP_DEG:
                        matched_idx = nearest_idx
                        out["matched"] = True
                        out["match_reason"] = "coord_review"
                        out["matched_sep_deg"] = nearest_sep

        if matched_idx is not None:
            out["matched_sparc_index"] = matched_idx
            out["matched_sparc_name"] = sparc.at[matched_idx, "sparc_name"]

        records.append(out)

    all_df = pd.DataFrame(records)
    nearest_df = pd.DataFrame(nearest_records)

    exact_df = all_df[all_df["match_reason"].isin(["name_exact", "catalog_exact"])].copy()
    coord_review_df = all_df[all_df["match_reason"].isin(["coord_strict", "coord_exact", "coord_review"])].copy()
    matched_df = all_df[all_df["matched"]].copy()
    unmatched_df = all_df[~all_df["matched"]].copy()

    all_path = OUTPUT_DIR / "sparc_kautsch2006_crossmatch_all.csv"
    exact_path = OUTPUT_DIR / "sparc_kautsch2006_crossmatch_exact.csv"
    coord_review_path = OUTPUT_DIR / "sparc_kautsch2006_crossmatch_coord_review.csv"
    matched_path = OUTPUT_DIR / "sparc_kautsch2006_crossmatch_matched.csv"
    unmatched_path = OUTPUT_DIR / "sparc_kautsch2006_crossmatch_unmatched.csv"
    nearest_path = OUTPUT_DIR / "sparc_kautsch2006_nearest_neighbors.csv"
    summary_path = OUTPUT_DIR / "sparc_kautsch2006_crossmatch_summary.json"

    all_df.to_csv(all_path, index=False)
    exact_df.to_csv(exact_path, index=False)
    coord_review_df.to_csv(coord_review_path, index=False)
    matched_df.to_csv(matched_path, index=False)
    unmatched_df.to_csv(unmatched_path, index=False)
    nearest_df.to_csv(nearest_path, index=False)

    summary = {
        "rows_total": int(len(all_df)),
        "rows_matched_any": int(len(matched_df)),
        "rows_unmatched": int(len(unmatched_df)),
        "match_reason_counts": {str(k): int(v) for k, v in all_df["match_reason"].value_counts(dropna=False).to_dict().items()},
        "rows_with_catalog_token": int((all_df["kautsch_catalog_token"].fillna("") != "").sum()),
        "rows_with_coords": int(all_df["RA_match_deg"].notna().sum()),
        "median_axial_ratio_r_matched": (
            float(pd.to_numeric(matched_df.get("axial_ratio_r"), errors="coerce").median())
            if "axial_ratio_r" in matched_df.columns and len(matched_df) > 0 else None
        ),
        "median_edge_likelihood_r_matched": (
            float(pd.to_numeric(matched_df.get("edge_likelihood_r"), errors="coerce").median())
            if "edge_likelihood_r" in matched_df.columns and len(matched_df) > 0 else None
        ),
        "median_sep_deg_matched": (
            float(pd.to_numeric(matched_df.get("matched_sep_deg"), errors="coerce").median())
            if "matched_sep_deg" in matched_df.columns and len(matched_df) > 0 else None
        ),
        "kautsch_file": str(KAUTSCH_FILE),
        "sparc_index_file": str(SPARC_INDEX_FILE),
        "output_dir": str(OUTPUT_DIR),
        "strict_coord_sep_deg": STRICT_COORD_SEP_DEG,
        "max_coord_sep_deg": EXACT_COORD_SEP_DEG,
        "review_coord_sep_deg": REVIEW_COORD_SEP_DEG,
        "detected_sparc_columns": {"name": s_name_col, "ra": s_ra_col, "dec": s_dec_col},
        "detected_kautsch_columns": {"name": k_name_col, "ra": k_ra_col, "dec": k_dec_col},
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] total={len(all_df)} matched={len(matched_df)} unmatched={len(unmatched_df)}")
    print(f"[OK] exact file: {exact_path}")
    print(f"[OK] coord review file: {coord_review_path}")
    print(f"[OK] nearest neighbors file: {nearest_path}")
    print(f"[OK] summary file: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
