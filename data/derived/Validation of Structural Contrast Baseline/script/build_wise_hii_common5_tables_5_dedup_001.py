#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a de-duplicated common comparison shortlist of 5 independent regions
for fair standard/topological comparison.

Version intent
--------------
This script is the next step after tables_4 ranked candidate pooling.
It creates a tables_5-style shortlist with region/membership de-duplication.

Placement
---------
data/derived/Validation of Structural Contrast Baseline/script/

Inputs
------
Primary:
- data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/tables_4/wise_hii_tables_4_ranked_pool.csv

Fallback:
- data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/tables_4/wise_hii_tables_4_common5_shortlist.csv

Outputs
-------
data/derived/Validation of Structural Contrast Baseline/wise_hii_catalog/tables_5/

Purpose
-------
- remove repeated rows from the same practical region/group
- keep the strongest representative row for each independent region-like bucket
- produce a fairer 5-target common comparison shortlist

Important
---------
This output is still NOT final input.
It is a fair-comparison shortlist for accelerated source gathering.

Windows example
---------------
python "data\\derived\\Validation of Structural Contrast Baseline\\script\\build_wise_hii_common5_tables_5_dedup_001.py"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


def read_csv_optional(path: Path):
    if path.exists():
        return pd.read_csv(path), str(path)
    return None, None


def read_primary_or_fallback(primary: Path, fallback: Path):
    if primary.exists():
        return pd.read_csv(primary), str(primary)
    if fallback.exists():
        return pd.read_csv(fallback), str(fallback)
    raise SystemExit(f"Missing both primary and fallback inputs:\n- {primary}\n- {fallback}")


def has_text(v) -> bool:
    return pd.notna(v) and str(v).strip() != ""


def normalize_text(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r'[\[\]\(\)\{\};:,/\\]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def build_region_bucket(row) -> str:
    region = row.get("hii_region_name", "")
    membership = row.get("membership", "")
    wise_name = row.get("wise_name", "")

    region_n = normalize_text(region)
    membership_n = normalize_text(membership)

    # Prefer explicit region name when available
    if region_n:
        return f"region::{region_n}"

    # Otherwise use the first meaningful membership token
    if membership_n:
        tokens = [t.strip() for t in membership_n.split() if t.strip()]
        joined = " ".join(tokens[:4]).strip()
        if joined:
            return f"membership::{joined}"

    return f"wise::{normalize_text(wise_name)}"


def normalize_priority(v):
    try:
        return float(v)
    except Exception:
        return 999999.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of topological_gravity_project")
    parser.add_argument("--top-k", default=5, type=int, help="Number of independent region candidates to select")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    in_dir = root / "data" / "derived" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "tables_4"
    out_dir = root / "data" / "derived" / "Validation of Structural Contrast Baseline" / "wise_hii_catalog" / "tables_5"
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_path = in_dir / "wise_hii_tables_4_ranked_pool.csv"
    fallback_path = in_dir / "wise_hii_tables_4_common5_shortlist.csv"

    df, used_path = read_primary_or_fallback(ranked_path, fallback_path)
    df["wise_name"] = df["wise_name"].astype(str).str.strip()

    if "score_total" not in df.columns:
        raise SystemExit("Input ranked pool does not contain score_total. Use tables_4 ranked pool as input.")

    df["priority_rank_numeric"] = df["priority_rank_numeric"] if "priority_rank_numeric" in df.columns else df["priority_rank"].apply(normalize_priority)
    df["region_bucket"] = df.apply(build_region_bucket, axis=1)

    # sort by strongest candidate first
    df = df.sort_values(
        ["score_total", "score_fits_ready", "score_proxy_ready", "score_mass_ready", "flag_known_group", "flag_has_region_name", "flag_has_membership", "priority_rank_numeric"],
        ascending=[False, False, False, False, False, False, False, True]
    ).reset_index(drop=True)

    # keep only best row per region bucket
    dedup = df.drop_duplicates(subset=["region_bucket"], keep="first").copy().reset_index(drop=True)

    dedup["tables_5_role"] = "deduplicated_ranked_pool"
    dedup["common_comparison_shortlist"] = False
    dedup.loc[dedup.index < int(args.top_k), "common_comparison_shortlist"] = True

    shortlist = dedup[dedup["common_comparison_shortlist"]].copy()

    # region coverage table
    coverage = df.groupby("region_bucket", dropna=False).agg(
        representative_wise_name=("wise_name", "first"),
        representative_region_name=("hii_region_name", "first"),
        representative_membership=("membership", "first"),
        rows_in_bucket=("wise_name", "count"),
        max_score_total=("score_total", "max"),
    ).reset_index()

    # Save outputs
    df.to_csv(out_dir / "wise_hii_tables_5_input_ranked_with_buckets.csv", index=False)
    dedup.to_csv(out_dir / "wise_hii_tables_5_deduplicated_pool.csv", index=False)
    shortlist.to_csv(out_dir / "wise_hii_tables_5_common5_shortlist.csv", index=False)
    coverage.to_csv(out_dir / "wise_hii_tables_5_region_bucket_coverage.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "input_rows", "value": int(len(df))},
        {"metric": "distinct_region_buckets", "value": int(dedup["region_bucket"].nunique())},
        {"metric": "common_comparison_shortlist", "value": int(len(shortlist))},
    ])
    summary.to_csv(out_dir / "wise_hii_tables_5_summary.csv", index=False)

    manifest_lines = [
        "WISE H II tables_5 fair common-comparison shortlist",
        "",
        f"Project root: {root}",
        f"Input source used: {used_path}",
        f"Output dir: {out_dir}",
        "",
        "Purpose:",
        "Convert row-level top ranking into region-level fair comparison ranking.",
        "Only one representative row is kept for each practical region/membership bucket.",
        "",
        f"Input rows: {len(df)}",
        f"Distinct region buckets: {dedup['region_bucket'].nunique()}",
        f"Shortlist size: {len(shortlist)}",
        "",
        "Operational note:",
        "Use wise_hii_tables_5_common5_shortlist.csv as the accelerated fair-comparison target set.",
        "This output is still NOT final input.",
    ]
    (out_dir / "wise_hii_tables_5_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")

    print("\n".join(manifest_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
