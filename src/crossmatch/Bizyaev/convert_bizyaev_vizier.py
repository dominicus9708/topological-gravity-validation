
#!/usr/bin/env python3
"""
Convert Bizyaev et al. (2014) VizieR exports into project-ready CSV files.

Project-aware defaults (matching current repository layout):
- input raw files:
  data/raw/Bizyaev/
- output derived files:
  data/derived/Bizyaev/

This script supports both:
1) VizieR XML/VOTable exports that contain a CSV block inside <![CDATA[ ... ]]>
2) Plain TSV / CSV exports

Outputs:
- bizyaev_table4_raw.csv
- bizyaev_table4_rband.csv
- bizyaev_table4_rband_for_crossmatch.csv
- bizyaev_table6_raw.csv
- bizyaev_merged_rband_1d_3d.csv
- bizyaev_conversion_summary.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_whitespace(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else pd.NA


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    a_num = to_numeric(a)
    b_num = to_numeric(b)
    out = a_num / b_num
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------------
# Reading VizieR exports
# ---------------------------------------------------------------------

CSV_BLOCK_RE = re.compile(
    r"<DATA>\s*<CSV[^>]*><!\[CDATA\[(.*?)\]\]>\s*</CSV>\s*</DATA>",
    re.DOTALL | re.IGNORECASE,
)


def read_vizier_xml_csv_block(path: str) -> pd.DataFrame:
    """
    Read VizieR XML/VOTable export with embedded CSV <![CDATA[...]]>.
    The embedded CSV usually has:
      line 1: header
      line 2: units
      line 3: visual separator
      line 4+: data
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    match = CSV_BLOCK_RE.search(text)
    if not match:
        raise ValueError(f"No VizieR embedded CSV block found in: {path}")

    block = match.group(1).strip("\n")
    lines = [line.rstrip("\r") for line in block.splitlines() if line.strip()]

    if len(lines) < 4:
        raise ValueError(f"Embedded CSV block too short in: {path}")

    header = lines[0]
    data_lines = lines[3:]  # skip units and dashed separator

    csv_text = "\n".join([header] + data_lines)
    df = pd.read_csv(io.StringIO(csv_text), sep=";", dtype=str, keep_default_na=False)
    return df


def read_plain_delimited(path: str) -> pd.DataFrame:
    """
    Fallback reader for plain TSV/CSV exports.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".tsv":
        sep = "\t"
    elif ext == ".csv":
        sep = ","
    else:
        sep = None

    if sep is not None:
        return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)

    # best effort fallback
    return pd.read_csv(path, sep=None, engine="python", dtype=str, keep_default_na=False)


def read_vizier_export(path: str) -> pd.DataFrame:
    """
    Auto-detect VizieR XML/VOTable export vs plain TSV/CSV.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        prefix = f.read(4096)

    if "<VOTABLE" in prefix or "<?xml" in prefix:
        return read_vizier_xml_csv_block(path)

    return read_plain_delimited(path)


# ---------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------

CATALOG_PATTERNS = [
    ("NGC",  re.compile(r"\bNGC\s*0*([0-9]{1,5})\b", re.IGNORECASE)),
    ("UGC",  re.compile(r"\bUGC\s*0*([0-9]{1,5})\b", re.IGNORECASE)),
    ("IC",   re.compile(r"\bIC\s*0*([0-9]{1,5})\b", re.IGNORECASE)),
    ("PGC",  re.compile(r"\bPGC\s*0*([0-9]{1,7})\b", re.IGNORECASE)),
    ("UGCA", re.compile(r"\bUGCA\s*0*([0-9]{1,5})\b", re.IGNORECASE)),
    ("ESO",  re.compile(r"\bESO\s*([0-9]{1,4})\s*[- ]?\s*G\s*0*([0-9]{1,4})\b", re.IGNORECASE)),
    ("DDO",  re.compile(r"\bDDO\s*0*([0-9]{1,5})\b", re.IGNORECASE)),
    ("KK",   re.compile(r"\bKK\s*([0-9]{1,4})\s*[- ]?\s*0*([0-9]{1,5})\b", re.IGNORECASE)),
    ("F",    re.compile(r"\bF\s*([0-9]{2,4})\s*[- ]?\s*V?\s*([0-9]{1,3})\b", re.IGNORECASE)),
    ("CAMB", re.compile(r"\bCAMB\b", re.IGNORECASE)),
]


def normalize_catalog_name(text: object) -> Optional[str]:
    if pd.isna(text):
        return None

    s = str(text).strip().upper()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)

    for prefix, pattern in CATALOG_PATTERNS:
        m = pattern.search(s)
        if not m:
            continue

        if prefix in {"NGC", "UGC", "IC", "PGC", "UGCA", "DDO"}:
            return f"{prefix}{int(m.group(1)):05d}" if prefix != "PGC" else f"{prefix}{int(m.group(1))}"
        if prefix == "ESO":
            return f"ESO{int(m.group(1)):03d}-G{int(m.group(2)):03d}"
        if prefix == "KK":
            return f"KK{int(m.group(1))}-{int(m.group(2))}"
        if prefix == "F":
            left = int(m.group(1))
            right = int(m.group(2))
            # preserve the common SPARC-like style as F563-V2 only when source contains V
            has_v = "V" in s
            return f"F{left}-V{right}" if has_v else f"F{left}-{right}"
        if prefix == "CAMB":
            return "CamB"

    return None


def choose_best_alias(row: pd.Series) -> object:
    candidates = [
        row.get("AName"),
        row.get("NED"),
        row.get("Name"),
    ]

    for value in candidates:
        normalized = normalize_catalog_name(value)
        if normalized:
            return normalized

    # If no recognizable catalog alias exists, keep the best available human-readable name
    for value in candidates:
        if pd.notna(value) and str(value).strip():
            return str(value).strip()

    return pd.NA


# ---------------------------------------------------------------------
# Table conversions
# ---------------------------------------------------------------------

TABLE4_NUMERIC_COLUMNS = [
    "RAJ2000", "DEJ2000", "PA", "h", "e_h", "z0", "e_z0",
    "S0", "e_S0", "gradz0", "mag", "B/T", "RV",
]

TABLE6_NUMERIC_COLUMNS = ["h", "z0", "S0"]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.columns:
        out[col] = out[col].map(normalize_whitespace)
    return out


def build_table4_products(df4: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = clean_dataframe(df4)

    for col in TABLE4_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = to_numeric(out[col])

    if "Band" in out.columns:
        out["Band"] = out["Band"].astype("string").str.strip().str.lower()

    out["preferred_name"] = out.apply(choose_best_alias, axis=1)
    out["catalog_name_normalized"] = out["preferred_name"].map(normalize_catalog_name)
    out["z0_over_h"] = safe_divide(out["z0"], out["h"])
    out["h_over_z0"] = safe_divide(out["h"], out["z0"])
    out["thickness_flag"] = pd.cut(
        out["z0_over_h"],
        bins=[-np.inf, 0.10, 0.20, np.inf],
        labels=["thin_like", "intermediate", "thick_like"],
    )

    # r-band subset for direct use
    if "Band" in out.columns:
        rband = out.loc[out["Band"] == "r"].copy()
    else:
        rband = out.copy()

    # crossmatch-oriented compact table
    keep_cols = [
        "Name", "preferred_name", "catalog_name_normalized",
        "RAJ2000", "DEJ2000", "Band",
        "h", "e_h", "z0", "e_z0",
        "z0_over_h", "h_over_z0",
        "S0", "e_S0", "gradz0",
        "mag", "B/T", "Type", "RV",
        "AName", "NED",
        "thickness_flag",
    ]
    keep_cols = [c for c in keep_cols if c in rband.columns]
    crossmatch = rband[keep_cols].copy()

    return out, rband, crossmatch


def build_table6_products(df6: pd.DataFrame) -> pd.DataFrame:
    out = clean_dataframe(df6)
    for col in TABLE6_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = to_numeric(out[col])

    out["z0_over_h"] = safe_divide(out["z0"], out["h"])
    out["h_over_z0"] = safe_divide(out["h"], out["z0"])
    out["thickness_flag"] = pd.cut(
        out["z0_over_h"],
        bins=[-np.inf, 0.10, 0.20, np.inf],
        labels=["thin_like", "intermediate", "thick_like"],
    )
    return out


def merge_table4_table6(table4_r: pd.DataFrame, table6: pd.DataFrame) -> pd.DataFrame:
    left = table4_r.copy()
    right = table6.copy()

    common_key = "Name" if "Name" in left.columns and "Name" in right.columns else None
    if common_key is None:
        return pd.DataFrame()

    merged = left.merge(
        right,
        on=common_key,
        how="left",
        suffixes=("_1d", "_3d"),
    )

    if "h_1d" in merged.columns and "h_3d" in merged.columns:
        merged["h_ratio_3d_to_1d"] = safe_divide(merged["h_3d"], merged["h_1d"])
    if "z0_1d" in merged.columns and "z0_3d" in merged.columns:
        merged["z0_ratio_3d_to_1d"] = safe_divide(merged["z0_3d"], merged["z0_1d"])

    return merged


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Bizyaev VizieR exports into project-ready CSV files."
    )
    parser.add_argument(
        "--table4",
        default=r"data/raw/Bizyaev/J_ApJ_787_24_table4_The_Structural_Parameters_of_True_Edge-on_Galaxies_from the_1-D_Analysis.tsv",
        help="Path to Bizyaev Table 4 raw export.",
    )
    parser.add_argument(
        "--table6",
        default=r"data/raw/Bizyaev/J_ApJ_787_24_table6_The_Structural_Parameters_of_true_Edge_on_Galaxies_in_the_r-band_from_our_3D_Analysis.tsv",
        help="Path to Bizyaev Table 6 raw export.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"data/derived/Bizyaev",
        help="Directory to write converted outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    df4 = read_vizier_export(args.table4)
    df6 = read_vizier_export(args.table6)

    table4_raw, table4_r, table4_cross = build_table4_products(df4)
    table6_raw = build_table6_products(df6)
    merged = merge_table4_table6(table4_r, table6_raw)

    # Output files
    out_table4_raw = os.path.join(args.output_dir, "bizyaev_table4_raw.csv")
    out_table4_r = os.path.join(args.output_dir, "bizyaev_table4_rband.csv")
    out_table4_cross = os.path.join(args.output_dir, "bizyaev_table4_rband_for_crossmatch.csv")
    out_table6_raw = os.path.join(args.output_dir, "bizyaev_table6_raw.csv")
    out_merged = os.path.join(args.output_dir, "bizyaev_merged_rband_1d_3d.csv")
    out_summary = os.path.join(args.output_dir, "bizyaev_conversion_summary.json")

    table4_raw.to_csv(out_table4_raw, index=False, encoding="utf-8-sig")
    table4_r.to_csv(out_table4_r, index=False, encoding="utf-8-sig")
    table4_cross.to_csv(out_table4_cross, index=False, encoding="utf-8-sig")
    table6_raw.to_csv(out_table6_raw, index=False, encoding="utf-8-sig")
    merged.to_csv(out_merged, index=False, encoding="utf-8-sig")

    summary = {
        "input": {
            "table4": args.table4,
            "table6": args.table6,
        },
        "output_dir": args.output_dir,
        "table4_raw_rows": int(len(table4_raw)),
        "table4_rband_rows": int(len(table4_r)),
        "table4_crossmatch_rows": int(len(table4_cross)),
        "table6_rows": int(len(table6_raw)),
        "merged_rows": int(len(merged)),
        "table4_unique_preferred_name": int(table4_raw["preferred_name"].dropna().nunique()) if "preferred_name" in table4_raw.columns else 0,
        "table4_unique_catalog_name_normalized": int(table4_raw["catalog_name_normalized"].dropna().nunique()) if "catalog_name_normalized" in table4_raw.columns else 0,
        "table4_rband_unique_catalog_name_normalized": int(table4_r["catalog_name_normalized"].dropna().nunique()) if "catalog_name_normalized" in table4_r.columns else 0,
        "notes": [
            "Table 4 is the main crossmatch-ready structural table.",
            "Table 6 is kept as a cleaner 3D-analysis companion table.",
            "Use z0_over_h or h_over_z0 as thickness-sensitive structural proxies.",
            "preferred_name is chosen from AName, NED, then Name.",
            "catalog_name_normalized attempts to map aliases into SPARC-like galaxy identifiers.",
        ],
    }

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Bizyaev conversion complete.")
    print(f"  - {out_table4_raw}")
    print(f"  - {out_table4_r}")
    print(f"  - {out_table4_cross}")
    print(f"  - {out_table6_raw}")
    print(f"  - {out_merged}")
    print(f"  - {out_summary}")


if __name__ == "__main__":
    main()
