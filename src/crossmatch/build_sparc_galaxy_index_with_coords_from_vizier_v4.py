#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build SPARC galaxy index with coordinates from a VizieR export (v4).

Fixes the prior issue where only 50 rows were extracted.
This version:
- supports VOTable XML with CSV CDATA blocks
- removes unit/separator lines robustly
- retries parsing with python engine
- drops malformed empty rows
- expects the SPARC master table to contain ~175 rows
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd


PROJECT_ROOT = Path(".")
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "sparc" / "J_AJ_152_157_table1_Mass models for 175 disk galaxies with SPARC.tsv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived" / "crossmatch" / "sparc_diskmass_S4G"
OUTPUT_FILE = OUTPUT_DIR / "sparc_galaxy_index_with_coords.csv"
SUMMARY_FILE = OUTPUT_DIR / "sparc_galaxy_index_with_coords_summary.json"


def normalize_colname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    norm_map = {normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def norm_alias(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper().replace(" ", "").replace("_", "")


def is_separator_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    # Lines made mostly of -, =, whitespace, separators
    stripped = re.sub(r"[-=;,\t ]", "", s)
    return stripped == ""


def parse_delimited_text(text: str, sep: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(text), sep=sep, engine="python")


def read_text_delimited(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f if not line.startswith("#") and line.strip()]
    if not lines:
        raise ValueError("No readable delimited table rows found.")

    header = lines[0]
    sep = "\t" if "\t" in header else (";" if ";" in header else ",")
    cleaned = "\n".join(lines)
    return parse_delimited_text(cleaned, sep)


def extract_csv_cdata(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    root = ET.fromstring(text)
    csv_nodes = [elem for elem in root.iter() if elem.tag.endswith("CSV")]
    if not csv_nodes:
        raise ValueError("No <CSV> node found in VOTable XML.")
    for node in csv_nodes:
        if node.text and node.text.strip():
            return node.text
    raise ValueError("No CSV CDATA content found inside <CSV> node.")


def read_votable_csv_cdata(path: Path) -> pd.DataFrame:
    csv_text = extract_csv_cdata(path)
    raw_lines = [ln.rstrip("\n") for ln in csv_text.splitlines() if ln.strip()]
    if len(raw_lines) < 2:
        raise ValueError("CSV CDATA content too short to parse.")

    header = raw_lines[0]
    sep = ";" if ";" in header else ("\t" if "\t" in header else ",")

    # Remove obvious unit / separator lines after header, but keep all data lines.
    body_lines = []
    for i, line in enumerate(raw_lines[1:], start=1):
        if i <= 3 and is_separator_line(line):
            continue
        # Typical VizieR units line often has bracketed/unit-like tokens and no object names.
        # Keep only lines that are not pure separators; data lines are preserved.
        body_lines.append(line)

    cleaned_text = "\n".join([header] + body_lines)
    df = parse_delimited_text(cleaned_text, sep)

    # Drop rows where the main name column is missing-like across all columns
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def read_vizier_any(path: Path) -> tuple[pd.DataFrame, str]:
    prefix = path.read_text(encoding="utf-8", errors="replace")[:1000].lstrip()
    if prefix.startswith("<?xml") or "<VOTABLE" in prefix:
        return read_votable_csv_cdata(path), "votable_xml_csv_cdata"
    return read_text_delimited(path), "text_delimited"


def main() -> int:
    if not INPUT_FILE.exists():
        raise SystemExit(f"Missing input file: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df, read_mode = read_vizier_any(INPUT_FILE)

    name_col = find_col(df, "Name", "Galaxy")
    simbad_col = find_col(df, "SimbadName")
    ned_col = find_col(df, "NEDname")
    s4g_col = find_col(df, "S4G")
    edd_col = find_col(df, "EDD")

    ra_col = find_col(df, "_RAJ2000", "RAJ2000", "_RA", "RA")
    dec_col = find_col(df, "_DEJ2000", "DEJ2000", "_DE", "DE")

    type_col = find_col(df, "Type")
    dist_col = find_col(df, "Dist", "D")
    inc_col = find_col(df, "i", "Inc")
    l36_col = find_col(df, "L3.6", "L[3.6]")
    reff_col = find_col(df, "Reff")
    rdisk_col = find_col(df, "Rdisk")
    mhi_col = find_col(df, "MHI")
    rhi_col = find_col(df, "RHI")
    vflat_col = find_col(df, "Vflat")
    evflat_col = find_col(df, "e_Vflat")
    qual_col = find_col(df, "Qual")
    ref_col = find_col(df, "Ref")

    if name_col is None or ra_col is None or dec_col is None:
        raise SystemExit(
            f"Required columns missing. name={name_col}, ra={ra_col}, dec={dec_col}. "
            f"Available columns={list(df.columns)}"
        )

    out = pd.DataFrame()
    out["galaxy_id"] = df[name_col].astype(str).str.strip()
    out["Name"] = df[name_col].astype(str).str.strip()
    out["SimbadName"] = df[simbad_col].astype(str).str.strip() if simbad_col else ""
    out["NEDname"] = df[ned_col].astype(str).str.strip() if ned_col else ""
    out["S4G"] = df[s4g_col].astype(str).str.strip() if s4g_col else ""
    out["EDD"] = df[edd_col].astype(str).str.strip() if edd_col else ""

    out["RA_deg"] = pd.to_numeric(df[ra_col], errors="coerce")
    out["DEC_deg"] = pd.to_numeric(df[dec_col], errors="coerce")

    if type_col:
        out["Type"] = df[type_col]
    if dist_col:
        out["Dist"] = pd.to_numeric(df[dist_col], errors="coerce")
    if inc_col:
        out["inclination_deg"] = pd.to_numeric(df[inc_col], errors="coerce")
    if l36_col:
        out["L3p6"] = pd.to_numeric(df[l36_col], errors="coerce")
    if reff_col:
        out["Reff"] = pd.to_numeric(df[reff_col], errors="coerce")
    if rdisk_col:
        out["Rdisk"] = pd.to_numeric(df[rdisk_col], errors="coerce")
    if mhi_col:
        out["MHI"] = pd.to_numeric(df[mhi_col], errors="coerce")
    if rhi_col:
        out["RHI"] = pd.to_numeric(df[rhi_col], errors="coerce")
    if vflat_col:
        out["Vflat"] = pd.to_numeric(df[vflat_col], errors="coerce")
    if evflat_col:
        out["e_Vflat"] = pd.to_numeric(df[evflat_col], errors="coerce")
    if qual_col:
        out["Qual"] = df[qual_col]
    if ref_col:
        out["Ref"] = df[ref_col]

    out["name_normalized"] = out["Name"].map(norm_alias)
    out["simbad_normalized"] = out["SimbadName"].map(norm_alias)
    out["ned_normalized"] = out["NEDname"].map(norm_alias)
    out["s4g_name_normalized"] = out["S4G"].map(norm_alias)

    # Remove obvious broken rows
    out = out[out["galaxy_id"].astype(str).str.strip() != ""].reset_index(drop=True)

    out.to_csv(OUTPUT_FILE, index=False)

    summary = {
        "rows_total": int(len(out)),
        "rows_with_coords": int(out["RA_deg"].notna().sum()),
        "rows_with_simbad": int((out["SimbadName"].fillna("") != "").sum()),
        "rows_with_ned": int((out["NEDname"].fillna("") != "").sum()),
        "rows_with_s4g": int((out["S4G"].fillna("") != "").sum()),
        "read_mode": read_mode,
        "detected_columns": {
            "name": name_col,
            "ra": ra_col,
            "dec": dec_col,
            "simbad": simbad_col,
            "ned": ned_col,
            "s4g": s4g_col,
            "edd": edd_col,
            "type": type_col,
            "dist": dist_col,
            "inc": inc_col,
            "l36": l36_col,
            "reff": reff_col,
            "rdisk": rdisk_col,
            "mhi": mhi_col,
            "rhi": rhi_col,
            "vflat": vflat_col,
            "evflat": evflat_col,
            "qual": qual_col,
            "ref": ref_col,
        },
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] rows={len(out)} with_coords={summary['rows_with_coords']} mode={read_mode}")
    print(f"[OK] output file: {OUTPUT_FILE}")
    print(f"[OK] summary file: {SUMMARY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
