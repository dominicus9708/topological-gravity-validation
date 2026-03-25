from __future__ import annotations

import csv
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------
# Edge-on candidate catalog conversion
# ---------------------------------------------------------------------
# Converts the newly downloaded VizieR XML/CSV hybrid files into CSV.
# Designed for the current project layout:
# - raw inputs under data/external_catalogs/edge_on_candidates/
# - outputs under data/derived/structure/edge_on_candidates/
# ---------------------------------------------------------------------

VOTABLE_NS = {"v": "http://www.ivoa.net/xml/VOTable/v1.2"}


@dataclass
class CatalogSpec:
    key: str
    label: str
    raw_dir: Path
    output_dir: Path
    filename_pattern: str


def normalize_catalog_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    if not text or text.lower() == "nan":
        return ""
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "")
    text = text.replace("/", "")
    text = text.replace("-", "")
    return text


def clean_string(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def to_float_or_nan(value: object) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if not text or text in {"---", "--", "-", "..."}:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def extract_votable_csv_block(votable_path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    tree = ET.parse(votable_path)
    root = tree.getroot()

    resource = root.find(".//v:RESOURCE", VOTABLE_NS)
    table = root.find(".//v:TABLE", VOTABLE_NS)
    csv_node = root.find(".//v:TABLE/v:DATA/v:CSV", VOTABLE_NS)

    if resource is None or table is None or csv_node is None or not csv_node.text:
        raise ValueError(f"Could not locate VOTable CSV block in: {votable_path}")

    csv_text = csv_node.text.strip("\n")
    if not csv_text.strip():
        raise ValueError(f"CSV block is empty in: {votable_path}")

    rows = list(csv.reader(io.StringIO(csv_text), delimiter=";"))
    if len(rows) < 4:
        raise ValueError(f"Unexpected CSV block structure in: {votable_path}")

    header = [clean_string(x) for x in rows[0]]
    units = [clean_string(x) for x in rows[1]]
    separators = rows[2]
    data_rows = rows[3:]

    # Drop empty trailing rows.
    while data_rows and not any(clean_string(cell) for cell in data_rows[-1]):
        data_rows.pop()

    df = pd.DataFrame(data_rows, columns=header)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(clean_string)

    metadata = {
        "resource_name": resource.get("name", ""),
        "table_name": table.get("name", ""),
        "description": clean_string((table.find("v:DESCRIPTION", VOTABLE_NS).text if table.find("v:DESCRIPTION", VOTABLE_NS) is not None else "")),
        "units": dict(zip(header, units)),
        "separator_row": separators,
        "row_count": len(df),
        "column_count": len(df.columns),
    }
    return df, metadata


def prepare_bizyaev_2002(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = ["_Glon", "_Glat", "_RAJ2000", "_DEJ2000", "Dist", "Re", "S0", "Z0", "_RA", "_DE"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].map(to_float_or_nan)

    if "RFGC" in out.columns:
        out["RFGC_int"] = pd.to_numeric(out["RFGC"], errors="coerce").astype("Int64")
        out["preferred_name"] = out["RFGC_int"].map(lambda x: f"RFGC{int(x):04d}" if pd.notna(x) else "")
    else:
        out["RFGC_int"] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"))
        out["preferred_name"] = ""

    out["catalog_name_normalized"] = out["preferred_name"].map(normalize_catalog_name)
    out["RAJ2000_deg"] = out.get("_RAJ2000", pd.Series([math.nan] * len(out)))
    out["DEJ2000_deg"] = out.get("_DEJ2000", pd.Series([math.nan] * len(out)))
    out["z0_over_re"] = out.apply(
        lambda row: row["Z0"] / row["Re"] if pd.notna(row.get("Z0")) and pd.notna(row.get("Re")) and row.get("Re") not in (0, 0.0) else math.nan,
        axis=1,
    )
    out["has_ned_link_flag"] = out.get("NED", pd.Series([""] * len(out))).astype(str).str.upper().eq("NED")
    out["has_simbad_link_flag"] = out.get("Simbad", pd.Series([""] * len(out))).astype(str).str.upper().eq("SIMBAD")
    out["source_catalog"] = "Bizyaev 2002"

    preferred_cols = [
        "preferred_name",
        "catalog_name_normalized",
        "RFGC_int",
        "Dist",
        "Re",
        "Z0",
        "z0_over_re",
        "S0",
        "RAJ2000_deg",
        "DEJ2000_deg",
        "Note",
        "has_ned_link_flag",
        "has_simbad_link_flag",
        "source_catalog",
    ]
    other_cols = [c for c in out.columns if c not in preferred_cols]
    return out[preferred_cols + other_cols]


def prepare_kautsch_2006(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "_Glon", "_Glat", "_RAJ2000", "_DEJ2000", "gmag", "e_gmag", "rmag", "e_rmag",
        "imag", "e_imag", "gmue", "rmue", "imue", "z", "e_z", "el(g)", "el(r)", "el(i)",
        "CI(g)", "CI(r)", "CI(i)", "Diam(g)", "Diam(r)", "Diam(i)", "a/b(g)", "a/b(r)", "a/b(i)",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = out[col].map(to_float_or_nan)

    out["preferred_name"] = out.get("SDSS", pd.Series([""] * len(out))).map(clean_string)
    out["catalog_name_normalized"] = out["preferred_name"].map(normalize_catalog_name)
    out["RAJ2000_deg"] = out.get("_RAJ2000", pd.Series([math.nan] * len(out)))
    out["DEJ2000_deg"] = out.get("_DEJ2000", pd.Series([math.nan] * len(out)))
    out["axial_ratio_r"] = out.get("a/b(r)", pd.Series([math.nan] * len(out)))
    out["major_diam_r_arcsec"] = out.get("Diam(r)", pd.Series([math.nan] * len(out)))
    out["edge_likelihood_r"] = out.get("el(r)", pd.Series([math.nan] * len(out)))
    out["source_catalog"] = "Kautsch 2006"

    preferred_cols = [
        "preferred_name",
        "catalog_name_normalized",
        "MType",
        "gmag",
        "rmag",
        "imag",
        "gmue",
        "rmue",
        "imue",
        "z",
        "e_z",
        "RAJ2000_deg",
        "DEJ2000_deg",
        "RAJ2000",
        "DEJ2000",
        "axial_ratio_r",
        "major_diam_r_arcsec",
        "edge_likelihood_r",
        "source_catalog",
    ]
    other_cols = [c for c in out.columns if c not in preferred_cols]
    return out[preferred_cols + other_cols]


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def build_catalog_specs(project_root: Path) -> List[CatalogSpec]:
    base_raw = project_root / "data" / "external_catalogs" / "edge_on_candidates"
    base_out = project_root / "data" / "derived" / "structure" / "edge_on_candidates"
    return [
        CatalogSpec(
            key="bizyaev_2002",
            label="Bizyaev 2002",
            raw_dir=base_raw / "Bizyaev 2002",
            output_dir=base_out / "Bizyaev_2002",
            filename_pattern="*.tsv",
        ),
        CatalogSpec(
            key="kautsch_2006",
            label="Kautsch 2006",
            raw_dir=base_raw / "Kautsch 2006",
            output_dir=base_out / "Kautsch_2006",
            filename_pattern="*.tsv",
        ),
    ]


def convert_one_catalog(spec: CatalogSpec) -> Dict[str, object]:
    matches = sorted(spec.raw_dir.glob(spec.filename_pattern))
    if not matches:
        raise FileNotFoundError(f"No input file found in {spec.raw_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one TSV in {spec.raw_dir}, found {len(matches)}")

    input_path = matches[0]
    raw_df, metadata = extract_votable_csv_block(input_path)

    if spec.key == "bizyaev_2002":
        prepared_df = prepare_bizyaev_2002(raw_df)
        raw_name = "bizyaev_2002_raw.csv"
        prepared_name = "bizyaev_2002_prepared.csv"
    elif spec.key == "kautsch_2006":
        prepared_df = prepare_kautsch_2006(raw_df)
        raw_name = "kautsch_2006_raw.csv"
        prepared_name = "kautsch_2006_prepared.csv"
    else:
        raise ValueError(f"Unsupported catalog key: {spec.key}")

    raw_path = spec.output_dir / raw_name
    prepared_path = spec.output_dir / prepared_name
    write_csv(raw_df, raw_path)
    write_csv(prepared_df, prepared_path)

    summary = {
        "catalog": spec.label,
        "input_file": str(input_path),
        "raw_csv": str(raw_path),
        "prepared_csv": str(prepared_path),
        "row_count": int(len(raw_df)),
        "column_count_raw": int(len(raw_df.columns)),
        "column_count_prepared": int(len(prepared_df.columns)),
        "metadata": metadata,
    }
    return summary


def main() -> int:
    project_root = Path.cwd()
    specs = build_catalog_specs(project_root)

    all_summaries: List[Dict[str, object]] = []
    summary_dir = project_root / "data" / "derived" / "structure" / "edge_on_candidates"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        summary = convert_one_catalog(spec)
        all_summaries.append(summary)
        print(f"[OK] {spec.label}: rows={summary['row_count']} -> {summary['prepared_csv']}")

    summary_path = summary_dir / "edge_on_candidate_conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"[OK] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
