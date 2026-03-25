#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Helpers: column finding / normalization
# ------------------------------------------------------------


def norm_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def canonical_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def find_first_matching_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    by_key = {canonical_key(c): c for c in columns}
    for cand in candidates:
        hit = by_key.get(canonical_key(cand))
        if hit is not None:
            return hit
    return None


def find_all_matching_columns(columns: Sequence[str], keywords: Sequence[str]) -> List[str]:
    out: List[str] = []
    lowered = [(c, c.lower()) for c in columns]
    for c, low in lowered:
        if any(k.lower() in low for k in keywords):
            out.append(c)
    return out


# ------------------------------------------------------------
# Helpers: catalog / galaxy-name normalization
# ------------------------------------------------------------

CATALOG_PAT = re.compile(r"\b(UGC|NGC|IC|PGC|ESO|MCG)\s*0*([0-9]+)\b", re.IGNORECASE)


def extract_catalog_id(text: object) -> Tuple[str, Optional[str], Optional[str]]:
    raw = norm_text(text).upper()
    if not raw:
        return "", None, None
    cleaned = re.sub(r"[_\-]+", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    m = CATALOG_PAT.search(cleaned)
    if not m:
        squashed = re.sub(r"\s+", "", cleaned)
        m = CATALOG_PAT.search(squashed)
    if not m:
        return cleaned, None, None
    prefix = m.group(1).upper()
    number = str(int(m.group(2)))
    return cleaned, prefix, number


def galaxy_name_normalized(text: object) -> str:
    cleaned, prefix, number = extract_catalog_id(text)
    if prefix and number:
        return f"{prefix}{number}"
    cleaned = cleaned.upper()
    cleaned = re.sub(r"[^A-Z0-9]+", "", cleaned)
    return cleaned


# ------------------------------------------------------------
# Helpers: coordinate parsing
# ------------------------------------------------------------


def _to_float_if_possible(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)
    text = norm_text(value)
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def parse_ra_to_deg(value: object) -> Optional[float]:
    numeric = _to_float_if_possible(value)
    if numeric is not None:
        if 0.0 <= numeric <= 24.0:
            return numeric * 15.0
        if 0.0 <= numeric <= 360.0:
            return numeric
        return None

    text = norm_text(value)
    if not text:
        return None
    text = text.replace(":", " ")
    parts = [p for p in re.split(r"\s+", text) if p]
    if len(parts) < 1:
        return None
    try:
        h = float(parts[0])
        m = float(parts[1]) if len(parts) >= 2 else 0.0
        s = float(parts[2]) if len(parts) >= 3 else 0.0
    except Exception:
        return None
    if not (0.0 <= h <= 24.0 and 0.0 <= m < 60.0 and 0.0 <= s < 60.0):
        return None
    return 15.0 * (h + m / 60.0 + s / 3600.0)


def parse_dec_to_deg(value: object) -> Optional[float]:
    numeric = _to_float_if_possible(value)
    if numeric is not None:
        if -90.0 <= numeric <= 90.0:
            return numeric
        return None

    text = norm_text(value)
    if not text:
        return None
    sign = -1.0 if text.strip().startswith("-") else 1.0
    text = text.replace("+", " ").replace("-", " ").replace(":", " ")
    parts = [p for p in re.split(r"\s+", text) if p]
    if len(parts) < 1:
        return None
    try:
        d = float(parts[0])
        m = float(parts[1]) if len(parts) >= 2 else 0.0
        s = float(parts[2]) if len(parts) >= 3 else 0.0
    except Exception:
        return None
    if not (0.0 <= abs(d) <= 90.0 and 0.0 <= m < 60.0 and 0.0 <= s < 60.0):
        return None
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


# ------------------------------------------------------------
# Reading and enriching tables
# ------------------------------------------------------------

SEED_NAME_CANDIDATES = [
    "galaxy_id", "galaxy", "name", "diskmass_name", "diskmass_galaxy",
    "UGC_sample", "UGC_survey1", "UGC", "primary_name"
]

SEED_RA_CANDIDATES = [
    "dm_ra_deg", "ra_deg", "_RAJ2000_sample", "_RAJ2000_survey1", "RAJ2000", "RA"
]
SEED_DEC_CANDIDATES = [
    "dm_dec_deg", "dec_deg", "_DEJ2000_sample", "_DEJ2000_survey1", "DEJ2000", "DEC"
]

SPARC_NAME_CANDIDATES = [
    "sparc_galaxy_id", "galaxy_id", "galaxy", "name", "SPARC_name", "s4g_name", "primary_name"
]
SPARC_RA_CANDIDATES = [
    "ra_deg", "RA_deg", "raj2000_deg", "RAJ2000_deg", "RAJ2000", "RA"
]
SPARC_DEC_CANDIDATES = [
    "dec_deg", "DEC_deg", "dej2000_deg", "DEJ2000_deg", "DEJ2000", "DEC"
]


def choose_primary_name_column(df: pd.DataFrame, preferred: Sequence[str]) -> Optional[str]:
    return find_first_matching_column(list(df.columns), preferred)


def choose_primary_coord_columns(df: pd.DataFrame, ra_candidates: Sequence[str], dec_candidates: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    return (
        find_first_matching_column(list(df.columns), ra_candidates),
        find_first_matching_column(list(df.columns), dec_candidates),
    )


def possible_alias_columns(df: pd.DataFrame) -> List[str]:
    cols = find_all_matching_columns(list(df.columns), ["alias", "alt", "name", "ugc", "ngc", "ic", "pgc"])
    # Deduplicate while keeping order
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:20]


def enrich_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    if kind == "seed":
        name_col = choose_primary_name_column(df, SEED_NAME_CANDIDATES)
        ra_col, dec_col = choose_primary_coord_columns(df, SEED_RA_CANDIDATES, SEED_DEC_CANDIDATES)
    else:
        name_col = choose_primary_name_column(df, SPARC_NAME_CANDIDATES)
        ra_col, dec_col = choose_primary_coord_columns(df, SPARC_RA_CANDIDATES, SPARC_DEC_CANDIDATES)

    alias_cols = possible_alias_columns(df)

    def combine_names(row: pd.Series) -> str:
        values: List[str] = []
        if name_col is not None:
            values.append(norm_text(row.get(name_col, "")))
        for c in alias_cols:
            values.append(norm_text(row.get(c, "")))
        merged = " | ".join([v for v in values if v])
        return merged

    df["__primary_name__"] = df[name_col].astype(str) if name_col is not None else ""
    df["__all_names__"] = df.apply(combine_names, axis=1)
    df["__name_norm__"] = df["__primary_name__"].map(galaxy_name_normalized)

    # catalog id from the broadest combined-name text
    cat = df["__all_names__"].map(extract_catalog_id)
    df["__catalog_clean__"] = cat.map(lambda x: x[0])
    df["__catalog_prefix__"] = cat.map(lambda x: x[1])
    df["__catalog_number__"] = cat.map(lambda x: x[2])
    df["__catalog_norm__"] = df.apply(
        lambda r: f"{r['__catalog_prefix__']}{r['__catalog_number__']}" if pd.notna(r['__catalog_prefix__']) and pd.notna(r['__catalog_number__']) and r['__catalog_prefix__'] and r['__catalog_number__'] else "",
        axis=1,
    )

    if ra_col is not None:
        df["__ra_src__"] = ra_col
        df["__ra_deg__"] = df[ra_col].map(parse_ra_to_deg)
    else:
        df["__ra_src__"] = ""
        df["__ra_deg__"] = np.nan

    if dec_col is not None:
        df["__dec_src__"] = dec_col
        df["__dec_deg__"] = df[dec_col].map(parse_dec_to_deg)
    else:
        df["__dec_src__"] = ""
        df["__dec_deg__"] = np.nan

    df["__has_coords__"] = df["__ra_deg__"].notna() & df["__dec_deg__"].notna()
    return df


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------


def value_fill_stats(df: pd.DataFrame) -> List[Dict[str, object]]:
    rows = []
    n = len(df)
    for c in df.columns:
        non_null = int(df[c].notna().sum())
        rows.append(
            {
                "column": c,
                "non_null": non_null,
                "null": n - non_null,
                "fill_ratio": round(non_null / n, 6) if n else 0.0,
                "dtype": str(df[c].dtype),
            }
        )
    rows.sort(key=lambda x: (x["fill_ratio"], x["column"]))
    return rows


def coord_summary(df: pd.DataFrame, label: str) -> Dict[str, object]:
    has = df["__has_coords__"]
    return {
        "label": label,
        "rows": int(len(df)),
        "rows_with_coords": int(has.sum()),
        "rows_without_coords": int((~has).sum()),
        "coord_fraction": round(float(has.mean()), 6) if len(df) else 0.0,
        "ra_min_deg": None if df["__ra_deg__"].dropna().empty else float(df["__ra_deg__"].dropna().min()),
        "ra_max_deg": None if df["__ra_deg__"].dropna().empty else float(df["__ra_deg__"].dropna().max()),
        "dec_min_deg": None if df["__dec_deg__"].dropna().empty else float(df["__dec_deg__"].dropna().min()),
        "dec_max_deg": None if df["__dec_deg__"].dropna().empty else float(df["__dec_deg__"].dropna().max()),
    }


def overlap_by_catalog(seed: pd.DataFrame, sparc: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    seed_c = seed[seed["__catalog_norm__"] != ""].copy()
    sparc_c = sparc[sparc["__catalog_norm__"] != ""].copy()

    seed_keys = set(seed_c["__catalog_norm__"])
    sparc_keys = set(sparc_c["__catalog_norm__"])
    inter = seed_keys & sparc_keys

    merged = seed_c.merge(
        sparc_c,
        on="__catalog_norm__",
        how="left",
        suffixes=("_seed", "_sparc"),
    )
    merged["catalog_match_found"] = merged["__primary_name___sparc"].notna()

    summary = {
        "seed_catalog_unique": len(seed_keys),
        "sparc_catalog_unique": len(sparc_keys),
        "catalog_intersection_unique": len(inter),
        "seed_rows_with_catalog": int(len(seed_c)),
        "sparc_rows_with_catalog": int(len(sparc_c)),
        "seed_rows_having_any_catalog_match": int(merged["catalog_match_found"].sum()),
    }
    return merged, summary


def spherical_sep_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    ra1r = math.radians(ra1)
    dec1r = math.radians(dec1)
    ra2r = math.radians(ra2)
    dec2r = math.radians(dec2)
    cosang = (
        math.sin(dec1r) * math.sin(dec2r)
        + math.cos(dec1r) * math.cos(dec2r) * math.cos(ra1r - ra2r)
    )
    cosang = min(1.0, max(-1.0, cosang))
    return math.degrees(math.acos(cosang))


def nearest_coord_matches(seed: pd.DataFrame, sparc: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    seed_c = seed[seed["__has_coords__"]].copy()
    sparc_c = sparc[sparc["__has_coords__"]].copy()
    if max_rows is not None:
        seed_c = seed_c.head(max_rows).copy()

    results: List[Dict[str, object]] = []
    if seed_c.empty or sparc_c.empty:
        return pd.DataFrame(results)

    sparc_rows = list(sparc_c[["__primary_name__", "__catalog_norm__", "__ra_deg__", "__dec_deg__"]].itertuples(index=False, name=None))

    for seed_name, seed_cat, seed_ra, seed_dec in seed_c[["__primary_name__", "__catalog_norm__", "__ra_deg__", "__dec_deg__"]].itertuples(index=False, name=None):
        best_sep = None
        best_name = ""
        best_cat = ""
        for s_name, s_cat, s_ra, s_dec in sparc_rows:
            sep = spherical_sep_deg(float(seed_ra), float(seed_dec), float(s_ra), float(s_dec))
            if best_sep is None or sep < best_sep:
                best_sep = sep
                best_name = s_name
                best_cat = s_cat
        results.append(
            {
                "seed_name": seed_name,
                "seed_catalog_norm": seed_cat,
                "seed_ra_deg": seed_ra,
                "seed_dec_deg": seed_dec,
                "nearest_sparc_name": best_name,
                "nearest_sparc_catalog_norm": best_cat,
                "nearest_sep_deg": best_sep,
            }
        )
    return pd.DataFrame(results)


def sample_unmatched_catalog_rows(seed_cat_merge: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    if seed_cat_merge.empty:
        return pd.DataFrame(columns=["seed_name", "seed_catalog_norm"])
    mask = ~seed_cat_merge["catalog_match_found"]
    cols = [c for c in ["__primary_name___seed", "__catalog_norm__"] if c in seed_cat_merge.columns]
    out = seed_cat_merge.loc[mask, cols].copy().head(limit)
    rename_map = {"__primary_name___seed": "seed_name", "__catalog_norm__": "seed_catalog_norm"}
    return out.rename(columns=rename_map)


def sample_exact_catalog_matches(seed_cat_merge: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    if seed_cat_merge.empty:
        return pd.DataFrame(columns=["seed_name", "sparc_name", "catalog_norm"])
    mask = seed_cat_merge["catalog_match_found"]
    cols = [
        "__primary_name___seed",
        "__primary_name___sparc",
        "__catalog_norm__",
        "__ra_deg___seed",
        "__dec_deg___seed",
        "__ra_deg___sparc",
        "__dec_deg___sparc",
    ]
    cols = [c for c in cols if c in seed_cat_merge.columns]
    out = seed_cat_merge.loc[mask, cols].copy().head(limit)
    return out.rename(
        columns={
            "__primary_name___seed": "seed_name",
            "__primary_name___sparc": "sparc_name",
            "__catalog_norm__": "catalog_norm",
            "__ra_deg___seed": "seed_ra_deg",
            "__dec_deg___seed": "seed_dec_deg",
            "__ra_deg___sparc": "sparc_ra_deg",
            "__dec_deg___sparc": "sparc_dec_deg",
        }
    )


def choose_reader(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    # csv/tsv/txt fallback with sep inference
    return pd.read_csv(path, sep=None, engine="python")


def write_text_report(path: Path, report: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("DiskMass ↔ SPARC input diagnostic report")
    lines.append("=" * 48)
    lines.append("")

    for key in ["seed_file", "sparc_file"]:
        lines.append(f"{key}: {report[key]}")
    lines.append("")

    lines.append("[Primary column selection]")
    prim = report["primary_columns"]
    for k, v in prim.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("[Coordinate summary]")
    for side in ["seed_coord_summary", "sparc_coord_summary"]:
        cs = report[side]
        lines.append(f"- {cs['label']}: rows={cs['rows']}, rows_with_coords={cs['rows_with_coords']}, rows_without_coords={cs['rows_without_coords']}, coord_fraction={cs['coord_fraction']}")
        lines.append(f"  RA range: {cs['ra_min_deg']} .. {cs['ra_max_deg']}")
        lines.append(f"  Dec range: {cs['dec_min_deg']} .. {cs['dec_max_deg']}")
    lines.append("")

    lines.append("[Catalog overlap]")
    for k, v in report["catalog_overlap_summary"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("[Interpretation hints]")
    hints = report["interpretation_hints"]
    for h in hints:
        lines.append(f"- {h}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_interpretation_hints(report: Dict[str, object]) -> List[str]:
    hints: List[str] = []
    seed_cs = report["seed_coord_summary"]
    sparc_cs = report["sparc_coord_summary"]
    cat = report["catalog_overlap_summary"]

    if seed_cs["coord_fraction"] < 0.5:
        hints.append("DiskMass 입력에서 좌표 파싱 성공률이 낮습니다. RA/DEC 원본 컬럼 또는 sexagesimal 형식을 우선 점검해야 합니다.")
    else:
        hints.append("DiskMass 입력의 좌표 파싱 성공률은 크게 문제되지 않아 보입니다.")

    if sparc_cs["coord_fraction"] < 0.5:
        hints.append("SPARC 입력에서 좌표 열 선택 또는 좌표 형식 파싱이 실패하고 있을 가능성이 큽니다.")
    else:
        hints.append("SPARC 입력의 좌표 파싱 성공률은 기본적으로 확보된 것으로 보입니다.")

    if cat["seed_rows_having_any_catalog_match"] == 0:
        hints.append("정규화된 UGC/NGC 번호 exact match가 0건입니다. 실제 이름 스키마가 현재 정규화 규칙과 다를 가능성이 큽니다.")
    elif cat["seed_rows_having_any_catalog_match"] < max(5, int(0.1 * max(cat["seed_rows_with_catalog"], 1))):
        hints.append("카탈로그 번호 exact match 비율이 낮습니다. SPARC 쪽 alias/name 열을 더 폭넓게 포함해야 할 수 있습니다.")
    else:
        hints.append("카탈로그 번호 exact match는 일정 수준 존재합니다. 이후 병목은 좌표 반경 또는 alias 통합 규칙일 가능성이 큽니다.")

    return hints


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose DiskMass alias seed and SPARC crossmatch input tables.")
    parser.add_argument("--seed-file", required=True, help="Path to diskmass_alias_seed.csv (or xlsx/tsv).")
    parser.add_argument("--sparc-file", required=True, help="Path to sparc_s4g_crossmatch_direct.csv (or xlsx/tsv).")
    parser.add_argument("--output-dir", required=True, help="Directory to write diagnostic outputs.")
    parser.add_argument("--nearest-limit", type=int, default=200, help="How many seed rows with coordinates to sample for nearest-neighbor coordinate diagnostics.")
    args = parser.parse_args()

    seed_path = Path(args.seed_file)
    sparc_path = Path(args.sparc_file)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    seed_raw = choose_reader(seed_path)
    sparc_raw = choose_reader(sparc_path)

    seed = enrich_table(seed_raw, kind="seed")
    sparc = enrich_table(sparc_raw, kind="sparc")

    seed_cat_merge, cat_summary = overlap_by_catalog(seed, sparc)
    nearest_df = nearest_coord_matches(seed, sparc, max_rows=args.nearest_limit)

    primary_columns = {
        "seed_name_column": choose_primary_name_column(seed_raw, SEED_NAME_CANDIDATES),
        "seed_ra_column": choose_primary_coord_columns(seed_raw, SEED_RA_CANDIDATES, SEED_DEC_CANDIDATES)[0],
        "seed_dec_column": choose_primary_coord_columns(seed_raw, SEED_RA_CANDIDATES, SEED_DEC_CANDIDATES)[1],
        "sparc_name_column": choose_primary_name_column(sparc_raw, SPARC_NAME_CANDIDATES),
        "sparc_ra_column": choose_primary_coord_columns(sparc_raw, SPARC_RA_CANDIDATES, SPARC_DEC_CANDIDATES)[0],
        "sparc_dec_column": choose_primary_coord_columns(sparc_raw, SPARC_RA_CANDIDATES, SPARC_DEC_CANDIDATES)[1],
    }

    report: Dict[str, object] = {
        "seed_file": str(seed_path),
        "sparc_file": str(sparc_path),
        "primary_columns": primary_columns,
        "seed_coord_summary": coord_summary(seed, "seed"),
        "sparc_coord_summary": coord_summary(sparc, "sparc"),
        "catalog_overlap_summary": cat_summary,
        "seed_alias_candidate_columns": possible_alias_columns(seed_raw),
        "sparc_alias_candidate_columns": possible_alias_columns(sparc_raw),
    }
    report["interpretation_hints"] = build_interpretation_hints(report)

    # Output files
    (outdir / "seed_fill_stats.csv").write_text(pd.DataFrame(value_fill_stats(seed_raw)).to_csv(index=False), encoding="utf-8")
    (outdir / "sparc_fill_stats.csv").write_text(pd.DataFrame(value_fill_stats(sparc_raw)).to_csv(index=False), encoding="utf-8")
    seed_cat_merge.to_csv(outdir / "catalog_overlap_detail.csv", index=False, encoding="utf-8-sig")
    sample_unmatched_catalog_rows(seed_cat_merge).to_csv(outdir / "sample_unmatched_catalog_rows.csv", index=False, encoding="utf-8-sig")
    sample_exact_catalog_matches(seed_cat_merge).to_csv(outdir / "sample_exact_catalog_matches.csv", index=False, encoding="utf-8-sig")
    nearest_df.to_csv(outdir / "nearest_coord_matches_sample.csv", index=False, encoding="utf-8-sig")
    seed[["__primary_name__", "__name_norm__", "__catalog_norm__", "__ra_deg__", "__dec_deg__", "__has_coords__"]].to_csv(
        outdir / "seed_enriched_preview.csv", index=False, encoding="utf-8-sig"
    )
    sparc[["__primary_name__", "__name_norm__", "__catalog_norm__", "__ra_deg__", "__dec_deg__", "__has_coords__"]].to_csv(
        outdir / "sparc_enriched_preview.csv", index=False, encoding="utf-8-sig"
    )

    with open(outdir / "diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_text_report(outdir / "diagnostic_report.txt", report)

    print("[DONE] Diagnostic files written to:", outdir)
    print(" - diagnostic_report.txt")
    print(" - diagnostic_summary.json")
    print(" - seed_fill_stats.csv")
    print(" - sparc_fill_stats.csv")
    print(" - catalog_overlap_detail.csv")
    print(" - sample_unmatched_catalog_rows.csv")
    print(" - sample_exact_catalog_matches.csv")
    print(" - nearest_coord_matches_sample.csv")
    print(" - seed_enriched_preview.csv")
    print(" - sparc_enriched_preview.csv")


if __name__ == "__main__":
    main()
