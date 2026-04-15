from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Name normalization helpers
# ---------------------------------------------------------------------

CATALOG_PATTERNS = [
    ("UGC", re.compile(r"\bUGC\s*0*(\d+)\b", re.IGNORECASE)),
    ("NGC", re.compile(r"\bNGC\s*0*(\d+)\b", re.IGNORECASE)),
    ("IC", re.compile(r"\bIC\s*0*(\d+)\b", re.IGNORECASE)),
    ("PGC", re.compile(r"\bPGC\s*0*(\d+)\b", re.IGNORECASE)),
    ("ESO", re.compile(r"\bESO\s*0*([0-9]+)\s*[- ]\s*0*([0-9]+)\b", re.IGNORECASE)),
    ("DDO", re.compile(r"\bDDO\s*0*(\d+)\b", re.IGNORECASE)),
    ("UGCA", re.compile(r"\bUGCA\s*0*(\d+)\b", re.IGNORECASE)),
    ("KK", re.compile(r"\bKK\s*0*([0-9]+)\s*[- ]\s*0*([0-9]+)\b", re.IGNORECASE)),
    ("F", re.compile(r"\bF\s*0*([0-9]+)\s*[- ]\s*([A-Z]?\d+)\b", re.IGNORECASE)),
]


def clean_text(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    return s


def normalize_simple_name(value: object) -> str:
    s = clean_text(value).upper()
    if not s:
        return ""
    s = re.sub(r"[^A-Z0-9]+", "", s)
    return s


def extract_catalog_token(value: object) -> str:
    s = clean_text(value)
    if not s:
        return ""
    up = s.upper()
    for prefix, pattern in CATALOG_PATTERNS:
        m = pattern.search(up)
        if not m:
            continue
        parts = [prefix] + [p.lstrip("0") or "0" for p in m.groups()]
        return "_".join(parts)
    return ""


# ---------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------


def angular_sep_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    cos_sep = (
        math.sin(dec1) * math.sin(dec2)
        + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2)
    )
    cos_sep = max(-1.0, min(1.0, cos_sep))
    return math.degrees(math.acos(cos_sep))


# ---------------------------------------------------------------------
# Column picking
# ---------------------------------------------------------------------


def first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    existing = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in existing:
            return existing[c.lower()]
    return None


@dataclass
class MatchConfig:
    max_coord_sep_deg: float = 0.10
    review_coord_sep_deg: float = 0.30


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------


def load_bizyaev(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["preferred_name", "RAJ2000", "DEJ2000", "h", "z0", "z0_over_h", "h_over_z0"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Bizyaev file missing required columns: {missing}")

    df = df.copy()
    df["bizyaev_name"] = df["preferred_name"].fillna(df.get("AName", pd.Series(index=df.index))).fillna(df["Name"])
    df["bizyaev_name_norm"] = df["bizyaev_name"].map(normalize_simple_name)
    df["bizyaev_catalog_token"] = df["bizyaev_name"].map(extract_catalog_token)
    if "catalog_name_normalized" in df.columns:
        raw_catalog = df["catalog_name_normalized"].fillna("")
        df["bizyaev_catalog_token"] = np.where(
            df["bizyaev_catalog_token"].eq(""),
            raw_catalog.map(extract_catalog_token),
            df["bizyaev_catalog_token"],
        )
    df["RAJ2000"] = pd.to_numeric(df["RAJ2000"], errors="coerce")
    df["DEJ2000"] = pd.to_numeric(df["DEJ2000"], errors="coerce")
    return df


SPARC_NAME_CANDIDATES = [
    "galaxy_id",
    "filename",
    "Name",
    "name",
    "name_normalized",
    "s4g_name_normalized",
]
SPARC_RA_CANDIDATES = ["RAJ2000", "ra_deg", "RAdeg", "_RAJ2000"]
SPARC_DEC_CANDIDATES = ["DEJ2000", "dec_deg", "DEdeg", "_DEJ2000"]
SPARC_ALIAS_CANDIDATES = [
    "galaxy_id",
    "filename",
    "Name",
    "name_normalized",
    "s4g_name_normalized",
    "s4g_name",
]


def load_sparc_index(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    name_col = first_existing(df, SPARC_NAME_CANDIDATES)
    if not name_col:
        raise ValueError("Could not identify SPARC primary name column.")

    ra_col = first_existing(df, SPARC_RA_CANDIDATES)
    dec_col = first_existing(df, SPARC_DEC_CANDIDATES)

    out = df.copy()
    out["sparc_name"] = out[name_col]
    out["sparc_name_norm"] = out["sparc_name"].map(normalize_simple_name)

    # Aggregate alias tokens from multiple columns.
    alias_tokens: list[list[str]] = []
    for _, row in out.iterrows():
        tokens: set[str] = set()
        names: set[str] = set()
        for col in SPARC_ALIAS_CANDIDATES:
            if col in out.columns:
                text = clean_text(row[col])
                if text:
                    names.add(text)
                    token = extract_catalog_token(text)
                    if token:
                        tokens.add(token)
                    names.add(normalize_simple_name(text))
        alias_tokens.append(sorted(tokens.union({t for t in names if t})))
    out["sparc_alias_tokens"] = alias_tokens

    if ra_col and dec_col:
        out["sparc_ra_deg"] = pd.to_numeric(out[ra_col], errors="coerce")
        out["sparc_dec_deg"] = pd.to_numeric(out[dec_col], errors="coerce")
    else:
        out["sparc_ra_deg"] = np.nan
        out["sparc_dec_deg"] = np.nan
    return out


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------


def build_exact_maps(sparc: pd.DataFrame):
    name_map: dict[str, list[int]] = {}
    token_map: dict[str, list[int]] = {}
    for idx, row in sparc.iterrows():
        n = clean_text(row["sparc_name_norm"])
        if n:
            name_map.setdefault(n, []).append(idx)
        for token in row["sparc_alias_tokens"]:
            token_map.setdefault(token, []).append(idx)
    return name_map, token_map



def choose_best_candidate(candidates: list[tuple[int, str, Optional[float]]], sparc: pd.DataFrame):
    def key_fn(item):
        idx, reason, sep = item
        # exact catalog > exact normalized name > coordinate
        reason_rank = {
            "catalog_exact": 0,
            "name_exact": 1,
            "coord_close": 2,
            "coord_review": 3,
        }.get(reason, 9)
        sep_val = sep if sep is not None and not math.isnan(sep) else 999.0
        return (reason_rank, sep_val, clean_text(sparc.at[idx, "sparc_name"]))

    return sorted(candidates, key=key_fn)[0]



def match_bizyaev_to_sparc(biz: pd.DataFrame, sparc: pd.DataFrame, config: MatchConfig) -> pd.DataFrame:
    name_map, token_map = build_exact_maps(sparc)
    records = []

    sparc_coord = sparc.dropna(subset=["sparc_ra_deg", "sparc_dec_deg"]).copy()

    for _, row in biz.iterrows():
        candidates: list[tuple[int, str, Optional[float]]] = []
        biz_name_norm = clean_text(row["bizyaev_name_norm"])
        biz_token = clean_text(row["bizyaev_catalog_token"])

        if biz_token and biz_token in token_map:
            for idx in token_map[biz_token]:
                candidates.append((idx, "catalog_exact", None))

        if biz_name_norm and biz_name_norm in name_map:
            for idx in name_map[biz_name_norm]:
                candidates.append((idx, "name_exact", None))

        ra = row["RAJ2000"]
        dec = row["DEJ2000"]
        nearest_sep = np.nan
        nearest_name = ""
        if pd.notna(ra) and pd.notna(dec) and not sparc_coord.empty:
            seps = sparc_coord.apply(
                lambda s: angular_sep_deg(float(ra), float(dec), float(s["sparc_ra_deg"]), float(s["sparc_dec_deg"])),
                axis=1,
            )
            min_pos = seps.idxmin()
            nearest_sep = float(seps.loc[min_pos])
            nearest_name = clean_text(sparc_coord.loc[min_pos, "sparc_name"])
            if nearest_sep <= config.max_coord_sep_deg:
                candidates.append((min_pos, "coord_close", nearest_sep))
            elif nearest_sep <= config.review_coord_sep_deg:
                candidates.append((min_pos, "coord_review", nearest_sep))

        if candidates:
            best_idx, best_reason, best_sep = choose_best_candidate(candidates, sparc)
            best_name = clean_text(sparc.at[best_idx, "sparc_name"])
            best_sep_final = best_sep
            if best_sep_final is None or math.isnan(best_sep_final):
                # compute sep if coords exist for both
                sra = sparc.at[best_idx, "sparc_ra_deg"]
                sdec = sparc.at[best_idx, "sparc_dec_deg"]
                if pd.notna(ra) and pd.notna(dec) and pd.notna(sra) and pd.notna(sdec):
                    best_sep_final = angular_sep_deg(float(ra), float(dec), float(sra), float(sdec))
                else:
                    best_sep_final = np.nan
        else:
            best_idx = None
            best_reason = "unmatched"
            best_name = ""
            best_sep_final = np.nan

        records.append(
            {
                "bizyaev_name": row["bizyaev_name"],
                "bizyaev_name_norm": row["bizyaev_name_norm"],
                "bizyaev_catalog_token": row["bizyaev_catalog_token"],
                "RAJ2000": row["RAJ2000"],
                "DEJ2000": row["DEJ2000"],
                "h": row["h"],
                "z0": row["z0"],
                "z0_over_h": row["z0_over_h"],
                "h_over_z0": row["h_over_z0"],
                "B/T": row.get("B/T", np.nan),
                "Type": row.get("Type", ""),
                "thickness_flag": row.get("thickness_flag", ""),
                "sparc_match_name": best_name,
                "sparc_match_reason": best_reason,
                "match_sep_deg": best_sep_final,
                "nearest_sparc_name": nearest_name,
                "nearest_sparc_sep_deg": nearest_sep,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------


def summarize(df: pd.DataFrame) -> dict:
    matched = df[~df["sparc_match_reason"].eq("unmatched")]
    return {
        "rows_total": int(len(df)),
        "rows_matched_any": int(len(matched)),
        "rows_unmatched": int((df["sparc_match_reason"] == "unmatched").sum()),
        "match_reason_counts": df["sparc_match_reason"].value_counts(dropna=False).to_dict(),
        "rows_with_catalog_token": int(df["bizyaev_catalog_token"].fillna("").ne("").sum()),
        "rows_with_coords": int(df[["RAJ2000", "DEJ2000"]].notna().all(axis=1).sum()),
        "median_z0_over_h_matched": float(matched["z0_over_h"].median()) if len(matched) else None,
        "median_sep_deg_matched": float(matched["match_sep_deg"].median()) if len(matched["match_sep_deg"].dropna()) else None,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Crossmatch Bizyaev r-band structural table with SPARC galaxy index.")
    parser.add_argument(
        "--bizyaev-file",
        default=r"data/derived/Bizyaev/bizyaev_table4_rband_for_crossmatch.csv",
        help="Path to Bizyaev crossmatch-ready CSV.",
    )
    parser.add_argument(
        "--sparc-index-file",
        default=r"data/derived/crossmatch/sparc_diskmass_S4G/sparc_galaxy_index.csv",
        help="Path to SPARC galaxy index CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=r"data/derived/crossmatch/sparc_bizyaev",
        help="Output directory for crossmatch results.",
    )
    parser.add_argument("--max-coord-sep-deg", type=float, default=0.10)
    parser.add_argument("--review-coord-sep-deg", type=float, default=0.30)
    args = parser.parse_args()

    bizyaev_file = Path(args.bizyaev_file)
    sparc_index_file = Path(args.sparc_index_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    biz = load_bizyaev(bizyaev_file)
    sparc = load_sparc_index(sparc_index_file)
    config = MatchConfig(
        max_coord_sep_deg=args.max_coord_sep_deg,
        review_coord_sep_deg=args.review_coord_sep_deg,
    )

    result = match_bizyaev_to_sparc(biz, sparc, config)

    matched = result[result["sparc_match_reason"] != "unmatched"].copy()
    unmatched = result[result["sparc_match_reason"] == "unmatched"].copy()
    exact = result[result["sparc_match_reason"].isin(["catalog_exact", "name_exact"])].copy()
    coord_review = result[result["sparc_match_reason"].isin(["coord_close", "coord_review"])].copy()

    result.to_csv(output_dir / "sparc_bizyaev_crossmatch_all.csv", index=False)
    matched.to_csv(output_dir / "sparc_bizyaev_crossmatch_matched.csv", index=False)
    exact.to_csv(output_dir / "sparc_bizyaev_crossmatch_exact.csv", index=False)
    coord_review.to_csv(output_dir / "sparc_bizyaev_crossmatch_coord_review.csv", index=False)
    unmatched.to_csv(output_dir / "sparc_bizyaev_crossmatch_unmatched.csv", index=False)

    summary = summarize(result)
    summary.update(
        {
            "bizyaev_file": str(bizyaev_file),
            "sparc_index_file": str(sparc_index_file),
            "output_dir": str(output_dir),
            "max_coord_sep_deg": args.max_coord_sep_deg,
            "review_coord_sep_deg": args.review_coord_sep_deg,
        }
    )
    with open(output_dir / "sparc_bizyaev_crossmatch_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] SPARC x Bizyaev crossmatch complete")
    print(f"[INFO] all={len(result)} matched={len(matched)} unmatched={len(unmatched)}")
    print(f"[INFO] reason counts={summary['match_reason_counts']}")
    print(f"[INFO] output_dir={output_dir}")


if __name__ == "__main__":
    main()
