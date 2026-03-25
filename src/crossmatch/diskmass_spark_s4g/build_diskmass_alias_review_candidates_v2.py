from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Conservative defaults: close matches can be auto-confirmed later,
# broader matches are kept for manual review.
MAX_COORD_MATCH_DEG = 0.20
MAX_REVIEW_MATCH_DEG = 0.50
TOP_N_CANDIDATES = 3

STRING_COLUMNS = [
    "ugc_norm",
    "diskmass_display_name",
    "diskmass_name_normalized",
    "alias_match_status",
    "candidate_sparc_galaxy_id",
    "candidate_sparc_name_normalized",
    "alias_source",
    "review_notes",
    "sparc_reference_note",
    "auto_exact_name_hint",
    "coord_alias_recommendation",
    "candidate_1_sparc_galaxy_id",
    "candidate_1_catalog_prefix",
    "candidate_2_sparc_galaxy_id",
    "candidate_2_catalog_prefix",
    "candidate_3_sparc_galaxy_id",
    "candidate_3_catalog_prefix",
]

FLOAT_COLUMNS = [
    "dm_ra_deg",
    "dm_dec_deg",
    "candidate_1_sep_deg",
    "candidate_2_sep_deg",
    "candidate_3_sep_deg",
    "best_coord_sep_deg",
]

INTLIKE_COLUMNS = [
    "coord_candidate_count",
    "candidate_1_catalog_number",
    "candidate_2_catalog_number",
    "candidate_3_catalog_number",
]

NAME_CANDIDATE_COLUMNS = [
    "ugc_norm",
    "diskmass_name_normalized",
    "diskmass_display_name",
    "UGC_sample",
    "UGC_survey1",
]

DM_RA_CANDIDATE_COLUMNS = [
    "dm_ra_deg",
    "_RAJ2000_sample",
    "_RAJ2000_survey1",
    "RAJ2000",
    "RA_ICRS",
    "RAdeg",
    "RA_deg",
    "ra_deg",
    "ra",
]

DM_DEC_CANDIDATE_COLUMNS = [
    "dm_dec_deg",
    "_DEJ2000_sample",
    "_DEJ2000_survey1",
    "DEJ2000",
    "DE_ICRS",
    "DEdeg",
    "DEC_deg",
    "dec_deg",
    "dec",
]

SPARC_NAME_CANDIDATE_COLUMNS = [
    "sparc_galaxy_id",
    "galaxy_id",
    "sparc_name",
    "galaxy",
    "name",
    "object",
    "s4g_name",
]

SPARC_RA_CANDIDATE_COLUMNS = [
    "sparc_ra_deg",
    "s4g_ra_deg",
    "ra_deg",
    "RAdeg",
    "RA_deg",
    "RAJ2000",
    "_RAJ2000",
    "ra",
]

SPARC_DEC_CANDIDATE_COLUMNS = [
    "sparc_dec_deg",
    "s4g_dec_deg",
    "dec_deg",
    "DEC_deg",
    "DEdeg",
    "DEJ2000",
    "_DEJ2000",
    "dec",
]


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "", text)
    return text


CATALOG_RE = re.compile(r"^([A-Z]+)(\d+)$")


def split_catalog_id(text: str) -> tuple[str | None, int | None]:
    norm = normalize_name(text)
    match = CATALOG_RE.match(norm)
    if not match:
        return None, None
    return match.group(1), int(match.group(2))


def first_existing(columns: Iterable[str], df: pd.DataFrame) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None


# ----- dtype helpers -----
def ensure_string_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df[col] = df[col].astype("string")



def ensure_float_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = pd.Series(np.nan, index=df.index, dtype="float64")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")



def ensure_intlike_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        df[col] = pd.Series(np.nan, index=df.index, dtype="float64")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ----- coordinate parsing -----
def _clean_coord_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = (
        text.replace("h", " ")
        .replace("m", " ")
        .replace("s", " ")
        .replace("d", " ")
        .replace("°", " ")
        .replace("′", " ")
        .replace("″", " ")
        .replace(":", " ")
        .replace(",", " ")
    )
    text = re.sub(r"\s+", " ", text).strip()
    return text



def parse_ra_to_deg(value: object) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        v = float(value)
        if 0.0 <= v <= 360.0:
            return v
        # Sometimes RA may be stored as hours in numeric form.
        if 0.0 <= v <= 24.0:
            return v * 15.0
        return math.nan

    text = _clean_coord_text(value)
    if not text:
        return math.nan
    parts = text.split(" ")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return math.nan

    if len(nums) == 1:
        v = nums[0]
        if 0.0 <= v <= 360.0:
            return v
        if 0.0 <= v <= 24.0:
            return v * 15.0
        return math.nan

    h = nums[0]
    m = nums[1] if len(nums) >= 2 else 0.0
    s = nums[2] if len(nums) >= 3 else 0.0
    if not (0.0 <= h <= 24.0 and 0.0 <= m < 60.0 and 0.0 <= s < 60.0):
        return math.nan
    return 15.0 * (h + m / 60.0 + s / 3600.0)



def parse_dec_to_deg(value: object) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        v = float(value)
        return v if -90.0 <= v <= 90.0 else math.nan

    text = _clean_coord_text(value)
    if not text:
        return math.nan

    sign = -1.0 if text.startswith("-") else 1.0
    text = text.lstrip("+-").strip()
    parts = text.split(" ")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return math.nan

    if len(nums) == 1:
        v = sign * nums[0]
        return v if -90.0 <= v <= 90.0 else math.nan

    d = nums[0]
    m = nums[1] if len(nums) >= 2 else 0.0
    s = nums[2] if len(nums) >= 3 else 0.0
    if not (0.0 <= abs(d) <= 90.0 and 0.0 <= m < 60.0 and 0.0 <= s < 60.0):
        return math.nan
    return sign * (abs(d) + m / 60.0 + s / 3600.0)



def first_valid_ra(row: pd.Series, candidate_columns: list[str]) -> float:
    for col in candidate_columns:
        if col not in row.index:
            continue
        value = row[col]
        out = parse_ra_to_deg(value)
        if not math.isnan(out):
            return out
    return math.nan



def first_valid_dec(row: pd.Series, candidate_columns: list[str]) -> float:
    for col in candidate_columns:
        if col not in row.index:
            continue
        value = row[col]
        out = parse_dec_to_deg(value)
        if not math.isnan(out):
            return out
    return math.nan



def angular_sep_deg(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
    if any(math.isnan(v) for v in [ra1_deg, dec1_deg, ra2_deg, dec2_deg]):
        return math.nan

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


# ----- SPARC preparation -----
def prepare_sparc_catalog(sparc_direct: pd.DataFrame) -> pd.DataFrame:
    df = sparc_direct.copy()

    name_col = first_existing(SPARC_NAME_CANDIDATE_COLUMNS, df)
    if name_col is None:
        raise ValueError(
            f"SPARC/S4G direct CSV must contain one of these name columns: {SPARC_NAME_CANDIDATE_COLUMNS}"
        )

    ensure_string_column(df, name_col)
    df["sparc_galaxy_id"] = df[name_col].astype("string")
    df["sparc_name_normalized"] = df["sparc_galaxy_id"].map(normalize_name).astype("string")

    df["sparc_ra_deg"] = df.apply(lambda row: first_valid_ra(row, SPARC_RA_CANDIDATE_COLUMNS), axis=1)
    df["sparc_dec_deg"] = df.apply(lambda row: first_valid_dec(row, SPARC_DEC_CANDIDATE_COLUMNS), axis=1)

    prefixes: list[str | None] = []
    numbers: list[int | None] = []
    for name in df["sparc_galaxy_id"].astype("string").fillna(""):
        prefix, number = split_catalog_id(str(name))
        prefixes.append(prefix)
        numbers.append(number)
    df["catalog_prefix"] = pd.Series(prefixes, index=df.index, dtype="string")
    df["catalog_number"] = pd.to_numeric(pd.Series(numbers, index=df.index), errors="coerce")

    df = df.drop_duplicates(subset=["sparc_name_normalized", "sparc_ra_deg", "sparc_dec_deg"], keep="first")
    df = df.reset_index(drop=True)
    return df


# ----- candidate generation -----
def add_name_hints(alias_df: pd.DataFrame, sparc_df: pd.DataFrame) -> None:
    sparc_names = set(sparc_df["sparc_name_normalized"].astype("string").fillna(""))
    hints: list[str | None] = []
    for _, row in alias_df.iterrows():
        hint = None
        for col in NAME_CANDIDATE_COLUMNS:
            if col not in alias_df.columns:
                continue
            norm = normalize_name(row[col])
            if norm and norm in sparc_names:
                hint = norm
                break
        hints.append(hint)
    alias_df["auto_exact_name_hint"] = pd.Series(hints, index=alias_df.index, dtype="string")



def build_coord_candidates(dm_ra_deg: float, dm_dec_deg: float, sparc_df: pd.DataFrame) -> list[dict[str, object]]:
    if math.isnan(dm_ra_deg) or math.isnan(dm_dec_deg):
        return []

    work = sparc_df[[
        "sparc_galaxy_id",
        "sparc_name_normalized",
        "sparc_ra_deg",
        "sparc_dec_deg",
        "catalog_prefix",
        "catalog_number",
    ]].copy()
    work["sep_deg"] = work.apply(
        lambda row: angular_sep_deg(dm_ra_deg, dm_dec_deg, float(row["sparc_ra_deg"]), float(row["sparc_dec_deg"])),
        axis=1,
    )
    work = work[pd.to_numeric(work["sep_deg"], errors="coerce").notna()].copy()
    work = work.sort_values("sep_deg", kind="stable")
    work = work[work["sep_deg"] <= MAX_REVIEW_MATCH_DEG].head(TOP_N_CANDIDATES)

    out: list[dict[str, object]] = []
    for _, row in work.iterrows():
        out.append(
            {
                "sparc_galaxy_id": row["sparc_galaxy_id"],
                "sparc_name_normalized": row["sparc_name_normalized"],
                "sep_deg": float(row["sep_deg"]),
                "catalog_prefix": row["catalog_prefix"],
                "catalog_number": row["catalog_number"],
            }
        )
    return out



def build_review_candidates(alias_seed: pd.DataFrame, sparc_direct: pd.DataFrame) -> pd.DataFrame:
    alias_df = alias_seed.copy()
    sparc_df = prepare_sparc_catalog(sparc_direct)

    for col in STRING_COLUMNS:
        ensure_string_column(alias_df, col)
    for col in FLOAT_COLUMNS:
        ensure_float_column(alias_df, col)
    for col in INTLIKE_COLUMNS:
        ensure_intlike_column(alias_df, col)

    # ----- preserve existing names and compute normalized names -----
    if "diskmass_display_name" not in alias_df.columns:
        source_name_col = first_existing(NAME_CANDIDATE_COLUMNS, alias_df)
        if source_name_col is not None:
            alias_df["diskmass_display_name"] = alias_df[source_name_col].astype("string")
        else:
            alias_df["diskmass_display_name"] = pd.Series(pd.NA, index=alias_df.index, dtype="string")
    else:
        alias_df["diskmass_display_name"] = alias_df["diskmass_display_name"].astype("string")

    if "diskmass_name_normalized" not in alias_df.columns:
        alias_df["diskmass_name_normalized"] = alias_df["diskmass_display_name"].map(normalize_name).astype("string")
    else:
        alias_df["diskmass_name_normalized"] = alias_df["diskmass_name_normalized"].astype("string")
        missing = alias_df["diskmass_name_normalized"].isna() | (alias_df["diskmass_name_normalized"].str.len() == 0)
        alias_df.loc[missing, "diskmass_name_normalized"] = alias_df.loc[missing, "diskmass_display_name"].map(normalize_name)

    # ----- compute DiskMass coordinates from multiple possible sources -----
    alias_df["dm_ra_deg"] = alias_df.apply(lambda row: first_valid_ra(row, DM_RA_CANDIDATE_COLUMNS), axis=1)
    alias_df["dm_dec_deg"] = alias_df.apply(lambda row: first_valid_dec(row, DM_DEC_CANDIDATE_COLUMNS), axis=1)

    add_name_hints(alias_df, sparc_df)

    name_to_row = {
        normalize_name(name): row
        for _, row in sparc_df.iterrows()
        for name in [row["sparc_galaxy_id"]]
        if normalize_name(name)
    }

    if "sparc_reference_note" not in alias_df.columns or alias_df["sparc_reference_note"].isna().all():
        note = " | ".join(sorted(sparc_df["sparc_galaxy_id"].astype("string").dropna().unique().tolist()))
        alias_df["sparc_reference_note"] = note

    # ----- row-by-row candidate filling -----
    for idx, row in alias_df.iterrows():
        hint_norm = normalize_name(row.get("auto_exact_name_hint"))
        dm_ra = float(row["dm_ra_deg"]) if pd.notna(row["dm_ra_deg"]) else math.nan
        dm_dec = float(row["dm_dec_deg"]) if pd.notna(row["dm_dec_deg"]) else math.nan

        coord_candidates = build_coord_candidates(dm_ra, dm_dec, sparc_df)
        chosen_candidates = coord_candidates.copy()

        # If an exact name hint exists but coordinates are missing or do not include it,
        # inject that candidate at the front so later review can still use it.
        if hint_norm and hint_norm in name_to_row:
            hint_row = name_to_row[hint_norm]
            hint_sep = angular_sep_deg(
                dm_ra,
                dm_dec,
                float(hint_row["sparc_ra_deg"]),
                float(hint_row["sparc_dec_deg"]),
            )
            hint_candidate = {
                "sparc_galaxy_id": hint_row["sparc_galaxy_id"],
                "sparc_name_normalized": hint_row["sparc_name_normalized"],
                "sep_deg": float(hint_sep) if not math.isnan(hint_sep) else math.nan,
                "catalog_prefix": hint_row["catalog_prefix"],
                "catalog_number": hint_row["catalog_number"],
            }
            present_names = {normalize_name(c["sparc_galaxy_id"]) for c in chosen_candidates}
            if hint_norm not in present_names:
                chosen_candidates = [hint_candidate] + chosen_candidates
            else:
                chosen_candidates = sorted(
                    chosen_candidates,
                    key=lambda c: (0 if normalize_name(c["sparc_galaxy_id"]) == hint_norm else 1, c["sep_deg"] if not math.isnan(c["sep_deg"]) else 999.0),
                )

        # keep unique by normalized galaxy id
        deduped: list[dict[str, object]] = []
        seen: set[str] = set()
        for cand in chosen_candidates:
            key = normalize_name(cand["sparc_galaxy_id"])
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(cand)
        chosen_candidates = deduped[:TOP_N_CANDIDATES]

        for n in range(1, TOP_N_CANDIDATES + 1):
            alias_df.at[idx, f"candidate_{n}_sparc_galaxy_id"] = pd.NA
            alias_df.at[idx, f"candidate_{n}_sep_deg"] = np.nan
            alias_df.at[idx, f"candidate_{n}_catalog_prefix"] = pd.NA
            alias_df.at[idx, f"candidate_{n}_catalog_number"] = np.nan

        for n, cand in enumerate(chosen_candidates, start=1):
            alias_df.at[idx, f"candidate_{n}_sparc_galaxy_id"] = cand["sparc_galaxy_id"]
            alias_df.at[idx, f"candidate_{n}_sep_deg"] = cand["sep_deg"]
            alias_df.at[idx, f"candidate_{n}_catalog_prefix"] = cand["catalog_prefix"]
            alias_df.at[idx, f"candidate_{n}_catalog_number"] = cand["catalog_number"]

        alias_df.at[idx, "coord_candidate_count"] = float(len(chosen_candidates))

        if chosen_candidates:
            best = chosen_candidates[0]
            alias_df.at[idx, "candidate_sparc_galaxy_id"] = best["sparc_galaxy_id"]
            alias_df.at[idx, "candidate_sparc_name_normalized"] = best["sparc_name_normalized"]
            alias_df.at[idx, "best_coord_sep_deg"] = best["sep_deg"]
            alias_df.at[idx, "alias_source"] = "coord_candidate_1"
            alias_df.at[idx, "alias_match_status"] = "pending_review"

            sep = best["sep_deg"]
            if not math.isnan(sep) and sep <= MAX_COORD_MATCH_DEG:
                alias_df.at[idx, "coord_alias_recommendation"] = "coord_candidate_within_threshold"
            elif hint_norm and normalize_name(best["sparc_galaxy_id"]) == hint_norm:
                alias_df.at[idx, "coord_alias_recommendation"] = "exact_name_hint_outside_threshold"
            else:
                alias_df.at[idx, "coord_alias_recommendation"] = "review_candidate_found"
        else:
            alias_df.at[idx, "candidate_sparc_galaxy_id"] = pd.NA
            alias_df.at[idx, "candidate_sparc_name_normalized"] = pd.NA
            alias_df.at[idx, "best_coord_sep_deg"] = np.nan
            alias_df.at[idx, "alias_source"] = pd.NA
            alias_df.at[idx, "alias_match_status"] = "pending_review"
            if math.isnan(dm_ra) or math.isnan(dm_dec):
                if hint_norm:
                    alias_df.at[idx, "coord_alias_recommendation"] = "exact_name_hint_only"
                    alias_df.at[idx, "candidate_sparc_galaxy_id"] = name_to_row[hint_norm]["sparc_galaxy_id"]
                    alias_df.at[idx, "candidate_sparc_name_normalized"] = hint_norm
                    alias_df.at[idx, "coord_candidate_count"] = 1.0
                    alias_df.at[idx, "candidate_1_sparc_galaxy_id"] = name_to_row[hint_norm]["sparc_galaxy_id"]
                    alias_df.at[idx, "candidate_1_catalog_prefix"] = name_to_row[hint_norm]["catalog_prefix"]
                    alias_df.at[idx, "candidate_1_catalog_number"] = name_to_row[hint_norm]["catalog_number"]
                    alias_df.at[idx, "alias_source"] = "auto_exact_name_hint"
                else:
                    alias_df.at[idx, "coord_alias_recommendation"] = "no_diskmass_coords"
            else:
                alias_df.at[idx, "coord_alias_recommendation"] = "no_candidate_within_threshold"

        # Helpful human-readable notes.
        if alias_df.at[idx, "coord_alias_recommendation"] == "exact_name_hint_only":
            alias_df.at[idx, "review_notes"] = "Exact catalog-name hint exists in SPARC, but DiskMass coordinates could not be parsed."
        elif alias_df.at[idx, "coord_alias_recommendation"] == "exact_name_hint_outside_threshold":
            alias_df.at[idx, "review_notes"] = "Exact catalog-name hint exists, but coordinate separation is larger than the conservative auto-confirm threshold."
        elif alias_df.at[idx, "coord_alias_recommendation"] == "review_candidate_found":
            alias_df.at[idx, "review_notes"] = "Coordinate candidate found only in broader review radius; manual inspection recommended."
        else:
            alias_df.at[idx, "review_notes"] = alias_df.at[idx, "review_notes"]

    return alias_df


# ----- CLI -----
def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DiskMass alias review candidates from alias seed and SPARC/S4G direct crossmatch tables."
    )
    parser.add_argument("--alias-seed", required=True)
    parser.add_argument("--sparc-s4g-direct", required=True)
    parser.add_argument("--output-file", required=True)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    alias_seed = pd.read_csv(args.alias_seed)
    sparc_direct = pd.read_csv(args.sparc_s4g_direct)
    out = build_review_candidates(alias_seed, sparc_direct)
    write_csv(out, args.output_file)
    print(f"[OK] wrote {len(out)} rows -> {args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
