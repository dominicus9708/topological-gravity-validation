from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STRING_COLUMNS = [
    "diskmass_name",
    "diskmass_display_name",
    "sparc_name",
    "best_candidate_name",
    "coord_candidate_1",
    "coord_candidate_2",
    "coord_candidate_3",
    "candidate_1_catalog_prefix",
    "candidate_2_catalog_prefix",
    "candidate_3_catalog_prefix",
    "alias_source",
    "auto_decision",
    "review_reason",
    "coord_alias_recommendation",
]

FLOAT_COLUMNS = [
    "best_coord_sep_deg",
    "coord_sep_deg",
    "coord_sep_arcmin",
    "coord_sep_arcsec",
    "candidate_1_sep_deg",
    "candidate_2_sep_deg",
    "candidate_3_sep_deg",
    "score",
]

INTLIKE_COLUMNS = [
    "coord_candidate_count",
    "candidate_count",
    "candidate_1_catalog_number",
    "candidate_2_catalog_number",
    "candidate_3_catalog_number",
]

BOOLEAN_COLUMNS = [
    "is_confirmed",
    "needs_review",
    "is_weak_or_none",
]


AUTO_CONFIRM_MAX_SEP_DEG = 0.05


def ensure_string_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df[col] = df[col].astype("string")
    return df



def ensure_float_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = pd.Series(np.nan, index=df.index, dtype="float64")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def ensure_intlike_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = pd.Series(np.nan, index=df.index, dtype="float64")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def ensure_bool_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = pd.Series(False, index=df.index, dtype="boolean")
    else:
        s = df[col]
        if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            df[col] = s.astype("boolean")
        else:
            normalized = (
                s.astype("string")
                .str.strip()
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                        "y": True,
                        "n": False,
                    }
                )
            )
            df[col] = normalized.fillna(False).astype("boolean")
    return df



def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in STRING_COLUMNS:
        if col in out.columns or col in {"alias_source", "auto_decision", "review_reason", "coord_alias_recommendation"}:
            ensure_string_column(out, col)

    for col in FLOAT_COLUMNS:
        if col in out.columns or col == "best_coord_sep_deg":
            ensure_float_column(out, col)

    for col in INTLIKE_COLUMNS:
        if col in out.columns:
            ensure_intlike_column(out, col)

    for col in BOOLEAN_COLUMNS:
        if col in out.columns:
            ensure_bool_column(out, col)

    ensure_string_column(out, "alias_source")
    ensure_string_column(out, "auto_decision")
    ensure_string_column(out, "review_reason")
    ensure_string_column(out, "coord_alias_recommendation")
    ensure_bool_column(out, "is_confirmed")
    ensure_bool_column(out, "needs_review")
    ensure_bool_column(out, "is_weak_or_none")
    ensure_float_column(out, "best_coord_sep_deg")
    return out



def resolve_sep_series(df: pd.DataFrame) -> pd.Series:
    priority = [
        "best_coord_sep_deg",
        "candidate_1_sep_deg",
        "coord_sep_deg",
        "coord_sep_arcmin",
        "coord_sep_arcsec",
    ]

    sep_deg = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in priority:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if col == "coord_sep_arcmin":
            s = s / 60.0
        elif col == "coord_sep_arcsec":
            s = s / 3600.0
        sep_deg = sep_deg.where(sep_deg.notna(), s)
    return sep_deg



def resolve_candidate_count(df: pd.DataFrame) -> pd.Series:
    if "coord_candidate_count" in df.columns:
        count = pd.to_numeric(df["coord_candidate_count"], errors="coerce")
    elif "candidate_count" in df.columns:
        count = pd.to_numeric(df["candidate_count"], errors="coerce")
    else:
        count = pd.Series(np.nan, index=df.index, dtype="float64")

    fallback_cols = [
        c for c in [
            "candidate_1_sparc_galaxy_id",
            "candidate_2_sparc_galaxy_id",
            "candidate_3_sparc_galaxy_id",
            "coord_candidate_1",
            "coord_candidate_2",
            "coord_candidate_3",
            "best_candidate_name",
            "sparc_name",
        ] if c in df.columns
    ]

    if fallback_cols:
        fallback_count = df[fallback_cols].astype("string").notna().sum(axis=1).astype("float64")
        count = count.where(count.notna(), fallback_count)

    return count.fillna(0.0)



def resolve_has_candidate(df: pd.DataFrame, candidate_count: pd.Series) -> pd.Series:
    has_candidate = candidate_count > 0

    if "candidate_1_sparc_galaxy_id" in df.columns:
        has_candidate = has_candidate | df["candidate_1_sparc_galaxy_id"].astype("string").notna()
    if "coord_candidate_1" in df.columns:
        has_candidate = has_candidate | df["coord_candidate_1"].astype("string").notna()
    if "best_candidate_name" in df.columns:
        has_candidate = has_candidate | df["best_candidate_name"].astype("string").notna()

    return has_candidate.fillna(False)



def resolve_alias_source(df: pd.DataFrame) -> pd.Series:
    source = pd.Series(pd.NA, index=df.index, dtype="string")

    if "candidate_1_sparc_galaxy_id" in df.columns:
        mask = df["candidate_1_sparc_galaxy_id"].astype("string").notna()
        source = source.where(~mask, "coord_candidate_1")

    if "best_candidate_name" in df.columns:
        mask = source.isna() & df["best_candidate_name"].astype("string").notna()
        source = source.where(~mask, "best_candidate_name")

    if "sparc_name" in df.columns:
        mask = source.isna() & df["sparc_name"].astype("string").notna()
        source = source.where(~mask, "sparc_name")

    return source



def build_alias_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = normalize_dataframe(df)

    sep_deg = resolve_sep_series(out)
    candidate_count = resolve_candidate_count(out)
    has_candidate = resolve_has_candidate(out, candidate_count)
    recommendation = out["coord_alias_recommendation"].astype("string").str.strip().str.lower()

    weak_reasons = recommendation.isin({
        "no_diskmass_coords",
        "no_candidate_within_threshold",
        "no_viable_candidate",
        "weak_or_none",
    })

    strong_single_candidate = has_candidate & (candidate_count == 1)
    close_enough = sep_deg.notna() & (sep_deg <= AUTO_CONFIRM_MAX_SEP_DEG)

    confirmed_mask = strong_single_candidate & close_enough & (~weak_reasons)
    review_mask = has_candidate & (~confirmed_mask) & (~weak_reasons)
    weak_mask = (~has_candidate) | weak_reasons

    out["best_coord_sep_deg"] = sep_deg
    out["alias_source"] = out["alias_source"].where(out["alias_source"].notna(), resolve_alias_source(out))

    out.loc[confirmed_mask, "auto_decision"] = "confirmed"
    out.loc[confirmed_mask, "review_reason"] = pd.NA

    out.loc[review_mask, "auto_decision"] = "review_needed"
    out.loc[review_mask, "review_reason"] = out.loc[review_mask, "review_reason"].where(
        out.loc[review_mask, "review_reason"].notna(),
        "candidate_exists_but_not_auto_confirmed",
    )

    out.loc[weak_mask, "auto_decision"] = "weak_or_none"
    out.loc[weak_mask, "review_reason"] = out.loc[weak_mask, "review_reason"].where(
        out.loc[weak_mask, "review_reason"].notna(),
        "no_viable_candidate",
    )

    out["is_confirmed"] = confirmed_mask.astype("boolean")
    out["needs_review"] = review_mask.astype("boolean")
    out["is_weak_or_none"] = weak_mask.astype("boolean")

    confirmed_df = out.loc[confirmed_mask].copy()
    review_df = out.loc[review_mask].copy()
    weak_df = out.loc[weak_mask].copy()
    return confirmed_df, review_df, weak_df



def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split DiskMass alias review candidates into auto-confirmed, review-needed, and weak/none outputs."
    )
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--confirmed-file", required=True)
    parser.add_argument("--review-file", required=True)
    parser.add_argument("--weak-file", required=True)
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input_file)
    confirmed_df, review_df, weak_df = build_alias_outputs(df)

    write_csv(confirmed_df, args.confirmed_file)
    write_csv(review_df, args.review_file)
    write_csv(weak_df, args.weak_file)

    print(
        f"[OK] confirmed={len(confirmed_df)} review_needed={len(review_df)} weak_or_none={len(weak_df)}"
    )
    print(f"[OK] confirmed file: {args.confirmed_file}")
    print(f"[OK] review file: {args.review_file}")
    print(f"[OK] weak/no-candidate file: {args.weak_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
