from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NAME_KEYS = [
    "diskmass_name",
    "diskmass_galaxy",
    "galaxy",
    "name",
]

MATCH_NAME_KEYS = [
    "diskmass_name",
    "diskmass_galaxy",
    "galaxy",
    "name",
    "sparc_name",
    "s4g_name",
]

SEP_KEYS = [
    "best_coord_sep_deg",
    "coord_sep_deg",
    "sep_deg",
    "separation_deg",
    "ang_sep_deg",
]

STRING_OUTPUT_COLUMNS = [
    "diskmass_name",
    "coord_candidate_1",
    "coord_candidate_2",
    "coord_candidate_3",
    "best_candidate_name",
    "candidate_count",
]



def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    for ch in [" ", "-", "_", ".", "/", "(", ")"]:
        text = text.replace(ch, "")
    return text



def first_existing(columns: Iterable[str], df: pd.DataFrame) -> str | None:
    for col in columns:
        if col in df.columns:
            return col
    return None



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



def prepare_seed(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    name_col = first_existing(NAME_KEYS, seed_df)
    if name_col is None:
        raise ValueError(
            f"Alias seed CSV must contain one of these columns: {NAME_KEYS}"
        )
    out = seed_df.copy()
    ensure_string_column(out, name_col)
    out["_join_key"] = out[name_col].map(normalize_name)
    out = out.drop_duplicates(subset=["_join_key"], keep="first").reset_index(drop=True)
    return out, name_col



def prepare_direct(direct_df: pd.DataFrame) -> tuple[pd.DataFrame, str, str | None]:
    name_col = first_existing(MATCH_NAME_KEYS, direct_df)
    if name_col is None:
        raise ValueError(
            f"Direct crossmatch CSV must contain one of these columns: {MATCH_NAME_KEYS}"
        )
    sep_col = first_existing(SEP_KEYS, direct_df)
    out = direct_df.copy()
    ensure_string_column(out, name_col)
    out["_join_key"] = out[name_col].map(normalize_name)
    if sep_col is not None:
        ensure_float_column(out, sep_col)
        out = out.sort_values(by=[sep_col, name_col], na_position="last").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out, name_col, sep_col



def build_review_candidates(alias_seed: pd.DataFrame, sparc_direct: pd.DataFrame) -> pd.DataFrame:
    alias_df, alias_name_col = prepare_seed(alias_seed)
    direct_df, direct_name_col, sep_col = prepare_direct(sparc_direct)

    for col in STRING_OUTPUT_COLUMNS:
        ensure_string_column(alias_df, col)
    ensure_float_column(alias_df, "best_coord_sep_deg")

    grouped = {k: g.copy() for k, g in direct_df.groupby("_join_key", dropna=False)}

    for idx, row in alias_df.iterrows():
        join_key = row["_join_key"]
        candidates = grouped.get(join_key)
        if candidates is None or candidates.empty:
            alias_df.at[idx, "candidate_count"] = "0"
            continue

        names = (
            candidates[direct_name_col]
            .astype("string")
            .dropna()
            .drop_duplicates()
            .tolist()
        )
        candidate_names = names[:3]
        alias_df.at[idx, "candidate_count"] = str(len(names))

        for n, candidate_name in enumerate(candidate_names, start=1):
            alias_df.at[idx, f"coord_candidate_{n}"] = candidate_name

        if candidate_names:
            alias_df.at[idx, "best_candidate_name"] = candidate_names[0]

        if sep_col is not None and sep_col in candidates.columns:
            best_sep = pd.to_numeric(candidates.iloc[0][sep_col], errors="coerce")
            if pd.notna(best_sep):
                alias_df.at[idx, "best_coord_sep_deg"] = float(best_sep)

    alias_df.rename(columns={alias_name_col: "diskmass_name"}, inplace=True)
    alias_df.drop(columns=[c for c in ["_join_key"] if c in alias_df.columns], inplace=True)

    ordered_front = [
        "diskmass_name",
        "best_candidate_name",
        "best_coord_sep_deg",
        "candidate_count",
        "coord_candidate_1",
        "coord_candidate_2",
        "coord_candidate_3",
    ]
    remaining = [c for c in alias_df.columns if c not in ordered_front]
    return alias_df[ordered_front + remaining]



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
