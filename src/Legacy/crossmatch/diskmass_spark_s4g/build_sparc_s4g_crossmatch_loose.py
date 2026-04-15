from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def norm_basic(name: object) -> str:
    if name is None:
        return ""
    text = str(name).strip().upper()
    if text == "" or text == "NAN":
        return ""
    return re.sub(r"[^A-Z0-9]", "", text)


def strip_leading_zeros_after_prefix(text: str) -> str:
    """
    Convert common catalog names to a looser comparison form.

    Examples:
      UGC00128 -> UGC128
      DDO064   -> DDO64
      NGC0024  -> NGC24
      ESO079G014 -> ESO79G14
      IC2574 -> IC2574 (unchanged enough)
    """
    m = re.match(r"^([A-Z]+)([0-9A-Z]+)$", text)
    if not m:
        return text

    prefix, tail = m.groups()

    # Split tail into digit groups / alpha groups
    parts = re.findall(r"[0-9]+|[A-Z]+", tail)
    new_parts = []
    for part in parts:
        if part.isdigit():
            stripped = part.lstrip("0")
            new_parts.append(stripped if stripped != "" else "0")
        else:
            new_parts.append(part)
    return prefix + "".join(new_parts)


def norm_loose(name: object) -> str:
    return strip_leading_zeros_after_prefix(norm_basic(name))


def choose_s4g_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "Name", "RAJ2000", "DEJ2000", "amaj", "PA", "ell",
        "3.6", "e_3.6", "4.5", "e_4.5", "c31", "c42",
        "M3.6", "M4.5", "logM*", "Dmean", "e_Dmean",
        "TT", "e_TT", "Vr", "e_Vr"
    ]
    return [c for c in preferred if c in df.columns]


def build_loose_match(
    unmatched_input: Path,
    s4g_catalog: Path,
    output_file: Path,
    still_unmatched_file: Optional[Path] = None,
) -> tuple[Path, Optional[Path], int, int]:
    sparc = pd.read_csv(unmatched_input)
    s4g = pd.read_csv(s4g_catalog)

    if "galaxy_id" not in sparc.columns:
        raise ValueError("Input unmatched SPARC file must contain 'galaxy_id'.")
    if "Name" not in s4g.columns:
        raise ValueError("S4G catalog must contain 'Name'.")

    sparc = sparc.copy()
    s4g = s4g.copy()

    sparc["galaxy_name_loose"] = sparc["galaxy_id"].map(norm_loose)
    s4g["s4g_name_loose"] = s4g["Name"].map(norm_loose)

    keep_cols = choose_s4g_columns(s4g)
    s4g_small = s4g[["s4g_name_loose"] + keep_cols].copy()

    merged = sparc.merge(
        s4g_small,
        left_on="galaxy_name_loose",
        right_on="s4g_name_loose",
        how="left",
    )

    merged["match_status"] = merged["Name"].notna().map(lambda x: "matched_loose" if x else "still_unmatched")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False)

    matched_count = int((merged["match_status"] == "matched_loose").sum())
    unmatched_count = int((merged["match_status"] == "still_unmatched").sum())

    still_path = None
    if still_unmatched_file is not None:
        still_path = still_unmatched_file
        still_unmatched_file.parent.mkdir(parents=True, exist_ok=True)
        merged.loc[merged["match_status"] == "still_unmatched"].to_csv(still_unmatched_file, index=False)

    return output_file, still_path, matched_count, unmatched_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Second-pass loose-name SPARC↔S4G crossmatch.")
    parser.add_argument(
        "--unmatched-input",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_s4g_unmatched_direct.csv"),
        help="Unmatched SPARC list from the direct-name pass.",
    )
    parser.add_argument(
        "--s4g-catalog",
        type=Path,
        default=Path("data/derived/s4g/S4G_catalog.csv"),
        help="Normalized S4G catalog CSV.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_s4g_crossmatch_loose.csv"),
        help="Output CSV for loose-name matches.",
    )
    parser.add_argument(
        "--still-unmatched-file",
        type=Path,
        default=Path("data/derived/crossmatch/sparc_s4g_unmatched_loose.csv"),
        help="Output CSV for SPARC galaxies still unmatched after loose-name pass.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path, still_path, matched_count, unmatched_count = build_loose_match(
        args.unmatched_input,
        args.s4g_catalog,
        args.output_file,
        args.still_unmatched_file,
    )
    print(f"[OK] loose SPARC↔S4G crossmatch written: {out_path}")
    print(f"[OK] matched_loose={matched_count} still_unmatched={unmatched_count}")
    if still_path is not None:
        print(f"[OK] remaining unmatched list written: {still_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
