from __future__ import annotations
# -----
# normalize_diskmass_survey1_fixed.py
#
# DiskMass Survey I 전용 정규화 스크립트
# -----

import argparse
import csv
from pathlib import Path

DEFAULT_INPUT = Path(r"data/raw/diskmass/J_ApJ_716_198_The_ DiskMass_survey_I.tsv")
DEFAULT_OUTPUT = Path(r"data/derived/diskmass/J_ApJ_716_198_The_ DiskMass_survey_I.csv")

HEADER_LINE_1BASED = 132
DATA_LINE_1BASED = 135

def split_line(line: str) -> list[str]:
    line = line.rstrip("\n\r")
    if ";" in line:
        return [c.strip() for c in line.split(";")]
    if "\t" in line:
        return [c.strip() for c in line.split("\t")]
    return [c.strip() for c in line.split(",")]

def clean_header(cells: list[str]) -> list[str]:
    out = []
    used = {}
    for i, c in enumerate(cells, 1):
        name = c.strip() or f"col_{i}"
        name = name.replace(" ", "_").replace("/", "_")
        name = name.replace("(", "").replace(")", "")
        name = name.replace("[", "").replace("]", "")
        if name in used:
            used[name] += 1
            name = f"{name}_{used[name]}"
        else:
            used[name] = 1
        out.append(name)
    return out

def normalize(input_file: Path, output_file: Path) -> Path:
    lines = input_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    header = clean_header(split_line(lines[HEADER_LINE_1BASED - 1]))
    ncols = len(header)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for raw in lines[DATA_LINE_1BASED - 1:]:
            s = raw.strip()
            if not s:
                continue
            up = s.upper()
            if "]]></CSV>" in up or "</CSV>" in up or "</DATA>" in up:
                break
            if s.startswith("<"):
                continue
            cells = split_line(raw)
            if len(cells) < ncols:
                cells += [""] * (ncols - len(cells))
            elif len(cells) > ncols:
                cells = cells[:ncols]
            writer.writerow(cells)
    return output_file

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = normalize(args.input_file, args.output_file)
    print(f"[OK] normalized Survey I file -> {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
