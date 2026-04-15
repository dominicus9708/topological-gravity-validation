#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
WISE H II bootstrap downloader

Purpose
-------
1) Download the core WISE H II raw materials for the project's baseline validation.
2) Save them under data/raw/Validation of Structural Contrast Baseline/
3) Build simple candidate tables for first-pass target selection.

Notes
-----
- This script intentionally focuses on the most stable first step:
  catalog + reference docs + candidate shortlist tables.
- It does NOT attempt to perform morphology recognition automatically.
  That is better handled after a first human pass over the shortlist.
- Designed for Windows 11 + standard Python environment.
"""

from __future__ import annotations

import csv
import io
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# --------------------------------------------------
# User-adjustable defaults
# --------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")
RAW_SUBDIR = Path("data") / "raw" / "Validation of Structural Contrast Baseline"
SCRIPT_SUBDIR = RAW_SUBDIR / "script"
WISE_SUBDIR = RAW_SUBDIR / "wise_hii_catalog"
DOC_SUBDIR = WISE_SUBDIR / "docs"
CAT_SUBDIR = WISE_SUBDIR / "catalog"
CANDIDATE_SUBDIR = WISE_SUBDIR / "candidate_lists"

URLS: Dict[str, str] = {
    "wise_catalog_v2_3_csv": "https://astro.phys.wvu.edu/wise/wise_hii_V2.3.csv",
    "wise_catalog_v1_3_csv": "https://astro.phys.wvu.edu/wise/wise_hii_V1.3.csv",
    "wise_catalog_paper_pdf": "https://astro.phys.wvu.edu/wise/wise_catalog_hii_regions.pdf",
    "wise_table2_txt": "https://astro.phys.wvu.edu/wise/Table2.txt",
    "wise_flux_densities_pdf": "https://astro.phys.wvu.edu/wise/flux_densities.pdf",
    "wise_site_index_html": "https://astro.phys.wvu.edu/wise/",
}

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-WISE-HII-Bootstrap/1.0"


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def fetch_bytes(url: str, timeout: int = 60) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def save_bytes(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_bytes(url)
    out_path.write_bytes(data)
    print(f"[OK] {out_path}")


def save_text(url: str, out_path: Path, encoding: str = "utf-8") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_bytes(url)
    out_path.write_text(data.decode(encoding, errors="replace"), encoding="utf-8")
    print(f"[OK] {out_path}")


def try_download(url: str, out_path: Path, kind: str) -> Tuple[bool, str]:
    try:
        if kind == "text":
            save_text(url, out_path)
        else:
            save_bytes(url, out_path)
        return True, "ok"
    except HTTPError as e:
        return False, f"HTTPError {e.code}"
    except URLError as e:
        return False, f"URLError {e.reason}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def sniff_delimiter(header_line: str) -> str:
    candidates = [",", "\t", ";", "|"]
    counts = {c: header_line.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def load_csv_rows(csv_path: Path) -> List[dict]:
    raw_text = csv_path.read_text(encoding="utf-8", errors="replace")
    first_line = raw_text.splitlines()[0]
    delim = sniff_delimiter(first_line)
    reader = csv.DictReader(io.StringIO(raw_text), delimiter=delim)
    rows = []
    for row in reader:
        cleaned = {}
        for k, v in row.items():
            ck = (k or "").replace("<br>", " ").replace("\n", " ").strip()
            cleaned[ck] = (v or "").strip()
        rows.append(cleaned)
    return rows


def normalize_columns(rows: List[dict]) -> List[dict]:
    """
    Harmonize likely column names across WISE catalog versions.
    """
    out = []
    for row in rows:
        new = dict(row)

        # Preferred standard names
        mapping_candidates = {
            "wise_name": ["WISE Name", "WISE", "WISE Name "],
            "catalog_class": ["Catalog", "Class", "Classification"],
            "glon": ["GLong", "WGLON", "GLong (deg.)", "GLong (deg.) "],
            "glat": ["GLat", "WGLAT", "GLat (deg.)", "GLat (deg.) "],
            "ra": ["RA"],
            "dec": ["Dec", "DEC"],
            "radius_arcsec": ["Radius", "R", "Radius (arcsec.)", "Radius (arcsec.) "],
            "hii_region_name": ["HII Region", "HIIName"],
            "membership": ["Membership", "Mem", "Group membership identifier"],
            "vlsr": ["VLSR", "VLSR (Mol.)"],
            "dist_kpc": ["Dist.", "Dist", "Dist. "],
            "w3_jy": ["WISE 12um (Jy)", "WISE 12um", "WISE 12μm (Jy)"],
            "w4_jy": ["WISE 22um (Jy)", "WISE 22um", "WISE 22μm (Jy)"],
            "mips24_jy": ["MIPSGAL 24um (Jy)", "MIPSGAL 24μm (Jy)"],
        }

        def pick(names: List[str]) -> str:
            for n in names:
                if n in row and row[n] != "":
                    return row[n]
            return ""

        normalized = {
            "wise_name": pick(mapping_candidates["wise_name"]),
            "catalog_class": pick(mapping_candidates["catalog_class"]),
            "glon": pick(mapping_candidates["glon"]),
            "glat": pick(mapping_candidates["glat"]),
            "ra": pick(mapping_candidates["ra"]),
            "dec": pick(mapping_candidates["dec"]),
            "radius_arcsec": pick(mapping_candidates["radius_arcsec"]),
            "hii_region_name": pick(mapping_candidates["hii_region_name"]),
            "membership": pick(mapping_candidates["membership"]),
            "vlsr": pick(mapping_candidates["vlsr"]),
            "dist_kpc": pick(mapping_candidates["dist_kpc"]),
            "w3_jy": pick(mapping_candidates["w3_jy"]),
            "w4_jy": pick(mapping_candidates["w4_jy"]),
            "mips24_jy": pick(mapping_candidates["mips24_jy"]),
        }
        out.append(normalized)
    return out


def safe_float(x: str) -> float | None:
    if x is None:
        return None
    x = x.strip()
    if x == "":
        return None
    x = x.replace(",", "")
    # extract first valid float
    m = re.search(r"[-+]?\d+(?:\.\d+)?", x)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[OK] {path}")


def build_candidate_lists(rows: List[dict], out_dir: Path) -> None:
    """
    Build practical shortlist tables for first-pass visual inspection.

    Strategy:
    - Prefer known/group/classified entries over generic candidates.
    - Prefer moderate angular size for simple shell-like baseline tests.
    - Keep columns small and human-readable.
    """
    enriched = []
    for r in rows:
        radius = safe_float(r.get("radius_arcsec", ""))
        glon = safe_float(r.get("glon", ""))
        glat = safe_float(r.get("glat", ""))
        dist = safe_float(r.get("dist_kpc", ""))
        w4 = safe_float(r.get("w4_jy", ""))
        w3 = safe_float(r.get("w3_jy", ""))
        m24 = safe_float(r.get("mips24_jy", ""))

        cat = (r.get("catalog_class", "") or "").strip().upper()
        # Heuristic priority:
        # K/G are generally safer than faint/ambiguous candidates for first pass.
        if cat.startswith("K"):
            priority = 1
        elif cat.startswith("G"):
            priority = 2
        elif cat.startswith("C"):
            priority = 3
        elif cat.startswith("Q"):
            priority = 4
        else:
            priority = 5

        size_bucket = ""
        if radius is None:
            size_bucket = "unknown"
        elif 60 <= radius <= 300:
            size_bucket = "preferred_simple_shell_range"
        elif 30 <= radius < 60 or 300 < radius <= 600:
            size_bucket = "usable"
        else:
            size_bucket = "edge_case"

        enriched.append({
            **r,
            "radius_arcsec_num": radius if radius is not None else "",
            "glon_num": glon if glon is not None else "",
            "glat_num": glat if glat is not None else "",
            "dist_kpc_num": dist if dist is not None else "",
            "w3_jy_num": w3 if w3 is not None else "",
            "w4_jy_num": w4 if w4 is not None else "",
            "mips24_jy_num": m24 if m24 is not None else "",
            "priority_rank": priority,
            "size_bucket": size_bucket,
        })

    base_fields = [
        "wise_name", "catalog_class", "glon", "glat", "ra", "dec",
        "radius_arcsec", "hii_region_name", "membership",
        "dist_kpc", "w3_jy", "w4_jy", "mips24_jy",
        "priority_rank", "size_bucket"
    ]

    # 1) All normalized rows
    write_csv(out_dir / "wise_hii_normalized_full.csv", enriched, base_fields)

    # 2) First-pass shortlist: known/group + moderate size
    shortlist = [
        r for r in enriched
        if r["priority_rank"] <= 2 and r["size_bucket"] == "preferred_simple_shell_range"
    ]
    shortlist.sort(key=lambda x: (
        x["priority_rank"],
        safe_float(str(x["radius_arcsec_num"])) if x["radius_arcsec_num"] != "" else 999999,
        x["wise_name"]
    ))
    write_csv(out_dir / "wise_hii_shortlist_known_group_simple_shell.csv", shortlist, base_fields)

    # 3) Backup shortlist: candidates with moderate size
    shortlist_candidates = [
        r for r in enriched
        if r["priority_rank"] == 3 and r["size_bucket"] == "preferred_simple_shell_range"
    ]
    shortlist_candidates.sort(key=lambda x: (
        safe_float(str(x["radius_arcsec_num"])) if x["radius_arcsec_num"] != "" else 999999,
        x["wise_name"]
    ))
    write_csv(out_dir / "wise_hii_shortlist_candidate_simple_shell.csv", shortlist_candidates, base_fields)

    # 4) Size-focused table
    size_sorted = sorted(
        enriched,
        key=lambda x: safe_float(str(x["radius_arcsec_num"])) if x["radius_arcsec_num"] != "" else 999999
    )
    write_csv(out_dir / "wise_hii_sorted_by_radius.csv", size_sorted, base_fields)

    # 5) Small README for what to inspect first
    readme = (
        "WISE H II bootstrap candidate notes\n"
        "===================================\n\n"
        "Recommended first-pass visual selection rule:\n"
        "1. Start from wise_hii_shortlist_known_group_simple_shell.csv\n"
        "2. Prefer objects with a single, clean ring/shell morphology\n"
        "3. Avoid obviously overlapping complexes in the first test\n"
        "4. Avoid highly irregular fragmented regions in the first test\n"
        "5. Keep 3 final candidates for manual inspection\n\n"
        "Suggested first validation goal:\n"
        "- Build a radial profile from one clean shell-like target\n"
        "- Define outer background ring\n"
        "- Check whether sigma separates shell region from background\n"
    )
    (out_dir / "README_first_pass.txt").write_text(readme, encoding="utf-8")
    print(f"[OK] {out_dir / 'README_first_pass.txt'}")


def write_project_notes(root: Path) -> None:
    notes = f"""Validation of Structural Contrast Baseline
=========================================

This raw-data bootstrap is designed for the 4th paper's foundational validation stage.

Primary source family
---------------------
- WISE Catalog of Galactic H II Regions
- Core use: identify clean shell-like or ring-like structures for sigma baseline tests

Why this source is enough for first-pass validation
---------------------------------------------------
- very large target pool
- shell/ring morphology is common
- suited to background-vs-structure separation
- can support a simple radial-profile test before more complex observational cases

Folder intent
-------------
- docs/      : papers and source descriptions
- catalog/   : original downloaded catalog files
- candidate_lists/ : normalized and filtered tables for manual target selection
- script/    : downloader/bootstrap scripts

Recommended next step after running this script
-----------------------------------------------
1. Open candidate_lists/wise_hii_shortlist_known_group_simple_shell.csv
2. Manually choose 3 shell-like targets
3. Download image cutouts for those 3 targets
4. Build the first standard radial profile
"""
    (root / "RAW_DATASET_OVERVIEW.txt").write_text(notes, encoding="utf-8")
    print(f"[OK] {root / 'RAW_DATASET_OVERVIEW.txt'}")


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    raw_root = project_root / RAW_SUBDIR
    script_dir = project_root / SCRIPT_SUBDIR
    wise_dir = project_root / WISE_SUBDIR
    docs_dir = project_root / DOC_SUBDIR
    cat_dir = project_root / CAT_SUBDIR
    candidate_dir = project_root / CANDIDATE_SUBDIR

    for d in [raw_root, script_dir, wise_dir, docs_dir, cat_dir, candidate_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - WISE H II Bootstrap")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Raw root     : {raw_root}")
    print("")

    downloads = [
        (URLS["wise_catalog_v2_3_csv"], cat_dir / "wise_hii_V2.3.csv", "binary"),
        (URLS["wise_catalog_v1_3_csv"], cat_dir / "wise_hii_V1.3.csv", "binary"),
        (URLS["wise_catalog_paper_pdf"], docs_dir / "wise_catalog_hii_regions.pdf", "binary"),
        (URLS["wise_table2_txt"], docs_dir / "Table2.txt", "text"),
        (URLS["wise_flux_densities_pdf"], docs_dir / "flux_densities.pdf", "binary"),
        (URLS["wise_site_index_html"], docs_dir / "wise_site_index.html", "text"),
    ]

    report_lines = []
    for url, out_path, kind in downloads:
        ok, msg = try_download(url, out_path, kind)
        report_lines.append(f"{out_path.name}\t{url}\t{ok}\t{msg}")
        time.sleep(0.5)

    # Use V2.3 if available, otherwise fall back to V1.3
    catalog_path = cat_dir / "wise_hii_V2.3.csv"
    if not catalog_path.exists():
        catalog_path = cat_dir / "wise_hii_V1.3.csv"

    if catalog_path.exists():
        rows = load_csv_rows(catalog_path)
        normalized = normalize_columns(rows)
        build_candidate_lists(normalized, candidate_dir)
    else:
        print("[WARN] No catalog CSV downloaded successfully. Candidate lists were not built.")

    write_project_notes(raw_root)

    report_path = raw_root / "download_report.tsv"
    report_path.write_text(
        "filename\turl\tsuccess\tmessage\n" + "\n".join(report_lines),
        encoding="utf-8"
    )
    print(f"[OK] {report_path}")

    print("")
    print("Done.")
    print("Recommended next file to open:")
    print(candidate_dir / "wise_hii_shortlist_known_group_simple_shell.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
