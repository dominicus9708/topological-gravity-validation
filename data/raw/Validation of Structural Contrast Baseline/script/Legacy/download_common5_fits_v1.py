#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
common5 FITS fetch pipeline (v1)

Author: Kwon Dominicus

Placement
---------
data/raw/Validation of Structural Contrast Baseline/script/

Purpose
-------
Download real FITS files for the common5 targets and store them under:

data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/fits/

Primary inputs
--------------
1) data/derived/Validation of Structural Contrast Baseline/input/standard/wise_hii_common5/wise_hii_common5_standard_final_input.csv
2) optional:
   data/derived/Validation of Structural Contrast Baseline/source_registry/wise_hii_common5/wise_hii_common5_radius_source_registry.csv

Behavior
--------
- Reads the IRSA FinderChart API query URL from fits_url.
- Requests the XML result.
- Extracts candidate FITS URLs from the XML.
- Downloads the first matching FITS URL (or tries multiple candidates).
- Saves one FITS file per target.
- Writes a download summary CSV and manifest TXT.

Important
---------
- This pipeline only downloads files; it does not build radial profiles.
- It is intended to be run before the radial-profile build pipeline.

Windows example
---------------
python "data\\raw\\Validation of Structural Contrast Baseline\\script\\download_common5_fits_v1.py"
"""

from __future__ import annotations

import gzip
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

STANDARD_INPUT_FILE = (
    Path("data")
    / "derived"
    / "Validation of Structural Contrast Baseline"
    / "input"
    / "standard"
    / "wise_hii_common5"
    / "wise_hii_common5_standard_final_input.csv"
)

FITS_OUT_DIR = (
    Path("data")
    / "raw"
    / "Validation of Structural Contrast Baseline"
    / "wise_hii_catalog"
    / "fits"
    / "fits"
)

SUMMARY_CSV = "common5_fits_download_summary.csv"
MANIFEST_TXT = "common5_fits_download_manifest.txt"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChatGPT common5 FITS downloader)",
    "Accept": "*/*",
}

TIMEOUT_SEC = 60
SLEEP_BETWEEN_TARGETS_SEC = 1.0


@dataclass
class DownloadResult:
    wise_name: str
    hii_region_name: str
    fits_url_seed: str
    xml_status: str
    candidate_fits_count: int
    selected_fits_url: str
    download_status: str
    output_path: str
    error: str


def sanitize_wise_name(name: str) -> str:
    text = str(name).strip()
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")


def load_standard_input(project_root: Path) -> pd.DataFrame:
    path = project_root / STANDARD_INPUT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Standard final input not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = ["wise_name", "hii_region_name", "fits_url"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Standard final input is missing required columns: " + ", ".join(missing))
    df["wise_name"] = df["wise_name"].astype(str).str.strip()
    return df


def http_get_text(url: str) -> str:
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
        return raw.decode(charset, errors="replace")


def http_get_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        return resp.read()


def extract_fits_urls_from_xml(xml_text: str) -> List[str]:
    urls: List[str] = []

    # 1) XML parse: collect text nodes that look like FITS URLs
    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            if elem.text:
                txt = elem.text.strip()
                if txt.lower().startswith("http") and ".fit" in txt.lower():
                    urls.append(txt)
            for k, v in elem.attrib.items():
                vv = str(v).strip()
                if vv.lower().startswith("http") and ".fit" in vv.lower():
                    urls.append(vv)
    except Exception:
        pass

    # 2) Regex fallback
    regex_urls = re.findall(r'https?://[^\s"<>\']+\.fits?(?:\.gz)?[^\s"<>\']*', xml_text, flags=re.IGNORECASE)
    urls.extend(regex_urls)

    # 3) De-duplicate while preserving order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def choose_best_fits_url(urls: List[str]) -> Optional[str]:
    if not urls:
        return None

    # Prefer WISE-related URLs when possible, then any FITS-like candidate
    def score(u: str):
        ul = u.lower()
        return (
            1 if "wise" in ul else 0,
            1 if "fits" in ul or ".fit" in ul else 0,
            1 if "image" in ul else 0,
            -len(u),
        )

    urls_sorted = sorted(urls, key=score, reverse=True)
    return urls_sorted[0]


def save_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def maybe_decompress_gz_to_fits(path_gz: Path) -> Path:
    if path_gz.suffix.lower() != ".gz":
        return path_gz

    out_path = path_gz.with_suffix("")  # remove .gz
    with gzip.open(path_gz, "rb") as f_in:
        out_path.write_bytes(f_in.read())
    return out_path


def download_one_target(row: pd.Series, out_dir: Path) -> DownloadResult:
    wise_name = str(row["wise_name"]).strip()
    hii_region_name = str(row.get("hii_region_name", "")).strip()
    fits_url_seed = str(row.get("fits_url", "")).strip()

    if not fits_url_seed or not fits_url_seed.lower().startswith("http"):
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed=fits_url_seed,
            xml_status="missing_seed_url",
            candidate_fits_count=0,
            selected_fits_url="",
            download_status="failed",
            output_path="",
            error="No usable fits_url seed present in standard final input",
        )

    try:
        xml_text = http_get_text(fits_url_seed)
    except Exception as exc:
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed=fits_url_seed,
            xml_status="xml_request_failed",
            candidate_fits_count=0,
            selected_fits_url="",
            download_status="failed",
            output_path="",
            error=f"{type(exc).__name__}: {exc}",
        )

    candidate_urls = extract_fits_urls_from_xml(xml_text)
    selected = choose_best_fits_url(candidate_urls)

    if not selected:
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed=fits_url_seed,
            xml_status="xml_ok_but_no_fits_url",
            candidate_fits_count=0,
            selected_fits_url="",
            download_status="failed",
            output_path="",
            error="No FITS-like URL found in FinderChart XML response",
        )

    # Try selected first, then all others if selected fails
    ordered_urls = [selected] + [u for u in candidate_urls if u != selected]
    safe = sanitize_wise_name(wise_name)

    last_error = ""
    for idx, url in enumerate(ordered_urls, start=1):
        try:
            payload = http_get_bytes(url)

            parsed = urllib.parse.urlparse(url)
            suffix = Path(parsed.path).suffix.lower()
            if parsed.path.lower().endswith(".fits.gz"):
                filename = f"{safe}.fits.gz"
            elif suffix in {".fits", ".fit", ".fts"}:
                filename = f"{safe}{suffix}"
            else:
                filename = f"{safe}.fits"

            out_path = out_dir / filename
            save_bytes(out_path, payload)

            final_path = maybe_decompress_gz_to_fits(out_path)
            return DownloadResult(
                wise_name=wise_name,
                hii_region_name=hii_region_name,
                fits_url_seed=fits_url_seed,
                xml_status="xml_ok",
                candidate_fits_count=len(candidate_urls),
                selected_fits_url=url,
                download_status="ok",
                output_path=str(final_path),
                error="",
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    return DownloadResult(
        wise_name=wise_name,
        hii_region_name=hii_region_name,
        fits_url_seed=fits_url_seed,
        xml_status="xml_ok",
        candidate_fits_count=len(candidate_urls),
        selected_fits_url=selected,
        download_status="failed",
        output_path="",
        error=last_error or "Download failed for all candidate FITS URLs",
    )


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = project_root / FITS_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_standard_input(project_root)

    results: List[DownloadResult] = []
    for _, row in df.iterrows():
        results.append(download_one_target(row, out_dir))
        time.sleep(SLEEP_BETWEEN_TARGETS_SEC)

    summary_df = pd.DataFrame([r.__dict__ for r in results])
    summary_path = out_dir / SUMMARY_CSV
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lines = [
        "Validation of Structural Contrast Baseline",
        "common5 FITS download manifest",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"standard_input: {project_root / STANDARD_INPUT_FILE}",
        f"fits_output_dir: {out_dir}",
        "",
        f"target_count: {len(df)}",
        f"download_ok: {int((summary_df['download_status'] == 'ok').sum()) if not summary_df.empty else 0}",
        f"download_failed: {int((summary_df['download_status'] == 'failed').sum()) if not summary_df.empty else 0}",
        "",
        "Note:",
        "This pipeline downloads local FITS files for common5 targets from the service URLs already prepared in the input.",
        "A successful download here should remove the current 'missing_local_fits' bottleneck in the radial-profile build stage.",
    ]
    (out_dir / MANIFEST_TXT).write_text("\n".join(lines), encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - common5 FITS fetch v1")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Input file   : {project_root / STANDARD_INPUT_FILE}")
    print(f"Output dir   : {out_dir}")
    print("")
    print("[OK] Created:")
    print(summary_path)
    print(out_dir / MANIFEST_TXT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
