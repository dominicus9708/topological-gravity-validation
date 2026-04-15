#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
common5 FITS fetch pipeline (v2, URL-normalized)

Author: Kwon Dominicus

Placement
---------
data/raw/Validation of Structural Contrast Baseline/script/

Purpose
-------
Download real FITS files for the common5 targets and store them under:

data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/fits/

Fixes over v1
-------------
- Normalizes relative IRSA API URLs to absolute URLs
- URL-encodes query parameters safely, especially locstr with spaces
- Preserves already-valid absolute URLs
- Records normalized seed URL in the summary
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

SUMMARY_CSV = "common5_fits_download_summary_v2.csv"
MANIFEST_TXT = "common5_fits_download_manifest_v2.txt"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChatGPT common5 FITS downloader v2)",
    "Accept": "*/*",
}

TIMEOUT_SEC = 60
SLEEP_BETWEEN_TARGETS_SEC = 1.0
IRSA_BASE = "https://irsa.ipac.caltech.edu"


@dataclass
class DownloadResult:
    wise_name: str
    hii_region_name: str
    fits_url_seed_raw: str
    fits_url_seed_normalized: str
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


def normalize_seed_url(url: str) -> str:
    raw = str(url).strip()
    if not raw:
        return ""

    # Relative IRSA path -> absolute
    if raw.startswith("/"):
        raw = IRSA_BASE + raw

    parsed = urllib.parse.urlsplit(raw)

    if not parsed.scheme or not parsed.netloc:
        return raw

    # Parse query safely and re-encode
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)

    # If parse_qsl missed malformed spacing, fallback to manual parse
    if not pairs and "?" in raw:
        query = raw.split("?", 1)[1]
        temp_pairs = []
        for part in query.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
            else:
                k, v = part, ""
            temp_pairs.append((k, v))
        pairs = temp_pairs

    normalized_query = urllib.parse.urlencode(pairs, doseq=True, safe=":/")
    rebuilt = urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, normalized_query, parsed.fragment)
    )
    return rebuilt


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

    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            if elem.text:
                txt = elem.text.strip()
                if txt.lower().startswith("http") and ".fit" in txt.lower():
                    urls.append(txt)
            for _, v in elem.attrib.items():
                vv = str(v).strip()
                if vv.lower().startswith("http") and ".fit" in vv.lower():
                    urls.append(vv)
    except Exception:
        pass

    regex_urls = re.findall(r'https?://[^\s"<>\']+\.fits?(?:\.gz)?[^\s"<>\']*', xml_text, flags=re.IGNORECASE)
    urls.extend(regex_urls)

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

    def score(u: str):
        ul = u.lower()
        return (
            1 if "wise" in ul else 0,
            1 if "image" in ul else 0,
            1 if ".fits" in ul or ".fit" in ul else 0,
            -len(u),
        )

    return sorted(urls, key=score, reverse=True)[0]


def save_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def maybe_decompress_gz_to_fits(path_gz: Path) -> Path:
    if path_gz.suffix.lower() != ".gz":
        return path_gz

    out_path = path_gz.with_suffix("")
    with gzip.open(path_gz, "rb") as f_in:
        out_path.write_bytes(f_in.read())
    return out_path


def download_one_target(row: pd.Series, out_dir: Path) -> DownloadResult:
    wise_name = str(row["wise_name"]).strip()
    hii_region_name = str(row.get("hii_region_name", "")).strip()
    fits_url_seed_raw = str(row.get("fits_url", "")).strip()
    fits_url_seed_normalized = normalize_seed_url(fits_url_seed_raw)

    if not fits_url_seed_normalized or not fits_url_seed_normalized.lower().startswith("http"):
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed_raw=fits_url_seed_raw,
            fits_url_seed_normalized=fits_url_seed_normalized,
            xml_status="missing_or_invalid_seed_url",
            candidate_fits_count=0,
            selected_fits_url="",
            download_status="failed",
            output_path="",
            error="No usable normalized fits_url seed present in standard final input",
        )

    try:
        xml_text = http_get_text(fits_url_seed_normalized)
    except Exception as exc:
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed_raw=fits_url_seed_raw,
            fits_url_seed_normalized=fits_url_seed_normalized,
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
            fits_url_seed_raw=fits_url_seed_raw,
            fits_url_seed_normalized=fits_url_seed_normalized,
            xml_status="xml_ok_but_no_fits_url",
            candidate_fits_count=0,
            selected_fits_url="",
            download_status="failed",
            output_path="",
            error="No FITS-like URL found in FinderChart XML response",
        )

    ordered_urls = [selected] + [u for u in candidate_urls if u != selected]
    safe = sanitize_wise_name(wise_name)

    last_error = ""
    for url in ordered_urls:
        try:
            payload = http_get_bytes(url)
            parsed = urllib.parse.urlparse(url)
            path_lower = parsed.path.lower()

            if path_lower.endswith(".fits.gz"):
                filename = f"{safe}.fits.gz"
            elif path_lower.endswith(".fits"):
                filename = f"{safe}.fits"
            elif path_lower.endswith(".fit"):
                filename = f"{safe}.fit"
            elif path_lower.endswith(".fts"):
                filename = f"{safe}.fts"
            else:
                filename = f"{safe}.fits"

            out_path = out_dir / filename
            save_bytes(out_path, payload)
            final_path = maybe_decompress_gz_to_fits(out_path)

            return DownloadResult(
                wise_name=wise_name,
                hii_region_name=hii_region_name,
                fits_url_seed_raw=fits_url_seed_raw,
                fits_url_seed_normalized=fits_url_seed_normalized,
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
        fits_url_seed_raw=fits_url_seed_raw,
        fits_url_seed_normalized=fits_url_seed_normalized,
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
        "common5 FITS download manifest v2",
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
        "Fix note:",
        "This v2 normalizes relative IRSA API URLs into absolute, URL-encoded request URLs before fetching XML.",
    ]
    (out_dir / MANIFEST_TXT).write_text("\n".join(lines), encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - common5 FITS fetch v2")
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
