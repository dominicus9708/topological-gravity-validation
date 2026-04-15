#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
common5 FITS fetch pipeline (v3, XML debug + broader candidate extraction)

Author: Kwon Dominicus

Placement
---------
data/raw/Validation of Structural Contrast Baseline/script/

Purpose
-------
Inspect IRSA FinderChart XML responses in detail and attempt broader extraction of
downloadable image/FITS candidates for common5 targets.

Outputs
-------
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/fits/
    common5_fits_download_summary_v3.csv
    common5_fits_download_manifest_v3.txt
    debug_xml/<wise_name>_finderchart_response.xml
    debug_xml/<wise_name>_finderchart_flattened_text.txt
    downloaded files if successful

Notes
-----
- This version saves the XML responses for inspection.
- It extracts not only '.fits' URLs, but also general HTTP links, then ranks them.
- It is still conservative and does not fabricate files.
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

DEBUG_XML_DIRNAME = "debug_xml"
SUMMARY_CSV = "common5_fits_download_summary_v3.csv"
MANIFEST_TXT = "common5_fits_download_manifest_v3.txt"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChatGPT common5 FITS downloader v3)",
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
    candidate_http_count: int
    candidate_fits_like_count: int
    selected_url: str
    selected_kind: str
    download_status: str
    output_path: str
    debug_xml_path: str
    debug_text_path: str
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

    if raw.startswith("/"):
        raw = IRSA_BASE + raw

    parsed = urllib.parse.urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw

    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
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
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, normalized_query, parsed.fragment)
    )


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


def flatten_xml_text(xml_text: str) -> str:
    lines = []
    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            tag = elem.tag
            text = (elem.text or "").strip()
            attrs = " ".join([f'{k}="{v}"' for k, v in elem.attrib.items()])
            if text or attrs:
                lines.append(f"TAG={tag} ATTRS={attrs} TEXT={text}")
    except Exception as exc:
        lines.append(f"[XML_PARSE_FAILED] {type(exc).__name__}: {exc}")

    for m in re.findall(r'https?://[^\s"<>\']+', xml_text, flags=re.IGNORECASE):
        lines.append(f"REGEX_URL={m}")

    return "\n".join(lines)


def extract_all_http_urls(xml_text: str) -> List[str]:
    urls: List[str] = []

    try:
        root = ET.fromstring(xml_text)
        for elem in root.iter():
            if elem.text:
                txt = elem.text.strip()
                if txt.lower().startswith("http"):
                    urls.append(txt)
            for _, v in elem.attrib.items():
                vv = str(v).strip()
                if vv.lower().startswith("http"):
                    urls.append(vv)
    except Exception:
        pass

    regex_urls = re.findall(r'https?://[^\s"<>\']+', xml_text, flags=re.IGNORECASE)
    urls.extend(regex_urls)

    seen = set()
    out = []
    for u in urls:
        u2 = u.rstrip(').,;\'"')
        if u2 not in seen:
            seen.add(u2)
            out.append(u2)
    return out


def classify_url_kind(url: str) -> str:
    ul = url.lower()
    if ".fits.gz" in ul:
        return "fits_gz"
    if ".fits" in ul or ".fit" in ul or ".fts" in ul:
        return "fits_like"
    if "image" in ul and ("wise" in ul or "finderchart" in ul):
        return "image_like"
    if "download" in ul:
        return "download_like"
    return "generic_http"


def choose_best_url(urls: List[str]) -> tuple[Optional[str], str]:
    if not urls:
        return None, ""

    def score(u: str):
        kind = classify_url_kind(u)
        return (
            5 if kind == "fits_gz" else 0,
            4 if kind == "fits_like" else 0,
            3 if kind == "download_like" else 0,
            2 if kind == "image_like" else 0,
            1 if "wise" in u.lower() else 0,
            -len(u),
        )

    best = sorted(urls, key=score, reverse=True)[0]
    return best, classify_url_kind(best)


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


def extension_for_url(url: str) -> str:
    ul = url.lower()
    if ul.endswith(".fits.gz"):
        return ".fits.gz"
    if ul.endswith(".fits"):
        return ".fits"
    if ul.endswith(".fit"):
        return ".fit"
    if ul.endswith(".fts"):
        return ".fts"
    return ".bin"


def download_one_target(row: pd.Series, out_dir: Path, debug_dir: Path) -> DownloadResult:
    wise_name = str(row["wise_name"]).strip()
    hii_region_name = str(row.get("hii_region_name", "")).strip()
    safe = sanitize_wise_name(wise_name)

    fits_url_seed_raw = str(row.get("fits_url", "")).strip()
    fits_url_seed_normalized = normalize_seed_url(fits_url_seed_raw)

    debug_xml_path = debug_dir / f"{safe}_finderchart_response.xml"
    debug_text_path = debug_dir / f"{safe}_finderchart_flattened_text.txt"

    if not fits_url_seed_normalized or not fits_url_seed_normalized.lower().startswith("http"):
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed_raw=fits_url_seed_raw,
            fits_url_seed_normalized=fits_url_seed_normalized,
            xml_status="missing_or_invalid_seed_url",
            candidate_http_count=0,
            candidate_fits_like_count=0,
            selected_url="",
            selected_kind="",
            download_status="failed",
            output_path="",
            debug_xml_path="",
            debug_text_path="",
            error="No usable normalized fits_url seed present in standard final input",
        )

    try:
        xml_text = http_get_text(fits_url_seed_normalized)
        debug_xml_path.write_text(xml_text, encoding="utf-8")
        debug_text_path.write_text(flatten_xml_text(xml_text), encoding="utf-8")
    except Exception as exc:
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed_raw=fits_url_seed_raw,
            fits_url_seed_normalized=fits_url_seed_normalized,
            xml_status="xml_request_failed",
            candidate_http_count=0,
            candidate_fits_like_count=0,
            selected_url="",
            selected_kind="",
            download_status="failed",
            output_path="",
            debug_xml_path=str(debug_xml_path),
            debug_text_path=str(debug_text_path),
            error=f"{type(exc).__name__}: {exc}",
        )

    candidate_urls = extract_all_http_urls(xml_text)
    fits_like_count = sum(1 for u in candidate_urls if classify_url_kind(u) in {"fits_gz", "fits_like"})
    selected, selected_kind = choose_best_url(candidate_urls)

    if not selected:
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed_raw=fits_url_seed_raw,
            fits_url_seed_normalized=fits_url_seed_normalized,
            xml_status="xml_ok_but_no_http_url",
            candidate_http_count=0,
            candidate_fits_like_count=0,
            selected_url="",
            selected_kind="",
            download_status="failed",
            output_path="",
            debug_xml_path=str(debug_xml_path),
            debug_text_path=str(debug_text_path),
            error="No HTTP URL found in FinderChart XML response",
        )

    try:
        payload = http_get_bytes(selected)
        ext = extension_for_url(selected)
        out_path = out_dir / f"{safe}{ext}"
        save_bytes(out_path, payload)
        final_path = maybe_decompress_gz_to_fits(out_path)

        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed_raw=fits_url_seed_raw,
            fits_url_seed_normalized=fits_url_seed_normalized,
            xml_status="xml_ok",
            candidate_http_count=len(candidate_urls),
            candidate_fits_like_count=fits_like_count,
            selected_url=selected,
            selected_kind=selected_kind,
            download_status="ok",
            output_path=str(final_path),
            debug_xml_path=str(debug_xml_path),
            debug_text_path=str(debug_text_path),
            error="",
        )
    except Exception as exc:
        return DownloadResult(
            wise_name=wise_name,
            hii_region_name=hii_region_name,
            fits_url_seed_raw=fits_url_seed_raw,
            fits_url_seed_normalized=fits_url_seed_normalized,
            xml_status="xml_ok",
            candidate_http_count=len(candidate_urls),
            candidate_fits_like_count=fits_like_count,
            selected_url=selected,
            selected_kind=selected_kind,
            download_status="failed",
            output_path="",
            debug_xml_path=str(debug_xml_path),
            debug_text_path=str(debug_text_path),
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = project_root / FITS_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / DEBUG_XML_DIRNAME
    debug_dir.mkdir(parents=True, exist_ok=True)

    df = load_standard_input(project_root)

    results: List[DownloadResult] = []
    for _, row in df.iterrows():
        results.append(download_one_target(row, out_dir, debug_dir))
        time.sleep(SLEEP_BETWEEN_TARGETS_SEC)

    summary_df = pd.DataFrame([r.__dict__ for r in results])
    summary_path = out_dir / SUMMARY_CSV
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lines = [
        "Validation of Structural Contrast Baseline",
        "common5 FITS download manifest v3",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"standard_input: {project_root / STANDARD_INPUT_FILE}",
        f"fits_output_dir: {out_dir}",
        f"debug_xml_dir: {debug_dir}",
        "",
        f"target_count: {len(df)}",
        f"download_ok: {int((summary_df['download_status'] == 'ok').sum()) if not summary_df.empty else 0}",
        f"download_failed: {int((summary_df['download_status'] == 'failed').sum()) if not summary_df.empty else 0}",
        "",
        "Note:",
        "This v3 stores the full XML response and flattened XML text for each target.",
        "Use these files when XML is reachable but FITS-link extraction still fails.",
    ]
    (out_dir / MANIFEST_TXT).write_text("\n".join(lines), encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - common5 FITS fetch v3")
    print("=" * 72)
    print(f"Project root : {project_root}")
    print(f"Input file   : {project_root / STANDARD_INPUT_FILE}")
    print(f"Output dir   : {out_dir}")
    print(f"Debug XML dir: {debug_dir}")
    print("")
    print("[OK] Created:")
    print(summary_path)
    print(out_dir / MANIFEST_TXT)
    print(debug_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())