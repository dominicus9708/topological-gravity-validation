#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
common5 FITS fetch pipeline (v4, resolve fitsurl response)

Author: Kwon Dominicus

Placement
---------
data/raw/Validation of Structural Contrast Baseline/script/

Purpose
-------
Download actual FITS files for common5 targets by:
1) reading FinderChart XML
2) extracting <fitsurl> candidates
3) resolving the fitsurl response one more step if needed
4) saving real FITS files into the raw fits folder

Output
------
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/fits/fits/
    common5_fits_download_summary_v4.csv
    common5_fits_download_manifest_v4.txt
    debug_xml/
    debug_resolved/
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
from typing import List, Optional, Tuple

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
DEBUG_RESOLVED_DIRNAME = "debug_resolved"
SUMMARY_CSV = "common5_fits_download_summary_v4.csv"
MANIFEST_TXT = "common5_fits_download_manifest_v4.txt"

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ChatGPT common5 FITS downloader v4)",
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
    fitsurl_candidate_count: int
    selected_fitsurl: str
    resolve_status: str
    resolved_url: str
    download_status: str
    output_path: str
    debug_xml_path: str
    debug_resolved_path: str
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


def http_get(url: str) -> Tuple[bytes, dict]:
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
        payload = resp.read()
        headers = dict(resp.info())
        headers["_final_url"] = resp.geturl()
        return payload, headers


def decode_text(payload: bytes, headers: dict) -> str:
    ctype = headers.get("Content-Type", "")
    m = re.search(r"charset=([^\s;]+)", ctype, flags=re.IGNORECASE)
    charset = m.group(1) if m else "utf-8"
    return payload.decode(charset, errors="replace")


def extract_fitsurl_candidates(xml_text: str) -> List[str]:
    urls: List[str] = []
    root = ET.fromstring(xml_text)
    for elem in root.iter():
        tag = str(elem.tag).lower()
        txt = (elem.text or "").strip()
        if tag.endswith("fitsurl") and txt:
            urls.append(txt)
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def looks_like_fits_bytes(payload: bytes) -> bool:
    head = payload[:80]
    return head.startswith(b"SIMPLE  =") or head.startswith(b"XTENSION=")


def looks_like_gzip(payload: bytes) -> bool:
    return len(payload) >= 2 and payload[:2] == b"\x1f\x8b"


def extract_url_from_text(text: str) -> Optional[str]:
    m = re.search(r'https?://[^\s"<>\']+', text, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(0).rstrip(').,;\'"]')


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


def choose_best_fitsurl(urls: List[str]) -> Optional[str]:
    if not urls:
        return None

    def score(u: str):
        ul = u.lower()
        return (
            4 if "wise_bands=3" in ul else 0,
            3 if "wise_bands=4" in ul else 0,
            2 if "wise_bands=2" in ul else 0,
            1 if "wise_bands=1" in ul else 0,
            -len(u),
        )
    return sorted(urls, key=score, reverse=True)[0]


def resolve_fitsurl_response(url: str, debug_resolved_path: Path) -> Tuple[str, bytes, str]:
    payload, headers = http_get(url)
    final_url = headers.get("_final_url", url)

    if looks_like_fits_bytes(payload):
        return final_url, payload, "direct_fits_bytes"

    if looks_like_gzip(payload):
        return final_url, payload, "direct_gzip_bytes"

    text = decode_text(payload, headers)
    debug_resolved_path.write_text(text, encoding="utf-8")

    extracted = extract_url_from_text(text)
    if extracted and extracted != url:
        payload2, headers2 = http_get(extracted)
        final_url2 = headers2.get("_final_url", extracted)
        if looks_like_fits_bytes(payload2):
            return final_url2, payload2, "resolved_text_to_fits"
        if looks_like_gzip(payload2):
            return final_url2, payload2, "resolved_text_to_gzip"
        return final_url2, payload2, "resolved_text_but_not_fits"

    return final_url, payload, "non_fits_response"


def extension_for_payload(resolved_url: str, payload: bytes) -> str:
    if looks_like_gzip(payload):
        return ".fits.gz"
    if looks_like_fits_bytes(payload):
        return ".fits"
    ul = resolved_url.lower()
    if ul.endswith(".fits.gz"):
        return ".fits.gz"
    if ul.endswith(".fits"):
        return ".fits"
    return ".bin"


def download_one_target(row: pd.Series, out_dir: Path, debug_xml_dir: Path, debug_resolved_dir: Path) -> DownloadResult:
    wise_name = str(row["wise_name"]).strip()
    hii_region_name = str(row.get("hii_region_name", "")).strip()
    safe = sanitize_wise_name(wise_name)

    fits_url_seed_raw = str(row.get("fits_url", "")).strip()
    fits_url_seed_normalized = normalize_seed_url(fits_url_seed_raw)

    debug_xml_path = debug_xml_dir / f"{safe}_finderchart_response.xml"
    debug_resolved_path = debug_resolved_dir / f"{safe}_fitsurl_resolved_response.txt"

    if not fits_url_seed_normalized or not fits_url_seed_normalized.lower().startswith("http"):
        return DownloadResult(
            wise_name, hii_region_name, fits_url_seed_raw, fits_url_seed_normalized,
            "missing_or_invalid_seed_url", 0, "", "", "", "failed", "",
            "", "", "No usable normalized fits_url seed present in standard final input"
        )

    try:
        payload, headers = http_get(fits_url_seed_normalized)
        xml_text = decode_text(payload, headers)
        debug_xml_path.write_text(xml_text, encoding="utf-8")
    except Exception as exc:
        return DownloadResult(
            wise_name, hii_region_name, fits_url_seed_raw, fits_url_seed_normalized,
            "xml_request_failed", 0, "", "", "", "failed", "",
            str(debug_xml_path), str(debug_resolved_path), f"{type(exc).__name__}: {exc}"
        )

    try:
        fitsurl_candidates = extract_fitsurl_candidates(xml_text)
    except Exception as exc:
        return DownloadResult(
            wise_name, hii_region_name, fits_url_seed_raw, fits_url_seed_normalized,
            "xml_parse_failed", 0, "", "", "", "failed", "",
            str(debug_xml_path), str(debug_resolved_path), f"{type(exc).__name__}: {exc}"
        )

    selected = choose_best_fitsurl(fitsurl_candidates)
    if not selected:
        return DownloadResult(
            wise_name, hii_region_name, fits_url_seed_raw, fits_url_seed_normalized,
            "xml_ok_but_no_fitsurl_tag", 0, "", "", "", "failed", "",
            str(debug_xml_path), str(debug_resolved_path), "No <fitsurl> candidates found in FinderChart XML"
        )

    try:
        resolved_url, payload2, resolve_status = resolve_fitsurl_response(selected, debug_resolved_path)
        ext = extension_for_payload(resolved_url, payload2)
        out_path = out_dir / f"{safe}{ext}"

        if ext == ".bin":
            return DownloadResult(
                wise_name, hii_region_name, fits_url_seed_raw, fits_url_seed_normalized,
                "xml_ok", len(fitsurl_candidates), selected, resolve_status, resolved_url,
                "failed", "",
                str(debug_xml_path), str(debug_resolved_path), "Resolved response was not FITS/gzip"
            )

        save_bytes(out_path, payload2)
        final_path = maybe_decompress_gz_to_fits(out_path)

        return DownloadResult(
            wise_name, hii_region_name, fits_url_seed_raw, fits_url_seed_normalized,
            "xml_ok", len(fitsurl_candidates), selected, resolve_status, resolved_url,
            "ok", str(final_path),
            str(debug_xml_path), str(debug_resolved_path), ""
        )
    except Exception as exc:
        return DownloadResult(
            wise_name, hii_region_name, fits_url_seed_raw, fits_url_seed_normalized,
            "xml_ok", len(fitsurl_candidates), selected, "", "",
            "failed", "",
            str(debug_xml_path), str(debug_resolved_path), f"{type(exc).__name__}: {exc}"
        )


def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    if len(sys.argv) >= 2:
        project_root = Path(sys.argv[1]).expanduser().resolve()

    out_dir = project_root / FITS_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_xml_dir = out_dir / DEBUG_XML_DIRNAME
    debug_xml_dir.mkdir(parents=True, exist_ok=True)
    debug_resolved_dir = out_dir / DEBUG_RESOLVED_DIRNAME
    debug_resolved_dir.mkdir(parents=True, exist_ok=True)

    df = load_standard_input(project_root)

    results: List[DownloadResult] = []
    for _, row in df.iterrows():
        results.append(download_one_target(row, out_dir, debug_xml_dir, debug_resolved_dir))
        time.sleep(SLEEP_BETWEEN_TARGETS_SEC)

    summary_df = pd.DataFrame([r.__dict__ for r in results])
    summary_path = out_dir / SUMMARY_CSV
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    lines = [
        "Validation of Structural Contrast Baseline",
        "common5 FITS download manifest v4",
        "=" * 60,
        "",
        f"project_root: {project_root}",
        f"standard_input: {project_root / STANDARD_INPUT_FILE}",
        f"fits_output_dir: {out_dir}",
        f"debug_xml_dir: {debug_xml_dir}",
        f"debug_resolved_dir: {debug_resolved_dir}",
        "",
        f"target_count: {len(df)}",
        f"download_ok: {int((summary_df['download_status'] == 'ok').sum()) if not summary_df.empty else 0}",
        f"download_failed: {int((summary_df['download_status'] == 'failed').sum()) if not summary_df.empty else 0}",
        "",
        "Note:",
        "This v4 resolves the <fitsurl> response one more step before deciding whether the payload is real FITS.",
    ]
    (out_dir / MANIFEST_TXT).write_text("\n".join(lines), encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - common5 FITS fetch v4")
    print("=" * 72)
    print(f"Project root      : {project_root}")
    print(f"Input file        : {project_root / STANDARD_INPUT_FILE}")
    print(f"Output dir        : {out_dir}")
    print(f"Debug XML dir     : {debug_xml_dir}")
    print(f"Debug resolved dir: {debug_resolved_dir}")
    print("")
    print("[OK] Created:")
    print(summary_path)
    print(out_dir / MANIFEST_TXT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
