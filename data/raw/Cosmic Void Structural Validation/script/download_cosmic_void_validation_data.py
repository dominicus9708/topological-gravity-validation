#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_cosmic_void_validation_data.py

Downloads selected public DESI DR1 VAC files into:

    data/raw/Cosmic Void Structural Validation/

This script is intentionally conservative:
- It only targets files whose direct public URLs were verified.
- It supports selective download by dataset or individual file keys.
- It avoids auto-downloading the entire 52 GB stellar-mass catalog unless requested.

Verified public source pages:
- DESIVAST docs and public v1.0 index
- Stellar Mass and Emission Line docs and public v1.0 index
- Gfinder docs and public v1.0 index
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

BASE_TARGET = Path("data") / "raw" / "Cosmic Void Structural Validation"

FILES: Dict[str, Dict[str, str]] = {
    # DESIVAST
    "desivast_voidfinder_ngc": {
        "folder": "DESIVAST",
        "filename": "DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits",
        "size_bytes": "3864960",
        "dataset": "desivast",
    },
    "desivast_voidfinder_sgc": {
        "folder": "DESIVAST",
        "filename": "DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits",
        "size_bytes": "578880",
        "dataset": "desivast",
    },
    "desivast_v2_vide_ngc": {
        "folder": "DESIVAST",
        "filename": "DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits",
        "size_bytes": "459138240",
        "dataset": "desivast",
    },
    "desivast_v2_vide_sgc": {
        "folder": "DESIVAST",
        "filename": "DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits",
        "size_bytes": "61980480",
        "dataset": "desivast",
    },
    "desivast_v2_zobov_ngc": {
        "folder": "DESIVAST",
        "filename": "DESIVAST_BGS_VOLLIM_V2_ZOBOV_NGC.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/DESIVAST_BGS_VOLLIM_V2_ZOBOV_NGC.fits",
        "size_bytes": "44026560",
        "dataset": "desivast",
    },
    "desivast_v2_zobov_sgc": {
        "folder": "DESIVAST",
        "filename": "DESIVAST_BGS_VOLLIM_V2_ZOBOV_SGC.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/DESIVAST_BGS_VOLLIM_V2_ZOBOV_SGC.fits",
        "size_bytes": "6644160",
        "dataset": "desivast",
    },
    "desivast_v2_revolver_ngc": {
        "folder": "DESIVAST",
        "filename": "DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits",
        "size_bytes": "611812800",
        "dataset": "desivast",
    },
    "desivast_v2_revolver_sgc": {
        "folder": "DESIVAST",
        "filename": "DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits",
        "size_bytes": "87442560",
        "dataset": "desivast",
    },
    "desivast_sha256": {
        "folder": "DESIVAST",
        "filename": "dr1_vac_dr1_desivast_v1.0.sha256sum",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/desivast/v1.0/dr1_vac_dr1_desivast_v1.0.sha256sum",
        "size_bytes": "840",
        "dataset": "desivast",
    },

    # Stellar mass
    "stellar_mass_main": {
        "folder": "DESI_Stellar_Mass_Emission",
        "filename": "dr1_galaxy_stellarmass_lineinfo_v1.0.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/stellar-mass-emline/v1.0/dr1_galaxy_stellarmass_lineinfo_v1.0.fits",
        "size_bytes": "52339092480",
        "dataset": "stellar_mass",
    },
    "stellar_mass_sha256": {
        "folder": "DESI_Stellar_Mass_Emission",
        "filename": "dr1_vac_dr1_stellar-mass-emline_v1.0.sha256sum",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/stellar-mass-emline/v1.0/dr1_vac_dr1_stellar-mass-emline_v1.0.sha256sum",
        "size_bytes": "108",
        "dataset": "stellar_mass",
    },

    # Gfinder
    "gfinder_galaxy": {
        "folder": "DESI_Gfinder",
        "filename": "DESIDR9.y1.v1_galaxy.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/gfinder/v1.0/DESIDR9.y1.v1_galaxy.fits",
        "size_bytes": "8353368000",
        "dataset": "gfinder",
    },
    "gfinder_group": {
        "folder": "DESI_Gfinder",
        "filename": "DESIDR9.y1.v1_group.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/gfinder/v1.0/DESIDR9.y1.v1_group.fits",
        "size_bytes": "5577586560",
        "dataset": "gfinder",
    },
    "gfinder_gal2grp": {
        "folder": "DESI_Gfinder",
        "filename": "iDESIDR9.y1.v1_1.fits",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/gfinder/v1.0/iDESIDR9.y1.v1_1.fits",
        "size_bytes": "3233566080",
        "dataset": "gfinder",
    },
    "gfinder_sha256": {
        "folder": "DESI_Gfinder",
        "filename": "dr1_vac_dr1_gfinder_v1.0.sha256sum",
        "url": "https://data.desi.lbl.gov/public/dr1/vac/dr1/gfinder/v1.0/dr1_vac_dr1_gfinder_v1.0.sha256sum",
        "size_bytes": "271",
        "dataset": "gfinder",
    },
}

DATASET_PRESETS: Dict[str, List[str]] = {
    "desivast_minimal": [
        "desivast_voidfinder_ngc",
        "desivast_voidfinder_sgc",
        "desivast_sha256",
    ],
    "desivast_all": [
        "desivast_voidfinder_ngc",
        "desivast_voidfinder_sgc",
        "desivast_v2_vide_ngc",
        "desivast_v2_vide_sgc",
        "desivast_v2_zobov_ngc",
        "desivast_v2_zobov_sgc",
        "desivast_v2_revolver_ngc",
        "desivast_v2_revolver_sgc",
        "desivast_sha256",
    ],
    "gfinder_all": [
        "gfinder_galaxy",
        "gfinder_group",
        "gfinder_gal2grp",
        "gfinder_sha256",
    ],
    "stellar_mass_all": [
        "stellar_mass_main",
        "stellar_mass_sha256",
    ],
    "starter_small": [
        "desivast_voidfinder_ngc",
        "desivast_voidfinder_sgc",
        "desivast_v2_zobov_ngc",
        "desivast_v2_zobov_sgc",
        "desivast_sha256",
        "gfinder_sha256",
        "stellar_mass_sha256",
    ],
}

SAFE_DEFAULT_PRESET = "starter_small"


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def plan_from_args(preset: str | None, keys: List[str]) -> List[str]:
    selected: List[str] = []
    if preset:
        if preset not in DATASET_PRESETS:
            raise ValueError(f"Unknown preset: {preset}")
        selected.extend(DATASET_PRESETS[preset])
    for key in keys:
        if key not in FILES:
            raise ValueError(f"Unknown file key: {key}")
        selected.append(key)
    # preserve order, remove duplicates
    seen = set()
    ordered = []
    for key in selected:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def download_file(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, out_path.open("wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)


def sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_sha256sum_file(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not path.exists():
        return mapping
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            checksum = parts[0]
            filename = parts[-1].lstrip("*")
            mapping[filename] = checksum
    return mapping


def verify_against_sumfile(downloaded_file: Path, sumfile: Path) -> Tuple[bool, str]:
    expected = parse_sha256sum_file(sumfile)
    if downloaded_file.name not in expected:
        return False, "checksum entry not found in sha256sum file"
    actual = sha256_of_file(downloaded_file)
    return (actual.lower() == expected[downloaded_file.name].lower(),
            f"expected={expected[downloaded_file.name]} actual={actual}")


def write_manifest(target_root: Path, selected_keys: List[str]) -> Path:
    manifest = target_root / "script" / "download_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    total = 0
    for key in selected_keys:
        meta = FILES[key]
        size = int(meta["size_bytes"])
        total += size
        lines.append(f"{key}")
        lines.append(f"  folder   : {meta['folder']}")
        lines.append(f"  filename : {meta['filename']}")
        lines.append(f"  size     : {human_size(size)}")
        lines.append(f"  url      : {meta['url']}")
        lines.append("")
    lines.append(f"Total planned size: {human_size(total)}")
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument(
        "--target-root",
        default=str(BASE_TARGET),
        help="Target root relative to project root. Default: data/raw/Cosmic Void Structural Validation/",
    )
    parser.add_argument(
        "--preset",
        default=SAFE_DEFAULT_PRESET,
        help=f"Preset download group. Default: {SAFE_DEFAULT_PRESET}",
    )
    parser.add_argument(
        "--file-key",
        action="append",
        default=[],
        help="Additional individual file key to download. May be repeated.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Write manifest and print plan without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="If a matching sha256sum file is downloaded, verify files against it after download.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    target_root = (project_root / args.target_root).resolve() if not Path(args.target_root).is_absolute() else Path(args.target_root).resolve()

    selected_keys = plan_from_args(args.preset, args.file_key)
    manifest = write_manifest(target_root, selected_keys)

    total = sum(int(FILES[k]["size_bytes"]) for k in selected_keys)
    print(f"Target root: {target_root}")
    print(f"Manifest: {manifest}")
    print(f"Files planned: {len(selected_keys)}")
    print(f"Total planned size: {human_size(total)}")

    for key in selected_keys:
        meta = FILES[key]
        out_path = target_root / meta["folder"] / meta["filename"]
        print(f"- {key}: {out_path.relative_to(project_root)} [{human_size(int(meta['size_bytes']))}]")

    if args.list_only:
        print("List-only mode enabled. No files downloaded.")
        return 0

    for key in selected_keys:
        meta = FILES[key]
        out_path = target_root / meta["folder"] / meta["filename"]
        if out_path.exists() and not args.force:
            print(f"[SKIP] {key} already exists: {out_path}")
            continue

        print(f"[DOWNLOAD] {key}")
        print(f"  URL : {meta['url']}")
        print(f"  OUT : {out_path}")
        download_file(meta["url"], out_path)
        print(f"  DONE: {out_path}")

    if args.verify:
        print("Running checksum verification where possible...")
        # verify within each folder if sha256sum file exists
        folders = sorted({FILES[k]["folder"] for k in selected_keys})
        for folder in folders:
            folder_path = target_root / folder
            sumfiles = list(folder_path.glob("*.sha256sum"))
            if not sumfiles:
                print(f"[VERIFY-SKIP] {folder}: no sha256sum file")
                continue
            sumfile = sumfiles[0]
            expected_map = parse_sha256sum_file(sumfile)
            if not expected_map:
                print(f"[VERIFY-SKIP] {folder}: could not parse {sumfile.name}")
                continue
            for filename in expected_map:
                fpath = folder_path / filename
                if not fpath.exists():
                    print(f"[VERIFY-MISS] {folder}/{filename}")
                    continue
                ok, detail = verify_against_sumfile(fpath, sumfile)
                tag = "VERIFY-OK" if ok else "VERIFY-FAIL"
                print(f"[{tag}] {folder}/{filename} :: {detail}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
