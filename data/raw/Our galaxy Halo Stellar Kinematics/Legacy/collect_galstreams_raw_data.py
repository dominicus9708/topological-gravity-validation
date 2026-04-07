from __future__ import annotations

import csv
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# ============================================================
# Our galaxy Halo Stellar Kinematics
# Raw data collector for galstreams
# ------------------------------------------------------------
# Intended placement:
#   data/raw/Our galaxy Halo Stellar Kinematics/scripts/
#     collect_galstreams_raw_data.py
#
# Intended outputs:
#   data/raw/Our galaxy Halo Stellar Kinematics/galstreams/
#     references/galstreams_raw_reference_links.csv
#     source_snapshot/...
#     tracks/*.csv
#
# Purpose:
# - Download or snapshot the public galstreams repository
# - Locate TOPCAT-friendly CSV track files (default repo location:
#   galstreams/tracks according to the upstream README)
# - Copy them into the project's raw-data folder without altering
#   their contents
# ============================================================

REPO_ZIP_URL = "https://github.com/cmateu/galstreams/archive/refs/heads/main.zip"
REPO_PAGE_URL = "https://github.com/cmateu/galstreams"
PROJECT_REL_RAW = Path("data/raw/Our galaxy Halo Stellar Kinematics")
REFERENCES_FILENAME = "galstreams_raw_reference_links.csv"


def find_project_root(start_file: Path) -> Path:
    current = start_file.resolve()
    for parent in current.parents:
        if (parent / "data").exists() and (parent / "results").exists():
            return parent
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    raise FileNotFoundError("Could not infer project root. Place this script inside the project tree.")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_zip(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url) as response, open(dest, "wb") as fh:
        fh.write(response.read())


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)


def find_tracks_dir(search_root: Path) -> Path | None:
    candidates = []
    for p in search_root.rglob("tracks"):
        if p.is_dir():
            csvs = list(p.glob("*.csv"))
            if csvs:
                candidates.append((len(csvs), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def copy_tree(src: Path, dst: Path) -> int:
    ensure_dir(dst)
    count = 0
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
            count += sum(1 for _ in target.rglob("*") if _.is_file())
        else:
            shutil.copy2(item, target)
            count += 1
    return count


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "stored_path",
                "source_kind",
                "upstream_url",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = find_project_root(Path(__file__))
    raw_root = project_root / PROJECT_REL_RAW
    galstreams_root = raw_root / "galstreams"
    tracks_out = galstreams_root / "tracks"
    snapshot_out = galstreams_root / "source_snapshot"
    references_out = galstreams_root / "references"
    manifest_path = galstreams_root / "collection_manifest.csv"

    ensure_dir(tracks_out)
    ensure_dir(snapshot_out)
    ensure_dir(references_out)

    # Copy bundled references CSV if the user placed it next to the script.
    local_refs = Path(__file__).with_name(REFERENCES_FILENAME)
    if local_refs.exists():
        shutil.copy2(local_refs, references_out / REFERENCES_FILENAME)

    manifest_rows: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        zip_path = tmpdir / "galstreams_main.zip"
        extract_dir = tmpdir / "extracted"

        print(f"[INFO] Downloading archive: {REPO_ZIP_URL}")
        download_zip(REPO_ZIP_URL, zip_path)
        print(f"[INFO] Extracting archive to: {extract_dir}")
        extract_zip(zip_path, extract_dir)

        # Store a full raw snapshot for traceability.
        extracted_roots = [p for p in extract_dir.iterdir() if p.is_dir()]
        if not extracted_roots:
            raise FileNotFoundError("No extracted repository root found in downloaded archive.")
        repo_root = extracted_roots[0]

        if snapshot_out.exists():
            shutil.rmtree(snapshot_out)
        shutil.copytree(repo_root, snapshot_out)
        manifest_rows.append(
            {
                "stored_path": str(snapshot_out),
                "source_kind": "github_repo_snapshot",
                "upstream_url": REPO_PAGE_URL,
                "notes": "Raw snapshot of the downloaded galstreams repository archive.",
            }
        )

        tracks_dir = find_tracks_dir(repo_root)
        if tracks_dir is None:
            raise FileNotFoundError(
                "Could not find a tracks directory containing CSV files. "
                "Inspect the source snapshot manually and update the script if upstream layout changed."
            )

        print(f"[INFO] Found track CSV directory: {tracks_dir}")
        if tracks_out.exists():
            shutil.rmtree(tracks_out)
        copied_count = copy_tree(tracks_dir, tracks_out)
        manifest_rows.append(
            {
                "stored_path": str(tracks_out),
                "source_kind": "track_csv_copy",
                "upstream_url": REPO_PAGE_URL,
                "notes": f"Copied TOPCAT-friendly CSV tables from upstream tracks directory. File count={copied_count}.",
            }
        )

    write_manifest(manifest_path, manifest_rows)

    print("[DONE] Saved:")
    print(f" - {references_out / REFERENCES_FILENAME}")
    print(f" - {snapshot_out}")
    print(f" - {tracks_out}")
    print(f" - {manifest_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FAILED] {exc}", file=sys.stderr)
        raise
