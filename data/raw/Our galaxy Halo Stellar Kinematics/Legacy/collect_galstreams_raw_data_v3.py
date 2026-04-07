from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

ARCHIVE_URL = "https://github.com/cmateu/galstreams/archive/refs/heads/main.zip"


def find_project_root(start_file: Path) -> Path:
    current = start_file.resolve()
    for parent in current.parents:
        if (parent / "data").exists() and (parent / "results").exists():
            return parent
    for parent in current.parents:
        if (parent / "data").exists():
            return parent
    return start_file.resolve().parents[5]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url) as response, destination.open("wb") as f:
        shutil.copyfileobj(response, f)


def find_repo_root(extract_root: Path) -> Path:
    candidates = [p for p in extract_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError("No extracted repository directory found.")
    for p in candidates:
        if (p / "pyproject.toml").exists() and (p / "galstreams").exists():
            return p
    return candidates[0]


def write_manifest(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_runtime_dependencies() -> None:
    deps = ["numpy", "scipy", "astropy", "gala", "matplotlib", "pandas", "packaging"]
    cmd = [sys.executable, "-m", "pip", "install", *deps]
    print(f"[INFO] Installing runtime dependencies only: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def patch_local_version_file(repo_root: Path) -> Path:
    version_file = repo_root / "galstreams" / "_version.py"
    if not version_file.exists():
        version_file.write_text('version = "0+local"\n', encoding="utf-8")
    return version_file


def generate_topcat_csvs_from_source(repo_root: Path, output_root: Path) -> None:
    sys.path.insert(0, str(repo_root))
    patch_local_version_file(repo_root)

    print("[INFO] Importing galstreams directly from local source snapshot")
    from galstreams import MWStreams  # type: ignore

    print("[INFO] Generating TOPCAT-friendly galstreams CSV files")
    mws = MWStreams(verbose=False, print_topcat_friendly_files=False)
    mws.print_topcat_friendly_compilation(output_root=str(output_root))


def main() -> None:
    project_root = find_project_root(Path(__file__))

    raw_root = project_root / "data" / "raw" / "Our galaxy Halo Stellar Kinematics" / "galstreams"
    refs_dir = raw_root / "references"
    snapshot_dir = raw_root / "source_snapshot"
    compiled_dir = raw_root / "compiled_tracks"

    ensure_dir(refs_dir)
    ensure_dir(snapshot_dir)
    ensure_dir(compiled_dir)

    archive_path = snapshot_dir / "galstreams_main.zip"

    print(f"[INFO] Downloading archive: {ARCHIVE_URL}")
    download_file(ARCHIVE_URL, archive_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        extract_root = tmpdir_path / "extracted"
        ensure_dir(extract_root)

        print(f"[INFO] Extracting archive to: {extract_root}")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_root)

        repo_root = find_repo_root(extract_root)
        snapshot_copy_root = snapshot_dir / repo_root.name
        if snapshot_copy_root.exists():
            shutil.rmtree(snapshot_copy_root)
        shutil.copytree(repo_root, snapshot_copy_root)

    ensure_runtime_dependencies()

    output_root = compiled_dir / "galstreams_compilation"
    generate_topcat_csvs_from_source(snapshot_copy_root, output_root)

    expected_files = [
        compiled_dir / "galstreams_compilation.tracks.csv",
        compiled_dir / "galstreams_compilation.end_points.csv",
        compiled_dir / "galstreams_compilation.mid_points.csv",
        compiled_dir / "galstreams_compilation.summary.csv",
    ]

    missing = [str(p) for p in expected_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "galstreams CSV generation finished, but some expected files are missing:\n"
            + "\n".join(missing)
        )

    manifest_rows = []
    for p in expected_files:
        manifest_rows.append(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "kind": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            }
        )

    write_manifest(raw_root / "collection_manifest.csv", manifest_rows)

    print("[DONE] Saved:")
    for p in expected_files:
        print(f" - {p}")
    print(f" - {raw_root / 'collection_manifest.csv'}")


if __name__ == "__main__":
    main()
