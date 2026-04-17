#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List

DEFAULT_TARGET = Path("data") / "raw" / "Cosmic Void Structural Validation"


def sanitize_folder_name(raw_name: str) -> str:
    mapping = {
        "DESIVAST Catalog (DESI DR1)": "DESIVAST",
        "Stellar Mass and Emission Line Catalog (DESI DR1)": "DESI_Stellar_Mass_Emission",
        "Gfinder Halo-based Group Catalog (DESI DR1)": "DESI_Gfinder",
        "SDSS DR7 VoidFinder Catalog (Pan et al. 2012)": "SDSS_DR7_VoidFinder",
        "SDSS DR7 Watershed/ZOBOV-based Public Void Catalog (Sutter et al. 2012; Nadathur & Hotchkiss 2014 lineage)": "SDSS_DR7_Watershed",
    }
    return mapping.get(raw_name, raw_name.replace(" ", "_").replace("/", "_"))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def copy_if_needed(src: Path, dst: Path) -> None:
    src_r = src.resolve()
    dst_r = dst.resolve()
    if src_r == dst_r:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_r, dst_r)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_dataset_readme(dataset: Dict[str, str], related_sources: List[Dict[str, str]]) -> str:
    lines = []
    lines.append(f"Dataset: {dataset.get('dataset_name', '')}")
    lines.append(f"Category: {dataset.get('category', '')}")
    lines.append(f"Provider: {dataset.get('provider', '')}")
    lines.append("")
    lines.append("Recommended use")
    lines.append("---------------")
    lines.append(dataset.get("recommended_use", ""))
    lines.append("")
    lines.append("Main files or products")
    lines.append("----------------------")
    lines.append(dataset.get("main_files_or_products", ""))
    lines.append("")
    lines.append("Key columns or quantities")
    lines.append("-------------------------")
    lines.append(dataset.get("key_columns_or_quantities", ""))
    lines.append("")
    lines.append("Primary documentation")
    lines.append("---------------------")
    lines.append(dataset.get("download_or_doc_url", ""))
    lines.append("")
    lines.append("Mass info available directly")
    lines.append("----------------------------")
    lines.append(dataset.get("mass_info_available_directly", ""))
    lines.append("")
    lines.append("Volume info available")
    lines.append("---------------------")
    lines.append(dataset.get("volume_info_available", ""))
    lines.append("")
    lines.append("Suggested raw subpath in project design")
    lines.append("---------------------------------------")
    lines.append(dataset.get("suggested_raw_subpath", ""))
    lines.append("")
    lines.append("Related sources")
    lines.append("---------------")
    if related_sources:
        for src in related_sources:
            lines.append(f"- [{src.get('source_id', '')}] {src.get('title', '')}")
            lines.append(f"  {src.get('citation_or_reference', '')}")
            lines.append(f"  {src.get('url', '')}")
    else:
        lines.append("- No separate source entry matched this dataset.")
    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append(dataset.get("notes", ""))
    lines.append("")
    lines.append("Manual download note")
    lines.append("--------------------")
    lines.append(
        "This folder may remain empty until the user manually downloads the actual dataset files "
        "from the official page above or later adds a downloader script with direct file URLs."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--datasets-csv", required=True, help="Path to recommended datasets CSV")
    parser.add_argument("--sources-csv", required=True, help="Path to sources CSV")
    parser.add_argument(
        "--target-root",
        default=str(DEFAULT_TARGET),
        help="Target root relative to project root. Default: data/raw/Cosmic Void Structural Validation/",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    datasets_csv = (project_root / args.datasets_csv).resolve() if not Path(args.datasets_csv).is_absolute() else Path(args.datasets_csv).resolve()
    sources_csv = (project_root / args.sources_csv).resolve() if not Path(args.sources_csv).is_absolute() else Path(args.sources_csv).resolve()
    target_root = (project_root / args.target_root).resolve() if not Path(args.target_root).is_absolute() else Path(args.target_root).resolve()

    if not datasets_csv.exists():
        raise FileNotFoundError(f"datasets csv not found: {datasets_csv}")
    if not sources_csv.exists():
        raise FileNotFoundError(f"sources csv not found: {sources_csv}")

    script_dir = target_root / "script"
    target_root.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    datasets = read_csv_rows(datasets_csv)
    sources = read_csv_rows(sources_csv)

    datasets_local_copy = target_root / "recommended_datasets_local_copy.csv"
    sources_local_copy = target_root / "sources_local_copy.csv"
    copy_if_needed(datasets_csv, datasets_local_copy)
    copy_if_needed(sources_csv, sources_local_copy)

    if datasets_csv.parent.resolve() != target_root.resolve():
        copy_if_needed(datasets_csv, target_root / datasets_csv.name)
    if sources_csv.parent.resolve() != target_root.resolve():
        copy_if_needed(sources_csv, target_root / sources_csv.name)

    source_registry_rows = []
    download_plan_rows = []
    created_folders = []

    for ds in datasets:
        dataset_name = ds.get("dataset_name", "").strip()
        folder_name = sanitize_folder_name(dataset_name)
        ds_dir = target_root / folder_name
        ds_dir.mkdir(parents=True, exist_ok=True)
        created_folders.append(ds_dir)

        related = [s for s in sources if s.get("applies_to_dataset", "").strip() in (dataset_name, "DESI DR1 combination")]
        readme_text = build_dataset_readme(ds, related)
        write_text(ds_dir / "README.txt", readme_text)

        for src in related:
            source_registry_rows.append({
                "dataset_name": dataset_name,
                "dataset_folder": folder_name,
                "source_id": src.get("source_id", ""),
                "source_type": src.get("source_type", ""),
                "title": src.get("title", ""),
                "citation_or_reference": src.get("citation_or_reference", ""),
                "url": src.get("url", ""),
                "why_it_matters": src.get("why_it_matters", ""),
            })

        download_plan_rows.append({
            "dataset_name": dataset_name,
            "dataset_folder": folder_name,
            "category": ds.get("category", ""),
            "provider": ds.get("provider", ""),
            "primary_doc_url": ds.get("download_or_doc_url", ""),
            "main_files_or_products": ds.get("main_files_or_products", ""),
            "mass_info_available_directly": ds.get("mass_info_available_directly", ""),
            "volume_info_available": ds.get("volume_info_available", ""),
            "action_required": "manual_download_or_future_direct_downloader",
            "status": "pending",
            "notes": ds.get("notes", ""),
        })

    download_plan_csv = target_root / "download_plan_expanded.csv"
    source_registry_csv = target_root / "source_registry_expanded.csv"

    with download_plan_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "dataset_name", "dataset_folder", "category", "provider", "primary_doc_url",
            "main_files_or_products", "mass_info_available_directly", "volume_info_available",
            "action_required", "status", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(download_plan_rows)

    with source_registry_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "dataset_name", "dataset_folder", "source_id", "source_type", "title",
            "citation_or_reference", "url", "why_it_matters"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_registry_rows)

    summary = []
    summary.append(f"Project root: {project_root}")
    summary.append(f"Target raw root: {target_root}")
    summary.append(f"Datasets copied: {len(datasets)}")
    summary.append(f"Sources copied: {len(sources)}")
    summary.append("Created dataset subfolders:")
    for ds_dir in created_folders:
        summary.append(f"- {ds_dir.relative_to(project_root)}")
    summary.append("")
    summary.append("Generated files:")
    summary.append(f"- {datasets_local_copy.relative_to(project_root)}")
    summary.append(f"- {sources_local_copy.relative_to(project_root)}")
    summary.append(f"- {download_plan_csv.relative_to(project_root)}")
    summary.append(f"- {source_registry_csv.relative_to(project_root)}")
    summary.append("")
    summary.append("Important note:")
    summary.append(
        "This script builds the raw-data registry and helper files only. "
        "It does not guarantee automatic download because many official pages here are documentation pages, "
        "not direct file URLs."
    )

    write_text(script_dir / "build_summary.txt", "\n".join(summary))
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
