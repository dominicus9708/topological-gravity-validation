from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_project_subpath(raw_value: str) -> Path:
    raw_value = (raw_value or "").strip().replace("\\", "/")
    if not raw_value:
        return Path("raw") / "Cosmic Void Structural Validation" / "unclassified"
    return Path(raw_value)


def build_source_index(source_rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    index: Dict[str, List[Dict[str, str]]] = {}
    for row in source_rows:
        key = row.get("applies_to_dataset", "").strip()
        index.setdefault(key, []).append(row)
    return index


def write_dataset_folder_readme(dataset_dir: Path, dataset_row: Dict[str, str], related_sources: List[Dict[str, str]]) -> None:
    readme = dataset_dir / "README.txt"
    lines: List[str] = []
    lines.append(f"Dataset: {dataset_row.get('dataset_name', '')}")
    lines.append(f"Category: {dataset_row.get('category', '')}")
    lines.append(f"Provider: {dataset_row.get('provider', '')}")
    lines.append("")
    lines.append("Recommended use")
    lines.append("---------------")
    lines.append(dataset_row.get("recommended_use", ""))
    lines.append("")
    lines.append("Main files or products")
    lines.append("----------------------")
    lines.append(dataset_row.get("main_files_or_products", ""))
    lines.append("")
    lines.append("Key columns or quantities")
    lines.append("-------------------------")
    lines.append(dataset_row.get("key_columns_or_quantities", ""))
    lines.append("")
    lines.append("Primary documentation")
    lines.append("---------------------")
    lines.append(dataset_row.get("download_or_doc_url", ""))
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
    lines.append(dataset_row.get("notes", ""))
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")



def build_download_plan_rows(dataset_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    plan_rows: List[Dict[str, str]] = []
    for row in dataset_rows:
        plan_rows.append(
            {
                "priority": row.get("priority", ""),
                "dataset_name": row.get("dataset_name", ""),
                "category": row.get("category", ""),
                "target_folder": str(normalize_project_subpath(row.get("suggested_raw_subpath", ""))),
                "primary_url": row.get("download_or_doc_url", ""),
                "main_files_or_products": row.get("main_files_or_products", ""),
                "mass_info_available_directly": row.get("mass_info_available_directly", ""),
                "volume_info_available": row.get("volume_info_available", ""),
                "recommended_use": row.get("recommended_use", ""),
                "notes": row.get("notes", ""),
            }
        )
    return plan_rows



def build_source_registry_rows(
    dataset_rows: List[Dict[str, str]], source_index: Dict[str, List[Dict[str, str]]]
) -> List[Dict[str, str]]:
    expanded: List[Dict[str, str]] = []
    for ds in dataset_rows:
        matches = source_index.get(ds.get("dataset_name", "").strip(), [])
        if not matches:
            expanded.append(
                {
                    "dataset_name": ds.get("dataset_name", ""),
                    "source_id": "",
                    "source_type": "",
                    "title": "",
                    "citation_or_reference": "",
                    "url": ds.get("download_or_doc_url", ""),
                    "why_it_matters": "Fallback to dataset documentation URL from dataset table",
                }
            )
            continue
        for src in matches:
            expanded.append(
                {
                    "dataset_name": ds.get("dataset_name", ""),
                    "source_id": src.get("source_id", ""),
                    "source_type": src.get("source_type", ""),
                    "title": src.get("title", ""),
                    "citation_or_reference": src.get("citation_or_reference", ""),
                    "url": src.get("url", ""),
                    "why_it_matters": src.get("why_it_matters", ""),
                }
            )
    return expanded



def write_master_readme(raw_root: Path, dataset_rows: List[Dict[str, str]]) -> None:
    readme = raw_root / "README.txt"
    lines: List[str] = []
    lines.append("Cosmic Void Structural Validation raw package")
    lines.append("============================================")
    lines.append("")
    lines.append("Purpose")
    lines.append("-------")
    lines.append(
        "This folder stores recommended public refined datasets and their source registry for the cosmic void validation stage of the fourth paper."
    )
    lines.append("")
    lines.append("Generated files")
    lines.append("---------------")
    lines.append("- recommended_datasets_local_copy.csv")
    lines.append("- sources_local_copy.csv")
    lines.append("- download_plan_expanded.csv")
    lines.append("- source_registry_expanded.csv")
    lines.append("")
    lines.append("Recommended dataset priority")
    lines.append("----------------------------")
    for row in dataset_rows:
        lines.append(
            f"{row.get('priority', '')}. {row.get('dataset_name', '')} -> {row.get('category', '')}"
        )
    lines.append("")
    lines.append("Operational note")
    lines.append("----------------")
    lines.append(
        "This script does not scrape or bulk-download survey files automatically. It creates the project folder structure, copies the registry CSVs, and writes per-dataset README files so that manual download and later pipeline work remain reproducible."
    )
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")



def copy_input_csv(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8-sig")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw/Cosmic Void Structural Validation/ from recommended dataset and source CSV files."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Root of the topological_gravity_project repository. Default: current directory",
    )
    parser.add_argument(
        "--datasets-csv",
        default="cosmic_void_structural_validation_recommended_datasets.csv",
        help="Path to the recommended datasets CSV",
    )
    parser.add_argument(
        "--sources-csv",
        default="cosmic_void_structural_validation_sources.csv",
        help="Path to the sources CSV",
    )
    parser.add_argument(
        "--raw-folder-name",
        default="Cosmic Void Structural Validation",
        help="Folder name to create under raw/",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    datasets_csv = Path(args.datasets_csv).resolve()
    sources_csv = Path(args.sources_csv).resolve()

    if not datasets_csv.exists():
        raise FileNotFoundError(f"Recommended datasets CSV not found: {datasets_csv}")
    if not sources_csv.exists():
        raise FileNotFoundError(f"Sources CSV not found: {sources_csv}")

    raw_root = project_root / "raw" / args.raw_folder_name
    script_root = raw_root / "script"
    raw_root.mkdir(parents=True, exist_ok=True)
    script_root.mkdir(parents=True, exist_ok=True)

    dataset_rows = read_csv_rows(datasets_csv)
    source_rows = read_csv_rows(sources_csv)
    source_index = build_source_index(source_rows)

    # Copy registry CSVs into raw root
    copy_input_csv(datasets_csv, raw_root / "recommended_datasets_local_copy.csv")
    copy_input_csv(sources_csv, raw_root / "sources_local_copy.csv")

    # Create dataset folders and per-folder readmes
    for ds in dataset_rows:
        rel_subpath = normalize_project_subpath(ds.get("suggested_raw_subpath", ""))
        ds_dir = project_root / rel_subpath
        ds_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_folder_readme(ds_dir, ds, source_index.get(ds.get("dataset_name", "").strip(), []))

    # Create expanded helper CSVs inside raw root
    plan_rows = build_download_plan_rows(dataset_rows)
    write_csv_rows(
        raw_root / "download_plan_expanded.csv",
        plan_rows,
        [
            "priority",
            "dataset_name",
            "category",
            "target_folder",
            "primary_url",
            "main_files_or_products",
            "mass_info_available_directly",
            "volume_info_available",
            "recommended_use",
            "notes",
        ],
    )

    expanded_sources = build_source_registry_rows(dataset_rows, source_index)
    write_csv_rows(
        raw_root / "source_registry_expanded.csv",
        expanded_sources,
        [
            "dataset_name",
            "source_id",
            "source_type",
            "title",
            "citation_or_reference",
            "url",
            "why_it_matters",
        ],
    )

    write_master_readme(raw_root, dataset_rows)

    summary_lines = [
        f"Project root: {project_root}",
        f"Raw root created: {raw_root}",
        f"Datasets copied: {len(dataset_rows)}",
        f"Sources copied: {len(source_rows)}",
        "Created dataset subfolders:",
    ]
    for ds in dataset_rows:
        summary_lines.append(f"- {normalize_project_subpath(ds.get('suggested_raw_subpath', ''))}")

    summary_path = raw_root / "script" / "build_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Prepared raw package at: {raw_root}")
    print(f"Summary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
