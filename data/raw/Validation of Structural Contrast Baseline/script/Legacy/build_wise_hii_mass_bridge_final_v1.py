#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation of Structural Contrast Baseline
WISE HII mass-bridge finalizer v1

Author: Kwon Dominicus

Purpose
-------
Reads the three seed CSV files already created under:

    data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/

and builds a single finalized bridge input file for later V3 use:

    wise_hii_mass_bridge.csv

This script does NOT invent stellar masses.
Instead, it consolidates:
- direct mass fields if already filled by the user
- preferred mass type and status
- radio proxy fields such as log_Nly and 1.4 GHz flux where available
- source tracking metadata

Recommended placement
---------------------
data/raw/Validation of Structural Contrast Baseline/script/

Inputs
------
wise_hii_mass_bridge_template_seed.csv
wise_hii_mass_bridge_sources_seed.csv
wise_hii_mass_bridge_radio_proxy_seed.csv

Outputs
-------
wise_hii_mass_bridge.csv
wise_hii_mass_bridge_manifest.txt
"""

from pathlib import Path
import pandas as pd

DEFAULT_PROJECT_ROOT = Path(r"C:\Users\mincu\Desktop\topological_gravity_project")

RAW_DIR = (
    Path("data")
    / "raw"
    / "Validation of Structural Contrast Baseline"
    / "wise_hii_catalog"
)

def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path, low_memory=False)

def _join_source_rows(src_df: pd.DataFrame) -> pd.DataFrame:
    if src_df.empty:
        return pd.DataFrame(columns=["wise_name", "source_bundle"])
    cols = ["wise_name", "source_role", "source_short", "source_url", "supported_fields", "source_note"]
    missing = [c for c in cols if c not in src_df.columns]
    if missing:
        raise ValueError("Source seed file missing columns: " + ", ".join(missing))

    src_df = src_df.copy()
    for c in cols:
        src_df[c] = src_df[c].fillna("").astype(str)

    def pack(group: pd.DataFrame) -> str:
        rows = []
        for _, r in group.iterrows():
            rows.append(
                f"[{r['source_role']}] {r['source_short']} | {r['source_url']} | "
                f"{r['supported_fields']} | {r['source_note']}"
            )
        return " || ".join(rows)

    out = src_df.groupby("wise_name", as_index=False).apply(pack)
    out = out.reset_index()
    out.columns = ["wise_name", "source_bundle"]
    return out

def main() -> int:
    project_root = DEFAULT_PROJECT_ROOT
    raw_dir = project_root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    template_path = raw_dir / "wise_hii_mass_bridge_template_seed.csv"
    sources_path = raw_dir / "wise_hii_mass_bridge_sources_seed.csv"
    radio_path = raw_dir / "wise_hii_mass_bridge_radio_proxy_seed.csv"

    template_df = _safe_read_csv(template_path)
    sources_df = _safe_read_csv(sources_path)
    radio_df = _safe_read_csv(radio_path) if radio_path.exists() else pd.DataFrame(columns=["wise_name"])

    required_template = [
        "wise_name",
        "object_type",
        "mass_bridge_status",
        "preferred_mass_type",
        "mass_value_msun",
        "mass_value_min_msun",
        "mass_value_max_msun",
        "mass_source_short",
        "mass_source_url",
        "mass_source_note",
        "recommended_next_step",
    ]
    missing_template = [c for c in required_template if c not in template_df.columns]
    if missing_template:
        raise ValueError("Template seed missing columns: " + ", ".join(missing_template))

    source_bundle_df = _join_source_rows(sources_df)

    out_df = template_df.copy()
    out_df = out_df.merge(source_bundle_df, on="wise_name", how="left")

    if not radio_df.empty:
        keep_radio = [
            "wise_name",
            "distance_kpc",
            "galactocentric_distance_kpc",
            "effective_radius_pc",
            "flux_1p4GHz_Jy",
            "log_Nly_s_minus_1",
            "stromgren_radius_pc",
            "dynamical_age_Myr",
            "cluster_candidate",
            "source_short",
            "source_url",
        ]
        for c in keep_radio:
            if c not in radio_df.columns:
                radio_df[c] = ""
        radio_df = radio_df[keep_radio].copy()
        radio_df = radio_df.rename(
            columns={
                "source_short": "radio_proxy_source_short",
                "source_url": "radio_proxy_source_url",
            }
        )
        out_df = out_df.merge(radio_df, on="wise_name", how="left")
    else:
        out_df["distance_kpc"] = ""
        out_df["galactocentric_distance_kpc"] = ""
        out_df["effective_radius_pc"] = ""
        out_df["flux_1p4GHz_Jy"] = ""
        out_df["log_Nly_s_minus_1"] = ""
        out_df["stromgren_radius_pc"] = ""
        out_df["dynamical_age_Myr"] = ""
        out_df["cluster_candidate"] = ""
        out_df["radio_proxy_source_short"] = ""
        out_df["radio_proxy_source_url"] = ""

    # Add explicit V3-facing interpretation fields
    out_df["bridge_ready_for_v3"] = out_df["mass_bridge_status"].map(
        {
            "radio_proxy_available": "radio_proxy_only",
            "needs_followup": "no_mass_bridge_yet",
        }
    ).fillna("review_needed")

    out_df["mass_input_mode_for_v3"] = out_df["preferred_mass_type"].map(
        {
            "equivalent_ionizing_star_mass": "requires_conversion_or_manual_fill",
            "direct_star_mass": "direct_mass_ready",
            "cluster_or_system_mass": "cluster_mass_ready",
        }
    ).fillna("review_needed")

    final_path = raw_dir / "wise_hii_mass_bridge.csv"
    out_df.to_csv(final_path, index=False, encoding="utf-8-sig")

    manifest_lines = [
        "Validation of Structural Contrast Baseline",
        "WISE HII mass-bridge finalizer manifest",
        "=" * 60,
        "",
        f"raw_dir: {raw_dir}",
        f"template_seed: {template_path}",
        f"sources_seed: {sources_path}",
        f"radio_proxy_seed: {radio_path}",
        f"final_bridge: {final_path}",
        "",
        "Meaning",
        "-" * 20,
        "This file is the consolidated mass-bridge input for later V3 use.",
        "It does not invent stellar masses.",
        "Where radio proxy data exist, they are included but kept distinct from direct mass values.",
        "Rows with no direct mass or radio proxy remain as follow-up targets.",
    ]
    manifest_path = raw_dir / "wise_hii_mass_bridge_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    print("=" * 72)
    print("Validation of Structural Contrast Baseline - mass bridge finalizer v1")
    print("=" * 72)
    print("[OK] Created:")
    print(final_path)
    print(manifest_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
