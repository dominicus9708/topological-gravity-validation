from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
import numpy as np


# ============================================================
# Our Galaxy Halo Stellar Kinematics
# galstreams final input pipeline for halo
# ------------------------------------------------------------
# Placement:
#   data/derived/Our galaxy Halo Stellar Kinematics/scripts/
#     build_galstreams_input_halo_v1.py
#
# Input:
#   data/derived/Our galaxy Halo Stellar Kinematics/first_processed_v1/halo/
#
# Output:
#   data/derived/Our galaxy Halo Stellar Kinematics/input/halo/
#
# Purpose:
# - Convert first processed galstreams data into final halo input files
# - Keep only columns needed for later skeleton/standard/topological use
# - Produce stable, minimal, explicit input tables
# ============================================================


KEEP_TRACK_POINTS = [
    "point_uid",
    "TrackName",
    "StreamName",
    "ID",
    "track_point_index",
    "ra",
    "dec",
    "distance",
    "pm_ra_cosdec",
    "pm_dec",
    "radial_velocity",
]

KEEP_STREAM_CATALOG = [
    "TrackName",
    "Name",
    "StreamName",
    "ID",
    "On",
    "InfoFlags_raw",
    "InfoFlags_str",
    "has_empirical_track",
    "has_D",
    "has_pm",
    "has_vrad",
    "has_empirical_track_from_flag",
    "has_D_from_flag",
    "has_pm_from_flag",
    "has_vrad_from_flag",
    "length",
    "ra_mid",
    "dec_mid",
    "distance_mid",
    "ra_pole",
    "dec_pole",
    "width_phi2",
    "width_pm_phi1_cosphi2",
    "width_pm_phi2",
    "n_track_points",
    "n_mid_points",
    "n_end_points",
    "availability_score",
    "availability_score_from_flag",
    "halo_usefulness_label",
]

KEEP_SUMMARY = [
    "TrackName",
    "Name",
    "ID",
    "On",
    "InfoFlags_raw",
    "InfoFlags_str",
    "has_empirical_track",
    "has_D",
    "has_pm",
    "has_vrad",
    "has_empirical_track_from_flag",
    "has_D_from_flag",
    "has_pm_from_flag",
    "has_vrad_from_flag",
    "length",
    "ra_o",
    "dec_o",
    "distance_o",
    "ra_f",
    "dec_f",
    "distance_f",
    "ra_mid",
    "dec_mid",
    "distance_mid",
    "ra_pole",
    "dec_pole",
    "width_phi2",
    "width_pm_phi1_cosphi2",
    "width_pm_phi2",
]


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


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def keep_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    return df[cols].copy()


def normalize_infoflags_str(series: pd.Series) -> pd.Series:
    def conv(x):
        if pd.isna(x):
            return pd.NA
        s = str(x).strip()
        try:
            s = str(int(float(s)))
        except Exception:
            pass
        if s.isdigit():
            return s.zfill(4)
        return s
    return series.apply(conv)


def bit_to_int(series: pd.Series, idx: int) -> pd.Series:
    vals = []
    for v in series.fillna(""):
        if isinstance(v, str) and len(v) > idx and v[idx] in ("0", "1"):
            vals.append(int(v[idx]))
        else:
            vals.append(pd.NA)
    return pd.Series(vals, dtype="Int64")


def fill_missing_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "InfoFlags_str" in out.columns:
        out["InfoFlags_str"] = normalize_infoflags_str(out["InfoFlags_str"])
    elif "InfoFlags_raw" in out.columns:
        out["InfoFlags_str"] = normalize_infoflags_str(out["InfoFlags_raw"])

    for col_name, idx in [
        ("has_empirical_track_from_flag", 0),
        ("has_D_from_flag", 1),
        ("has_pm_from_flag", 2),
        ("has_vrad_from_flag", 3),
    ]:
        if col_name not in out.columns and "InfoFlags_str" in out.columns:
            out[col_name] = bit_to_int(out["InfoFlags_str"], idx)

    return out


def prioritize_boollike_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [
        "has_empirical_track",
        "has_D",
        "has_pm",
        "has_vrad",
        "has_empirical_track_from_flag",
        "has_D_from_flag",
        "has_pm_from_flag",
        "has_vrad_from_flag",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")
    return out


def build_overlay_candidates(track_points: pd.DataFrame, stream_catalog: pd.DataFrame) -> pd.DataFrame:
    # Minimal point-level overlay table for later halo shell matching
    points = keep_existing_columns(
        track_points,
        [
            "point_uid",
            "TrackName",
            "StreamName",
            "ID",
            "track_point_index",
            "ra",
            "dec",
            "distance",
            "pm_ra_cosdec",
            "pm_dec",
            "radial_velocity",
        ],
    )

    catalog = keep_existing_columns(
        stream_catalog,
        [
            "TrackName",
            "halo_usefulness_label",
            "availability_score",
            "availability_score_from_flag",
            "InfoFlags_str",
            "width_phi2",
            "ra_mid",
            "dec_mid",
            "distance_mid",
        ],
    ).drop_duplicates(subset=["TrackName"])

    out = points.merge(catalog, on="TrackName", how="left")

    # Conservative coarse bins for later fast shell triage
    if "distance" in out.columns:
        d = pd.to_numeric(out["distance"], errors="coerce")
        out["distance_band"] = pd.cut(
            d,
            bins=[0, 5, 10, 20, 40, 80, np.inf],
            labels=["0-5", "5-10", "10-20", "20-40", "40-80", "80+"],
            right=False,
        )

    return out


def write_readme(path: Path, *, input_dir: Path, output_dir: Path, catalog_df: pd.DataFrame, overlay_df: pd.DataFrame) -> None:
    text = f"""galstreams halo final input v1
==============================

작성 시각
---------
{datetime.now().isoformat(timespec="seconds")}

입력 경로
---------
{input_dir}

출력 경로
---------
{output_dir}

입력 파일 성격
-------------
이번 단계의 입력은 first_processed_v1/halo 의 1차 가공본이다.
이번 단계의 출력은 이후 halo 관련 skeleton / standard / topological 파이프라인이
직접 참조할 수 있는 최종 input 데이터이다.

생성 파일
---------
- galstreams_stream_catalog_input.csv
- galstreams_track_points_input.csv
- galstreams_summary_input.csv
- galstreams_halo_overlay_candidates_input.csv
- input_manifest.csv

최종 input 구성 원칙
-------------------
1. raw/중간 처리용 점검 열은 제거하거나 최소화한다.
2. 후속 파이프라인이 직접 참조할 열만 유지한다.
3. InfoFlags는 4자리 문자열 기준으로 통일한다.
4. stream catalog 와 point table 을 분리 유지한다.
5. overlay_candidates 는 halo shell 보조 매칭용 경량 입력이다.

기본 규모
---------
- stream catalog row 수: {len(catalog_df)}
- overlay candidate row 수: {len(overlay_df)}

권장 후속 단계
--------------
1. halo skeleton 에서 overlay_candidates_input 을 읽는 경로 테스트
2. shell summary 와 distance_band 기반의 느슨한 겹침 점검
3. 필요시 topological shell 결과와 stream label 을 연결하는 보조 파이프라인 작성
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    project_root = find_project_root(Path(__file__))

    input_dir = project_root / "data" / "derived" / "Our galaxy Halo Stellar Kinematics" / "first_processed_v1" / "halo"
    output_dir = project_root / "data" / "derived" / "Our galaxy Halo Stellar Kinematics" / "input" / "halo"
    ensure_dir(output_dir)

    tracks_path = input_dir / "galstreams_tracks_stage1.csv"
    catalog_path = input_dir / "galstreams_stream_catalog_stage1.csv"
    summary_path = input_dir / "galstreams_summary_stage1.csv"

    for p in [tracks_path, catalog_path, summary_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required stage1 input not found: {p}")

    tracks_df = read_csv(tracks_path)
    catalog_df = read_csv(catalog_path)
    summary_df = read_csv(summary_path)

    catalog_input = keep_existing_columns(catalog_df, KEEP_STREAM_CATALOG)
    summary_input = keep_existing_columns(summary_df, KEEP_SUMMARY)
    tracks_input = keep_existing_columns(tracks_df, KEEP_TRACK_POINTS)

    catalog_input = fill_missing_flag_columns(catalog_input)
    summary_input = fill_missing_flag_columns(summary_input)

    catalog_input = prioritize_boollike_columns(catalog_input)
    summary_input = prioritize_boollike_columns(summary_input)

    # Ensure availability scores are numeric if present
    for col in ["availability_score", "availability_score_from_flag", "n_track_points", "n_mid_points", "n_end_points"]:
        if col in catalog_input.columns:
            catalog_input[col] = pd.to_numeric(catalog_input[col], errors="coerce").astype("Int64")

    # De-duplicate conservatively
    if "TrackName" in catalog_input.columns:
        catalog_input = catalog_input.drop_duplicates(subset=["TrackName"]).reset_index(drop=True)
    if "TrackName" in summary_input.columns:
        summary_input = summary_input.drop_duplicates(subset=["TrackName"]).reset_index(drop=True)

    overlay_candidates = build_overlay_candidates(tracks_input, catalog_input)

    out_catalog = output_dir / "galstreams_stream_catalog_input.csv"
    out_tracks = output_dir / "galstreams_track_points_input.csv"
    out_summary = output_dir / "galstreams_summary_input.csv"
    out_overlay = output_dir / "galstreams_halo_overlay_candidates_input.csv"
    out_readme = output_dir / "README_input.txt"
    out_manifest = output_dir / "input_manifest.csv"

    write_csv(catalog_input, out_catalog)
    write_csv(tracks_input, out_tracks)
    write_csv(summary_input, out_summary)
    write_csv(overlay_candidates, out_overlay)
    write_readme(out_readme, input_dir=input_dir, output_dir=output_dir, catalog_df=catalog_input, overlay_df=overlay_candidates)

    manifest_rows = []
    for p in [out_catalog, out_tracks, out_summary, out_overlay, out_readme]:
        manifest_rows.append(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "filename": p.name,
                "path": str(p),
                "size_bytes": p.stat().st_size,
            }
        )

    with out_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["created_at", "filename", "path", "size_bytes"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("[DONE] Saved:")
    for p in [out_catalog, out_tracks, out_summary, out_overlay, out_readme, out_manifest]:
        print(f" - {p}")


if __name__ == "__main__":
    main()
