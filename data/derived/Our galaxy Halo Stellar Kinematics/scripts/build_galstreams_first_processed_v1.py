from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
import numpy as np


# ============================================================
# Our Galaxy Halo Stellar Kinematics
# galstreams first processed v1 pipeline
# ------------------------------------------------------------
# Placement:
#   data/derived/Our galaxy Halo Stellar Kinematics/scripts/
#     build_galstreams_first_processed_v1.py
#
# Input:
#   data/raw/Our galaxy Halo Stellar Kinematics/galstreams/compiled_tracks/
#
# Output:
#   data/derived/Our galaxy Halo Stellar Kinematics/first_processed_v1/halo/
#
# Purpose:
# - Preserve raw files
# - Produce first-pass cleaned tables
# - Normalize point-level and track-level metadata
# - Create a stream catalog for later halo cross-labeling
# ============================================================


POINT_NUMERIC_COLS = [
    "ra", "dec", "distance", "pm_ra_cosdec", "pm_dec", "radial_velocity"
]

SUMMARY_NUMERIC_COLS = [
    "width_phi2",
    "width_pm_phi1_cosphi2",
    "width_pm_phi2",
    "total_v_sigma",
    "Lv_stream",
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
    "ID",
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


def to_numeric_if_present(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def drop_artifact_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    to_drop = [c for c in out.columns if c in ("Unnamed: 0", "index")]
    if to_drop:
        out = out.drop(columns=to_drop)
    return out


def normalize_identifier_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["TrackName", "StreamName", "Name", "TrackRefs", "TrackName.1"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


def clean_point_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    out = df.copy()
    out = drop_artifact_index_cols(out)
    out = normalize_identifier_cols(out)
    out = to_numeric_if_present(out, POINT_NUMERIC_COLS + ["ID"])

    if "ID" in out.columns:
        out["ID"] = out["ID"].astype("Int64")

    if "TrackName" in out.columns:
        out["track_point_index"] = out.groupby("TrackName").cumcount()
    else:
        out["track_point_index"] = pd.Series(range(len(out)), dtype="Int64")

    out["source_table"] = table_name
    out["point_uid"] = [f"{table_name}_{i:07d}" for i in range(len(out))]

    return out


def infoflags_to_str(x) -> object:
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


def bit_to_int(series: pd.Series, idx: int) -> pd.Series:
    vals = []
    for v in series.fillna(""):
        if isinstance(v, str) and len(v) > idx and v[idx] in ("0", "1"):
            vals.append(int(v[idx]))
        else:
            vals.append(pd.NA)
    return pd.Series(vals, dtype="Int64")


def clean_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = drop_artifact_index_cols(out)
    out = normalize_identifier_cols(out)

    if "TrackName.1" in out.columns:
        out = out.drop(columns=["TrackName.1"])

    out = to_numeric_if_present(out, SUMMARY_NUMERIC_COLS)

    if "ID" in out.columns:
        out["ID"] = out["ID"].astype("Int64")

    if "On" in out.columns:
        # Conservative conversion
        out["On"] = out["On"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        ).fillna(False)

    if "InfoFlags" in out.columns:
        out["InfoFlags_raw"] = out["InfoFlags"]
        out["InfoFlags_str"] = out["InfoFlags"].apply(infoflags_to_str)
        out["has_empirical_track_from_flag"] = bit_to_int(out["InfoFlags_str"], 0)
        out["has_D_from_flag"] = bit_to_int(out["InfoFlags_str"], 1)
        out["has_pm_from_flag"] = bit_to_int(out["InfoFlags_str"], 2)
        out["has_vrad_from_flag"] = bit_to_int(out["InfoFlags_str"], 3)

    return out


def build_point_counts(tracks_df: pd.DataFrame, mid_df: pd.DataFrame, end_df: pd.DataFrame) -> pd.DataFrame:
    track_counts = (
        tracks_df.groupby("TrackName", dropna=False)
        .size()
        .reset_index(name="n_track_points")
    )
    mid_counts = (
        mid_df.groupby("TrackName", dropna=False)
        .size()
        .reset_index(name="n_mid_points")
    )
    end_counts = (
        end_df.groupby("TrackName", dropna=False)
        .size()
        .reset_index(name="n_end_points")
    )

    out = track_counts.merge(mid_counts, on="TrackName", how="outer")
    out = out.merge(end_counts, on="TrackName", how="outer")

    for col in ["n_track_points", "n_mid_points", "n_end_points"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("Int64")

    return out


def build_stream_catalog(summary_df: pd.DataFrame, point_counts_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.merge(point_counts_df, on="TrackName", how="left")

    for col in ["n_track_points", "n_mid_points", "n_end_points"]:
        if col not in out.columns:
            out[col] = pd.Series([0] * len(out), dtype="Int64")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("Int64")

    availability_cols = [c for c in ["has_empirical_track", "has_D", "has_pm", "has_vrad"] if c in out.columns]
    if availability_cols:
        out["availability_score"] = (
            out[availability_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        ).astype("Int64")

    flag_cols = [
        c for c in [
            "has_empirical_track_from_flag",
            "has_D_from_flag",
            "has_pm_from_flag",
            "has_vrad_from_flag",
        ] if c in out.columns
    ]
    if flag_cols:
        out["availability_score_from_flag"] = (
            out[flag_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
        ).astype("Int64")

    # Simple halo-usefulness label for later triage
    if "availability_score_from_flag" in out.columns:
        score = out["availability_score_from_flag"].fillna(0)
    elif "availability_score" in out.columns:
        score = out["availability_score"].fillna(0)
    else:
        score = pd.Series([0] * len(out))

    labels = []
    for s in score:
        if s >= 4:
            labels.append("A_full_6D_like")
        elif s >= 3:
            labels.append("B_partial_dynamic")
        elif s >= 2:
            labels.append("C_position_only_plus")
        else:
            labels.append("D_minimal")
    out["halo_usefulness_label"] = labels

    return out


def build_track_points_normalized(tracks_df: pd.DataFrame) -> pd.DataFrame:
    out = tracks_df.copy()

    ordered_cols = []
    for c in [
        "point_uid",
        "source_table",
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
    ]:
        if c in out.columns:
            ordered_cols.append(c)

    other_cols = [c for c in out.columns if c not in ordered_cols]
    out = out[ordered_cols + other_cols]

    return out


def write_readme(path: Path, *, input_dir: Path, output_dir: Path, summary_df: pd.DataFrame, tracks_df: pd.DataFrame) -> None:
    track_n = len(summary_df)
    point_n = len(tracks_df)

    text = f"""galstreams first processed v1
=================================

작성 시각
---------
{datetime.now().isoformat(timespec="seconds")}

입력 경로
---------
{input_dir}

출력 경로
---------
{output_dir}

생성 파일
---------
- galstreams_tracks_stage1.csv
- galstreams_mid_points_stage1.csv
- galstreams_end_points_stage1.csv
- galstreams_summary_stage1.csv
- galstreams_stream_catalog_stage1.csv
- galstreams_track_points_normalized_stage1.csv
- stage1_manifest.csv

1차 가공 내용
-------------
1. raw CSV의 불필요한 인덱스 열 제거
2. 점(point) 테이블의 핵심 수치 열 타입 정리
3. TrackName 기준 point index 추가
4. summary 메타데이터 정리
5. InfoFlags 정규화 및 bit 파싱
6. Track별 point 수 집계
7. halo 연결 전용 stream catalog 생성

기본 규모
---------
- summary row 수: {track_n}
- track point row 수: {point_n}

해석상 의미
-----------
- tracks_stage1: 개별 stream knot/track point 수준 정제본
- summary_stage1: track 수준 메타데이터 정제본
- stream_catalog_stage1: halo 파이프라인 보조 라벨/필터링용 기본 카탈로그
- track_points_normalized_stage1: 후속 교차 매칭용 기준 point 테이블

권장 다음 단계
--------------
1. halo shell과의 좌표/거리 기반 보조 매칭 규칙 설계
2. stream catalog에서 availability_score 및 halo_usefulness_label 기준으로 우선순위 분류
3. 이후 second_processed_v1에서 halo 보조 라벨 생성
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    project_root = find_project_root(Path(__file__))

    input_dir = project_root / "data" / "raw" / "Our galaxy Halo Stellar Kinematics" / "galstreams" / "compiled_tracks"
    output_dir = project_root / "data" / "derived" / "Our galaxy Halo Stellar Kinematics" / "first_processed_v1" / "halo"
    ensure_dir(output_dir)

    tracks_path = input_dir / "galstreams_compilation.tracks.csv"
    summary_path = input_dir / "galstreams_compilation.summary.csv"
    mid_path = input_dir / "galstreams_compilation.mid_points.csv"
    end_path = input_dir / "galstreams_compilation.end_points.csv"

    for p in [tracks_path, summary_path, mid_path, end_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")

    tracks_raw = read_csv(tracks_path)
    summary_raw = read_csv(summary_path)
    mid_raw = read_csv(mid_path)
    end_raw = read_csv(end_path)

    tracks_stage1 = clean_point_table(tracks_raw, "tracks")
    mid_stage1 = clean_point_table(mid_raw, "mid_points")
    end_stage1 = clean_point_table(end_raw, "end_points")
    summary_stage1 = clean_summary_table(summary_raw)

    point_counts = build_point_counts(tracks_stage1, mid_stage1, end_stage1)
    stream_catalog = build_stream_catalog(summary_stage1, point_counts)
    track_points_normalized = build_track_points_normalized(tracks_stage1)

    out_tracks = output_dir / "galstreams_tracks_stage1.csv"
    out_mid = output_dir / "galstreams_mid_points_stage1.csv"
    out_end = output_dir / "galstreams_end_points_stage1.csv"
    out_summary = output_dir / "galstreams_summary_stage1.csv"
    out_catalog = output_dir / "galstreams_stream_catalog_stage1.csv"
    out_norm = output_dir / "galstreams_track_points_normalized_stage1.csv"
    out_readme = output_dir / "README_stage1.txt"
    out_manifest = output_dir / "stage1_manifest.csv"

    write_csv(tracks_stage1, out_tracks)
    write_csv(mid_stage1, out_mid)
    write_csv(end_stage1, out_end)
    write_csv(summary_stage1, out_summary)
    write_csv(stream_catalog, out_catalog)
    write_csv(track_points_normalized, out_norm)
    write_readme(out_readme, input_dir=input_dir, output_dir=output_dir, summary_df=summary_stage1, tracks_df=tracks_stage1)

    manifest_rows = []
    for p in [out_tracks, out_mid, out_end, out_summary, out_catalog, out_norm, out_readme]:
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
    for p in [out_tracks, out_mid, out_end, out_summary, out_catalog, out_norm, out_readme, out_manifest]:
        print(f" - {p}")


if __name__ == "__main__":
    main()
