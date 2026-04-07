galstreams first processed v1
=================================

작성 시각
---------
2026-04-07T10:12:52

입력 경로
---------
C:\Users\mincu\Desktop\topological_gravity_project\data\raw\Our galaxy Halo Stellar Kinematics\galstreams\compiled_tracks

출력 경로
---------
C:\Users\mincu\Desktop\topological_gravity_project\data\derived\Our galaxy Halo Stellar Kinematics\first_processed_v1\halo

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
- summary row 수: 217
- track point row 수: 414799

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
