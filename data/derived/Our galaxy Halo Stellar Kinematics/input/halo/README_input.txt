galstreams halo final input v1
==============================

작성 시각
---------
2026-04-07T10:30:08

입력 경로
---------
C:\Users\mincu\Desktop\topological_gravity_project\data\derived\Our galaxy Halo Stellar Kinematics\first_processed_v1\halo

출력 경로
---------
C:\Users\mincu\Desktop\topological_gravity_project\data\derived\Our galaxy Halo Stellar Kinematics\input\halo

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
- stream catalog row 수: 217
- overlay candidate row 수: 414799

권장 후속 단계
--------------
1. halo skeleton 에서 overlay_candidates_input 을 읽는 경로 테스트
2. shell summary 와 distance_band 기반의 느슨한 겹침 점검
3. 필요시 topological shell 결과와 stream label 을 연결하는 보조 파이프라인 작성
