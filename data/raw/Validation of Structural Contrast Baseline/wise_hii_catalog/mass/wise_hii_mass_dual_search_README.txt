네 번째 논문 WISE H II mass raw 작업 지침 (direct + proxy 동시 탐색)
=================================================================

핵심 원칙
---------
- direct 질량 탐색과 proxy 탐색을 순차가 아니라 병행한다.
- raw 단계에서 source를 모은 뒤, working -> derived staging으로 올린다.
- final input은 마지막에 별도 builder로만 만든다.

권장 폴더 구조
--------------
data/raw/Validation of Structural Contrast Baseline/wise_hii_catalog/mass/
├─ raw/
│  ├─ direct/
│  │  └─ 직접 질량 또는 질량 범위가 적힌 논문 정리 csv
│  ├─ proxy_log_nly/
│  │  └─ log Nly, ionizing photon 관련 논문 정리 csv
│  ├─ proxy_spectral_type/
│  │  └─ spectral type 관련 논문 정리 csv
│  ├─ proxy_radio/
│  │  └─ radio continuum 관련 논문 정리 csv
│  └─ proxy_ionizing_source/
│     └─ ionizing source 식별 논문 정리 csv
├─ wise_hii_mass_search_registry.csv
├─ wise_hii_mass_dual_search_input.csv
├─ wise_hii_mass_dual_candidates_initial.csv
├─ wise_hii_mass_dual_source_registry.csv
└─ wise_hii_mass_dual_init_manifest.txt

먼저 할 일
----------
1. wise_hii_mass_dual_search_input.csv를 생성한다.
2. raw/direct, raw/proxy_log_nly, raw/proxy_spectral_type, raw/proxy_radio, raw/proxy_ionizing_source 폴더를 만든다.
3. 논문을 찾으면 해당 폴더에 csv를 저장한다.
4. source_track 값으로 direct/proxy 종류를 명시한다.
5. 이후 working update 스크립트가 이 자료를 읽도록 확장한다.

프록시 우선순위
---------------
1. direct_mass
2. proxy_log_nly
3. proxy_spectral_type
4. proxy_radio
5. proxy_ionizing_source

최소 권장 열
------------
source_key, wise_name, matched_object_name, paper_title, authors, year, journal, doi,
ads_url, arxiv_url, source_track, source_type, match_method, match_quality,
mass_field_description, mass_value_msun, mass_range_lower_msun, mass_range_upper_msun,
proxy_kind, proxy_value, proxy_value_unit, log_nly, spectral_type,
radio_proxy_available, distance_kpc, matching_notes

주의
----
- direct 질량이 없다고 proxy 탐색을 미루지 않는다.
- proxy 값은 direct mass와 같은 지위로 과장하지 않는다.
- source_track와 proxy_kind를 반드시 남겨서 나중에 구분 가능하게 한다.
