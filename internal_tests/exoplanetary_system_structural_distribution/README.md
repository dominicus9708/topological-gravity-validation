# Exoplanetary System Structural Distribution Test

이 패키지는 내부 토의용 소형 천체관측 테스트입니다. 목적은 외계행성계의 질량-거리 분포를 `D_w`형 구조 기술량과 `sigma_system`으로 재표현할 수 있는지 확인하는 것입니다.

## 성격

- 논문용 최종 검증이 아닙니다.
- 위상중력의 관측 증명이 아닙니다.
- NASA Exoplanet Archive `PSCompPars` 표의 공개 관측/편집 자료를 사용합니다.
- 결과는 내부 노트, 후속 토의, 구조량 후보 선별에 쓰는 것이 적절합니다.

## 실행 위치

기존 저장소 루트에서 실행하는 것을 기준으로 합니다.

```bat
cd C:\Users\mincu\Desktop\topological_gravity_project
python internal_tests\exoplanetary_system_structural_distribution\run_exoplanet_structural_test.py --root .
```

## 출력 구조

```text
data/raw/Exoplanetary System Structural Distribution/
data/derived/Exoplanetary System Structural Distribution/input/
results/Exoplanetary System Structural Distribution/output/YYYYMMDD_HHMMSS/
```

주요 출력:

```text
standard_planet_level_working.csv
topological_system_level_working.csv
standard_orbital_span_vs_mass_concentration.png
topological_structural_map.png
representative_high_contrast_systems.png
exoplanet_structural_test_summary.txt
```

## 핵심 계산 사슬

```text
mass proxy + semi-major axis
→ cumulative mass slope alpha_obs
→ D_w_standard
→ D_bg_by_n
→ sigma_system = D_w_standard - D_bg_by_n
```

## 해석 수위

`D_w_standard`는 행성계 내부의 질량-거리 구조 기술량입니다. 정수 차원이나 실제 공간 차원을 뜻하지 않습니다.

`sigma_system`은 같은 유효 행성 수를 가진 계의 중앙 배경값과 비교한 구조 대비입니다. 중력 측정값이 아닙니다.
