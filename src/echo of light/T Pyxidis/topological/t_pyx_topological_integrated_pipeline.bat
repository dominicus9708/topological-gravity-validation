@echo off
cd /d C:\Users\mincu\Desktop\topological_gravity_project
python "src\echo of light\T Pyxidis\topological\t_pyx_topological_integrated_pipeline.py" ^
  --patch-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv" ^
  --output-root "results\Echo of Light\output\T Pyxidis\topological\echo_medium_centered_integrated" ^
  --d-bg 3.0 ^
  --lambda-sigma 0.8 ^
  --radius-power 3.0 ^
  --w-medium-structure 0.60 ^
  --w-observer-gap 0.40 ^
  --w-gradient-driver 0.40 ^
  --w-gap-driver 0.35 ^
  --w-delay-structure 0.25
pause