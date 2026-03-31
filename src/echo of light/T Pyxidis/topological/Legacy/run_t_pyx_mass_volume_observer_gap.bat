@echo off
cd /d C:\Users\mincu\Desktop\topological_gravity_project
python "src\echo of light\T Pyxidis\topological\t_pyx_mass_volume_observer_gap_pipeline.py" ^
  --patch-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv" ^
  --output-root "results\Echo of Light\output\T Pyxidis\topological\mass_volume_observer_gap" ^
  --d-bg 3.0 ^
  --lambda-sigma 0.8 ^
  --radius-power 3.0 ^
  --w-mass-volume 0.35 ^
  --w-local-burst 0.35 ^
  --w-observer-gap 0.30 ^
  --local-strength 1.0
pause
