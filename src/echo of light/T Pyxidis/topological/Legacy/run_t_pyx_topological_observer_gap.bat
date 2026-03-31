@echo off
cd /d C:\Users\mincu\Desktop\topological_gravity_project
python "src\echo of light\T Pyxidis\topological\t_pyx_topological_observer_gap_pipeline.py" ^
  --patch-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv" ^
  --output-root "results\Echo of Light\output" ^
  --object-name "T Pyxidis" ^
  --mode-name "topological_observer_gap" ^
  --w-local 0.60 ^
  --w-gap 0.40 ^
  --local-strength 1.0
pause
