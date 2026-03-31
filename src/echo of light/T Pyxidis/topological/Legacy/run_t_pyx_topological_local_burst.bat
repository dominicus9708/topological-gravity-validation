@echo off
cd /d C:\Users\mincu\Desktop\topological_gravity_project
python "src\echo of light\T Pyxidis\topological\t_pyx_topological_local_burst_pipeline.py" ^
  --patch-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv" ^
  --output-root "results\Echo of Light\output" ^
  --object-name "T Pyxidis" ^
  --mode-name "topological_local_burst" ^
  --w-obs 0.65 ^
  --w-burst 0.35 ^
  --burst-strength 1.0
pause
