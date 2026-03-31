@echo off
cd /d C:\Users\mincu\Desktop\topological_gravity_project
python "src\echo of light\T Pyxidis\standard\t_pyx_standard_baseline.py" ^
  --observation-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_observation_epochs.csv" ^
  --geometry-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_geometry_reference.csv" ^
  --output-dir "results\Echo of Light\output\T Pyxidis\standard\baseline"
pause
