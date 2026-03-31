@echo off
cd /d C:\Users\mincu\Desktop\topological_gravity_project
python "src\echo of light\T Pyxidis\standard\t_pyx_standard_original_paper_pipeline.py" ^
  --observation-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_observation_epochs_original_paper.csv" ^
  --echo-patch-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv" ^
  --output-dir "results\Echo of Light\output\T Pyxidis\standard"
pause
