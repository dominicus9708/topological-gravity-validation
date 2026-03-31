@echo off
cd /d C:\Users\mincu\Desktop\topological_gravity_project
python "src\echo of light\T Pyxidis\standard\t_pyx_standard_integrated_pipeline.py" ^
  --patch-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_echo_patches_original_paper.csv" ^
  --interpretation-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_interpretation_table.csv" ^
  --timeline-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_observation_timeline_table.csv" ^
  --output-root "results\Echo of Light\output\T Pyxidis\standard\integrated_standard"
pause