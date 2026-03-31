T Pyx standard pipeline setup
============================

Recommended project placement
-----------------------------
Python:
src/echo of light/T Pyxidis/standard/t_pyx_standard_pipeline.py

Input CSV:
data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_input_seed.csv

Output base folder:
results/Echo of Light/output/T Pyxidis/standard

Batch:
batch/run_t_pyx_standard.bat

What this pipeline currently does
---------------------------------
1. Reads the second-processed standard input seed CSV.
2. Validates required columns.
3. Creates a timestamped output folder.
4. Copies the processed input CSV into the output folder.
5. Filters rows with ready_for_standard_plot = 1.
6. Produces:
   - t_pyx_standard_input_processed.csv
   - t_pyx_standard_plot_input.csv
   - t_pyx_standard_radius_vs_time.png
   - run_summary.txt

Command example
---------------
python "src\echo of light\T Pyxidis\standard\t_pyx_standard_pipeline.py" ^
  --input-csv "data\derived\Echo of Light\input\T Pyxidis\t_pyx_standard_input_seed.csv" ^
  --output-dir "results\Echo of Light\output\T Pyxidis\standard"

Caution
-------
The current radius values are seed-scale placeholders based on representative literature structure scale.
So this pipeline is appropriate as a first standard-baseline skeleton, not yet as a final paper-grade measured-radius pipeline.
