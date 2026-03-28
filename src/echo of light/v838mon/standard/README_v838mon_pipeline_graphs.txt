README_v838mon_pipeline_graphs.txt
==================================

Purpose
-------
This script is the graph-producing version of the V838 Mon standard-reference pipeline.

Default input folder
--------------------
data/derived/Echo of Light/input

Default output root
-------------------
output/V838Mon

Output pattern
--------------
output/V838Mon/YYYYMMDD_HHMMSS/

What it writes
--------------
Tables:
- verified_copy.csv
- model_input_all_copy.csv
- model_input_tylenda_copy.csv
- model_input_crause_copy.csv
- verified_tylenda_only.csv
- verified_crause_only.csv
- dataset_row_counts.csv
- summary.json
- README_results.txt

Plots:
- plots/crause_radius_vs_time.png
- plots/tylenda_radius_vs_time.png
- plots/tylenda_center_shift.png
- plots/crause_center_offset.png

Interpretation note
-------------------
This version is still a standard-reference reproduction step.
It is meant to reproduce and visualize the prepared Tylenda/Crause input data
before any topological-gravity correction term is added.
