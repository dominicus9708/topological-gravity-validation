README_v838mon_pipeline_updated.txt
===================================

Why this updated version exists
-------------------------------
The actual project folder listing shows that:

data/derived/Echo of Light/input

currently contains:
- README_model_inputs.txt
- v838mon_measurements_verified.csv
- v838mon_model_input_all.csv
- v838mon_model_input_tylenda.csv
- v838mon_model_input_crause.csv

Therefore, this updated pipeline uses those files directly as defaults.

Default input folder
--------------------
data/derived/Echo of Light/input

Default results folder
----------------------
results/Echo of Light/output

What it does
------------
- loads the verified table and the three model-input CSV files
- copies them into a timestamped results folder
- separates verified rows into Tylenda-only and Crause-only copies
- writes dataset_row_counts.csv
- writes summary.json and README_results.txt

Important
---------
This version is matched to the real project file names listed in 목록.txt.
