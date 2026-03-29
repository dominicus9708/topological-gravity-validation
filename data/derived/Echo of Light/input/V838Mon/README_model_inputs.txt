README_model_inputs.txt
=======================

Purpose
-------
This folder contains model-input CSV files derived from the verified V838 Mon processed measurement table.

Files
-----
1. v838mon_model_input_all.csv
   - Combined model-input table
   - Keeps both Tylenda and Crause rows
   - Useful for general comparison work
   - IMPORTANT: time_reference must be respected because the two datasets use different day-count conventions

2. v838mon_model_input_tylenda.csv
   - Tylenda-only input table
   - Best for early HST-based outer-rim and center-shift analysis

3. v838mon_model_input_crause.csv
   - Crause-only input table
   - Best for longer-baseline radius and center-offset analysis
   - Most useful for sheet/shell comparison over time

4. v838mon_measurements_verified.csv
   - Full verified table
   - Includes additional columns kept from the processed stage
   - This is the best archival bridge between processed data and model input data

Column meaning
--------------
- dataset: source group (Tylenda or Crause)
- observation_date: observation date as stored in the processed table
- time_value: numerical time value from the source table
- time_reference: explicit definition of what the time value means
- radius_arcsec: observed echo radius in arcseconds
- ra_offset_arcsec / dec_offset_arcsec: echo center offsets when provided in RA/Dec style
- x_center_arcsec / y_center_arcsec: center coordinates when provided in x/y style
- source_id: source registry identifier

Important caution
-----------------
Do not merge or reinterpret Tylenda and Crause time systems without explicitly converting them.
They are intentionally preserved as separate time_reference conventions.

Recommended next step
---------------------
Use:
- v838mon_model_input_tylenda.csv for Tylenda-only geometric checks
- v838mon_model_input_crause.csv for Crause-only sheet/shell baseline comparison
- then compute residuals only after choosing one time convention at a time
