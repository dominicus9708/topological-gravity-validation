README_v838mon_topological_pipeline.txt
=======================================

Purpose
-------
This is the first topological-side trial pipeline for V838 Mon.

Script location
---------------
src/echo of light/v838mon/topological/v838mon_topological_pipeline.py

Default input folder
--------------------
data/derived/Echo of Light/input

Default output root
-------------------
results/Echo of Light/output/V838Mon/topological

Output pattern
--------------
results/Echo of Light/output/V838Mon/topological/YYYYMMDD_HHMMSS/

What this version does
----------------------
- loads the verified and model-input V838 Mon CSV files
- copies them into the timestamped output folder
- creates topological trial CSV files for Tylenda and Crause
- applies configurable trial correction terms using sigma_proxy and dsigma_dt_proxy if available
- creates comparison plots for radius and center/offset quantities

Important interpretation note
-----------------------------
This version is not the final topological-gravity derivation.
It is a trial comparison framework.

Proxy behavior
--------------
If sigma_proxy or dsigma_dt_proxy do not exist in the input CSV files,
this version creates them internally as zeros.
That means the first run can still be used to verify the pipeline structure.
