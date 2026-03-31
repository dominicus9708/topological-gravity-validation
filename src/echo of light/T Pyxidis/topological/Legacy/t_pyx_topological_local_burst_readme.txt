T Pyxidis topological local-burst pipeline
=======================================

Purpose
-------
This version adds a local burst-induced low-effective-dimension term to the earlier
T Pyxidis topological trial.

Interpretive structure
----------------------
- sigma_obs:
  observer-side geometric distortion term based on theta and |z|
- sigma_burst_local:
  local burst-induced low-effective-dimension term near T Pyxidis itself
- sigma_total:
  weighted combination of sigma_obs and sigma_burst_local

This version does NOT model the entire T Pyxidis-Earth path.

Expected project placement
--------------------------
src/echo of light/T Pyxidis/topological/t_pyx_topological_local_burst_pipeline.py
batch/run_t_pyx_topological_local_burst.bat

Expected input
--------------
data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_echo_patches_original_paper.csv

Default output
--------------
results/Echo of Light/output/T Pyxidis/topological_local_burst/<timestamp>/

Produced files
--------------
- t_pyx_topological_local_burst_patch_comparison.csv
- t_pyx_topological_local_burst_circular_layout.png
- t_pyx_topological_local_burst_delay_theta_comparison.png
- t_pyx_topological_local_burst_components.png
- run_summary.txt
