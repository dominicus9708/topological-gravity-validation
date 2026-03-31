T Pyxidis topological observer-gap pipeline
========================================

Purpose
-------
This version reconstructs the T Pyxidis topological trial in a way that is closer to
the user's theory framing.

Interpretive structure
----------------------
- sigma_local_burst:
  local low-effective-dimension term caused by the eruptive disturbance itself
- sigma_observer_gap:
  external-observer describability-gap term inspired by the difference between
  internal and external describability in Paper 1
- sigma_total:
  weighted combination of the two terms

This version does NOT model the entire T Pyxidis-Earth path.
It only models the locally disturbed region and the external-observer description gap.

Expected project placement
--------------------------
src/echo of light/T Pyxidis/topological/t_pyx_topological_observer_gap_pipeline.py
batch/run_t_pyx_topological_observer_gap.bat

Expected input
--------------
data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_echo_patches_original_paper.csv

Default output
--------------
results/Echo of Light/output/T Pyxidis/topological_observer_gap/<timestamp>/

Produced files
--------------
- t_pyx_topological_observer_gap_patch_comparison.csv
- t_pyx_topological_observer_gap_circular_layout.png
- t_pyx_topological_observer_gap_delay_theta_comparison.png
- t_pyx_topological_observer_gap_components.png
- t_pyx_topological_observer_gap_delay_driver.png
- run_summary.txt
