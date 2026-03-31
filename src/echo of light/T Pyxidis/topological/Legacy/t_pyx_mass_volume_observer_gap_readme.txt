T Pyxidis mass-volume + observer-gap pipeline
==========================================

Files
-----
1. t_pyx_mass_volume_observer_gap_pipeline.py
   - Main Python script for the current T Pyxidis theory-aligned trial.
   - Keeps the T Pyx patch-based interface.
   - Imports V838Mon-style mass-volume proxy logic into T Pyx.
   - Produces a circular layout graph, a family-separated delay-theta comparison,
     a structural-components graph, and a delay-driver graph.

2. run_t_pyx_mass_volume_observer_gap.bat
   - Example Windows batch file.

Expected project placement
--------------------------
src/echo of light/T Pyxidis/topological/t_pyx_mass_volume_observer_gap_pipeline.py
batch/run_t_pyx_mass_volume_observer_gap.bat

Expected input
--------------
data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_echo_patches_original_paper.csv

Default output
--------------
results/Echo of Light/output/T Pyxidis/topological/mass_volume_observer_gap/<timestamp>/

Produced files
--------------
- t_pyx_mass_volume_observer_gap_patch_comparison.csv
- plots/t_pyx_mass_volume_observer_gap_circular_layout.png
- plots/t_pyx_mass_volume_observer_gap_delay_theta_comparison.png
- plots/t_pyx_mass_volume_observer_gap_components.png
- plots/t_pyx_mass_volume_observer_gap_delay_driver.png
- summary.json
- README_results.txt

Interpretation note
-------------------
This pipeline is not identical to the V838 Mon code, but it is reconstructed to preserve
the same research logic:
- structural disturbance
- mass-volume proxy
- trial effective dimension
- standard-vs-topological comparison
