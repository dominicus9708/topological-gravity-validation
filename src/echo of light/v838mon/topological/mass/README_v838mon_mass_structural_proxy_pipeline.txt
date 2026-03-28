README_v838mon_mass_structural_proxy_pipeline.txt
=================================================

Purpose
-------
This pipeline builds a mass-informed structural proxy for V838 Mon by combining:
- auxiliary mass/ejecta comparison rows
- observed light-echo radius rows

Input files
-----------
Comparison:
- data/derived/Echo of Light/comparison/v838mon_paper_comparison_auxiliary.csv

Light-echo input:
- data/derived/Echo of Light/input/v838mon_model_input_tylenda.csv
- data/derived/Echo of Light/input/v838mon_model_input_crause.csv

Output root
-----------
results/Echo of Light/output/V838Mon/topological/mass/YYYYMMDD_HHMMSS/

Interpretation notes
--------------------
- mass_proxy_raw:
  cumulative heuristic event-strength proxy from literature rows

- echo_volume_proxy_raw:
  observed radius ^ radius_power
  This is an echo-geometric volume proxy, not a direct ejecta volume.

- sigma_mass_volume:
  normalized mass/volume structural proxy

- Dw_trial_mass_volume:
  heuristic trial effective dimension based on sigma_mass_volume

This pipeline is meant for consistency-oriented comparison, not strict derivation.
