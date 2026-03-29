README_v838mon_chi_square_pipeline.txt
======================================

Purpose
-------
This pipeline computes radius-based chi-square metrics for V838 Mon by comparing:
- observed radius
- standard baseline radius
- topological mass-volume trial radius

Inputs
------
Observed input tables:
- data/derived/Echo of Light/input/v838mon_model_input_tylenda.csv
- data/derived/Echo of Light/input/v838mon_model_input_crause.csv

Mass/topological trial inputs:
- results/Echo of Light/output/V838Mon/topological/mass/<timestamp>/v838mon_tylenda_mass_volume_trial.csv
- results/Echo of Light/output/V838Mon/topological/mass/<timestamp>/v838mon_crause_mass_volume_trial.csv

Output root
-----------
results/Echo of Light/output/V838Mon/topological/chi_square/YYYYMMDD_HHMMSS/

Important note
--------------
The standard baseline is intentionally a trivial observational baseline copy.
Therefore its chi-square is zero by construction.

This means the practical use of this pipeline is:
- to evaluate whether the topological radius trial remains within acceptable residual size
- to compare residual magnitude under assumed observation uncertainty
- not to claim superiority over the standard geometric interpretation from this number alone
