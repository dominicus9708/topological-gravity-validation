T Pyx standard baseline package
=============================

Purpose
-------
This package separates the earlier skeleton test from the new standard baseline.
The standard baseline is intended to reconstruct the original-paper interpretation layer
more explicitly.

Files
-----
1. t_pyx_standard_observation_epochs.csv
   - Observation/event timeline.
   - Contains eruption reference, HST public image epochs, and Halpha structure epochs.

2. t_pyx_standard_geometry_reference.csv
   - Global geometry reference table.
   - Contains the representative clumpy-ring geometry from the light-echo interpretation.

3. t_pyx_standard_baseline.py
   - Pipeline that reads the two CSVs and creates a baseline plot plus processed copies.

4. run_t_pyx_standard_baseline.bat
   - Example Windows batch launcher.

Placement
---------
Put the files here in the project:

data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_observation_epochs.csv
data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_geometry_reference.csv
src/echo of light/T Pyxidis/standard/t_pyx_standard_baseline.py
batch/run_t_pyx_standard_baseline.bat

Important caution
-----------------
The baseline is still constrained by the currently available public/light-echo summary information.
It is better than the earlier skeleton because it separates observation epochs from geometry references,
but it is not yet a dense epoch-by-epoch direct-radius measurement table.
