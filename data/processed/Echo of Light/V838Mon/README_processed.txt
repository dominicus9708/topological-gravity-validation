README_processed.txt
====================

Folder purpose
--------------
This folder contains processed-stage CSV files for the V838 Mon light-echo workflow.

Intended placement
------------------
data/processed/Echo of Light/V838Mon/

Files
-----
1. v838mon_sources_processed.csv
   - processed source registry for the papers and visual materials

2. v838mon_epochs_processed.csv
   - normalized epoch registry
   - separates exposure_date from page_release_year
   - keeps image presence explicit

3. v838mon_measurements_processed.csv
   - normalized measurement table derived from the uploaded seed CSV
   - keeps Tylenda and Crause rows together
   - preserves different time-count conventions through time_reference

Processing rules
----------------
- observation_date and exposure_date are kept separate where relevant
- page_release_year is not treated as equivalent to the physical observation epoch
- Tylenda and Crause measurements are kept together, but dataset identity is preserved
- manual_verification_status remains 'seed_verified_pending'
- these files are processed project-side organization files, not replacements for the raw papers

Recommended next step
---------------------
Open v838mon_measurements_processed.csv in a spreadsheet and verify each row directly against:
- On the light echo in V838 Mon.pdf
- V838 Mon: light echo evolution and distance estimate.pdf

Only after that should a final analysis CSV be exported for direct modeling work.
