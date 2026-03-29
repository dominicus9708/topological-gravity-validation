README_SN2014J_extracted.txt
============================

Purpose
-------
This bundle organizes a first-pass extracted-data set for SN 2014J.

What is included
----------------
1. SN2014J_source_registry.csv
   - source list with role and URL

2. SN2014J_epochs_extracted.csv
   - directly extractable dated milestones from accessible sources
   - discovery date
   - early HST UV epoch range
   - light-echo imaging start/end dates

3. SN2014J_auxiliary_extracted.csv
   - auxiliary comparison values
   - distance to M82
   - B-band maximum JD
   - total ejected mass
   - Ni-56 mass
   - light-echo dust-cloud distance range

Important limitation
--------------------
This is not yet a detailed radius/center-shift measurement table comparable to the V838 Mon Crause/Tylenda CSVs.
I did not find a readily accessible paper table with per-epoch light-echo radii in the sources gathered here.
So this bundle should be treated as:

- source registry
- epoch reference table
- auxiliary comparison table

Recommended next use
--------------------
Use this bundle to start the SN 2014J raw/comparison structure.
If a detailed light-echo measurement paper or archival measurement table is later located,
it can be added as:
- SN2014J_measurements_extracted.csv
without changing the current registry structure.
