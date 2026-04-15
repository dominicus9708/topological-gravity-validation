WISE H II common5 radius/source registry
=====================================

Purpose
-------
This CSV is separate from the final input CSVs.
It records which literature pages, data services, or catalog/service sources were used
to determine or support radius-related geometry and radial-profile extraction.

Recommended storage location
----------------------------
data/derived/Validation of Structural Contrast Baseline/source_registry/wise_hii_common5/

Recommended files
-----------------
- wise_hii_common5_radius_source_registry.csv
- optional backups / seed files

Separation rule
---------------
- input/... = pipeline-readable final input and radial profile CSVs
- source_registry/... = evidence record for radius / profile generation

Practical rule
--------------
Do not mix executable input CSVs with source-evidence CSVs.
For radius/profile work, service-level sources are allowed in addition to paper citations.
