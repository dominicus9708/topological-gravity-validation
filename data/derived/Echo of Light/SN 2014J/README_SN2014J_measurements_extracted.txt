README_SN2014J_measurements_extracted.txt
=========================================

Purpose
-------
This bundle provides the first measurement-oriented extracted CSV for SN 2014J.

What is included
----------------
1. SN2014J_measurements_extracted.csv
   - directly usable measurement-like rows
   - includes:
     * Crotts 2015 direct arcsec values
     * Crotts 2015 derived physical radius / foreground distance context
     * Yang 2017 multiple-epoch markers
     * Yang 2017 foreground cloud distance range

2. SN2014J_measurement_sources.csv
   - source registry specifically for the measurement CSV

Important limitation
--------------------
This is still not as complete as the V838 Mon Crause/Tylenda tables.

Why
---
- Crotts 2015 gives directly useful radius values in accessible preview form
- Yang 2017 accessible abstract confirms multiple epochs and foreground-distance structure
- But a full per-epoch radius table for all epochs is still not yet available in the gathered accessible sources

How to use
----------
Treat this as a first-pass measurement CSV.
Good roles:
- standard baseline seed
- comparison/reference CSV
- source for later manual augmentation

Recommended next step
---------------------
Use this file to start:
- SN2014J standard pipeline skeleton
or
- SN2014J comparison folder seed

Then later, if a fuller paper table or figure digitization is secured, add:
- more rows with direct radius measurements
- component-specific brightness or width values
without changing the schema
