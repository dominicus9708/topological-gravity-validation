README_V838Mon_CSVs.txt
=======================

Purpose
-------
This bundle contains three CSV files prepared for V838 Mon light-echo data organization.
These are not the final processed science tables, but structured seed files for manual verification
before later CSV refinement or spreadsheet-based comparison.

Included CSV files
------------------
1. V838Mon_sources.csv
   Description:
   - A source registry for the main papers and visual materials currently used in the V838 Mon workflow.
   - This file identifies which paper or visual asset is being used and why.

   Main source roles:
   - SRC-01: "On the light echo in V838 Mon"
     * Core theory / light-echo geometry / slab-shell model
   - SRC-02: "V838 Mon: light echo evolution and distance estimate"
     * Core measurements / radius-offset table / sheet-shell fit
   - SRC-03: "Hubble Space Telescope Observations of the Light Echoes around V838 Monocerotis"
     * HST overview / 3D dust mapping context
   - SRC-04: montage image PDF
     * Visual epoch summary
   - SRC-05: STScI/NASA video
     * Raw visual reference

2. V838Mon_epochs.csv
   Description:
   - An observation-epoch registry for the currently collected image materials.
   - Its purpose is to separate the true observation date (exposure_date) from the
     page release year or later publication page identity.

   Important rule:
   - page_release_year and exposure_date must not be treated as the same thing.
   - For this project, exposure_date is the physically relevant date for the image epoch.

   Current epoch coverage:
   - 2002-05-20
   - 2002-09-02
   - 2002-10-28
   - 2002-12-17
   - 2004-02-08
   - 2004-10-24

3. V838Mon_measurement_seed.csv
   Description:
   - A manually reviewable seed table of numerical measurements collected from the two main papers.
   - This file is intended as the starting point for later spreadsheet comparison and eventual cleaned export.

   Internal sections:
   - Tylenda_Table1 rows
     * Early HST-related outer-rim circle fit values
     * Includes radius and x/y center coordinates
   - Crause_Table4 rows
     * Longer baseline measurements
     * Includes radius and RA/Dec echo-center offsets

   Verification note:
   - The 'verify_against_pdf' column is intentionally kept.
   - Do not delete it until the values have been checked by hand against the original PDF pages.

Source basis used for these CSV files
-------------------------------------
These CSV files were prepared from the currently uploaded and discussed project materials:

- On the light echo in V838 Mon.pdf
- V838 Mon light echo evolution and distance estimate.pdf
- Hubble Space Telescope Observations of the Light Echoes.pdf
- The Expanding Light Echo of Red Supergiant Star V838 Monocerotis.pdf
- Uploaded individual epoch images:
  * V838 Monocerotis 2002 05.jpg
  * V838 Monocerotis 2002 09.jpeg
  * V838 Monocerotis 2002 12.jpeg
  * V838 Monocerotis 2004 10.jpeg

Recommended workflow
--------------------
1. Keep the original PDFs, images, and videos in raw storage.
2. Use these CSV files as human-readable seed tables.
3. Load them into a spreadsheet for row-by-row comparison.
4. Only after manual verification should a final cleaned CSV be produced for analysis.

Suggested folder placement
--------------------------
- raw/
  Original PDFs, images, video, and downloads

- working/
  These CSV seed tables, spreadsheet comparison files, and handwritten review notes

- processed/
  Final cleaned analysis CSV files after verification

Practical caution
-----------------
Do not force image dates and paper-table dates to match when they are actually different.
Keep both if necessary, and preserve the distinction in notes.

This README is specific to the V838 Mon CSV bundle.
It is not a full project-wide bibliography README.
