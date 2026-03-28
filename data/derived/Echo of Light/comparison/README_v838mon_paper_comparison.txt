README_v838mon_paper_comparison.txt
===================================

Purpose
-------
This bundle reshapes the earlier mass/ejecta reference material into a paper-comparison
auxiliary table that can sit next to the 2002-2004 light-echo geometry tables.

Main file
---------
- v838mon_paper_comparison_auxiliary.csv

How to use
----------
This table is NOT the main light-echo geometry table.
It is an auxiliary comparison table.

Recommended comparison logic
----------------------------
1. Geometry main comparison table:
   - radius evolution
   - centre / offset evolution
   - source table and paper source
   - direct comparison with Tylenda / Crause measurements

2. Auxiliary comparison table:
   - progenitor interpretations
   - event markers
   - ejecta or shell mass bounds
   - qualitative interpretation notes

Important rules
---------------
- Do not directly merge these rows into the geometry rows as if they are the same type of measurement.
- Use 'comparison_role' and 'can_join_to_geometry' to decide how strongly a value may be linked.
- Rows marked 'no_direct_join' should remain contextual only.
- Rows marked 'phase_level_only' may be linked only at the event-phase level.
- Rows marked 'annotation_only' are interpretive notes, not fit parameters.

Why this structure matters
--------------------------
V838 Mon is not a clean supernova mass-budget case.
The 2002-2004 light echo geometry and the 2002 outburst/ejecta literature must remain
related but distinct.
