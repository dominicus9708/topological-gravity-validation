T Pyx first processed timeline
=============================

Purpose
-------
This is the first processed input draft for the T Pyxidis light-echo / structure project.
It is intended as an intermediate CSV for pipeline design, not as a final measurement table.

Design rules
------------
1. Structure epochs are treated as the primary timeline axis.
2. Mass values are mostly interval anchors or integrated constraints, not dense epoch-by-epoch measurements.
3. Volume values are mostly indirect proxies derived from representative structure scale, not direct measured physical volumes.
4. Public Hubble image dates are retained explicitly because they are useful for standard-baseline plotting.
5. Historical shell mass is kept as a separate anchor because it is not synchronized to the 2011 epochs.

Important cautions
------------------
- The '5 arcsec' ring radius is a representative literature structural scale, not a per-epoch direct measurement.
- 'arcsec3_proxy' is only a simple structural proxy (= radius^3 in angular units), not a literal physical volume.
- T Pyx does not currently provide a V838 Mon-like fully synchronized mass-volume time series.
- This CSV is therefore suitable for:
  * standard baseline skeleton,
  * structure-event timeline plotting,
  * mass-anchor assisted topological trial design,
  but not for claiming a direct fully observed mass-volume evolution curve.

Recommended next step
---------------------
Build two derived inputs from this file:
1. t_pyx_standard_seed.csv
   - keep only eruption reference + public HST image epochs + any future measured radii
2. t_pyx_mass_anchor_comparison.csv
   - keep mass-related anchors, delayed ejection notes, and historical shell values

Files created
-------------
- t_pyx_first_processed_timeline.csv
- t_pyx_first_processed_readme.txt