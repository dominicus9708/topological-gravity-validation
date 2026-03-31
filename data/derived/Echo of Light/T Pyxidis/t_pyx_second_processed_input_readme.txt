T Pyxidis second processed input set
===================================

Target folder
-------------
data/derived/Echo of Light/input/T Pyxidis/

Files
-----
1. t_pyx_standard_input_seed.csv
   - First-pass standard pipeline input seed.
   - Contains eruption reference, HST public light-echo image epochs, and Halpha structure-event epochs.
   - Only the HST public image epochs currently carry a representative radius scale (5 arcsec).
   - That 5 arcsec value is a literature representative structural scale, not a strict per-epoch measured radius table.

2. t_pyx_topological_input_mass_volume_seed.csv
   - First-pass topological / mass-volume input seed.
   - Combines structure-event epochs with mass anchors and volume proxies.
   - Suitable for early trial pipeline design where synchronized direct mass-volume time series do not exist.

Important caution
-----------------
These are input-seed CSVs, not final cleaned measurement tables.
They are designed to allow:
- standard baseline skeleton plotting,
- structure-event timeline plotting,
- mass-anchor assisted topological trial design.

Recommended next filename placement
-----------------------------------
data/derived/Echo of Light/input/T Pyxidis/t_pyx_standard_input_seed.csv
data/derived/Echo of Light/input/T Pyxidis/t_pyx_topological_input_mass_volume_seed.csv
