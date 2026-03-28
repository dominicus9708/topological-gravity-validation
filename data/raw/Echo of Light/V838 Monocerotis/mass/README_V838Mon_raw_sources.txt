README_V838Mon_raw_sources.txt
==============================

Purpose
-------
This bundle organizes source information for V838 Mon in a way that can be kept
in a raw/reference folder before later structured pipeline use.

Important interpretation note
-----------------------------
V838 Mon is not usually treated as a standard supernova case in the literature.
For this reason, the most useful raw-source split is:

1. light echo direct sources
2. mass/ejecta-related sources
3. public explanatory or image-series references

Files in this bundle
--------------------
1. V838Mon_source_registry.csv
   - master source index
   - includes category, role, main points, and source URL

2. V838Mon_light_echo_raw_notes.txt
   - notes on the direct light-echo references most relevant to the 2002-2004 phenomenon

3. V838Mon_mass_ejecta_raw_notes.txt
   - notes on outburst character and ejecta/shell-mass-related references

Recommended project placement
-----------------------------
data/raw/Echo of Light/V838Mon/
or
docs/raw_references/V838Mon/

Recommended internal usage
--------------------------
- Use light echo direct sources as the standard-reference baseline layer
- Use mass/ejecta-related sources as auxiliary parameters or constraints
- Do not treat the light echo dust and the 2002 ejected material as automatically identical

Practical note
--------------
This bundle is a source organization layer.
It is not yet a model input table.
