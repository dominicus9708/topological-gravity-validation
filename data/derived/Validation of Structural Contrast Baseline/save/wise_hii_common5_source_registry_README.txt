WISE H II common5 source registry usage
=====================================

Purpose
-------
This CSV is separate from final input CSVs.
It exists only to record which literature sources were used to fill rows in the common5 inputs.

Recommended storage location
----------------------------
data/derived/Validation of Structural Contrast Baseline/source_registry/wise_hii_common5/

Recommended files
-----------------
- wise_hii_common5_source_registry.csv
- optional backups / seed files

Separation rule
---------------
- input/... = pipeline-readable final input
- source_registry/... = literature/source evidence only

Column meaning
--------------
- wise_name: official WISE target identifier
- hii_region_name: human-readable region name
- track: standard / topological / both
- source_key: internal unique source label
- paper_title, authors, year, journal, doi, ads_url, arxiv_url: citation metadata
- source_type: direct_mass / proxy_log_nly / proxy_spectral_type / proxy_radio / ionizing_source / fits_service / etc.
- used_for_field: which input field(s) this source helped fill
- match_method: how this source was matched to the target
- match_quality: high / medium / low
- value_text: text value such as spectral type or comments
- value_numeric: numeric proxy or mass value if applicable
- value_unit: unit of numeric value
- notes: freeform memo

Practical rule
--------------
Keep literature-source CSV separate from final input CSV to avoid confusion about which file is executable input and which file is evidence record.
