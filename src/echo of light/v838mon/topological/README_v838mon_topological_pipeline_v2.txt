README_v838mon_topological_pipeline_v2.txt
==========================================

Purpose
-------
This is the second topological-side trial pipeline for V838 Mon.

Main improvement over v1
------------------------
If sigma_proxy is not already present, v2 computes it automatically from
observed center-offset magnitude.

Proxy definitions
-----------------
- Crause:
  sigma_proxy_raw = sqrt(ra_offset_arcsec^2 + dec_offset_arcsec^2)

- Tylenda:
  sigma_proxy_raw = sqrt(x_center_arcsec^2 + y_center_arcsec^2)

Then:
- sigma_proxy = normalized sigma_proxy_raw in [0, 1]
- dsigma_dt_proxy = finite difference derivative over time_value

Output root
-----------
results/Echo of Light/output/V838Mon/topological/YYYYMMDD_HHMMSS/

Interpretation note
-------------------
This is still a heuristic trial framework.
It is stronger than the zero-proxy v1, but it is not yet a final physical derivation.
