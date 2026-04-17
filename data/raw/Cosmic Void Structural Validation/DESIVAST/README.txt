Dataset: DESIVAST Catalog (DESI DR1)
Category: void_catalog_primary
Provider: DESI Data Release 1

Recommended use
---------------
Primary void catalog for representative cosmic void selection and stacked baseline construction

Main files or products
----------------------
DESIVAST_BGS_VOLLIM_VoidFinder_NGC.fits; DESIVAST_BGS_VOLLIM_VoidFinder_SGC.fits; DESIVAST_BGS_VOLLIM_V2_VIDE_NGC.fits; DESIVAST_BGS_VOLLIM_V2_VIDE_SGC.fits; DESIVAST_BGS_VOLLIM_V2_REVOLVER_NGC.fits; DESIVAST_BGS_VOLLIM_V2_REVOLVER_SGC.fits

Key columns or quantities
-------------------------
EFFECTIVE_RADIUS; EFFECTIVE_RADIUS_UNCERT; EDGE_AREA; TOT_AREA; algorithm-specific void geometry columns

Primary documentation
---------------------
https://data.desi.lbl.gov/doc/releases/dr1/vac/desivast/

Mass info available directly
----------------------------
No direct mass column in void catalog itself

Volume info available
---------------------
Yes

Suggested raw subpath in project design
---------------------------------------
raw/Cosmic Void Structural Validation/DESIVAST/

Related sources
---------------
- [SRC001] DESIVAST Catalog - DESI Data
  DESIVAST Catalog documentation page
  https://data.desi.lbl.gov/doc/releases/dr1/vac/desivast/
- [SRC004] DR1 (latest) - DESI Data
  DESI DR1 release overview
  https://data.desi.lbl.gov/doc/releases/dr1/

Notes
-----
Best primary starting point because it offers multiple void-finder algorithms on a common DESI Y1 BGS Bright base

Manual download note
--------------------
This folder may remain empty until the user manually downloads the actual dataset files from the official page above or later adds a downloader script with direct file URLs.
