# SPARC Structure Test 32

## Purpose

This folder contains a dedicated crossmatched test set for the
structure-linked galaxy rotation analysis used in the fourth paper
of the Topological Gravity project.

The purpose of this dataset is **not** to replace the full SPARC sample,
but to construct a **structure-enriched subsample** in which
SPARC rotation-curve data are combined with external edge-on
galaxy structural measurements.

In this design, **SPARC remains the base dynamical dataset**,
and external catalogs are used only to append
**structural quantities not originally contained in SPARC**,
such as disk thickness-related parameters.

---

## Conceptual Role in This Project

This dataset is intended for a **consistency-oriented structural test**
of the working hypothesis that galactic rotation behavior may be
partly correlated with structural or thickness-related properties
of disk galaxies.

This folder therefore represents a **derived experimental dataset**
for internal analysis and methodological validation,
rather than a direct public survey product.

---

## Source Data

### 1. SPARC rotation-curve base sample
- Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016)
- *SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves*
- *The Astronomical Journal*, **152**, 157
- DOI: 10.3847/0004-6256/152/6/157

SPARC is used here as the **base dynamical dataset**.

### 2. Bizyaev et al. (2014) edge-on structural catalog
- Bizyaev, D. et al. (2014)
- *Structural parameters of true edge-on galaxies*
- *The Astrophysical Journal*, **787**, 24
- VizieR catalog: `J/ApJ/787/24`

This catalog is used as a source of
**structural / thickness-related parameters**
for crossmatched galaxies.

### 3. Bizyaev & Mitronova (2002) edge-on structural catalog
- Bizyaev, D., & Mitronova, S. (2002)
- *Photometric parameters of edge-on galaxies from 2MASS observations*
- *Astronomy & Astrophysics*, **389**, 795–801
- DOI: 10.1051/0004-6361:20020633

This catalog is used as an additional source of
**edge-on disk structural information**.

### VizieR acknowledgement
This research made use of the VizieR catalogue access tool,
CDS, Strasbourg, France.

---

## Folder Contents

### `structure_test32_master_catalog.csv`
Master index of the final unique structure-linked SPARC galaxies
used in this test set.

This file summarizes:
- matched SPARC galaxy name
- selected source catalog
- duplicate status
- source object names
- input/output file locations
- selected external structural quantities

### `structure_test32_summary.json`
Machine-readable summary of the exported dataset.

This includes:
- number of final unique galaxies
- number of successfully exported per-galaxy files
- source-catalog counts
- duplicate-status counts
- path references used during construction

### `missing_source_files.csv`
List of galaxies that could not be exported
because either:
- the normalized SPARC file was missing, or
- the external structural match row could not be recovered

### `per_galaxy/`
Per-galaxy enriched CSV files.

Each file contains:
- the original SPARC normalized rotation-curve table
- appended structural metadata from the selected external catalog
- duplicate / source-tracking metadata

These files are the **main working input files**
for the next-stage structural analysis.

---

## Construction Logic

The dataset was built under the following principle:

> **Use SPARC as the base observational dynamical table,**
> and append only those quantities that are not originally contained in SPARC.

This means:

- SPARC columns are preserved as the primary dynamical observables
- external catalog columns are treated as **added structural descriptors**
- duplicate matches between Bizyaev (2014) and Bizyaev (2002)
  are resolved at the catalog level before export

Where duplicate matches existed,
the current pipeline preferentially retained the
**Bizyaev (2014)** entry as the primary structural source.

---

## Important Interpretation Note

This dataset should **not** be interpreted as a statistically complete
edge-on galaxy survey.

Rather, it should be interpreted as a **crossmatched structure-linked test sample**
constructed for targeted consistency-oriented analysis.

Therefore:

- this folder is suitable for
  - structural residual analysis
  - thickness–rotation comparisons
  - exploratory model checks
- but it should **not** be over-interpreted as a complete population sample

---

## Intended Use in This Project

This dataset is intended to support tests such as:

- comparison between structural thickness indicators and
  rotation-curve residuals
- comparison between structural quantities and
  derived coupling / deviation parameters
- exploratory checks of whether structural morphology
  is systematically related to dynamical discrepancies

Its role is therefore **methodological and interpretive**,
not final proof by itself.

---

## Author

Kwon Dominicus  
Independent Researcher
