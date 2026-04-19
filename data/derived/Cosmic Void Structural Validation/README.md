# Cosmic Void Structural Validation

## Purpose

This folder stores the derived-data layer for the cosmic-void validation domain
used in Section 9.3 of the fourth paper.

Its purpose is to preserve the processed datasets, comparison tables,
and executable inputs needed to compare:

- a standard proxy-density reading of void structure, and
- a topological effective-dimension response.

---

## Role in the paper

This folder supports the validation domain described as
**Cosmic Void Structural Validation**.

The paper-facing role of this target is to test whether large-scale underdensity structure
can be translated into a reproducible structural comparison without forcing
the standard and topological quantities to be identical observables.

Accordingly, the validation should be read as a consistency-oriented structural comparison.

---

## Internal logic

The derived files in this domain may include:

- processed void catalog tables
- background support tables
- proxy-density summaries
- effective-dimension support tables
- integrated comparison inputs
- final executable paper-facing tables stored in `input/`

The standard and topological layers are not identical quantities.
The purpose of the derived layer is to preserve the processed comparison frame
used to place them into a common structural interpretation.

---

## Reproducibility role

This folder preserves the intermediate and executable processed data products
needed to move from source catalogs to paper-facing comparison inputs.

Without this layer, later readers could see the raw catalog or the final plot,
but not the practical steps that prepared the comparable structural-response inputs.

---

## Important interpretation note

This validation target may depend on operational choices such as:

- sample construction
- proxy-density support
- background selection
- processed comparison framing

These choices should be interpreted conservatively.

The existence of processed derived variables here does not imply
that the target has been reduced to a single direct physical observable.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. `input/README.md`
2. the corresponding script README
3. the paper-facing dated output folders in `results/`