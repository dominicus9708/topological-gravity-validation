# Data

## Purpose

This folder stores the data layer of the validation archive.

Its role is to preserve the distinction between:

- original observational or literature-linked source material,
- processed and executable inputs,
- intermediate bridge tables used for reproducible validation workflows.

In this repository, reproducibility depends not only on scripts,
but also on keeping the data path traceable from source to executable input.

For that reason, the `data/` folder should be understood as the
main provenance and execution-input layer of the repository.

---

## Core principle

The data workflow is organized so that raw materials and executable inputs
are not mixed together.

The intended logic is:

**raw -> derived -> input -> script -> results**

This means:

- `raw` preserves original source material and provenance anchors
- `derived` stores processed or intermediate data products
- `input` stores the final processed files actually consumed by runnable pipelines

The purpose of this separation is to make later review easier and to prevent
confusion between original materials, working intermediates, and paper-facing inputs.

---

## Main internal roles

### raw

The `raw` layer preserves the closest available form of the original source material.

Typical examples include:

- downloaded observational tables
- source images or derived image-source registries
- literature-linked reference tables
- service-level exports
- manually collected provenance records

Files in `raw` should be treated as provenance anchors.
They should not be casually overwritten or silently replaced.

### derived

The `derived` layer stores processed data created from raw materials.

Typical examples include:

- normalized tables
- cleaned profile tables
- merged or matched datasets
- proxy-support tables
- bridge-construction tables
- intermediate products used before final executable input is fixed

This layer exists because many validation targets cannot be executed directly from raw data.

### input

The `input` layer stores the final processed files directly consumed by paper-facing pipelines.

This is the most important execution interface in the data hierarchy.

If a result in the manuscript depends on a specific executable data table,
that table should appear in the relevant `input` folder or be traceable to it.

---

## Reproducibility role

The `data/` folder supports reproducibility in three different ways:

1. **source traceability**  
   It records where the information came from before analysis.

2. **execution reproducibility**  
   It preserves the processed files required to run the current validated scripts.

3. **manuscript traceability**  
   It allows later readers to understand which processed inputs supported
   particular plots, summaries, and validation interpretations.

Without this separation, it becomes too easy to confuse cited source material
with pipeline-ready input.

---

## Editing rule

The data folder is not a single uniform editing zone.

Different sublayers should be handled differently.

- `raw` should be preserved as conservatively as possible.
- `derived` may be updated when preprocessing logic changes, but changes should remain traceable.
- `input` should not be silently replaced once it has served as a paper-facing execution input.

When a major input changes, versioning or explicit notes are strongly recommended.

---

## Relation to other folders

The repository should be read in layers.

- The root `README.md` explains the archive as a whole.
- `docs/` explains workflow logic, provenance notes, and manuscript-support guidance.
- `data/` preserves provenance and executable input structure.
- `src/` or `script/` preserves the runnable pipelines.
- `results/` preserves dated outputs and summaries.

In this layered reading order, `data/` is the bridge between
documentation/provenance and executable analysis.

---

## Interpretation boundary

Data stored in this folder do not automatically imply direct physical measurement
of every quantity used in the manuscript.

Some validation targets use:

- proxy-supported quantities
- density-like summaries
- mass-volume bridge variables
- component-wise auxiliary tables
- shell-wise aggregated inputs

These should be interpreted conservatively.
The role of this folder is to preserve the data pathway clearly,
not to exaggerate the epistemic status of every derived field.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. `data/raw/README.md`
2. `data/derived/README.md`
3. the target-specific `input/README.md`

That order makes it easier to understand how each validation target moves
from provenance-preserving raw material to executable paper-facing input.