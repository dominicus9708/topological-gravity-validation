# Source Pipelines

## Purpose

This folder stores the executable pipeline layer of the validation archive.

Its role is to connect processed input data to reproducible generated outputs.

In the repository workflow, `src/` should be understood as the main execution layer between:

- `data/` as the provenance and input layer, and
- `results/` as the generated output layer.

Accordingly, this folder preserves the runnable scripts used to produce
paper-facing validation outputs for the fourth paper:

**Topological Gravity: Structural Derivation of an Effective Gravitational Field**

---

## Role in the workflow

The intended repository logic is:

**raw -> derived -> input -> src -> results**

This means:

- raw data preserve provenance,
- derived and input folders preserve processed executable tables,
- `src/` executes the current paper-facing workflows,
- results folders preserve dated outputs and summaries.

Without the `src/` layer, reproducibility would stop at stored inputs.
This folder is what turns those inputs into manuscript-facing outputs.

---

## What this folder contains

This folder may contain target-specific executable pipelines such as:

- preprocessing helpers
- acquisition or restructuring scripts
- standard baseline pipelines
- topological interpretation pipelines
- bridge-support scripts
- summary or comparison generators

Not every file in `src/` has the same status.
Some files are directly paper-facing current execution scripts.
Others may be support scripts used before final execution.

For that reason, readers should rely on local target-level README files
to identify which scripts are current and which are auxiliary.

---

## Standard and topological roles

### Standard

The `standard` layer reconstructs or summarizes the observational baseline
as closely as possible from the cited observational materials.

Its role is not to prove the theory.
Its role is to secure a reproducible observational baseline.

If the standard layer is unstable or unclear,
the topological layer should not be treated as meaningful.

### Topological

The `topological` layer applies the fourth-paper structural interpretation
on top of the observational baseline.

It does not replace the baseline.
It reuses the same observational structure and adds structural-response logic,
such as effective structural dimension, structural contrast,
or other conservative comparison quantities.

### Conditional bridge or support scripts

Some targets may include additional scripts for:

- mass-volume bridge construction
- normalization support
- target-specific restructuring
- integrated comparison summaries

These are conditional supports, not universal default assumptions.

---

## Current versus legacy logic

This repository may preserve older or replaced script states elsewhere,
especially under `legacy/`.

Therefore, files in `src/` should be read carefully.

In general:

- current paper-facing execution should be documented in target-level README files,
- older or discontinued logic should remain in legacy folders,
- readers should not assume that every script here is equally manuscript-facing
  without checking the local documentation.

---

## Reproducibility role

This folder supports reproducibility by preserving:

1. **execution reproducibility**  
   It stores the runnable logic that transforms processed input tables into outputs.

2. **workflow transparency**  
   It helps later readers understand how baseline reconstruction and topological
   interpretation were operationally separated.

3. **manuscript traceability**  
   It provides the executable path behind dated result folders, plots, and summary tables.

This is especially important in a validation archive where the goal is not only to
store data and outputs, but to preserve the route between them.

---

## Interpretation boundary

Scripts in this folder should not be interpreted as proof by themselves.

Even when a script produces a promising comparison output,
that output should still be interpreted according to the paper's
consistency-oriented validation philosophy.

In particular:

- proxy-supported quantities remain proxy-supported,
- target-dependent workflows remain target-dependent,
- reproducible structural response is not identical to final empirical closure.

This distinction matters for both manuscript interpretation and repository reading.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the target-specific README inside the relevant `src/<target>/` folder
2. the corresponding input README in `data/derived/.../input/`
3. the corresponding dated output folders in `results/`

That order makes it easier to connect execution logic to actual paper-facing runs.