# Cosmic Void Structural Validation Pipelines

## Purpose

This folder stores the executable pipelines for the cosmic-void validation domain
used in Section 9.3 of the fourth paper.

Its role is to transform processed void inputs into reproducible outputs for:

- a standard proxy-density comparison layer, and
- a topological effective-dimension response layer.

This is the execution layer for paper-facing cosmic-void validation runs.

---

## Role in the workflow

The intended workflow is:

**raw -> derived -> input -> src -> results**

Within that workflow, this folder is responsible for running the current
manuscript-facing cosmic-void workflows.

The upstream dependency is the processed input layer.
The downstream destination is the corresponding dated results folders.

---

## Standard role

The `standard` scripts produce the baseline comparison layer from processed void inputs.

Typical products may include:

- proxy-density summaries
- background comparisons
- integrated standard-side comparison values
- baseline plots and csv outputs

The role of the standard layer is not to mimic the topological layer,
but to secure a reproducible comparison baseline.

---

## Topological role

The `topological` scripts produce the structural-response layer
based on the same processed comparison frame.

Typical products may include:

- effective-dimension based summaries
- structural-response classifications
- integrated comparison outputs
- topological plots and csv summaries

The standard and topological outputs are not identical observables.
They are placed into a common structural comparison frame.

---

## Auxiliary scripts

Some scripts may exist for:

- sample construction
- background support preparation
- proxy-density restructuring
- integrated comparison table generation
- summary file generation

These should be read as execution supports inside the reproducibility chain.

---

## Current versus auxiliary scripts

Readers should distinguish between:

- current paper-facing execution scripts
- helper or restructuring scripts
- exploratory or replaced variants preserved for continuity

When in doubt, priority should be given to scripts that are directly linked to:

- the current input folder
- dated results folders
- manuscript-facing integrated comparison outputs

---

## Interpretation boundary

The cosmic-void validation should be interpreted conservatively.

This folder preserves reproducible structural comparison logic,
not a reduction of the problem to one closed direct observable.

Operational choices such as background construction,
sample framing, and proxy-density support remain part of the workflow
and should be read within the consistency-oriented validation philosophy of the paper.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the corresponding input README
2. the dated standard output folders
3. the dated topological output folders