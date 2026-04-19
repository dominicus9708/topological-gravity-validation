# Results

## Purpose

This folder stores the generated output layer of the validation archive.

Its role is to preserve the dated results produced by the executable pipelines,
including plots, csv summaries, processed comparison tables, and supporting run records.

In the repository workflow, `results/` is the final execution-output layer following:

**raw -> derived -> input -> src -> results**

This folder therefore preserves the paper-facing outputs generated from the current
validation pipelines used for the fourth paper.

---

## What belongs here

Typical contents of `results/` may include:

- dated run folders
- plots
- csv summary tables
- processed comparison outputs
- integrated comparison files
- run summary txt files
- manuscript-facing figure-support outputs

This folder is not the place for raw-source materials or primary executable inputs.
Its role begins after execution.

---

## Output policy

Outputs should be organized in a way that reduces ambiguity and prevents accidental overwrite.

The recommended logic is:

- by validation target
- by interpretation layer
- by date and time where possible

This preserves execution history and makes manuscript traceability easier.

---

## Interpretation boundary

Files in `results/` are generated outputs.
They should be interpreted according to the paper's consistency-oriented validation philosophy.

In particular:

- a promising comparison output is not by itself final empirical proof
- standard outputs and topological outputs do not play identical roles
- some outputs depend on proxy-supported or target-dependent processed inputs
- dated results preserve execution state, not automatic theoretical closure

---

## Recommended reading order

For a given validation target, the recommended order is:

1. target-level README
2. `output/README.md`
3. `output/standard/README.md`
4. `output/topological/README.md`
5. dated run folders
6. summary txt and csv files
7. plots

This order reduces confusion between baseline reconstruction and topological interpretation.

---

## Relation to the manuscript

This folder is the most direct storage layer for manuscript-facing generated material.

When a manuscript figure, table, or validation summary depends on a generated output,
that output should appear here or be traceable to a dated run folder here.

Accordingly, this folder is central for later review of what was actually produced by execution.