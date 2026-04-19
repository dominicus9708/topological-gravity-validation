# Topological Output

## Purpose

This folder stores the generated outputs from the topological interpretation pipeline
for this validation target.

Its role is to preserve the reproducible structural-response layer generated
on top of the observational baseline.

---

## Meaning of topological

The topological layer does not replace the observational baseline.

Instead, it reuses the same fixed processed input structure and adds
the fourth-paper structural interpretation.

Depending on the target, this may include:

- structural contrast variables
- effective structural dimension
- normalized structural-response quantities
- bridge-style comparison variables
- topological summary plots
- comparison csv tables

---

## Relation to the standard layer

The topological output should always be read in relation to the standard output.

Its meaning depends on the baseline secured by the standard layer.

For that reason, readers should not begin with this folder in isolation.
The intended order is always:

1. standard output
2. topological output
3. dated run folder comparison
4. manuscript-facing summary interpretation

---

## Typical outputs

Files in this folder may include:

- topological summary csv files
- processed structural-response tables
- topological comparison plots
- bridge-aware plots
- target-specific topological summaries
- dated run folders

Not every validation target uses identical topological outputs,
but all files here should belong to the paper-facing structural interpretation layer.

---

## Interpretation boundary

These outputs should be interpreted conservatively.

They are part of a consistency-oriented validation workflow.
They do not by themselves establish final observational closure.

Where proxy-supported or bridge-style quantities appear,
those quantities should be read with the same caution stated in the manuscript.