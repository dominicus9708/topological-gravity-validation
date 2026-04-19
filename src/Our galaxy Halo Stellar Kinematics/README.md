# Our Galaxy Halo Stellar Kinematics Pipelines

## Purpose

This folder stores the executable pipelines for the halo stellar-kinematic validation domain
used in Section 9.4 of the fourth paper.

Its role is to transform processed shell-wise halo inputs into reproducible outputs for:

- the standard shell-wise observational comparison layer, and
- the topological shell-wise structural-response layer.

This folder should be treated as the execution layer for the current halo-validation workflow.

---

## Role in the workflow

The intended workflow is:

**raw -> derived -> input -> src -> results**

Within that workflow, this folder is responsible for running the current
paper-facing halo-validation pipelines.

The upstream dependency is the processed shell-wise input layer.
The downstream destination is the corresponding dated results folders.

---

## Standard role

The `standard` scripts reconstruct the shell-wise observational baseline
from the processed halo inputs.

Typical products may include:

- shell-wise kinematic summaries
- interval comparisons
- variance or spread summaries
- baseline plots and csv outputs

The standard layer secures the observational comparison frame for the halo domain.

---

## Topological role

The `topological` scripts apply the fourth-paper structural interpretation
to the same fixed shell-wise input structure.

Typical products may include:

- local structural-dimension summaries
- shell-wise contrast and spread quantities
- effective response plots
- topological csv summaries

These outputs remain dependent on the same fixed input logic as the standard layer,
even though the interpreted quantities differ.

---

## 5D and 6D logic

This domain may contain separate or partially separate logic for:

- 5D cases
- 6D cases

These cases do not necessarily have identical observational coverage or stability.
That difference should remain visible in the execution logic and in later interpretation.

---

## Auxiliary scripts

Some scripts may exist for:

- shell construction
- distance or velocity restructuring
- support-table generation
- summary assembly
- comparison output production

These belong to the reproducibility chain,
but not all of them are equally manuscript-facing.

---

## Current versus auxiliary scripts

Readers should distinguish between:

- current manuscript-facing execution scripts
- helper or restructuring scripts
- earlier or replaced variants

When in doubt, priority should be given to scripts that are directly linked to:

- current paper-facing input files
- dated results folders
- manuscript-facing halo comparison plots and summaries

---

## Interpretation boundary

The halo validation should be interpreted conservatively.

In particular:

- sparse shells remain sparse,
- 5D and 6D coverage differ,
- some output quantities are structural-response quantities rather than direct observables.

Accordingly, this execution layer should be read as part of a reproducible
consistency-oriented workflow, not as a strong quantitative closure by itself.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the corresponding input README
2. the dated output folders in `results/`
3. the summary files associated with manuscript-facing comparisons