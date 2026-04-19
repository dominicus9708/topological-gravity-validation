# Echo of Light Pipelines

## Purpose

This folder stores the executable pipelines for the light-echo validation domain
used in Section 9.2 of the fourth paper.

Its role is to transform processed light-echo inputs into reproducible outputs for
target-specific standard and topological comparisons.

Because the light-echo domain includes heterogeneous targets,
the execution logic is target-dependent rather than universally uniform.

---

## Role in the workflow

The intended workflow is:

**raw -> derived -> input -> src -> results**

Within that workflow, this folder is responsible for running the current
paper-facing light-echo pipelines.

The upstream dependency is the target-specific processed input layer.
The downstream destination is the corresponding dated results folders.

---

## Target-dependent execution logic

The light-echo validation is not a single-formula pipeline family.

Different targets may require different executable logic, such as:

- temporal radius and center-offset evolution
- patch-family geometry processing
- component-wise structural decomposition
- target-specific normalization and comparison

This difference is intentional and reflects observational heterogeneity,
not a mistake in repository organization.

---

## Standard role

The `standard` scripts reconstruct the observational morphology
as closely as possible from the processed target inputs.

Typical standard products may include:

- temporal evolution tables
- center-offset plots
- patch-family layout summaries
- component-wise observational context
- baseline comparison plots

Their role is to secure the observational baseline for each target.

---

## Topological role

The `topological` scripts apply the fourth-paper structural interpretation
on top of the corresponding standard observational structure.

Depending on the target, this may include:

- structural-response quantities
- normalized contrast variables
- trial effective-dimension summaries
- patch-family comparison logic
- component-wise structural hierarchy outputs

These outputs should be interpreted conservatively and target by target.

---

## Auxiliary scripts

Some scripts may exist for:

- geometry restructuring
- target-specific formatting
- bridge support
- intermediate comparison generation
- summary production for manuscript use

These are part of the reproducibility chain,
but not every such file is necessarily the main paper-facing execution script.

---

## Current versus auxiliary scripts

Readers should distinguish between:

- current manuscript-facing execution scripts
- helper or preprocessing scripts
- older or replaced target variants

When in doubt, priority should be given to scripts that are directly connected to:

- final target input files
- dated result folders
- manuscript-facing figures and summaries

---

## Interpretation boundary

The light-echo pipelines preserve a constrained consistency-oriented validation workflow.

They should not be interpreted as closed proof pipelines.

In particular:

- target heterogeneity remains real,
- target-specific normalization remains target-specific,
- reproducible structural response is not identical to final empirical closure.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the corresponding target input README
2. the dated results folders for the relevant target
3. any run summaries used for manuscript figure assembly