# Input

## Purpose

This folder stores the final executable paper-facing input files for the
halo stellar-kinematic validation domain used in Section 9.4 of the fourth paper.

These files are the processed shell-wise inputs directly consumed by the current
paper-facing halo-validation pipelines.

They preserve the stable execution interface between processed halo data
and the corresponding standard and topological runs.

---

## Role in the workflow

The intended logic is:

**raw -> derived -> input -> script -> results**

Within that logic, this folder is the point at which processed halo data become
stable executable inputs for paper-facing shell-wise execution.

Accordingly, the files here should be treated as the official input layer
for the current halo-validation workflow.

---

## Typical contents

Files in this folder may include:

- shell-assigned input tables
- processed 5D input tables
- processed 6D input tables
- distance and velocity support inputs
- structural-response support tables required for current execution
- final executable csv files used by current scripts

These files are not raw-source materials.
They are the processed execution interface used for reproducible runs.

---

## Paper-facing status

This folder is directly relevant to manuscript reproduction.

If a shell-wise figure, summary table, or paper-facing comparison depends on
current processed halo inputs, those files should appear here or be traceable to here.

This is therefore a key manuscript-facing input location for the halo domain.

---

## Interpretation note

The halo validation should be interpreted conservatively.

In particular:

- 5D and 6D cases do not have identical observational coverage,
- some shells are sparse,
- some processed fields represent structural-response quantities rather than direct observables.

The purpose of this folder is to preserve that execution pathway clearly,
not to overstate the status of every derived field.

---

## Editing rule

Do not silently replace important paper-facing input files once they support a manuscript run.

If a major change is needed:

- preserve the previous version where possible,
- make the change explicit,
- and ensure that later readers can still determine which input supported which run.

This is especially important for shell-wise workflows,
where sparse coverage can make later reconstruction difficult.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the corresponding script README
2. the dated halo output folders in `results/`
3. any summary files tied to manuscript figures or shell-wise comparisons