# Input

## Purpose

This folder stores the final executable paper-facing input files for the
WISE H II region validation domain used in Section 9.1 of the fourth paper.

These files are the processed inputs directly consumed by the current
paper-facing pipelines.

They are not raw-source materials, and they are not merely exploratory
intermediate tables.
They are the practical execution interface between processed preparation
and current reproducible script execution.

---

## Role in the workflow

The intended logic is:

**raw -> derived -> input -> script -> results**

Within that logic, this folder is the point at which processed data become
stable executable inputs.

Accordingly, the files here should be treated as the current official input layer
for the paper-facing WISE H II validation workflow.

---

## Typical contents

Files in this folder may include:

- final target tables
- processed radial profile inputs
- normalized executable input csv files
- bridge-support inputs required by current standard or topological scripts
- target-level consolidated input files

Not every file here has to be minimal.
However, every file here should have a clear execution role.

---

## Paper-facing status

This folder is directly relevant to manuscript reproduction.

If a plot, csv summary, or validation statement in the manuscript depends on a
current processed data table, that table should be located here or traceable to here.

This folder therefore has stronger paper-facing importance than a general derived working folder.

---

## Interpretation note

The WISE H II validation should be interpreted conservatively.

Some quantities may include bridge-style support or proxy-supported fields.
These do not imply direct closed mass determination.

Accordingly, files in this folder preserve a reproducible execution layer for
**structural comparison** and **consistency-oriented response**,
not a claim of final observational closure.

---

## Editing rule

Do not silently replace important input files once they support a paper-facing run.

If a major change is required:

- prefer explicit versioning,
- preserve the earlier file when practical,
- or record the change clearly in notes or commit history.

The purpose of this caution is to preserve manuscript traceability.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the corresponding script README
2. the dated output folders in `results/`
3. any run summary txt or csv produced from those inputs