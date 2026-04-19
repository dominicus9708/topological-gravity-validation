# Derived Data

## Purpose

This folder stores the derived-data layer of the validation archive.

Its role is to preserve processed data products created from raw materials
before or during the construction of executable paper-facing inputs.

In this repository, many validation targets cannot be executed directly from
raw observational sources.
They require cleaning, restructuring, normalization, crossmatching,
proxy-support construction, bridge tables, or intermediate summaries.

The `derived/` folder exists to preserve that intermediate logic clearly.

---

## What belongs here

Typical contents of `derived/` may include:

- cleaned tables
- normalized profiles
- merged datasets
- crossmatched tables
- bridge-support files
- intermediate proxy tables
- target-specific processed summaries
- manually verified derived tables
- folders named `input/` that store final executable inputs

This means `derived/` is not just temporary workspace.
It is an important reproducibility layer between source preservation and execution.

---

## Input as a special sublayer

Within many targets, `input/` appears inside `derived/`.

This is intentional.

The `input/` folder should be understood as the **paper-facing executable sublayer**
of derived data.

That is where the final processed files directly consumed by scripts should live.

So the practical logic becomes:

**raw -> derived -> derived/input -> script -> results**

This distinction matters because not every derived file is equally important
for final manuscript reproduction.

Some derived files are intermediate working products.
The `input/` files are the ones most directly tied to current execution.

---

## Reproducibility role

The `derived/` folder supports reproducibility in two important ways:

1. **processing traceability**  
   It shows how raw materials were converted into usable structured data.

2. **execution readiness**  
   It preserves the intermediate and final processed tables required for
   current runnable validation pipelines.

Without this layer, later readers would see only raw files and final outputs,
while the practical transformation logic would be hidden.

---

## Editing rule

Files in `derived/` may be updated more often than files in `raw/`,
but changes should still remain traceable.

Recommended practice:

- do not silently replace important processed inputs
- version or rename major revisions when practical
- document new columns or processing assumptions
- distinguish intermediate files from current executable inputs
- keep target-specific `input/` folders understandable to later readers

The goal is not to freeze all work.
The goal is to prevent confusion about which processed files actually supported the paper.

---

## Relation to scripts

The `derived/` layer is the main interface between data preparation and execution.

In most cases, scripts should not depend directly on scattered raw materials
once a stable executable input has been constructed.

Instead, scripts should read from the relevant `input/` folder
or from clearly documented derived tables.

This makes reruns more stable and manuscript support easier to audit.

---

## Relation to results

The relationship is:

- `derived/` stores prepared data for execution
- `results/` stores generated outputs after execution

Readers should therefore not confuse processed input tables with result tables.

Even when a processed table looks already structured,
it still belongs to the pre-execution side if it serves as a script input.

---

## Interpretation boundary

Files in `derived/` may contain:

- proxy-supported quantities
- normalized structural fields
- crossmatched identifiers
- bridge quantities
- shell-wise or zone-wise reorganizations

These should be interpreted according to the validation philosophy of the paper:
conservatively and in a consistency-oriented way.

The existence of such fields does not automatically make them
uniform direct measurements.
They remain part of a documented structural-response workflow.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the target-specific `derived/<target>/README.md`
2. the target-specific `derived/<target>/input/README.md`
3. the corresponding script README

That order makes it easier to identify which derived files were exploratory
and which ones served as the current executable paper-facing inputs.