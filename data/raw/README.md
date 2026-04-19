# Raw Data

## Purpose

This folder stores the raw-source layer of the validation archive.

Its role is to preserve the closest available form of the original source material
used for the validation workflows.

In this repository, `raw` does not mean that every file is perfectly untouched in a
strict archival sense.
It means that the files here are kept as close as practical to the original
observational, downloaded, cited, or manually collected source materials
before paper-facing executable preprocessing.

Accordingly, this folder functions as the main **provenance anchor**
for the repository.

---

## What belongs here

Typical contents of `raw/` may include:

- downloaded observational tables
- original csv or txt source files
- image-source materials
- service-level exports
- literature-linked source registries
- manually collected source notes tied directly to cited materials
- raw bridge-support materials before normalization or executable restructuring

These files are kept here so that later readers can understand
where the processed executable inputs came from.

---

## What does not belong here

This folder is not the place for:

- final executable paper-facing input tables
- normalized analysis-ready tables
- cleaned merged inputs
- postprocessed bridge tables intended for direct execution
- figure-ready summary outputs

Those should instead appear in `derived/`, `input/`, or `results/`
depending on their role.

---

## Reproducibility role

This folder supports reproducibility by preserving
**source traceability**.

It allows a later reader to answer questions such as:

- What was the original source of this processed input?
- Was this value downloaded, manually recorded, or derived later?
- Which files represent provenance anchors rather than execution-ready inputs?
- Which observational materials existed before script-based preprocessing?

Without this layer, it becomes too easy to confuse
original source material with pipeline-ready input.

---

## Editing rule

Files in `raw/` should be handled conservatively.

Recommended practice:

- do not silently overwrite important raw files
- if a corrected or re-downloaded version is needed, keep the change explicit
- preserve source names where practical
- if renaming is necessary, record the rule or source identity clearly
- prefer adding provenance notes instead of modifying raw content in place

The purpose of this caution is not rigidity for its own sake.
It is to preserve confidence in the source-to-input chain.

---

## Relation to derived and input

The intended data flow is:

**raw -> derived -> input**

That means:

- `raw` preserves source material
- `derived` preserves processed intermediate data
- `input` preserves final executable paper-facing data

Readers should therefore not expect `raw/` to be directly runnable
in every validation target.

Many targets require normalization, cleaning, matching,
or bridge construction before executable input becomes possible.

---

## Interpretation boundary

The existence of a file in `raw/` does not imply that it directly provides
all physical quantities used later in the manuscript.

Some later layers introduce:

- normalized structural summaries
- proxy-supported tables
- bridge variables
- shell-wise or component-wise reorganized forms

Those belong to later stages.
The role of `raw/` is to preserve the origin point of the workflow,
not to contain the full interpreted data structure by itself.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. `data/derived/README.md`
2. the relevant target-specific raw README
3. the corresponding target-specific input README

This makes it easier to follow how source material becomes executable input.