# Input

## Purpose

This folder stores the final executable paper-facing input files for the
light-echo validation domain used in Section 9.2 of the fourth paper.

Because the light-echo domain contains heterogeneous targets,
this folder preserves target-dependent executable inputs rather than one
single universal input format.

These files are the processed inputs directly consumed by the current
paper-facing light-echo pipelines.

---

## Role in the workflow

The intended logic is:

**raw -> derived -> input -> script -> results**

This folder is the point at which target-specific processed data become
stable executable inputs for the current paper-facing workflows.

Accordingly, the files here should be treated as the official input layer
for the current reproducible light-echo validation workflows.

---

## Typical contents

Files in this folder may include target-specific inputs such as:

- temporal radius tables
- center-offset inputs
- patch-family geometry inputs
- component-wise decomposition inputs
- processed normalized structural-response inputs
- final executable csv tables for V838 Mon, T Pyxidis, or Eta Carinae

Because the observational forms differ by target,
the exact structure of these files may differ as well.

That difference is intentional and should remain visible.

---

## Paper-facing status

This folder is directly relevant to manuscript reproduction.

The light-echo validation is not built from one single universal processed structure.
Instead, the manuscript-facing runs depend on target-specific executable inputs.

These files therefore preserve the paper-facing execution layer for the
light-echo validation domain.

---

## Interpretation note

The light-echo validation is a constrained consistency-oriented examination.

The small number of targets reflects rarity and limited observational accessibility,
not arbitrary cherry-picking.

Some processed fields in this folder may be normalized or proxy-supported.
They should be interpreted as part of a reproducible structural-response workflow,
not as final direct measurement of a completed covariant field.

---

## Editing rule

Do not silently replace important target inputs once they support a paper-facing run.

If a major revision is necessary:

- preserve the earlier version when possible,
- use explicit versioned names,
- or record the change clearly in notes and commit history.

This is especially important because target heterogeneity makes later confusion more likely.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the corresponding light-echo script README
2. the dated output folders for each target in `results/`
3. any run-summary files connected to manuscript figures