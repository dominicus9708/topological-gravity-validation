# Echo of Light

## Purpose

This folder stores the derived-data layer for the light-echo validation domain
used in Section 9.2 of the fourth paper.

Its purpose is to preserve the processed target-specific data needed to execute
reproducible comparisons for light-echo systems.

This validation domain does not impose one universal formula across all targets.
Instead, it preserves target-dependent processed inputs for several distinct
observational forms.

---

## Role in the paper

This folder supports the validation domain described as
**Validation Through Light-Echo Morphology**.

The targets in this domain are treated as a constrained consistency-oriented set.

Their role is to test whether topological structural response can be translated into
reproducible target-specific workflows for:

- temporal relaxation,
- anisotropic patch-family structure,
- component-wise structural decomposition.

This is not an exhaustive statistical survey and not a final proof.

---

## Internal logic

The derived layer for light echoes may include target-specific processed forms such as:

- radius and temporal evolution tables
- center-offset tables
- patch-family geometry tables
- component decomposition tables
- normalized bridge variables
- final executable target inputs stored in `input/`

The targets are not forced into a single identical processed representation,
because the observational forms themselves differ.

That difference is intentional and should remain visible in the derived structure.

---

## Reproducibility role

This folder preserves the intermediate and executable preparation logic needed to move from:

- source-linked observational material
to
- paper-facing executable target inputs.

Because the light-echo analysis depends on rare and heterogeneous observational cases,
the derived layer is especially important for preserving how each target was converted
into a reproducible structural-response workflow.

---

## Important interpretation note

The light-echo domain should be interpreted conservatively.

The small number of targets reflects rarity and limited observational accessibility,
not arbitrary cherry-picking.

Derived quantities in this folder may include proxy-supported or normalized structural variables.
They should be read as part of a consistency-oriented comparison framework,
not as final direct measurement of a completed covariant field.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. `input/README.md`
2. the corresponding script README
3. the dated output folders for each target in `results/`