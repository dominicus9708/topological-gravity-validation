# Output

## Purpose

This folder stores the organized generated outputs for this validation target.

Its role is to separate the output layer by interpretation type
and to preserve dated execution products in a readable structure.

In most cases, the main distinction inside this folder is between:

- `standard`
- `topological`

This structure reflects the validation logic of the paper.

---

## Why this separation exists

The fourth paper does not treat standard and topological results as interchangeable.

The standard layer secures the observational or baseline reconstruction.
The topological layer adds the structural interpretation on top of that baseline.

Because those two layers have different epistemic roles,
their outputs should remain separated in the repository as well.

---

## Typical contents

This folder may contain:

- `standard/`
- `topological/`
- dated run folders
- comparison summaries
- integrated figure-support outputs

If additional support layers exist for a specific target,
they should remain clearly labeled.

---

## Reading rule

The recommended order is:

1. read the standard output first
2. then read the topological output
3. then compare dated run folders and summaries

This order matches the intended interpretation flow of the manuscript.