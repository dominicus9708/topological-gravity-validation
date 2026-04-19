# Standard Output

## Purpose

This folder stores the generated outputs from the standard baseline pipeline
for this validation target.

Its role is to preserve the reproducible baseline reconstruction layer
used before any topological interpretation is applied.

---

## Meaning of standard

The standard layer reconstructs or summarizes the observational baseline
as closely as possible from the processed paper-facing input.

This is the baseline reproduction layer.

Its purpose is not to prove the theory.
Its purpose is to secure a readable and traceable observational reference point.

If this layer is unclear or unstable,
the corresponding topological layer should not be treated as meaningful.

---

## Typical outputs

Files in this folder may include:

- observational profile plots
- baseline comparison plots
- shell or zone summaries
- baseline csv tables
- target-specific standard summaries
- dated run folders

Not every validation target produces identical file types,
but all files here should belong to the baseline reconstruction side.

---

## Reading rule

The standard output should be read before the topological output.

The recommended order is:

1. summary txt
2. key csv tables
3. plots
4. dated run folders if exact execution tracing is needed

This order helps establish the observational baseline clearly.

---

## Interpretation boundary

Agreement in this folder means that the repository successfully reconstructs
or summarizes the baseline structure in a reproducible way.

It does not by itself validate the topological interpretation.