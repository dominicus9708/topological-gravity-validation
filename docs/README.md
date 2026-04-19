# Documentation

## Purpose

This folder stores the supporting documents needed to understand,
reproduce, and audit the validation workflow used for the fourth paper:

**Topological Gravity: Structural Derivation of an Effective Gravitational Field**

This folder is not the main location for raw observational data,
processed executable inputs, or final generated outputs.
Its role is different.

The purpose of `docs/` is to preserve the written guidance that explains
how the executable archive maps onto the manuscript.

---

## What this folder is for

The files in this folder are intended for readers who need to understand:

- how the validation workflow was organized
- what role each validation target played in the paper
- how provenance and source tracking were handled
- how execution notes and handoff summaries were recorded
- how manuscript-facing interpretation was connected to stored outputs

In other words, this folder explains the **logic around the workflow**,
not just the workflow outputs themselves.

---

## Typical contents

This folder may contain documents such as:

- validation policy summaries
- reproducibility notes
- source provenance notes
- manuscript insertion notes
- execution command guides
- chat handoff summaries
- figure assembly notes
- repository documentation plans
- folder-specific README drafting notes

Not every file in this folder is a paper-facing artifact.
Some files exist to preserve practical working context and decision history.

---

## Relation to other folders

The repository should be read in layers.

- The root `README.md` explains the repository as a whole.
- Target folders explain each validation domain.
- Raw and derived folders preserve provenance and executable inputs.
- Script folders preserve the runnable pipelines.
- Result folders preserve dated outputs and summaries.
- The `docs/` folder explains how those pieces connect to the manuscript and to one another.

Accordingly, `docs/` should be treated as a **supporting interpretation and provenance layer**.

---

## Reproducibility role

This folder contributes to reproducibility in an indirect but important way.

It does not usually contain the numerical executable input itself.
Instead, it preserves the reasoning and documentation needed to answer questions such as:

- Why was a certain folder kept?
- Which files are legacy and which are current?
- Which pipeline state was used for the manuscript?
- How were the outputs interpreted conservatively?
- How should a later reader navigate the archive without confusing trial material with paper-facing material?

For that reason, this folder helps preserve
**workflow reproducibility**, **provenance clarity**, and **manuscript traceability**.

---

## Important distinction

Files in `docs/` should not be confused with raw data or with final numerical outputs.

If a reader wants to reproduce an actual result,
they should use `docs/` together with:

1. the target-level README
2. the relevant input folder
3. the relevant script folder
4. the corresponding dated output folder

Thus, `docs/` is an explanatory layer, not a substitute for executable provenance.

---

## Interpretation boundary

This repository follows a consistency-oriented validation philosophy.
Accordingly, many documents in this folder are written to preserve
conservative interpretation boundaries, including:

- proxy-supported rather than uniformly direct-measured quantities
- limited sample size
- target-dependent observational accessibility
- differences between standard baselines and topological interpretation
- the distinction between reproducible structural response and final empirical proof

Readers should therefore treat this folder as part of the repository's
transparency structure, not as a collection of claims stronger than the paper itself.

---

## Recommended use

If you are reading this folder for the first time, the recommended order is:

1. root `README.md`
2. this `docs/README.md`
3. target-level README for the validation domain you need
4. any provenance or run-command note relevant to that target
5. the corresponding input, script, and output folders

This order reduces confusion between manuscript support material,
legacy notes, and executable artifacts.

---

## Notes

Some documents in this folder may describe planning,
organization, or repository maintenance rather than physics itself.

They are kept because reproducibility is not only a matter of code and data,
but also of preserving the practical logic that explains
how the archive should be read.