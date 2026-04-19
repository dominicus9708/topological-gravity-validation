# Input

## Purpose

This folder stores the final executable paper-facing input files for the
cosmic-void validation domain used in Section 9.3 of the fourth paper.

These files are the processed inputs directly consumed by the current
paper-facing cosmic-void pipelines.

They preserve the executable comparison frame between
standard proxy-density structure and topological effective-dimension response.

---

## Role in the workflow

The intended logic is:

**raw -> derived -> input -> script -> results**

Within that logic, this folder is the stable execution interface used by the
current manuscript-facing cosmic-void workflows.

Accordingly, the files here should be treated as the official input layer
for reproducible cosmic-void validation runs.

---

## Typical contents

Files in this folder may include:

- processed void sample tables
- background support tables
- proxy-density executable inputs
- topological-response support tables
- integrated comparison inputs
- final executable csv files used by current scripts

These are not raw catalogs and not merely exploratory intermediate products.
They are the practical execution interface for the paper-facing runs.

---

## Paper-facing status

This folder is directly relevant to manuscript reproduction.

If a figure, table, or summary in the manuscript depends on current processed
cosmic-void comparison inputs, those files should appear here or be traceable to here.

This makes the folder one of the most important data locations for this validation domain.

---

## Interpretation note

The standard and topological quantities used in this domain are not identical observables.

They are placed into a common structural comparison frame.

Therefore, some files in this folder preserve operational comparison quantities
rather than one single direct physical observable.

They should be interpreted conservatively and in accordance with the
consistency-oriented validation philosophy of the paper.

---

## Editing rule

Do not silently replace important executable inputs once they support a paper-facing run.

If a major change is required:

- preserve earlier versions when practical,
- document structural changes clearly,
- and keep later readers able to understand which input supported which manuscript result.

---

## Recommended next step

After reading this file, the recommended next step is to open:

1. the corresponding script README
2. the dated output folders in `results/`
3. any run summary files associated with manuscript figures or integrated comparison outputs