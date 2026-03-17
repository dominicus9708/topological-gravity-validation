# Galaxy Rotation Validation Pipeline

## 1. Purpose

This document describes the validation pipeline used to compare
the structural gravitational framework with observed galaxy rotation curves.

The goal is not to claim a replacement of existing models,
but to evaluate whether the structural formulation:

    can reproduce observed rotational behavior
    under controlled and reproducible conditions

---

## 2. Data Source

The primary dataset is based on galaxy rotation observations (SPARC-type data).

Raw data are stored in:

    data_raw/sparc/

These include:

- radial distance (r)
- observed rotation velocity (v_obs)
- baryonic mass components

Raw datasets must remain unchanged.

---

## 3. Preprocessing

All preprocessing is performed via:

    src/preprocessing.py

Steps include:

- unit normalization
- column standardization
- missing value handling
- galaxy-wise segmentation

The output is stored in:

    data_processed/sparc/

No manual data editing is allowed.

---

## 4. Structural Modeling Components

### 4.1 Sigma (σ)

Sigma represents structural deviation across the system.

It is computed using:

    weighted integration

instead of discrete segmentation.

Reason:

- preserves structural continuity
- avoids artificial discretization effects

---

### 4.2 Beta (β)

Beta represents an effective coupling between:

- baryonic contribution
- structural deviation

Two cases are evaluated.

#### Case A: Constant Beta

    β = constant

Used as a baseline model.

#### Case B: Structural Beta

    β = f(structural quantities)

Derived from internal structural properties.

Important constraint:

Beta must not be tuned to directly match observations.

---

## 5. Computational Pipeline

The full computation sequence is:

    raw data
    → preprocessing
    → sigma computation
    → beta assignment
    → acceleration calculation
    → rotation velocity reconstruction
    → comparison with observation
    → metric evaluation
    → result storage

Each step is implemented in a separate module under:

    src/

---

## 6. Output

The pipeline produces:

### 6.1 Individual Galaxy Results

Stored in:

    results/plots/individual/

Includes:

- observed rotation curve
- model prediction
- residuals

---

### 6.2 Summary Results

Stored in:

    results/tables/

Includes:

- RMSE
- relative error
- comparison across beta models

---

## 7. Evaluation Metrics

Quantitative evaluation is performed using:

- root mean square error (RMSE)
- mean fractional error
- residual distribution

Visual agreement alone is not considered sufficient.

---

## 8. Interpretation Constraints

The results must be interpreted under the following conditions:

- Agreement with observations does not imply
  replacement of dark matter or standard cosmology

- Disagreement does not invalidate the framework
  without structural analysis

- This pipeline evaluates consistency, not causality

---

## 9. Known Limitations

Current limitations include:

- incomplete modeling of environmental effects
- simplified structural coupling (beta)
- limited dataset coverage
- possible degradation in specific galaxies

These limitations must be considered in all interpretations.

---

## 10. Experimental Principles

The following rules are strictly enforced:

- no manual tuning of outputs
- no hidden parameter adjustment
- no modification of raw data
- full reproducibility of all results

---

## 11. Research Context

This pipeline corresponds to the validation stage
of the fourth research work.

It builds upon:

- structural axiomatic formulation
- function-theoretic aggregation
- dynamical structural interpretation

The purpose is to test whether these structures:

    can be mapped onto observational galaxy dynamics
