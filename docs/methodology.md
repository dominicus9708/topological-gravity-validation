# Methodology

## 1. Purpose of This Document

This document defines the methodological principles used in the validation framework.

The goal is to ensure that all numerical results are:

- reproducible
- structurally interpretable
- independent of arbitrary parameter tuning

This repository is not intended to present finalized physical claims,
but to evaluate structural consistency between the model and observational data.

---

## 2. Data Handling Policy

### 2.1 Raw Data

All observational datasets are stored in:

    data_raw/

These datasets must not be modified.

They represent:

- galaxy rotation data (SPARC-based)
- gravitational wave derived datasets
- cosmic microwave background related datasets
- light echo observational data

All preprocessing must be performed through code.

---

### 2.2 Processed Data

Processed data are stored in:

    data_processed/

These files are generated from raw data and must be reproducible.

No manual editing of processed data is allowed.

---

## 3. Structural Quantities

### 3.1 Local Scaling Exponent

The model assumes the existence of local structural scaling contributions:

    α(x)

This quantity represents the structural role of each region.

---

### 3.2 Weighted Structural Aggregation

The global structural quantity is defined through weighted aggregation:

    D_w = ∫ α(x) w(x) dμ

This formulation follows the mathematical structure established in the second paper.

It is important to note:

- This is not a probabilistic expectation value
- It is a structural aggregation functional

---

## 4. Sigma (σ) Modeling

### 4.1 Definition

The sigma term represents structural deviation from a reference configuration.

It encodes how local structural contributions differ across the system.

---

### 4.2 Computation Method

Sigma is computed using integration-based approaches.

Two possible methods exist:

    Method A: discrete segmentation
    Method B: weighted integration

The current implementation prioritizes:

    weighted integration

Reason:

- avoids artificial discretization effects
- preserves structural continuity

---

## 5. Beta (β) Treatment

### 5.1 Role of Beta

Beta represents an effective coupling between:

- baryonic gravitational contribution
- structural deviation (sigma)

---

### 5.2 Case A: Constant Beta

Initial validation uses:

    β = constant

Purpose:

- establish baseline behavior
- avoid overfitting
- verify structural consistency

---

### 5.3 Case B: Structural Beta

In advanced stages, beta is derived from structural properties.

This introduces:

- system-dependent coupling
- structural interpretation of parameter variation

Important constraint:

Beta must not be tuned to match observations directly.

---

## 6. Computational Pipeline

The full validation pipeline is:

    raw data
    → preprocessing
    → sigma computation
    → beta application
    → acceleration calculation
    → rotation curve reconstruction
    → comparison with observation

Each step must be traceable through code.

---

## 7. Interpretation Constraints

The results of this framework must be interpreted carefully.

In particular:

- Agreement with observational data does not imply
  replacement of existing physical theories

- Disagreement does not invalidate the framework
  without structural analysis

- Observational consistency must be distinguished from causal explanation

---

## 8. Experimental Principles

The following principles are strictly enforced:

- No manual tuning of outputs
- No hidden parameter adjustments
- No data manipulation outside code
- All results must be reproducible

---

## 9. Known Limitations

Current limitations include:

- incomplete modeling of environmental effects
- simplified structural coupling assumptions
- partial dataset coverage

These limitations must be considered when interpreting results.

---

## 10. Research Position

This repository represents a validation framework within a broader research program.

It is not a complete physical theory.

Its role is to test whether the structural framework:

    can reproduce observed phenomena
    under transparent and reproducible conditions
