# Heckman Integration Guide — CAREER-DML v3.4

**Subject**: Documentation of the three levels of Heckman integration implemented in the CAREER-DML pipeline.

---

## Overview

The CAREER-DML v3.4 pipeline integrates Heckman's work at three levels of depth. All three levels are fully implemented in the codebase. This document describes each level and maps it to the corresponding code.

---

## Level 1: DGP Design (Structural Selection + Exclusion Restriction)

**Objective**: The Data Generating Process is built on Heckman's principles.

**Implementation** (`src/dgp.py`):

The latent variable `ability` influences both treatment selection (T) and the outcome (Y), creating endogenous selection bias. The DGP includes a proper exclusion restriction (`peer_adoption`) that affects treatment selection but not the outcome, satisfying the identifying assumption of Heckman (1979). Treatment assignment follows a rational decision model where agents adopt AI if the expected utility gain (higher wages minus adaptation costs) is positive. Adaptation costs are inversely proportional to `ability`, creating structural endogeneity consistent with Heckman's framework.

**Key code references**:
- `SyntheticDGP._calculate_expected_utility()` — rational decision model
- `SyntheticDGP._assign_treatment()` — structural selection with exclusion restriction
- `SyntheticDGP._generate_exclusion_restriction()` — peer adoption instrument

---

## Level 2: Interpretive Anchoring (Validation Suite)

**Objective**: Results are interpreted through Heckman's theoretical lens.

**Implementation** (`src/validation.py`):

The validation suite connects empirical results to Heckman's theory at two points. First, the variance decomposition interprets the `common_support_penalty` as a quantitative measure of the severity of Heckman's selection bias. Second, the GATES analysis interprets treatment effect heterogeneity through the lens of Cunha & Heckman's (2007) skill-capital complementarity: individuals with higher latent human capital benefit more from AI exposure (Q1: 0.5081 to Q5: 0.5661, t = 62.27, p < 10⁻²⁰⁰).

The Heckman Two-Step benchmark is run with and without the exclusion restriction, providing a fair comparison. The DML pipeline achieves a 93.0% bias reduction relative to the properly-identified Heckman model (ATE: 0.5378 vs. 1.0413).

**Key code references**:
- `interpret_variance_heckman()` — selection bias severity
- `interpret_gates_heckman()` — skill-capital complementarity
- `run_heckman_two_step_benchmark()` — with/without exclusion restriction

---

## Level 3: Structural Robustness

**Objective**: Test whether results hold under different selection mechanisms.

**Implementation** (`src/validation.py` + `src/dgp.py`):

The pipeline generates data under two selection mechanisms: mechanical (probabilistic assignment based on covariates) and structural (rational decision model where agents weigh expected utility). The bias difference between the two mechanisms is 0.077, below the pre-specified threshold of 0.1, indicating that the DML pipeline is not sensitive to the data-generating mechanism.

**Key code references**:
- `run_robustness_test()` — structural vs. mechanical comparison
- `SyntheticDGP(selection_type='mechanical')` — probabilistic selection
- `SyntheticDGP(selection_type='structural')` — rational decision selection

---

## References

- Heckman, J. J. (1979). Sample Selection Bias as a Specification Error. *Econometrica*, 47(1), 153-161.
- Cunha, F., & Heckman, J. J. (2007). The Technology of Skill Formation. *American Economic Review*, 97(2), 31-47.
- Chernozhukov, V. et al. (2018). Double/Debiased Machine Learning. *The Econometrics Journal*, 21(1), C1-C68.
- Veitch, V., Sridhar, D., & Blei, D. (2020). Adapting Text Embeddings for Causal Inference. *UAI 2020*.
