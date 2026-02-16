# Changelog

All notable changes to CAREER-DML are documented in this file. This project follows [Semantic Versioning](https://semver.org/).

## [4.0] - 2026-02-16

This is a major release reflecting the culmination of the research journey, including robustness corrections and the implementation of a semi-synthetic DGP.

### Added
- **Semi-Synthetic DGP:** A new data generator calibrated with real-world parameters from the NLSY79 and AI exposure scores from Felten et al. (2021).
- **Corrected Pipeline:** A new main script (`main_board_corrected.py`) that implements two key corrections: equal dimensionality across all embedding variants (`phi_dim=64`) and a realistic treatment effect (`TRUE_ATE=0.08`).
- **Power Analysis Module:** `src/power_analysis.py` computes the Minimum Detectable Effect (MDE) for any given sample size, formalising the Signal-to-Noise Frontier.
- **Overlap Diagnostic:** A propensity score histogram is now generated and saved to visually inspect common support.
- **Utility Module:** `src/utils.py` extracts shared functions to eliminate code duplication.
- **Technical Improvements:**
  - Numerical stability clamp (`torch.clamp`) added to the VIB `logvar` to prevent `log(0)` errors.
  - Epsilon guard (`1e-10`) added to Welch's t-test in GATES analysis to prevent division by zero.

### Changed
- **README.md:** Rewritten to reflect the full project evolution and key findings.
- **Research Proposal and Motivation Letter:** Updated to v7.0 with the final project narrative.
- **Paper Drafts:** Updated to v4.0 with new Abstract, Introduction, and Results sections.
- **Tests:** All unit and integration tests rewritten for the current module interfaces (50 tests, all passing).

### Key Findings
- **Embedding Paradox Confirmed:** With equal dimensions (`phi_dim=64`), the VIB variant still exhibits the highest bias (160.3% error), confirming the result is not a dimensional artifact.
- **Signal-to-Noise Frontier Characterised:** With a realistic ATE of 0.08, all models fail (errors > 100%), demonstrating the limits of causal inference with small samples.
- **Consistent Superiority over Heckman:** The DML framework demonstrates 88-95% bias reduction over the classical Heckman model across all three pipeline configurations.
- **Robustness:** The semi-synthetic pipeline achieved an Oster delta of 75.95.

## [3.3] - 2026-02-13

### Added
- Statistical inference for all ATE estimates (SE, 95% CI, p-values).
- Formal GATES heterogeneity test (Welch's t-test).
- VIB sensitivity analysis (beta sweep).
- Exclusion restriction in the DGP for a properly-identified Heckman benchmark.

### Changed
- Rewrote README with a more academic tone and detailed results.

## [3.2] - 2026-02-08

### Added
- Debiased GRU with adversarial training.
- GATES and Oster delta validation.

## [3.1] - 2026-01-15

### Added
- Causal GRU with VIB regularisation.

## [3.0] - 2025-12-20

### Added
- Initial implementation of the CAREER-DML pipeline with a Predictive GRU and a basic Heckman comparator.
