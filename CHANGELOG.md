# Changelog

All notable changes to CAREER-DML are documented in this file. This project follows [Semantic Versioning](https://semver.org/).

## [4.0] - 2026-02-16

This is a major release reflecting the culmination of the entire research journey, including a full 3-Layer Board Review and the implementation of its findings.

### Added
- **3-Layer Board Review:** A comprehensive analysis of the project from technical, methodological, and strategic perspectives, leading to the v4.0 corrections.
- **Semi-Synthetic DGP:** A new data generator calibrated with real-world parameters from the NLSY79 and AI exposure scores from Felten et al. (2021).
- **Board-Corrected Pipeline:** A new main script (`main_board_corrected.py`) that implements the Board's key decisions: equal dimensionality (`phi_dim=64`) and a realistic treatment effect (`TRUE_ATE=0.08`).
- **Overlap Diagnostic:** A propensity score histogram is now generated and saved (`results/figures/overlap_diagnostic.png`) to visually inspect common support.
- **Technical Improvements:**
  - Numerical stability clamp (`torch.clamp`) added to the VIB `logvar` to prevent `log(0)` errors.
  - Epsilon guard (`1e-10`) added to Welch's t-test in GATES analysis to prevent division by zero.
- **New Documentation:**
  - `BOARD_ANALYSIS.md`: The full deliberation and verdict of the 3-Layer Board.
  - `provas_empiricas.md`: A summary of the 9 independent empirical proofs that validate the framework.
  - `gap_analysis.md`: A cross-analysis of the internal and external Board reviews.

### Changed
- **README.md:** Completely rewritten to v4.0, telling the full story of the project's evolution and highlighting the key findings of the "Embedding Paradox" and the "Signal-to-Noise Frontier".
- **Research Proposal & Motivation Letter:** Updated to v7.0, reflecting the final, mature narrative of the project, focusing on the validated framework and the need for large-scale data.
- **Paper Drafts:** Updated to v4.0 with new Abstract, Introduction, and Results sections that incorporate the final project narrative.
- **Citation:** Updated to reflect the final version 4.0 and the full scope of the project.

### Key Scientific Findings
- **Embedding Paradox Confirmed:** With equal dimensions (`phi_dim=64`), the VIB variant still exhibits the highest bias (160.3% error), proving the paradox is a genuine scientific finding.
- **Signal-to-Noise Frontier Discovered:** With a realistic ATE of 0.08, all models fail (errors > 100%), demonstrating the limits of causal inference with small samples and motivating the need for large-scale administrative data.
- **Robust Superiority over Heckman:** The DML framework demonstrates a consistent 88-95% bias reduction over the classical Heckman model across all three major pipeline runs (synthetic, semi-synthetic, Board-corrected).
- **Extreme Robustness:** The semi-synthetic pipeline achieved an Oster delta of **75.95**, indicating extreme robustness to unobserved confounding.

## [3.3] - 2026-02-13

### Added
- Statistical inference for all ATE estimates (SE, 95% CI, p-values).
- Formal GATES heterogeneity test (Welch's t-test).
- VIB sensitivity analysis (β sweep).
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
