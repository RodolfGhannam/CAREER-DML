# Changelog

All notable changes to CAREER-DML are documented in this file. This project follows [Semantic Versioning](https://semver.org/).

## [6.0] - 2026-02-20

This release focuses on refining the scientific narrative, strengthening the connection to economic literature, and preparing all documents for final submission.

### Changed
- **Scientific Narrative:** Re-framed the "Embedding Paradox" as the **"Sequential Embedding Ordering Phenomenon"** across all documents. This change adopts more cautious and precise scientific language, positioning the finding as a consistent empirical observation rather than a universal law, which strengthens the research proposal.
- **Structural Economics Dialogue:** Integrated an explicit bridge to classical structural labor economics models (Ben-Porath, Mincer, Autor, Keane & Wolpin) in the `README.md` and `PAPER_DRAFTS_V6.md`. This clarifies how the learned embeddings can be interpreted as rich, non-parametric approximations of key latent variables in those frameworks.
- **Data Source Clarity:** Added explicit explanatory notes in all public-facing documents (`README.md`, `PAPER_DRAFTS_V6.md`, `Research_Proposal_v9_Final.md`, `Motivation_Letter_v9_Final_CBS.md`) to clarify that the semi-synthetic DGP is calibrated with real-world data from the **NLSY79** and **Felten et al. (2021)**.
- **Updated Documents:** All key documents (`PAPER_DRAFTS`, `Research_Proposal`, `Motivation_Letter`, `README`) have been updated to version 6.0+ to reflect these changes.

## [5.0] - 2026-02-16

This was a major release reflecting the culmination of the research journey, including robustness corrections and the implementation of a semi-synthetic DGP.

### Added
- **Semi-Synthetic DGP:** A new data generator calibrated with real-world parameters from the NLSY79 and AI exposure scores from Felten et al. (2021).
- **Corrected Pipeline:** A new main script (`main_board_corrected.py`) that implements two key corrections: equal dimensionality across all embedding variants (`phi_dim=64`) and a realistic treatment effect (`TRUE_ATE=0.08`).
- **Power Analysis Module:** `src/power_analysis.py` computes the Minimum Detectable Effect (MDE) for any given sample size, formalising the Signal-to-Noise Frontier.

### Key Findings
- **Embedding Ordering Phenomenon Confirmed:** With equal dimensions (`phi_dim=64`), the VIB variant still exhibits the highest bias (160.6% error), confirming the result is not a dimensional artifact.
- **Signal-to-Noise Frontier Characterised:** With a realistic ATE of 0.08, all models fail (errors > 100%), demonstrating the limits of causal inference with small samples.
- **Consistent Superiority over Heckman:** The DML framework demonstrates 88-95% bias reduction over the classical Heckman model across all three pipeline configurations.

## [3.0 - 4.0] - 2026-01-15 to 2026-02-13

### Added
- Causal GRU (VIB) and Debiased GRU (Adversarial) variants.
- Statistical inference (SE, CI, p-values), GATES test, Oster delta, and VIB sensitivity analysis.
- Properly-identified Heckman benchmark with a valid exclusion restriction.

## [1.0 - 2.0] - 2025-12-20

### Added
- Initial implementation of the CAREER-DML pipeline with a Predictive GRU.
