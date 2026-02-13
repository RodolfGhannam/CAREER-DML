# Changelog

All notable changes to CAREER-DML are documented in this file. This project follows [Semantic Versioning](https://semver.org/).

## [3.3] - 2026-02-13

### Added
- Statistical inference for all ATE estimates: standard errors, 95% confidence intervals, and p-values via `CausalForestDML.ate_inference()` (Wager & Athey, 2018)
- Formal GATES heterogeneity test: Welch's t-test for Q1 vs Q5 with Cohen's d effect size
- VIB sensitivity analysis: β sweep across 7 values (0.0001 to 1.0)
- Exclusion restriction (`peer_adoption`) in the DGP for a properly-identified Heckman benchmark
- Robustness test comparing structural (rational decision) vs. mechanical (probabilistic) selection
- Limitations and Future Work section in README
- FAQ section in README
- Unit tests (`tests/`)
- CHANGELOG.md, expanded CONTRIBUTING.md

### Changed
- Heckman benchmark now uses proper exclusion restriction, satisfying the identifying assumption of Heckman (1979)
- README rewritten with figure captions, expanded How to Run, and improved Citation section
- All language revised to factual academic tone

### Results
- Predictive GRU: ATE = 0.5378, SE = 0.0520, 95% CI [0.4358, 0.6397], bias = 7.6%
- Debiased GRU: ATE = 0.5919, SE = 0.0563, 95% CI [0.4816, 0.7021], bias = 18.4%
- VIB GRU: ATE = 0.7996, SE = 0.0595, 95% CI [0.6830, 0.9162], bias = 59.9%
- Heckman (with exclusion): ATE = 1.0413, bias = 0.5413
- DML bias reduction vs. Heckman: 93.0%
- GATES Q1 vs Q5: t = 62.27, p < 10^-200
- Oster delta = 13.66

## [3.2] - 2026-02-08

### Added
- Debiased GRU with adversarial training (λ_adv = 1.0)
- Variance decomposition analysis
- GATES heterogeneity analysis (5 quantiles)
- Comprehensive validation suite (Oster delta, placebo tests)

### Changed
- Improved DGP with time-varying confounding
- Refined propensity trimming thresholds [0.05, 0.95]

### Results
- ATE = 0.6712 (bias = 0.1712, 34.2% error)
- Debiased GRU outperformed Predictive and VIB variants
- 90.6% bias reduction vs. Heckman (without exclusion restriction)

## [3.1] - 2026-01-15

### Added
- Causal GRU with VIB regularisation
- PLS dimensionality reduction
- Initial GATES implementation

### Results
- ATE = 0.7240 (bias = 0.2240)
- Identified over-regularisation issue with VIB

## [3.0] - 2025-12-20

### Added
- Initial implementation of CAREER-DML pipeline
- Predictive GRU baseline
- Heckman two-step comparator (without exclusion restriction)
- Basic DML pipeline with CausalForestDML

### Results
- Proof of concept
- ATE = 0.8156 (bias = 0.3156)
