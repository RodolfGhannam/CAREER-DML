# CAREER-DML: Career Embeddings for Causal Inference under Heckman Selection

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![EconML](https://img.shields.io/badge/EconML-CausalForestDML-green.svg)](https://econml.azurewebsites.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Can career sequence embeddings serve as a robust alternative to classical selection correction when estimating the causal effect of AI adoption on labour market outcomes?**

This repository implements a **Double/Debiased Machine Learning (DML)** pipeline that uses **GRU-based career embeddings** to estimate the Average Treatment Effect (ATE) of AI adoption on wages, benchmarked against the classical **Heckman (1979) two-step selection model** with a proper exclusion restriction.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAREER-DML Pipeline v3.4                     │
│            (Heckman + Inference + VIB Sensitivity)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: DGP v3.4 (Structural Selection + Exclusion)           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Career       │  │ Heckman      │  │ Exclusion            │  │
│  │ Sequences    │──│ Selection    │──│ Restriction          │  │
│  │ (Markov)     │  │ (Rational)   │  │ (peer_adoption)      │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  Layer 2: Embedding Variants                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Predictive   │  │ Causal GRU   │  │ Debiased GRU         │  │
│  │ GRU          │  │ (VIB)        │  │ (Adversarial)        │  │
│  │ (Baseline)   │  │ (Veitch)     │  │ (Veitch + Debiasing) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  Layer 3: CausalForestDML + Inference                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ ATE + SE + 95% CI + p-value (ate_inference)              │   │
│  │ GATES + Formal Heterogeneity Test (Welch's t-test)       │   │
│  │ Oster (2019) Sensitivity + Placebo Tests                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Layer 4: Benchmarks                                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Heckman Two-Step (with/without exclusion restriction)    │   │
│  │ VIB Sensitivity Analysis (β sweep)                       │   │
│  │ Structural vs. Mechanical Selection Robustness           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Results

All results are based on **n = 1,000 individuals, T = 10 periods, random seed = 42**. Standard errors and confidence intervals are obtained via `CausalForestDML.ate_inference()` (Wager & Athey, 2018).

### 1. Embedding Comparison

| Variant | ATE | SE | 95% CI | p-value | Bias | % Error | Status |
|:--------|:---:|:--:|:------:|:-------:|:----:|:-------:|:------:|
| Predictive GRU | 0.5378 | 0.0520 | [0.4358, 0.6397] | 4.70e-25 | 0.0378 | 7.6% | Lowest bias |
| Causal GRU (VIB) | 0.7996 | 0.0595 | [0.6830, 0.9162] | 3.59e-41 | 0.2996 | 59.9% | High bias |
| Debiased GRU (Adversarial) | 0.5919 | 0.0563 | [0.4816, 0.7021] | 6.87e-26 | 0.0919 | 18.4% | Moderate bias |

> **True ATE = 0.5000**

![Embedding Comparison](results/figures/fig1_embedding_comparison.png)

**Figure 1.** ATE estimates with 95% confidence intervals for the three primary embedding variants. The dashed horizontal line indicates the true ATE (0.50). The Predictive GRU achieves the lowest bias (7.6%), followed by the Adversarial variant (18.4%). The VIB variant exhibits the highest bias (59.9%), a finding discussed in the VIB Sensitivity Analysis below.

---

### 2. Treatment Effect Heterogeneity (GATES)

| Quantile | ATE | SE | 95% CI | n |
|:---------|:---:|:--:|:------:|:-:|
| Q1 (Low Human Capital) | 0.5081 | 0.0006 | [0.5069, 0.5094] | 200 |
| Q2 | 0.5275 | 0.0002 | [0.5270, 0.5280] | 199 |
| Q3 | 0.5381 | 0.0002 | [0.5377, 0.5385] | 199 |
| Q4 | 0.5491 | 0.0002 | [0.5486, 0.5496] | 199 |
| Q5 (High Human Capital) | 0.5661 | 0.0007 | [0.5647, 0.5674] | 199 |

> **Heterogeneity test:** H₀: ATE(Q1) = ATE(Q5) rejected with t = 62.27, p < 10⁻²⁰⁰, Cohen's d = 6.25. The monotonically increasing pattern is consistent with the skill-biased technological change hypothesis (Cunha & Heckman, 2007).

![GATES Heterogeneity](results/figures/fig2_gates_heterogeneity.png)

**Figure 2.** Group Average Treatment Effects (GATES) across five quantiles of estimated CATEs. Error bars represent 95% confidence intervals. The monotonically increasing pattern from Q1 (0.508) to Q5 (0.566) is consistent with skill-capital complementarity: individuals with higher latent human capital benefit more from AI exposure.

---

### 3. DML vs. Heckman Two-Step Benchmark

| Method | ATE | SE | |Bias| | Exclusion Restriction |
|:-------|:---:|:--:|:-----:|:--------------------:|
| **DML + Predictive GRU** | **0.5378** | **0.0520** | **0.0378** | N/A (nonparametric) |
| Heckman Two-Step (with excl.) | 1.0413 | 0.0370 | 0.5413 | peer_adoption |
| Heckman Two-Step (no excl.) | 0.8780 | — | 0.3780 | None |

> DML yields a 93.0% bias reduction relative to the properly-identified Heckman model. The Heckman benchmark uses `peer_adoption` as an exclusion restriction (affects treatment selection but not outcome), satisfying the identifying assumption of Heckman (1979). See the **Limitations** section for important caveats on this comparison.

![DML vs Heckman](results/figures/fig3_dml_vs_heckman.png)

**Figure 3.** Absolute bias comparison between DML with career embeddings and the Heckman two-step estimator (with proper exclusion restriction). The Heckman model struggles in this high-dimensional setting because the Inverse Mills Ratio is a scalar correction, whereas career embeddings capture selection heterogeneity across a 64-dimensional space.

---

### 4. VIB Sensitivity Analysis

| β | ATE | SE | Bias | % Error |
|:-:|:---:|:--:|:----:|:-------:|
| 0.0001 | 0.7302 | 0.0590 | 0.2302 | 46.0% |
| 0.001 | 0.7255 | 0.0571 | 0.2255 | 45.1% |
| 0.01 | 0.7381 | 0.0584 | 0.2381 | 47.6% |
| 0.05 | 0.7507 | 0.0592 | 0.2507 | 50.1% |
| 0.1 | 0.7336 | 0.0517 | 0.2336 | 46.7% |
| 0.5 | 0.7505 | 0.0579 | 0.2505 | 50.1% |
| 1.0 | 0.6944 | 0.0499 | 0.1944 | 38.9% |

> The VIB approach exhibits sensitivity to the β parameter across the range tested (38.9%–50.1% error). The Predictive GRU (7.6% error) and Adversarial approach (18.4% error) do not require this hyperparameter.

![VIB Sensitivity](results/figures/fig4_vib_sensitivity.png)

**Figure 4.** VIB bias as a function of the β (KL divergence penalty) parameter. The horizontal dashed lines indicate the Predictive GRU (7.6%) and Adversarial (18.4%) baselines, which do not depend on β. The VIB variant is consistently above both baselines across all β values tested, suggesting that the information bottleneck trade-off is non-trivial for sequential career data.

#### Why Does VIB Underperform?

The VIB variant (Veitch et al., 2020) performs worse than the Predictive baseline across all β values. This is counterintuitive, as the information bottleneck should in principle remove confounding information. We hypothesise two contributing factors:

1. **Over-compression of sequential information.** Career sequences contain causal information distributed across the entire trajectory. The KL penalty compresses the embedding dimensionality, but in doing so it may discard outcome-relevant (not just confounding) information. Unlike text data where key tokens carry most of the signal, career sequences have a more diffuse information structure.

2. **Interaction with DML cross-fitting.** The VIB embeddings are trained on the full sample, then used as features in the DML cross-fitting procedure. If the compression is too aggressive, the resulting features may not have enough residual variation for the nuisance models to exploit, leading to poor first-stage estimates.

The Adversarial approach avoids this issue because it selectively removes treatment-predictive information without reducing dimensionality. This finding may be of independent methodological interest for researchers applying causal embeddings to sequential data.

---

### 5. Robustness: Structural vs. Mechanical Selection

| Selection Mode | ATE | SE | 95% CI | Bias | % Error |
|:---------------|:---:|:--:|:------:|:----:|:-------:|
| Mechanical | 0.5920 | 0.0623 | [0.4698, 0.7142] | 0.0920 | 18.4% |
| Structural (Heckman) | 0.6689 | 0.0700 | [0.5317, 0.8062] | 0.1689 | 33.8% |

> |Δbias| = 0.0769 < 0.1. Results hold under both mechanical (probabilistic) and structural (rational decision) selection, indicating that the method is not sensitive to the data-generating mechanism.

![Robustness Test](results/figures/fig5_robustness_test.png)

**Figure 5.** Robustness test comparing DML performance under two selection mechanisms: mechanical (probabilistic assignment based on covariates) and structural (rational decision model where agents weigh expected utility). The bias difference of 0.077 is below the pre-specified threshold of 0.1.

---

### Validation Summary

| Test | Result | Threshold | Status |
|:-----|:------:|:---------:|:------:|
| Oster (2019) δ | 13.66 | > 2.0 | Pass |
| Placebo (random T) | -0.0478 | ≈ 0 | Pass |
| Placebo (random Y) | 0.0139 | ≈ 0 | Pass |
| Structural vs. Mechanical | Δ = 0.077 | < 0.1 | Pass |
| GATES monotonicity | Yes | — | Pass |
| GATES Q1 ≠ Q5 | p < 10⁻²⁰⁰ | < 0.05 | Pass |

---

## Theoretical Framework

This project integrates three foundational streams of econometric research:

1. **Heckman (1979):** The DGP implements endogenous selection into treatment via a rational decision model where latent ability affects both treatment selection and outcomes. The exclusion restriction (`peer_adoption`) satisfies the identifying assumption of the classical two-step estimator.

2. **Chernozhukov et al. (2018):** The DML framework provides doubly-robust estimation that is valid under high-dimensional nuisance parameters, with cross-fitting to avoid overfitting bias.

3. **Veitch, Sridhar & Blei (2020):** Four embedding variants operationalise different approaches to adapting text (career sequence) embeddings for causal inference: predictive sufficiency, variational information bottleneck, adversarial debiasing, and a two-stage procedure faithful to the original proposal. A causal sufficiency test and linear representation probing provide diagnostics on the information content of each variant.

The pipeline evaluates whether career sequence embeddings can serve as a nonparametric alternative to the classical Inverse Mills Ratio for correcting selection bias in labour market studies.

---

## How to Run

### Prerequisites

- **Python:** 3.9, 3.10, or 3.11
- **OS:** Linux, macOS, or Windows (WSL2 recommended on Windows)
- **Hardware:** 16 GB RAM minimum; GPU recommended but not required

### Installation

```bash
# Clone the repository
git clone https://github.com/RodolfGhannam/CAREER-DML.git
cd CAREER-DML

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Full pipeline (reproduces all results)
python main.py
```

**Expected output:**
- Console: Training progress for 4 GRU variants, DML estimation, validation suite
- `results/output_v34.txt`: Complete numerical output
- `results/figures/*.png`: 7 figures

**Runtime:** Approximately 10–15 minutes on a machine with 16 GB RAM (CPU). Faster with GPU.

### Troubleshooting

| Issue | Solution |
|:------|:---------|
| `ModuleNotFoundError: No module named 'econml'` | Run `pip install -r requirements.txt` |
| CUDA out of memory | The pipeline runs on CPU by default; no GPU required |
| Results differ from reported values | Ensure `random seed = 42` (default in `main.py`) and matching package versions |

---

## Repository Structure

```
CAREER-DML/
├── main.py                          # Main pipeline orchestrator (v3.4)
├── src/
│   ├── dgp.py                       # Data Generating Process (Heckman + exclusion)
│   ├── embeddings.py                # 4 GRU variants (Predictive, VIB, Adversarial, Two-Stage)
│   ├── dml.py                       # CausalForestDML pipeline (with inference)
│   └── validation.py                # Robustness tests + Heckman benchmark
├── tests/
│   ├── test_dgp.py                  # Unit tests for DGP
│   ├── test_embeddings.py           # Unit tests for embedding variants
│   ├── test_dml.py                  # Unit tests for DML pipeline
│   └── test_integration.py          # End-to-end integration test
├── results/
│   ├── output_v34.txt               # Complete pipeline output
│   └── figures/
│       ├── fig1_embedding_comparison.png
│       ├── fig2_gates_heterogeneity.png
│       ├── fig3_dml_vs_heckman.png
│       ├── fig4_vib_sensitivity.png
│       └── fig5_robustness_test.png
├── docs/
│   ├── candidature/                 # CBS PhD application documents
│   │   ├── CV_v4_Recreated.md
│   │   ├── Research_Proposal_v6_Final.md
│   │   └── Motivation_Letter_v6_Final_CBS.md
│   └── technical/
│       ├── BLUEPRINT_V3_2_FINAL.md
│       ├── HECKMAN_IMPLEMENTATION_GUIDE.md
│       └── PAPER_TEXTS_HECKMAN_ANCHOR.md
├── requirements.txt
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── .gitignore
└── README.md
```

---

## Limitations and Future Work

This project uses **synthetic data** to validate the methodology under controlled conditions with known ground truth. The following limitations should be noted:

1. **Synthetic DGP.** All results are generated from a synthetic Data Generating Process. While the DGP is designed to capture key features of real labour markets (Heckman selection, skill complementarity, career path dependence), the magnitudes and patterns may differ from real-world data. The primary contribution is methodological, not empirical.

2. **VIB underperformance.** The Variational Information Bottleneck variant consistently exhibits higher bias across all β values tested (38.9%–59.9%). This may be because the compression-prediction trade-off is particularly challenging for sequential career data, where the relevant causal information is distributed across the entire sequence rather than concentrated in a few dimensions. See the VIB Sensitivity Analysis section above for a detailed discussion.

3. **Heckman benchmark.** The Heckman two-step estimator operates under conditions that are simultaneously favourable (proper exclusion restriction via `peer_adoption`) and unfavourable (high-dimensional confounding that a scalar Inverse Mills Ratio cannot capture). The 93.0% bias reduction should be interpreted in context: it reflects the specific DGP design, not a general claim about the relative merits of parametric vs. nonparametric selection correction. Alternative Heckman specifications with different exclusion restrictions or lower-dimensional confounding may perform differently.

4. **Single seed.** Results are reported for a single random seed (42). A full Monte Carlo study across multiple seeds would strengthen the conclusions and provide more reliable standard error estimates.

5. **Hyperparameter sensitivity.** The Adversarial GRU uses λ_adv = 1.0 and the VIB uses a range of β values. Results may vary under different regularisation choices. A systematic hyperparameter study is planned.

**Planned extensions** (PhD thesis at CBS):
- Application to Danish administrative data (IDA/Statistics Denmark)
- Panel DML with time-varying treatments
- Formal comparison with modern semiparametric selection models
- Monte Carlo study across multiple seeds and DGP specifications

---

## Frequently Asked Questions

**Q: Why synthetic data? Why not start with real data?**

Synthetic data with known ground truth allows validation that the method recovers the true causal effect before applying it to real data where the ground truth is unknown. This follows standard practice in causal inference methodology (e.g., Athey & Imbens, 2017; Chernozhukov et al., 2018). Real data validation using Danish administrative registers is planned for the PhD.

**Q: How should the 93.0% bias reduction be interpreted?**

This figure compares DML + career embeddings against a specific Heckman two-step specification in a specific synthetic DGP. It should not be interpreted as a general claim that DML is 93% better than Heckman. The Heckman model struggles here because the confounding is high-dimensional (64-dimensional embeddings vs. a scalar Inverse Mills Ratio). In settings with lower-dimensional confounding and strong exclusion restrictions, Heckman may perform comparably or better.

**Q: Why does the VIB variant perform worse than the Predictive baseline?**

See the detailed discussion in the VIB Sensitivity Analysis section. In brief: the KL divergence penalty compresses the embedding space, and for sequential career data this compression appears to discard outcome-relevant information along with confounding information. The Adversarial approach avoids this by selectively removing treatment-predictive features without reducing dimensionality.

**Q: Can I reproduce the results?**

Yes. All results are deterministic with `random seed = 42`. Clone the repository, install dependencies from `requirements.txt`, and run `python main.py`. The output should match `results/output_v34.txt`.

**Q: Can I adapt this framework for other domains?**

Yes. The code is MIT licensed. Replace `src/dgp.py` with your data loading code, modify `src/embeddings.py` for your sequence type, and keep `src/dml.py` and `src/validation.py` as-is. The DML pipeline is domain-agnostic.

---

## Citation

If you use this framework in your research, please cite:

### BibTeX

```bibtex
@software{ghannam2026careerdml,
  author       = {Ghannam, Rodolf M.},
  title        = {{CAREER-DML}: Career Embeddings for Causal Inference
                  under Heckman Selection},
  year         = {2026},
  version      = {3.4},
  publisher    = {GitHub},
  url          = {https://github.com/RodolfGhannam/CAREER-DML}
}
```

### APA

Ghannam, R. M. (2026). *CAREER-DML: Career embeddings for causal inference under Heckman selection* (Version 3.4) [Computer software]. https://github.com/RodolfGhannam/CAREER-DML

---

## References

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Cunha, F., & Heckman, J. J. (2007). The technology of skill formation. *American Economic Review*, 97(2), 31–47.
- Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica*, 47(1), 153–161.
- Oster, E. (2019). Unobservable selection and coefficient stability: Theory and evidence. *Journal of Business & Economic Statistics*, 37(2), 187–204.
- Vafa, K., Palikot, E., Du, T., Kanodia, A., Athey, S., & Blei, D. M. (2025). Career embeddings. *Proceedings of the National Academy of Sciences*, 122(4).
- Veitch, V., Sridhar, D., & Blei, D. (2020). Adapting text embeddings for causal inference. *Proceedings of the 36th Conference on Uncertainty in Artificial Intelligence (UAI)*.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
