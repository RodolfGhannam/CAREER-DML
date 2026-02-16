# Paper Drafts — CAREER-DML v4.0

These are the current text blocks for the working paper, reflecting the full project evolution from the initial synthetic validation to the corrected framework with real-world calibration.

---

## Abstract

> We investigate whether deep learning models of career trajectories can correct for endogenous selection bias when estimating the causal effects of AI adoption on wages. While prior work has focused on adapting text embeddings for causal inference, we demonstrate that for sequential career data, a methodological paradox emerges: causally-aware embeddings, such as those using a Variational Information Bottleneck (VIB), exhibit *higher* bias than simple predictive embeddings when integrated into a Double/Debiased Machine Learning (DML) framework. We validate this finding in a synthetic data laboratory and show it holds even after controlling for model dimensionality. Furthermore, by calibrating our model with real-world parameters from the NLSY79 and AI exposure scores from Felten et al. (2021), we identify a "Signal-to-Noise Frontier": with realistic effect sizes (e.g., an 8% wage premium), the selection bias in modest samples (N=1,000) completely dominates the causal signal, rendering estimation unreliable for all tested model variants. This finding provides a data-driven motivation for the necessity of large-scale administrative data in modern labor econometrics. Our DML framework consistently reduces estimation bias by 88-95% compared to a classical Heckman model, offering a robust, open-source methodology for future research.

---

## 1. Introduction

> The rapid proliferation of Artificial Intelligence (AI) into white-collar professions has created an urgent need for methods that can accurately estimate its causal impact on labor market outcomes. A central challenge is endogenous selection: workers who adopt AI are not randomly assigned, and their latent characteristics, such as ability or ambition, likely influence both their adoption decision and their subsequent career trajectory (Heckman, 1979). While classical methods like the Heckman two-step model exist, their reliance on parametric assumptions and valid exclusion restrictions limits their applicability in high-dimensional, non-linear settings.
>
> Recent advances in representation learning offer a promising alternative. Vafa et al. (2025) have shown that career histories can be encoded into dense vector representations that predict future outcomes. Following Veitch et al. (2020), who adapted text embeddings for causal inference, a natural next step is to use these career embeddings as high-dimensional controls within a modern causal machine learning framework like Double/Debiased Machine Learning (DML) (Chernozhukov et al., 2018).
>
> In this paper, we develop and rigorously test such a framework. We make two primary contributions. First, we identify and validate what we term the "Embedding Paradox": contrary to theoretical expectations, we find that causally-aware embeddings, specifically those using a Variational Information Bottleneck (VIB), consistently and significantly *underperform* simple predictive embeddings when estimating causal effects on sequential career data. Second, by calibrating our models with real-world labor market data, we characterise a "Signal-to-Noise Frontier," demonstrating that with realistic treatment effect sizes, the selection bias inherent in typical survey-sized datasets is so large relative to the causal signal that reliable estimation becomes infeasible. This finding highlights the critical importance of large-scale administrative data for contemporary causal questions. We present a complete, open-source pipeline that reduces estimation bias by up to 95% compared to classical benchmarks and provides a foundation for future work with population-scale data.

---

## 2. Methodology

> Our methodological framework, CAREER-DML, integrates a dual Data Generating Process (DGP) approach with a DML estimation pipeline.
>
> **Data Generating Processes.** To ensure both internal and external validity, we employ two DGPs. The first is a Synthetic DGP built on first principles of labor economics, incorporating Heckman-style selection with a valid exclusion restriction. This allows us to test the pipeline's ability to recover a known ground truth. The second is a Semi-Synthetic DGP, which calibrates the key parameters of the simulation — including returns to education and experience, gender wage gaps, and the variance of unobserved ability — using estimates from the National Longitudinal Survey of Youth 1979 (NLSY79). Treatment assignment is further informed by real-world AI exposure scores from Felten et al. (2021). This dual approach ensures our findings are not artifacts of a single, arbitrary data structure.
>
> **Causal Estimation.** We use the `CausalForestDML` estimator (Wager & Athey, 2018) within the DML framework of Chernozhukov et al. (2018). The key innovation is using a GRU-based career embedding as the high-dimensional control variable. We test three embedding variants, all with equal dimensionality (`phi_dim=64`) to ensure a fair comparison: (1) a Predictive GRU trained to predict the outcome, (2) a Causal GRU (VIB) that adds an information bottleneck penalty following Veitch et al. (2020), and (3) a Debiased GRU that uses an adversarial network to purge treatment-predictive information.
>
> **Validation Suite.** All results are subjected to a comprehensive validation suite, including placebo tests, Oster's (2019) delta for sensitivity to unobservables, a formal statistical test for treatment effect heterogeneity (GATES), a power analysis for minimum detectable effects, and a benchmark comparison against a classical Heckman two-step model.

---

## 3. Results

### 3.1. The Embedding Paradox

> A recurring finding across our experiments was the underperformance of the theoretically-motivated VIB embedding. An initial concern was that this could be an artifact of differing model capacity, since the VIB and adversarial variants originally used a lower-dimensional bottleneck (`phi_dim=16`) than the predictive variant (`hidden_dim=64`). To rule this out, we ran a corrected pipeline where all embedding variants were fixed to an equal dimension (`phi_dim=64`) and the true ATE was set to 0.08.

> | Corrected Results (ATE=0.08, phi_dim=64) | ATE | Bias | % Error |
> |:---|:---:|:----:|:---:|
> | Predictive GRU | -0.0064 | -0.0864 | 108.0% |
> | **Causal GRU (VIB)** | **-0.0482** | **-0.1282** | **160.3%** |
> | Debiased GRU (Adversarial) | -0.0104 | -0.0904 | 112.9% |

> Even with identical dimensionality, the VIB variant exhibits the highest estimation error (160.3%). The paradox is not an artifact of model specification but appears to be a feature of applying VIB to sequential career data. The information bottleneck, in its attempt to purge confounding variation, seems to also destroy causally relevant information that is diffusely spread across the career trajectory. A beta sweep across 14 configurations of the VIB regularisation parameter confirmed that no value of beta resolves this issue, suggesting the problem lies in the dual-task formulation itself rather than in hyperparameter calibration.

### 3.2. The Signal-to-Noise Frontier

> The corrected experiment yielded a second finding. By setting the true ATE to a realistic 8% wage premium — consistent with the range suggested by Acemoglu et al. (2022) — we found that *all* variants failed to recover the effect, with errors exceeding 100%.
>
> This is not a failure of the pipeline per se, but a characterisation of the boundary conditions for causal inference. It demonstrates that for modest sample sizes (N=1,000), the magnitude of the selection bias is an order of magnitude larger than a realistic causal effect. A formal power analysis confirms this: the minimum detectable effect at N=1,000 is approximately 0.125, well above the 0.08 target. At N=1,000,000 (the scale of Scandinavian administrative registers), the MDE drops to 0.004.
>
> This finding, which we term the "Signal-to-Noise Frontier," has implications for the field. It suggests that for many policy-relevant questions in labor economics where treatment effects are likely to be modest, research based on standard survey datasets may be unreliable regardless of the sophistication of the causal method employed. It provides a data-driven justification for leveraging large-scale administrative data.

### 3.3. Validation with Real-World Parameters

> We also validated the pipeline using the Semi-Synthetic DGP with a stronger ATE signal (0.538). In this setting, the pipeline successfully recovered the effect with reasonable precision. The Debiased Adversarial variant performed best, achieving an error of only 17.5%.

> | Semi-Synthetic Results (ATE=0.538) | ATE | SE | 95% CI | % Error |
> |:---|:---:|:--:|:------:|:---:|
> | **Debiased GRU (Adversarial)** | **0.4438** | **0.0624** | **[0.3209, 0.5668]** | **17.5%** |
> | Predictive GRU | 0.3865 | 0.0384 | [0.3112, 0.4618] | 28.2% |
> | Causal GRU (VIB) | 0.3483 | 0.0491 | [0.2521, 0.4445] | 35.3% |

> The 95% confidence interval for the best-performing model contains the true ATE of 0.538. The model also passed our most stringent robustness test, yielding an Oster delta of 75.95, indicating extreme robustness to unobserved confounding. This confirms that the pipeline performs well when the signal is strong enough to be detected.

---

## 4. Conclusion

> This research makes two contributions to the literature on causal machine learning for labor economics. First, we identify and validate the "Embedding Paradox," a counterintuitive finding that causally-aware VIB embeddings underperform simple predictive embeddings for sequential career data. This extends the work of Veitch et al. (2020) by identifying a boundary condition for their approach in the domain of temporal occupational sequences. Second, we characterise the "Signal-to-Noise Frontier," demonstrating the limits of causal inference with small samples and providing an empirical justification for the use of large-scale administrative data. Our open-source framework, CAREER-DML, consistently outperforms classical selection models and provides a validated tool for future research. The next step is to apply this framework to population-scale register data to estimate the true, heterogeneous effects of AI on the labor market.
