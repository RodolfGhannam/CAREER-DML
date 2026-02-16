# Research Proposal: Causal Inference on Career Trajectories with Large-Scale Administrative Data

**Author:** Rodolf Mikel Ghannam Neto

**Date:** February 16, 2026

**Version:** 8.0

---

## 1. Research Question and Motivation

The central question of this PhD project is: **What is the causal effect of career transitions, particularly those related to technological change like AI, on individual wage trajectories?**

Understanding this question is of paramount importance for individuals navigating their careers, for firms managing human capital, and for governments designing labor market policies. However, estimating this effect is notoriously difficult due to high-dimensional confounding from individuals' past career histories.

To address this, I have developed **CAREER-DML**, a novel framework combining deep learning for representation learning with Double/Debiased Machine Learning (DML) for robust causal estimation. This framework has been validated on a semi-synthetic DGP and has already yielded two key scientific insights:

1.  **The Embedding Paradox:** A demonstration that causally-motivated embeddings can be counterproductive in sequential career data.
2.  **The Signal-to-Noise Frontier:** A characterization of the sample size limitations for detecting realistic, small-magnitude causal effects.

This proposal outlines the plan to scale this validated framework using the Danish administrative registers, leveraging the unique opportunity at CBS to move from simulation to real-world impact.

## 2. Contribution to the Literature

This project contributes to three main streams of literature:

1.  **Causal Machine Learning:** We extend the work of Chernozhukov et al. (2018) and Veitch et al. (2020) by applying their methods to the domain of sequential socio-economic data and identifying the boundary conditions under which they are effective.
2.  **Labor Economics:** We provide a new, more robust methodology for estimating treatment effects in the presence of complex, path-dependent career histories, moving beyond the limitations of classical models.
3.  **Economics of AI:** We aim to provide some of the first large-scale, causally-identified estimates of the returns to AI exposure at the individual level.

## 3. Methodology: From MVP to Large-Scale Research

My research plan for the PhD is structured as a direct response to the (scientifically productive) limitations identified in the CAREER-DML MVP. The external review of my work noted that to strengthen its contribution, it should "explore more the dimensão dinâmica, ampliar o diálogo com literatura estrutural, e explicitar com maior clareza os limites de generalização". My PhD agenda does precisely that.

**Year 1: Scaling and Replication**

- **Objective:** Apply the existing CAREER-DML framework to the full Danish administrative dataset. This directly addresses the **Signal-to-Noise Frontier**; our power analysis shows that with N > 1,000,000, we can detect effects as small as 0.26%, far below the 8% wage premium we are interested in.
- **Method:** Replicate the gain decomposition analysis (Heckman vs. LASSO/RF vs. Embeddings) on the full dataset. This will test the hypothesis that the incremental value of sequential embeddings becomes apparent at large N.

**Year 2: Dynamic Treatment Effects**

- **Objective:** Extend the framework to estimate dynamic treatment effects, τ(t). This addresses the reviewer's point about the "dimensão dinâmica". We will seek to answer: How does the wage premium for AI exposure evolve in the years following the transition? Does it grow, decay, or remain constant?
- **Method:** Adapt the DML framework for dynamic treatment effects, potentially using methods like sequence-to-sequence models or by estimating effects for different time horizons post-treatment.

**Year 3: Heterogeneity and Structural Dialogue**

- **Objective:** Explore deep heterogeneity in the treatment effects and connect our findings to structural models of human capital. This addresses the reviewer's call for a "diálogo com literatura estrutural".
- **Method:** Estimate Conditional Average Treatment Effects (CATEs) based on career stage, prior skill set, and industry. We will then use these rich, non-parametric estimates to calibrate or test the predictions of structural models of career choice and human capital investment.

## 4. Data

This project will leverage the world-class administrative data available through Statistics Denmark, including longitudinal data on employment, education, income, and firm characteristics for the entire Danish population. This dataset is uniquely suited to this research due to its scale (N > 1M), temporal depth (T > 30 years), and granularity.

## 5. Supervision and Fit with CBS

This research is a perfect fit for the research environment at CBS and the supervision of Prof. Kongsted. The project directly aligns with the strategic research area of "Digital Transformations" and leverages the unique data infrastructure of Denmark. My background in econometrics and machine learning, combined with the proven CAREER-DML framework, provides a strong foundation to execute this ambitious agenda.
