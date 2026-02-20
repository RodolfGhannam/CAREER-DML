# Research Proposal: Causal Inference on Career Trajectories with Large-Scale Administrative Data

**Author:** Rodolf Mikel Ghannam Neto

**Date:** February 20, 2026

**Version:** 9.0

---

## 1. Research Question and Motivation

The central question of this PhD project is: **What is the causal effect of career transitions, particularly those related to technological change like AI, on individual wage trajectories?**

To address this, I have developed **CAREER-DML**, a novel framework combining deep learning for representation learning with Double/Debiased Machine Learning (DML) for robust causal estimation. The framework has been validated in a **semi-synthetic data laboratory**, where the simulation parameters are calibrated with real-world data to ensure relevance. This initial work has already yielded two key scientific insights that form the foundation of this proposal:

1.  **A Sequential Embedding Ordering Phenomenon:** A consistent empirical finding that simpler predictive embeddings outperform more complex, causally-motivated embeddings in our specific domain, opening a clear avenue for future methodological investigation.
2.  **The Signal-to-Noise Frontier:** A characterization of the sample size limitations for detecting realistic, small-magnitude causal effects, providing a rigorous justification for the use of large-scale administrative data.

This proposal outlines the plan to scale this validated framework using the Danish administrative registers, leveraging the unique opportunity at CBS to move from simulation to real-world impact.

### Data Source Note

> The semi-synthetic data laboratory uses a Data Generating Process (DGP) calibrated with two real-world US data sources: the **National Longitudinal Survey of Youth 1979 (NLSY79)**, which informs the parameters of labor market dynamics, and the **Felten et al. (2021) AI Occupational Exposure (AIOE) scores**, which define the treatment variable. This ensures that our initial findings are grounded in realistic labor market structures.

## 2. Contribution to the Literature

This project contributes to three main streams of literature:

1.  **Causal Machine Learning:** We extend the work of Chernozhukov et al. (2018) and Veitch et al. (2020) by applying their methods to the domain of sequential socio-economic data and identifying the boundary conditions under which they are effective.
2.  **Labor Economics:** We provide a new, more robust methodology for estimating treatment effects in the presence of complex, path-dependent career histories. Crucially, we build an explicit bridge to classical structural models (e.g., Ben-Porath, 1967; Mincer, 1974), showing how our learned embeddings can be interpreted as rich, non-parametric approximations of key latent variables like human capital.
3.  **Economics of AI:** We aim to provide some of the first large-scale, causally-identified estimates of the returns to AI exposure at the individual level, moving beyond correlations to identify causal impacts.

## 3. PhD Research Plan

My research plan is structured to build directly upon the findings from the CAREER-DML proof-of-concept.

**Year 1: Scaling, Replication, and the Structural Bridge**

- **Objective:** Apply the existing CAREER-DML framework to the full Danish administrative dataset. This directly addresses the **Signal-to-Noise Frontier** by leveraging a population-scale dataset (N > 1M).
- **Method:** 
    1. Replicate the gain decomposition analysis (Heckman vs. LASSO/RF vs. Embeddings) on the full dataset to test if the **Embedding Ordering Phenomenon** holds at scale.
    2. Deepen the **dialogue with structural models**. We will conduct a formal analysis of the learned embeddings (`z_i`) to test their correlation with concepts from structural models, such as mapping the principal components of `z_i` to measures of task complexity or career volatility.

**Year 2: Dynamic Treatment Effects**

- **Objective:** Extend the framework to estimate dynamic treatment effects, τ(t). We will seek to answer: How does the wage premium for AI exposure evolve in the years following the transition?
- **Method:** Adapt the DML framework for dynamic treatment effects, potentially using methods like sequence-to-sequence models or by estimating effects for different time horizons post-treatment.

**Year 3: Deep Heterogeneity and Policy Simulation**

- **Objective:** Explore deep heterogeneity in the treatment effects and use the estimated model for policy simulations.
- **Method:** Estimate Conditional Average Treatment Effects (CATEs) based on career stage, prior skill set, and industry. We will then use these rich, non-parametric estimates to simulate the potential impact of different policy interventions, such as retraining subsidies targeted at specific worker profiles.

## 4. Data and Supervision

This project will leverage the world-class administrative data available through Statistics Denmark. This dataset is uniquely suited to this research due to its scale, temporal depth, and granularity. This research is a perfect fit for the research environment at CBS and the supervision of Prof. Kongsted, whose work on labor mobility and innovation using Danish register data aligns perfectly with the project's goals. My background in econometrics and machine learning, combined with the proven CAREER-DML framework, provides a strong foundation to execute this ambitious agenda.
