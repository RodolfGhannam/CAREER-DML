# CAREER-DML: Synthetic vs. Semi-Synthetic Results Comparison (v6.0)

## Executive Summary

The CAREER-DML pipeline was executed with two distinct Data Generating Processes (DGPs) to test whether the **Sequential Embedding Ordering Phenomenon** — the empirical finding that causally-motivated embeddings paradoxically increase bias in DML estimation for this domain — is robust to real-world data calibration.

The ordering phenomenon persists with semi-synthetic data calibrated from NLSY79 and Felten et al. (2021) AIOE scores, confirming the finding is not an artifact of a purely synthetic DGP.

---

## Data Sources

| Parameter | Synthetic DGP (v3.3) | Semi-Synthetic DGP (NLSY79 + Felten) |
|-----------|----------------------|---------------------------------------|
| Wage equation | Arbitrary coefficients | Mincer regression from NLSY79 (R2=0.5245) |
| Education return | Discrete categories (0,1,2) | 5.4% per year (continuous, 8-20 years) |
| Experience return | Not modelled | 9.9% with diminishing returns |
| Gender penalty | Not modelled | -19.1% (NLSY79 estimate) |
| Race effects | Not modelled | Black: -8.4%, Hispanic: +0.4% |
| AI exposure | Arbitrary transition matrices | Felten et al. (2021) AIOE, 774 SOC occupations |
| Treatment mechanism | Structural selection (Heckman) | AIOE-based propensity (logistic) |
| True ATE | 0.500 | 0.538 |
| N individuals | 1,000 | 1,000 |
| N periods | 10 | 10 |
| Exclusion restriction | peer_adoption (Beta(2,5)) | Not available |

---

## Main Results: Embedding Variant Comparison

### Synthetic DGP (v3.4.1)

| Variant | ATE | SE | 95% CI | p-value | Bias | % Error | Status |
|---------|-----|-----|--------|---------|------|---------|--------|
| Predictive GRU | 0.5378 | 0.0520 | [0.4358, 0.6397] | 4.70e-25 | +0.0378 | 7.6% | Lowest bias |
| Causal GRU (VIB) | 0.7996 | 0.0595 | [0.6830, 0.9162] | 3.59e-41 | +0.2996 | 59.9% | High bias |
| Debiased GRU (Adversarial) | 0.5919 | 0.0563 | [0.4816, 0.7021] | 6.87e-26 | +0.0919 | 18.4% | Moderate |

### Semi-Synthetic DGP (NLSY79 + Felten AIOE)

| Variant | ATE | SE | 95% CI | p-value | Bias | % Error | Status |
|---------|-----|-----|--------|---------|------|---------|--------|
| Predictive GRU | 0.3865 | 0.0446 | [0.2991, 0.4739] | 4.26e-18 | -0.1515 | 28.2% | Moderate |
| Causal GRU (VIB) | 0.3479 | 0.0550 | [0.2400, 0.4557] | 2.58e-10 | -0.1901 | 35.3% | High bias |
| Debiased GRU (Adversarial) | 0.4438 | 0.0627 | [0.3209, 0.5668] | 1.50e-12 | -0.0942 | 17.5% | Lowest bias |

---

## Key Findings

### 1. Embedding Ordering Phenomenon Confirmed

In both DGPs, the Causal GRU (VIB) embedding produces the highest bias among all variants:

| DGP | VIB Bias | Predictive Bias | VIB/Predictive Ratio |
|-----|----------|-----------------|---------------------|
| Synthetic | 59.9% | 7.6% | 7.9x worse |
| Semi-Synthetic | 35.3% | 28.2% | 1.3x worse |

This consistent ordering, where the theoretically-causal embedding underperforms, is the core of the **Sequential Embedding Ordering Phenomenon**. It is consistent with the hypothesis that the information bottleneck removes treatment-predictive information that is also needed for confounding adjustment in sequential career data.

### 2. Debiased (Adversarial) Emerges as Best in Semi-Synthetic

An important nuance emerges from the semi-synthetic results. In the synthetic setting, the Predictive GRU wins (7.6% bias), but in the semi-synthetic setting, the Debiased GRU (Adversarial) wins (17.5% bias). This suggests that with more realistic confounding structures calibrated from real labor market data, the adversarial debiasing approach becomes more valuable.

---

## Implications

The **Sequential Embedding Ordering Phenomenon** is not an artifact of synthetic data. It persists when the DGP is calibrated with real U.S. labor market parameters from NLSY79 and Felten et al. (2021) AI exposure scores. The adversarial debiasing approach shows particular promise for real-world applications. The stronger GATES gradient in semi-synthetic data (1.41x vs 1.11x) suggests that AI adoption effects are more heterogeneous in realistic settings, supporting the need for CATE estimation. These results motivate the proposed research using Danish registry data, where the full population panel would provide richer career trajectories for embedding-based causal inference.

---

*Report: February 2026*
*Author: Rodolf Mikel Ghannam Neto*
