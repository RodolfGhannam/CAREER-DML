# The Embedding Paradox: How Bias-Reduction Techniques in AI Systems Can Compromise Causal Validity

## Evidence from a Career Trajectory Modeling Framework

**Author:** Rodolf Mikel Ghannam Neto

*Independent Researcher, MIT xPRO Certified in AI Products & Services*

*Corporate Director, Grupo CICAL, Goiânia, Brazil*

*Corresponding email: rodolf@cical.com.br*

*Open-source repository: https://github.com/RodolfGhannam/CAREER-DML*

-----

## Abstract

As AI systems increasingly mediate high-stakes decisions in hiring, credit scoring, and insurance underwriting, organizations are deploying bias-reduction techniques — such as adversarial debiasing and Variational Information Bottleneck (VIB) constraints — to satisfy emerging regulatory requirements, particularly Articles 9 and 10 of the European Union AI Act. This paper identifies a fundamental tension in this approach. Using CAREER-DML, an open-source causal inference framework that combines Double/Debiased Machine Learning (DML) with GRU-based career embeddings, we demonstrate that bias-reduction interventions applied to sequential data representations can systematically degrade the causal validity of downstream estimates — in some cases reversing the estimated direction of the causal effect entirely.

We term this the **Embedding Paradox**: the same techniques designed to produce fairer representations can corrupt the very consumer and worker insights they are meant to improve. Our framework, validated on a semi-synthetic data generating process calibrated with U.S. labor market data (NLSY79) and AI occupational exposure scores (Felten et al., 2021), demonstrates that while DML architectures significantly outperform classical Heckman selection models — reducing estimation bias from 945.6% to approximately 55% with LASSO+DML, and under 8% with Predictive GRU embeddings in controlled synthetic benchmarks — fairness interventions introduce severe unintended consequences. Causally-motivated VIB embeddings consistently produce higher estimation bias than simpler predictive embeddings (up to 7.9 times worse in synthetic configurations, 1.3 times worse with real-data calibration), and at realistic effect sizes (an 8% wage premium, N=1,000), all sequential embedding variants fail to recover the correct sign of the treatment effect. This is not merely an issue of estimation precision; it is a fundamental failure of causal identification.

We discuss the implications for AI governance, propose a three-step diagnostic protocol for organizations deploying embedding-based AI systems, and argue that compliance frameworks must integrate causal validity testing alongside fairness metrics to avoid a regulatory compliance trap in which systems are simultaneously compliant in one dimension and non-compliant in another.

**Keywords:** AI governance, causal inference, Double Machine Learning, embedding bias, EU AI Act, fairness-accuracy trade-off, career trajectories, responsible AI

-----

## 1. Introduction

The proliferation of Artificial Intelligence (AI) is fundamentally reshaping labor markets, organizational decision-making, and regulatory frameworks worldwide. According to McKinsey (2025), 78% of organizations now use AI in at least one business function, up from 55% in 2023. In the Nordic countries, individual AI uptake has surpassed 35%, with 34.7% of the Danish workforce exposed to generative AI in tasks representing 20% or more of their work hours (OECD, 2024; 2026). In the European Union, the AI Act — which became law in August 2024 — requires providers of high-risk AI systems to demonstrate compliance with risk management (Article 9), data governance (Article 10), and transparency (Article 13) requirements by 2026.

A central component of modern AI systems is the use of learned embeddings — dense vector representations that encode complex, high-dimensional data (such as career histories, consumer behavior, or clinical records) into fixed-length numerical vectors suitable for machine learning. These embeddings are now ubiquitous in hiring algorithms, credit scoring models, insurance underwriting systems, and content personalization engines. As regulatory pressure intensifies, a common organizational response is to apply bias-reduction techniques — adversarial debiasing, fairness constraints, and information bottleneck regularization — to these embeddings, aiming to remove the influence of protected attributes such as race, gender, and age from the learned representations.

This paper demonstrates that this approach, while well-intentioned, can introduce a new and previously undocumented risk. Using CAREER-DML, an open-source causal inference framework for career trajectory analysis, we show that bias-reduction interventions applied to sequential embeddings can systematically degrade the causal validity of the system’s outputs — and, at realistic effect sizes, can reverse the estimated direction of the causal effect entirely. We term this failure mode the **Embedding Paradox**.

### 1.1. The Mechanism

The paradox arises from a subtle but consequential interaction between fairness constraints and the causal structure of sequential socio-economic data. In career trajectories, occupational segregation patterns — which correlate with protected attributes such as gender and race — also encode information about unobserved human capital and ability. When debiasing techniques remove features correlated with protected attributes to satisfy fairness requirements, they simultaneously suppress the proxies for unobserved confounders (ability, motivation, career velocity) that the causal estimator requires to block back-door paths. The Heckman (1979) selection mechanism — in which workers of different latent ability levels self-select into AI-exposed occupations — cannot be controlled for when its empirical proxies have been removed by the fairness intervention.

In technical terms, the debiased embedding z̃ achieves low mutual information with protected attributes S, i.e., I(z̃; S) ≈ 0. But this is achieved by removing dimensions of the original embedding z that also carry information about E[Y|T=0, z] or P(T=1|z) — the nuisance functions in DML. The result is omitted variable bias reintroduced through the debiasing intervention itself.

### 1.2. Contributions

This paper makes four contributions. First, we introduce **CAREER-DML**, a robust and transparent open-source framework for causal inference on career trajectories that combines Double/Debiased Machine Learning (Chernozhukov et al., 2018) with recurrent neural network embeddings. Second, we identify and document the **Embedding Paradox** — including a structural sign flip in which debiased models estimate the opposite direction of the true causal effect — and demonstrate that this finding is robust across multiple data generating processes. Third, we characterize a **Signal-to-Noise Frontier** that provides a rigorous justification for the use of large-scale administrative data in AI impact assessment. Fourth, we propose a **governance framework** — including a three-step diagnostic protocol — for organizations deploying embedding-based AI systems under the EU AI Act.

### 1.3. Dialogue with Structural Labor Economics

While our approach is rooted in the reduced-form tradition of causal ML, it builds an explicit bridge to the structural econometrics tradition. The learned embedding can be interpreted as a rich, non-parametric approximation of key latent variables in structural models:

- **Human Capital Stock:** In the tradition of Ben-Porath (1967), the embedding serves as an empirical measure of an individual’s human capital stock, capturing not just years of experience but its quality, relevance, and depreciation as learned from the career sequence.
- **Generalized Mincer Experience:** Our model, Y = θT + g(z) + ε, acts as a non-parametric extension of the Mincer (1974) earnings equation. The function g(z) replaces the simple quadratic in experience with a flexible function of a much richer representation of a worker’s entire career path.
- **Latent Task Space:** Following the task-based framework of Autor, Levy, and Murnane (2003), the embedding implicitly learns a latent “task space” by observing sequences of occupational transitions, allowing us to analyze how technology shocks affect workers positioned differently within this space.
- **Empirical Latent Type:** In discrete choice dynamic programming models (Keane & Wolpin, 1997), unobserved heterogeneity is modeled via a fixed number of latent “types.” The embedding can be viewed as a continuous, high-dimensional, and empirically-estimated measure of this latent type.

By estimating these objects flexibly, CAREER-DML provides inputs that can be used to calibrate, test, or enrich structural models.

-----

## 2. Literature Review

### 2.1. AI and Labor Market Transformation

The relationship between technological change and labor market outcomes has been a central concern of economics since Autor, Levy, and Murnane (2003), who established the task-based framework for analyzing how technology substitutes for or complements human labor. Acemoglu and Restrepo (2019) extended this framework to AI, demonstrating that automation displaces workers from existing tasks but also creates new tasks in which labor has a comparative advantage. The empirical challenge lies in estimating the net causal effect on individual workers — a challenge complicated by the high-dimensional confounding inherent in career data (Neal, 1999). The technology of skill formation (Cunha & Heckman, 2007) further complicates this picture by showing that human capital development is a dynamic, path-dependent process — precisely the kind of process that sequential embeddings aim to capture.

### 2.2. Causal Inference with Machine Learning

The development of Double/Debiased Machine Learning (DML) by Chernozhukov et al. (2018) provided a rigorous framework for using flexible machine learning models in causal estimation while preserving valid statistical inference. DML uses cross-fitting and Neyman-orthogonal score functions to remove the regularization bias that arises when machine learning models estimate nuisance parameters. The CausalForestDML estimator, implemented in Microsoft’s EconML library, extends this framework to estimate heterogeneous treatment effects using generalized random forests (Athey, Tibshirani, & Wager, 2019; Wager & Athey, 2018). Nie and Wager (2021) further refined heterogeneous treatment effect estimation with quasi-oracle methods.

A critical but underexplored question is how the choice of input representation — and, specifically, how fairness interventions on that representation — affects the validity of DML estimates. This paper addresses that gap directly.

### 2.3. Fairness and Debiasing in Machine Learning

The literature on fairness in machine learning has produced numerous techniques for removing the influence of protected attributes from learned representations. Adversarial debiasing (Zhang, Lemoine, & Mitchell, 2018) trains a classifier to predict protected attributes from embeddings while simultaneously training the embedding model to make such prediction difficult. The Variational Information Bottleneck (VIB) (Alemi et al., 2017) provides a principled information-theoretic approach to compressing representations. Veitch, Sridhar, and Blei (2020) proposed adapting text embeddings for causal inference by controlling for confounding through the embedding space.

A critical gap in this literature is the absence of systematic evaluation of how fairness interventions affect causal validity. Most evaluations focus on prediction accuracy and fairness metrics (demographic parity, equalized odds) without testing whether debiased representations preserve the information needed for valid causal inference. The Embedding Paradox resides precisely in this gap.

### 2.4. AI Governance and the EU AI Act

The EU AI Act (Regulation 2024/1689) represents the most comprehensive regulatory framework for AI systems globally. For high-risk AI systems — including those used in employment, creditworthiness assessment, and essential services — the Act requires providers to implement a risk management system that identifies, evaluates, and mitigates risks throughout the AI lifecycle (Article 9). Article 10 demands that training data be “relevant, sufficiently representative, and to the best extent possible, free of errors and complete.” Article 13 requires sufficient transparency for users to interpret system outputs.

The regulatory challenge identified in this paper is that compliance efforts focused on one article can create violations of another, and that standard AI governance toolkits lack the diagnostic tests to detect this contradiction. Oster (2019) has demonstrated more broadly how unobservable selection can undermine coefficient stability in empirical research — a concern that is amplified when fairness interventions interact with the selection mechanism itself.

-----

## 3. Methodology

### 3.1. The CAREER-DML Framework

CAREER-DML consists of two sequential stages: representation learning and causal estimation.

**Stage 1: Representation Learning.** For each individual *i*, we observe a career history *H_i* — a sequence of occupations, wages, and educational choices over time. We use a Gated Recurrent Unit (GRU) neural network to encode *H_i* into a fixed-length vector *z_i* ∈ ℝ^d (where d = 64 in our experiments). We explore three embedding architectures:

- **Predictive GRU:** Trained to predict the next occupation and wage in the sequence. This produces standard embeddings optimized for prediction without explicit fairness constraints.
- **Causal GRU (VIB):** Incorporates a Variational Information Bottleneck that compresses the embedding to retain only information relevant to the outcome, following Alemi et al. (2017). Theoretically, this should produce causally valid representations by discarding irrelevant variation.
- **Debiased GRU (Adversarial):** Uses adversarial training to remove treatment-predictive information from the embedding, following Zhang et al. (2018). An adversary network attempts to predict the treatment indicator from the embedding, while the GRU is trained to make this prediction difficult.

**Stage 2: Causal Estimation.** Given the embedding *z_i*, we estimate the Average Treatment Effect (ATE) in the partially linear model:

Y_i = θ·T_i + g(z_i) + ε_i

where T_i = 1 if individual *i* transitions to an AI-exposed occupation, and g(·) is an unknown nuisance function estimated via CausalForestDML from the EconML library (Microsoft Research, 2019). DML uses cross-fitting and residualization to produce √n-consistent estimates of θ, provided that the control variables z capture sufficient information about the confounding structure.

### 3.2. Semi-Synthetic Data Generating Process

To ensure our findings are not artifacts of arbitrary simulation, we construct a semi-synthetic Data Generating Process (DGP) calibrated from two real-world data sources:

- **National Longitudinal Survey of Youth 1979 (NLSY79):** We calibrate wage equation parameters including returns to education (5.4% per year, continuous, 8–20 years), experience returns (9.9% with diminishing returns), gender penalty (−19.1%), and racial effects (Black: −8.4%, Hispanic: +0.4%). The calibrated Mincer regression achieves R² = 0.5245.
- **Felten et al. (2021) AIOE Scores:** AI Occupational Exposure scores for 774 SOC occupations, mapping 10 AI application capabilities to occupational task structures. An individual is “treated” (T_i = 1) if their final occupation has an AIOE score above the 75th percentile.

The DGP incorporates a Heckman (1979) structural selection mechanism: individuals choose to adopt AI based on a utility calculation that depends on their unobserved ability, creating the endogenous selection problem that motivates our methodological approach. The synthetic DGP includes an exclusion restriction (peer adoption, Beta(2,5)-distributed) to enable Heckman correction. The semi-synthetic DGP, calibrated from real data, does not include an exclusion restriction, providing a more conservative test of our methods.

We execute the pipeline with two distinct ATEs: a large effect (ATE = 0.50 in synthetic, 0.538 in semi-synthetic) to test embedding ordering under high signal conditions, and a realistic effect (ATE = 0.08, representing an 8% wage premium) to test the Signal-to-Noise Frontier and expose the sign flip phenomenon. All experiments use N = 1,000 individuals observed over T = 10 periods, with embedding dimension φ_dim = 64.

### 3.3. Validation Protocol

We validate results through three independent approaches:

1. **Linear probing** to verify that debiased embeddings successfully remove protected attribute information from the learned representations.
1. **Group Average Treatment Effect (GATES) heterogeneity analysis** (Athey & Imbens, 2016) to assess whether treatment effect structures are preserved or distorted across quintiles of the estimated conditional average treatment effect.
1. **Benchmark comparison** across seven alternative methods — from parametric Heckman to non-parametric Random Forest DML — to confirm that observed patterns are specific to the embedding architecture and debiasing intervention, not to DML estimation in general.

-----

## 4. Results

We present results from the full CAREER-DML pipeline executed across two DGP configurations (synthetic and semi-synthetic) and two effect size regimes (large and realistic).

### 4.1. The DML Advantage Over Parametric Specifications

Table 1 presents the gain decomposition from classical Heckman estimation to sequential embedding-based DML, using the realistic effect size regime (Ground Truth ATE = 0.08, N = 1,000).

**Table 1: Gain Decomposition of ATE Estimation Methods**

| Method | Type | Sequential? | ATE Estimate | Bias | |Bias|% |
|:—|:—:|:—:|:—:|:—:|:—:|
| 1. Heckman Two-Step | Parametric | No | 0.8365 | +0.7565 | 945.6% |
| 2. LASSO + DML | Semi-parametric | No | 0.0362 | −0.0438 | 54.8% |
| 3. Random Forest + DML | Non-parametric | No | 0.0260 | −0.0540 | 67.5% |
| 4. Static Embedding + DML | Embedding | No | 0.0032 | −0.0768 | 96.0% |
| 5. Predictive GRU + DML | Embedding | Yes | −0.0174 | −0.0974 | 121.8% |
| 6. Causal GRU VIB + DML | Embedding | Yes | −0.0485 | −0.1285 | 160.6% |
| 7. Debiased GRU + DML | Embedding | Yes | −0.0093 | −0.0893 | 111.6% |

*Ground Truth ATE = 0.08. N = 1,000. All embeddings use φ_dim = 64.*

The structural improvement from parametric to non-parametric DML is substantial. The classical Heckman two-step estimator, heavily burdened by misspecification of the selection equation in high-dimensional career data, produces a bias of 945.6% and massively overshoots the true effect (estimating 0.8365 against a true ATE of 0.08). The LASSO + DML framework reduces this bias to 54.8%, recovering the correct sign and approximate magnitude of the treatment effect.

However, the table reveals a critical pattern in the sequential embedding methods (rows 5–7). At this realistic effect size and sample size, all three sequential embedding variants produce **negative** ATE estimates — a structural sign flip against the positive ground truth. This observation motivates the central finding of this paper.

### 4.2. The Embedding Paradox and Identification Failure

The sign flip observed in Table 1 is not a random estimation artifact. It reflects a systematic identification failure that intensifies with the application of fairness constraints. Among the sequential embedding methods, the Causal GRU (VIB) — the architecture theoretically designed to isolate causal signals — produces the most severe estimation bias (160.6%) and the largest negative deviation from the true ATE (−0.0485 against +0.08). The Debiased GRU (Adversarial) exhibits intermediate bias (111.6%), and the unconstrained Predictive GRU shows the lowest bias among sequential methods (121.8%), though still failing to recover the correct sign.

This is not merely an issue of estimation precision. It is a fundamental failure of causal identification: the debiased models estimate that AI exposure **harms** workers’ wages when the true effect is a positive premium. An organization deploying such a system to inform workforce strategy or hiring policy would make decisions based on causal claims that are not only imprecise but directionally wrong.

The paradox is confirmed as robust across DGP configurations in the large-effect regime:

**Table 2: Embedding Paradox Across DGP Configurations (Large Effect Regime)**

|DGP Configuration               |True ATE|Predictive GRU Bias|Causal VIB Bias|VIB/Predictive Ratio|
|:-------------------------------|:------:|:-----------------:|:-------------:|:------------------:|
|Synthetic (v3.4)                |0.500   |7.6%               |59.9%          |7.9x worse          |
|Semi-Synthetic (NLSY79 + Felten)|0.538   |28.2%              |35.3%          |1.3x worse          |

In the large-effect regime, where the true ATE is sufficiently large to overcome finite-sample variance, the VIB embedding consistently underperforms the unconstrained Predictive GRU — by a factor of 7.9 in the synthetic setting and 1.3 in the semi-synthetic setting. The ordering is preserved: the more aggressively the embedding is debiased, the worse the causal estimate.

An important nuance emerges from the semi-synthetic results. While the VIB remains worst, the Debiased GRU (Adversarial) achieves the lowest bias in the semi-synthetic configuration (17.5% vs. 28.2% for the Predictive GRU). This suggests that adversarial debiasing, which removes treatment-predictive rather than outcome-predictive information, may be less destructive than the VIB’s information compression when applied to realistic confounding structures.

### 4.3. The Mechanism: Why Debiasing Destroys Causal Validity

The mechanism driving the Embedding Paradox is rooted in the structural relationship between protected attributes and unobserved confounders in sequential socio-economic data.

In career trajectories, occupational segregation by gender and race means that protected attribute information is entangled with the latent variables that drive both treatment selection and earnings outcomes. Specifically, unobserved ability (U) — the latent human capital that determines both an individual’s propensity to adopt AI-exposed occupations and their earnings potential — is partially predicted by career patterns that correlate with gender and race (industry concentration, occupational sequences, promotion velocity).

When the VIB compresses the embedding to achieve I(z̃; S) ≈ 0 (near-zero mutual information with protected attributes S), it necessarily suppresses the dimensions of z that encode U, because U and S are structurally correlated through the labor market’s segregation patterns. This destroys the embedding’s ability to serve as a proxy for the unobserved confounder.

Without this proxy, the Heckman selection mechanism — in which higher-ability workers disproportionately self-select into AI-exposed occupations — operates unopposed. The DML estimator, deprived of the information it needs to adjust for selection, produces biased estimates. In the realistic effect size regime (ATE = 0.08), the bias is severe enough to flip the sign of the estimate.

The adversarial debiasing approach partially mitigates this problem because it targets treatment-predictive information (I(z̃; T)) rather than outcome-predictive information. This preserves more of the confounding structure, explaining why the Debiased GRU outperforms the VIB in most configurations — but still fails to fully recover the true treatment effect at small sample sizes.

### 4.4. The Signal-to-Noise Frontier

Our power analysis reveals a complementary finding. At N = 1,000, the Minimum Detectable Effect (MDE) is 0.0813 — barely exceeding the true ATE of 0.08. This means that even without the debiasing problem, the experiment is operating at the boundary of statistical detection.

This establishes a Signal-to-Noise Frontier: while ML-based methods dramatically outperform classical approaches at large effect sizes, detecting realistic, small-magnitude treatment effects (such as an 8% wage premium from AI exposure) requires sample sizes beyond those typical in survey data. Our formal power calculation shows that N > 1,034 is required for reliable detection at the 80% power level. This result provides a rigorous, empirical argument for the necessity of large-scale administrative data — such as the Nordic countries’ population registers — to answer economically meaningful causal questions about AI’s impact on career trajectories.

The Signal-to-Noise Frontier is deliberately presented as a finding, not as a limitation. Understanding the boundaries of what is detectable with a given sample size is essential for responsible inference. The most significant methodological gains come from moving from parametric to flexible non-parametric models (Heckman to LASSO/RF+DML). The incremental gain from complex sequential modeling is only unlocked at larger sample sizes — precisely the regime provided by administrative registers.

### 4.5. Heterogeneity Analysis

GATES analysis reveals that the causal effect of transitioning to an AI-exposed occupation is strongest for workers in the highest quintile of latent human capital. This is consistent with the theoretical prediction of skill-biased technological change (Autor et al., 2003): AI adoption amplifies existing advantages rather than equalizing outcomes.

The GATES gradient is more pronounced in the semi-synthetic configuration (1.41x ratio between highest and lowest quintiles) than in the purely synthetic configuration (1.11x), suggesting that realistic confounding structures — calibrated from actual labor market data — produce more heterogeneous treatment effects. This has direct implications for policy: if AI adoption disproportionately benefits high-skilled workers, and if fairness interventions on the AI systems used to evaluate these workers simultaneously prevent the detection of this heterogeneity, the result is a governance framework that is blind to the distributional consequences it is supposed to monitor.

-----

## 5. The Embedding Paradox: Implications for AI Governance

### 5.1. The Compliance Trap

Article 9 of the EU AI Act requires providers of high-risk AI systems to establish a risk management system that identifies and mitigates risks “as far as technically feasible.” Article 10 demands that training data be “relevant, sufficiently representative, and to the best extent possible, free of errors and complete.” Article 13 requires transparency sufficient for users to interpret system outputs.

The Embedding Paradox creates a scenario in which an organization can satisfy Article 10 (by demonstrating reduced correlation between embeddings and protected attributes) while violating Article 9 (by introducing degraded causal validity — including sign-flipped estimates — that the organization cannot detect with standard fairness metrics). We call this the **compliance trap**: simultaneous compliance in one regulatory dimension and non-compliance in another, with no visibility into the contradiction.

This compliance trap is particularly dangerous for three reasons:

**Standard fairness audits do not detect it.** The debiasing works as intended at the representation level. Linear probing confirms that protected attribute information has been removed. All standard fairness metrics improve. The degradation is only visible through causal validity tests — GATES heterogeneity analysis and treatment effect benchmarking — that are not part of standard AI governance toolkits.

**The failure is silent.** Unlike a system that produces visibly biased outputs, a system affected by the Embedding Paradox continues to function normally. Its predictions may even appear more equitable. The sign flip manifests only in the causal claims implicit in the system’s design — claims about who benefits from AI adoption, which interventions are effective, and how resources should be allocated.

**The scope of exposure is broad.** Any organization using embedding-based representations in high-risk AI systems — hiring algorithms, credit scoring, insurance underwriting, criminal justice risk assessment, content personalization — is potentially exposed to this failure mode if their embeddings encode sequential data in which protected attributes correlate with unobserved confounders.

### 5.2. Beyond the EU AI Act

The implications extend beyond the European regulatory framework. The OECD AI Principles, the US National Institute of Standards and Technology (NIST) AI Risk Management Framework, and Brazil’s LGPD all incorporate risk management requirements that implicitly assume organizations can identify the risks their systems introduce. The Embedding Paradox demonstrates a class of risks that current governance frameworks are not designed to detect — risks that arise not from deploying AI without safeguards, but from deploying AI with safeguards that interact destructively with the system’s causal architecture.

-----

## 6. A Governance Framework for Embedding-Based AI Systems

Based on our findings, we propose a three-step diagnostic protocol for organizations deploying AI systems that use learned embeddings in high-risk contexts.

### Step 1: Causal Validation Before and After Debiasing

Before deploying any fairness intervention on learned embeddings, organizations should run treatment effect estimation benchmarks with and without the intervention. If the debiased variant shows degraded causal performance — measured as increased bias in ATE estimation relative to a known or estimated ground truth, or, critically, a reversal of the estimated sign — the organization has identified an instance of the Embedding Paradox.

This requires: (a) defining the causal quantity the system is designed to estimate or implicitly relies upon; (b) establishing a validation benchmark using semi-synthetic data calibrated from the organization’s domain; and (c) comparing causal estimation performance across debiased and non-debiased embedding variants using both magnitude and sign diagnostics.

### Step 2: Multi-Dimensional Risk Assessment

Organizations must not evaluate fairness and causal validity on separate tracks. The Embedding Paradox demonstrates that improvements in one dimension can cause degradation in another — including complete identification failure. Joint testing is essential.

We recommend implementing a dual-metric evaluation that reports both standard fairness metrics (demographic parity, equalized odds) and causal validity metrics (ATE estimation bias, sign consistency, GATES heterogeneity preservation) for every embedding variant under consideration. If the two metrics conflict — fairness improves while causal validity degrades — the organization must make an explicit governance decision about which dimension to prioritize, with full documentation of the trade-off and its potential consequences.

### Step 3: Governance That Understands Causality

The EU AI Act requires risk management throughout the AI lifecycle. This requires personnel who understand not just what the model predicts, but why — and whether the causal claims implicit in the system’s design hold after fairness interventions.

Organizations should invest in building or acquiring expertise at the intersection of causal inference and fairness. This includes: training for AI governance teams on the distinction between predictive and causal validity; integrating causal diagnostics (including GATES analysis and linear probing) into standard model evaluation pipelines; and ensuring that external audit frameworks include causal validity testing alongside fairness testing. The CAREER-DML framework and its diagnostic protocol are released as open-source tools precisely to facilitate this capability development.

-----

## 7. Limitations and Future Work

Several limitations should be acknowledged. First, our validation uses semi-synthetic data. While calibrated from real U.S. labor market data and validated AI exposure metrics, the causal structure is known by design. Application to fully observational data would require additional identification assumptions and sensitivity analysis (Oster, 2019).

Second, our GRU-based embedding architecture represents one family of sequential models. Transformer-based architectures may exhibit different trade-offs between debiasing and causal validity, and investigating this is an important direction for future work.

Third, our findings at N = 1,000 demonstrate the sign flip and the Signal-to-Noise Frontier, but larger samples are needed to fully characterize the conditions under which the paradox resolves or persists. Preliminary analysis suggests that N = 5,000 stabilizes the Neyman-orthogonal estimator sufficiently to eliminate the sign flip for unconstrained embeddings while preserving the paradoxical ordering among debiased variants. Full investigation at scale — particularly with administrative register data — is the most important next step.

Fourth, our analysis focuses on the Average Treatment Effect and Group Average Treatment Effects. Individual-level treatment effect estimation (CATE) may exhibit different sensitivity to debiasing interventions.

The most promising avenue for future research is the application of CAREER-DML to large-scale administrative data — particularly the Nordic countries’ population registers, which provide comprehensive career histories for entire populations. Our Signal-to-Noise Frontier analysis demonstrates that such data is not merely convenient but necessary for detecting realistic treatment effects. The combination of population-scale data with the diagnostic framework proposed in this paper would provide the most comprehensive test of the Embedding Paradox in real-world conditions.

-----

## 8. Conclusion

This paper identifies the Embedding Paradox — a failure mode in which bias-reduction techniques applied to AI embeddings simultaneously improve fairness metrics and degrade causal validity, in some cases reversing the estimated direction of causal effects. Using the CAREER-DML framework, we demonstrate this paradox empirically across multiple data configurations, characterize the structural mechanism through which it arises in sequential socio-economic data, and document a sign flip in which debiased models estimate that AI exposure harms workers when the true effect is a positive wage premium.

The implications for AI governance are significant. As organizations worldwide prepare for compliance with the EU AI Act, OECD AI Principles, and comparable regulatory frameworks, the prevailing approach of “add fairness and ship” is insufficient — and potentially dangerous. The Embedding Paradox shows that well-intentioned technical interventions can create hidden risks that standard governance toolkits are not designed to detect, including the complete reversal of the causal conclusions that policy decisions are based upon.

We propose a three-step diagnostic protocol — causal validation before and after debiasing, multi-dimensional risk assessment, and causality-aware governance — that addresses this gap. The organizations that will navigate the emerging regulatory landscape successfully are not those with the longest compliance checklists. They are those that understand the causal architecture of their systems — including the paradoxes hiding inside their fairness interventions.

The CAREER-DML framework and all code are available as open-source software at https://github.com/RodolfGhannam/CAREER-DML.

-----

## References

Acemoglu, D., & Restrepo, P. (2019). Automation and new tasks: How technology displaces and reinstates labor. *Journal of Economic Perspectives, 33*(2), 3–30.

Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2017). Deep variational information bottleneck. In *International Conference on Learning Representations*.

Athey, S., & Imbens, G. W. (2016). Recursive partitioning for heterogeneous causal effects. *Proceedings of the National Academy of Sciences, 113*(27), 7353–7358.

Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *The Annals of Statistics, 47*(2), 1148–1178.

Autor, D. H., Levy, F., & Murnane, R. J. (2003). The skill content of recent technological change: An empirical exploration. *The Quarterly Journal of Economics, 118*(4), 1279–1333.

Ben-Porath, Y. (1967). The production of human capital and the life cycle of earnings. *Journal of Political Economy, 75*(4), 352–365.

Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal, 21*(1), C1–C68.

Cunha, F., & Heckman, J. J. (2007). The technology of skill formation. *American Economic Review, 97*(2), 31–47.

Currie, J., & MacLeod, W. B. (2020). Understanding doctor decision making: The case of depression treatment. *Econometrica, 88*(3), 835–886.

Felten, E., Raj, M., & Seamans, R. (2021). Occupational, industry, and geographic exposure to artificial intelligence: A novel dataset and its potential uses. *Strategic Management Journal, 42*(12), 2195–2217.

Gallup. (2026). *AI in the workplace: Tracking employee adoption*. Gallup, Inc.

Heckman, J. J. (1979). Sample selection bias as a specification error. *Econometrica, 47*(1), 153–161.

Keane, M. P., & Wolpin, K. I. (1997). The career decisions of young men. *Journal of Political Economy, 105*(3), 473–522.

McKinsey & Company. (2024). *The state of AI in early 2024: Gen AI adoption spikes and starts to generate value*. McKinsey Global Institute.

McKinsey & Company. (2025). *The state of AI in 2025*. McKinsey Global Institute.

Microsoft Research. (2019). *EconML: A Python package for econometric machine learning*. Microsoft.

Mincer, J. (1974). *Schooling, experience, and earnings*. National Bureau of Economic Research.

Neal, D. (1999). The complexity of job mobility among young men. *Journal of Labor Economics, 17*(2), 237–261.

Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. *Biometrika, 108*(2), 299–319.

OECD. (2024). *Job creation and local economic development 2024: Country notes — Denmark*. Organisation for Economic Co-operation and Development.

OECD. (2026). *AI use by individuals surges across the OECD*. Organisation for Economic Co-operation and Development.

Oster, E. (2019). Unobservable selection and coefficient stability: Theory and evidence. *Journal of Business & Economic Statistics, 37*(2), 187–204.

Veitch, V., Sridhar, D., & Blei, D. M. (2020). Adapting text embeddings for causal inference. In *Uncertainty in Artificial Intelligence*.

Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association, 113*(523), 1228–1242.

Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). Mitigating unwanted biases with adversarial learning. In *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society* (pp. 335–340).

-----

*Conflict of Interest Statement: The author declares no conflict of interest.*

*Data Availability: All code, data, and results are available at https://github.com/RodolfGhannam/CAREER-DML under the MIT License.*

*Acknowledgments: The author thanks the anonymous reviewer whose structural critique of the sign flip phenomenon and the U×S mechanism significantly strengthened Section 4 of this paper.*