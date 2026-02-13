# Paper Texts — Heckman Anchor (v3.4)

**Instruction**: These texts are drafts for the corresponding paper sections. Adapt as needed.

---

## Text 1: Methodology Section (after describing the DGP)

> The design of our Data Generating Process (DGP) is grounded in the foundations of labour econometrics. Specifically, our latent variable `ability`, which influences both treatment selection ($T$) and the outcome ($Y$), constitutes a direct implementation of the **sample selection bias** formalised by **Heckman (1979)** [1]. The DGP includes a proper exclusion restriction (`peer_adoption`) that affects treatment selection but not the outcome, satisfying the identifying assumption of the classical two-step estimator. By doing so, we create a controlled setting where the Heckman benchmark receives a fair test.
>
> Additionally, the way `ability` modulates the treatment effect (CATE) and career transitions draws on the work on **human capital formation and skill dynamics** by **Cunha and Heckman (2007)** [2]. Our *career embeddings* can therefore be seen as a nonparametric attempt to capture the manifestation of this latent human capital along an individual's professional trajectory. We evaluate four embedding variants — Predictive, Causal VIB (Veitch et al., 2020), Adversarial Debiasing, and Two-Stage Causal — to assess which approach to adapting embeddings for causal inference is most effective in this setting.

---

## Text 2: Results Section (after the Variance Decomposition table)

> The `common_support_penalty` component of the variance decomposition can be interpreted as a **quantitative measure of the severity of Heckman's selection bias** in our data. A high value indicates weak overlap between the treatment and control groups due to confounding by `ability`, making causal inference particularly challenging. This finding underscores the necessity of methods that explicitly address the overlap problem, whether through adversarial debiasing or through the DML cross-fitting procedure that partitions the sample to avoid overfitting bias.

---

## Text 3: Results Section (after the GATES table)

> The GATES analysis reveals treatment effect heterogeneity consistent with the theory of **skill-capital complementarity (Cunha & Heckman, 2007)** [2]. The return to AI exposure ranges from 0.5081 (Q1, lowest latent human capital) to 0.5661 (Q5, highest latent human capital), a 1.11x difference. A formal heterogeneity test rejects the null hypothesis H₀: ATE(Q1) = ATE(Q5) with t = 62.27 and p < 10⁻²⁰⁰ (Cohen's d = 6.25). This monotonically increasing pattern is consistent with the hypothesis that individuals with greater latent human capital — proxied by higher `ability` and `education` — benefit disproportionately from AI exposure, in line with **skill-biased technological change (SBTC)** and with Heckman's model of human capital formation, where initial endowments amplify the returns to subsequent investments.

---

## Text 4: Robustness Section

> A potential concern is that our selection mechanism follows a reduced-form specification. To address this, we developed an alternative DGP where treatment selection results from a **rational decision model**: agents choose to adopt AI if the expected utility gain (higher wages minus adaptation costs) is positive. Adaptation costs are inversely proportional to `ability`, creating a structural form of endogenous selection consistent with Heckman's framework. We find that our main results hold under this specification change: the Predictive GRU achieves an ATE of 0.5920 under mechanical selection versus 0.6689 under structural selection, with a bias difference of 0.077. This is below the pre-specified threshold of 0.1, suggesting that the DML pipeline is not sensitive to whether selection arises from a mechanical rule or from optimising behaviour.

---

## Text 5: Limitations and Future Work Section

> While our DGP incorporates key features of Heckman's selection framework — endogenous treatment assignment driven by latent ability, with a proper exclusion restriction — it remains a simplified model using synthetic data. The 93.0% bias reduction over the Heckman two-step estimator should be interpreted in context: it reflects the specific DGP design (high-dimensional confounding that a scalar Inverse Mills Ratio cannot capture), not a general claim about the relative merits of parametric vs. nonparametric selection correction. A natural extension would be to embed our career representations within a **full dynamic structural model** where agents solve an intertemporal optimisation problem, choosing their career path to maximise lifetime utility subject to human capital accumulation constraints (Keane & Wolpin, 1997; Heckman & Navarro, 2007) [3] [4]. We leave this integration for future work, alongside application to Danish administrative data (IDA/Statistics Denmark) and a Monte Carlo study across multiple seeds.

---

## References

[1]: Heckman, J. J. (1979). Sample Selection Bias as a Specification Error. *Econometrica*, 47(1), 153-161.

[2]: Cunha, F., & Heckman, J. J. (2007). The Technology of Skill Formation. *American Economic Review*, 97(2), 31-47.

[3]: Keane, M. P., & Wolpin, K. I. (1997). The Career Decisions of Young Men. *Journal of Political Economy*, 105(3), 473-522.

[4]: Heckman, J. J., & Navarro, S. (2007). Dynamic Discrete Choice and Dynamic Treatment Effects. *Journal of Econometrics*, 136(2), 341-396.
