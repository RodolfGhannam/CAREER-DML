# Textos Prontos para o Paper — Ancoragem Heckman (Nível 1)

**Instrução**: Copie estes textos diretamente para as secções indicadas do seu paper.

---

## Texto 1: Para a Secção de Metodologia (após descrever o DGP)

> The design of our Data Generating Process (DGP) is intentionally grounded in the foundations of labor econometrics. Specifically, our latent variable `ability`, which influences both treatment selection ($T$) and the outcome ($Y$), constitutes a direct implementation of the **sample selection bias** formalized by **Heckman (1979)** [1]. By doing so, we create a realistic challenge where simple correlation methods, such as predictive embeddings, are expected to produce biased estimates — precisely because they absorb the confounding path $\text{ability} \to T \leftarrow U \to Y$.
>
> Additionally, the way `ability` modulates the treatment effect (CATE) and career transitions draws on the work on **human capital formation and skill dynamics** by **Cunha and Heckman (2007)** [2]. Our *career embeddings* can therefore be seen as a non-parametric attempt to capture the manifestation of this latent human capital along an individual's professional trajectory. The Debiased GRU variant, trained with adversarial objectives, aims to preserve the causal information in these representations while removing the confounding signal — effectively achieving what Heckman's Inverse Mills Ratio does parametrically, but in a richer, high-dimensional feature space.

---

## Texto 2: Para a Secção de Resultados (após a tabela de Variance Decomposition)

> The `common_support_penalty` component of the variance decomposition can be interpreted as a **quantitative measure of the severity of Heckman's selection bias** in our data. A high value indicates weak overlap between the treatment and control groups due to confounding by `ability`, making causal inference particularly challenging. In our setting, this penalty accounts for [X]% of the total variance, confirming that the selection mechanism — whether mechanical or structural — creates substantial separation between groups. This finding underscores the necessity of adversarial debiasing methods that explicitly address the overlap problem.

---

## Texto 3: Para a Secção de Resultados (após a tabela de GATES)

> The GATES analysis reveals strong treatment effect heterogeneity, consistent with the theory of **skill-capital complementarity (Cunha & Heckman, 2007)** [2]. The return to AI exposure ranges from [Q1 value] (Q1, lowest latent human capital) to [Q5 value] (Q5, highest latent human capital), a [X]x difference. This monotonically increasing pattern is consistent with the hypothesis that individuals with greater latent human capital — proxied by higher `ability` and `education` — benefit disproportionately from AI exposure, in line with **skill-biased technological change (SBTC)** and with Heckman's model of human capital formation, where initial endowments amplify the returns to subsequent investments.

---

## Texto 4: Para a Secção de Robustez (novo parágrafo)

> A potential concern is that our selection mechanism follows a reduced-form specification. To address this, we developed an alternative DGP (v3.2) where treatment selection results from a **rational decision model**: agents choose to adopt AI if the expected utility gain (higher wages minus adaptation costs) is positive. Adaptation costs are inversely proportional to `ability`, creating a structural form of endogenous selection consistent with Heckman's framework. We find that our main results are robust to this specification change: the Debiased GRU achieves an ATE of [value] under structural selection versus [value] under mechanical selection, with a bias difference of only [value]. This suggests that adversarial debiasing is not sensitive to whether selection arises from a mechanical rule or from optimizing behaviour.

---

## Texto 5: Para a Secção de Limitações e Pesquisa Futura

> While our DGP incorporates key features of Heckman's selection framework — endogenous treatment assignment driven by latent ability — it remains a simplified model. A natural extension would be to embed our debiased career representations within a **full dynamic structural model** where agents solve an intertemporal optimization problem, choosing their career path to maximize lifetime utility subject to human capital accumulation constraints (Keane & Wolpin, 1997; Heckman & Navarro, 2007) [3] [4]. Such a model would allow for richer counterfactual analyses, such as evaluating the welfare effects of retraining subsidies or predicting how changes in AI adoption costs would reshape career trajectories. We leave this integration for future work.

---

## Referências

[1]: Heckman, J. J. (1979). Sample Selection Bias as a Specification Error. *Econometrica*, 47(1), 153-161.

[2]: Cunha, F., & Heckman, J. J. (2007). The Technology of Skill Formation. *American Economic Review*, 97(2), 31-47.

[3]: Keane, M. P., & Wolpin, K. I. (1997). The Career Decisions of Young Men. *Journal of Political Economy*, 105(3), 473-522.

[4]: Heckman, J. J., & Navarro, S. (2007). Dynamic Discrete Choice and Dynamic Treatment Effects. *Journal of Econometrics*, 136(2), 341-396.
