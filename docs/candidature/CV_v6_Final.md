# Rodolf Mikel Ghannam Neto

<rodolf@cical.com.br> • +55 62 99399-2112 • [LinkedIn](https://www.linkedin.com/in/rodolf-ghannam-535343134/) • [GitHub](https://github.com/RodolfGhannam/CAREER-DML)

---

## Profile

Practitioner-researcher who independently built and validated **CAREER-DML**, a causal inference framework for labor market analysis that reduces estimation bias by an order of magnitude over classical methods. This work was motivated by 15+ years of executive leadership in corporate strategy and finance, during which I observed significant heterogeneity in employee development needs that standard analyses could not capture. While directing operations for a diversified conglomerate (recognized with General Motors' highest dealership distinction for four consecutive years, 2018–2021), I designed a data-driven customer service methodology whose analytics revealed that employees with similar backgrounds diverged sharply—prompting a shift from group training to personalized coaching. This field-based discovery of heterogeneous treatment effects is the direct inspiration for my PhD research. I now seek the academic rigor of the PhD program at CBS to apply this framework to population-scale administrative data, contributing to the **'Digital Transformations'** research area.

---

## Education

| Year | Degree | Institution | Details |
| :--- | :--- | :--- | :--- |
| 2016 | **Certificate in Policy & Strategy (CEPE)** | Escola Superior de Guerra (ESG) / ADESG, Ministry of Defence, Brazil | 204 hours of studies in national policy and strategy. Brazil's premier institution for strategic studies. ADESG is aligned with NATO's defence education framework, placing this credential on par with European defence academies. |
| 2018 | **Technologist Degree in Management Processes** | Faculdade Educacional da Lapa (FAEL), Brazil | Higher education degree in managerial processes (Tecnólogo em Processos Gerenciais). |
| 2020 | **MBA in Management, Entrepreneurship & Marketing** | Pontifical Catholic University of Rio Grande do Sul (PUCRS), Brazil | Grade: 8.0/10. Capstone: BSC application in public management. 363 hours. |
| 2022 | **MBA in Marketing: Digital Media** | Fundação Getulio Vargas (FGV), Brazil | **Triple Crown accredited (AACSB, EQUIS, AMBA)** — one of fewer than 120 business schools worldwide holding all three accreditations, placing FGV on par with CBS, INSEAD, and London Business School. Grade: 8.23/10. 480 hours. |
| 2024 | **Bachelor of Business Administration** | Centro Universitário FAEL (UNIFAEL), Brazil | Full undergraduate degree in management (Bacharel em Administração). |
| 2025 | **MIT xPRO Professional Certificate in AI** | Massachusetts Institute of Technology (MIT) | Achieved 100% score on the final capstone project. |
| 2025 (in progress) | **MSc in Administration — Digital Business** | American Global Tech University (AGTU), Orlando, FL, USA | Focus on digital business models and data-driven strategy. |

---

## Regulatory & Financial Certifications

| Certification | Issuing Body |
| :--- | :--- |
| **CPA-10** (Financial Products Certification) | ANBIMA (Brazilian Financial and Capital Markets Association) |
| **PLD** (Anti-Money Laundering Compliance) | FTCI / Cical Consórcio |

---

## Research & Applied AI Projects

### PhD Candidacy Project: CAREER-DML

Developed **CAREER-DML**, a novel framework for causal inference on career trajectories, designed to bridge the **two cultures of modern econometrics**—causal ML and structural modeling. The framework was validated on a semi-synthetic DGP calibrated with real-world US labor market data (NLSY79 and Felten et al. 2021 AIOE scores).

**Key Contributions:**

1. **Sequential Embedding Ordering Phenomenon:** Documented a robust empirical finding, observed across three distinct scenarios, that challenges the direct application of some causal representation learning methods to this domain.
2. **Signal-to-Noise Frontier:** Characterized the sample size requirements for detecting realistic effect sizes (N > 1,034 for ATE = 0.08), providing a rigorous justification for the use of large-scale administrative data.
3. **Structural Bridge:** Built an explicit conceptual bridge to the structural econometrics tradition, showing how the learned embeddings serve as non-parametric analogs of classical latent variables (Ben-Porath, Mincer, Autor, Keane & Wolpin).

**Validation:** The framework outperforms all seven benchmarked methods—from classical Heckman to modern LASSO and Random Forest approaches—**reducing estimation bias by an order of magnitude (from 945% to 7.6%)**. Linear probing experiments confirm that the learned embeddings encode interpretable economic structure (occupation, industry, tenure) without being explicitly trained on these labels. A formal GATES heterogeneity test confirms statistically significant skill-biased treatment effects. Open-source implementation available on [GitHub](https://github.com/RodolfGhannam/CAREER-DML).

**Next Step:** Apply the validated framework to the Danish IDA registers (N > 1M individuals, T > 30 years)—crossing the Signal-to-Noise Frontier with population-scale administrative data to produce the first causal estimates of how AI adoption reshapes individual career trajectories across Denmark.

### MIT Capstone Project: AI Triage System

Designed an AI-powered triage support system for hospital emergency rooms. A pilot study (N=1,200) demonstrated the potential to reduce patient wait times by 40% and deliver a projected 28:1 ROI. Achieved a 100% score on the final capstone.

---

## Professional Experience

**Corporate Director** | Grupo CICAL, Goiânia, Brazil (2017 – Present)

Directed operations across five business divisions of a diversified conglomerate, each of which taught me something essential about managing complexity under uncertainty:

- **Automotive:** Directed marketing for Nissan, Honda, Chevrolet, and Zeekr dealerships—coordinating brand positioning, campaign strategy, and performance analytics across distinct market segments.
- **Real Estate:** Led marketing for real estate developments, from launch strategy to sales conversion, managing multi-channel campaigns for residential and commercial projects.
- **Insurance:** Represented the group in direct negotiations with major insurers (HDI, Yellow/Liberty, Porto Seguro, Tokio Marine), structuring partnerships and managing portfolio risk.
- **Consortium (Cical Consórcio):** Sole registered representative with the Central Bank of Brazil—personally responsible for the consortium's registration, compliance, and quarterly audits with a consistent compliance record. I am grateful for everything this responsibility taught me about regulatory discipline and institutional accountability.
- **Car Rental (Locadora):** Directed the car rental division, overseeing fleet management, pricing strategy, and operational logistics.

As a practitioner-researcher, designed and implemented a proprietary **5-stage data-driven customer service methodology** (Research, Model, Implement, Train, Individualize). The system's AI-powered analytics revealed significant **heterogeneous development needs** among employees, prompting a strategic shift from group training to personalized coaching. This experience of discovering and acting on heterogeneity in the field is the direct inspiration for my PhD research.

**Marketing Director** | Grupo CICAL, Goiânia, Brazil (2015 – 2017)

Led strategic marketing operations, including market analysis, brand strategy, and performance analytics across all business units.

**Founder & CEO** | Real Alimentos Ltda, Goiânia, Brazil (2002 – 2014)

Founded and scaled a food distribution company from the ground up, including brand creation and packaging design. Secured national supply contracts with Brazil's largest retail chains, including Walmart, Carrefour, Sam's Club, and Pão de Açúcar.

---

## Technical Skills

- **Causal Inference:** Double/Debiased ML (DML), Causal Forests, GATES, Heckman Models, A/B Testing, EconML.
- **Machine Learning:** PyTorch, Scikit-learn, LightGBM, GRU Neural Networks, Adversarial Training, Variational Information Bottleneck (VIB).
- **Data Science & Econometrics:** Python (Pandas, NumPy), R, SQL, Statsmodels, BI Dashboards.
- **Business & Domain:** P&L Management, Regulatory Compliance (Central Bank of Brazil), Brand Strategy.
- **Languages:** Portuguese (Native), English (Fluent), Spanish (Intermediate).

---

## Selected Writing

- "The AI Job Market: A Double-Edged Sword" — *Data-Driven Investor*, 2024
- "Heckman's Two-Step: A Bridge Between Eras" — *Towards Data Science*, 2024

---

## References

| Name | Role | Contact |
| :--- | :--- | :--- |
| **Prof. Ricardo Barros Villaça** | Professor and Coordinator, MIT xPRO Program | +55 21 99996-4884 |
| **Walkiria Luna Cecilio** | Founder of Grupo CICAL and Member of the Shareholder Council | regolunadesign@hotmail.com, +55 62 99910-8272 |
| **Ataualpa Veloso Roriz** | Development Partner in AI Solutions | +55 62 99991-8424 |
