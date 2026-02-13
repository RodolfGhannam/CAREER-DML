# BLUEPRINT v3.2 — CAREER-DML (HECKMAN STRUCTURAL)

**Data:** 12 de Fevereiro de 2026
**Autor:** Rodolf Mikel Ghannam Neto
**Status:** FINAL (para submissão CBS)

## 1. Visão Geral do Projeto

O **CAREER-DML** é um framework de pesquisa para estimar o impacto causal da adoção de IA na trajetória salarial de trabalhadores, usando Double/Debiased Machine Learning (DML) sobre representações de carreira (career embeddings). A versão 3.2 integra os 3 níveis de profundidade do trabalho de James Heckman para garantir robustez econométrica e relevância teórica.

## 2. Arquitetura do MVP v3.2 (Código)

O MVP v3.2 é um pipeline Python executável (`main.py`) que demonstra o framework completo em 6 etapas, usando dados sintéticos gerados por um Processo Gerador de Dados (DGP) que incorpora o viés de seleção de Heckman.

### Componentes Core (`src/`):

1.  **`dgp.py` (Nível 1 Heckman):**
    -   **Seleção Estrutural:** Agentes sintéticos decidem adotar IA com base num cálculo de utilidade esperada (salário futuro vs. custo de adaptação), criando um viés de seleção endógeno e estrutural.
    -   **Capital Humano (Cunha & Heckman):** A `ability` latente influencia o custo de adaptação e o retorno da IA, modelando a complementaridade capital-competência.

2.  **`embeddings.py`:**
    -   **Predictive GRU:** Baseline que maximiza a predição de Y (falha causalmente).
    -   **Causal GRU (VIB):** Implementa o Variational Information Bottleneck de Veitch et al. (2020).
    -   **Debiased GRU (Adversarial):** **Variante vencedora.** Usa uma rede adversária para purgar a informação do tratamento (T) do embedding, tornando-o causalmente válido.

3.  **`dml.py`:**
    -   **CausalForestDML:** Usa o estimador de Athey & Wager para capturar heterogeneidade, com LightGBM para os modelos de nuisance.
    -   **GATES (Group Average Treatment Effects):** Estima o ATE para quantis de CATEs, revelando como o efeito da IA varia com o capital humano latente.

4.  **`validation.py` (Nível 2 Heckman):**
    -   **`interpret_variance_heckman`:** Decompõe a variância do estimador e interpreta a `common_support_penalty` como uma medida quantitativa do viés de seleção.
    -   **`interpret_gates_heckman`:** Interpreta os GATES como proxies de capital humano latente, conectando os resultados à teoria de Cunha & Heckman (2007).
    -   **`run_heckman_two_step_benchmark`:** Compara o ATE do DML com o do método clássico de Heckman, demonstrando a superioridade dos career embeddings sobre a Inverse Mills Ratio (IMR).

### Orquestração (`main.py`):

-   **Etapa 1: Geração de Dados (DGP v3.2)**
-   **Etapa 2: Treino das 3 Variantes de Embedding**
-   **Etapa 3: Estimação DML e Comparação de Bias**
-   **Etapa 4: Validação Completa da Melhor Variante** (Variance Decomp, GATES, Oster, Placebos)
-   **Etapa 5: Benchmark Heckman Two-Step**
-   **Etapa 6: Teste de Robustez Estrutural vs. Mecânico (Nível 3 Heckman)**

## 4. Os 3 Níveis de Integração Heckman (v3.2)

Esta versão integra o trabalho de Heckman em 3 níveis de profundidade:

-   **Nível 1 (Narrativo/DGP):** O próprio DGP é construído sobre os princípios de Heckman (seleção endógena, capital humano latente). O código gera texto para a secção de metodologia do paper explicando esta escolha.

-   **Nível 2 (Interpretativo/Validação):** Os resultados (variância, GATES) não são apenas reportados, mas **interpretados através da lente teórica de Heckman**. O DML é explicitamente comparado ao método clássico de Heckman, provando a sua superioridade.

-   **Nível 3 (Estrutural/Robustez):** O pipeline prova que os resultados são robustos a diferentes modelos de seleção (mecânico vs. estrutural), respondendo à crítica central de Heckman sobre modelos de forma reduzida.

## 5. Próximos Passos

1.  **Submissão CBS:** Usar os outputs e a documentação (CV, Proposal, Blueprint) para finalizar a candidatura ao PhD da Copenhagen Business School.
2.  **Paper (Pós-aceitação):** Expandir o MVP para um paper completo, usando os textos gerados pelo `main.py` como base para a secção de metodologia.
3.  **Dados Reais:** Aplicar o framework validado aos Danish Register Data para gerar resultados empíricos.
