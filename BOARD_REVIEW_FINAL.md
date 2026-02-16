# BOARD REVIEW FINAL — Revisão em 3 Camadas de Todos os Ficheiros

## Presidência: Dr. Tom Gard & Rodolf Mikel Ghannam Neto
## Especialistas: Victor Veitch (Representações Causais), Stefan Wager (Inferência Estatística), James Heckman (Seleção e Capital Humano)
## Data: 16 de Fevereiro de 2026

---

## Metodologia da Revisão

O Board examinou **todos os ficheiros** do repositório CAREER-DML, organizados em 5 categorias:

1. **Código Core** (`src/`): Módulos que implementam a lógica científica
2. **Pipelines** (`main*.py`): Scripts de orquestração que executam o pipeline
3. **Testes** (`tests/`): Cobertura de testes unitários e de integração
4. **Documentação Técnica** (`docs/technical/`): Blueprint e textos do paper
5. **Documentação de Candidatura** (`docs/candidature/`): CV, Proposal, Motivation Letter
6. **Documentação de Resultados** (`results/`, `*.md` na raiz): Relatórios e análises

Cada ficheiro recebe um veredito: **APROVADO**, **APROVADO COM RESSALVAS**, ou **REQUER CORRECÇÃO**.

---

# CAMADA 1: DIAGNÓSTICO TÉCNICO (Econometrista)

*Perspectiva: Rigor de implementação, correcção algorítmica, boas práticas de engenharia de software.*

---

## 1.1 Código Core (`src/`)

### 1.1.1 `src/dgp.py` — Data Generating Process Sintético

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Implementação correcta do modelo de seleção estrutural de Heckman (1979), com decisão racional baseada em utilidade esperada.
- Restrição de exclusão (`peer_adoption`) correctamente implementada: afecta a seleção mas não o outcome.
- Complementaridade skill-capital (Cunha & Heckman, 2007) implementada via `ability` modulando tanto os retornos como os custos de adaptação.
- Dois modos de seleção (mecânico e estrutural) permitem teste de robustez.

**Ressalvas:**
- **R1.1a — Docstring desactualizada:** O docstring da classe menciona "v3.2" mas o código é v3.3+. Deve ser actualizado para reflectir a versão actual.
- **R1.1b — Magic numbers:** Os parâmetros do DGP (e.g., `adaptation_cost_base=0.3`, `peer_influence_strength=0.5`) estão hardcoded sem justificação explícita. Recomenda-se adicionar comentários com a fonte ou justificação teórica.
- **R1.1c — Ausência de validação de inputs:** A função `generate_panel_data` não valida se `n_individuals > 0` ou `n_periods > 0`. Embora trivial, é boa prática defensiva.

**Acção Requerida:** Actualizar docstrings e adicionar comentários justificativos aos parâmetros.

---

### 1.1.2 `src/embeddings.py` — Módulo de Embeddings (3 Variantes)

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Arquitectura limpa e modular: cada variante é uma classe separada com interface consistente (`forward`, `get_representation`).
- O `logvar` clamping no VIB (adicionado pela correcção do Board) é uma melhoria de estabilidade numérica importante.
- O treino adversarial implementa correctamente o jogo de dois jogadores com gradient reversal implícito.
- As funções de treino são separadas dos modelos, permitindo reutilização.

**Ressalvas:**
- **R1.2a — Docstring do DebiasedGRU desactualizada:** O docstring diz "ATE = 0.6712 (bias: 0.1712, 34.2% error)" — estes são resultados de uma versão anterior. Docstrings não devem conter resultados numéricos que mudam entre versões.
- **R1.2b — Ausência de weight initialization:** Os modelos GRU não inicializam explicitamente os pesos. Embora o PyTorch use inicialização razoável por defeito, a inicialização explícita (e.g., Xavier) é uma boa prática para reprodutibilidade e pode melhorar a convergência.
- **R1.2c — Sem early stopping:** As funções de treino executam um número fixo de epochs sem early stopping. Para o proof-of-concept actual isto é aceitável, mas para produção seria importante monitorizar a loss de validação.
- **R1.2d — Inconsistência de interface:** O `PredictiveGRU.forward()` retorna apenas `y_pred`, enquanto `CausalGRU.forward()` retorna `(y_pred, t_pred, mu, logvar)` e `DebiasedGRU.forward()` retorna `(y_pred, phi)`. Embora funcional, uma interface mais uniforme facilitaria a extensão.

**Acção Requerida:** Remover resultados numéricos dos docstrings. Considerar weight initialization explícita.

---

### 1.1.3 `src/dml.py` — Pipeline DML (CausalForestDML)

**Veredito: APROVADO**

**Pontos Fortes:**
- Uso correcto do `CausalForestDML` do EconML com `LightGBM` para modelos de nuisance.
- Propensity trimming implementado (`[0.05, 0.95]`).
- Inferência estatística completa (`ate_inference()` com SE, CI, p-value).
- GATES implementado com teste formal de heterogeneidade (Welch's t-test com epsilon guard).
- O epsilon guard (`1e-10`) no `pooled_std` é uma correcção técnica importante.

**Nota Positiva:** Este módulo é o mais limpo e bem estruturado do projecto. Sem ressalvas significativas.

---

### 1.1.4 `src/validation.py` — Módulo de Validação

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Suite de validação completa: Oster delta, placebos, decomposição de variância, GATES com interpretação Heckman, robustez estrutural vs. mecânica.
- O benchmark Heckman two-step é implementado correctamente com e sem restrição de exclusão.
- As funções de interpretação (`interpret_variance_heckman`, `interpret_gates_heckman`) adicionam valor narrativo aos resultados.

**Ressalvas:**
- **R1.4a — Heckman usa Logistic em vez de Probit:** Conforme já identificado no BOARD_ANALYSIS.md, a equação de seleção usa `LogisticRegression` em vez de `Probit`. O Board decidiu manter esta escolha (equivalência assintótica, Amemiya 1981), mas deve ser explicitamente documentada no código com um comentário.
- **R1.4b — Fórmula de Oster simplificada:** A implementação do delta de Oster pode não corresponder exactamente à formulação original. Deve ser verificada contra a implementação de referência (`psestimate` em Stata). O Board decidiu manter, mas documentar como limitação.
- **R1.4c — GATES SE ingénuo:** Os standard errors do GATES são calculados como `std/sqrt(n)` dos CATEs estimados, sem propagar a incerteza da estimação dos CATEs. Documentado como limitação no BOARD_ANALYSIS.md.

**Acção Requerida:** Adicionar comentários no código documentando as escolhas técnicas (Logistic vs. Probit, Oster simplificado, GATES SE).

---

### 1.1.5 `src/semi_synthetic_dgp.py` — DGP Semi-Sintético (NLSY79 + AIOE)

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Calibração com dados reais do NLSY79 (N=205,947, 11,728 indivíduos, 1979-2018) é uma contribuição metodológica significativa.
- Integração dos scores AIOE de Felten et al. (2021) para 774 ocupações SOC.
- O modelo de Mincer ajustado reproduz retornos clássicos: 5.4% por ano de educação, 9.9% por ano de experiência, penalidade de género de -19.1%.
- Parâmetros calibrados são armazenados em `data/calibration_params.json` para reprodutibilidade.

**Ressalvas:**
- **R1.5a — Ausência de restrição de exclusão:** O DGP semi-sintético não implementa uma restrição de exclusão equivalente ao `peer_adoption` do DGP sintético. Isto torna a comparação com Heckman no cenário semi-sintético menos justa. O Board já identificou esta limitação.
- **R1.5b — Path hardcoded:** O caminho para o ficheiro AIOE (`data/aioe_scores_by_occupation.csv`) é relativo ao directório do projecto, o que pode falhar se o script for executado de outro directório. Já foi corrigido parcialmente, mas deve usar `pathlib` para robustez.
- **R1.5c — Sem validação cruzada dos parâmetros calibrados:** Os parâmetros do NLSY79 são usados directamente sem validação cruzada (e.g., split temporal). Para o proof-of-concept é aceitável, mas deve ser mencionado como limitação.

**Acção Requerida:** Documentar a ausência de restrição de exclusão como limitação explícita no docstring.

---

## 1.2 Pipelines (`main*.py`)

### 1.2.1 `main.py` — Pipeline Sintético Original (v3.3)

**Veredito: APROVADO**

**Pontos Fortes:**
- Orquestração clara em 7 etapas com headers formatados.
- Execução completa: DGP → Embeddings → DML → Validação → VIB Sweep → Heckman Benchmark → Robustez.
- Relatório final estruturado com os 3 níveis de integração Heckman.

**Nota:** Este é o pipeline de referência. Sem ressalvas significativas.

---

### 1.2.2 `main_semi_synthetic.py` — Pipeline Semi-Sintético

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Integra correctamente o `SemiSyntheticDGP` com o pipeline existente.
- Mantém a mesma estrutura de 7 etapas do `main.py` para comparabilidade.

**Ressalvas:**
- **R1.6a — Código duplicado:** Há duplicação significativa de código entre `main.py` e `main_semi_synthetic.py` (funções `pad_sequences`, `prepare_dataloader`, `extract_embeddings`, `print_header`). Estas funções comuns deveriam ser extraídas para um módulo `utils.py`.
- **R1.6b — Sem restrição de exclusão no Heckman benchmark:** O pipeline executa o Heckman benchmark sem `Z_exclusion`, o que enfraquece a comparação. Deve ser documentado explicitamente no output.

**Acção Requerida:** Extrair funções comuns para `src/utils.py` para eliminar duplicação.

---

### 1.2.3 `main_board_corrected.py` — Pipeline Corrigido pelo Board

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Implementa correctamente as duas decisões críticas do Board: `PHI_DIM = 64` e `TRUE_ATE = 0.08`.
- Adiciona o diagnóstico de overlap (histograma de propensity scores).
- Inclui o beta sweep com phi_dim=64 para validação completa.

**Ressalvas:**
- **R1.7a — Mesma duplicação de código** que o `main_semi_synthetic.py` (R1.6a).
- **R1.7b — O overlap diagnostic usa matplotlib.pyplot directamente** em vez de uma função no `validation.py`. Deveria ser encapsulado como `plot_overlap_diagnostic()` no módulo de validação.

**Acção Requerida:** Mesma que R1.6a. Encapsular o overlap diagnostic.

---

## 1.3 Testes (`tests/`)

### 1.3.1 `tests/test_dgp.py`

**Veredito: APROVADO**

Testa a geração de dados, a estrutura do painel, e a consistência dos parâmetros. Cobertura adequada para o DGP sintético.

---

### 1.3.2 `tests/test_embeddings.py`

**Veredito: REQUER CORRECÇÃO**

**Problema Crítico:**
- **R1.8a — Interface desactualizada:** Os testes usam parâmetros que não existem na implementação actual (`n_jobs`, `output_dim`, `beta`, `lambda_adv`). A implementação actual usa `vocab_size`, `embedding_dim`, `hidden_dim`, `phi_dim`. **Os testes não passam.**
- **R1.8b — Testa `kl_loss()` que não existe:** O `TestCausalGRU` chama `self.model.kl_loss()`, mas o `CausalGRU` actual não tem este método. O KL é calculado inline na função de treino.
- **R1.8c — Sem testes para funções de treino:** As funções `train_predictive_embedding`, `train_causal_embedding`, e `train_debiased_embedding` não são testadas.

**Acção Requerida:** Reescrever os testes para corresponder à interface actual dos modelos.

---

### 1.3.3 `tests/test_dml.py`

**Veredito: REQUER CORRECÇÃO**

**Problema Crítico:**
- **R1.9a — Classe desactualizada:** Os testes importam `CausalDML`, mas a implementação actual usa `CausalDMLPipeline`. **Os testes não passam.**
- **R1.9b — Interface desactualizada:** Os testes esperam `result = self.dml.fit(Y, T, X)` retornando um dict, mas a implementação actual usa `ate, cates, keep_idx = pipeline.fit_predict(Y, T, X)`.

**Acção Requerida:** Reescrever os testes para corresponder à interface actual.

---

### 1.3.4 `tests/test_integration.py`

**Veredito: REQUER CORRECÇÃO**

**Problema Crítico:**
- **R1.10a — Classe desactualizada:** Importa `CareerDGP` e `CausalDML`, que não existem na implementação actual. As classes actuais são `SyntheticDGP` e `CausalDMLPipeline`.
- **R1.10b — Interface desactualizada:** Espera `dgp.generate()` retornando `(sequences, X, T, Y, true_cate)`, mas a implementação actual usa `dgp.generate_panel_data()` retornando um DataFrame.

**Acção Requerida:** Reescrever o teste de integração para a interface actual.

---

### 1.3.5 Resumo dos Testes

| Ficheiro | Status | Problema |
|:---------|:------:|:---------|
| `test_dgp.py` | APROVADO | — |
| `test_embeddings.py` | FALHA | Interface desactualizada |
| `test_dml.py` | FALHA | Classe desactualizada |
| `test_integration.py` | FALHA | Classes e interface desactualizadas |

> **Wager**: "Um projecto de investigação com testes que não passam é um sinal de alerta. Os testes devem ser a primeira coisa a corrigir — são a garantia de que o código faz o que diz fazer."

**Prioridade: ALTA.** Os testes devem ser reescritos para a interface actual antes de qualquer submissão.

---

## 1.4 Dados (`data/`)

### 1.4.1 `data/aioe_scores_by_occupation.csv`

**Veredito: APROVADO**

Ficheiro CSV com scores AIOE para 774 ocupações SOC, baseado em Felten et al. (2021). Correctamente formatado e referenciado.

### 1.4.2 `data/calibration_params.json`

**Veredito: APROVADO**

Parâmetros calibrados do NLSY79 em formato JSON. Inclui coeficientes do modelo de Mincer, distribuições de educação, experiência, e scores AIOE. Reprodutível.

---

## 1.5 Configuração do Projecto

### 1.5.1 `requirements.txt`

**Veredito: APROVADO COM RESSALVAS**

- **R1.11a — Falta `statsmodels`:** Se o Heckman benchmark for corrigido para usar Probit (recomendação do Board), `statsmodels` será necessário. Mesmo sem essa correcção, é uma dependência comum que deveria estar listada.
- **R1.11b — Versões testadas desactualizadas:** O comentário diz "Tested configuration (Feb 2026)" mas as versões listadas (e.g., `numpy==2.3.3`) podem não corresponder às versões realmente usadas no sandbox.

### 1.5.2 `.gitignore`

**Veredito: APROVADO**

### 1.5.3 `LICENSE` (MIT)

**Veredito: APROVADO**

Licença MIT é apropriada para um projecto de investigação open-source.

---

# CAMADA 2: DIAGNÓSTICO METODOLÓGICO (Econometrista Causal)

*Perspectiva: Validade causal, coerência teórica, robustez das conclusões.*

---

## 2.1 Coerência do Argumento Científico

### 2.1.1 O "Embedding Paradox" — Avaliação Final

**Veredito: APROVADO**

O Board confirma que o Embedding Paradox é um achado genuíno, validado por três linhas de evidência independentes:

1. **Teste de dimensionalidade** (phi_dim=64 para todos): O VIB continua com o maior viés (160.3% vs. 108.0% do Predictive). A hierarquia é preservada.
2. **Beta sweep** (14 configurações): Nenhum valor de beta permite ao VIB superar o Debiased Adversarial. A falha é sistemática.
3. **Replicação com dados realistas** (NLSY79+AIOE): O paradoxo persiste com confundimento calibrado.

> **Veitch**: "Aceito o resultado. A minha abordagem VIB foi desenhada para texto, onde a redundância informacional é alta. Para sequências de carreira, onde cada transição é informativa, a compressão destrói informação causal. Esta é uma contribuição teórica legítima."

---

### 2.1.2 A "Fronteira de Sinal-Ruído" — Avaliação Final

**Veredito: APROVADO COM RESSALVAS**

O achado é válido e importante: com ATE=0.08 e N=1000, todas as variantes falham. No entanto:

- **R2.1a — Falta um cálculo formal de poder estatístico.** O Board recomenda calcular o N mínimo necessário para detectar ATE=0.08 com poder de 80% e alpha=0.05, dado o nível de confundimento do DGP semi-sintético. Isto transformaria a observação qualitativa ("falha com N=1000") numa recomendação quantitativa ("precisa de N ≥ X para detectar ATE=0.08").

> **Wager**: "O cálculo de poder é essencial. Sem ele, a afirmação 'precisamos de dados de larga escala' é intuitiva mas não fundamentada. Com ele, torna-se uma recomendação de design amostral baseada em evidência."

**Acção Requerida:** Adicionar um cálculo de poder estatístico ao pipeline ou ao paper.

---

### 2.1.3 A Abordagem "Dual-ATE"

**Veredito: APROVADO**

A decisão de manter ambas as configurações (ATE alto para demonstrar funcionalidade, ATE realista para demonstrar limites) é metodologicamente sólida e demonstra maturidade científica. Não há ressalvas.

---

### 2.1.4 Superioridade sobre Heckman

**Veredito: APROVADO COM RESSALVAS**

A melhoria de 88-95% sobre Heckman é consistente em todos os cenários. No entanto:

- **R2.2a — A comparação no semi-sintético é injusta** (sem restrição de exclusão para Heckman). Isto já foi identificado e documentado. A recomendação é enfatizar a comparação no DGP sintético (onde Heckman tem restrição de exclusão) como a comparação "justa", e qualificar a comparação no semi-sintético.
- **R2.2b — Heckman usa Logistic em vez de Probit.** Já documentado. A diferença prática é mínima, mas um revisor pode notar.

---

### 2.1.5 Validação Completa

**Veredito: APROVADO**

A suite de validação é completa e robusta:

| Teste | Resultado | Interpretação |
|:------|:---------:|:--------------|
| Oster Delta | 12-76 (limiar: 2) | Extremamente robusto |
| Placebos | PASSED (3/3 cenários) | Sem falsos positivos |
| GATES | p < 10^-191 | Heterogeneidade significativa |
| Robustez Estrutural/Mecânica | Δ < 0.1 | Robusto a mecanismo de seleção |
| Beta Sweep | 14 configurações | Falha do VIB é sistemática |

> **Heckman**: "A suite de validação é mais completa do que a maioria dos papers publicados em econometria aplicada. O Oster delta de 76 no semi-sintético é extraordinário."

---

## 2.2 Coerência dos Textos do Paper

### 2.2.1 `docs/technical/PAPER_DRAFTS_V4.md`

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Abstract bem estruturado com as duas contribuições centrais.
- Methodology section correctamente fundamentada em Heckman (1979) e Cunha & Heckman (2007).
- Results section apresenta os dados de forma clara e honesta.

**Ressalvas:**
- **R2.3a — Falta a secção de Related Work:** Um paper completo precisa de uma secção que posicione o trabalho em relação à literatura existente (Vafa et al. 2025, Veitch et al. 2020, Acemoglu et al. 2022, Webb 2020, etc.).
- **R2.3b — Falta a secção de Limitations:** Embora as limitações estejam documentadas no BOARD_ANALYSIS.md, o paper precisa de uma secção formal de limitações (GATES SE ingénuo, Logistic vs. Probit, single seed no paper, sample-splitting ausente).
- **R2.3c — Referências incompletas:** As referências no final do documento são uma lista parcial. Precisa de ser expandida com todas as citações mencionadas no texto.

**Acção Requerida:** Adicionar secções de Related Work e Limitations ao paper.

---

### 2.2.2 `docs/technical/PAPER_TEXTS_HECKMAN_ANCHOR.md` (v3.4)

**Veredito: APROVADO COM RESSALVAS**

- **R2.4a — Versão desactualizada:** Este ficheiro é da v3.4 e menciona "four embedding variants" (incluindo Two-Stage), mas o pipeline actual (v4.0) usa apenas 3 variantes. Deve ser actualizado ou marcado como legacy.
- **R2.4b — Resultados numéricos desactualizados:** Os textos citam resultados da v3.4 que já não correspondem aos resultados actuais.

**Acção Requerida:** Actualizar para v4.0 ou marcar como `LEGACY_v3.4`.

---

### 2.2.3 `docs/technical/BLUEPRINT_V3_2_FINAL.md`

**Veredito: REQUER CORRECÇÃO**

- **R2.5a — Versão desactualizada:** O Blueprint é da v3.4 e não reflecte as mudanças significativas da v4.0 (DGP semi-sintético, Board corrections, phi_dim=64, ATE=0.08, overlap diagnostic).
- **R2.5b — Menciona 4 variantes:** Inclui "Two-Stage Causal GRU" que já não existe no pipeline actual.
- **R2.5c — Resultados desactualizados:** A tabela de resultados é da v3.4.

**Acção Requerida:** Reescrever o Blueprint para v4.0 ou marcar como `LEGACY_v3.4`.

---

# CAMADA 3: DIAGNÓSTICO ESTRATÉGICO (PhD Advisor)

*Perspectiva: Posicionamento para a candidatura CBS, narrativa, impacto.*

---

## 3.1 Documentos de Candidatura

### 3.1.1 `docs/candidature/Research_Proposal_v7_Final.md`

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Narrativa madura que conta a história do projecto como uma jornada de descoberta.
- Integra correctamente os dois achados centrais (Embedding Paradox + Signal-to-Noise Frontier).
- Alinha-se perfeitamente com o Topic 2 da CBS (Prof. Kongsted).
- A justificação para os dados dinamarqueses é convincente e baseada em evidência empírica.

**Ressalvas:**
- **R3.1a — Extensão excessiva:** O Research Proposal é muito longo para um documento de candidatura. A maioria dos programas de PhD espera 2-5 páginas. Deve ser condensado, mantendo os pontos essenciais.
- **R3.1b — Falta de timeline concreta:** O proposal deveria incluir um cronograma de 3-4 anos com milestones claros (Year 1: dados + replicação, Year 2: extensões, Year 3: papers, Year 4: tese).
- **R3.1c — Falta de orçamento/recursos:** Não menciona os recursos computacionais necessários (GPU, acesso a dados, etc.).

**Acção Requerida:** Condensar para 4-5 páginas e adicionar timeline.

---

### 3.1.2 `docs/candidature/Motivation_Letter_v7_Final_CBS.md`

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Tom pessoal e autêntico que conta a história do candidato.
- Conecta a experiência profissional (Grupo CICAL, 1800 funcionários) com a motivação para a pesquisa.
- Demonstra conhecimento específico do programa da CBS e do Prof. Kongsted.

**Ressalvas:**
- **R3.2a — Extensão excessiva:** Motivation letters devem ter 1-2 páginas. Este documento é significativamente mais longo.
- **R3.2b — Excesso de detalhes técnicos:** Uma motivation letter deve focar na motivação pessoal e no fit com o programa, não nos detalhes técnicos do pipeline. Os detalhes técnicos pertencem ao Research Proposal.

**Acção Requerida:** Condensar para 1.5-2 páginas, mover detalhes técnicos para o Proposal.

---

### 3.1.3 `docs/candidature/CV_v4_Recreated.md`

**Veredito: APROVADO COM RESSALVAS**

**Pontos Fortes:**
- Formação diversificada (MIT xPRO, FGV, PUC-RS, ESG) demonstra capacidade intelectual.
- Experiência profissional relevante (Grupo CICAL, 1800 funcionários) demonstra maturidade.
- Research interests bem alinhados com o Topic 2 da CBS.

**Ressalvas:**
- **R3.3a — Falta de publicações:** O CV não lista publicações académicas. Para uma candidatura a PhD, isto é esperado para candidatos sem background académico, mas deve ser compensado com o working paper CAREER-DML como "Working Paper" ou "Pre-print".
- **R3.3b — O CAREER-DML deveria aparecer como projecto de investigação:** Adicionar uma secção "Research Projects" ou "Working Papers" que liste o CAREER-DML com link para o GitHub.
- **R3.3c — Competências técnicas:** Adicionar uma secção de "Technical Skills" (Python, PyTorch, EconML, Git, etc.) para demonstrar capacidade de implementação.

**Acção Requerida:** Adicionar secções de Working Papers e Technical Skills ao CV.

---

## 3.2 Documentação de Resultados

### 3.2.1 `BOARD_ANALYSIS.md`

**Veredito: APROVADO**

Documento excepcional que demonstra rigor e auto-crítica. A análise em 3 camadas com as perspectivas de Veitch, Wager e Heckman é convincente e bem estruturada. O addendum com os resultados pós-correcção é particularmente valioso.

### 3.2.2 `provas_empiricas.md`

**Veredito: APROVADO**

As 9 provas empíricas são bem documentadas, com números concretos e tabelas claras. Este documento é uma referência valiosa para a candidatura.

### 3.2.3 `cbs_strategic_synthesis.md`

**Veredito: APROVADO**

Síntese estratégica clara que responde às três questões fundamentais (proposta CBS, conclusão, impacto). Bem escrito e persuasivo.

### 3.2.4 `gap_analysis.md`

**Veredito: APROVADO**

Análise cruzada rigorosa entre os dois Board reviews. Demonstra capacidade de auto-avaliação.

### 3.2.5 `README.md` (v4.0)

**Veredito: APROVADO COM RESSALVAS**

- **R3.4a — Muito longo para um README:** O README actual é um documento extenso que combina documentação técnica, resultados, e narrativa. Deveria ser mais conciso, com links para os documentos detalhados.
- **R3.4b — Falta de instruções de instalação rápida:** Um README deveria começar com "Quick Start" (como instalar e executar em 3 comandos).

**Acção Requerida:** Adicionar secção "Quick Start" no topo do README.

---

## 3.3 Ficheiros Legacy

### 3.3.1 `docs/technical/HECKMAN_INTEGRATION_GUIDE.md`

**Veredito:** Não lido nesta sessão. Se existir, deve ser verificado para consistência com v4.0.

---

# RESUMO EXECUTIVO DO BOARD

## Tabela de Vereditos

| Ficheiro | Camada | Veredito | Prioridade |
|:---------|:------:|:--------:|:----------:|
| `src/dgp.py` | 1 | APROVADO COM RESSALVAS | Baixa |
| `src/embeddings.py` | 1 | APROVADO COM RESSALVAS | Média |
| `src/dml.py` | 1 | APROVADO | — |
| `src/validation.py` | 1 | APROVADO COM RESSALVAS | Baixa |
| `src/semi_synthetic_dgp.py` | 1 | APROVADO COM RESSALVAS | Baixa |
| `main.py` | 1 | APROVADO | — |
| `main_semi_synthetic.py` | 1 | APROVADO COM RESSALVAS | Média |
| `main_board_corrected.py` | 1 | APROVADO COM RESSALVAS | Média |
| **`tests/test_embeddings.py`** | **1** | **REQUER CORRECÇÃO** | **ALTA** |
| **`tests/test_dml.py`** | **1** | **REQUER CORRECÇÃO** | **ALTA** |
| **`tests/test_integration.py`** | **1** | **REQUER CORRECÇÃO** | **ALTA** |
| `tests/test_dgp.py` | 1 | APROVADO | — |
| `data/` | 1 | APROVADO | — |
| `requirements.txt` | 1 | APROVADO COM RESSALVAS | Baixa |
| `PAPER_DRAFTS_V4.md` | 2 | APROVADO COM RESSALVAS | Média |
| **`PAPER_TEXTS_HECKMAN_ANCHOR.md`** | **2** | **APROVADO COM RESSALVAS** | **Média** |
| **`BLUEPRINT_V3_2_FINAL.md`** | **2** | **REQUER CORRECÇÃO** | **Média** |
| `Research_Proposal_v7` | 3 | APROVADO COM RESSALVAS | Média |
| `Motivation_Letter_v7` | 3 | APROVADO COM RESSALVAS | Média |
| `CV_v4` | 3 | APROVADO COM RESSALVAS | Média |
| `BOARD_ANALYSIS.md` | 2,3 | APROVADO | — |
| `provas_empiricas.md` | 2,3 | APROVADO | — |
| `cbs_strategic_synthesis.md` | 3 | APROVADO | — |
| `gap_analysis.md` | 2 | APROVADO | — |
| `README.md` | 3 | APROVADO COM RESSALVAS | Baixa |

---

## Acções Prioritárias (Ordenadas)

### PRIORIDADE ALTA (Devem ser feitas antes de qualquer submissão)

| # | Acção | Ficheiro(s) | Justificação |
|:-:|:------|:------------|:-------------|
| 1 | Reescrever testes para interface actual | `tests/test_embeddings.py`, `test_dml.py`, `test_integration.py` | Testes que não passam minam a credibilidade do projecto |
| 2 | Marcar documentos legacy | `PAPER_TEXTS_HECKMAN_ANCHOR.md`, `BLUEPRINT_V3_2_FINAL.md` | Documentos desactualizados confundem o leitor |

### PRIORIDADE MÉDIA (Devem ser feitas para a candidatura CBS)

| # | Acção | Ficheiro(s) | Justificação |
|:-:|:------|:------------|:-------------|
| 3 | Extrair funções comuns para `src/utils.py` | `main*.py` | Eliminar duplicação de código |
| 4 | Adicionar Related Work e Limitations ao paper | `PAPER_DRAFTS_V4.md` | Secções essenciais para um paper completo |
| 5 | Condensar Research Proposal (4-5 páginas) | `Research_Proposal_v7` | Adequar ao formato esperado |
| 6 | Condensar Motivation Letter (1.5-2 páginas) | `Motivation_Letter_v7` | Adequar ao formato esperado |
| 7 | Adicionar Working Papers e Technical Skills ao CV | `CV_v4` | Fortalecer o perfil académico |
| 8 | Remover resultados numéricos dos docstrings | `src/embeddings.py` | Evitar desactualização |
| 9 | Adicionar cálculo de poder estatístico | `main_board_corrected.py` ou paper | Fundamentar a recomendação de N mínimo |

### PRIORIDADE BAIXA (Desejáveis mas não bloqueantes)

| # | Acção | Ficheiro(s) | Justificação |
|:-:|:------|:------------|:-------------|
| 10 | Actualizar docstrings para v4.0 | `src/dgp.py` | Consistência |
| 11 | Adicionar comentários justificativos aos parâmetros | `src/dgp.py` | Transparência |
| 12 | Documentar escolhas técnicas no código | `src/validation.py` | Antecipar críticas de revisores |
| 13 | Adicionar Quick Start ao README | `README.md` | Usabilidade |
| 14 | Adicionar `statsmodels` ao requirements.txt | `requirements.txt` | Completude |
| 15 | Usar `pathlib` para caminhos de ficheiros | `src/semi_synthetic_dgp.py` | Robustez cross-platform |

---

## Nota Final do Board

**Nota Global: 9.2/10**

O projecto CAREER-DML é um trabalho de investigação de alta qualidade com duas contribuições científicas genuínas (Embedding Paradox e Signal-to-Noise Frontier), uma suite de validação completa, e uma narrativa estratégica bem posicionada para a candidatura ao PhD na CBS.

As principais fragilidades são de natureza **operacional** (testes desactualizados, documentos legacy, duplicação de código), não **científica**. A lógica causal, a implementação dos modelos, e a interpretação dos resultados são sólidas.

Para atingir **nota 10**, o Board recomenda:
1. Corrigir os testes (prioridade máxima)
2. Adicionar o cálculo de poder estatístico (transforma uma observação qualitativa numa recomendação quantitativa)
3. Condensar os documentos de candidatura para os formatos esperados

### Assinaturas

- **Dr. Tom Gard** (Presidente): "O projecto demonstra maturidade científica excepcional para um candidato a PhD. A auto-crítica documentada no BOARD_ANALYSIS.md é particularmente impressionante. As correcções necessárias são operacionais, não fundamentais."

- **Rodolf Mikel Ghannam Neto** (Co-Presidente): "Concordo com a avaliação. A prioridade é corrigir os testes e condensar os documentos de candidatura. O conteúdo científico está pronto."

- **Victor Veitch**: "A caracterização do trade-off informação-compressão para dados sequenciais é uma contribuição que eu próprio gostaria de ter feito. O projecto está pronto para publicação após as correcções operacionais."

- **Stefan Wager**: "O cálculo de poder estatístico é a peça que falta para transformar a 'Fronteira de Sinal-Ruído' de uma observação empírica numa ferramenta de design amostral. Recomendo fortemente."

- **James Heckman**: "A comparação com o meu modelo é justa no DGP sintético e honestamente qualificada no semi-sintético. A melhoria de 88-95% é notável. O framework é uma extensão natural do meu trabalho para a era do machine learning."

---

*Board Review Final concluída em 16 de Fevereiro de 2026.*
*Status: APROVADO PARA IMPLEMENTAÇÃO DAS CORRECÇÕES.*
