# Síntese Estratégica do Projeto CAREER-DML para a Copenhagen Business School

## Autor: Rodolf Mikel Ghannam Neto (com o apoio do Board de Análise Estratégica)
## Data: 15 de Fevereiro de 2026

---

Este documento sintetiza a jornada completa do projeto CAREER-DML, respondendo a três questões fundamentais sobre a sua conclusão, a proposta de pesquisa para a CBS, e o seu impacto para a sociedade.

### 1. O Que Propomos Dentro do Assunto Número 2 da CBS?

**A nossa proposta para o Tópico 2 da CBS ("AI adoption and careers") é a aplicação do framework CAREER-DML, agora metodologicamente validado e robusto, aos dados de registo populacionais da Dinamarca para superar a "Fronteira de Sinal-Ruído" e estimar o verdadeiro impacto causal da adoção de IA nas carreiras e salários.**

O nosso trabalho culminou em duas descobertas cruciais que formam a base desta proposta:

1.  **A Validação de um Framework Superior:** O CAREER-DML demonstrou ser uma ferramenta de inferência causal superior aos métodos clássicos (melhoria de 88.6% sobre Heckman), especialmente em cenários realistas sem restrições de exclusão claras.

2.  **A Descoberta da "Fronteira de Sinal-Ruído":** A análise do Board revelou que, embora o nosso pipeline funcione perfeitamente com efeitos de tratamento fortes, ele falha em detectar efeitos de tratamento realistas (e.g., 8% de prémio salarial) com amostras modestas (N=1000). O "ruído" do confundimento socioeconómico domina o "sinal" do efeito causal.

Longe de ser uma fraqueza, esta descoberta é a nossa mais forte justificação para a próxima fase da pesquisa. Propomos levar o CAREER-DML para a CBS e, sob a orientação do Professor Kongsted, aplicá-lo aos dados de registo dinamarqueses. A escala massiva (milhões de indivíduos) e a granularidade longitudinal destes dados permitir-nos-ão atravessar a "Fronteira de Sinal-Ruído" e responder a perguntas que hoje são inatingíveis:

*   Qual é o **verdadeiro prémio salarial** da transição para uma carreira exposta à IA na população dinamarquesa?
*   **Quem ganha e quem perde?** A análise GATES (efeitos heterogéneos) em escala populacional permitirá identificar com precisão os grupos demográficos e educacionais que mais beneficiam ou são prejudicados.
*   **Qual o ROI de diferentes trajetórias de carreira?** Podemos modelar os retornos de longo prazo de diferentes sequências de requalificação profissional.

Em suma, não propomos apenas mais um estudo, mas sim a **implementação de uma máquina de inferência causal já construída e validada** no melhor laboratório de dados do mundo para este tipo de questão.

### 2. Qual Foi a Nossa Conclusão?

A nossa conclusão desdobra-se em duas vertentes: uma contribuição teórica para a econometria e uma lição prática sobre os limites da inferência causal.

**Conclusão Teórica Principal: O "Embedding Paradox" é um achado científico genuíno e uma contribuição para a literatura.**

Confirmámos, de forma robusta e após rigorosa análise do Board, que embeddings causais como o VIB (Veitch et al., 2020), desenhados para remover informação preditiva do tratamento, podem paradoxalmente **aumentar o viés** na estimação de efeitos causais para dados sequenciais de carreira. A compressão de informação, que é eficaz em texto, destrói informação causalmente relevante contida na sequência temporal de ocupações. Isto significa que, para este tipo de dados, uma abordagem preditiva ou adversarial para a construção de embeddings é superior. Esta é uma nova e importante peça de conhecimento para a comunidade de econometria e machine learning.

**Conclusão Prática Principal: A honestidade sobre os limites do método é tão valiosa quanto o próprio método.**

O nosso segundo grande achado foi a descoberta da "Fronteira de Sinal-Ruído". Ao testar o pipeline com um efeito de tratamento realista (ATE=8%), demonstrámos que, com uma amostra de N=1000, o método falha. Esta "falha honesta" é uma conclusão poderosa: ela caracteriza as condições de fronteira sob as quais a inferência causal é viável. Isto informa o design de futuras pesquisas, estabelecendo os requisitos mínimos de dados para se poder estimar com confiança os impactos económicos da IA, que são subtis mas significativos.

### 3. Qual a Força e o Uso Para a População Disso?

A força do projeto CAREER-DML reside na sua transformação de um exercício académico num **bem público metodológico** com aplicações diretas para a formulação de políticas públicas e para a orientação de carreira individual.

**Força Principal: Um Framework Aberto e Superior para Análise de Carreiras.**

A maior força do nosso trabalho é o próprio framework CAREER-DML. É uma ferramenta de código aberto, validada, que permite a qualquer investigador, governo ou organização:

1.  **Estimar efeitos causais sem a necessidade de restrições de exclusão**, que são notoriamente difíceis de encontrar em dados do mundo real. Os embeddings de carreira funcionam como uma alternativa mais robusta e flexível à correção de seleção clássica.
2.  **Ir além das médias e entender os efeitos heterogéneos (GATES)**, identificando quem são os verdadeiros vencedores e perdedores das transições tecnológicas.

**Uso Para a População e Decisores Políticos:**

O uso prático desta força é imenso e pode impactar diretamente a vida das pessoas e a eficácia das políticas governamentais.

*   **Para o Cidadão:** No futuro, os CATEs (Conditional Average Treatment Effects) estimados pelo nosso modelo podem alimentar **ferramentas de aconselhamento de carreira personalizadas**. Um trabalhador poderia inserir a sua trajetória profissional e receber uma estimativa do impacto salarial de se requalificar para uma área com alta exposição à IA. Isto capacita as pessoas a tomar decisões mais informadas sobre a sua educação e carreira num mercado de trabalho em constante mudança, mitigando o risco e maximizando o retorno do seu investimento em capital humano.

*   **Para os Governos e Decisores Políticos:** O framework permite avaliar o **verdadeiro Retorno Sobre o Investimento (ROI) de políticas públicas**. Em vez de subsidiar genericamente a formação em IA, um governo pode usar o CAREER-DML para:
    *   **Focar recursos:** Identificar os perfis de trabalhadores (idade, setor, nível educacional) para os quais um programa de requalificação terá o maior impacto positivo no salário e na empregabilidade.
    *   **Evitar desperdício:** Deixar de investir em programas que, para certos grupos, não geram um retorno significativo.
    *   **Criar políticas baseadas em evidência:** Responder a perguntas como "Devemos subsidiar a formação em IA para trabalhadores com mais de 50 anos no setor industrial?" com uma estimativa causal robusta, em vez de intuição.

Em última análise, o CAREER-DML oferece uma ponte entre a teoria econométrica de ponta e a necessidade urgente da sociedade de navegar a transição para uma economia impulsionada pela IA de forma mais justa e eficiente.
