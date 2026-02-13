"""
CAREER-DML: Data Generating Process (DGP) - v3.2 (HECKMAN STRUCTURAL)
Camada 1 - Gerador de dados sintéticos com ground truth causal conhecido.

Evolução v3.1 → v3.2:
    - NÍVEL 3 HECKMAN: O mecanismo de seleção para o tratamento agora segue um
      modelo de decisão racional (utility-based), onde o agente calcula a utilidade
      esperada de adotar IA vs. não adotar, considerando benefícios salariais e
      custos de adaptação. Isto transforma o modelo de "reduzido" para
      (minimamente) "estrutural", endereçando a crítica central de Heckman.
    - NÍVEL 2 HECKMAN: O DGP agora suporta dois modos de seleção ('mechanical'
      e 'structural') para permitir comparação de robustez.

Referências:
    - Heckman (1979), Sample Selection Bias as a Specification Error
    - Cunha & Heckman (2007), The Technology of Skill Formation
    - Chernozhukov et al. (2018), Double/Debiased ML
    - Veitch et al. (2020), Adapting Text Embeddings for Causal Inference
"""

import numpy as np
import pandas as pd
from scipy.special import expit


class SyntheticDGP:
    """Synthetic Data Generating Process for causal inference validation.

    v3.2: Suporta seleção mecânica (v3.1) e estrutural (Heckman-style).

    Args:
        true_ate: Average Treatment Effect verdadeiro.
        delta_e: Moderação do efeito por educação.
        delta_a: Moderação do efeito por habilidade (capital humano).
        gamma_a: Efeito direto da habilidade no outcome.
        lambda_f: Efeito da interação tratamento x flexicurity.
        n_occupations: Número de ocupações no vocabulário de carreira.
        selection_mode: 'mechanical' (v3.1 original) ou 'structural' (Heckman v3.2).
        adaptation_cost_base: Custo base de adaptação à IA (Nível 3 Heckman).
        adaptation_cost_ability_factor: Quanto a ability reduz o custo (Nível 3).
    """

    def __init__(
        self,
        true_ate: float = 0.5,
        delta_e: float = 0.15,
        delta_a: float = 0.2,
        gamma_a: float = 0.25,
        lambda_f: float = 0.05,
        n_occupations: int = 50,
        selection_mode: str = "structural",
        adaptation_cost_base: float = 1.2,
        adaptation_cost_ability_factor: float = 0.8,
    ):
        self.true_ate = true_ate
        self.delta_e = delta_e
        self.delta_a = delta_a
        self.gamma_a = gamma_a
        self.lambda_f = lambda_f
        self.n_occupations = n_occupations
        self.selection_mode = selection_mode
        self.adaptation_cost_base = adaptation_cost_base
        self.adaptation_cost_ability_factor = adaptation_cost_ability_factor
        self.beta_x = np.array([0.2, -0.1])
        self.sigma_y = 0.5
        self.covariates = ["education", "flexicurity"]

        # Matrizes de Transição de Carreira
        self.transition_matrix_baseline = self._init_transition_matrix()
        self.transition_matrix_ai = self._init_ai_biased_matrix()

    def _init_transition_matrix(self) -> np.ndarray:
        """Cria uma matriz de transição de Markov estocástica."""
        matrix = np.random.dirichlet(np.ones(self.n_occupations), size=self.n_occupations)
        return matrix

    def _init_ai_biased_matrix(self) -> np.ndarray:
        """Cria uma matriz de transição que favorece ocupações de 'tecnologia'."""
        matrix = self._init_transition_matrix()
        matrix[:, -10:] *= 1.5
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        return matrix

    def _generate_static_features(self, n_individuals: int) -> pd.DataFrame:
        """Gera características individuais imutáveis no tempo."""
        return pd.DataFrame(
            {
                "education": np.random.choice(
                    [0, 1, 2], size=n_individuals, p=[0.3, 0.5, 0.2]
                ),
                "flexicurity": np.random.uniform(0, 1, size=n_individuals),
            }
        )

    def _generate_latent_ability(self, X_static: pd.DataFrame) -> np.ndarray:
        """Gera o confounder não observado (habilidade latente).

        Heckman (1979): Esta é a variável que causa o viés de seleção.
        Cunha & Heckman (2007): Correlacionada com educação (capital humano).
        """
        return np.random.normal(X_static["education"] * 0.2, 1)

    # =========================================================================
    # NÍVEL 3 HECKMAN: Modelo de Decisão Racional (Structural Selection)
    # =========================================================================

    def _calculate_expected_utility(self, ability: float, education: int) -> tuple[float, float]:
        """Calcula a utilidade esperada de adotar vs. não adotar IA.

        Este método implementa um modelo de decisão racional simplificado,
        inspirado no framework de Heckman. O agente:
        1. Estima o seu salário futuro com e sem IA.
        2. Calcula o custo de adaptação (inversamente proporcional à ability).
        3. Escolhe o tratamento se a utilidade líquida for positiva.

        Isto cria uma forma de seleção ENDÓGENA e ESTRUTURAL:
        - Agentes com alta ability têm MENOR custo de adaptação → mais propensos a adotar.
        - Agentes com alta ability também têm MAIOR retorno da IA (via delta_a).
        - Isto gera o confounding: ability → T e ability → Y.

        A diferença em relação ao modelo mecânico (v3.1) é que aqui a seleção
        resulta de uma DECISÃO OTIMIZADORA, não de uma regra probabilística ad hoc.

        Args:
            ability: Habilidade latente do indivíduo.
            education: Nível de educação (0, 1, 2).

        Returns:
            Tupla (utility_no_ai, utility_with_ai).
        """
        # Salário esperado sem IA (baseline)
        expected_wage_no_ai = 1.5 + 0.8 * education + self.gamma_a * ability

        # Salário esperado com IA (inclui o efeito causal verdadeiro)
        expected_wage_with_ai = (
            expected_wage_no_ai
            + self.true_ate
            + self.delta_e * education
            + self.delta_a * ability
        )

        # Custo de adaptação à IA (Cunha & Heckman 2007: skill complementarity)
        # Quem tem mais ability tem MENOR custo de adaptação
        cost_of_effort = max(
            0, self.adaptation_cost_base - self.adaptation_cost_ability_factor * ability
        )

        utility_no_ai = expected_wage_no_ai
        utility_with_ai = expected_wage_with_ai - cost_of_effort

        return utility_no_ai, utility_with_ai

    def _assign_treatment_structural(self, ability: float, education: int) -> int:
        """v3.2 — Atribui tratamento com base num modelo de decisão racional.

        O agente escolhe adotar IA se a utilidade esperada com IA for maior
        do que sem IA, com algum ruído (bounded rationality).

        Referência: Heckman (1979) — a seleção é endógena e depende de
        variáveis não observáveis (ability) que também afetam o outcome.
        """
        utility_no_ai, utility_with_ai = self._calculate_expected_utility(ability, education)

        # O fator 2.0 controla a "racionalidade" do agente:
        # - Fator alto → decisão quase determinística
        # - Fator baixo → mais ruído (bounded rationality)
        prob_treatment = expit((utility_with_ai - utility_no_ai) * 1.0)
        return np.random.binomial(1, prob_treatment)

    def _assign_treatment_mechanical(self, ability: float, education: int) -> int:
        """v3.1 — Atribui tratamento com regra probabilística mecânica (baseline).

        Mantido para comparação de robustez.
        """
        prob = expit(0.5 * education + 1.5 * ability - 1)
        return np.random.binomial(1, prob)

    def _assign_treatment(self, ability: float, education: int) -> int:
        """Dispatcher: escolhe o modo de seleção configurado."""
        if self.selection_mode == "structural":
            return self._assign_treatment_structural(ability, education)
        else:
            return self._assign_treatment_mechanical(ability, education)

    # =========================================================================
    # Geração de Sequências e Outcomes (inalterados da v3.1)
    # =========================================================================

    def _generate_career_sequence(
        self, ability: float, education: int, treatment: int, treatment_time: int, n_periods: int
    ) -> list[int]:
        """Gera uma sequência de ocupações ao longo do tempo para um indivíduo."""
        sequence = []
        if education == 0:
            current_occ = np.random.choice(range(0, 20))
        elif education == 1:
            current_occ = np.random.choice(range(20, 35))
        else:
            current_occ = np.random.choice(range(35, 50))

        for t in range(n_periods):
            matrix = (
                self.transition_matrix_ai
                if treatment == 1 and t >= treatment_time
                else self.transition_matrix_baseline
            )
            probs = matrix[current_occ] * np.exp(0.3 * ability)
            probs /= probs.sum()

            next_occ = np.random.choice(self.n_occupations, p=probs)
            sequence.append(next_occ)
            current_occ = next_occ

        return sequence

    def _generate_outcome(
        self, static_features: pd.Series, treatment: int, ability: float
    ) -> tuple[float, float]:
        """Gera o resultado (salário) com base nos 5 mecanismos causais."""
        effect_x = np.dot(static_features[self.covariates].values.astype(float), self.beta_x)
        effect_a = self.gamma_a * ability
        cate = (
            self.true_ate
            + self.delta_e * static_features["education"]
            + self.delta_a * ability
        )
        effect_hte = cate * treatment
        effect_flex = self.lambda_f * treatment * static_features["flexicurity"]
        noise = np.random.normal(0, self.sigma_y)
        outcome = effect_x + effect_a + effect_hte + effect_flex + noise
        return outcome, cate

    def generate_panel_data(
        self, n_individuals: int = 1000, n_periods: int = 10
    ) -> pd.DataFrame:
        """Gera um DataFrame de painel completo, incluindo sequências de carreira.

        Inclui metadados sobre o modo de seleção para rastreabilidade.
        """
        X_static = self._generate_static_features(n_individuals)
        abilities = self._generate_latent_ability(X_static)

        panel_data = []
        for i in range(n_individuals):
            treatment = self._assign_treatment(abilities[i], X_static.loc[i, "education"])
            treatment_time = np.random.randint(2, n_periods - 2)

            career_sequence = self._generate_career_sequence(
                abilities[i],
                X_static.loc[i, "education"],
                treatment,
                treatment_time,
                n_periods,
            )

            for t in range(n_periods):
                outcome, true_cate = self._generate_outcome(
                    X_static.loc[i], treatment if t >= treatment_time else 0, abilities[i]
                )
                panel_data.append(
                    {
                        "individual_id": i,
                        "period": t,
                        "education": X_static.loc[i, "education"],
                        "flexicurity": X_static.loc[i, "flexicurity"],
                        "ability": abilities[i],
                        "treatment": treatment if t >= treatment_time else 0,
                        "outcome": outcome,
                        "true_cate": true_cate,
                        "occupation": career_sequence[t],
                        "career_sequence_history": career_sequence[: t + 1],
                        "selection_mode": self.selection_mode,
                    }
                )
        return pd.DataFrame(panel_data)

    def get_true_parameters(self) -> dict:
        """Retorna os parâmetros verdadeiros do DGP para validação.

        Útil para comparar estimativas com ground truth.
        """
        return {
            "true_ate": self.true_ate,
            "delta_e": self.delta_e,
            "delta_a": self.delta_a,
            "gamma_a": self.gamma_a,
            "lambda_f": self.lambda_f,
            "selection_mode": self.selection_mode,
            "adaptation_cost_base": self.adaptation_cost_base,
            "adaptation_cost_ability_factor": self.adaptation_cost_ability_factor,
        }
