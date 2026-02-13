"""
CAREER-DML: Data Generating Process (DGP) - v3.3 (HECKMAN STRUCTURAL + EXCLUSION)
Layer 1 - Synthetic data generator with known causal ground truth.

Evolution v3.2 → v3.3:
    - Added exclusion restriction variable (peer_adoption) that affects treatment
      selection but NOT the outcome. This provides a valid instrument for the
      Heckman two-step benchmark, making the comparison between DML and Heckman
      methodologically fair (Heckman, 1979: the exclusion restriction is the
      identifying assumption of the selection model).
    - Maintained both 'mechanical' and 'structural' selection modes.

References:
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

    v3.3: Adds exclusion restriction (peer_adoption) for valid Heckman benchmark.

    The exclusion restriction is a variable that:
        - Affects the probability of treatment (T) via social influence
        - Does NOT directly affect the outcome (Y)
        - Is observable to the econometrician

    This satisfies the identifying assumption of Heckman (1979) and allows
    the two-step estimator to operate under its intended conditions, making
    the comparison between DML and Heckman methodologically rigorous.

    Args:
        true_ate: Average Treatment Effect verdadeiro.
        delta_e: Moderação do efeito por educação.
        delta_a: Moderação do efeito por habilidade (capital humano).
        gamma_a: Efeito direto da habilidade no outcome.
        lambda_f: Efeito da interação tratamento x flexicurity.
        n_occupations: Número de ocupações no vocabulário de carreira.
        selection_mode: 'mechanical' (v3.1) ou 'structural' (Heckman v3.2+).
        adaptation_cost_base: Custo base de adaptação à IA (Nível 3 Heckman).
        adaptation_cost_ability_factor: Quanto a ability reduz o custo (Nível 3).
        peer_influence_strength: Strength of peer adoption effect on treatment (exclusion restriction).
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
        peer_influence_strength: float = 0.6,
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
        self.peer_influence_strength = peer_influence_strength
        self.beta_x = np.array([0.2, -0.1])
        self.sigma_y = 0.5
        self.covariates = ["education", "flexicurity"]

        # Career Transition Matrices
        self.transition_matrix_baseline = self._init_transition_matrix()
        self.transition_matrix_ai = self._init_ai_biased_matrix()

    def _init_transition_matrix(self) -> np.ndarray:
        """Create a stochastic Markov transition matrix."""
        matrix = np.random.dirichlet(np.ones(self.n_occupations), size=self.n_occupations)
        return matrix

    def _init_ai_biased_matrix(self) -> np.ndarray:
        """Create a transition matrix that favours 'tech' occupations."""
        matrix = self._init_transition_matrix()
        matrix[:, -10:] *= 1.5
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        return matrix

    def _generate_static_features(self, n_individuals: int) -> pd.DataFrame:
        """Generate time-invariant individual characteristics.

        Includes the exclusion restriction variable: peer_adoption.
        This represents the proportion of an individual's professional network
        that has already adopted AI tools. It affects treatment selection
        (via social influence / information diffusion) but NOT the outcome
        (your salary depends on YOUR skills, not on whether your peers use AI).
        """
        return pd.DataFrame(
            {
                "education": np.random.choice(
                    [0, 1, 2], size=n_individuals, p=[0.3, 0.5, 0.2]
                ),
                "flexicurity": np.random.uniform(0, 1, size=n_individuals),
                "peer_adoption": np.random.beta(2, 5, size=n_individuals),
            }
        )

    def _generate_latent_ability(self, X_static: pd.DataFrame) -> np.ndarray:
        """Generate the unobserved confounder (latent ability).

        Heckman (1979): This is the variable that causes selection bias.
        Cunha & Heckman (2007): Correlated with education (human capital).
        """
        return np.random.normal(X_static["education"] * 0.2, 1)

    # =========================================================================
    # LEVEL 3 HECKMAN: Rational Decision Model (Structural Selection)
    # =========================================================================

    def _calculate_expected_utility(self, ability: float, education: int, peer_adoption: float) -> tuple[float, float]:
        """Calculate expected utility of adopting vs. not adopting AI.

        This method implements a simplified rational decision model,
        inspired by the Heckman framework. The agent:
        1. Estimates future salary with and without AI.
        2. Calculates adaptation cost (inversely proportional to ability).
        3. Incorporates peer influence (exclusion restriction).
        4. Chooses treatment if net utility is positive.

        The peer_adoption variable enters ONLY the selection equation,
        satisfying the exclusion restriction required by Heckman (1979).

        Args:
            ability: Latent ability of the individual.
            education: Education level (0, 1, 2).
            peer_adoption: Proportion of peers who adopted AI (exclusion restriction).

        Returns:
            Tuple (utility_no_ai, utility_with_ai).
        """
        # Expected salary without AI (baseline)
        expected_wage_no_ai = 1.5 + 0.8 * education + self.gamma_a * ability

        # Expected salary with AI (includes true causal effect)
        expected_wage_with_ai = (
            expected_wage_no_ai
            + self.true_ate
            + self.delta_e * education
            + self.delta_a * ability
        )

        # Adaptation cost (Cunha & Heckman 2007: skill complementarity)
        cost_of_effort = max(
            0, self.adaptation_cost_base - self.adaptation_cost_ability_factor * ability
        )

        # Peer influence reduces perceived cost (information diffusion)
        # This is the exclusion restriction: affects T but NOT Y
        perceived_cost = cost_of_effort * (1 - self.peer_influence_strength * peer_adoption)

        utility_no_ai = expected_wage_no_ai
        utility_with_ai = expected_wage_with_ai - perceived_cost

        return utility_no_ai, utility_with_ai

    def _assign_treatment_structural(self, ability: float, education: int, peer_adoption: float) -> int:
        """v3.3 — Assign treatment based on rational decision model with exclusion restriction.

        The agent chooses to adopt AI if expected utility with AI exceeds
        utility without AI, with some noise (bounded rationality).

        Reference: Heckman (1979) — selection is endogenous and depends on
        unobservable variables (ability) that also affect the outcome,
        plus the exclusion restriction (peer_adoption) that only affects selection.
        """
        utility_no_ai, utility_with_ai = self._calculate_expected_utility(ability, education, peer_adoption)
        prob_treatment = expit((utility_with_ai - utility_no_ai) * 1.0)
        return np.random.binomial(1, prob_treatment)

    def _assign_treatment_mechanical(self, ability: float, education: int, peer_adoption: float) -> int:
        """v3.1 — Assign treatment with mechanical probabilistic rule (baseline).

        Maintained for robustness comparison. Peer adoption also enters here.
        """
        prob = expit(0.5 * education + 1.5 * ability - 1 + self.peer_influence_strength * peer_adoption)
        return np.random.binomial(1, prob)

    def _assign_treatment(self, ability: float, education: int, peer_adoption: float) -> int:
        """Dispatcher: selects the configured selection mode."""
        if self.selection_mode == "structural":
            return self._assign_treatment_structural(ability, education, peer_adoption)
        else:
            return self._assign_treatment_mechanical(ability, education, peer_adoption)

    # =========================================================================
    # Career Sequence and Outcome Generation
    # =========================================================================

    def _generate_career_sequence(
        self, ability: float, education: int, treatment: int, treatment_time: int, n_periods: int
    ) -> list[int]:
        """Generate a career occupation sequence over time for an individual."""
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
        """Generate outcome (salary) based on the 5 causal mechanisms.

        NOTE: peer_adoption does NOT enter the outcome equation.
        This is the exclusion restriction that identifies the Heckman model.
        """
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
        """Generate a complete panel DataFrame, including career sequences.

        Includes metadata about selection mode for traceability.
        """
        X_static = self._generate_static_features(n_individuals)
        abilities = self._generate_latent_ability(X_static)

        panel_data = []
        for i in range(n_individuals):
            treatment = self._assign_treatment(
                abilities[i], X_static.loc[i, "education"], X_static.loc[i, "peer_adoption"]
            )
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
                        "peer_adoption": X_static.loc[i, "peer_adoption"],
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
        """Return the true DGP parameters for validation.

        Useful for comparing estimates with ground truth.
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
            "peer_influence_strength": self.peer_influence_strength,
        }
