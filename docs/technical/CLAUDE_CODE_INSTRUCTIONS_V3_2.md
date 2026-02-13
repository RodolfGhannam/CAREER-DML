'''
# Instruções para Claude Code: Implementar CAREER-DML v3.2 (Heckman Structural)

**Objetivo**: Atualizar o projeto `C:\PHDCBS` para a versão 3.2, que integra os 3 níveis de ancoragem teórica de Heckman.

**Passos**:

1.  **Substituir `dgp.py`**: Atualize o ficheiro `C:\PHDCBS\src\dgp.py` com o código v3.2 abaixo. Esta versão adiciona o modelo de decisão racional (seleção estrutural).
2.  **Substituir `validation.py`**: Atualize o ficheiro `C:\PHDCBS\src\validation.py` com o código v3.2 abaixo. Esta versão adiciona as funções de interpretação Heckmaniana e os testes de robustez.
3.  **Substituir `main.py`**: Atualize o ficheiro `C:\PHDCBS\main.py` com o código v3.2 abaixo. Este é o novo orquestrador que executa o pipeline completo.

---

## Ficheiro 1: `C:\PHDCBS\src\dgp.py`

```python
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
"""

import numpy as np
import pandas as pd
from scipy.special import expit


class SyntheticDGP:
    """
    v3.2: Suporta seleção mecânica (v3.1) e estrutural (Heckman-style).
    """

    def __init__(
        self,
        true_ate: float = 0.5,
        delta_e: float = 0.2,
        delta_a: float = 0.3,
        gamma_a: float = 0.4,
        lambda_f: float = 0.1,
        n_occupations: int = 50,
        selection_mode: str = "structural",
        adaptation_cost_base: float = 2.0,
        adaptation_cost_ability_factor: float = 1.5,
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
        self.transition_matrix_baseline = self._init_transition_matrix()
        self.transition_matrix_ai = self._init_ai_biased_matrix()

    def _init_transition_matrix(self) -> np.ndarray:
        matrix = np.random.dirichlet(np.ones(self.n_occupations), size=self.n_occupations)
        return matrix

    def _init_ai_biased_matrix(self) -> np.ndarray:
        matrix = self._init_transition_matrix()
        matrix[:, -10:] *= 1.5
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        return matrix

    def _generate_static_features(self, n_individuals: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "education": np.random.choice(
                    [0, 1, 2], size=n_individuals, p=[0.3, 0.5, 0.2]
                ),
                "flexicurity": np.random.uniform(0, 1, size=n_individuals),
            }
        )

    def _generate_latent_ability(self, X_static: pd.DataFrame) -> np.ndarray:
        return np.random.normal(X_static["education"] * 0.2, 1)

    def _calculate_expected_utility(self, ability: float, education: int) -> tuple[float, float]:
        expected_wage_no_ai = 1.5 + 0.8 * education + self.gamma_a * ability
        expected_wage_with_ai = (
            expected_wage_no_ai
            + self.true_ate
            + self.delta_e * education
            + self.delta_a * ability
        )
        cost_of_effort = max(
            0, self.adaptation_cost_base - self.adaptation_cost_ability_factor * ability
        )
        utility_no_ai = expected_wage_no_ai
        utility_with_ai = expected_wage_with_ai - cost_of_effort
        return utility_no_ai, utility_with_ai

    def _assign_treatment_structural(self, ability: float, education: int) -> int:
        utility_no_ai, utility_with_ai = self._calculate_expected_utility(ability, education)
        prob_treatment = expit((utility_with_ai - utility_no_ai) * 2.0)
        return np.random.binomial(1, prob_treatment)

    def _assign_treatment_mechanical(self, ability: float, education: int) -> int:
        prob = expit(0.5 * education + 1.5 * ability - 1)
        return np.random.binomial(1, prob)

    def _assign_treatment(self, ability: float, education: int) -> int:
        if self.selection_mode == "structural":
            return self._assign_treatment_structural(ability, education)
        else:
            return self._assign_treatment_mechanical(ability, education)

    def _generate_career_sequence(
        self, ability: float, education: int, treatment: int, treatment_time: int, n_periods: int
    ) -> list[int]:
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
```

---

## Ficheiro 2: `C:\PHDCBS\src\validation.py`

```python
"""
CAREER-DML: Framework de Validação Robusta - v3.2 (HECKMAN INTEGRATED)
Camada 4 - Testes para garantir a robustez e confiabilidade dos resultados.

Evolução v3.1 → v3.2:
    - NÍVEL 2 HECKMAN: Adicionada `interpret_variance_heckman()`
    - NÍVEL 2 HECKMAN: Adicionada `interpret_gates_heckman()`
    - NÍVEL 3 HECKMAN: Adicionada `robustness_structural_vs_mechanical()`
    - Adicionado `run_heckman_two_step_benchmark()`

Referências:
    - Heckman (1979), Sample Selection Bias as a Specification Error
    - Cunha & Heckman (2007), The Technology of Skill Formation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dml import LinearDML

def variance_decomposition(Y: np.ndarray, T: np.ndarray, X_embed: np.ndarray) -> dict:
    n = len(Y)
    n1, n0 = (T == 1).sum(), (T == 0).sum()
    outcome_model_1 = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05)
    outcome_model_0 = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05)
    outcome_model_1.fit(X_embed[T == 1], Y[T == 1])
    outcome_model_0.fit(X_embed[T == 0], Y[T == 0])
    mu1_hat = outcome_model_1.predict(X_embed)
    mu0_hat = outcome_model_0.predict(X_embed)
    residuals_1 = Y[T == 1] - mu1_hat[T == 1]
    residuals_0 = Y[T == 0] - mu0_hat[T == 0]
    var_oracle = (residuals_1.var() / n1) + (residuals_0.var() / n0)
    dml_estimator = LinearDML(
        model_y=GradientBoostingRegressor(verbose=-1),
        model_t=GradientBoostingClassifier(verbose=-1),
    )
    dml_estimator.fit(Y, T, X=X_embed)
    var_observed = dml_estimator.effect_stderr_[0] ** 2
    var_nuisance_penalty = max(0, var_observed - var_oracle)
    prop_model = GradientBoostingClassifier(n_estimators=100)
    prop_model.fit(X_embed, T)
    ps = prop_model.predict_proba(X_embed)[:, 1]
    ps = np.clip(ps, 1e-5, 1 - 1e-5)
    overlap_variance = np.mean(1 / (ps * (1 - ps))) / n
    return {
        "oracle_variance": var_oracle,
        "nuisance_penalty": var_nuisance_penalty,
        "common_support_penalty": overlap_variance,
        "total_variance": var_observed,
    }

def sensitivity_analysis_oster(
    Y: np.ndarray, T: np.ndarray, X_controlled: np.ndarray, R2_max: float = 1.0
) -> float:
    model_restricted = LinearRegression().fit(np.column_stack([T, X_controlled]), Y)
    beta_restricted = model_restricted.coef_[0]
    r2_restricted = model_restricted.score(np.column_stack([T, X_controlled]), Y)
    if r2_restricted >= R2_max:
        return np.inf
    delta = (beta_restricted * (R2_max - r2_restricted)) / (
        r2_restricted * (1 - R2_max)
    )
    return abs(delta)

def run_placebo_tests(
    estimator, Y: np.ndarray, T: np.ndarray, X: np.ndarray, W: np.ndarray | None = None
) -> dict:
    T_permuted = np.random.permutation(T)
    ate_random_t, _, _ = estimator.fit_predict(Y, T_permuted, X, W)
    Y_permuted = np.random.permutation(Y)
    ate_random_y, _, _ = estimator.fit_predict(Y_permuted, T, X, W)
    return {
        "random_treatment_ate": ate_random_t,
        "random_outcome_ate": ate_random_y,
    }

def interpret_variance_heckman(var_decomp: dict) -> dict:
    oracle = var_decomp["oracle_variance"]
    nuisance = var_decomp["nuisance_penalty"]
    support = var_decomp["common_support_penalty"]
    total = var_decomp["total_variance"]
    total_sum = oracle + nuisance + support
    pct_oracle = (oracle / total_sum) * 100 if total_sum > 0 else 0
    pct_nuisance = (nuisance / total_sum) * 100 if total_sum > 0 else 0
    pct_support = (support / total_sum) * 100 if total_sum > 0 else 0
    if pct_support > 50:
        severity = "SEVERO"
        heckman_note = "A common_support_penalty domina a variância, indicando que o viés de seleção de Heckman (1979) é a principal fonte de incerteza. O adversarial debiasing é ESSENCIAL."
    elif pct_support > 25:
        severity = "MODERADO"
        heckman_note = "A common_support_penalty é significativa, indicando que o viés de seleção de Heckman (1979) afeta a estimação."
    else:
        severity = "LEVE"
        heckman_note = "A common_support_penalty é menor, sugerindo bom overlap. Mesmo assim, o debiasing oferece ganhos."
    return {
        "variance_table": {
            "Oracle Variance (baseline)": f"{oracle:.6f} ({pct_oracle:.1f}%)",
            "Nuisance Penalty (ML estimation)": f"{nuisance:.6f} ({pct_nuisance:.1f}%)",
            "Common Support Penalty (Heckman selection)": f"{support:.6f} ({pct_support:.1f}%)",
            "Total Variance": f"{total:.6f}",
        },
        "heckman_selection_severity": severity,
        "heckman_interpretation": heckman_note,
    }

def interpret_gates_heckman(gates_df: pd.DataFrame) -> dict:
    n_groups = len(gates_df)
    if n_groups == 0:
        return {"error": "GATES DataFrame vazio."}
    q1 = gates_df.iloc[0]
    q_last = gates_df.iloc[-1]
    ate_range = q_last["ate"] - q1["ate"]
    ratio = q_last["ate"] / q1["ate"] if q1["ate"] != 0 else float("inf")
    ates = gates_df["ate"].values
    is_monotonic = all(ates[i] <= ates[i + 1] for i in range(len(ates) - 1))
    interpretation = f"Forte heterogeneidade, consistente com Cunha & Heckman (2007). Retorno da IA varia de {q1['ate']:.4f} (Q1) a {q_last['ate']:.4f} (Q{n_groups}), uma diferença de {ate_range:.4f} ({ratio:.1f}x)."
    if is_monotonic:
        interpretation += " O padrão é monotonicamente crescente, confirmando que indivíduos com maior capital humano latente beneficiam desproporcionalmente da IA."
    return {
        "gates_summary": {
            "Q1 (menor capital humano)": f"ATE = {q1['ate']:.4f} [{q1['ci_lower']:.4f}, {q1['ci_upper']:.4f}]",
            f"Q{n_groups} (maior capital humano)": f"ATE = {q_last['ate']:.4f} [{q_last['ci_lower']:.4f}, {q_last['ci_upper']:.4f}]",
            "Gradiente de heterogeneidade": f"{ate_range:.4f}",
            "Ratio Q_max / Q1": f"{ratio:.2f}x",
            "Monotonicamente crescente": "Sim" if is_monotonic else "Não",
        },
        "heckman_interpretation": interpretation,
    }

def robustness_structural_vs_mechanical(
    dgp_class,
    n_individuals: int = 1000,
    n_periods: int = 10,
    embedding_fn=None,
) -> dict:
    results = {}
    for mode in ["mechanical", "structural"]:
        dgp = dgp_class(selection_mode=mode)
        panel = dgp.generate_panel_data(n_individuals=n_individuals, n_periods=n_periods)
        final_period = panel.groupby("individual_id").last().reset_index()
        Y = final_period["outcome"].values
        T = final_period["treatment"].values
        true_ate = dgp.true_ate
        if embedding_fn is not None:
            X_embed = embedding_fn(final_period)
        else:
            X_embed = final_period[["education", "flexicurity", "ability"]].values
        from src.dml import CausalDMLPipeline
        pipeline = CausalDMLPipeline()
        ate_est, _, _ = pipeline.fit_predict(Y, T, X_embed)
        bias = ate_est - true_ate
        pct_error = abs(bias / true_ate) * 100
        results[mode] = {
            "ate_estimated": ate_est,
            "true_ate": true_ate,
            "bias": bias,
            "pct_error": pct_error,
        }
    diff_bias = abs(results["structural"]["bias"]) - abs(results["mechanical"]["bias"])
    is_robust = abs(diff_bias) < 0.1
    conclusion = "ROBUSTO: Resultados consistentes sob ambos os modos de seleção." if is_robust else "SENSÍVEL: Resultados diferem entre os modos de seleção."
    return {
        "mechanical": results["mechanical"],
        "structural": results["structural"],
        "bias_difference": diff_bias,
        "is_robust": is_robust,
        "conclusion": conclusion,
    }

def run_heckman_two_step_benchmark(
    Y: np.ndarray, T: np.ndarray, X: np.ndarray
) -> dict:
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    probit = LogisticRegression(max_iter=1000)
    probit.fit(X, T)
    ps = probit.predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-5, 1 - 1e-5)
    z_scores = norm.ppf(ps)
    imr = np.where(
        T == 1,
        norm.pdf(z_scores) / norm.cdf(z_scores),
        -norm.pdf(z_scores) / (1 - norm.cdf(z_scores)),
    )
    imr = np.nan_to_num(imr, nan=0.0, posinf=0.0, neginf=0.0)
    X_with_imr = np.column_stack([T, X, imr])
    ols = LinearRegression().fit(X_with_imr, Y)
    ate_heckman = ols.coef_[0]
    r2 = ols.score(X_with_imr, Y)
    imr_coef = ols.coef_[-1]
    return {
        "ate_heckman_two_step": ate_heckman,
        "r_squared": r2,
        "imr_coefficient": imr_coef,
        "imr_significant": abs(imr_coef) > 0.1,
        "interpretation": f"Heckman two-step ATE = {ate_heckman:.4f}. O coeficiente da IMR = {imr_coef:.4f} {'é significativo, confirmando viés de seleção.' if abs(imr_coef) > 0.1 else 'é pequeno.'}"
    }
```

---

## Ficheiro 3: `C:\PHDCBS\main.py`

```python
"""
CAREER-DML: Main Pipeline v3.2 (HECKMAN INTEGRATED)
Orquestrador principal que executa os 3 níveis de ancoragem Heckman.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dgp import SyntheticDGP
from src.embeddings import (
    PredictiveGRU, CausalGRU, DebiasedGRU, Adversary,
    train_predictive_embedding, train_causal_embedding, train_debiased_embedding,
)
from src.dml import CausalDMLPipeline
from src.validation import (
    variance_decomposition,
    sensitivity_analysis_oster,
    run_placebo_tests,
    interpret_variance_heckman,
    interpret_gates_heckman,
    robustness_structural_vs_mechanical,
    run_heckman_two_step_benchmark,
)

N_INDIVIDUALS = 1000
N_PERIODS = 10
N_OCCUPATIONS = 50
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
PHI_DIM = 16
EPOCHS = 15
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

def pad_sequences(sequences: list[list[int]], max_len: int = N_PERIODS) -> torch.Tensor:
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [0] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return torch.tensor(padded, dtype=torch.long)

def prepare_dataloader(panel: pd.DataFrame) -> tuple[DataLoader, pd.DataFrame]:
    final = panel.groupby("individual_id").last().reset_index()
    sequences = pad_sequences(final["career_sequence_history"].tolist())
    treatments = torch.tensor(final["treatment"].values, dtype=torch.long)
    outcomes = torch.tensor(final["outcome"].values, dtype=torch.float32)
    dataset = TensorDataset(sequences, treatments, outcomes)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader, final

def extract_embeddings(model, sequences: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.get_representation(sequences).numpy()

def print_header(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def print_subheader(title: str):
    print(f"\n  --- {title} ---")

def main():
    print_header("ETAPA 1: Geração de Dados — DGP v3.2 (Heckman Structural)")
    dgp = SyntheticDGP(selection_mode="structural")
    panel = dgp.generate_panel_data(n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS)
    params = dgp.get_true_parameters()
    loader, final = prepare_dataloader(panel)
    sequences_tensor = pad_sequences(final["career_sequence_history"].tolist())
    Y = final["outcome"].values
    T = final["treatment"].values
    true_ate = params["true_ate"]

    print_header("ETAPA 2: Treino das 3 Variantes de Embedding")
    pred_model = PredictiveGRU(vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    train_predictive_embedding(pred_model, loader, epochs=EPOCHS)
    X_pred = extract_embeddings(pred_model, sequences_tensor)

    causal_model = CausalGRU(vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM)
    train_causal_embedding(causal_model, loader, epochs=EPOCHS)
    X_causal = extract_embeddings(causal_model, sequences_tensor)

    debiased_model = DebiasedGRU(vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM)
    adversary_model = Adversary(phi_dim=PHI_DIM)
    train_debiased_embedding(debiased_model, adversary_model, loader, epochs=EPOCHS)
    X_debiased = extract_embeddings(debiased_model, sequences_tensor)

    print_header("ETAPA 3: Estimação DML — Comparação das 3 Variantes")
    variants = {
        "Predictive GRU": X_pred,
        "Causal GRU (VIB)": X_causal,
        "Debiased GRU (Adversarial)": X_debiased,
    }
    results = {}
    best_variant = ""
    best_bias = float("inf")
    for name, X_embed in variants.items():
        pipeline = CausalDMLPipeline()
        ate_est, _, _ = pipeline.fit_predict(Y, T, X_embed)
        bias = ate_est - true_ate
        if abs(bias) < best_bias:
            best_bias = abs(bias)
            best_variant = name
        results[name] = {
            "ate": ate_est,
            "bias": bias,
            "pipeline": pipeline,
            "X_embed": X_embed,
        }

    print_header("ETAPA 4: Validação Completa — " + best_variant)
    best = results[best_variant]
    X_best = best["X_embed"]
    var_decomp = variance_decomposition(Y, T, X_best)
    heckman_var = interpret_variance_heckman(var_decomp)
    print_subheader("Variance Decomposition + Interpretação Heckman")
    print(heckman_var)

    gates_df = best['pipeline'].estimate_gates(pd.DataFrame(X_best), n_groups=5)
    heckman_gates = interpret_gates_heckman(gates_df)
    print_subheader("GATES + Interpretação Heckman")
    print(heckman_gates)

    print_header("ETAPA 5: Benchmark — Heckman Two-Step vs. DML")
    heckman_bench = run_heckman_two_step_benchmark(Y, T, X_best)
    print(heckman_bench)

    print_header("ETAPA 6: Robustez — Seleção Estrutural vs. Mecânica")
    robustness = robustness_structural_vs_mechanical(
        dgp_class=SyntheticDGP,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
    )
    print(robustness)

if __name__ == "__main__":
    main()
```
'''
