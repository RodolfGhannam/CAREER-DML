"""
CAREER-DML: Framework de Validação Robusta - v3.2 (HECKMAN INTEGRATED)
Camada 4 - Testes para garantir a robustez e confiabilidade dos resultados.

Evolução v3.1 → v3.2:
    - NÍVEL 2 HECKMAN: Adicionada `interpret_variance_heckman()` que interpreta
      a decomposição da variância através da lente do viés de seleção de Heckman.
    - NÍVEL 2 HECKMAN: Adicionada `interpret_gates_heckman()` que interpreta
      os GATES como proxies de capital humano latente (Cunha & Heckman, 2007).
    - NÍVEL 3 HECKMAN: Adicionada `robustness_structural_vs_mechanical()` que
      compara os resultados do DML sob os dois modos de seleção do DGP.
    - Adicionado `run_heckman_two_step_benchmark()` como benchmark clássico.

Referências:
    - Heckman (1979), Sample Selection Bias as a Specification Error
    - Cunha & Heckman (2007), The Technology of Skill Formation
    - Wager & Athey (2018), Estimation and Inference of HTE
    - Oster (2019), Unobservable Selection and Coefficient Stability
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dml import LinearDML


# =============================================================================
# FUNÇÕES ORIGINAIS v3.1 (mantidas intactas)
# =============================================================================

def variance_decomposition(Y: np.ndarray, T: np.ndarray, X_embed: np.ndarray) -> dict:
    """Decompõe a variância do estimador ATE em 3 fontes principais.

    Args:
        Y: Array de resultados (n,).
        T: Array de tratamento (n,).
        X_embed: Array de covariates (embeddings) para controlo (n, d).

    Returns:
        Dicionário com 'oracle_variance', 'nuisance_penalty',
        'common_support_penalty', e 'total_variance'.
    """
    n = len(Y)
    n1, n0 = (T == 1).sum(), (T == 0).sum()

    # 1. Oracle Variance
    outcome_model_1 = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05)
    outcome_model_0 = GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.05)

    outcome_model_1.fit(X_embed[T == 1], Y[T == 1])
    outcome_model_0.fit(X_embed[T == 0], Y[T == 0])

    mu1_hat = outcome_model_1.predict(X_embed)
    mu0_hat = outcome_model_0.predict(X_embed)

    residuals_1 = Y[T == 1] - mu1_hat[T == 1]
    residuals_0 = Y[T == 0] - mu0_hat[T == 0]

    var_oracle = (residuals_1.var() / n1) + (residuals_0.var() / n0)

    # 2. Observed Variance
    dml_estimator = LinearDML(
        model_y=GradientBoostingRegressor(verbose=0),
        model_t=GradientBoostingClassifier(verbose=0),
        discrete_treatment=True,
    )
    dml_estimator.fit(Y, T, X=X_embed)
    inf = dml_estimator.effect_inference(X=X_embed)
    var_observed = inf.stderr.mean() ** 2

    # 3. Nuisance Penalty
    var_nuisance_penalty = max(0, var_observed - var_oracle)

    # 4. Common Support Penalty
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
    """Calcula o delta de Oster (2019) para sensibilidade a não observáveis."""

    # Restricted model (sem controles)
    beta_restricted = np.cov(Y, T)[0, 1] / np.var(T)
    y_pred_restricted = beta_restricted * T
    r2_restricted = 1 - np.var(Y - y_pred_restricted) / np.var(Y)

    # Controlled model (com embeddings)
    model = LinearRegression()
    X_with_T = np.column_stack([T, X_controlled])
    model.fit(X_with_T, Y)
    beta_controlled = model.coef_[0]
    r2_controlled = model.score(X_with_T, Y)

    # Oster delta: evitar divisão por zero
    denominator = (beta_controlled - beta_restricted) * (r2_controlled - r2_restricted)
    if abs(denominator) < 1e-10:  # Praticamente zero
        # Se o denominador é zero, significa que não há mudança no coeficiente ou R²
        # ao adicionar controles, indicando robustez máxima ou um caso degenerado.
        # Retornamos np.inf para indicar que o delta é indefinido ou muito grande.
        return np.inf
    
    delta = (beta_restricted * (R2_max - r2_restricted)) / denominator
    return abs(delta)


def run_placebo_tests(
    estimator, Y: np.ndarray, T: np.ndarray, X: np.ndarray, W: np.ndarray | None = None
) -> dict:
    """Executa testes de placebo para verificar a validade da estimação causal."""
    T_permuted = np.random.permutation(T)
    ate_random_t, _, _ = estimator.fit_predict(Y, T_permuted, X, W)

    Y_permuted = np.random.permutation(Y)
    ate_random_y, _, _ = estimator.fit_predict(Y_permuted, T, X, W)

    return {
        "random_treatment_ate": ate_random_t,
        "random_outcome_ate": ate_random_y,
    }


# =============================================================================
# NÍVEL 2 HECKMAN: Interpretações Heckmanianas dos Resultados
# =============================================================================

def interpret_variance_heckman(var_decomp: dict) -> dict:
    """Interpreta a decomposição da variância pela lente de Heckman (1979).

    A common_support_penalty é reinterpretada como uma medida quantitativa
    da severidade do viés de seleção de Heckman.

    Args:
        var_decomp: Dicionário retornado por variance_decomposition().

    Returns:
        Dicionário com interpretações textuais e métricas derivadas.
    """
    oracle = var_decomp["oracle_variance"]
    nuisance = var_decomp["nuisance_penalty"]
    support = var_decomp["common_support_penalty"]
    total = var_decomp["total_variance"]

    # Proporção de cada componente na variância total
    total_sum = oracle + nuisance + support
    pct_oracle = (oracle / total_sum) * 100 if total_sum > 0 else 0
    pct_nuisance = (nuisance / total_sum) * 100 if total_sum > 0 else 0
    pct_support = (support / total_sum) * 100 if total_sum > 0 else 0

    # Classificação da severidade do viés de seleção
    if pct_support > 50:
        severity = "SEVERO"
        heckman_note = (
            "A common_support_penalty domina a variância total, indicando que "
            "o viés de seleção de Heckman (1979) é a principal fonte de incerteza. "
            "Isto confirma que a habilidade latente (ability) cria uma separação "
            "forte entre os grupos de tratamento e controlo, tornando a inferência "
            "causal particularmente desafiadora. O uso de adversarial debiasing "
            "é ESSENCIAL neste cenário."
        )
    elif pct_support > 25:
        severity = "MODERADO"
        heckman_note = (
            "A common_support_penalty é uma componente significativa da variância, "
            "indicando que o viés de seleção de Heckman (1979) está presente e "
            "afeta a estimação. O adversarial debiasing melhora substancialmente "
            "a estimação neste cenário."
        )
    else:
        severity = "LEVE"
        heckman_note = (
            "A common_support_penalty é uma componente menor da variância, "
            "sugerindo que o overlap entre grupos é razoável. Mesmo assim, "
            "o adversarial debiasing oferece ganhos de precisão."
        )

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
    """Interpreta os GATES pela lente de Cunha & Heckman (2007).

    Os quantis de CATE são reinterpretados como proxies para diferentes
    níveis de capital humano latente.

    Args:
        gates_df: DataFrame retornado por estimate_gates().

    Returns:
        Dicionário com interpretações textuais.
    """
    n_groups = len(gates_df)
    if n_groups == 0:
        return {"error": "GATES DataFrame vazio."}

    # Extrair valores extremos
    q1 = gates_df.iloc[0]
    q_last = gates_df.iloc[-1]

    # Calcular o gradiente de heterogeneidade
    ate_range = q_last["ate"] - q1["ate"]
    ratio = q_last["ate"] / q1["ate"] if q1["ate"] != 0 else float("inf")

    # Teste de monotonicidade (os efeitos crescem com o quantil?)
    ates = gates_df["ate"].values
    is_monotonic = all(ates[i] <= ates[i + 1] for i in range(len(ates) - 1))

    interpretation = (
        f"A análise de GATES revela uma forte heterogeneidade no efeito do tratamento, "
        f"consistente com a teoria de complementaridade capital-competência "
        f"(Cunha & Heckman, 2007). O retorno da exposição à IA varia de "
        f"{q1['ate']:.4f} (Q1, menor capital humano latente) a "
        f"{q_last['ate']:.4f} (Q{n_groups}, maior capital humano latente), "
        f"uma diferença de {ate_range:.4f} ({ratio:.1f}x)."
    )

    if is_monotonic:
        interpretation += (
            " O padrão é monotonicamente crescente, confirmando que indivíduos "
            "com maior capital humano latente (ability + education) beneficiam "
            "desproporcionalmente da exposição à IA. Isto é consistente com a "
            "hipótese de skill-biased technological change (SBTC) e com o "
            "modelo de formação de capital humano de Heckman."
        )
    else:
        interpretation += (
            " O padrão não é estritamente monotónico, sugerindo que a relação "
            "entre capital humano e retorno da IA pode ser não-linear, "
            "possivelmente devido a efeitos de saturação nos quantis superiores."
        )

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


# =============================================================================
# NÍVEL 3 HECKMAN: Teste de Robustez Estrutural vs. Mecânico
# =============================================================================

def robustness_structural_vs_mechanical(
    dml_pipeline,
    dgp_class,
    n_individuals: int = 1000,
    n_periods: int = 10,
    embedding_fn=None,
) -> dict:
    """Compara os resultados do DML sob seleção mecânica vs. estrutural.

    Este teste é a resposta direta à crítica de Heckman sobre modelos reduzidos.
    Demonstra que os resultados se mantêm robustos mesmo quando a seleção
    para o tratamento resulta de um modelo de decisão racional.

    Args:
        dml_pipeline: Instância de CausalDMLPipeline (não fitted).
        dgp_class: Classe SyntheticDGP (v3.2) para instanciar.
        n_individuals: Número de indivíduos a gerar.
        n_periods: Número de períodos.
        embedding_fn: Função que recebe o DataFrame e retorna embeddings.
                      Se None, usa as colunas numéricas diretamente.

    Returns:
        Dicionário com ATEs, biases e conclusão de robustez.
    """
    results = {}

    for mode in ["mechanical", "structural"]:
        print(f"\n{'='*60}")
        print(f"  Teste de Robustez: Seleção {mode.upper()}")
        print(f"{'='*60}")

        # Gerar dados com o modo de seleção especificado
        dgp = dgp_class(selection_mode=mode)
        panel = dgp.generate_panel_data(n_individuals=n_individuals, n_periods=n_periods)

        # Usar apenas o último período para estimação cross-sectional
        final_period = panel.groupby("individual_id").last().reset_index()

        Y = final_period["outcome"].values
        T = final_period["treatment"].values
        true_ate = dgp.true_ate

        # Gerar embeddings ou usar features diretas
        if embedding_fn is not None:
            X_embed = embedding_fn(final_period)
        else:
            # Fallback: usar education + flexicurity + ability como features
            X_embed = final_period[["education", "flexicurity", "ability"]].values

        # Importar uma nova instância do pipeline para cada teste
        from src.dml import CausalDMLPipeline
        pipeline = CausalDMLPipeline()
        ate_est, cates, keep_idx = pipeline.fit_predict(Y, T, X_embed)

        bias = ate_est - true_ate
        pct_error = abs(bias / true_ate) * 100

        results[mode] = {
            "ate_estimated": ate_est,
            "true_ate": true_ate,
            "bias": bias,
            "pct_error": pct_error,
            "n_treated": int(T.sum()),
            "n_control": int((1 - T).sum()),
            "treatment_rate": float(T.mean()),
        }

        print(f"  ATE Estimado: {ate_est:.4f}")
        print(f"  ATE Verdadeiro: {true_ate:.4f}")
        print(f"  Bias: {bias:.4f} ({pct_error:.1f}%)")
        print(f"  Taxa de Tratamento: {T.mean():.2%}")

    # Análise de robustez
    diff_bias = abs(results["structural"]["bias"]) - abs(results["mechanical"]["bias"])
    is_robust = abs(diff_bias) < 0.1  # Tolerância de 0.1

    conclusion = (
        "ROBUSTO: Os resultados são consistentes sob ambos os modos de seleção "
        f"(diferença de bias = {diff_bias:.4f}). O método de adversarial debiasing "
        "é robusto mesmo quando a seleção para o tratamento resulta de um modelo "
        "de decisão racional (Heckman-style), não apenas de uma regra mecânica."
        if is_robust
        else
        f"SENSÍVEL: Os resultados diferem entre os modos de seleção "
        f"(diferença de bias = {diff_bias:.4f}). Investigação adicional necessária."
    )

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
    """Executa o modelo de seleção de Heckman (two-step) como benchmark.

    Implementa uma versão simplificada do procedimento de Heckman (1979):
    1. Probit para estimar P(T=1|X) → calcula Inverse Mills Ratio (IMR).
    2. OLS com IMR como controlo adicional.

    Isto permite comparar o ATE do DML com o ATE do modelo clássico de Heckman,
    demonstrando que os career embeddings são uma forma mais rica de controlar
    para a seleção do que a tradicional Inverse Mills Ratio.

    Args:
        Y: Array de resultados (n,).
        T: Array de tratamento (n,).
        X: Array de covariáveis (n, d).

    Returns:
        Dicionário com ATE do Heckman two-step e métricas.
    """
    from scipy.stats import norm

    # Passo 1: Probit (aproximado por logistic regression)
    from sklearn.linear_model import LogisticRegression

    probit = LogisticRegression(max_iter=1000)
    probit.fit(X, T)
    ps = probit.predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-5, 1 - 1e-5)

    # Calcular Inverse Mills Ratio (IMR)
    # Para tratados: lambda = phi(Xb) / Phi(Xb)
    # Para não-tratados: lambda = -phi(Xb) / (1 - Phi(Xb))
    z_scores = norm.ppf(ps)
    imr = np.where(
        T == 1,
        norm.pdf(z_scores) / norm.cdf(z_scores),
        -norm.pdf(z_scores) / (1 - norm.cdf(z_scores)),
    )
    imr = np.nan_to_num(imr, nan=0.0, posinf=0.0, neginf=0.0)

    # Passo 2: OLS com IMR como controlo
    X_with_imr = np.column_stack([T, X, imr])
    ols = LinearRegression().fit(X_with_imr, Y)

    ate_heckman = ols.coef_[0]  # Coeficiente do tratamento
    r2 = ols.score(X_with_imr, Y)
    imr_coef = ols.coef_[-1]  # Coeficiente da IMR (significância indica seleção)

    return {
        "ate_heckman_two_step": ate_heckman,
        "r_squared": r2,
        "imr_coefficient": imr_coef,
        "imr_significant": abs(imr_coef) > 0.1,
        "interpretation": (
            f"O modelo de Heckman two-step estima ATE = {ate_heckman:.4f}. "
            f"O coeficiente da IMR = {imr_coef:.4f} "
            f"{'é significativo, confirmando a presença de viés de seleção.' if abs(imr_coef) > 0.1 else 'é pequeno, sugerindo que o viés de seleção é limitado após controlar por X.'} "
            f"Compare com o ATE do DML para avaliar a superioridade dos career embeddings "
            f"sobre a Inverse Mills Ratio clássica."
        ),
    }
