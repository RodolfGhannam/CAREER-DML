"""
CAREER-DML: Semi-Synthetic Pipeline
====================================
Runs the full CAREER-DML pipeline using the Semi-Synthetic DGP calibrated
with real-world data from NLSY79 (Mincer wage equation) and Felten et al.
(2021) AI Occupational Exposure Index.

This script mirrors the structure of main.py but uses SemiSyntheticDGP
instead of SyntheticDGP. The core modules (embeddings, DML, validation)
are NOT modified — only the data generation step changes.

Key differences from main.py:
    1. Data comes from SemiSyntheticDGP (calibrated with real parameters)
    2. Continuous AIOE scores are discretized into occupation bins for GRU input
    3. Cross-sectional data (no panel groupby needed)
    4. Heckman benchmark runs without exclusion restriction (no peer_adoption)
    5. Robustness structural vs. mechanical test is skipped (not applicable)

Pipeline Steps:
    1. Generate semi-synthetic data (NLSY79 + Felten AIOE calibration)
    2. Train 3 embedding variants (Predictive, Causal VIB, Debiased Adversarial)
    3. DML estimation with each variant (with SE, CI, p-values)
    4. Complete validation (variance decomposition, GATES, Oster, placebo)
    5. VIB sensitivity analysis (beta sweep)
    6. Heckman two-step benchmark (without exclusion restriction)
    7. Print comparative results

References:
    - Mincer, J. (1974). Schooling, Experience, and Earnings.
    - Felten, E., Raj, M., & Seamans, R. (2021). Occupational, industry, and
      geographic exposure to AI. Strategic Management Journal, 42(12), 2195-2217.
    - Chernozhukov et al. (2018). Double/Debiased ML.
    - Veitch et al. (2020). Adapting Text Embeddings for Causal Inference.
    - Heckman (1979). Sample Selection Bias as a Specification Error.

Author: Rodolf Mikel Ghannam Neto
Date: February 2026
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# Project imports (UNCHANGED from main.py)
from src.semi_synthetic_dgp import SemiSyntheticDGP
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
    run_heckman_two_step_benchmark,
    vib_sensitivity_analysis,
)

# =============================================================================
# CONFIGURATION (same as main.py for comparability)
# =============================================================================
N_INDIVIDUALS = 1000
N_PERIODS = 10          # Match original pipeline sequence length
N_OCCUPATIONS = 50      # Discretization bins for AIOE scores
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
PHI_DIM = 16
EPOCHS = 15
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def discretize_aioe_to_occupations(career_aioe: np.ndarray, n_bins: int = N_OCCUPATIONS) -> np.ndarray:
    """Convert continuous AIOE scores to discrete occupation IDs.

    The GRU embedding models expect integer occupation IDs (for nn.Embedding).
    We discretize the continuous AIOE scores into n_bins equal-width bins.

    Args:
        career_aioe: Array of shape (N, T) with continuous AIOE scores.
        n_bins: Number of discrete occupation bins.

    Returns:
        Array of shape (N, T) with integer occupation IDs in [0, n_bins-1].
    """
    # Use percentile-based bins for better distribution
    flat = career_aioe.flatten()
    bin_edges = np.percentile(flat, np.linspace(0, 100, n_bins + 1))
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < n_bins + 1:
        # If not enough unique percentiles, use linspace
        bin_edges = np.linspace(flat.min() - 0.001, flat.max() + 0.001, n_bins + 1)

    career_discrete = np.digitize(career_aioe, bin_edges) - 1
    career_discrete = np.clip(career_discrete, 0, n_bins - 1)
    return career_discrete


def pad_sequences(sequences: list[list[int]], max_len: int = N_PERIODS) -> torch.Tensor:
    """Pad sequences with zeros to uniform length (same as main.py)."""
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [0] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return torch.tensor(padded, dtype=torch.long)


def extract_embeddings(model, sequences: torch.Tensor) -> np.ndarray:
    """Extract embeddings from a trained model (same as main.py)."""
    model.eval()
    with torch.no_grad():
        return model.get_representation(sequences).numpy()


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subheader(title: str):
    """Print a formatted sub-header."""
    print(f"\n  --- {title} ---")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    # =========================================================================
    # STEP 1: Generate Semi-Synthetic Data (NLSY79 + Felten AIOE)
    # =========================================================================
    print_header("STEP 1: Data Generation -- Semi-Synthetic DGP (NLSY79 + Felten AIOE)")

    dgp = SemiSyntheticDGP(n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED)
    data = dgp.generate()

    # Extract arrays
    career_aioe = data["career_sequences"]      # (N, T), continuous AIOE scores
    Y = data["Y"]                                # (N,), log wages
    T = data["T_treatment"].astype(int)           # (N,), binary treatment
    true_ate = data["true_ate"]                   # scalar
    propensity = data["propensity"]               # (N,), true propensity scores
    hte = data["hte"]                             # (N,), heterogeneous treatment effects
    covariates = data["covariates"]               # dict of covariate arrays

    # Discretize continuous AIOE scores into occupation IDs for GRU
    career_discrete = discretize_aioe_to_occupations(career_aioe, N_OCCUPATIONS)
    sequences = [career_discrete[i].tolist() for i in range(len(career_discrete))]
    sequences_tensor = pad_sequences(sequences, max_len=N_PERIODS)

    # Create DataLoader (cross-sectional, no panel groupby needed)
    dataset = TensorDataset(
        sequences_tensor,
        torch.tensor(T, dtype=torch.long),
        torch.tensor(Y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Print data summary
    print(f"  Calibration: {data['calibration_source']}")
    print(f"  True ATE: {true_ate:.4f}")
    print(f"  Individuals: {N_INDIVIDUALS}")
    print(f"  Periods: {N_PERIODS}")
    print(f"  Occupation bins: {N_OCCUPATIONS}")
    print(f"  Treatment rate: {T.mean():.2%}")
    print(f"  Outcome mean (treated): {Y[T == 1].mean():.4f}")
    print(f"  Outcome mean (control): {Y[T == 0].mean():.4f}")
    print(f"  Naive ATE (diff-in-means): {Y[T == 1].mean() - Y[T == 0].mean():.4f}")
    print(f"  Naive bias: {abs((Y[T == 1].mean() - Y[T == 0].mean()) - true_ate):.4f}")
    print(f"  Mean propensity: {propensity.mean():.4f}")
    print(f"  Mean HTE: {hte.mean():.4f}")
    print(f"  HTE range: [{hte.min():.4f}, {hte.max():.4f}]")

    # Covariate distributions
    print(f"\n  Covariate Distributions (should match NLSY79):")
    print(f"    Female %: {covariates['female'].mean()*100:.1f}% (target: 48.2%)")
    print(f"    Black %: {covariates['black'].mean()*100:.1f}% (target: 26.3%)")
    print(f"    Hispanic %: {covariates['hispanic'].mean()*100:.1f}% (target: 17.5%)")
    print(f"    Education mean: {covariates['education'].mean():.1f} (target: 13.5)")
    print(f"    Experience init mean: {covariates['experience_init'].mean():.1f}")

    # Career AIOE trajectory info
    print(f"\n  Career AIOE Trajectories:")
    print(f"    Mean AIOE (period 1): {career_aioe[:, 0].mean():.4f}")
    print(f"    Mean AIOE (period {N_PERIODS}): {career_aioe[:, -1].mean():.4f}")
    print(f"    AIOE range: [{career_aioe.min():.4f}, {career_aioe.max():.4f}]")
    print(f"    Discrete occupation IDs range: [{career_discrete.min()}, {career_discrete.max()}]")

    # =========================================================================
    # STEP 2: Train the 3 Embedding Variants
    # =========================================================================
    print_header("STEP 2: Training the 3 Embedding Variants")

    # --- Variant 1: Predictive GRU ---
    print_subheader("Variant 1: Predictive GRU (Baseline)")
    pred_model = PredictiveGRU(vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    train_predictive_embedding(pred_model, loader, epochs=EPOCHS)
    X_pred = extract_embeddings(pred_model, sequences_tensor)
    print(f"  Embedding shape: {X_pred.shape}")

    # --- Variant 2: Causal GRU (VIB) ---
    print_subheader("Variant 2: Causal GRU (VIB)")
    causal_model = CausalGRU(vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM)
    train_causal_embedding(causal_model, loader, epochs=EPOCHS)
    X_causal = extract_embeddings(causal_model, sequences_tensor)
    print(f"  Embedding shape: {X_causal.shape}")

    # --- Variant 3: Debiased GRU (Adversarial) ---
    print_subheader("Variant 3: Debiased GRU (Adversarial)")
    debiased_model = DebiasedGRU(vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM)
    adversary_model = Adversary(phi_dim=PHI_DIM)
    train_debiased_embedding(debiased_model, adversary_model, loader, epochs=EPOCHS)
    X_debiased = extract_embeddings(debiased_model, sequences_tensor)
    print(f"  Embedding shape: {X_debiased.shape}")

    # =========================================================================
    # STEP 3: DML Estimation with Each Variant (with full inference)
    # =========================================================================
    print_header("STEP 3: DML Estimation -- Comparison of 3 Variants (with Inference)")

    variants = {
        "Predictive GRU": X_pred,
        "Causal GRU (VIB)": X_causal,
        "Debiased GRU (Adversarial)": X_debiased,
    }

    results = {}
    best_variant = None
    best_bias = float("inf")

    for name, X_embed in variants.items():
        print_subheader(name)
        pipeline = CausalDMLPipeline()
        ate_est, cates, keep_idx = pipeline.fit_predict(Y, T, X_embed)

        bias = ate_est - true_ate
        pct_error = abs(bias / true_ate) * 100

        results[name] = {
            "ate": ate_est,
            "se": pipeline.ate_se,
            "ci": pipeline.ate_ci,
            "pvalue": pipeline.ate_pvalue,
            "bias": bias,
            "pct_error": pct_error,
            "cates": cates,
            "keep_idx": keep_idx,
            "pipeline": pipeline,
            "embedding": X_embed,
        }

        print(f"  ATE Estimated: {ate_est:.4f} (SE: {pipeline.ate_se:.4f})")
        print(f"  95% CI: [{pipeline.ate_ci[0]:.4f}, {pipeline.ate_ci[1]:.4f}]")
        print(f"  p-value: {pipeline.ate_pvalue:.4e}")
        print(f"  True ATE: {true_ate:.4f}")
        print(f"  Bias: {bias:.4f} ({pct_error:.1f}%)")

        if abs(bias) < best_bias:
            best_bias = abs(bias)
            best_variant = name

    # Print comparative table
    print_header("COMPARATIVE RESULTS TABLE")
    print(f"  {'Variant':<30} {'ATE':<10} {'SE':<10} {'95% CI':<24} {'p-value':<12} {'Bias':<10} {'% Error':<10} {'Status'}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*24} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in results.items():
        status = "Lowest bias" if name == best_variant else ("High bias" if r["pct_error"] > 30 else "Moderate bias")
        ci_str = f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
        print(f"  {name:<30} {r['ate']:<10.4f} {r['se']:<10.4f} {ci_str:<24} {r['pvalue']:<12.4e} {r['bias']:<10.4f} {r['pct_error']:<10.1f} {status}")
    print(f"  Lowest-bias variant: {best_variant}")

    # Store best variant info
    best = results[best_variant]
    X_best = best["embedding"]

    # =========================================================================
    # STEP 4: Complete Validation (Lowest-Bias Variant)
    # =========================================================================
    print_header(f"STEP 4: Complete Validation -- {best_variant}")

    # --- 4a. Variance Decomposition ---
    print_subheader("4a. Variance Decomposition + Heckman Interpretation")
    var_decomp = variance_decomposition(Y, T, X_best)
    print(f"  Variance Decomposition:")
    for key, val in var_decomp.items():
        if isinstance(val, (int, float)):
            total = var_decomp.get("total_variance", 1)
            pct = (val / total * 100) if total > 0 and key != "total_variance" else 0
            print(f"    {key}: {val:.6f} ({pct:.1f}%)")

    heckman_var = interpret_variance_heckman(var_decomp)
    print(f"  Selection Bias Severity (Heckman): {heckman_var['heckman_selection_severity']}")
    print(f"  Interpretation: {heckman_var['heckman_interpretation']}")

    # --- 4b. GATES ---
    print_subheader("4b. GATES + Heckman Interpretation (Human Capital)")
    X_best_df = pd.DataFrame(X_best[best["keep_idx"]])
    gates_df = best["pipeline"].estimate_gates(X_best_df)
    print(f"  GATES (Group Average Treatment Effects):")
    print(gates_df.to_string(index=False))

    heckman_gates = interpret_gates_heckman(gates_df)
    print(f"\n  Heckman Summary:")
    for k, v in heckman_gates['gates_summary'].items():
        print(f"    {k}: {v}")
    print(f"  Interpretation: {heckman_gates['heckman_interpretation']}")

    # --- 4b-ii. Formal GATES Heterogeneity Test ---
    print_subheader("4b-ii. Formal GATES Heterogeneity Test: H0: ATE(Q1) = ATE(Q5)")
    gates_test = best["pipeline"].test_gates_heterogeneity(gates_df)
    print(f"  Q1 mean CATE: {gates_test['q1_mean']:.4f}")
    print(f"  Q5 mean CATE: {gates_test['q5_mean']:.4f}")
    print(f"  Difference: {gates_test['difference']:.4f}")
    print(f"  t-statistic: {gates_test['t_statistic']:.2f}")
    print(f"  p-value: {gates_test['p_value']:.4e}")
    print(f"  Cohen's d: {gates_test['cohens_d']:.2f}")
    print(f"  Significant (alpha=0.05): {'YES' if gates_test['significant'] else 'NO'}")
    print(f"  Interpretation: {gates_test['interpretation']}")

    # --- 4c. Oster Sensitivity ---
    print_subheader("4c. Sensitivity Analysis (Oster Delta)")
    oster_delta = sensitivity_analysis_oster(Y, T, X_best)
    oster_interp = f"delta = {oster_delta:.2f} {'> 2 -> robust (Oster, 2019)' if oster_delta > 2 else '< 2 -> potentially sensitive (Oster, 2019)'}"
    print(f"  Oster Delta: {oster_delta:.4f}")
    print(f"  Interpretation: {oster_interp}")

    # --- 4d. Placebo Tests ---
    print_subheader("4d. Placebo Tests")
    placebo_pipeline = CausalDMLPipeline()
    placebo = run_placebo_tests(placebo_pipeline, Y, T, X_best)
    ate_rand_t = placebo['random_treatment_ate']
    ate_rand_y = placebo['random_outcome_ate']
    placebo_status = 'PASSED' if abs(ate_rand_t) < 0.3 and abs(ate_rand_y) < 0.3 else 'FAILED'
    print(f"  ATE with random treatment: {ate_rand_t:.4f} (expected ~= 0)")
    print(f"  ATE with random outcome: {ate_rand_y:.4f} (expected ~= 0)")
    print(f"  Status: {placebo_status}")

    # =========================================================================
    # STEP 5: VIB Sensitivity Analysis (Beta Sweep)
    # =========================================================================
    print_header("STEP 5: VIB Sensitivity Analysis -- Beta Sweep (VIB Sensitivity)")

    vib_results = vib_sensitivity_analysis(
        model_class=CausalGRU,
        adversary_class=Adversary,
        train_fn_causal=train_causal_embedding,
        train_fn_debiased=train_debiased_embedding,
        extract_fn=extract_embeddings,
        dml_pipeline_class=CausalDMLPipeline,
        loader=loader,
        sequences_tensor=sequences_tensor,
        Y=Y,
        T=T,
        true_ate=true_ate,
        beta_values=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        vocab_size=N_OCCUPATIONS,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        phi_dim=PHI_DIM,
        epochs=EPOCHS,
    )

    print_subheader("VIB Beta Sweep Results")
    print(vib_results.to_string(index=False))

    # Compare with adversarial (no beta to tune)
    print(f"\n  Adversarial Debiased (no beta): ATE = {results['Debiased GRU (Adversarial)']['ate']:.4f}, bias = {results['Debiased GRU (Adversarial)']['bias']:.4f} ({results['Debiased GRU (Adversarial)']['pct_error']:.1f}%)")
    print(f"  Lowest-error VIB beta: {vib_results.loc[vib_results['pct_error'].idxmin(), 'beta']:.4f}")
    print(f"  Lowest-error VIB ATE: {vib_results.loc[vib_results['pct_error'].idxmin(), 'ate']:.4f}")

    # =========================================================================
    # STEP 6: Heckman Two-Step Benchmark (WITHOUT exclusion restriction)
    # =========================================================================
    print_header("STEP 6: Benchmark -- Heckman Two-Step vs. DML (No Exclusion Restriction)")
    print("  Note: Semi-synthetic DGP does not have a peer_adoption exclusion")
    print("  restriction. Running Heckman without exclusion for comparison.")

    heckman_bench = run_heckman_two_step_benchmark(Y, T, X_best, Z_exclusion=None)
    print(f"  Heckman Two-Step ATE: {heckman_bench['ate_heckman_two_step']:.4f} (SE: {heckman_bench['ate_se']:.4f})")
    print(f"  DML ATE ({best_variant}): {best['ate']:.4f} (SE: {best['se']:.4f})")
    print(f"  True ATE: {true_ate:.4f}")
    print(f"  Has exclusion restriction: {heckman_bench['has_exclusion_restriction']}")
    print(f"  IMR coefficient: {heckman_bench['imr_coefficient']:.4f}")
    print(f"  IMR significant: {'Yes' if heckman_bench['imr_significant'] else 'No'}")
    print(f"\n  Interpretation: {heckman_bench['interpretation']}")

    bias_heckman = abs(heckman_bench["ate_heckman_two_step"] - true_ate)
    bias_dml = abs(best["bias"])
    improvement = ((bias_heckman - bias_dml) / bias_heckman) * 100 if bias_heckman > 0 else 0

    print(f"\n  Bias Comparison:")
    print(f"    Heckman Two-Step: |bias| = {bias_heckman:.4f}")
    print(f"    DML + {best_variant}: |bias| = {bias_dml:.4f}")
    print(f"    DML improvement over Heckman: {improvement:.1f}%")

    # =========================================================================
    # FINAL REPORT
    # =========================================================================
    print_header("FINAL REPORT -- CAREER-DML Semi-Synthetic (NLSY79 + Felten AIOE)")
    print(f"""
  DATA CALIBRATION:
    Source: NLSY79 (N=205,947; 11,728 individuals; 1979-2018)
    Mincer R²: 0.5245
    Education return: 5.4% per year
    Experience return: 9.9% (with diminishing returns)
    Gender penalty: -19.1%
    AI Exposure: Felten et al. (2021), 774 SOC occupations

  EMBEDDING PARADOX TEST:
    Predictive GRU:              ATE = {results['Predictive GRU']['ate']:.4f}, bias = {results['Predictive GRU']['pct_error']:.1f}%
    Causal GRU (VIB):            ATE = {results['Causal GRU (VIB)']['ate']:.4f}, bias = {results['Causal GRU (VIB)']['pct_error']:.1f}%
    Debiased GRU (Adversarial):  ATE = {results['Debiased GRU (Adversarial)']['ate']:.4f}, bias = {results['Debiased GRU (Adversarial)']['pct_error']:.1f}%

  LOWEST-BIAS VARIANT: {best_variant} (bias = {best['pct_error']:.1f}%)

  EMBEDDING PARADOX PERSISTS: {"YES" if results['Causal GRU (VIB)']['pct_error'] > results['Predictive GRU']['pct_error'] else "NO"}
    The causal VIB embedding {"underperforms" if results['Causal GRU (VIB)']['pct_error'] > results['Predictive GRU']['pct_error'] else "outperforms"} the predictive baseline,
    {"confirming" if results['Causal GRU (VIB)']['pct_error'] > results['Predictive GRU']['pct_error'] else "refuting"} that the Embedding Paradox is NOT an artifact of synthetic data.

  VALIDATION:
    Oster Delta: {oster_delta:.4f} ({"robust" if oster_delta > 2 else "not robust"})
    GATES heterogeneity: {"Significant" if gates_test['significant'] else "Not significant"} (p = {gates_test['p_value']:.4e})
    Placebo tests: {placebo_status}
    Heckman benchmark: DML {"improves" if improvement > 0 else "does not improve"} over Heckman by {improvement:.1f}%

  CONCLUSION:
    The CAREER-DML results are {"ROBUST" if results['Causal GRU (VIB)']['pct_error'] > results['Predictive GRU']['pct_error'] else "SENSITIVE"} to the choice of DGP.
    Semi-synthetic data calibrated with real U.S. labor market parameters
    {"confirms" if results['Causal GRU (VIB)']['pct_error'] > results['Predictive GRU']['pct_error'] else "does not confirm"} the central finding: causal embeddings designed to remove
    treatment-predictive information can paradoxically increase bias in
    DML estimation of AI adoption effects on wages.
    """)

    print_header("END OF SEMI-SYNTHETIC PIPELINE")

    # Return results for programmatic access
    return {
        "results": results,
        "best_variant": best_variant,
        "true_ate": true_ate,
        "oster_delta": oster_delta,
        "gates_test": gates_test,
        "placebo": placebo,
        "heckman": heckman_bench,
        "vib_results": vib_results,
    }


if __name__ == "__main__":
    main()
