"""
CAREER-DML: Board-Corrected Semi-Synthetic Pipeline
=====================================================
Implements the two CRITICAL decisions from the 3-Layer Board Analysis:

  1. PHI_DIM = 64 (same as HIDDEN_DIM) — eliminates the dimensional confound
     that could explain the Embedding Paradox as an artifact.
  2. TRUE_ATE = 0.08 (8% wage premium) — calibrated with literature
     (Acemoglu et al., 2022; Felten et al., 2021).

If the Embedding Paradox persists with equal dimensions and a realistic ATE,
it is a GENUINE scientific finding. If it disappears, it was an artifact.

Board: Dr. Tom Gard (President), Rodolf Mikel Ghannam Neto (Co-President)
Consultants: Victor Veitch, Stefan Wager, James Heckman (theoretical perspectives)
Date: February 2026
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

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
# BOARD-CORRECTED CONFIGURATION
# =============================================================================
N_INDIVIDUALS = 1000
N_PERIODS = 10
N_OCCUPATIONS = 50
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

# BOARD DECISION 1: PHI_DIM = HIDDEN_DIM (eliminate dimensional confound)
PHI_DIM = 64  # Was 16 — now equal to HIDDEN_DIM

# BOARD DECISION 2: TRUE_ATE = 0.08 (realistic 8% wage premium)
BOARD_TRUE_ATE = 0.08

EPOCHS = 15
BATCH_SIZE = 64
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


def discretize_aioe_to_occupations(career_aioe, n_bins=N_OCCUPATIONS):
    flat = career_aioe.flatten()
    bin_edges = np.percentile(flat, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < n_bins + 1:
        bin_edges = np.linspace(flat.min() - 0.001, flat.max() + 0.001, n_bins + 1)
    career_discrete = np.digitize(career_aioe, bin_edges) - 1
    career_discrete = np.clip(career_discrete, 0, n_bins - 1)
    return career_discrete


def pad_sequences(sequences, max_len=N_PERIODS):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [0] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return torch.tensor(padded, dtype=torch.long)


def extract_embeddings(model, sequences):
    model.eval()
    with torch.no_grad():
        return model.get_representation(sequences).numpy()


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_subheader(title):
    print(f"\n  --- {title} ---")


def main():
    print_header("BOARD-CORRECTED PIPELINE")
    print(f"  Board Decision 1: PHI_DIM = {PHI_DIM} (was 16)")
    print(f"  Board Decision 2: TRUE_ATE = {BOARD_TRUE_ATE} (was 0.538)")
    print(f"  Purpose: Test if Embedding Paradox is genuine or dimensional artifact")

    # =========================================================================
    # STEP 1: Generate data with corrected TRUE_ATE
    # =========================================================================
    print_header("STEP 1: Data Generation (Board-Corrected ATE)")

    dgp = SemiSyntheticDGP(n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED)

    # Override TRUE_ATE in the DGP module before generating
    import src.semi_synthetic_dgp as dgp_module
    original_ate = dgp_module.TRUE_ATE
    dgp_module.TRUE_ATE = BOARD_TRUE_ATE

    data = dgp.generate()

    # Restore original (for safety)
    dgp_module.TRUE_ATE = original_ate

    career_aioe = data["career_sequences"]
    Y = data["Y"]
    T = data["T_treatment"].astype(int)
    true_ate = BOARD_TRUE_ATE  # Use Board-corrected value
    propensity = data["propensity"]
    hte = data["hte"]
    covariates = data["covariates"]

    career_discrete = discretize_aioe_to_occupations(career_aioe, N_OCCUPATIONS)
    sequences = [career_discrete[i].tolist() for i in range(len(career_discrete))]
    sequences_tensor = pad_sequences(sequences, max_len=N_PERIODS)

    dataset = TensorDataset(
        sequences_tensor,
        torch.tensor(T, dtype=torch.long),
        torch.tensor(Y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"  True ATE (Board-corrected): {true_ate:.4f}")
    print(f"  Individuals: {N_INDIVIDUALS}")
    print(f"  Treatment rate: {T.mean():.2%}")
    print(f"  Outcome mean (treated): {Y[T == 1].mean():.4f}")
    print(f"  Outcome mean (control): {Y[T == 0].mean():.4f}")
    print(f"  Naive ATE (diff-in-means): {Y[T == 1].mean() - Y[T == 0].mean():.4f}")
    print(f"  Mean HTE: {hte.mean():.4f}")
    print(f"  HTE range: [{hte.min():.4f}, {hte.max():.4f}]")

    # =========================================================================
    # STEP 2: Train 3 Embedding Variants (ALL with PHI_DIM = 64)
    # =========================================================================
    print_header("STEP 2: Training 3 Embedding Variants (PHI_DIM = 64 for ALL)")

    # Variant 1: Predictive GRU (hidden_dim = 64, output dim = 64)
    print_subheader("Variant 1: Predictive GRU (Baseline, dim=64)")
    pred_model = PredictiveGRU(
        vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM
    )
    train_predictive_embedding(pred_model, loader, epochs=EPOCHS)
    X_pred = extract_embeddings(pred_model, sequences_tensor)
    print(f"  Embedding shape: {X_pred.shape}")

    # Variant 2: Causal GRU VIB (phi_dim = 64, same as hidden_dim)
    print_subheader("Variant 2: Causal GRU VIB (phi_dim=64, BOARD-CORRECTED)")
    causal_model = CausalGRU(
        vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM
    )
    train_causal_embedding(causal_model, loader, epochs=EPOCHS)
    X_causal = extract_embeddings(causal_model, sequences_tensor)
    print(f"  Embedding shape: {X_causal.shape}")

    # Variant 3: Debiased GRU Adversarial (phi_dim = 64, same as hidden_dim)
    print_subheader("Variant 3: Debiased GRU Adversarial (phi_dim=64, BOARD-CORRECTED)")
    debiased_model = DebiasedGRU(
        vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM
    )
    adversary_model = Adversary(phi_dim=PHI_DIM)
    train_debiased_embedding(debiased_model, adversary_model, loader, epochs=EPOCHS)
    X_debiased = extract_embeddings(debiased_model, sequences_tensor)
    print(f"  Embedding shape: {X_debiased.shape}")

    # =========================================================================
    # STEP 3: DML Estimation
    # =========================================================================
    print_header("STEP 3: DML Estimation (Board-Corrected)")

    variants = {
        "Predictive GRU (dim=64)": X_pred,
        "Causal GRU VIB (dim=64)": X_causal,
        "Debiased Adversarial (dim=64)": X_debiased,
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

    # Comparative table
    print_header("BOARD-CORRECTED COMPARATIVE RESULTS")
    print(f"  {'Variant':<35} {'ATE':<10} {'SE':<10} {'Bias':<10} {'% Error':<10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in results.items():
        print(f"  {name:<35} {r['ate']:<10.4f} {r['se']:<10.4f} {r['bias']:<10.4f} {r['pct_error']:<10.1f}")
    print(f"\n  Lowest-bias variant: {best_variant}")

    best = results[best_variant]
    X_best = best["embedding"]

    # =========================================================================
    # STEP 4: Validation
    # =========================================================================
    print_header(f"STEP 4: Validation ({best_variant})")

    # 4a. Variance Decomposition
    print_subheader("4a. Variance Decomposition + Heckman")
    var_decomp = variance_decomposition(Y, T, X_best)
    heckman_var = interpret_variance_heckman(var_decomp)
    print(f"  Selection Severity: {heckman_var['heckman_selection_severity']}")
    for key, val in heckman_var['variance_table'].items():
        print(f"    {key}: {val}")

    # 4b. GATES
    print_subheader("4b. GATES + Heterogeneity Test")
    X_best_df = pd.DataFrame(X_best[best["keep_idx"]])
    gates_df = best["pipeline"].estimate_gates(X_best_df)
    print(gates_df.to_string(index=False))

    gates_test = best["pipeline"].test_gates_heterogeneity(gates_df)
    print(f"  Heterogeneity significant: {'YES' if gates_test['significant'] else 'NO'}")
    print(f"  p-value: {gates_test['p_value']:.4e}")
    print(f"  Cohen's d: {gates_test['cohens_d']:.2f}")

    # 4c. Oster
    print_subheader("4c. Oster Sensitivity")
    oster_delta = sensitivity_analysis_oster(Y, T, X_best)
    print(f"  Oster Delta: {oster_delta:.4f}")
    print(f"  Robust (>2): {'YES' if oster_delta > 2 else 'NO'}")

    # 4d. Placebo
    print_subheader("4d. Placebo Tests")
    placebo_pipeline = CausalDMLPipeline()
    placebo = run_placebo_tests(placebo_pipeline, Y, T, X_best)
    ate_rand_t = placebo['random_treatment_ate']
    ate_rand_y = placebo['random_outcome_ate']
    placebo_status = 'PASSED' if abs(ate_rand_t) < 0.3 and abs(ate_rand_y) < 0.3 else 'FAILED'
    print(f"  Random treatment ATE: {ate_rand_t:.4f}")
    print(f"  Random outcome ATE: {ate_rand_y:.4f}")
    print(f"  Status: {placebo_status}")

    # =========================================================================
    # STEP 5: VIB Beta Sweep (with corrected phi_dim=64)
    # =========================================================================
    print_header("STEP 5: VIB Beta Sweep (phi_dim=64)")

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
    print_subheader("VIB Beta Sweep Results (phi_dim=64)")
    print(vib_results.to_string(index=False))

    # =========================================================================
    # STEP 6: Heckman Benchmark
    # =========================================================================
    print_header("STEP 6: Heckman Two-Step Benchmark")

    heckman_bench = run_heckman_two_step_benchmark(Y, T, X_best, Z_exclusion=None)
    print(f"  Heckman ATE: {heckman_bench['ate_heckman_two_step']:.4f}")
    print(f"  DML ATE ({best_variant}): {best['ate']:.4f}")
    print(f"  True ATE: {true_ate:.4f}")

    bias_heckman = abs(heckman_bench["ate_heckman_two_step"] - true_ate)
    bias_dml = abs(best["bias"])
    improvement = ((bias_heckman - bias_dml) / bias_heckman) * 100 if bias_heckman > 0 else 0
    print(f"  Heckman |bias|: {bias_heckman:.4f}")
    print(f"  DML |bias|: {bias_dml:.4f}")
    print(f"  DML improvement: {improvement:.1f}%")

    # =========================================================================
    # BOARD VERDICT
    # =========================================================================
    print_header("BOARD VERDICT: EMBEDDING PARADOX TEST")

    pred_name = "Predictive GRU (dim=64)"
    vib_name = "Causal GRU VIB (dim=64)"
    deb_name = "Debiased Adversarial (dim=64)"

    pred_err = results[pred_name]["pct_error"]
    vib_err = results[vib_name]["pct_error"]
    deb_err = results[deb_name]["pct_error"]

    paradox_persists = vib_err > pred_err

    print(f"""
  BOARD-CORRECTED RESULTS (PHI_DIM = 64, TRUE_ATE = {BOARD_TRUE_ATE}):

    Predictive GRU:     ATE = {results[pred_name]['ate']:.4f}, bias = {pred_err:.1f}%
    Causal GRU (VIB):   ATE = {results[vib_name]['ate']:.4f}, bias = {vib_err:.1f}%
    Debiased (Adv):     ATE = {results[deb_name]['ate']:.4f}, bias = {deb_err:.1f}%

  EMBEDDING PARADOX (VIB bias > Predictive bias):
    With phi_dim=16 (original):  YES (confirmed in both synthetic and semi-synthetic)
    With phi_dim=64 (corrected): {"YES — GENUINE FINDING" if paradox_persists else "NO — DIMENSIONAL ARTIFACT"}

  BOARD CONCLUSION:
    {"The Embedding Paradox is a GENUINE scientific finding. Even with equal embedding dimensions, the VIB strategy produces higher bias than the predictive baseline. This confirms that the information bottleneck destroys causally relevant information in sequential career data." if paradox_persists else "The Embedding Paradox DISAPPEARS when embedding dimensions are equalized. The original finding was a DIMENSIONAL ARTIFACT: the VIB with 16 dimensions simply had insufficient capacity to capture confounding, while the Predictive GRU with 64 dimensions had more room. The paper narrative must be revised to focus on the dimensionality sensitivity of causal embeddings."}

  VALIDATION SUMMARY:
    Oster Delta: {oster_delta:.4f} ({"robust" if oster_delta > 2 else "not robust"})
    GATES heterogeneity: {"Significant" if gates_test['significant'] else "Not significant"} (p = {gates_test['p_value']:.4e})
    Placebo tests: {placebo_status}
    Heckman comparison: DML {"improves" if improvement > 0 else "does not improve"} by {improvement:.1f}%
    """)

    print_header("END OF BOARD-CORRECTED PIPELINE")

    return {
        "results": results,
        "best_variant": best_variant,
        "true_ate": true_ate,
        "paradox_persists": paradox_persists,
        "oster_delta": oster_delta,
        "gates_test": gates_test,
        "placebo_status": placebo_status,
        "heckman": heckman_bench,
        "vib_results": vib_results,
    }


if __name__ == "__main__":
    main()
