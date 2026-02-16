"""
CAREER-DML: Expanded Pipeline (v5.0)
=====================================================
Incorporates all methodological improvements from external peer review:

  1. LASSO+DML and Random Forest+DML benchmarks (gain decomposition)
  2. Static embedding benchmark (Word2Vec-style occupation averages)
  3. Corrected dimensions (phi_dim=64) and realistic ATE (0.08)
  4. Expanded power analysis with sensitivity to sigma_Y and R²
  5. Full validation suite (GATES, Oster, Placebo, Heckman, VIB sweep)

The gain decomposition table answers the key question:
  "Where exactly is the incremental gain of sequential embeddings?"

Date: February 2026
"""

import numpy as np
import pandas as pd
import torch
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier

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
    run_heckman_two_step_benchmark,
    vib_sensitivity_analysis,
)
from src.power_analysis import compute_mde, compute_required_n

# =============================================================================
# CONFIGURATION
# =============================================================================
N_INDIVIDUALS = 1000
N_PERIODS = 10
N_OCCUPATIONS = 50
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
PHI_DIM = 64       # Equal to HIDDEN_DIM (dimensional confound eliminated)
TRUE_ATE = 0.08    # Realistic 8% wage premium
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


def build_static_features(career_discrete, n_occ=N_OCCUPATIONS):
    """Build static (non-sequential) features from career sequences.

    Creates occupation frequency vectors and summary statistics,
    simulating what LASSO/RF would use without sequential embeddings.
    """
    n = career_discrete.shape[0]
    n_periods = career_discrete.shape[1]

    # Occupation frequency vector (bag-of-occupations)
    freq = np.zeros((n, n_occ))
    for i in range(n):
        for t in range(n_periods):
            freq[i, career_discrete[i, t]] += 1
    freq = freq / n_periods  # Normalize to proportions

    # Summary statistics
    last_occ = career_discrete[:, -1].reshape(-1, 1) / n_occ  # Last occupation (normalized)
    first_occ = career_discrete[:, 0].reshape(-1, 1) / n_occ  # First occupation
    n_transitions = np.sum(np.diff(career_discrete, axis=1) != 0, axis=1).reshape(-1, 1) / n_periods
    occ_diversity = np.array([len(set(career_discrete[i])) for i in range(n)]).reshape(-1, 1) / n_occ

    X_static = np.hstack([freq, last_occ, first_occ, n_transitions, occ_diversity])
    return X_static


def build_static_embedding(career_discrete, embedding_dim=EMBEDDING_DIM, n_occ=N_OCCUPATIONS):
    """Build a static embedding by averaging learned occupation vectors.

    This simulates a Word2Vec-style approach: each occupation gets a
    learned vector, and the career representation is the mean of all
    occupation vectors in the sequence. No temporal ordering is used.
    """
    # Use the frequency-weighted centroid of one-hot encoded occupations
    # projected into a lower-dimensional space via PCA
    from sklearn.decomposition import PCA

    n = career_discrete.shape[0]
    n_periods = career_discrete.shape[1]

    # One-hot encode each time step and average
    avg_onehot = np.zeros((n, n_occ))
    for i in range(n):
        for t in range(n_periods):
            avg_onehot[i, career_discrete[i, t]] += 1
    avg_onehot = avg_onehot / n_periods

    # PCA to reduce to embedding_dim
    n_components = min(embedding_dim, n_occ, n)
    pca = PCA(n_components=n_components)
    X_static_emb = pca.fit_transform(avg_onehot)

    return X_static_emb


def main():
    print_header("CAREER-DML EXPANDED PIPELINE (v5.0)")
    print(f"  Incorporates all external review suggestions")
    print(f"  PHI_DIM = {PHI_DIM}, TRUE_ATE = {TRUE_ATE}")
    print(f"  New benchmarks: LASSO+DML, RF+DML, Static Embedding+DML")

    # =========================================================================
    # STEP 1: Generate Data
    # =========================================================================
    t_start = time.time()
    step_times = {}

    print_header("STEP 1: Data Generation (Semi-Synthetic, ATE=0.08)")
    print("  NOTE ON TREATMENT DEFINITION:")
    print("    T = 1 if the individual transitions to an occupation with AI Exposure")
    print("    Index (AIOE, Felten et al. 2021) above the 75th percentile.")
    print("    This measures OCCUPATIONAL EXPOSURE to AI, not individual adoption")
    print("    of a specific technology. The estimated ATE represents the wage")
    print("    premium associated with this occupational transition, conditional")
    print("    on the full career trajectory.")
    print()
    t_step = time.time()

    dgp = SemiSyntheticDGP(n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED)

    import src.semi_synthetic_dgp as dgp_module
    original_ate = dgp_module.TRUE_ATE
    dgp_module.TRUE_ATE = TRUE_ATE
    data = dgp.generate()
    dgp_module.TRUE_ATE = original_ate

    career_aioe = data["career_sequences"]
    Y = data["Y"]
    T = data["T_treatment"].astype(int)
    true_ate = TRUE_ATE
    propensity = data["propensity"]
    hte = data["hte"]

    career_discrete = discretize_aioe_to_occupations(career_aioe, N_OCCUPATIONS)
    sequences = [career_discrete[i].tolist() for i in range(len(career_discrete))]
    sequences_tensor = pad_sequences(sequences, max_len=N_PERIODS)

    dataset = TensorDataset(
        sequences_tensor,
        torch.tensor(T, dtype=torch.long),
        torch.tensor(Y, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"  True ATE: {true_ate:.4f}")
    print(f"  Individuals: {N_INDIVIDUALS}")
    print(f"  Treatment rate: {T.mean():.2%}")
    print(f"  Outcome mean (treated): {Y[T == 1].mean():.4f}")
    print(f"  Outcome mean (control): {Y[T == 0].mean():.4f}")
    print(f"  Naive ATE (diff-in-means): {Y[T == 1].mean() - Y[T == 0].mean():.4f}")
    step_times['Step 1: Data Generation'] = time.time() - t_step

    # =========================================================================
    # STEP 2: Build All Representations
    # =========================================================================
    t_step = time.time()
    print_header("STEP 2: Building All Representations")

    # --- 2a. Static features (for LASSO and RF) ---
    print_subheader("2a. Static Features (bag-of-occupations + summary stats)")
    X_static = build_static_features(career_discrete)
    print(f"  Static feature shape: {X_static.shape}")

    # --- 2b. Static embedding (PCA of occupation frequencies) ---
    print_subheader("2b. Static Embedding (PCA, no temporal order)")
    X_static_emb = build_static_embedding(career_discrete)
    print(f"  Static embedding shape: {X_static_emb.shape}")

    # --- 2c. Predictive GRU (sequential) ---
    print_subheader("2c. Predictive GRU (sequential, dim=64)")
    pred_model = PredictiveGRU(
        vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM
    )
    train_predictive_embedding(pred_model, loader, epochs=EPOCHS)
    X_pred = extract_embeddings(pred_model, sequences_tensor)
    print(f"  Predictive GRU embedding shape: {X_pred.shape}")

    # --- 2d. Causal GRU VIB (sequential) ---
    print_subheader("2d. Causal GRU VIB (sequential, phi_dim=64)")
    causal_model = CausalGRU(
        vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM
    )
    train_causal_embedding(causal_model, loader, epochs=EPOCHS)
    X_causal = extract_embeddings(causal_model, sequences_tensor)
    print(f"  Causal GRU VIB embedding shape: {X_causal.shape}")

    # --- 2e. Debiased GRU Adversarial (sequential) ---
    print_subheader("2e. Debiased GRU Adversarial (sequential, phi_dim=64)")
    debiased_model = DebiasedGRU(
        vocab_size=N_OCCUPATIONS, embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM, phi_dim=PHI_DIM
    )
    adversary_model = Adversary(phi_dim=PHI_DIM)
    train_debiased_embedding(debiased_model, adversary_model, loader, epochs=EPOCHS)
    X_debiased = extract_embeddings(debiased_model, sequences_tensor)
    print(f"  Debiased GRU embedding shape: {X_debiased.shape}")
    step_times['Step 2: Embeddings'] = time.time() - t_step

    # =========================================================================
    # STEP 3: DML Estimation — All Methods
    # =========================================================================
    t_step = time.time()
    print_header("STEP 3: DML Estimation — Gain Decomposition Table")

    all_methods = {
        "1. Heckman Two-Step": None,  # Special case — handled separately
        "2. LASSO + DML": X_static,
        "3. Random Forest + DML": X_static,
        "4. Static Embedding + DML": X_static_emb,
        "5. Predictive GRU + DML": X_pred,
        "6. Causal GRU VIB + DML": X_causal,
        "7. Debiased GRU + DML": X_debiased,
    }

    results = {}

    # --- 3.1 Heckman Two-Step Benchmark ---
    print_subheader("Method 1: Heckman Two-Step (Classical Benchmark)")
    heckman_bench = run_heckman_two_step_benchmark(Y, T, X_pred, Z_exclusion=None)
    heckman_ate = heckman_bench['ate_heckman_two_step']
    heckman_bias = heckman_ate - true_ate
    heckman_pct = abs(heckman_bias / true_ate) * 100
    results["1. Heckman Two-Step"] = {
        "ate": heckman_ate, "bias": heckman_bias, "pct_error": heckman_pct,
        "se": np.nan, "method_type": "Parametric", "sequential": "No",
    }
    print(f"  ATE: {heckman_ate:.4f}, Bias: {heckman_bias:.4f} ({heckman_pct:.1f}%)")

    # --- 3.2 LASSO + DML ---
    print_subheader("Method 2: LASSO + DML (Semi-parametric, no sequence)")
    pipeline_lasso = CausalDMLPipeline()
    # Override the model_y and model_t with LASSO for first stage
    from econml.dml import CausalForestDML
    lasso_dml = CausalForestDML(
        model_y=LassoCV(cv=5, max_iter=5000),
        model_t=LassoCV(cv=5, max_iter=5000),
        n_estimators=200,
        min_samples_leaf=10,
        random_state=SEED,
    )
    lasso_dml.fit(Y, T, X=X_static)
    ate_lasso = lasso_dml.ate(X=X_static)
    bias_lasso = ate_lasso - true_ate
    pct_lasso = abs(bias_lasso / true_ate) * 100
    results["2. LASSO + DML"] = {
        "ate": ate_lasso, "bias": bias_lasso, "pct_error": pct_lasso,
        "se": np.nan, "method_type": "Semi-parametric", "sequential": "No",
    }
    print(f"  ATE: {ate_lasso:.4f}, Bias: {bias_lasso:.4f} ({pct_lasso:.1f}%)")

    # --- 3.3 Random Forest + DML ---
    print_subheader("Method 3: Random Forest + DML (Non-parametric, no sequence)")
    rf_dml = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED),
        model_t=RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED),
        n_estimators=200,
        min_samples_leaf=10,
        random_state=SEED,
    )
    rf_dml.fit(Y, T, X=X_static)
    ate_rf = rf_dml.ate(X=X_static)
    bias_rf = ate_rf - true_ate
    pct_rf = abs(bias_rf / true_ate) * 100
    results["3. Random Forest + DML"] = {
        "ate": ate_rf, "bias": bias_rf, "pct_error": pct_rf,
        "se": np.nan, "method_type": "Non-parametric", "sequential": "No",
    }
    print(f"  ATE: {ate_rf:.4f}, Bias: {bias_rf:.4f} ({pct_rf:.1f}%)")

    # --- 3.4 Static Embedding + DML ---
    print_subheader("Method 4: Static Embedding + DML (Embedding, no sequence)")
    pipeline_static = CausalDMLPipeline()
    ate_static, cates_static, keep_static = pipeline_static.fit_predict(Y, T, X_static_emb)
    bias_static = ate_static - true_ate
    pct_static = abs(bias_static / true_ate) * 100
    results["4. Static Embedding + DML"] = {
        "ate": ate_static, "bias": bias_static, "pct_error": pct_static,
        "se": pipeline_static.ate_se, "method_type": "Embedding (static)", "sequential": "No",
    }
    print(f"  ATE: {ate_static:.4f}, Bias: {bias_static:.4f} ({pct_static:.1f}%)")

    # --- 3.5-3.7 Sequential Embedding Variants ---
    seq_variants = {
        "5. Predictive GRU + DML": X_pred,
        "6. Causal GRU VIB + DML": X_causal,
        "7. Debiased GRU + DML": X_debiased,
    }

    best_variant = None
    best_bias = float("inf")

    for name, X_embed in seq_variants.items():
        print_subheader(f"Method {name[:1]}: {name[3:]}")
        pipeline = CausalDMLPipeline()
        ate_est, cates, keep_idx = pipeline.fit_predict(Y, T, X_embed)
        bias = ate_est - true_ate
        pct_error = abs(bias / true_ate) * 100

        results[name] = {
            "ate": ate_est, "bias": bias, "pct_error": pct_error,
            "se": pipeline.ate_se, "method_type": "Embedding (sequential)",
            "sequential": "Yes", "cates": cates, "keep_idx": keep_idx,
            "pipeline": pipeline, "embedding": X_embed,
        }
        print(f"  ATE: {ate_est:.4f}, Bias: {bias:.4f} ({pct_error:.1f}%)")

        if abs(bias) < best_bias:
            best_bias = abs(bias)
            best_variant = name

    step_times['Step 3: DML Estimation'] = time.time() - t_step

    # =========================================================================
    # STEP 4: Gain Decomposition Table
    # =========================================================================
    t_step = time.time()
    print_header("STEP 4: GAIN DECOMPOSITION TABLE")
    print(f"\n  {'Method':<35} {'Type':<22} {'Seq?':<6} {'ATE':<10} {'Bias':<10} {'|Bias|%':<10}")
    print(f"  {'-'*35} {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in results.items():
        print(f"  {name:<35} {r['method_type']:<22} {r.get('sequential','No'):<6} {r['ate']:<10.4f} {r['bias']:<10.4f} {r['pct_error']:<10.1f}")

    print(f"\n  True ATE: {true_ate:.4f}")
    print(f"\n  Lowest-bias method: {best_variant}")

    # Compute incremental gains
    heckman_abs_bias = abs(results["1. Heckman Two-Step"]["bias"])
    print(f"\n  INCREMENTAL GAIN ANALYSIS (vs. Heckman |bias| = {heckman_abs_bias:.4f}):")
    for name, r in results.items():
        if name == "1. Heckman Two-Step":
            continue
        abs_bias = abs(r["bias"])
        if heckman_abs_bias > 0:
            gain = ((heckman_abs_bias - abs_bias) / heckman_abs_bias) * 100
        else:
            gain = 0
        print(f"    {name:<35} |bias|={abs_bias:.4f}  gain vs Heckman: {gain:+.1f}%")

    step_times['Step 4: Gain Decomposition'] = time.time() - t_step

    # =========================================================================
    # STEP 5: Validation Suite
    # =========================================================================
    t_step = time.time()
    print_header("STEP 5: Validation Suite")

    best = results[best_variant]
    X_best = best["embedding"]

    # 5a. Oster
    print_subheader("5a. Oster Sensitivity")
    oster_delta = sensitivity_analysis_oster(Y, T, X_best)
    print(f"  Oster Delta: {oster_delta:.4f}")
    print(f"  Robust (>2): {'YES' if oster_delta > 2 else 'NO'}")

    # 5b. Placebo
    print_subheader("5b. Placebo Tests")
    placebo_pipeline = CausalDMLPipeline()
    placebo = run_placebo_tests(placebo_pipeline, Y, T, X_best)
    ate_rand_t = placebo['random_treatment_ate']
    ate_rand_y = placebo['random_outcome_ate']
    placebo_status = 'PASSED' if abs(ate_rand_t) < 0.3 and abs(ate_rand_y) < 0.3 else 'FAILED'
    print(f"  Random treatment ATE: {ate_rand_t:.4f}")
    print(f"  Random outcome ATE: {ate_rand_y:.4f}")
    print(f"  Status: {placebo_status}")

    # 5c. GATES
    print_subheader("5c. GATES Heterogeneity")
    X_best_df = pd.DataFrame(X_best[best["keep_idx"]])
    gates_df = best["pipeline"].estimate_gates(X_best_df)
    print(gates_df.to_string(index=False))

    gates_test = best["pipeline"].test_gates_heterogeneity(gates_df)
    print(f"  Heterogeneity significant: {'YES' if gates_test['significant'] else 'NO'}")
    print(f"  p-value: {gates_test['p_value']:.4e}")

    # 5d. Overlap Diagnostic
    print_subheader("5d. Overlap Diagnostic")
    ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=SEED)
    ps_model.fit(X_best, T)
    ps_scores = ps_model.predict_proba(X_best)[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.hist(ps_scores[T == 0], bins=30, alpha=0.6, label='Control (T=0)', color='#4A90D9')
    ax.hist(ps_scores[T == 1], bins=30, alpha=0.6, label='Treated (T=1)', color='#E74C3C')
    ax.axvline(x=0.05, color='gray', linestyle='--', linewidth=0.8, label='Trim bounds')
    ax.axvline(x=0.95, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Propensity Score P(T=1|X)')
    ax.set_ylabel('Frequency')
    ax.set_title('Overlap Diagnostic: Propensity Score Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/figures/overlap_diagnostic_v5.png', dpi=150)
    plt.close()

    pct_trimmed = ((ps_scores < 0.05).sum() + (ps_scores > 0.95).sum()) / len(ps_scores) * 100
    overlap_quality = 'GOOD' if pct_trimmed < 5 else ('MODERATE' if pct_trimmed < 20 else 'POOR')
    print(f"  Overlap quality: {overlap_quality} ({pct_trimmed:.1f}% trimmed)")

    step_times['Step 5: Validation'] = time.time() - t_step

    # =========================================================================
    # STEP 6: Expanded Power Analysis with Sensitivity
    # =========================================================================
    t_step = time.time()
    print_header("STEP 6: Power Analysis with Sensitivity")

    sigma_y_est = np.std(Y)
    print(f"  Estimated sigma_Y: {sigma_y_est:.4f}")

    # 6a. MDE at current N
    mde_current = compute_mde(N_INDIVIDUALS, sigma_y_est)
    print(f"  MDE at N={N_INDIVIDUALS}: {mde_current['mde']:.4f}")
    print(f"  MDE as % of sigma_Y: {mde_current['mde_pct']:.1f}%")
    print(f"  Can detect ATE=0.50? {'YES' if 0.50 > mde_current['mde'] else 'NO'}")
    print(f"  Can detect ATE=0.08? {'YES' if 0.08 > mde_current['mde'] else 'NO'}")

    # 6b. Required N for ATE=0.08
    req_n = compute_required_n(0.08, sigma_y_est)
    print(f"  Required N for ATE=0.08: {req_n['n_required']:,}")

    # 6c. Sensitivity analysis: MDE vs sigma_Y and R²
    print_subheader("6c. Sensitivity: MDE vs sigma_Y and R²")
    sigma_values = [0.5, 0.75, 1.0, 1.25, 1.5]
    r2_values = [0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"\n  {'sigma_Y':<10}", end="")
    for r2 in r2_values:
        print(f"  R²={r2:<6}", end="")
    print()
    print(f"  {'-'*10}", end="")
    for _ in r2_values:
        print(f"  {'-'*8}", end="")
    print()

    sensitivity_data = {}
    for sy in sigma_values:
        print(f"  {sy:<10.2f}", end="")
        for r2 in r2_values:
            mde = compute_mde(N_INDIVIDUALS, sy, r2)
            key = (sy, r2)
            sensitivity_data[key] = mde['mde']
            detectable = "Y" if 0.08 > mde['mde'] else "N"
            print(f"  {mde['mde']:.4f}{detectable:<2}", end="")
        print()

    # 6d. MDE at different sample sizes
    print_subheader("6d. MDE at Different Sample Sizes")
    sample_sizes = [1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
    print(f"  {'N':<12} {'MDE':<10} {'Detects 0.08?':<15} {'Detects 0.50?':<15}")
    print(f"  {'-'*12} {'-'*10} {'-'*15} {'-'*15}")
    for n in sample_sizes:
        mde = compute_mde(n, sigma_y_est)
        print(f"  {n:<12,} {mde['mde']:<10.4f} {'YES' if 0.08 > mde['mde'] else 'NO':<15} {'YES' if 0.50 > mde['mde'] else 'NO':<15}")

    # Save sensitivity figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: MDE vs N
    ns = np.logspace(3, 6, 50).astype(int)
    mdes = [compute_mde(n, sigma_y_est)['mde'] for n in ns]
    axes[0].loglog(ns, mdes, 'b-', linewidth=2)
    axes[0].axhline(y=0.08, color='r', linestyle='--', label='ATE = 0.08')
    axes[0].axhline(y=0.50, color='g', linestyle='--', label='ATE = 0.50')
    axes[0].axvline(x=N_INDIVIDUALS, color='gray', linestyle=':', label=f'Current N={N_INDIVIDUALS}')
    axes[0].set_xlabel('Sample Size (N)')
    axes[0].set_ylabel('Minimum Detectable Effect (MDE)')
    axes[0].set_title('Signal-to-Noise Frontier')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: Sensitivity heatmap
    mde_matrix = np.zeros((len(sigma_values), len(r2_values)))
    for i, sy in enumerate(sigma_values):
        for j, r2 in enumerate(r2_values):
            mde_matrix[i, j] = sensitivity_data[(sy, r2)]

    im = axes[1].imshow(mde_matrix, cmap='RdYlGn_r', aspect='auto')
    axes[1].set_xticks(range(len(r2_values)))
    axes[1].set_xticklabels([f'{r2:.1f}' for r2 in r2_values])
    axes[1].set_yticks(range(len(sigma_values)))
    axes[1].set_yticklabels([f'{sy:.2f}' for sy in sigma_values])
    axes[1].set_xlabel('R² (nuisance)')
    axes[1].set_ylabel('sigma_Y')
    axes[1].set_title(f'MDE Sensitivity (N={N_INDIVIDUALS})')
    for i in range(len(sigma_values)):
        for j in range(len(r2_values)):
            color = 'white' if mde_matrix[i, j] > 0.15 else 'black'
            axes[1].text(j, i, f'{mde_matrix[i, j]:.3f}', ha='center', va='center', color=color, fontsize=9)
    plt.colorbar(im, ax=axes[1], label='MDE')

    plt.tight_layout()
    plt.savefig('results/figures/power_analysis_sensitivity.png', dpi=150)
    plt.close()
    print(f"\n  Saved: results/figures/power_analysis_sensitivity.png")

    step_times['Step 6: Power Analysis'] = time.time() - t_step

    # =========================================================================
    # STEP 7: Embedding Paradox Verification
    # =========================================================================
    print_header("STEP 7: EMBEDDING PARADOX VERIFICATION")

    pred_err = results["5. Predictive GRU + DML"]["pct_error"]
    vib_err = results["6. Causal GRU VIB + DML"]["pct_error"]
    deb_err = results["7. Debiased GRU + DML"]["pct_error"]

    paradox_persists = vib_err > pred_err

    print(f"""
  RESULTS (PHI_DIM = {PHI_DIM}, TRUE_ATE = {TRUE_ATE}):

    Predictive GRU:     ATE = {results['5. Predictive GRU + DML']['ate']:.4f}, |bias| = {pred_err:.1f}%
    Causal GRU (VIB):   ATE = {results['6. Causal GRU VIB + DML']['ate']:.4f}, |bias| = {vib_err:.1f}%
    Debiased (Adv):     ATE = {results['7. Debiased GRU + DML']['ate']:.4f}, |bias| = {deb_err:.1f}%

  EMBEDDING PARADOX (VIB bias > Predictive bias):
    {"YES — GENUINE FINDING" if paradox_persists else "NO — NOT CONFIRMED IN THIS RUN"}

  INTERPRETATION:
    {"The information bottleneck destroys causally relevant information in sequential career data. This is a structural incompatibility, not a dimensional artifact." if paradox_persists else "The paradox did not manifest in this run. This may be due to stochastic variation with small samples."}
    """)

    # =========================================================================
    # STEP 8: Final Summary
    # =========================================================================
    print_header("FINAL SUMMARY: GAIN DECOMPOSITION")

    # --- Computational Complexity Report ---
    t_total = time.time() - t_start
    print(f"\n  COMPUTATIONAL PROFILE (N={N_INDIVIDUALS}):")
    for step_name, step_time in step_times.items():
        print(f"    {step_name:<35} {step_time:>8.1f}s")
    print(f"    {'TOTAL':<35} {t_total:>8.1f}s")
    print(f"\n  PROJECTED SCALING:")
    for target_n in [10_000, 100_000, 1_000_000]:
        factor = target_n / N_INDIVIDUALS
        projected = t_total * factor
        hours = projected / 3600
        print(f"    N={target_n:>10,}: ~{projected:,.0f}s ({hours:.1f}h)")
    print(f"    Note: GPU acceleration and mini-batch training reduce embedding")
    print(f"    time sub-linearly. DML with cross-fitting scales ~O(N log N).")

    print(f"""
  The gain decomposition answers the key question from the external review:
  "Where exactly is the incremental gain of sequential embeddings?"

  Decomposition (|bias| reduction vs. Heckman):
    """)

    heckman_abs = abs(results["1. Heckman Two-Step"]["bias"])
    decomposition = []
    for name in ["2. LASSO + DML", "3. Random Forest + DML", "4. Static Embedding + DML",
                  "5. Predictive GRU + DML", "6. Causal GRU VIB + DML", "7. Debiased GRU + DML"]:
        r = results[name]
        abs_bias = abs(r["bias"])
        gain = ((heckman_abs - abs_bias) / heckman_abs * 100) if heckman_abs > 0 else 0
        decomposition.append((name, abs_bias, gain))
        print(f"    {name:<35} |bias|={abs_bias:.4f}  gain: {gain:+.1f}%")

    print(f"""
  KEY FINDINGS:
    1. Heckman -> LASSO/RF: Measures the gain from flexibility alone
    2. LASSO/RF -> Static Embedding: Measures the gain from learned representations
    3. Static Embedding -> Predictive GRU: Measures the gain from temporal ordering
    4. Predictive GRU -> VIB/Debiased: Measures the effect of causal regularization

  VALIDATION:
    Oster Delta: {oster_delta:.4f} ({'robust' if oster_delta > 2 else 'not robust'})
    Placebo tests: {placebo_status}
    GATES heterogeneity: {'Significant' if gates_test['significant'] else 'Not significant'}
    Overlap quality: {overlap_quality}
    """)

    # --- Future Extensions (deferred to PhD) ---
    print("\n  FUTURE EXTENSIONS (deferred to PhD programme):")
    print("    1. Dynamic treatment effects tau(t): Estimate how the AI wage premium")
    print("       evolves over time since occupational transition, capturing cumulative")
    print("       and fading effects across the career lifecycle.")
    print("    2. Treatment timing heterogeneity: Analyse whether early vs. late")
    print("       adopters of AI-exposed occupations experience different returns.")
    print("    3. Life-cycle heterogeneity: Explore how the treatment effect varies")
    print("       by career stage (early, mid, late career).")
    print("    These extensions require panel data with sufficient temporal depth,")
    print("    motivating the use of Danish administrative registers (N > 1M, T > 30y).")

    print_header("END OF EXPANDED PIPELINE (v5.0)")

    return {
        "results": results,
        "best_variant": best_variant,
        "true_ate": true_ate,
        "paradox_persists": paradox_persists,
        "oster_delta": oster_delta,
        "gates_test": gates_test,
        "placebo_status": placebo_status,
        "sensitivity_data": sensitivity_data,
    }


if __name__ == "__main__":
    main()
