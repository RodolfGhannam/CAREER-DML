"""
CAREER-DML: Double/Debiased Machine Learning Pipeline - v3.3
Layer 3 - Causal estimation using CausalForestDML with career embeddings.

This module implements:
    - CausalForestDML with GradientBoosting as nuisance models
    - Propensity score trimming for common support
    - ATE inference with standard errors, confidence intervals, and p-values
    - GATES (Group Average Treatment Effects) with formal hypothesis testing
    - Cross-fitting via econml

References:
    - Chernozhukov et al. (2018), Double/Debiased ML
    - Wager & Athey (2018), Estimation and Inference of HTE using Random Forests
    - Athey, Tibshirani, Wager (2019), Generalized Random Forests
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dml import CausalForestDML


class CausalDMLPipeline:
    """Pipeline for causal estimation using DML with Causal Forests.

    Flow:
        1. Propensity score trimming (common support enforcement)
        2. CausalForestDML fit with cross-fitting
        3. ATE estimation with inference (SE, CI, p-value)
        4. CATE estimation (individual-level)
        5. GATES estimation with formal hypothesis testing

    Attributes:
        model: CausalForestDML estimator (fitted).
        ate: Estimated Average Treatment Effect.
        ate_se: Standard error of the ATE estimate.
        ate_ci: 95% confidence interval for the ATE.
        ate_pvalue: p-value for H0: ATE = 0.
        cates: Array of individual-level CATEs.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_leaf: int = 30,
        max_depth: int = 10,
        ps_trim_lower: float = 0.05,
        ps_trim_upper: float = 0.95,
    ):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.ps_trim_lower = ps_trim_lower
        self.ps_trim_upper = ps_trim_upper
        self.model = None
        self.ate = None
        self.ate_se = None
        self.ate_ci = None
        self.ate_pvalue = None
        self.cates = None

    def _propensity_trim(
        self, Y: np.ndarray, T: np.ndarray, X: np.ndarray, W: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
        """Trim observations with extreme propensity scores.

        Enforces common support by removing units where P(T=1|X) is
        too close to 0 or 1, following best practices in causal inference.
        """
        features = X if W is None else np.column_stack([X, W])
        ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3)
        ps_model.fit(features, T)
        ps = ps_model.predict_proba(features)[:, 1]

        keep = (ps > self.ps_trim_lower) & (ps < self.ps_trim_upper)
        keep_idx = np.where(keep)[0]

        n_trimmed = (~keep).sum()
        if n_trimmed > 0:
            print(f"    Propensity trimming: {n_trimmed} observations removed ({n_trimmed/len(T)*100:.1f}%)")

        Y_trim = Y[keep]
        T_trim = T[keep]
        X_trim = X[keep]
        W_trim = W[keep] if W is not None else None

        return Y_trim, T_trim, X_trim, W_trim, keep_idx

    def fit_predict(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: np.ndarray,
        W: np.ndarray | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Fit the CausalForestDML and estimate ATE + CATEs with full inference.

        Args:
            Y: Outcome array (n,).
            T: Treatment array (n,).
            X: Embedding features for heterogeneity (n, d).
            W: Additional controls (n, p). Optional.

        Returns:
            Tuple of (ATE, CATEs array, keep_idx from trimming).
        """
        # Step 1: Propensity trimming
        Y_t, T_t, X_t, W_t, keep_idx = self._propensity_trim(Y, T, X, W)

        # Step 2: Fit CausalForestDML
        self.model = CausalForestDML(
            model_y=GradientBoostingRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05
            ),
            model_t=GradientBoostingClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05
            ),
            discrete_treatment=True,
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=42,
        )

        self.model.fit(Y_t, T_t, X=X_t, W=W_t)

        # Step 3: ATE with full inference (Wager & Athey, 2018)
        ate_inference = self.model.ate_inference(X=X_t)
        self.ate = float(ate_inference.mean_point)
        self.ate_se = float(ate_inference.stderr_mean)
        ci = ate_inference.conf_int_mean(alpha=0.05)
        # Handle both scalar and array returns from different EconML versions
        ci_low = ci[0].item() if hasattr(ci[0], 'item') else float(ci[0])
        ci_high = ci[1].item() if hasattr(ci[1], 'item') else float(ci[1])
        self.ate_ci = (ci_low, ci_high)
        pval = ate_inference.pvalue(value=0)
        self.ate_pvalue = pval.item() if hasattr(pval, 'item') else float(pval)

        # Step 4: Estimate CATEs
        self.cates = self.model.effect(X=X_t).flatten()

        print(f"    ATE estimate: {self.ate:.4f}")
        print(f"    SE: {self.ate_se:.4f}")
        print(f"    95% CI: [{self.ate_ci[0]:.4f}, {self.ate_ci[1]:.4f}]")
        print(f"    p-value: {self.ate_pvalue:.4e}")
        print(f"    CATEs: mean={self.cates.mean():.4f}, std={self.cates.std():.4f}")

        return self.ate, self.cates, keep_idx

    def estimate_gates(
        self, X_df: pd.DataFrame, n_groups: int = 5
    ) -> pd.DataFrame:
        """Estimate Group Average Treatment Effects (GATES) with inference.

        Divides observations into quantile groups based on predicted CATEs
        and estimates the average effect within each group with standard
        errors and confidence intervals.

        Interpreted through the lens of Cunha & Heckman (2007) as proxies
        for different levels of latent human capital.

        Args:
            X_df: DataFrame of features (same as used in fit).
            n_groups: Number of quantile groups.

        Returns:
            DataFrame with columns: group, ate, se, ci_lower, ci_upper, n_obs.
        """
        if self.model is None or self.cates is None:
            raise ValueError("Must call fit_predict() before estimate_gates().")

        quantile_labels = pd.qcut(self.cates, q=n_groups, labels=False, duplicates="drop")

        gates_data = []
        for g in sorted(np.unique(quantile_labels)):
            mask = quantile_labels == g
            group_cates = self.cates[mask]

            ate_g = group_cates.mean()
            se_g = group_cates.std() / np.sqrt(len(group_cates))
            ci_lower = ate_g - 1.96 * se_g
            ci_upper = ate_g + 1.96 * se_g

            gates_data.append({
                "group": int(g) + 1,
                "ate": ate_g,
                "se": se_g,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_obs": int(mask.sum()),
            })

        return pd.DataFrame(gates_data)

    def test_gates_heterogeneity(self, gates_df: pd.DataFrame) -> dict:
        """Formal hypothesis test: H0: ATE(Q1) = ATE(Q_max).

        Uses Welch's t-test on the CATE distributions of the lowest
        and highest quintiles to determine if treatment effect
        heterogeneity is statistically significant.

        Returns:
            Dictionary with test statistic, p-value, and interpretation.
        """
        if self.cates is None:
            raise ValueError("Must call fit_predict() before testing heterogeneity.")

        n_groups = len(gates_df)
        quantile_labels = pd.qcut(self.cates, q=n_groups, labels=False, duplicates="drop")

        q1_cates = self.cates[quantile_labels == 0]
        q5_cates = self.cates[quantile_labels == quantile_labels.max()]

        # Welch's t-test (does not assume equal variances)
        t_stat, p_value = stats.ttest_ind(q5_cates, q1_cates, equal_var=False)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((q1_cates.std()**2 + q5_cates.std()**2) / 2)
        cohens_d = (q5_cates.mean() - q1_cates.mean()) / pooled_std if pooled_std > 0 else 0

        significant = p_value < 0.05

        return {
            "q1_mean": float(q1_cates.mean()),
            "q5_mean": float(q5_cates.mean()),
            "difference": float(q5_cates.mean() - q1_cates.mean()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "significant": significant,
            "interpretation": (
                f"The treatment effect for Q{n_groups} (high human capital) is "
                f"{q5_cates.mean():.4f}, compared to {q1_cates.mean():.4f} for Q1 "
                f"(low human capital), a difference of {q5_cates.mean() - q1_cates.mean():.4f}. "
                f"This difference is {'statistically significant' if significant else 'not statistically significant'} "
                f"(t={t_stat:.2f}, p={p_value:.4e}, Cohen's d={cohens_d:.2f}), "
                f"{'confirming' if significant else 'failing to confirm'} the hypothesis of "
                f"skill-biased technological change (Cunha & Heckman, 2007)."
            ),
        }
