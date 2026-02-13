"""
CAREER-DML: Double/Debiased Machine Learning Pipeline - v3.2
Camada 3 - Estimação causal usando CausalForestDML com career embeddings.

Este módulo implementa:
    - CausalForestDML com LightGBM como modelos de nuisance
    - Propensity score trimming para common support
    - GATES (Group Average Treatment Effects) para heterogeneidade
    - Cross-fitting automático via econml

Referências:
    - Chernozhukov et al. (2018), Double/Debiased ML
    - Wager & Athey (2018), Estimation and Inference of HTE using Random Forests
    - Athey, Tibshirani, Wager (2019), Generalized Random Forests
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from econml.dml import CausalForestDML


class CausalDMLPipeline:
    """Pipeline de estimação causal usando DML com Causal Forests.

    Fluxo:
        1. Propensity score trimming (common support enforcement)
        2. CausalForestDML fit com cross-fitting
        3. ATE estimation
        4. CATE estimation (individual-level)
        5. GATES estimation (group-level heterogeneity)

    Attributes:
        model: CausalForestDML estimator (fitted).
        ate: Estimated Average Treatment Effect.
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
            print(f"    Propensity trimming: {n_trimmed} observações removidas ({n_trimmed/len(T)*100:.1f}%)")

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
        """Fit the CausalForestDML and estimate ATE + CATEs.

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

        # Step 3: Estimate ATE (must pass X since X was used in fit)
        self.ate = float(self.model.ate(X=X_t))

        # Step 4: Estimate CATEs
        self.cates = self.model.effect(X=X_t).flatten()

        print(f"    ATE estimado: {self.ate:.4f}")
        print(f"    CATEs — mean: {self.cates.mean():.4f}, std: {self.cates.std():.4f}")

        return self.ate, self.cates, keep_idx

    def estimate_gates(
        self, X_df: pd.DataFrame, n_groups: int = 5
    ) -> pd.DataFrame:
        """Estimate Group Average Treatment Effects (GATES).

        Divides observations into quantile groups based on predicted CATEs
        and estimates the average effect within each group.

        This is the key tool for discovering heterogeneity, interpreted
        through the lens of Cunha & Heckman (2007) as proxies for
        different levels of latent human capital.

        Args:
            X_df: DataFrame of features (same as used in fit).
            n_groups: Number of quantile groups.

        Returns:
            DataFrame with columns: group, ate, ci_lower, ci_upper, n_obs.
        """
        if self.model is None or self.cates is None:
            raise ValueError("Must call fit_predict() before estimate_gates().")

        # Assign groups based on CATE quantiles
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
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_obs": int(mask.sum()),
            })

        return pd.DataFrame(gates_data)
