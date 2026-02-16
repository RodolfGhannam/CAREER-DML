"""
CAREER-DML: Statistical Power Analysis Module

Computes the minimum detectable effect (MDE) for the DML estimator
given sample size, noise level, and significance level.

This module addresses Board Review finding R2.3:
"The Signal-to-Noise Frontier should be quantified with a formal
power calculation, not just observed empirically."

Reference:
    Chernozhukov et al. (2018), "Double/Debiased Machine Learning
    for Treatment and Structural Parameters", Econometrics Journal.
"""

import numpy as np
from scipy import stats


def compute_mde(
    n: int,
    sigma_y: float,
    r2_nuisance: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    treatment_share: float = 0.5,
) -> dict:
    """Compute the Minimum Detectable Effect (MDE) for DML.

    The MDE is the smallest true ATE that the estimator can detect
    with probability `power` at significance level `alpha`.

    For a partially linear model Y = tau*T + g(X) + epsilon, the
    asymptotic variance of the DML estimator is:

        Var(hat{tau}) = sigma_epsilon^2 / (n * Var(T|X))

    where sigma_epsilon^2 = sigma_y^2 * (1 - R^2_nuisance) is the
    residual variance after partialling out X.

    Args:
        n: Sample size.
        sigma_y: Standard deviation of the outcome variable.
        r2_nuisance: R-squared of the nuisance function g(X) for Y.
            Higher values mean better first-stage fit and lower MDE.
        alpha: Significance level (two-sided test).
        power: Desired statistical power (1 - beta).
        treatment_share: Proportion of treated units (p).

    Returns:
        Dictionary with:
            - mde: Minimum detectable effect size.
            - mde_pct: MDE as percentage of sigma_y.
            - se_ate: Estimated standard error of the ATE.
            - n_required_for_ate: Sample size needed to detect a given ATE.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Residual variance after partialling out X
    sigma_epsilon = sigma_y * np.sqrt(1 - r2_nuisance)

    # Variance of T|X (for binary treatment with share p)
    var_t = treatment_share * (1 - treatment_share)

    # Standard error of the ATE estimator
    se_ate = sigma_epsilon / np.sqrt(n * var_t)

    # Minimum detectable effect
    mde = (z_alpha + z_beta) * se_ate

    return {
        "mde": mde,
        "mde_pct": (mde / sigma_y) * 100 if sigma_y > 0 else np.inf,
        "se_ate": se_ate,
        "sigma_epsilon": sigma_epsilon,
        "z_alpha": z_alpha,
        "z_beta": z_beta,
    }


def compute_required_n(
    target_ate: float,
    sigma_y: float,
    r2_nuisance: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
    treatment_share: float = 0.5,
) -> dict:
    """Compute the sample size required to detect a given ATE.

    Args:
        target_ate: The true ATE to detect.
        sigma_y: Standard deviation of the outcome variable.
        r2_nuisance: R-squared of the nuisance function g(X) for Y.
        alpha: Significance level (two-sided test).
        power: Desired statistical power (1 - beta).
        treatment_share: Proportion of treated units (p).

    Returns:
        Dictionary with:
            - n_required: Minimum sample size.
            - se_required: Required standard error.
            - mde_at_n: MDE at the computed sample size (should equal target_ate).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    sigma_epsilon = sigma_y * np.sqrt(1 - r2_nuisance)
    var_t = treatment_share * (1 - treatment_share)

    # Required SE: target_ate = (z_alpha + z_beta) * SE
    se_required = target_ate / (z_alpha + z_beta)

    # SE = sigma_epsilon / sqrt(n * var_t)
    # => n = (sigma_epsilon / SE)^2 / var_t
    n_required = int(np.ceil((sigma_epsilon / se_required) ** 2 / var_t))

    # Verify
    mde_check = compute_mde(n_required, sigma_y, r2_nuisance, alpha, power, treatment_share)

    return {
        "n_required": n_required,
        "se_required": se_required,
        "mde_at_n": mde_check["mde"],
        "target_ate": target_ate,
    }


def power_analysis_report(
    n_current: int = 1000,
    sigma_y: float = 1.0,
    r2_nuisance: float = 0.5,
    true_ate_high: float = 0.50,
    true_ate_low: float = 0.08,
) -> str:
    """Generate a formatted power analysis report.

    Computes MDE for the current sample and required N for both
    high and low ATE scenarios.

    Args:
        n_current: Current sample size.
        sigma_y: Standard deviation of the outcome.
        r2_nuisance: R-squared of nuisance function.
        true_ate_high: High ATE scenario (synthetic DGP).
        true_ate_low: Low ATE scenario (realistic calibration).

    Returns:
        Formatted string report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  CAREER-DML: Statistical Power Analysis")
    lines.append("=" * 60)
    lines.append(f"  Current sample size:     N = {n_current:,}")
    lines.append(f"  Outcome std. deviation:  sigma_Y = {sigma_y:.2f}")
    lines.append(f"  Nuisance R-squared:      R2 = {r2_nuisance:.2f}")
    lines.append(f"  Significance level:      alpha = 0.05 (two-sided)")
    lines.append(f"  Target power:            1 - beta = 0.80")
    lines.append("")

    # MDE at current N
    mde_result = compute_mde(n_current, sigma_y, r2_nuisance)
    lines.append(f"  MDE at N = {n_current:,}:")
    lines.append(f"    Minimum Detectable Effect = {mde_result['mde']:.4f}")
    lines.append(f"    MDE as % of sigma_Y       = {mde_result['mde_pct']:.1f}%")
    lines.append(f"    Estimated SE(ATE)          = {mde_result['se_ate']:.4f}")
    lines.append("")

    # Can we detect the high ATE?
    lines.append(f"  Scenario 1: High ATE = {true_ate_high}")
    if true_ate_high >= mde_result["mde"]:
        lines.append(f"    DETECTABLE at N = {n_current:,} (ATE > MDE)")
    else:
        lines.append(f"    NOT DETECTABLE at N = {n_current:,} (ATE < MDE)")
    req_high = compute_required_n(true_ate_high, sigma_y, r2_nuisance)
    lines.append(f"    Required N for 80% power: {req_high['n_required']:,}")
    lines.append("")

    # Can we detect the low ATE?
    lines.append(f"  Scenario 2: Low ATE = {true_ate_low}")
    if true_ate_low >= mde_result["mde"]:
        lines.append(f"    DETECTABLE at N = {n_current:,} (ATE > MDE)")
    else:
        lines.append(f"    NOT DETECTABLE at N = {n_current:,} (ATE < MDE)")
    req_low = compute_required_n(true_ate_low, sigma_y, r2_nuisance)
    lines.append(f"    Required N for 80% power: {req_low['n_required']:,}")
    lines.append("")

    # Conclusion
    lines.append("  CONCLUSION:")
    if true_ate_low < mde_result["mde"]:
        lines.append(f"    The Signal-to-Noise Frontier lies between N = {n_current:,}")
        lines.append(f"    and N = {req_low['n_required']:,}. With the current sample,")
        lines.append(f"    only effects larger than {mde_result['mde']:.4f} are detectable.")
        lines.append(f"    Danish register data (N > 1M) would provide MDE < 0.01,")

        mde_1m = compute_mde(1_000_000, sigma_y, r2_nuisance)
        lines.append(f"    enabling detection of effects as small as {mde_1m['mde']:.4f}.")
    else:
        lines.append(f"    Both scenarios are detectable at N = {n_current:,}.")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    print(power_analysis_report())
