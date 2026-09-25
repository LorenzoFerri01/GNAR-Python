import numpy as np

from gnar.utils.neighbour_sets import degree_terms
from gnar.utils.gnar_profile_likelihood import ResidualisedSums


def weighted_degree_sums(g: np.ndarray, degrees: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Sums over nodes with N_i >= 1 of a weight g_i against log N_i, and the two quadratic forms of the (beta, kappa) block:

        S_kappa(g) = sum_i g_i log^2 N_i - (sum_i g_i log N_i)^2 / sum_i g_i,
        S_beta(g)  = sum_i g_i - (sum_i g_i log N_i)^2 / sum_i g_i log^2 N_i.

    When sum_i g_i log^2 N_i = 0 (every node with neighbours has degree 1) S_beta(g) is its limit, sum_i g_i.

    Params:
        g: np.array. Weights g_i. Shape (d,)
        degrees: np.array. Degrees N_i. Shape (d,)

    Returns:
        (sum g, sum g log N, sum g log^2 N, S_kappa(g), S_beta(g))
    """
    mask, log_N, _ = degree_terms(degrees, 1.0)
    g, log_N = np.asarray(g, dtype=float)[mask], log_N[mask]
    s0, s1, s2 = float(np.sum(g)), float(g @ log_N), float(g @ (log_N * log_N))
    S_kappa = s2 - s1 * s1 / s0 if s0 > 0 else 0.0
    S_beta = s0 - s1 * s1 / s2 if s2 > 0 else s0
    return s0, s1, s2, max(S_kappa, 0.0), max(S_beta, 0.0)


def beta_kappa_cov(R: ResidualisedSums, degrees: np.ndarray, beta: float, kappa: float, sigma_2: float, kappa_estimated: bool = True) -> np.ndarray:
    """
    Covariance of (beta_hat, kappa_hat) after concentrating out the node-specific alphas (the fast path). The information
    is

        F~ = sum_i w_i u_i u_i^T,   w_i = N_i^(-2 kappa) d_i,   u_i = (1, -beta log N_i)^T,

    over nodes with N_i >= 1, and Cov = sigma^2 F~^(-1), which in closed form is

        Var(kappa_hat) = sigma^2 / (beta^2 S_kappa(w)),   Var(beta_hat) = sigma^2 / S_beta(w),
        Cov(beta_hat, kappa_hat) = sigma^2 sum_i w_i log N_i / (beta sum_i w_i S_kappa(w)).

    With kappa fixed (known) Var(beta_hat) = sigma^2 / sum_i w_i and the kappa entries are NaN. When kappa is not identified
    (S_kappa(w) = 0, e.g. a regular graph) or beta = 0, Var(kappa_hat) is infinite and the covariance NaN; Var(beta_hat) is
    then infinite too unless every node with neighbours has degree 1, where beta is still identified.

    Params:
        R: ResidualisedSums.
        degrees: np.array. Degrees N_i. Shape (d,)
        beta: float. Estimate of beta.
        kappa: float. Estimate (or fixed value) of kappa.
        sigma_2: float. Noise variance.
        kappa_estimated: bool. Whether kappa was estimated.

    Returns:
        np.array. Covariance matrix of (beta_hat, kappa_hat). Shape (2, 2)
    """
    _, _, w_kappa = degree_terms(degrees, kappa)
    w = w_kappa * w_kappa * R.d
    s0, s1, s2, S_kappa, S_beta = weighted_degree_sums(w, degrees)
    cov = np.full((2, 2), np.nan)
    if not kappa_estimated:
        cov[0, 0] = sigma_2 / s0 if s0 > 0 else np.inf
        return cov
    identified = S_kappa > 1e-12 * s2 and beta != 0
    if not identified:
        cov[1, 1] = np.inf
        cov[0, 0] = sigma_2 / s0 if (s2 == 0 and s0 > 0) else np.inf
        return cov
    cov[1, 1] = sigma_2 / (beta * beta * S_kappa)
    cov[0, 0] = sigma_2 / S_beta
    cov[0, 1] = cov[1, 0] = sigma_2 * s1 / (beta * s0 * S_kappa)
    return cov
