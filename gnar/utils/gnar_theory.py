import warnings

import numpy as np

from gnar.utils.neighbour_sets import degree_terms
from gnar.utils.gnar_profile_likelihood import kappa_identifiable
from gnar.utils.gnar_inference import weighted_degree_sums
from gnar.utils.simulating import _gnar1_setup, stationary_cov, row_sum_bound


def _per_period(n: int | None) -> float:
    # Factor 1 / (T - 1) turning per-period quantities into variances for a series of T = n time points
    if n is None:
        return 1.0
    if n < 2:
        raise ValueError("n (the number of time points T) must be at least 2.")
    return 1.0 / (n - 1)


def _network_moments(A, alpha, beta: float, kappa: float, sigma_2: float) -> tuple:
    # Gamma0 and, for each node, Cov(S_i, X_i) = (A Gamma0)_ii and Var(S_i) = (A Gamma0 A^T)_ii
    A, alpha, beta, kappa, degrees = _gnar1_setup(A, alpha, beta, kappa)
    Gamma0 = stationary_cov(A, alpha, beta, kappa, sigma_2)
    AG = np.asarray(A @ Gamma0)
    cov_sx = np.diag(AG).copy()
    var_s = np.einsum("ij,ij->i", AG, A.toarray())
    return A, alpha, beta, kappa, degrees, Gamma0, cov_sx, var_s


def asymptotic_cov(A, alpha, beta: float, kappa: float, sigma_2: float = 1.0, n: int | None = None) -> np.ndarray:
    """
    Exact asymptotic covariance of theta_hat = (alpha_1, ..., alpha_d, beta, kappa) from the true parameters and the
    graph. With Gamma0 the stationary covariance (Gamma0 = Phi Gamma0 Phi^T + sigma^2 I), Cov(S_i, X_i) = (A Gamma0)_ii and
    Var(S_i) = (A Gamma0 A^T)_ii, the per-period information matrix F has entries

        (alpha_i, alpha_i): Gamma0_ii,                              (alpha_i, alpha_j), i != j: 0,
        (alpha_i, beta):    N_i^(-kappa) Cov(S_i, X_i),             (alpha_i, kappa): -beta log N_i N_i^(-kappa) Cov(S_i, X_i),
        (beta, beta):       sum_i N_i^(-2 kappa) Var(S_i),          (beta, kappa):    -beta sum_i log N_i N_i^(-2 kappa) Var(S_i),
        (kappa, kappa):     beta^2 sum_i log^2 N_i N_i^(-2 kappa) Var(S_i),

    with sums over nodes with N_i >= 1, and the asymptotic covariance is sigma^2 F^(-1) / (T - 1). Since Gamma0 scales with
    sigma^2, the result does not depend on sigma_2.

    If kappa is not identified a warning is raised and Var(kappa_hat) is infinite:
        - every node with neighbours has degree 1: N_i^(-kappa) = 1, so the (alpha, beta) block is the one with kappa
          known;
        - all nodes with neighbours share a degree above 1 (a regular graph): only beta N^(-kappa) is identified, so the
          alpha block is the one with kappa known and Var(beta_hat) is infinite;
        - beta = 0 (otherwise): kappa does not enter the model, the estimators are non-regular and no information-matrix
          variance applies, so the other entries are NaN (use a small non-zero beta for the limit).

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        alpha: float or np.array. Autoregressive coefficients. Shape (d,) or scalar.
        beta: float. Network coefficient.
        kappa: float. Normalisation exponent.
        sigma_2: float. Noise variance. Defaults to 1.
        n: int, optional. Number of time points T. None returns the per-period covariance sigma^2 F^(-1) (divide by T - 1).

    Returns:
        np.array. Shape (d + 2, d + 2)
    """
    A, alpha, beta, kappa, degrees, Gamma0, cov_sx, var_s = _network_moments(A, alpha, beta, kappa, sigma_2)
    d = len(degrees)
    _, log_N, w = degree_terms(degrees, kappa)
    F = np.zeros((d + 2, d + 2))
    F[np.arange(d), np.arange(d)] = np.diag(Gamma0)
    F[:d, d] = F[d, :d] = w * cov_sx
    F[:d, d + 1] = F[d + 1, :d] = -beta * log_N * w * cov_sx
    F[d, d] = np.sum(w * w * var_s)
    F[d, d + 1] = F[d + 1, d] = -beta * np.sum(log_N * w * w * var_s)
    F[d + 1, d + 1] = beta ** 2 * np.sum(log_N ** 2 * w * w * var_s)
    factor = sigma_2 * _per_period(n)
    if kappa_identifiable(degrees) and beta != 0:
        return factor * np.linalg.inv(F)
    warnings.warn("kappa is not identified (a regular graph, or beta = 0): its asymptotic variance is infinite.", UserWarning, stacklevel=2)
    cov = np.full((d + 2, d + 2), np.nan)
    cov[d + 1, d + 1] = np.inf
    degree_one = bool(np.all(degrees[degrees > 0] == 1))
    if beta == 0 and not degree_one:
        # kappa does not enter the model: non-regular estimators, no information-matrix variance
        return cov
    # The information with kappa known gives the identified block: (alpha, beta) when every node with neighbours has
    # degree 1 (N_i^(-kappa) = 1 for all kappa), alpha alone on a regular graph (only beta N^(-kappa) is identified)
    known = factor * np.linalg.inv(F[:d + 1, :d + 1])
    keep = d + 1 if degree_one else d
    cov[:keep, :keep] = known[:keep, :keep]
    if not degree_one:
        cov[d, d] = np.inf
    return cov


def closed_form_variances(A, alpha, beta: float, kappa: float, sigma_2: float = 1.0, n: int | None = None) -> dict:
    """
    Closed forms of the asymptotic variances of beta_hat and kappa_hat. With v_i = Var(S_i) - Cov(S_i, X_i)^2 / Gamma0_ii
    and w_i = N_i^(-2 kappa) v_i,

        Var(kappa_hat) ~= sigma^2 / ((T - 1) beta^2 S_kappa(w)),   Var(beta_hat) ~= sigma^2 / ((T - 1) S_beta(w)),

    where S_kappa(g) = sum_i g_i log^2 N_i - (sum_i g_i log N_i)^2 / sum_i g_i and
    S_beta(g) = sum_i g_i - (sum_i g_i log N_i)^2 / sum_i g_i log^2 N_i, over nodes with N_i >= 1. On a regular graph
    S_kappa = 0 and the variances are infinite (with a warning), except Var(beta_hat) when every node with neighbours
    has degree 1. At beta = 0 kappa does not enter the model and Var(beta_hat) is NaN (non-regular; see asymptotic_cov).

    Params: as asymptotic_cov.

    Returns:
        dict with keys "kappa" and "beta".
    """
    A, alpha, beta, kappa, degrees, Gamma0, cov_sx, var_s = _network_moments(A, alpha, beta, kappa, sigma_2)
    _, _, w_kappa = degree_terms(degrees, kappa)
    v = var_s - cov_sx ** 2 / np.diag(Gamma0)
    _, _, s2, S_kappa, S_beta = weighted_degree_sums(w_kappa ** 2 * v, degrees)
    factor = sigma_2 * _per_period(n)
    if not kappa_identifiable(degrees) or beta == 0:
        warnings.warn("kappa is not identified (a regular graph, or beta = 0): S_kappa = 0 and its variance is infinite.", UserWarning, stacklevel=2)
        if s2 == 0:
            var_beta = factor / S_beta
        else:
            var_beta = np.nan if beta == 0 else np.inf
        return {"kappa": np.inf, "beta": var_beta}
    return {"kappa": factor / (beta ** 2 * S_kappa), "beta": factor / S_beta}


def variance_bounds(A, alpha, beta: float, kappa: float, n: int | None = None) -> dict:
    """
    Degree-based bounds on the asymptotic variances, for sigma^2 = 1 (the variances do not depend on sigma^2). Write S(N^p)
    for S evaluated at g_i = N_i^p (nodes with N_i >= 1), and gamma_bar = max_q Gamma0_qq <= 1 / (1 - rho^2), where
    rho = ||Phi||_inf:

        1 / ((T-1) beta^2 gamma_bar S_kappa(N^(2-2kappa))) <= Var(kappa_hat) <= 1 / ((T-1) beta^2 S_kappa(N^(1-2kappa))),
        1 / ((T-1) gamma_bar S_beta(N^(2-2kappa)))         <= Var(beta_hat)  <= 1 / ((T-1) S_beta(N^(1-2kappa))),
        1 / ((T-1) Gamma0_ii)                               <= Var(alpha_i)   <= 1 / (T-1).

    They follow from N_i <= v_i <= gamma_bar N_i^2 for v_i = Var(S_i | X_i), because S_kappa and S_beta are
    non-decreasing and homogeneous of degree one in g. Each lower bound is returned twice: with the exact gamma_bar (or
    Gamma0_ii), and with 1 / (1 - rho^2), which needs only the graph and the parameters (NaN, with a warning, if rho >= 1).

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        alpha: float or np.array. Autoregressive coefficients. Shape (d,) or scalar.
        beta: float. Network coefficient.
        kappa: float. Normalisation exponent.
        n: int, optional. Number of time points T. None returns per-period bounds (divide by T - 1).

    Returns:
        dict with keys "kappa" and "beta" (each a dict with "lower", "lower_rho" and "upper"), "alpha" (the same keys with
        arrays of shape (d,)), "gamma_bar" and "rho".
    """
    A, alpha, beta, kappa, degrees, Gamma0, _, _ = _network_moments(A, alpha, beta, kappa, 1.0)
    factor = _per_period(n)
    rho = row_sum_bound(A, alpha, beta, kappa)
    gamma_bar = float(np.max(np.diag(Gamma0)))
    if rho < 1:
        gamma_rho = 1.0 / (1.0 - rho ** 2)
    else:
        warnings.warn(f"rho = {rho:.6g} >= 1, so the bounds based on 1 / (1 - rho^2) are not available.", UserWarning, stacklevel=2)
        gamma_rho = np.nan
    mask, log_N, _ = degree_terms(degrees, kappa)
    g1, g2 = np.zeros_like(degrees), np.zeros_like(degrees)
    g1[mask] = np.exp((1 - 2 * kappa) * log_N[mask])
    g2[mask] = np.exp((2 - 2 * kappa) * log_N[mask])
    _, _, _, Sk1, Sb1 = weighted_degree_sums(g1, degrees)
    _, _, _, Sk2, Sb2 = weighted_degree_sums(g2, degrees)
    identified = kappa_identifiable(degrees) and beta != 0
    if not identified:
        warnings.warn("kappa is not identified (a regular graph, or beta = 0): S_kappa = 0 and its bounds are infinite.", UserWarning, stacklevel=2)

    def inv(x):
        return np.inf if x == 0 or not np.isfinite(x) else 1.0 / x

    b2 = beta ** 2
    out = {
        "kappa": {"lower": factor * inv(b2 * gamma_bar * Sk2), "lower_rho": factor * inv(b2 * gamma_rho * Sk2) if np.isfinite(gamma_rho) else np.nan,
                  "upper": factor * inv(b2 * Sk1)},
        "beta": {"lower": factor * inv(gamma_bar * Sb2), "lower_rho": factor * inv(gamma_rho * Sb2) if np.isfinite(gamma_rho) else np.nan,
                 "upper": factor * inv(Sb1)},
        "alpha": {"lower": factor / np.diag(Gamma0), "lower_rho": factor * np.full(len(degrees), 1 - rho ** 2) if rho < 1 else np.full(len(degrees), np.nan),
                  "upper": np.full(len(degrees), factor)},
        "gamma_bar": gamma_bar,
        "rho": rho,
    }
    return out
