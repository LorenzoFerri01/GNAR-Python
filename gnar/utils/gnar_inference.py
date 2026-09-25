import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from gnar.utils.neighbour_sets import degree_terms, node_degrees
from gnar.utils.gnar_linear_regression import design_matrix
from gnar.utils.gnar_profile_likelihood import ResidualisedSums, fixed_kappa_ols, deviance


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


def jacobian(ts: np.ndarray, A, beta: float, kappa: float) -> np.ndarray:
    """
    Jacobian of the conditional mean mu_{i,t} = alpha_i X_{i,t-1} + beta N_i^(-kappa) S_{i,t-1} with respect to
    theta = (alpha_1, ..., alpha_d, beta, kappa). For the row of node i at time t:

        d mu_{i,t} / d alpha_j = X_{i,t-1} 1{j = i},
        d mu_{i,t} / d beta    = N_i^(-kappa) S_{i,t-1},
        d mu_{i,t} / d kappa   = -beta log N_i N_i^(-kappa) S_{i,t-1},

    with 0 in the network columns for isolated nodes. Rows are node-major, as in design_matrix. This dense matrix is the
    reference for the fast covariance, cov_fast.

    Params:
        ts: np.array. Time series (after any demeaning), time x nodes. Shape (n, d)
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix. Shape (d, d)
        beta: float. Network coefficient.
        kappa: float. Normalisation exponent.

    Returns:
        J: np.array. Shape (d * (n - 1), d + 2)
    """
    D, _ = design_matrix(ts, A, kappa)
    n, d = np.shape(ts)
    _, log_N, _ = degree_terms(node_degrees(A), kappa)
    return np.hstack([D, (-beta * np.repeat(log_N, n - 1) * D[:, d])[:, None]])


def cov_dense(J: np.ndarray, sigma_2: float) -> np.ndarray:
    """
    Reference covariance of the least-squares estimator from the Jacobian, Cov(theta_hat) = sigma^2 (J^T J)^(-1).

    Params:
        J: np.array. Jacobian. Shape (m, k)
        sigma_2: float. Noise variance.

    Returns:
        np.array. Shape (k, k)
    """
    return sigma_2 * np.linalg.inv(J.T @ J)


def cov_fast(M, R: ResidualisedSums, degrees: np.ndarray, beta: float, kappa: float, sigma_2: float, kappa_estimated: bool = True) -> np.ndarray:
    """
    Covariance sigma^2 (J^T J)^(-1) of theta_hat = (alpha_1, ..., alpha_d, beta, kappa) from the per-node sums, in O(d^2)
    (O(d) for the diagonal) instead of forming J. With u_i = (1, -beta log N_i)^T and w_i = N_i^(-kappa) (0 for isolated
    nodes), the blocks of J^T J are

        (alpha_i, alpha_i): xx_i,   (alpha_i, (beta, kappa)): w_i xs_i u_i^T,   ((beta, kappa), (beta, kappa)): sum_i w_i^2 ss_i u_i u_i^T,

    and after concentrating out alpha the (beta, kappa) block is F~ = sum_i w_i^2 d_i u_i u_i^T. With G the d x 2 matrix of
    rows g_i = w_i xs_i / xx_i u_i^T, the block inverse is

        Cov((beta, kappa)) = sigma^2 F~^(-1),   Cov(alpha, (beta, kappa)) = -sigma^2 G F~^(-1),
        Cov(alpha) = sigma^2 (diag(1 / xx_i) + G F~^(-1) G^T).

    With kappa fixed, u_i = 1 and the kappa row and column are NaN (kappa is known). A node with xx_i = 0 (its alpha is not
    estimated) gets NaN in its alpha row and column.

    Params:
        M: NodeSums.
        R: ResidualisedSums.
        degrees: np.array. Degrees N_i. Shape (d,)
        beta: float. Estimate of beta.
        kappa: float. Estimate (or fixed value) of kappa.
        sigma_2: float. Noise variance.
        kappa_estimated: bool. Whether kappa was estimated.

    Returns:
        np.array. Covariance matrix. Shape (d + 2, d + 2)
    """
    d = len(degrees)
    _, log_N, w = degree_terms(degrees, kappa)
    U = np.vstack([np.ones(d), -beta * log_N]).T if kappa_estimated else np.ones((d, 1))
    k = U.shape[1]
    has_lag = M.xx > 0
    inv_xx = np.divide(1.0, M.xx, out=np.zeros(d), where=has_lag)
    F = (U * (w * w * R.d)[:, None]).T @ U
    G = (w * M.xs * inv_xx)[:, None] * U
    F_inv = np.linalg.inv(F)
    cov = np.full((d + 2, d + 2), np.nan)
    cov[:d, :d] = sigma_2 * (np.diag(inv_xx) + G @ F_inv @ G.T)
    cov[:d, d:d + k] = -sigma_2 * G @ F_inv
    cov[d:d + k, :d] = cov[:d, d:d + k].T
    cov[d:d + k, d:d + k] = sigma_2 * F_inv
    no_lag = np.concatenate([~has_lag, [False, False]])
    cov[no_lag, :] = np.nan
    cov[:, no_lag] = np.nan
    return cov


def var_fast(M, R: ResidualisedSums, degrees: np.ndarray, beta: float, kappa: float, sigma_2: float, kappa_estimated: bool = True) -> np.ndarray:
    """
    Diagonal of cov_fast in O(d) without forming the full matrix:

        Var(alpha_i) = sigma^2 (1 / xx_i + g_i F~^(-1) g_i^T),   Var(beta_hat), Var(kappa_hat) = diagonal of sigma^2 F~^(-1).

    Params: as cov_fast.

    Returns:
        np.array. Variances of (alpha_1, ..., alpha_d, beta, kappa). Shape (d + 2,)
    """
    d = len(degrees)
    _, log_N, w = degree_terms(degrees, kappa)
    U = np.vstack([np.ones(d), -beta * log_N]).T if kappa_estimated else np.ones((d, 1))
    k = U.shape[1]
    has_lag = M.xx > 0
    inv_xx = np.divide(1.0, M.xx, out=np.zeros(d), where=has_lag)
    F_inv = np.linalg.inv((U * (w * w * R.d)[:, None]).T @ U)
    G = (w * M.xs * inv_xx)[:, None] * U
    out = np.full(d + 2, np.nan)
    out[:d] = sigma_2 * (inv_xx + np.einsum("ij,jk,ik->i", G, F_inv, G))
    out[:d][~has_lag] = np.nan
    out[d:d + k] = sigma_2 * np.diag(F_inv)
    return out


def wald_ci(estimate: float | np.ndarray, se: float | np.ndarray, level: float = 0.95) -> tuple:
    """
    Wald interval estimate +- z_{(1 + level) / 2} SE.

    Returns:
        (lower, upper)
    """
    z = norm.ppf((1 + level) / 2)
    return estimate - z * se, estimate + z * se


def wald_test(estimate: float, se: float, value: float) -> tuple[float, float]:
    """
    Wald test of kappa = value: z = (kappa_hat - value) / SE(kappa_hat), with the two-sided normal p-value.

    Returns:
        (z, p_value)
    """
    z = (estimate - value) / se
    return float(z), float(2 * norm.sf(abs(z)))


def profile_ci(M, R: ResidualisedSums, degrees: np.ndarray, kappa_hat: float, rss_hat: float, grid: np.ndarray,
               n_obs: int, sigma_2: float | None = None, level: float = 0.95) -> dict:
    """
    Profile-likelihood interval for kappa, {kappa : dev(kappa) <= chi-square(1) quantile at level}, with
    dev(kappa) = n log(RSS(kappa) / RSS(kappa_hat)) (or (RSS(kappa) - RSS(kappa_hat)) / sigma^2 with sigma^2 fixed).

    Each end is found by root-finding (brentq) on dev(kappa) - cutoff, bracketed by the first grid point moving outward
    from kappa_hat at which the deviance exceeds the cutoff. If the deviance stays below the cutoff up to a bound, that
    side is reported as open and the bound is returned as the end. If the deviance dips below the cutoff again beyond an
    end, the confidence set is not an interval and "disconnected" is True.

    Params:
        M: NodeSums.
        R: ResidualisedSums.
        degrees: np.array. Degrees N_i. Shape (d,)
        kappa_hat: float. Estimate of kappa.
        rss_hat: float. RSS(kappa_hat).
        grid: np.array. The grid of kappa values of the profile, spanning kappa_bounds.
        n_obs: int. Number of observations n = d (T - 1).
        sigma_2: float, optional. Known noise variance.
        level: float. Confidence level.

    Returns:
        dict with keys "lower", "upper", "lower_open", "upper_open", "disconnected".
    """
    cutoff = chi2.ppf(level, 1)

    def excess(kappa):
        return deviance(fixed_kappa_ols(M, degrees, float(kappa), R)[2], rss_hat, n_obs, sigma_2) - cutoff

    grid = np.asarray(grid, dtype=float)
    out = {"disconnected": False}
    for side, points in (("lower", grid[grid < kappa_hat][::-1]), ("upper", grid[grid > kappa_hat])):
        inner = kappa_hat
        end, is_open = None, True
        for j, point in enumerate(points):
            if excess(point) > 0:
                end = float(brentq(excess, min(inner, point), max(inner, point), xtol=1e-12))
                is_open = False
                # Beyond the end, the deviance should stay above the cutoff for an interval
                out["disconnected"] |= bool(any(excess(q) <= 0 for q in points[j + 1:]))
                break
            inner = point
        if is_open:
            end = float(grid[0] if side == "lower" else grid[-1])
        out[side], out[f"{side}_open"] = end, is_open
    return out


def beta_at_reference(beta: float, kappa: float, cov_beta_kappa: np.ndarray, N0: float, kappa_estimated: bool = True) -> tuple[float, float]:
    """
    Network coefficient at a reference degree, beta_0 = beta N_0^(-kappa), and its delta-method standard error

        Var(beta_0) ~= beta_0^2 [Var(beta)/beta^2 - 2 log N_0 Cov(beta, kappa)/beta + log^2 N_0 Var(kappa)],

    computed in the equivalent gradient form g^T Cov(beta, kappa) g with g = N_0^(-kappa) (1, -beta log N_0)^T, which needs
    no division by beta. With kappa fixed, Var(beta_0) = N_0^(-2 kappa) Var(beta).

    Params:
        beta: float. Estimate of beta.
        kappa: float. Estimate (or fixed value) of kappa.
        cov_beta_kappa: np.array. Covariance of (beta_hat, kappa_hat). Shape (2, 2)
        N0: float. Reference degree (>= 1).
        kappa_estimated: bool. Whether kappa was estimated.

    Returns:
        (beta_0, se)
    """
    scale = np.exp(-kappa * np.log(N0))
    beta_0 = beta * scale
    if not kappa_estimated:
        return float(beta_0), float(scale * np.sqrt(cov_beta_kappa[0, 0]))
    g = scale * np.array([1.0, -beta * np.log(N0)])
    if not np.all(np.isfinite(cov_beta_kappa)):
        return float(beta_0), float("inf")
    return float(beta_0), float(np.sqrt(g @ cov_beta_kappa @ g))
