import numpy as np
from typing import NamedTuple
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import chi2

from gnar.utils.neighbour_sets import degree_terms

# Distance from a bound below which the estimate of kappa is reported as lying on the bound
BOUNDARY_TOL = 1e-3
# Relative size of RSS differences treated as ties when looking for local minima on the grid
TIE_RTOL = 1e-12


class NodeSums(NamedTuple):
    """
    Per-node sums of the kappa-normalised GNAR(1, [1]) model over t = 2, ..., T, each of shape (d,):

        xx_i = sum_t X_{i,t-1}^2,         xy_i = sum_t X_{i,t-1} X_{i,t},   xs_i = sum_t X_{i,t-1} S_{i,t-1},
        ss_i = sum_t S_{i,t-1}^2,         sy_i = sum_t S_{i,t-1} X_{i,t},   yy_i = sum_t X_{i,t}^2,

    where S_{i,t-1} = sum_{q in N(i)} X_{q,t-1}. n is the number of time points T.
    """
    xx: np.ndarray
    xy: np.ndarray
    xs: np.ndarray
    ss: np.ndarray
    sy: np.ndarray
    yy: np.ndarray
    n: int


class ResidualisedSums(NamedTuple):
    """
    Sums residualised on the own lag, each of shape (d,):

        c_i = sy_i - xs_i xy_i / xx_i,   d_i = ss_i - xs_i^2 / xx_i,   e_i = yy_i - xy_i^2 / xx_i.

    For a node whose lagged series is identically zero (xx_i = 0) no residualisation is possible and alpha_i is set to
    0, as in gnar_lr; then xs_i = xy_i = 0 and c_i = sy_i, d_i = ss_i, e_i = yy_i.
    """
    c: np.ndarray
    d: np.ndarray
    e: np.ndarray


class KappaEstimate(NamedTuple):
    """
    Result of the profile-likelihood optimisation over kappa.

    kappa is NaN when kappa is not identified; warnings holds (code, message) pairs with code one of "boundary",
    "multimodal", "flat" or "identifiability".
    """
    kappa: float
    rss: float
    grid: np.ndarray
    rss_grid: np.ndarray
    identified: bool
    warnings: list


def node_sums(ts: np.ndarray, A) -> NodeSums:
    """
    Compute the per-node sums (see NodeSums) in one O(nd) pass.

    Params:
        ts: np.array. Time series, time x nodes. Shape (n, d)
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix. Shape (d, d)

    Returns:
        NodeSums
    """
    ts = np.asarray(ts, dtype=float)
    # Neighbour sums S[t, i] = sum_q ts[t, q] A[q, i]
    S = np.asarray(A.T @ ts.T).T
    x, y, s = ts[:-1], ts[1:], S[:-1]
    return NodeSums(xx=np.sum(x * x, axis=0), xy=np.sum(x * y, axis=0), xs=np.sum(x * s, axis=0),
                    ss=np.sum(s * s, axis=0), sy=np.sum(s * y, axis=0), yy=np.sum(y * y, axis=0), n=ts.shape[0])


def residualised_sums(M: NodeSums) -> ResidualisedSums:
    """
    Residualise the neighbour sums and the target on the own lag (see ResidualisedSums).

    Params:
        M: NodeSums.

    Returns:
        ResidualisedSums
    """
    has_lag = M.xx > 0
    xy_ratio = np.divide(M.xy, M.xx, out=np.zeros_like(M.xx), where=has_lag)
    xs_ratio = np.divide(M.xs, M.xx, out=np.zeros_like(M.xx), where=has_lag)
    return ResidualisedSums(c=M.sy - M.xs * xy_ratio, d=M.ss - M.xs * xs_ratio, e=M.yy - M.xy * xy_ratio)


def _beta_parts(R: ResidualisedSums, degrees: np.ndarray, kappa: float) -> tuple[np.ndarray, float, float]:
    # Weights w_i = N_i^(-kappa) (0 for isolated nodes) and the sums sum_i w_i c_i, sum_i w_i^2 d_i over nodes with N_i >= 1
    _, _, w = degree_terms(degrees, kappa)
    return w, float(w @ R.c), float((w * w) @ R.d)


def informative_nodes(M: NodeSums, R: ResidualisedSums, degrees: np.ndarray) -> np.ndarray:
    """
    Nodes that carry information on the network term: N_i >= 1 and a neighbour sum that is not a multiple of the node's
    own lag, d_i > 0 (up to rounding, relative to ss_i). Only these nodes enter beta_hat and the kappa information.

    Returns:
        np.array of bool. Shape (d,)
    """
    return (np.asarray(degrees) > 0) & (R.d > 1e-12 * M.ss)


def has_network_information(M: NodeSums, R: ResidualisedSums, degrees: np.ndarray) -> bool:
    """
    Whether the data carry information on beta: some node with N_i >= 1 has a neighbour sum that is not a multiple of its
    own lag (see informative_nodes).
    """
    return bool(np.any(informative_nodes(M, R, degrees)))


def fixed_kappa_ols(M: NodeSums, degrees: np.ndarray, kappa: float, R: ResidualisedSums | None = None) -> tuple[np.ndarray, float, float]:
    """
    Closed-form least squares for the kappa-normalised GNAR(1, [1]) standard model with kappa fixed:

        beta(kappa)    = sum_i N_i^(-kappa) c_i / sum_i N_i^(-2 kappa) d_i,
        alpha_i(kappa) = (xy_i - beta(kappa) N_i^(-kappa) xs_i) / xx_i,
        RSS(kappa)     = sum_i e_i - (sum_i N_i^(-kappa) c_i)^2 / sum_i N_i^(-2 kappa) d_i,

    where the sums defining beta run over nodes with N_i >= 1. Isolated nodes add e_i to RSS and get the AR(1) estimate
    alpha_i = xy_i / xx_i. If the data carry no information on beta (no edges, or every neighbour sum is a multiple of the
    own lag), beta is set to 0, as gnar_lr does for an all-zero regressor.

    Params:
        M: NodeSums.
        degrees: np.array. Degrees N_i. Shape (d,)
        kappa: float. Normalisation exponent.
        R: ResidualisedSums, optional. Precomputed residualised sums.

    Returns:
        alpha: np.array. Shape (d,)
        beta: float
        rss: float
    """
    if R is None:
        R = residualised_sums(M)
    w, num, den = _beta_parts(R, degrees, kappa)
    if has_network_information(M, R, degrees) and den > 0:
        beta = num / den
        rss = float(np.sum(R.e) - num * beta)
    else:
        beta = 0.0
        rss = float(np.sum(R.e))
    alpha = np.divide(M.xy - beta * w * M.xs, M.xx, out=np.zeros_like(M.xx), where=M.xx > 0)
    return alpha, float(beta), max(rss, 0.0)


def rss_profile(M: NodeSums, degrees: np.ndarray, kappas: np.ndarray, R: ResidualisedSums | None = None) -> np.ndarray:
    """
    Residual sum of squares RSS(kappa) = sum_i e_i - (sum_i N_i^(-kappa) c_i)^2 / sum_i N_i^(-2 kappa) d_i on a grid of
    kappa values, at O(d) cost per value.

    Params:
        M: NodeSums.
        degrees: np.array. Degrees N_i. Shape (d,)
        kappas: np.array. Values of kappa. Shape (k,)
        R: ResidualisedSums, optional. Precomputed residualised sums.

    Returns:
        np.array. RSS(kappa) for each value. Shape (k,)
    """
    if R is None:
        R = residualised_sums(M)
    return np.array([fixed_kappa_ols(M, degrees, float(k), R)[2] for k in np.atleast_1d(kappas)])


def rss_derivative(M: NodeSums, degrees: np.ndarray, kappa: float, R: ResidualisedSums | None = None) -> float:
    """
    Derivative of RSS(kappa) = sum_i e_i - P(kappa)^2 / Q(kappa), with P = sum_i N_i^(-kappa) c_i and
    Q = sum_i N_i^(-2 kappa) d_i:

        RSS'(kappa) = -P (2 P' Q - P Q') / Q^2,   P' = -sum_i log N_i N_i^(-kappa) c_i,   Q' = -2 sum_i log N_i N_i^(-2 kappa) d_i.

    Params:
        M: NodeSums.
        degrees: np.array. Degrees N_i. Shape (d,)
        kappa: float. Normalisation exponent.
        R: ResidualisedSums, optional. Precomputed residualised sums.

    Returns:
        float. dRSS / dkappa
    """
    if R is None:
        R = residualised_sums(M)
    _, log_N, w = degree_terms(degrees, kappa)
    P, Q = float(w @ R.c), float((w * w) @ R.d)
    if Q <= 0:
        return 0.0
    dP, dQ = -float((log_N * w) @ R.c), -2.0 * float((log_N * w * w) @ R.d)
    return -P * (2 * dP * Q - P * dQ) / (Q * Q)


def profile_loglik(rss: float | np.ndarray, n_obs: int, sigma_2: float | None = None) -> float | np.ndarray:
    """
    Gaussian log-likelihood with alpha and beta profiled out, as a function of RSS(kappa). With n = n_obs = d (T - 1):

        sigma^2 estimated:  l_p(kappa) = -(n / 2) log(RSS(kappa) / n) - (n / 2) (1 + log 2 pi),
        sigma^2 fixed:      l_p(kappa) = -RSS(kappa) / (2 sigma^2) - (n / 2) log(2 pi sigma^2).

    Params:
        rss: float or np.array. Residual sum(s) of squares.
        n_obs: int. Number of observations n = d (T - 1).
        sigma_2: float, optional. Known noise variance; None when it is profiled out.

    Returns:
        float or np.array. Profile log-likelihood.
    """
    rss = np.asarray(rss, dtype=float)
    if sigma_2 is None:
        out = -n_obs / 2 * np.log(rss / n_obs) - n_obs / 2 * (1 + np.log(2 * np.pi))
    else:
        out = -rss / (2 * sigma_2) - n_obs / 2 * np.log(2 * np.pi * sigma_2)
    return float(out) if out.ndim == 0 else out


def deviance(rss: float | np.ndarray, rss_min: float, n_obs: int, sigma_2: float | None = None) -> float | np.ndarray:
    """
    Profile deviance dev(kappa) = 2 (l_p(kappa_hat) - l_p(kappa)):

        sigma^2 estimated:  dev(kappa) = n log(RSS(kappa) / RSS(kappa_hat)),
        sigma^2 fixed:      dev(kappa) = (RSS(kappa) - RSS(kappa_hat)) / sigma^2,

    clipped at 0 against rounding.

    Params:
        rss: float or np.array. RSS(kappa).
        rss_min: float. RSS(kappa_hat).
        n_obs: int. Number of observations n = d (T - 1).
        sigma_2: float, optional. Known noise variance; None when it is profiled out.

    Returns:
        float or np.array. Deviance.
    """
    rss = np.asarray(rss, dtype=float)
    if sigma_2 is None:
        out = n_obs * np.log(rss / rss_min)
    else:
        out = (rss - rss_min) / sigma_2
    out = np.maximum(out, 0.0)
    return float(out) if out.ndim == 0 else out


def kappa_identifiable(degrees: np.ndarray, informative: np.ndarray | None = None) -> bool:
    """
    Whether the degrees can identify kappa: at least two nodes with N_i >= 1 (by default), or at least two of the given
    informative nodes, have different degrees. (kappa also needs beta != 0; on a regular graph only beta N^(-kappa) is
    identified.)
    """
    degrees = np.asarray(degrees)
    keep = degrees > 0 if informative is None else np.asarray(informative)
    return np.unique(degrees[keep]).size >= 2


def _count_local_minima(values: np.ndarray, tol: float) -> int:
    # Count local minima of a sequence, including the ends, treating differences of at most tol as ties
    steps = np.diff(values)
    signs = np.sign(steps) * (np.abs(steps) > tol)
    signs = signs[signs != 0]
    if signs.size == 0:
        return 0
    count = int(np.sum((signs[:-1] < 0) & (signs[1:] > 0)))
    count += int(signs[0] > 0) + int(signs[-1] < 0)
    return count


def _polish(M: NodeSums, degrees: np.ndarray, R: ResidualisedSums, x0: float, a: float, b: float) -> float:
    # Near its minimum RSS(kappa) is flat to rounding, so a minimiser resolves kappa only to about sqrt(machine epsilon)
    # times its standard error; the root of the analytic derivative is resolved to machine precision. The root is
    # bracketed around x0 (growing the bracket until RSS' changes sign) so that it is the stationary point the minimiser
    # found, and it is kept only if its RSS is no worse than at x0
    def slope(x):
        return rss_derivative(M, degrees, x, R)

    h = max(1e-9, 1e-6 * (b - a))
    lower, upper = max(a, x0 - h), min(b, x0 + h)
    while not (slope(lower) < 0 < slope(upper)):
        if lower == a and upper == b:
            return x0
        h *= 4
        lower, upper = max(a, x0 - h), min(b, x0 + h)
    root = float(brentq(slope, lower, upper, xtol=1e-15, rtol=4 * np.finfo(float).eps))
    rss_root, rss_x0 = fixed_kappa_ols(M, degrees, root, R)[2], fixed_kappa_ols(M, degrees, x0, R)[2]
    return root if rss_root <= rss_x0 * (1 + TIE_RTOL) else x0


def estimate_kappa(
    M: NodeSums,
    degrees: np.ndarray,
    kappa_bounds: tuple[float, float] = (0.0, 1.5),
    grid: int = 201,
    sigma_2: float | None = None,
    flat_level: float = 0.95,
    R: ResidualisedSums | None = None,
) -> KappaEstimate:
    """
    Estimate kappa by minimising RSS(kappa) (maximising the profile likelihood) over kappa_bounds:

        1. evaluate RSS(kappa) on a grid of `grid` equally spaced points spanning kappa_bounds;
        2. refine the best grid point with bounded Brent (scipy.optimize.minimize_scalar, method="bounded") on its two
           neighbouring grid intervals, then polish Brent's point by solving RSS'(kappa) = 0 (see rss_derivative) with
           brentq on a small bracket around it, which locates the minimum to machine precision rather than to the square
           root of it. When the best grid point is a bound and RSS still decreases towards it, the estimate is that
           bound;
        3. warn when the optimum lies within 1e-3 of a bound ("boundary"), when the grid shows several local minima
           ("multimodal"), or when dev(kappa) stays below the chi-square(1) quantile at level flat_level at both bounds
           ("flat").

    If kappa is not identified, no optimisation is done, kappa is NaN and only the "identifiability" warning is returned.
    That is the case when the data carry no information on the network term, when the nodes that do (see
    informative_nodes) all have the same degree, and when RSS(kappa) is constant to rounding (beta_hat = 0 for every kappa).

    Params:
        M: NodeSums.
        degrees: np.array. Degrees N_i. Shape (d,)
        kappa_bounds: tuple. Lower and upper bound for kappa.
        grid: int. Number of grid points (at least 3).
        sigma_2: float, optional. Known noise variance, used in the deviance of the flat-profile check.
        flat_level: float. Level of the chi-square(1) cutoff for the flat-profile check.
        R: ResidualisedSums, optional. Precomputed residualised sums.

    Returns:
        KappaEstimate
    """
    lo, hi = (float(b) for b in kappa_bounds)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("kappa_bounds must be two finite numbers with lower < upper.")
    if int(grid) != grid or grid < 3:
        raise ValueError("grid must be an integer of at least 3.")
    if R is None:
        R = residualised_sums(M)
    kappas = np.linspace(lo, hi, int(grid))
    rss_grid = rss_profile(M, degrees, kappas, R)
    n_obs = len(degrees) * (M.n - 1)

    informative = informative_nodes(M, R, degrees)
    message = None
    if not np.any(informative):
        message = "kappa is not identified: the data carry no information on the network term (no edges, or neighbour sums proportional to the own lags)."
    elif not kappa_identifiable(degrees, informative):
        common = np.unique(np.asarray(degrees)[informative])
        message = (f"kappa is not identified: every node that carries information on the network term has degree {common[0]:g}, "
                   "so only beta * N^(-kappa) is identified (use beta_at at that degree).")
    elif np.ptp(rss_grid) <= TIE_RTOL * np.min(rss_grid):
        message = "kappa is not identified: RSS(kappa) is constant, so beta_hat = 0 for every kappa (no network effect in the data)."
    if message is not None:
        return KappaEstimate(np.nan, float(np.min(rss_grid)), kappas, rss_grid, False, [("identifiability", message)])

    k = int(np.argmin(rss_grid))
    last = len(kappas) - 1
    if k == last and rss_derivative(M, degrees, hi, R) <= 0:
        # RSS still decreases at the upper bound: the constrained optimum is the bound itself
        kappa_hat = hi
    elif k == 0 and rss_derivative(M, degrees, lo, R) >= 0:
        kappa_hat = lo
    else:
        a, b = kappas[max(k - 1, 0)], kappas[min(k + 1, last)]
        res = minimize_scalar(lambda x: fixed_kappa_ols(M, degrees, float(x), R)[2], bounds=(a, b), method="bounded",
                              options={"xatol": 1e-10})
        kappa_hat = _polish(M, degrees, R, float(res.x), a, b)
        # Keep the grid point if it is genuinely better (beyond rounding) than the refined point
        if rss_grid[k] < fixed_kappa_ols(M, degrees, kappa_hat, R)[2] * (1 - TIE_RTOL):
            kappa_hat = float(kappas[k])
    rss_hat = fixed_kappa_ols(M, degrees, kappa_hat, R)[2]

    warns = []
    if min(abs(kappa_hat - lo), abs(kappa_hat - hi)) < BOUNDARY_TOL:
        warns.append(("boundary", f"The estimate kappa = {kappa_hat:.6g} lies on the boundary of kappa_bounds = ({lo:g}, {hi:g}); "
                                  "the Wald standard error and test are not valid there."))
    if _count_local_minima(rss_grid, TIE_RTOL * rss_hat) > 1:
        warns.append(("multimodal", "The profile of RSS(kappa) has several local minima on the grid; the profile confidence set may not be an interval."))
    cutoff = chi2.ppf(flat_level, 1)
    dev_bounds = deviance(np.array([rss_grid[0], rss_grid[-1]]), rss_hat, n_obs, sigma_2)
    if np.all(dev_bounds < cutoff):
        warns.append(("flat", f"The profile deviance stays below the chi-square(1) {flat_level:.0%} cutoff at both bounds: kappa is weakly identified "
                              "(beta may be close to 0 or the degrees too similar)."))
    return KappaEstimate(kappa_hat, rss_hat, kappas, rss_grid, True, warns)
