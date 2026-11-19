from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse.linalg import eigsh
from scipy.stats import chi2, norm

from gnar.utils.data_utils import set_mean
from gnar.utils.simulating import generate_noise
from gnar.var import VAR

# Graphs with at most this many nodes use dense symmetric eigensolvers for the extreme eigenvalues; larger graphs use
# sparse Lanczos iterations (scipy.sparse.linalg.eigsh), which cost O(number of edges) per iteration.
_DENSE_EIG_MAX_NODES = 200
# Number of equally spaced kappa values in [0, 1] at which the profile likelihood is evaluated before refinement
_PROFILE_GRID_SIZE = 201


def _as_csr(A) -> sp.csr_array:
    """Convert a dense NumPy array or a SciPy sparse matrix/array to a float CSR array without explicit zeros."""
    if sp.issparse(A):
        A = sp.csr_array(A, dtype=float)
    elif isinstance(A, np.ndarray):
        A = sp.csr_array(np.asarray(A, dtype=float))
    else:
        raise ValueError("Adjacency matrix A must be a NumPy array or a SciPy sparse matrix.")
    A.eliminate_zeros()
    return A


def check_kappa_adjacency(A) -> None:
    """
    Check that A (a NumPy array or SciPy sparse matrix) is a valid adjacency matrix for the kappa-GNAR model: square,
    binary, symmetric, without self-loops or isolated nodes.
    """
    if not (sp.issparse(A) or isinstance(A, np.ndarray)) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix A must be a square NumPy array or SciPy sparse matrix.")
    A = _as_csr(A)
    if not np.all(A.data == 1):
        raise ValueError("Adjacency matrix A must be binary (0 or 1): the kappa-GNAR model is defined for unweighted networks.")
    if (A != A.T).nnz > 0:
        raise ValueError("Adjacency matrix A must be symmetric: the kappa-GNAR model is defined for undirected networks.")
    if np.any(A.diagonal() != 0):
        raise ValueError("Adjacency matrix A must not contain self-loops.")
    isolated = np.flatnonzero(np.asarray(A.sum(axis=1)).ravel() == 0)
    if isolated.size > 0:
        shown = isolated[:10].tolist()
        raise ValueError(f"The graph must not contain isolated nodes (found {isolated.size}, e.g. at indices {shown}).")


def _degrees(A: sp.csr_array) -> np.ndarray:
    return np.asarray(A.sum(axis=1)).ravel()


def kappa_weight_matrix(A, kappa: float):
    """
    Compute D^{-kappa} A, where D is the diagonal degree matrix. Row i holds the weights node i assigns to its
    neighbours: (D^{-kappa} A x)_i = d_i^{-kappa} * sum_{j in N(i)} x_j. Assumes A has no isolated nodes.
    Returns a CSR array if A is sparse, and a dense array otherwise.
    """
    A_csr = _as_csr(A)
    W = (sp.diags_array(_degrees(A_csr) ** -kappa) @ A_csr).tocsr()
    return W if sp.issparse(A) else W.toarray()


def _symmetrised_weight_matrix(A: sp.csr_array, kappa: float) -> sp.csr_array:
    """
    S = D^{-kappa/2} A D^{-kappa/2} (sparse), which is symmetric and similar to D^{-kappa} A
    (S = D^{kappa/2} (D^{-kappa} A) D^{-kappa/2}), so it has the same, real, eigenvalues.
    """
    s = sp.diags_array(_degrees(A) ** (-kappa / 2))
    return (s @ A @ s).tocsr()


def _spectral_info(S: sp.csr_array) -> tuple[float, float, np.ndarray]:
    """
    Smallest eigenvalue, largest eigenvalue (Perron root) and unit-norm Perron eigenvector of the sparse symmetric
    non-negative matrix S. The Perron vector is returned with non-negative entries.
    """
    N = S.shape[0]
    if N <= _DENSE_EIG_MAX_NODES:
        ev, vecs = eigh(S.toarray())
        smallest, largest, v = ev[0], ev[-1], vecs[:, -1]
    else:
        # Deterministic starting vectors: a positive vector is close to the Perron vector for the largest eigenvalue
        vals, vecs = eigsh(S, k=1, which="LA", v0=np.ones(N))
        largest, v = vals[0], vecs[:, 0]
        smallest = eigsh(S, k=1, which="SA", v0=np.random.default_rng(0).standard_normal(N),
                         return_eigenvectors=False)[0]
    v = v / np.linalg.norm(v)
    return float(smallest), float(largest), np.abs(v)


def kappa_spectral_radius(A, kappa: float) -> float:
    """
    Spectral radius rho_kappa of D^{-kappa} A for a symmetric adjacency matrix without isolated nodes. Computed as the
    largest eigenvalue of the similar symmetric matrix D^{-kappa/2} A D^{-kappa/2}; since that matrix is non-negative, its
    largest eigenvalue is its spectral radius (Perron root).
    """
    return kappa_spectrum_bounds(A, kappa)[1]


def kappa_spectrum_bounds(A, kappa: float) -> tuple[float, float]:
    """
    Smallest and largest eigenvalues (mu_min, rho_kappa) of D^{-kappa} A for a symmetric adjacency matrix without isolated
    nodes. Only the two extreme eigenvalues of the similar symmetric matrix D^{-kappa/2} A D^{-kappa/2} are computed:
    with a dense solver for small graphs, and with sparse Lanczos iterations (cost per iteration proportional to the number
    of edges) for graphs with more than 200 nodes.
    """
    smallest, largest, _ = _spectral_info(_symmetrised_weight_matrix(_as_csr(A), kappa))
    return smallest, largest


class _ProfileStatistics:
    """
    Sufficient statistics for fitting the kappa-GNAR model by least squares, and the profile likelihood in kappa.

    With y_t the demeaned data and S_t = A y_{t-1} the raw neighbour sums, the model for fixed kappa is the pooled
    regression y_{i,t} = alpha y_{i,t-1} + beta d_i^{-kappa} S_{i,t} + e_{i,t}. Since kappa only enters through the per-node
    factor d_i^{-kappa}, all the sums needed for the 2 x 2 normal equations are weighted sums over nodes of the vectors
    q_i = sum_t S_{i,t}^2, r_i = sum_t y_{i,t-1} S_{i,t} and u_i = sum_t y_{i,t} S_{i,t}. After a single sparse product
    (O(T x edges)), each evaluation of the profile likelihood therefore costs O(N).
    """

    def __init__(self, Y: np.ndarray, A: sp.csr_array, log_deg: np.ndarray) -> None:
        y_lag, y_cur = Y[:-1], Y[1:]
        S = np.asarray(A @ y_lag.T).T                      # (T - 1, N) raw neighbour sums
        self.sxx = float(np.einsum("ij,ij->", y_lag, y_lag))
        self.sxy = float(np.einsum("ij,ij->", y_lag, y_cur))
        self.syy = float(np.einsum("ij,ij->", y_cur, y_cur))
        self.q = np.einsum("ij,ij->j", S, S)
        self.r = np.einsum("ij,ij->j", y_lag, S)
        self.u = np.einsum("ij,ij->j", y_cur, S)
        self.log_deg = log_deg
        self.n_obs = y_cur.size                            # (T - 1) N
        if self.sxx == 0:
            raise ValueError("The (demeaned) time series is identically zero: the model cannot be fitted.")

    def solve(self, kappa) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Least-squares (alpha, beta) and RSS at one or several values of kappa (vectorised over kappa).
        """
        kappa = np.atleast_1d(np.asarray(kappa, dtype=float))
        w = np.exp(-np.outer(kappa, self.log_deg))        # (K, N): d_i^{-kappa}
        szz = (w * w) @ self.q
        sxz = w @ self.r
        szy = w @ self.u
        det = self.sxx * szz - sxz ** 2
        if np.any(det <= 0):
            raise ValueError("The regressors are collinear: the model cannot be fitted (is the network effect absent?).")
        alpha = (szz * self.sxy - sxz * szy) / det
        beta = (self.sxx * szy - sxz * self.sxy) / det
        rss = self.syy - alpha * self.sxy - beta * szy
        return alpha, beta, np.maximum(rss, np.finfo(float).tiny)

    def loglik(self, rss) -> np.ndarray:
        """Profile Gaussian log-likelihood -(M/2)(log(2 pi RSS / M) + 1), M = (T - 1) N."""
        m = self.n_obs
        return -0.5 * m * (np.log(2 * np.pi * np.asarray(rss) / m) + 1)

    def gram(self, beta: float, kappa: float, with_kappa: bool) -> np.ndarray:
        """
        J^T J for the parameters (alpha, beta[, kappa]), where the columns of the Jacobian J of the fitted values are
        y_{i,t-1}, d_i^{-kappa} S_{i,t} and -beta log(d_i) d_i^{-kappa} S_{i,t}.
        """
        w = np.exp(-kappa * self.log_deg)
        xx, xz, zz = self.sxx, w @ self.r, (w * w) @ self.q
        if not with_kappa:
            return np.array([[xx, xz], [xz, zz]])
        lw = self.log_deg * w
        xk = -beta * (lw @ self.r)
        zk = -beta * ((lw * w) @ self.q)
        kk = beta ** 2 * ((lw * lw) @ self.q)
        return np.array([[xx, xz, xk], [xz, zz, zk], [xk, zk, kk]])


class KappaGNAR:
    """
    Global GNAR(1, [1]) model with a degree-normalisation exponent kappa, for unweighted undirected networks:

        X_{i,t} - mu_i = alpha (X_{i,t-1} - mu_i) + beta d_i^{-kappa} sum_{j in N(i)} (X_{j,t-1} - mu_j) + e_{i,t},
        e_t ~ N(0, sigma_2 I).

    The default parametrisation uses the spectrally normalised network coefficient

        b = beta * rho_kappa,    rho_kappa = spectral radius of D^{-kappa} A,

    so that, in VAR(1) form, X_t - mu = Phi (X_{t-1} - mu) + e_t with Phi = alpha I + (b / rho_kappa) D^{-kappa} A.
    kappa = 1 gives the standard global GNAR(1, [1]) model (rho_1 = 1, so b = beta); kappa = 0 uses raw neighbour sums.

    The model can be initialised either by fitting it to data (pass ts) or from known parameters (pass alpha and b or beta).

    The graph is stored as a sparse matrix, so construction, fitting, stationarity checks, simulation and forecasting all
    scale with the number of edges. Only the second moments (autocovariances) are dense and cost O(N^3).

    Parameters:
        A (np.ndarray or scipy sparse matrix): Binary, symmetric adjacency matrix without self-loops or isolated nodes.
            Shape (N, N).
        alpha (float): Autoregressive coefficient, shared by all nodes. Only used if ts is not provided.
        b (float): Spectrally normalised network coefficient. Provide exactly one of b and beta when ts is not provided.
        kappa (float or None): Degree-normalisation exponent, in [0, 1]. When building from parameters, None means 1.
            When fitting, None means that kappa is estimated, and a value fixes it.
        beta (float): Unnormalised network coefficient, beta = b / rho_kappa. Alternative to b.
        mean (float, int, np.ndarray or pd.DataFrame): The mean of the process. Only used if ts is not provided. Defaults to 0.
        sigma_2 (float): Noise variance, common to all nodes. Only used if ts is not provided. Defaults to 1.
        names (list-like): Node names, in the order of the rows of A. Defaults to 0, ..., N - 1 (matching the node labels of
            networkx.from_numpy_array), or to the columns of ts if it is a DataFrame.
        ts (np.ndarray or pd.DataFrame): Time series to fit the model to. Shape (T, N).
        demean (bool): Whether to remove the sample mean of each node before fitting. Only used if ts is provided.
            Defaults to True.

    Methods:
        fit: Fit the model to time series data (estimating kappa or keeping it fixed).
        summary / kappa_confint: Estimates, standard errors and confidence intervals.
        loglik / aic / bic: Gaussian log-likelihood and information criteria of a fitted model.
        plot_profile: Plot the profile likelihood of kappa.
        phi: Sparse VAR(1) coefficient matrix.
        to_var: Convert to an equivalent (dense) VAR(1) model.
        is_stationary / phi_spectral_radius: Exact stationarity check and spectral radius of Phi.
        is_regular: Whether all nodes have the same degree (kappa is then not identifiable).
        simulate: Simulate a realisation from the model.
        predict: Forecast future values.
        compute_autocov_mats / compute_autocorr_mats / marginal_variances: Theoretical second moments.
        clear_cache: Free the cached eigendecomposition used for the second moments.
    """

    def __init__(
        self,
        A,
        alpha: float | None = None,
        b: float | None = None,
        kappa: float | None = None,
        *,
        beta: float | None = None,
        mean: float | int | np.ndarray | pd.DataFrame = 0,
        sigma_2: float = 1.0,
        names=None,
        ts: np.ndarray | pd.DataFrame | None = None,
        demean: bool = True,
    ) -> None:
        check_kappa_adjacency(A)
        self._A = _as_csr(A)
        self._N = self._A.shape[0]
        self._degrees = _degrees(self._A)
        self._log_deg = np.log(self._degrees)
        self._names = self._make_names(names)
        self._ts = None
        self._fit = None

        if ts is not None:
            if alpha is not None or b is not None or beta is not None:
                raise ValueError("Provide either the time series ts (to fit the model) or the parameters, not both.")
            self.fit(ts, kappa=kappa, demean=demean)
        else:
            if alpha is None:
                raise ValueError("Either the time series ts or the model parameters (alpha and b or beta) are required.")
            self._set_parameters(alpha, b, beta, 1.0 if kappa is None else kappa, mean, sigma_2)

    def _make_names(self, names) -> pd.Index:
        index = pd.RangeIndex(self._N) if names is None else pd.Index(names)
        if len(index) != self._N:
            raise ValueError(f"The number of names ({len(index)}) does not match the number of nodes ({self._N}).")
        if not index.is_unique:
            raise ValueError("Node names must be unique.")
        return index

    @staticmethod
    def _check_kappa(kappa) -> float:
        if not np.isscalar(kappa) or not 0 <= kappa <= 1:
            raise ValueError("kappa must be a scalar in [0, 1].")
        return float(kappa)

    def _set_parameters(self, alpha, b, beta, kappa, mean, sigma_2) -> None:
        kappa = self._check_kappa(kappa)
        if (b is None) == (beta is None):
            raise ValueError("Provide exactly one of b (spectrally normalised) and beta (unnormalised).")
        if np.ndim(sigma_2) != 0 or not sigma_2 > 0:
            raise ValueError("sigma_2 must be a positive scalar: the noise variance is common to all nodes.")
        self.alpha = float(alpha)
        self.kappa = kappa
        self._mu_min, self.rho, self._perron = _spectral_info(_symmetrised_weight_matrix(self._A, kappa))
        self.b = float(b) if b is not None else float(beta) * self.rho
        self.mu = set_mean(mean, self._N)
        self.sigma_2 = float(sigma_2)
        self._phi = None
        self._eig_cache = None
        if not self.is_stationary():
            warnings.warn("The kappa-GNAR model is non-stationary based on the provided parameters!", UserWarning)

    # ------------------------------------------------------------------ fitting
    def fit(self, ts: np.ndarray | pd.DataFrame, kappa: float | None = None, demean: bool = True) -> None:
        """
        Fit the model to time series data by Gaussian (conditional) maximum likelihood, i.e. pooled least squares.

        For fixed kappa, (alpha, beta) solve a 2 x 2 least-squares problem; kappa is estimated by maximising the profile
        likelihood over [0, 1]: on a grid of 201 values, then refined with Brent's method, always comparing with the exact
        endpoints so that the estimate can be exactly 0 or 1. Each evaluation of the profile costs O(N) after a single
        sparse product, so fitting scales to large graphs. The noise variance is estimated as RSS / ((T - 1) N).

        Parameters:
            ts (np.ndarray or pd.DataFrame): Time series of shape (T, N). If a DataFrame, its columns become the node names.
            kappa (float or None): None to estimate kappa; a value in [0, 1] to keep it fixed. Defaults to None.
            demean (bool): Whether to remove the sample mean of each node. Defaults to True.
        """
        is_df = isinstance(ts, pd.DataFrame)
        values = ts.to_numpy(dtype=float) if is_df else np.asarray(ts, dtype=float)
        if values.ndim != 2 or values.shape[1] != self._N:
            raise ValueError(f"ts must have shape (T, {self._N}): one column per node of the graph.")
        if values.shape[0] < 3:
            raise ValueError("At least three time steps are required to fit the model.")
        if not np.all(np.isfinite(values)):
            raise ValueError("ts contains missing or infinite values.")
        estimate_kappa = kappa is None
        if not estimate_kappa:
            kappa = self._check_kappa(kappa)
        elif self.is_regular():
            raise ValueError("The graph is regular, so kappa is not identifiable: fix it, e.g. fit(ts, kappa=1.0).")

        names = self._make_names(ts.columns) if is_df else self._names
        mu = values.mean(axis=0, keepdims=True) if demean else np.zeros((1, self._N))
        stats = _ProfileStatistics(values - mu, self._A, self._log_deg)

        grid = np.linspace(0.0, 1.0, _PROFILE_GRID_SIZE)
        g_alpha, g_beta, g_rss = stats.solve(grid)
        if estimate_kappa:
            kappa = self._maximise_profile(stats, grid, g_rss)
        alpha, beta, rss = (x[0] for x in stats.solve(kappa))

        self._names = names
        self._ts = ts.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)        # stationarity is checked below with a fit-specific message
            self._set_parameters(alpha, None, beta, kappa, mu, rss / stats.n_obs)
        self._fit = {
            "stats": stats,
            "kappa_estimated": estimate_kappa,
            "T": values.shape[0],
            "rss": float(rss),
            "profile": pd.DataFrame({"kappa": grid, "loglik": stats.loglik(g_rss), "alpha": g_alpha, "beta": g_beta}),
        }
        if not self.is_stationary():
            warnings.warn("The fitted kappa-GNAR model is non-stationary.", UserWarning)
        if estimate_kappa:
            self._check_identification()

    @staticmethod
    def _maximise_profile(stats: _ProfileStatistics, grid: np.ndarray, rss: np.ndarray) -> float:
        """Minimise RSS(kappa) on [0, 1]: grid search, Brent refinement around the best grid point, exact endpoints."""
        i = int(np.argmin(rss))
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
        refined = minimize_scalar(lambda k: stats.solve(k)[2][0], bounds=(lo, hi), method="bounded",
                                  options={"xatol": 1e-10})
        candidates = np.array([grid[i], refined.x, 0.0, 1.0])
        best = candidates[np.argmin(stats.solve(candidates)[2])]
        # Warn if the profile has a second local maximum that is not clearly worse than the best one
        interior = (rss[1:-1] < rss[:-2]) & (rss[1:-1] < rss[2:])
        local = np.r_[rss[0] < rss[1], interior, rss[-1] < rss[-2]]
        if local.sum() > 1:
            ll = stats.loglik(rss[local])
            if np.sort(ll)[-2] > ll.max() - chi2.ppf(0.95, 1) / 2:
                warnings.warn("The profile likelihood of kappa has several local maxima of similar height; the estimate "
                              "of kappa may be unstable. Inspect plot_profile().", UserWarning)
        return float(best)

    def _require_fit(self) -> dict:
        if self._fit is None:
            raise ValueError("The model was not fitted to data.")
        return self._fit

    @property
    def is_fitted(self) -> bool:
        """Whether the model was fitted to data."""
        return self._fit is not None

    @property
    def profile_(self) -> pd.DataFrame:
        """
        Profile likelihood on the grid of 201 kappa values in [0, 1]: columns kappa, loglik and the conditional
        least-squares estimates alpha and beta at each kappa (b = beta * rho_kappa).
        """
        return self._require_fit()["profile"].copy()

    def _profile_loglik(self, kappa) -> np.ndarray:
        stats = self._require_fit()["stats"]
        return stats.loglik(stats.solve(kappa)[2])

    def kappa_confint(self, level: float = 0.95) -> tuple[float, float]:
        """
        Profile-likelihood confidence interval for kappa: {kappa in [0, 1] : 2 [l(kappa_hat) - l(kappa)] <= chi2_1(level)},
        truncated to [0, 1]. The chi2_1 cut-off is exact asymptotically for interior kappa and conservative when the true
        kappa is 0 or 1 (where the likelihood-ratio statistic is a 1/2 chi2_0 + 1/2 chi2_1 mixture).
        """
        fit = self._require_fit()
        if not fit["kappa_estimated"]:
            raise ValueError("kappa was fixed when fitting, so it has no confidence interval.")
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1).")
        cutoff = chi2.ppf(level, 1)
        l_hat = float(self._profile_loglik(self.kappa)[0])
        excess = lambda k: 2 * (l_hat - float(self._profile_loglik(k)[0])) - cutoff
        grid = fit["profile"]["kappa"].to_numpy()

        def bound(direction: int) -> float:
            end = 0.0 if direction < 0 else 1.0
            if excess(end) <= 0:
                return end
            # Walk along the grid away from kappa_hat until the first point outside the interval, then solve for the crossing
            pts = np.r_[self.kappa, grid[grid < self.kappa][::-1]] if direction < 0 else np.r_[self.kappa, grid[grid > self.kappa]]
            prev = pts[0]
            for k in pts[1:]:
                if excess(k) > 0:
                    return float(brentq(excess, min(prev, k), max(prev, k), xtol=1e-10))
                prev = k
            return float(brentq(excess, min(prev, end), max(prev, end), xtol=1e-10))

        lower, upper = bound(-1), bound(+1)
        outside = (grid < lower - 1e-9) | (grid > upper + 1e-9)
        if np.any(2 * (l_hat - fit["profile"]["loglik"].to_numpy()[outside]) <= cutoff):
            warnings.warn("The profile-likelihood confidence region for kappa is not an interval; the interval around the "
                          "estimate is reported. Inspect plot_profile().", UserWarning)
        return lower, upper

    def _covariance(self) -> tuple[np.ndarray, list[str]]:
        """
        Asymptotic covariance of the estimates of (alpha, beta[, kappa]): sigma_2 (J^T J)^{-1}. kappa is included only if it
        was estimated and lies in the interior of [0, 1].
        """
        fit = self._require_fit()
        with_kappa = fit["kappa_estimated"] and 0 < self.kappa < 1
        gram = fit["stats"].gram(self.beta, self.kappa, with_kappa)
        params = ["alpha", "beta", "kappa"] if with_kappa else ["alpha", "beta"]
        return self.sigma_2 * np.linalg.inv(gram), params

    def _rho_derivative(self) -> float:
        """d rho_kappa / d kappa = -rho_kappa * sum_i log(d_i) v_i^2, with v the unit Perron vector of D^{-kappa/2} A D^{-kappa/2}."""
        return -self.rho * float(self._log_deg @ self._perron ** 2)

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        """
        Estimates, standard errors and confidence intervals of a fitted model, one row per parameter (alpha, b, beta, kappa).

        Standard errors come from the asymptotic covariance sigma_2 (J^T J)^{-1} of the least-squares estimates of
        (alpha, beta, kappa), with the delta method for b = beta * rho_kappa; intervals for alpha, b and beta are Wald
        intervals. The interval for kappa is the profile-likelihood interval (see kappa_confint). When kappa is fixed, or
        estimated on the boundary of [0, 1], its standard error is NaN and the other standard errors are conditional on
        kappa; near the boundary they understate the uncertainty.
        """
        fit = self._require_fit()
        cov, params = self._covariance()
        se = dict(zip(params, np.sqrt(np.diag(cov))))
        grad = np.array([0.0, self.rho] + ([self.beta * self._rho_derivative()] if "kappa" in params else []))
        se["b"] = float(np.sqrt(grad @ cov @ grad))
        z = norm.ppf(0.5 + level / 2)
        rows = {}
        for name in ["alpha", "b", "beta"]:
            est = getattr(self, name)
            rows[name] = [est, se[name], est - z * se[name], est + z * se[name]]
        if fit["kappa_estimated"]:
            rows["kappa"] = [self.kappa, se.get("kappa", np.nan), *self.kappa_confint(level)]
        else:
            rows["kappa"] = [self.kappa, np.nan, np.nan, np.nan]
        pct = f"{100 * level:g}%"
        return pd.DataFrame.from_dict(rows, orient="index",
                                      columns=["estimate", "std_error", f"ci_lower ({pct})", f"ci_upper ({pct})"])

    def _check_identification(self) -> None:
        lower, upper = self.kappa_confint(0.95)
        if lower == 0.0 and upper == 1.0:
            warnings.warn("kappa is weakly identified: its 95% profile-likelihood interval covers all of [0, 1].", UserWarning)
        b_row = self.summary().loc["b"]
        if abs(b_row["estimate"]) < 2 * b_row["std_error"]:
            warnings.warn("The network coefficient b is not clearly different from zero (|b| < 2 standard errors), so kappa "
                          "is poorly identified.", UserWarning)

    def _num_params(self) -> int:
        # alpha, b, sigma_2, plus kappa if it was estimated (the node means are not counted)
        return 3 + int(self._require_fit()["kappa_estimated"])

    def loglik(self) -> float:
        """Maximised Gaussian log-likelihood of the fitted model, conditional on the first observation."""
        fit = self._require_fit()
        return float(fit["stats"].loglik(fit["rss"]))

    def aic(self) -> float:
        """Akaike information criterion -2 loglik + 2k, with k = 4 (alpha, b, kappa, sigma_2), or 3 if kappa was fixed."""
        return -2 * self.loglik() + 2 * self._num_params()

    def bic(self) -> float:
        """Bayesian information criterion -2 loglik + k log((T - 1) N), with k as in aic()."""
        return -2 * self.loglik() + self._num_params() * np.log(self._require_fit()["stats"].n_obs)

    def plot_profile(self, level: float = 0.95, zoom: bool = True, n_points: int = 51):
        """
        Plot the profile likelihood of kappa (left) and the conditional estimates alpha(kappa) and b(kappa) along it (right).

        The left panel shows l(kappa) - l(kappa_hat), the estimate, the chi2_1(level) / 2 cut-off and the profile-likelihood
        confidence interval. With zoom=True it shows the range of kappa where the profile is within four times the cut-off
        of its maximum (the profile is recomputed on a fine grid there, which is cheap); with zoom=False it shows [0, 1].
        The right panel always covers [0, 1]; it needs rho_kappa at each point, so it uses n_points values of kappa.

        Returns:
            matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        fit = self._require_fit()
        blue, orange, muted = "#2a78d6", "#eb6834", "#52514e"
        prof = fit["profile"]
        l_hat = float(self._profile_loglik(self.kappa)[0])
        half_cut = chi2.ppf(level, 1) / 2
        fig, (ax_l, ax_p) = plt.subplots(1, 2, figsize=(11, 4))

        if zoom:
            y_low = -4 * half_cut
            grid, diff = prof["kappa"].to_numpy(), prof["loglik"].to_numpy() - l_hat
            inside = np.r_[grid[diff >= y_low], self.kappa]
            step = grid[1] - grid[0]
            lo, hi = max(inside.min() - step, 0.0), min(inside.max() + step, 1.0)
            if hi - lo < 0.02:                               # keep a minimal width so the curve stays visible
                centre = min(max((lo + hi) / 2, 0.01), 0.99)
                lo, hi = centre - 0.01, centre + 0.01
            ks_l = np.linspace(lo, hi, 401)
            ax_l.plot(ks_l, self._profile_loglik(ks_l) - l_hat, color=blue, lw=1.8)
            ax_l.set_xlim(lo, hi)
            ax_l.set_ylim(y_low * 1.05, 0.15 * half_cut)
        else:
            ks_l = np.linspace(0, 1, 401)
            diff = self._profile_loglik(ks_l) - l_hat
            ax_l.plot(ks_l, diff, color=blue, lw=1.8)
            ax_l.set_xlim(0, 1)
            ax_l.set_ylim(max(diff.min(), -20 * half_cut) * 1.05, 0.15 * half_cut)
        ax_l.axhline(-half_cut, color=muted, lw=1, ls="--")
        ax_l.text(0.01, -half_cut, f" {100 * level:g}% cut-off", color=muted, va="bottom",
                  transform=ax_l.get_yaxis_transform())
        if fit["kappa_estimated"]:
            lower, upper = self.kappa_confint(level)
            ax_l.axvspan(lower, upper, color=blue, alpha=0.08, lw=0)
            ax_l.axvline(self.kappa, color=blue, lw=1, ls=":")
            ax_l.set_title(fr"Profile likelihood: $\hat\kappa$ = {self.kappa:.3f}, CI [{lower:.3f}, {upper:.3f}]")
        else:
            ax_l.axvline(self.kappa, color=blue, lw=1, ls=":")
            ax_l.set_title(fr"Profile likelihood ($\kappa$ fixed at {self.kappa:g})")
        ax_l.set_xlabel(r"$\kappa$")
        ax_l.set_ylabel(r"$\ell_p(\kappa) - \ell_p(\hat\kappa)$")

        ks = np.linspace(0, 1, n_points)
        alpha_k, beta_k, _ = fit["stats"].solve(ks)
        rho_k = np.array([_spectral_info(_symmetrised_weight_matrix(self._A, k))[1] for k in ks])
        ax_p.plot(ks, alpha_k, color=blue, lw=1.8, label=r"$\hat\alpha(\kappa)$")
        ax_p.plot(ks, beta_k * rho_k, color=orange, lw=1.8, label=r"$\hat b(\kappa)$")
        ax_p.axvline(self.kappa, color=muted, lw=1, ls=":")
        ax_p.set_xlim(0, 1)
        ax_p.set_xlabel(r"$\kappa$")
        ax_p.set_title("Conditional estimates along the profile")
        ax_p.legend(frameon=False)
        for ax in (ax_l, ax_p):
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(True, color="#e4e3df", lw=0.6)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------ parameters and structure
    @property
    def beta(self) -> float:
        """Unnormalised network coefficient beta = b / rho_kappa."""
        return self.b / self.rho

    @property
    def degrees(self) -> pd.Series:
        """Node degrees, indexed by node name."""
        return pd.Series(self._degrees, index=self._names, name="degree")

    @property
    def names(self) -> pd.Index:
        """Node names."""
        return self._names

    def is_regular(self) -> bool:
        """True if all nodes have the same degree, in which case kappa is not identifiable from data."""
        return bool(np.all(self._degrees == self._degrees[0]))

    def phi(self) -> sp.csr_array:
        """
        Sparse VAR(1) coefficient matrix Phi = alpha I + (b / rho_kappa) D^{-kappa} A (a scipy.sparse CSR array; use
        .toarray() for a dense copy). Row i gives the coefficients of node i's equation: X_t - mu = Phi (X_{t-1} - mu) + e_t.
        """
        if self._phi is None:
            W = sp.diags_array(self._degrees ** -self.kappa) @ self._A
            self._phi = (self.alpha * sp.eye_array(self._N) + (self.b / self.rho) * W).tocsr()
        return self._phi

    def phi_spectral_radius(self) -> float:
        """
        Spectral radius of Phi. The eigenvalues of D^{-kappa} A are real and lie in [mu_min, rho_kappa], so the eigenvalues
        of Phi are alpha + b * mu / rho_kappa and the extremes are attained at mu = rho_kappa and mu = mu_min.
        """
        return max(abs(self.alpha + self.b), abs(self.alpha + self.b * self._mu_min / self.rho))

    def is_stationary(self) -> bool:
        """
        Exact stationarity check: the process is stationary if and only if

            |alpha + b| < 1   and   |alpha + b * mu_min / rho_kappa| < 1,

        where mu_min is the smallest eigenvalue of D^{-kappa} A. In particular |alpha| + |b| < 1 is sufficient, and it is
        also necessary when the graph has a bipartite component (mu_min = -rho_kappa).
        """
        return self.phi_spectral_radius() < 1

    # ------------------------------------------------------------------ VAR conversion
    def to_var(self) -> VAR:
        """
        Return the equivalent VAR(1) model. Note that VAR stores dense coefficients, so this uses O(N^2) memory.
        """
        # VAR stores coefficients as (source, target), i.e. the transpose of Phi. Stationarity is known exactly here, so
        # VAR's own (dense eigenvalue) check is skipped.
        coeffs = pd.DataFrame(self.phi().T.toarray(), columns=self._names)
        return VAR(p=1, coeffs=coeffs, mean=self.mu, sigma_2=self.sigma_2, check_stationarity=False)

    # ------------------------------------------------------------------ simulation and forecasting
    def simulate(self, n: int, sigma_2: float | None = None, burn_in: int = 50,
                 as_frame: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Simulate a realisation from the model. Each step costs one sparse matrix-vector product. With the same random
        seed the output equals that of to_var().simulate(...) (up to floating-point rounding).

        Parameters:
            n (int): The number of time steps to simulate.
            sigma_2 (float): The noise variance. Defaults to the model's sigma_2.
            burn_in (int): The number of burn-in steps to discard.
            as_frame (bool): If True, return a DataFrame with the node names as columns. Defaults to False.

        Returns:
            np.ndarray or pd.DataFrame: The simulated time series. Shape (n, N).
        """
        # The first row is the all-zero pre-sample used to start the recursion; it is always discarded
        total = 1 + burn_in + n
        ts_sim = generate_noise(self.sigma_2 if sigma_2 is None else sigma_2, total, self._N)
        ts_sim[0] = 0.0
        phi = self.phi()
        for t in range(1, total):
            # X_t = Phi X_{t-1} + e_t, with the noise already stored in row t
            ts_sim[t] += phi @ ts_sim[t - 1]
        sim = ts_sim[1 + burn_in:] + self.mu
        return pd.DataFrame(sim, columns=self._names) if as_frame else sim

    def predict(self, ts: np.ndarray | pd.DataFrame | None = None, h: int = 1) -> np.ndarray | pd.DataFrame:
        """
        Forecast 1 to h steps ahead from each row of ts, using sparse products (O(number of edges) per row and step).
        Same conventions as GNAR.predict with p = 1: a single row gives an output of shape (h, N); T > 1 rows give forecasts
        from every origin, of shape (T, N, h), or a DataFrame with MultiIndex columns (node, "h = k") if ts is a DataFrame.
        If ts is None, the forecasts start from the last observation used in fitting.
        """
        if ts is None:
            if self._ts is None:
                raise ValueError("No time series provided and the model was not fitted to data. Pass a time series to predict().")
            ts = self._ts.iloc[[-1]] if isinstance(self._ts, pd.DataFrame) else np.asarray(self._ts)[[-1]]
        if h < 1:
            raise ValueError("The forecast horizon h must be at least 1.")
        is_df = isinstance(ts, pd.DataFrame)
        values = ts.to_numpy(dtype=float) if is_df else np.asarray(ts, dtype=float)
        if values.ndim != 2 or values.shape[1] != self._N:
            raise ValueError(f"ts must have shape (T, {self._N}).")
        n = values.shape[0]
        if n < 1:
            raise ValueError("At least one observation is required for forecasting.")

        phi = self.phi()
        preds = np.empty((n, self._N, h))
        current = (values - self.mu).T                     # (N, T): one column per forecast origin
        for i in range(h):
            current = phi @ current
            preds[:, :, i] = current.T
        preds += self.mu.reshape(1, self._N, 1)

        if n == 1:
            if is_df:
                return pd.DataFrame(preds[0].T, index=range(1, h + 1), columns=ts.columns, dtype=float)
            return preds[0].T
        if is_df:
            columns = pd.MultiIndex.from_product([ts.columns, [f"h = {i}" for i in range(1, h + 1)]])
            return pd.DataFrame(preds.reshape(n, self._N * h), index=ts.index, columns=columns, dtype=float)
        return preds

    # ------------------------------------------------------------------ second moments
    def _spectral_cache(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Eigendecomposition S = U diag(s) U^T of S = D^{-kappa/2} A D^{-kappa/2} and the matrix B = U^T D^kappa U. Both
        depend only on the graph and kappa. Computed once (O(N^3)) and cached; see clear_cache().
        """
        if self._eig_cache is None:
            s, U = eigh(_symmetrised_weight_matrix(self._A, self.kappa).toarray(), driver="evd", overwrite_a=True)
            B = U.T @ (U * (self._degrees ** self.kappa)[:, None])
            self._eig_cache = (s, U, B)
        return self._eig_cache

    def clear_cache(self) -> None:
        """Free the cached eigendecomposition (about 2 N^2 floats) used by the second-moment methods."""
        self._eig_cache = None

    def _gamma0_core(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (U, C) such that Gamma(0) = sigma_2 D^{-kappa/2} U C U^T D^{-kappa/2}.

        With M = alpha I + (b / rho_kappa) S = U diag(lam) U^T, Phi = D^{-kappa/2} M D^{kappa/2}, so
        Phi^k Phi^kT = D^{-kappa/2} U diag(lam)^k B diag(lam)^k U^T D^{-kappa/2}. Summing over k gives C = B / (1 - lam lam^T).
        """
        if not self.is_stationary():
            raise ValueError("Second moments are only defined for a stationary model.")
        s, U, B = self._spectral_cache()
        lam = self.alpha + (self.b / self.rho) * s
        C = np.multiply.outer(lam, lam)
        np.subtract(1.0, C, out=C)
        np.divide(B, C, out=C)
        return U, C

    def compute_autocov_mats(self, max_lag: int = 1) -> np.ndarray:
        """
        Theoretical autocovariance matrices Gamma(0), ..., Gamma(max_lag), with Gamma(k) = Cov(X_t, X_{t-k}).
        Shape (max_lag + 1, N, N).

        Gamma(0) is computed in closed form from one symmetric eigendecomposition (cached per model, so repeated calls are
        much cheaper); Gamma(k) = Phi Gamma(k - 1) uses sparse products. Total cost O(N^3) time and O(N^2) memory.
        """
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        U, C = self._gamma0_core()
        gammas = np.empty((max_lag + 1, self._N, self._N))
        np.matmul(U @ C, U.T, out=gammas[0])
        scale = np.sqrt(self.sigma_2) * self._degrees ** (-self.kappa / 2)
        gammas[0] *= scale[:, None]
        gammas[0] *= scale[None, :]
        # Remove the rounding asymmetry
        gammas[0] += gammas[0].T.copy()
        gammas[0] *= 0.5
        phi = self.phi()
        for k in range(1, max_lag + 1):
            gammas[k] = phi @ gammas[k - 1]
        return gammas

    def compute_autocorr_mats(self, max_lag: int = 1) -> np.ndarray:
        """Theoretical autocorrelation matrices for lags 0, ..., max_lag. Shape (max_lag + 1, N, N)."""
        gammas = self.compute_autocov_mats(max_lag)
        inv_sd = 1.0 / np.sqrt(np.diag(gammas[0]).copy())
        gammas *= inv_sd[None, :, None]
        gammas *= inv_sd[None, None, :]
        return gammas

    def marginal_variances(self) -> pd.Series:
        """
        Theoretical marginal variances Var(X_i), i.e. the diagonal of Gamma(0), indexed by node name. Cheaper than
        compute_autocov_mats(0) because it avoids forming the full N x N matrix product.
        """
        U, C = self._gamma0_core()
        diag = np.einsum("ij,ij->i", U @ C, U)
        return pd.Series(self.sigma_2 * self._degrees ** -self.kappa * diag, index=self._names, name="variance")

    # ------------------------------------------------------------------ display
    def __repr__(self) -> str:
        return (f"KappaGNAR(N={self._N}, alpha={self.alpha:.4g}, b={self.b:.4g}, kappa={self.kappa:.4g}, "
                f"fitted={self.is_fitted})")

    def __str__(self) -> str:
        deg = self._degrees
        header = (f"Kappa-GNAR(1, [1]) Model (spectral normalisation)\n"
                  f"Graph with {self._N} nodes and {self._A.nnz // 2} edges; "
                  f"degrees from {deg.min():.0f} to {deg.max():.0f} (mean {deg.mean():.2f})\n")
        mu = self.mu.ravel()
        if np.all(mu == mu[0]):
            mean = f"Mean: {mu[0]:g} (all nodes)"
        elif self._N <= 10:
            mean = f"Mean:\n{pd.DataFrame(self.mu, index=['mean'], columns=self._names)}"
        else:
            mean = f"Mean: varies by node, from {mu.min():.4g} to {mu.max():.4g} (see .mu)"
        if self.is_fitted:
            fit = self._fit
            kappa_note = "estimated" if fit["kappa_estimated"] else "fixed"
            info = (f"Fitted to T = {fit['T']} time steps ({fit['stats'].n_obs} observations); kappa {kappa_note}\n"
                    f"{self.summary().to_string()}\n"
                    f"rho_kappa = {self.rho:.6g}, sigma_2 = {self.sigma_2:.6g}\n"
                    f"log-likelihood = {self.loglik():.6g}, AIC = {self.aic():.6g}, BIC = {self.bic():.6g}\n")
            return header + info + mean + "\n"
        params = pd.Series({"alpha": self.alpha, "b": self.b, "kappa": self.kappa,
                            "beta (= b / rho_kappa)": self.beta, "rho_kappa": self.rho, "sigma_2": self.sigma_2})
        return header + f"Parameters:\n{params.to_string()}\n{mean}\n"
