from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from gnar.utils.data_utils import set_mean
from gnar.utils.simulating import generate_noise
from gnar.var import VAR

# Graphs with at most this many nodes use dense symmetric eigensolvers for the extreme eigenvalues; larger graphs use
# sparse Lanczos iterations (scipy.sparse.linalg.eigsh), which cost O(number of edges) per iteration.
_DENSE_EIG_MAX_NODES = 200


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


def _extreme_eigenvalues(S: sp.csr_array) -> tuple[float, float]:
    """Smallest and largest eigenvalues of the sparse symmetric matrix S."""
    d = S.shape[0]
    if d <= _DENSE_EIG_MAX_NODES:
        ev = eigh(S.toarray(), eigvals_only=True)
        return float(ev[0]), float(ev[-1])
    # Deterministic starting vectors: a positive vector is close to the Perron vector for the largest eigenvalue
    largest = eigsh(S, k=1, which="LA", v0=np.ones(d), return_eigenvectors=False)[0]
    smallest = eigsh(S, k=1, which="SA", v0=np.random.default_rng(0).standard_normal(d), return_eigenvectors=False)[0]
    return float(smallest), float(largest)


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
    return _extreme_eigenvalues(_symmetrised_weight_matrix(_as_csr(A), kappa))


class KappaGNAR:
    """
    Global GNAR(1, [1]) model with a degree-normalisation exponent kappa, for unweighted undirected networks:

        X_{i,t} - mu_i = alpha (X_{i,t-1} - mu_i) + beta d_i^{-kappa} sum_{j in N(i)} (X_{j,t-1} - mu_j) + e_{i,t},
        e_t ~ N(0, sigma_2 I).

    The default parametrisation uses the spectrally normalised network coefficient

        b = beta * rho_kappa,    rho_kappa = spectral radius of D^{-kappa} A,

    so that, in VAR(1) form, X_t - mu = Phi (X_{t-1} - mu) + e_t with Phi = alpha I + (b / rho_kappa) D^{-kappa} A.
    kappa = 1 gives the standard global GNAR(1, [1]) model (rho_1 = 1, so b = beta); kappa = 0 uses raw neighbour sums.

    The graph is stored as a sparse matrix, so construction, stationarity checks, simulation and forecasting all cost
    O(number of edges) per step and scale to large graphs. Autocovariances are dense (d x d) and cost O(d^3).

    Parameters:
        A (np.ndarray or scipy sparse matrix): Binary, symmetric adjacency matrix without self-loops or isolated nodes.
            Shape (d, d).
        alpha (float): Autoregressive coefficient, shared by all nodes.
        b (float): Spectrally normalised network coefficient. Provide exactly one of b and beta.
        kappa (float): Degree-normalisation exponent, in [0, 1]. Defaults to 1.
        beta (float): Unnormalised network coefficient, beta = b / rho_kappa. Alternative to b.
        mean (float, int, np.ndarray or pd.DataFrame): The mean of the process. Defaults to 0.
        sigma_2 (float): Noise variance, common to all nodes. Defaults to 1.
        names (list-like): Node names, in the order of the rows of A. Defaults to 0, ..., d - 1 (matching the node labels of
            networkx.from_numpy_array). Used in printing, in to_var(), and in DataFrame outputs.

    Methods:
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
        alpha: float,
        b: float | None = None,
        kappa: float = 1.0,
        *,
        beta: float | None = None,
        mean: float | int | np.ndarray | pd.DataFrame = 0,
        sigma_2: float = 1.0,
        names=None,
    ) -> None:
        check_kappa_adjacency(A)
        if not np.isscalar(kappa) or not 0 <= kappa <= 1:
            raise ValueError("kappa must be a scalar in [0, 1].")
        if (b is None) == (beta is None):
            raise ValueError("Provide exactly one of b (spectrally normalised) and beta (unnormalised).")
        if np.ndim(sigma_2) != 0 or not sigma_2 > 0:
            raise ValueError("sigma_2 must be a positive scalar: the noise variance is common to all nodes.")

        self._A = _as_csr(A)
        self._d = self._A.shape[0]
        self._degrees = _degrees(self._A)
        self._names = pd.RangeIndex(self._d) if names is None else pd.Index(names)
        if len(self._names) != self._d:
            raise ValueError(f"The number of names ({len(self._names)}) does not match the number of nodes ({self._d}).")
        if not self._names.is_unique:
            raise ValueError("Node names must be unique.")

        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self._mu_min, self.rho = _extreme_eigenvalues(_symmetrised_weight_matrix(self._A, self.kappa))
        self.b = float(b) if b is not None else float(beta) * self.rho
        self.mu = set_mean(mean, self._d)
        self.sigma_2 = float(sigma_2)
        self._phi = None
        self._eig_cache = None

        if not self.is_stationary():
            warnings.warn("The kappa-GNAR model is non-stationary based on the provided parameters!", UserWarning)

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
            self._phi = (self.alpha * sp.eye_array(self._d) + (self.b / self.rho) * W).tocsr()
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
        Return the equivalent VAR(1) model. Note that VAR stores dense coefficients, so this uses O(d^2) memory.
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
            np.ndarray or pd.DataFrame: The simulated time series. Shape (n, d).
        """
        # The first row is the all-zero pre-sample used to start the recursion; it is always discarded
        total = 1 + burn_in + n
        ts_sim = generate_noise(self.sigma_2 if sigma_2 is None else sigma_2, total, self._d)
        ts_sim[0] = 0.0
        phi = self.phi()
        for t in range(1, total):
            # X_t = Phi X_{t-1} + e_t, with the noise already stored in row t
            ts_sim[t] += phi @ ts_sim[t - 1]
        sim = ts_sim[1 + burn_in:] + self.mu
        return pd.DataFrame(sim, columns=self._names) if as_frame else sim

    def predict(self, ts: np.ndarray | pd.DataFrame, h: int = 1) -> np.ndarray | pd.DataFrame:
        """
        Forecast 1 to h steps ahead from each row of ts, using sparse products (O(number of edges) per row and step).
        Same conventions as GNAR.predict with p = 1: a single row gives an output of shape (h, d); n > 1 rows give forecasts
        from every origin, of shape (n, d, h), or a DataFrame with MultiIndex columns (node, "h = k") if ts is a DataFrame.
        """
        if h < 1:
            raise ValueError("The forecast horizon h must be at least 1.")
        is_df = isinstance(ts, pd.DataFrame)
        values = ts.to_numpy(dtype=float) if is_df else np.asarray(ts, dtype=float)
        if values.ndim != 2 or values.shape[1] != self._d:
            raise ValueError(f"ts must have shape (n, {self._d}).")
        n = values.shape[0]
        if n < 1:
            raise ValueError("At least one observation is required for forecasting.")

        phi = self.phi()
        preds = np.empty((n, self._d, h))
        current = (values - self.mu).T                     # (d, n): one column per forecast origin
        for i in range(h):
            current = phi @ current
            preds[:, :, i] = current.T
        preds += self.mu.reshape(1, self._d, 1)

        if n == 1:
            if is_df:
                return pd.DataFrame(preds[0].T, index=range(1, h + 1), columns=ts.columns, dtype=float)
            return preds[0].T
        if is_df:
            columns = pd.MultiIndex.from_product([ts.columns, [f"h = {i}" for i in range(1, h + 1)]])
            return pd.DataFrame(preds.reshape(n, self._d * h), index=ts.index, columns=columns, dtype=float)
        return preds

    # ------------------------------------------------------------------ second moments
    def _spectral_cache(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Eigendecomposition S = U diag(s) U^T of S = D^{-kappa/2} A D^{-kappa/2} and the matrix B = U^T D^kappa U. Both
        depend only on the graph and kappa. Computed once (O(d^3)) and cached; see clear_cache().
        """
        if self._eig_cache is None:
            s, U = eigh(_symmetrised_weight_matrix(self._A, self.kappa).toarray(), driver="evd", overwrite_a=True)
            B = U.T @ (U * (self._degrees ** self.kappa)[:, None])
            self._eig_cache = (s, U, B)
        return self._eig_cache

    def clear_cache(self) -> None:
        """Free the cached eigendecomposition (about 2 d^2 floats) used by the second-moment methods."""
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
        Shape (max_lag + 1, d, d).

        Gamma(0) is computed in closed form from one symmetric eigendecomposition (cached per model, so repeated calls are
        much cheaper); Gamma(k) = Phi Gamma(k - 1) uses sparse products. Total cost O(d^3) time and O(d^2) memory.
        """
        if max_lag < 0:
            raise ValueError("max_lag must be non-negative.")
        U, C = self._gamma0_core()
        gammas = np.empty((max_lag + 1, self._d, self._d))
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
        """Theoretical autocorrelation matrices for lags 0, ..., max_lag. Shape (max_lag + 1, d, d)."""
        gammas = self.compute_autocov_mats(max_lag)
        inv_sd = 1.0 / np.sqrt(np.diag(gammas[0]).copy())
        gammas *= inv_sd[None, :, None]
        gammas *= inv_sd[None, None, :]
        return gammas

    def marginal_variances(self) -> pd.Series:
        """
        Theoretical marginal variances Var(X_i), i.e. the diagonal of Gamma(0), indexed by node name. Cheaper than
        compute_autocov_mats(0) because it avoids forming the full d x d matrix product.
        """
        U, C = self._gamma0_core()
        diag = np.einsum("ij,ij->i", U @ C, U)
        return pd.Series(self.sigma_2 * self._degrees ** -self.kappa * diag, index=self._names, name="variance")

    # ------------------------------------------------------------------ display
    def __repr__(self) -> str:
        return f"KappaGNAR(d={self._d}, alpha={self.alpha:.4g}, b={self.b:.4g}, kappa={self.kappa:.4g})"

    def __str__(self) -> str:
        params = pd.Series({"alpha": self.alpha, "b": self.b, "kappa": self.kappa,
                            "beta (= b / rho_kappa)": self.beta, "rho_kappa": self.rho, "sigma_2": self.sigma_2})
        mu = self.mu.ravel()
        if np.all(mu == mu[0]):
            mean = f"Mean: {mu[0]:g} (all nodes)"
        elif self._d <= 10:
            mean = f"Mean:\n{pd.DataFrame(self.mu, index=['mean'], columns=self._names)}"
        else:
            mean = f"Mean: varies by node, from {mu.min():.4g} to {mu.max():.4g} (see .mu)"
        deg = self._degrees
        return (f"Kappa-GNAR(1, [1]) Model (spectral normalisation)\n"
                f"Graph with {self._d} nodes and {self._A.nnz // 2} edges; "
                f"degrees from {deg.min():.0f} to {deg.max():.0f} (mean {deg.mean():.2f})\n"
                f"Parameters:\n{params.to_string()}\n{mean}\n")
