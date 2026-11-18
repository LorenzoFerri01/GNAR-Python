from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from gnar.utils.data_utils import set_mean
from gnar.var import VAR


def check_kappa_adjacency(A: np.ndarray) -> None:
    """
    Check that A is a valid adjacency matrix for the kappa-GNAR model: a square, binary, symmetric NumPy array without
    self-loops or isolated nodes.
    """
    if not isinstance(A, np.ndarray) or A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix A must be a square NumPy array.")
    if not np.all((A == 0) | (A == 1)):
        raise ValueError("Adjacency matrix A must be binary (0 or 1): the kappa-GNAR model is defined for unweighted networks.")
    if not np.array_equal(A, A.T):
        raise ValueError("Adjacency matrix A must be symmetric: the kappa-GNAR model is defined for undirected networks.")
    if np.any(np.diag(A) != 0):
        raise ValueError("Adjacency matrix A must not contain self-loops.")
    isolated = np.flatnonzero(A.sum(axis=1) == 0)
    if isolated.size > 0:
        raise ValueError(f"The graph must not contain isolated nodes (found at indices {isolated.tolist()}).")


def kappa_weight_matrix(A: np.ndarray, kappa: float) -> np.ndarray:
    """
    Compute D^{-kappa} A, where D is the diagonal degree matrix. Row i holds the weights node i assigns to its
    neighbours: (D^{-kappa} A x)_i = d_i^{-kappa} * sum_{j in N(i)} x_j. Assumes A has no isolated nodes.
    """
    d = A.sum(axis=1)
    return (d ** -kappa)[:, None] * A


def _symmetrised_weight_matrix(A: np.ndarray, kappa: float) -> np.ndarray:
    """
    S = D^{-kappa/2} A D^{-kappa/2}, which is symmetric and similar to D^{-kappa} A (S = D^{kappa/2} (D^{-kappa} A) D^{-kappa/2}),
    so it has the same, real, eigenvalues.
    """
    s = A.sum(axis=1) ** (-kappa / 2)
    return s[:, None] * A * s[None, :]


def _extreme_eigenvalue(S: np.ndarray, largest: bool) -> float:
    i = S.shape[0] - 1 if largest else 0
    return float(eigh(S, eigvals_only=True, subset_by_index=[i, i])[0])


def kappa_spectral_radius(A: np.ndarray, kappa: float) -> float:
    """
    Spectral radius rho_kappa of D^{-kappa} A for a symmetric adjacency matrix without isolated nodes. Computed as the
    largest eigenvalue of the similar symmetric matrix D^{-kappa/2} A D^{-kappa/2}; since that matrix is non-negative, its
    largest eigenvalue is its spectral radius (Perron root).
    """
    return _extreme_eigenvalue(_symmetrised_weight_matrix(A, kappa), largest=True)


def kappa_spectrum_bounds(A: np.ndarray, kappa: float) -> tuple[float, float]:
    """
    Smallest and largest eigenvalues (mu_min, rho_kappa) of D^{-kappa} A for a symmetric adjacency matrix without isolated
    nodes. Only the two extreme eigenvalues of the similar symmetric matrix D^{-kappa/2} A D^{-kappa/2} are computed.
    """
    S = _symmetrised_weight_matrix(A, kappa)
    return _extreme_eigenvalue(S, largest=False), _extreme_eigenvalue(S, largest=True)


class KappaGNAR:
    """
    Global GNAR(1, [1]) model with a degree-normalisation exponent kappa, for unweighted undirected networks:

        X_{i,t} - mu_i = alpha (X_{i,t-1} - mu_i) + beta d_i^{-kappa} sum_{j in N(i)} (X_{j,t-1} - mu_j) + e_{i,t},
        e_t ~ N(0, sigma_2 I).

    The default parametrisation uses the spectrally normalised network coefficient

        b = beta * rho_kappa,    rho_kappa = spectral radius of D^{-kappa} A,

    so that, in VAR(1) form, X_t - mu = Phi (X_{t-1} - mu) + e_t with Phi = alpha I + (b / rho_kappa) D^{-kappa} A.
    kappa = 1 gives the standard global GNAR(1, [1]) model (rho_1 = 1, so b = beta); kappa = 0 uses raw neighbour sums.

    Parameters:
        A (np.ndarray): Binary, symmetric adjacency matrix without self-loops or isolated nodes. Shape (d, d).
        alpha (float): Autoregressive coefficient, shared by all nodes.
        b (float): Spectrally normalised network coefficient. Provide exactly one of b and beta.
        kappa (float): Degree-normalisation exponent, in [0, 1]. Defaults to 1.
        beta (float): Unnormalised network coefficient, beta = b / rho_kappa. Alternative to b.
        mean (float, int, np.ndarray or pd.DataFrame): The mean of the process. Defaults to 0.
        sigma_2 (float): Noise variance, common to all nodes. Defaults to 1.
        names (list-like): Node names, in the order of the rows of A. Defaults to 0, ..., d - 1 (matching the node labels of
            networkx.from_numpy_array). Used in printing, in to_var(), and in DataFrame outputs.

    Methods:
        phi: VAR(1) coefficient matrix.
        to_var: Convert to an equivalent VAR(1) model.
        is_stationary: Exact stationarity check.
        is_regular: Whether all nodes have the same degree (kappa is then not identifiable).
        simulate: Simulate a realisation from the model.
        predict: Forecast future values.
        compute_autocov_mats / compute_autocorr_mats: Theoretical autocovariance / autocorrelation matrices.
    """

    def __init__(
        self,
        A: np.ndarray,
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

        self._A = A.astype(float)
        self._d = A.shape[0]
        self._degrees = self._A.sum(axis=1)
        self._names = pd.RangeIndex(self._d) if names is None else pd.Index(names)
        if len(self._names) != self._d:
            raise ValueError(f"The number of names ({len(self._names)}) does not match the number of nodes ({self._d}).")
        if not self._names.is_unique:
            raise ValueError("Node names must be unique.")

        self.alpha = float(alpha)
        self.kappa = float(kappa)
        self._mu_min, self.rho = kappa_spectrum_bounds(self._A, self.kappa)
        self.b = float(b) if b is not None else float(beta) * self.rho
        self.mu = set_mean(mean, self._d)
        self.sigma_2 = float(sigma_2)

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

    def phi(self) -> np.ndarray:
        """
        VAR(1) coefficient matrix Phi = alpha I + (b / rho_kappa) D^{-kappa} A. Row i gives the coefficients of node i's
        equation, so that X_t - mu = Phi (X_{t-1} - mu) + e_t.
        """
        return self.alpha * np.eye(self._d) + (self.b / self.rho) * kappa_weight_matrix(self._A, self.kappa)

    def is_stationary(self) -> bool:
        """
        Exact stationarity check. The eigenvalues of D^{-kappa} A are real and lie in [mu_min, rho_kappa], so the eigenvalues
        of Phi are alpha + b * mu / rho_kappa, and the process is stationary if and only if

            |alpha + b| < 1   and   |alpha + b * mu_min / rho_kappa| < 1.

        In particular |alpha| + |b| < 1 is sufficient, and it is also necessary when the graph has a bipartite component
        (mu_min = -rho_kappa).
        """
        return bool(abs(self.alpha + self.b) < 1 and abs(self.alpha + self.b * self._mu_min / self.rho) < 1)

    # ------------------------------------------------------------------ VAR-based functionality
    def to_var(self) -> VAR:
        """Return the equivalent VAR(1) model."""
        with warnings.catch_warnings():
            # Stationarity is already checked (and warned about) by this class
            warnings.simplefilter("ignore", UserWarning)
            # VAR stores coefficients as (source, target), i.e. the transpose of Phi
            return VAR(p=1, coeffs=pd.DataFrame(self.phi().T, columns=self._names), mean=self.mu, sigma_2=self.sigma_2)

    def simulate(self, n: int, sigma_2: float | None = None, burn_in: int = 50,
                 as_frame: bool = False) -> np.ndarray | pd.DataFrame:
        """
        Simulate a realisation from the model.

        Parameters:
            n (int): The number of time steps to simulate.
            sigma_2 (float): The noise variance. Defaults to the model's sigma_2.
            burn_in (int): The number of burn-in steps to discard.
            as_frame (bool): If True, return a DataFrame with the node names as columns. Defaults to False.

        Returns:
            np.ndarray or pd.DataFrame: The simulated time series. Shape (n, d).
        """
        sim = self.to_var().simulate(n, sigma_2=self.sigma_2 if sigma_2 is None else sigma_2, burn_in=burn_in)
        return pd.DataFrame(sim, columns=self._names) if as_frame else sim

    def predict(self, ts: np.ndarray | pd.DataFrame, h: int = 1) -> np.ndarray | pd.DataFrame:
        """
        Forecast 1 to h steps ahead, with the same conventions as GNAR.predict for p = 1: a single row gives an output of
        shape (h, d); n > 1 rows give forecasts from every origin, of shape (n, d, h), or a DataFrame with MultiIndex columns
        (node, horizon) if ts is a DataFrame.
        """
        return self.to_var().predict(ts, h=h)

    def compute_autocov_mats(self, max_lag: int = 1) -> np.ndarray:
        """Theoretical autocovariance matrices Gamma(0), ..., Gamma(max_lag), Gamma(k) = Cov(X_t, X_{t-k}). Shape (max_lag + 1, d, d)."""
        return self.to_var().compute_autocov_mats(max_lag)

    def compute_autocorr_mats(self, max_lag: int = 1) -> np.ndarray:
        """Theoretical autocorrelation matrices for lags 0, ..., max_lag. Shape (max_lag + 1, d, d)."""
        return self.to_var().compute_autocorr_mats(max_lag)

    # ------------------------------------------------------------------ display
    def __repr__(self) -> str:
        return f"KappaGNAR(d={self._d}, alpha={self.alpha:.4g}, b={self.b:.4g}, kappa={self.kappa:.4g})"

    def __str__(self) -> str:
        params = pd.Series({"alpha": self.alpha, "b": self.b, "kappa": self.kappa,
                            "beta (= b / rho_kappa)": self.beta, "rho_kappa": self.rho, "sigma_2": self.sigma_2})
        mean = pd.DataFrame(self.mu, index=["mean"], columns=self._names)
        return (f"Kappa-GNAR(1, [1]) Model (spectral normalisation)\n"
                f"Graph with {self._d} nodes and {int(self._A.sum() / 2)} edges\n"
                f"Parameters:\n{params.to_string()}\n{mean}\n")
