import numpy as np
from scipy.linalg import solve_discrete_lyapunov, cholesky
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from gnar.utils.data_utils import check_kappa, check_kappa_graph
from gnar.utils.neighbour_sets import node_degrees, degree_terms

# Above this number of nodes the spectral radius is computed with a sparse eigensolver
_DENSE_EIG_MAX = 2000

def shift_X(X: np.ndarray, sim: np.ndarray, ns: np.ndarray, p: int, s: np.ndarray) -> np.ndarray:
    """
    Shift the design matrix with the new simulated values and neighbour sums

    Params:
        X: np.array. Design matrix. Shape (d, p + sum(s))
        sim: np.array. Simulated values. Shape (d,)
        ns: np.array. Neighbour sums. Shape (p, d, max(s))
        p: int. Number of lags
        s: np.array. Maximum stage of neighbour dependence for each lag. Shape (p,)

    Returns:
        X: np.array. Design matrix. Shape (d, p + sum(s))
    """
    # Data shapes
    d, _ = np.shape(X)
    r = np.max(s)
    # Update the lagged time series observations
    X[:, 1 : p] = X[:, : p - 1]
    X[:, 0] = sim
    # Update the design matrix with the lagged neighbour sums
    for i in range(p):
        X[:, p + np.sum(s[:i]).astype(int) : p + np.sum(s[:i+1]).astype(int)] = ns[:, i * r : i * r + s[i]]
    return X

def generate_noise(sigma_2: float | int | np.ndarray, n: int, d: int) -> np.ndarray:
    """
    Generate noise for the GNAR model. The noise is assumed to be Gaussian with mean 0 and variance (or covariance) sigma_2.

    Params:
        sigma_2: float, int, np.array. The variance or covariance matrix.
        n: int. Number of observations.
        d: int. Number of nodes.

    Returns:
        np.array. Noise matrix. Shape (n, d)
    """
    if isinstance(sigma_2, (float, int)):
        return np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=(n, d))
    elif sigma_2.ndim == 1 or sigma_2.shape[0] == 1:
        return np.random.normal(loc=0, scale=np.sqrt(sigma_2), size=(n, d))
    return np.random.multivariate_normal(mean=np.zeros(d), cov=sigma_2, size=n)

def _gnar1_setup(A, alpha: float | np.ndarray, beta: float, kappa: float) -> tuple[csr_matrix, np.ndarray, float, float, np.ndarray]:
    """
    Validate the inputs of a kappa-normalised GNAR(1, [1]) model and broadcast alpha to one value per node.

    Returns:
        A (CSR), alpha (shape (d,)), beta, kappa, degrees N_i (shape (d,))
    """
    A = check_kappa_graph(A)
    kappa = check_kappa(kappa, allow_none=False)
    d = A.shape[0]
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (d,)).copy()
    beta = float(beta)
    return A, alpha, beta, kappa, node_degrees(A)

def gnar1_transition(A, alpha: float | np.ndarray, beta: float, kappa: float) -> csr_matrix:
    """
    Transition matrix of the kappa-normalised GNAR(1, [1]) model in vector form, X_t = Phi X_{t-1} + u_t, with

        Phi = diag(alpha) + beta diag(N_i^(-kappa)) A,

    so that Phi[i, q] = alpha_i 1{i = q} + beta N_i^(-kappa) A[i, q]. Isolated nodes (N_i = 0) have no network term.

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        alpha: float or np.array. Autoregressive coefficients alpha_i. Shape (d,) or scalar.
        beta: float. Network coefficient.
        kappa: float. Normalisation exponent.

    Returns:
        Phi: scipy.sparse.csr_matrix. Shape (d, d)
    """
    A, alpha, beta, kappa, degrees = _gnar1_setup(A, alpha, beta, kappa)
    _, _, w = degree_terms(degrees, kappa)
    return csr_matrix(diags(alpha) + beta * (diags(w) @ A))

def row_sum_bound(A, alpha: float | np.ndarray, beta: float, kappa: float) -> float:
    """
    Row-sum norm of the transition matrix,

        rho = ||Phi||_inf = max_i (|alpha_i| + |beta| N_i^(1 - kappa)),

    where the network term is 0 for isolated nodes. rho < 1 is sufficient (not necessary) for stationarity.

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        alpha: float or np.array. Autoregressive coefficients. Shape (d,) or scalar.
        beta: float. Network coefficient.
        kappa: float. Normalisation exponent.

    Returns:
        float. rho
    """
    A, alpha, beta, kappa, degrees = _gnar1_setup(A, alpha, beta, kappa)
    mask, log_N, _ = degree_terms(degrees, kappa)
    network = np.zeros_like(degrees)
    network[mask] = np.abs(beta) * np.exp((1 - kappa) * log_N[mask])
    return float(np.max(np.abs(alpha) + network))

def spectral_radius(A, alpha: float | np.ndarray, beta: float, kappa: float) -> float:
    """
    Spectral radius of the transition matrix Phi = diag(alpha) + beta diag(N_i^(-kappa)) A. The process is stationary if
    and only if it is below 1.

    For symmetric A, Phi is similar to the symmetric matrix diag(alpha) + beta D^(-kappa/2) A D^(-kappa/2), with
    D = diag(N_i), via D^(kappa/2) Phi D^(-kappa/2) (isolated nodes have zero rows and columns in A). Its eigenvalues are
    therefore real and are computed with a symmetric eigensolver.

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        alpha: float or np.array. Autoregressive coefficients. Shape (d,) or scalar.
        beta: float. Network coefficient.
        kappa: float. Normalisation exponent.

    Returns:
        float. Spectral radius of Phi.
    """
    A, alpha, beta, kappa, degrees = _gnar1_setup(A, alpha, beta, kappa)
    _, _, w_half = degree_terms(degrees, kappa / 2)
    Phi_sym = diags(alpha) + beta * (diags(w_half) @ A @ diags(w_half))
    d = A.shape[0]
    if d <= _DENSE_EIG_MAX:
        eigs = np.linalg.eigvalsh(Phi_sym.toarray())
        return float(np.max(np.abs(eigs)))
    # Largest-magnitude eigenvalue by Lanczos. The fixed, generic starting vector keeps the result deterministic; a
    # structured one such as a vector of ones is an eigenvector of regular graphs and would hide the rest of the spectrum
    v0 = np.random.default_rng(0).standard_normal(d)
    largest = eigsh(Phi_sym, k=1, which="LM", v0=v0, ncv=min(d - 1, 50), tol=1e-10, return_eigenvectors=False)
    return float(abs(largest[0]))

def stationary_cov(A, alpha: float | np.ndarray, beta: float, kappa: float, sigma_2: float = 1.0) -> np.ndarray:
    """
    Stationary covariance Gamma0 = Var(X_t) of the kappa-normalised GNAR(1, [1]) model, the solution of the discrete
    Lyapunov equation

        Gamma0 = Phi Gamma0 Phi^T + sigma^2 I_d.

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        alpha: float or np.array. Autoregressive coefficients. Shape (d,) or scalar.
        beta: float. Network coefficient.
        kappa: float. Normalisation exponent.
        sigma_2: float. Noise variance sigma^2.

    Returns:
        Gamma0: np.array. Shape (d, d)
    """
    _check_stationary(A, alpha, beta, kappa)
    Phi = gnar1_transition(A, alpha, beta, kappa).toarray()
    d = Phi.shape[0]
    Gamma0 = solve_discrete_lyapunov(Phi, float(sigma_2) * np.eye(d))
    return (Gamma0 + Gamma0.T) / 2

def _check_stationary(A, alpha: float | np.ndarray, beta: float, kappa: float) -> None:
    # Raise an error if the spectral radius of Phi is at least 1, reporting both the spectral radius and rho. Since the
    # spectral radius is at most rho = ||Phi||_inf, rho < 1 settles stationarity without an eigenvalue computation
    rho = row_sum_bound(A, alpha, beta, kappa)
    if rho < 1:
        return
    radius = spectral_radius(A, alpha, beta, kappa)
    if radius >= 1:
        raise ValueError(f"The GNAR model is not stationary: the spectral radius of Phi is {radius:.6g} >= 1 (row-sum bound rho = {rho:.6g}).")

def _random_state(rng):
    # None uses NumPy's global random state (so np.random.seed applies, as elsewhere in pygnar); a np.random.RandomState is
    # used as is; anything else (an int seed, a SeedSequence or a np.random.Generator) is passed to np.random.default_rng
    if rng is None:
        return np.random.mtrand._rand
    if isinstance(rng, np.random.RandomState):
        return rng
    return np.random.default_rng(rng)

def simulate_gnar1(
    A,
    alpha: float | np.ndarray,
    beta: float,
    kappa: float,
    n: int,
    sigma_2: float = 1.0,
    start: str = "stationary",
    burn_in: int = 0,
    rng=None,
) -> np.ndarray:
    """
    Simulate the kappa-normalised GNAR(1, [1]) model

        X_{i,t} = alpha_i X_{i,t-1} + beta N_i^(-kappa) S_{i,t-1} + u_{i,t},    S_{i,t-1} = sum_{q in N(i)} X_{q,t-1},

    i.e. X_t = Phi X_{t-1} + u_t with Phi = diag(alpha) + beta diag(N_i^(-kappa)) A and u_t ~ N(0, sigma^2 I_d) i.i.d.

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        alpha: float or np.array. Autoregressive coefficients alpha_i. Shape (d,) or scalar.
        beta: float. Network coefficient.
        kappa: float. Normalisation exponent.
        n: int. Number of time points returned (T), including the starting value.
        sigma_2: float. Noise variance sigma^2. Defaults to 1.
        start: str. "stationary" draws X_0 ~ N(0, Gamma0), where Gamma0 solves Gamma0 = Phi Gamma0 Phi^T + sigma^2 I_d
            (see stationary_cov), so the series is stationary from the first observation. "zero" starts from X = 0,
            which avoids the O(d^3) Lyapunov solve for large graphs; use it with a burn-in.
        burn_in: int. Number of initial time points generated and discarded before the n returned ones. Defaults to 0.
        rng: None, int, np.random.Generator or np.random.RandomState. Source of randomness. None uses NumPy's global
            random state, so np.random.seed makes the output reproducible; an int seed or a Generator gives an
            independent stream.

    Raises:
        ValueError: if the spectral radius of Phi is at least 1 (the message reports it and the row-sum bound rho).

    Returns:
        X: np.array. Simulated time series, time x nodes, with X[0] the starting value. Shape (n, d)
    """
    if n < 1 or burn_in < 0:
        raise ValueError("n must be at least 1 and burn_in must be non-negative.")
    if start not in {"stationary", "zero"}:
        raise ValueError("start must be 'stationary' or 'zero'.")
    if sigma_2 <= 0:
        raise ValueError("sigma_2 must be positive.")
    _check_stationary(A, alpha, beta, kappa)
    Phi = gnar1_transition(A, alpha, beta, kappa)
    d = Phi.shape[0]
    # Dense matrix-vector products are faster for small graphs; keep Phi sparse for large ones
    Phi_T = Phi.T.toarray() if d <= _DENSE_EIG_MAX else Phi.T.tocsr()
    gen = _random_state(rng)
    total = burn_in + n
    X = np.zeros((total, d))
    if start == "stationary":
        Gamma0 = stationary_cov(A, alpha, beta, kappa, sigma_2)
        L = cholesky(Gamma0, lower=True)
        X[0] = L @ gen.standard_normal(d)
    u = np.sqrt(sigma_2) * gen.standard_normal((total - 1, d))
    for t in range(1, total):
        # Row convention: X_t^T = X_{t-1}^T Phi^T + u_t^T
        X[t] = X[t - 1] @ Phi_T + u[t - 1]
    return X[burn_in:]

def stationary_params(A, kappa: float, b: float, alpha: float | np.ndarray) -> float:
    """
    Network coefficient that fixes the largest total network weight at b:

        |beta| = b / max_i N_i^(1 - kappa),   so that   max_i |beta| N_i^(1 - kappa) = b,

    with the maximum over nodes with N_i >= 1, and beta taking the sign of b. It checks that max_i |alpha_i| + |b| < 1, so
    the row-sum bound rho = ||Phi||_inf is below 1 and the process is stationary. Holding b fixed (rather than beta) keeps
    growing graphs stationary.

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        kappa: float. Normalisation exponent.
        b: float. Total network weight of the most heavily weighted node.
        alpha: float or np.array. Autoregressive coefficients. Shape (d,) or scalar.

    Returns:
        float. beta
    """
    A, alpha, _, kappa, degrees = _gnar1_setup(A, alpha, 0.0, kappa)
    mask, log_N, _ = degree_terms(degrees, kappa)
    if not np.any(mask):
        raise ValueError("The graph has no edges, so the network coefficient is undefined.")
    if np.max(np.abs(alpha)) + abs(b) >= 1:
        raise ValueError(f"max |alpha_i| + |b| = {np.max(np.abs(alpha)) + abs(b):.6g} must be below 1 for rho < 1.")
    return float(b / np.max(np.exp((1 - kappa) * log_N[mask])))
