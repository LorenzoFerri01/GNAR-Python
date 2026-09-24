import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

def check_kappa(kappa: float | int | None, allow_none: bool = True) -> float | None:
    """
    Check the normalisation exponent kappa. None means kappa is estimated; otherwise it must be a finite real number.

    Returns:
        kappa as a float, or None.
    """
    if kappa is None:
        if not allow_none:
            raise ValueError("kappa must be a number here; kappa=None (estimated) is not allowed.")
        return None
    if isinstance(kappa, bool) or not isinstance(kappa, (int, float, np.integer, np.floating)):
        raise ValueError("kappa must be a real number or None.")
    kappa = float(kappa)
    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite.")
    return kappa

def check_kappa_graph(A) -> csr_matrix:
    """
    Check that A is a valid graph for the kappa-normalised GNAR model: a square, binary, symmetric adjacency matrix
    with no self-loops (an unweighted, undirected graph). Accepts NumPy arrays and SciPy sparse matrices.

    Raises:
        NotImplementedError: if A has weights other than 0 and 1 (weighted networks are not supported for kappa != 1).
        ValueError: if A is not square, has negative entries, is not symmetric or has self-loops.

    Returns:
        A as a SciPy CSR matrix of floats.
    """
    if issparse(A):
        A = csr_matrix(A, dtype=float)
    elif isinstance(A, np.ndarray):
        if A.ndim != 2:
            raise ValueError("Adjacency matrix A must be a square matrix.")
        A = csr_matrix(A.astype(float))
    else:
        raise ValueError("Adjacency matrix A must be a NumPy array or a SciPy sparse matrix.")
    if A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix A must be a square matrix.")
    A.eliminate_zeros()
    if np.any(A.data < 0):
        raise ValueError("Adjacency matrix A must have non-negative weights.")
    if np.any(A.data != 1):
        raise NotImplementedError("Weighted networks are not supported for kappa != 1; A must be binary (0 or 1).")
    if np.any(A.diagonal() != 0):
        raise ValueError("Adjacency matrix A must not contain self-loops for kappa != 1.")
    if (A != A.T).nnz != 0:
        raise ValueError("Adjacency matrix A must be symmetric (an undirected graph) for kappa != 1.")
    return A

def gnar_checks(A: np.ndarray, p: int, s: np.ndarray, model_type: str, net_type: str = "unweighted", kappa: float | None = 1.0) -> None:
    if not isinstance(A, np.ndarray) or A.shape[0] != A.shape[1]:
        raise ValueError("Adjacency matrix A must be a square NumPy array.")
    if np.any(A < 0):
        raise ValueError("Adjacency matrix A must have non-negative weights.")
    valid_net_types = {"unweighted", "weighted", "distance"}
    if net_type not in valid_net_types:
        raise ValueError(f"Invalid net_type. Expected one of {valid_net_types}, got '{net_type}'.")
    if net_type == "unweighted" and not np.all((A == 0) | (A == 1)):
        raise ValueError("Adjacency matrix A must be binary (0 or 1) for unweighted networks.")
    if p < 1:
        raise ValueError("The number of lags p must be at least 1.")
    if not isinstance(s, np.ndarray) or len(s) != p:
        raise ValueError("Array s must be a NumPy array with length equal to p (number of lags).")
    if np.any(s < 0):
        raise ValueError("Values in s must be non-negative integers.")
    valid_model_types = {"global", "standard", "local"}
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type. Expected one of {valid_model_types}, got '{model_type}'.")
    kappa = check_kappa(kappa)
    if kappa is None or kappa != 1.0:
        # The kappa-normalised model is only defined for GNAR(1, [1]) on unweighted, undirected graphs
        if p != 1 or len(s) != 1 or s[0] != 1:
            raise NotImplementedError("kappa != 1 is only implemented for GNAR(1, [1]) models (p = 1, s = [1]).")
        if net_type != "unweighted":
            raise NotImplementedError("kappa != 1 is only implemented for unweighted networks.")
        if model_type == "local":
            raise NotImplementedError("kappa != 1 is not supported for local models: node-specific betas absorb N_i^(-kappa), so kappa is not identified.")
        if kappa is None and model_type == "global":
            raise NotImplementedError("Estimating kappa is only implemented for standard models (node-specific alpha).")
        check_kappa_graph(A)
    return None

def set_mean(mean: float | int | np.ndarray | pd.DataFrame, d: int) -> np.ndarray:
    if isinstance(mean, (float, int)):
        return np.full((1, d), mean)
    elif isinstance(mean, np.ndarray):
        return mean.reshape(1, d)
    elif isinstance(mean, pd.DataFrame):
        return mean.to_numpy().reshape(1, d)
    else:
        raise ValueError("Mean must be a float, int, NumPy array or Pandas DataFrame.")

def set_cov(sigma_2: float | int | np.ndarray | pd.DataFrame) -> float | int | np.ndarray:
    if isinstance(sigma_2, (float, int, np.ndarray)):
        return sigma_2
    elif isinstance(sigma_2, pd.DataFrame):
        return sigma_2.to_numpy()
    else:
        raise ValueError("Noise covariance matrix must be a float, int, NumPy array or Pandas DataFrame.")

def cov_mat(sigma_2: float | int | np.ndarray, d: int) -> np.ndarray:
    if isinstance(sigma_2, (int, float)):
        return sigma_2 * np.eye(d)
    elif isinstance(sigma_2, np.ndarray) and (sigma_2.ndim == 1 or sigma_2.shape[0] == 1):
        return np.diag(sigma_2.flatten())
    return sigma_2

def check_gnar_coeffs(coeffs: np.ndarray, d: int, p: int, s: np.ndarray, model_type: str) -> None:
    k, q = np.shape(coeffs)
    if d != q:
        raise ValueError("The number of nodes in the adjacency matrix does not match the number of nodes in the coefficients matrix.")
    if k != p + np.sum(s):
        raise ValueError("The number of coefficients does not match the number of parameters.")
    unique_params = np.apply_along_axis(lambda row: len(np.unique(row)), axis=1, arr=coeffs)
    if model_type == "global" and np.any(unique_params != 1):
        raise ValueError("The model type is global, but the coefficients are not the same for all nodes.")
    elif model_type == "standard" and np.any(unique_params[p:] != 1):
        raise ValueError("The model type is standard, but the beta coefficients are not the same for all nodes.")
    return None

def param_reorder(coeffs: np.ndarray, p: int, s: np.ndarray) -> np.ndarray:
    """
    Reorder the coefficients to match the structure of the weight matrices.
    """
    order = []
    s_tau = 0
    for i in range(p):
        order.append(i)
        order += list(range(p + s_tau, p + s_tau + s[i]))
        s_tau += s[i]
    return coeffs[order]
