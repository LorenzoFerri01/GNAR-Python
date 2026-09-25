from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

from gnar.utils.data_utils import check_kappa, check_kappa_graph
from gnar.utils.neighbour_sets import node_degrees
from gnar.utils.gnar_profile_likelihood import (node_sums, residualised_sums, fixed_kappa_ols, estimate_kappa,
                                                profile_loglik, deviance, has_network_information, informative_nodes)
from gnar.utils.simulating import _check_sigma_2
from gnar.utils.gnar_inference import beta_kappa_cov


class KappaWarning(UserWarning):
    """Base class of the warnings raised when fitting the kappa-normalised GNAR model."""


class KappaBoundaryWarning(KappaWarning):
    """The estimate of kappa lies within 1e-3 of a bound of kappa_bounds."""


class KappaMultimodalWarning(KappaWarning):
    """The profile of RSS(kappa) has several local minima on the grid."""


class KappaFlatProfileWarning(KappaWarning):
    """The profile deviance stays below the chi-square(1) cutoff at both bounds (weak identification)."""


class KappaIdentifiabilityWarning(KappaWarning):
    """kappa (or beta) is not identified by the graph and the data."""


_WARNING_CLASSES = {
    "boundary": KappaBoundaryWarning,
    "multimodal": KappaMultimodalWarning,
    "flat": KappaFlatProfileWarning,
    "identifiability": KappaIdentifiabilityWarning,
}

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _warn_outside_package(message: str, category: type) -> None:
    # Attribute the warning to the first caller outside the gnar package, whether the user called fit_gnar1 directly or
    # through GNAR, so that the location (and the default once-per-location filter) refers to the user's code
    frame, level = sys._getframe(1), 1
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(_PACKAGE_DIR):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, category, stacklevel=level + 1)


def check_kappa_bounds(kappa_bounds) -> tuple[float, float]:
    """
    Check the bounds of the search for kappa: two finite real numbers with lower < upper.

    Returns:
        (lower, upper) as floats.
    """
    try:
        lo, hi = kappa_bounds
    except (TypeError, ValueError):
        raise ValueError("kappa_bounds must be a pair (lower, upper).") from None
    for b in (lo, hi):
        if isinstance(b, (bool, np.bool_)) or not isinstance(b, (int, float, np.integer, np.floating)):
            raise ValueError("kappa_bounds must contain two real numbers.")
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError("kappa_bounds must be two finite numbers with lower < upper.")
    return lo, hi


class GNARKappaFit:
    """
    Fit of the kappa-normalised GNAR(1, [1]) standard model

        X_{i,t} = alpha_i X_{i,t-1} + beta N_i^(-kappa) S_{i,t-1} + u_{i,t},    S_{i,t-1} = sum_{q in N(i)} X_{q,t-1},

    returned by fit_gnar1.

    Attributes:
        alpha (np.ndarray): Estimates of alpha_i. Shape (d,)
        beta (float): Estimate of beta (NaN if not identified).
        kappa (float): Estimate of kappa, or its fixed value (NaN if estimated but not identified).
        sigma_2 (float): Noise variance: RSS / (n_obs - p) with p = d + 2 (kappa estimated) or d + 1 (kappa fixed), or
            the fixed value passed to fit_gnar1. A parameter that is not identified (kappa on a regular graph, beta
            without network information) is not counted in p.
        rss (float): Residual sum of squares RSS(kappa_hat).
        loglik (float): Profile log-likelihood at kappa_hat (see gnar.utils.gnar_profile_likelihood.profile_loglik).
        n_obs (int): Number of observations d (T - 1).
        n (int): Number of time points T.
        d (int): Number of nodes.
        degrees (np.ndarray): Degrees N_i. Shape (d,)
        mu (np.ndarray): Mean removed from the data before fitting (zeros if demean=False). Shape (1, d)
        names: Node names (DataFrame columns, or 1, ..., d).
        kappa_estimated (bool): Whether kappa was estimated.
        kappa_bounds (tuple): Bounds of the search for kappa.
        sigma_2_fixed (bool): Whether sigma_2 was fixed rather than estimated.
        profile (dict or None): For estimated kappa, the grid ("kappa"), RSS ("rss"), profile log-likelihood ("loglik")
            and deviance ("deviance") on the grid; None when kappa is fixed.
        warnings (list): (code, message) pairs, with code one of "boundary", "multimodal", "flat", "identifiability".
    """

    def __init__(self, **attributes) -> None:
        for key, value in attributes.items():
            setattr(self, key, value)

    @property
    def num_params(self) -> int:
        # Number of identified mean parameters (the p in sigma_2 = RSS / (n_obs - p)): d alphas, beta if the data carry
        # information on it, and kappa if it was estimated and is identified
        return self._num_params

    def _se_beta_kappa(self) -> tuple[float, float]:
        # Standard errors of beta_hat and kappa_hat from the concentrated (beta, kappa) information
        return float(np.sqrt(self._cov_beta_kappa[0, 0])), float(np.sqrt(self._cov_beta_kappa[1, 1]))

    def __repr__(self) -> str:
        status = "estimated" if self.kappa_estimated else "fixed"
        return f"GNARKappaFit(d={self.d}, n={self.n}, kappa={self.kappa:.6g} ({status}), beta={self.beta:.6g})"

    def __str__(self) -> str:
        se_beta, se_kappa = self._se_beta_kappa()
        status = "estimated" if self.kappa_estimated else "fixed"
        lines = [f"Kappa-normalised GNAR(1, [1]) fit: d = {self.d} nodes, T = {self.n} time points, n = {self.n_obs} observations",
                 f"kappa  = {self.kappa:.6g} ({status}" + (f", SE {se_kappa:.4g})" if self.kappa_estimated else ")"),
                 f"beta   = {self.beta:.6g} (SE {se_beta:.4g})",
                 f"sigma^2 = {self.sigma_2:.6g} ({'fixed' if self.sigma_2_fixed else 'estimated'}), RSS = {self.rss:.6g}, loglik = {self.loglik:.6g}"]
        alpha = pd.Series(self.alpha, index=self.names, name="alpha")
        lines.append(f"alpha:\n{alpha.to_string()}")
        for code, message in self.warnings:
            lines.append(f"Warning ({code}): {message}")
        return "\n".join(lines) + "\n"


def _check_ts(ts, d: int) -> tuple[np.ndarray, object]:
    # Validate the time series (time x nodes) and return it as a float array with the node names
    if isinstance(ts, pd.DataFrame):
        names = ts.columns
        ts = ts.to_numpy(dtype=float)
    else:
        ts = np.asarray(ts, dtype=float)
        names = np.arange(1, ts.shape[1] + 1) if ts.ndim == 2 else None
    if ts.ndim != 2:
        raise ValueError("The time series must be a 2D array of shape (T, d), time x nodes.")
    if ts.shape[1] != d:
        raise ValueError("The number of time series does not match the number of nodes in the adjacency matrix.")
    if ts.shape[0] < 3:
        raise ValueError("At least 3 time points are required.")
    if not np.all(np.isfinite(ts)):
        raise ValueError("The time series must not contain missing or non-finite values.")
    return ts, names


def fit_gnar1(
    A,
    ts: np.ndarray | pd.DataFrame,
    kappa: float | None = None,
    kappa_bounds: tuple[float, float] = (0.0, 1.5),
    grid: int = 201,
    sigma_2: float | None = None,
    *,
    demean: bool,
) -> GNARKappaFit:
    """
    Fit the kappa-normalised GNAR(1, [1]) standard model (node-specific alpha)

        X_{i,t} = alpha_i X_{i,t-1} + beta N_i^(-kappa) S_{i,t-1} + u_{i,t},    S_{i,t-1} = sum_{q in N(i)} X_{q,t-1},

    by least squares with kappa fixed, or with kappa estimated by profile likelihood.

    With kappa fixed the closed-form OLS of gnar.utils.gnar_profile_likelihood.fixed_kappa_ols is used. With kappa=None,
    RSS(kappa) is minimised over kappa_bounds (grid search refined by bounded Brent; see estimate_kappa), which maximises
    the profile likelihood l_p(kappa) = -(n/2) log(RSS(kappa)/n) - (n/2)(1 + log 2 pi), n = d (T - 1). Each evaluation
    of the profile costs O(d) after one O(dT) pass over the data. The noise variance is sigma^2 = RSS / (n - p), with
    p = d + 2 (kappa estimated) or d + 1 (kappa fixed), unless sigma_2 is given; a parameter that is not identified
    (kappa on a regular graph, beta without network information) is not counted in p.

    Warnings (subclasses of KappaWarning) are raised when the estimate lies on a bound, when the profile has several local
    minima, when it is flat, and when kappa is not identified (fewer than two distinct degrees among nodes with
    neighbours). They are also stored in the fit's warnings attribute.

    Params:
        A: np.array or scipy.sparse matrix. Binary, symmetric adjacency matrix with no self-loops. Shape (d, d)
        ts: np.array or pd.DataFrame. Time series, time x nodes. Shape (T, d)
        kappa: float or None. Fixed value of kappa, or None to estimate it. Defaults to None.
        kappa_bounds: tuple. Bounds of the search for kappa. Defaults to (0, 1.5).
        grid: int. Number of grid points for the profile. Defaults to 201.
        sigma_2: float or None. Known noise variance (e.g. 1 in simulations); None to estimate it.
        demean: bool. Whether to subtract the sample mean of each node before fitting. Required (no default yet).

    Returns:
        GNARKappaFit
    """
    A = check_kappa_graph(A)
    kappa = check_kappa(kappa)
    if sigma_2 is not None:
        sigma_2 = _check_sigma_2(sigma_2)
    kappa_bounds = check_kappa_bounds(kappa_bounds)
    if isinstance(grid, (bool, np.bool_)) or not isinstance(grid, (int, np.integer)) or grid < 3:
        raise ValueError("grid must be an integer of at least 3.")
    if not isinstance(demean, (bool, np.bool_)):
        raise ValueError("demean must be True or False.")
    d = A.shape[0]
    ts, names = _check_ts(ts, d)
    mu = np.mean(ts, axis=0, keepdims=True) if demean else np.zeros((1, d))
    x = ts - mu
    n = x.shape[0]
    n_obs = d * (n - 1)
    degrees = node_degrees(A)
    M = node_sums(x, A)
    R = residualised_sums(M)
    kappa_estimated = kappa is None
    if sigma_2 is None and n_obs <= d + 1 + int(kappa_estimated):
        raise ValueError(f"Too few observations ({n_obs}) for {d + 1 + int(kappa_estimated)} parameters.")

    profile = None
    if kappa_estimated:
        est = estimate_kappa(M, degrees, kappa_bounds, grid, sigma_2, R=R)
        found = list(est.warnings)
        if est.identified:
            kappa_hat = est.kappa
            alpha, beta, rss = fixed_kappa_ols(M, degrees, kappa_hat, R)
        else:
            # The fitted values do not depend on kappa, so alpha and RSS come from any kappa (here 1). beta is not
            # identified when the informative nodes share one degree above 1 (only beta N^(-kappa) is); with degree 1,
            # without network information (beta = 0) or with beta_hat = 0 for every kappa, beta_hat is well defined
            kappa_hat = np.nan
            alpha, beta, rss = fixed_kappa_ols(M, degrees, 1.0, R)
            common = np.unique(degrees[informative_nodes(M, R, degrees)])
            if common.size == 1 and common[0] > 1:
                beta = np.nan
        rss_min = min(float(np.min(est.rss_grid)), rss)
        profile = {"kappa": est.grid, "rss": est.rss_grid, "loglik": profile_loglik(est.rss_grid, n_obs, sigma_2),
                   "deviance": deviance(est.rss_grid, rss_min, n_obs, sigma_2)}
    else:
        kappa_hat = kappa
        alpha, beta, rss = fixed_kappa_ols(M, degrees, kappa, R)
        found = []
        if not has_network_information(M, R, degrees):
            found.append(("identifiability", "beta is not identified: the data carry no information on the network term; it is set to 0."))

    # Degrees of freedom: d alphas, beta if the data carry information on it, and kappa if it is estimated and identified
    p = d + int(has_network_information(M, R, degrees)) + int(kappa_estimated and np.isfinite(kappa_hat))
    sigma_2_hat = float(sigma_2) if sigma_2 is not None else rss / (n_obs - p)
    beta_for_cov = beta if np.isfinite(beta) else 0.0
    kappa_for_cov = kappa_hat if np.isfinite(kappa_hat) else 1.0
    cov_bk = beta_kappa_cov(R, degrees, beta_for_cov, kappa_for_cov, sigma_2_hat, kappa_estimated)
    if kappa_estimated and not np.isfinite(kappa_hat):
        cov_bk[1, 1] = np.inf

    for code, message in found:
        _warn_outside_package(message, _WARNING_CLASSES[code])

    return GNARKappaFit(
        alpha=alpha, beta=float(beta), kappa=float(kappa_hat), sigma_2=sigma_2_hat, rss=float(rss),
        loglik=profile_loglik(rss, n_obs, sigma_2), n_obs=n_obs, n=n, d=d, degrees=degrees, mu=mu, names=names,
        kappa_estimated=kappa_estimated, kappa_bounds=kappa_bounds, sigma_2_fixed=sigma_2 is not None, _num_params=p,
        profile=profile, warnings=found, _cov_beta_kappa=cov_bk, _A=A, _M=M, _R=R, _ts=x,
    )


def plot_profile(fit: GNARKappaFit, ax=None, level: float = 0.95, show_quadratic: bool = True):
    """
    Plot the profile deviance dev(kappa) = 2 (l_p(kappa_hat) - l_p(kappa)) against kappa, with a horizontal line at the
    chi-square(1) quantile for the chosen level (its intersection with the curve gives the profile-likelihood interval)
    and a vertical line at kappa_hat. With show_quadratic=True it overlays the Wald approximation
    (kappa - kappa_hat)^2 / SE(kappa_hat)^2; a large gap between the two curves signals weak identification.

    Params:
        fit: GNARKappaFit. A fit with kappa estimated.
        ax: matplotlib Axes, optional. Axes to draw on; a new figure is created if None.
        level: float. Confidence level of the cutoff line. Defaults to 0.95.
        show_quadratic: bool. Whether to overlay the Wald (quadratic) approximation. Defaults to True.

    Returns:
        matplotlib Axes
    """
    import matplotlib.pyplot as plt

    if not fit.kappa_estimated or fit.profile is None:
        raise ValueError("kappa was fixed in this fit, so there is no profile to plot.")
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1.")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    kappas, dev = fit.profile["kappa"], fit.profile["deviance"]
    cutoff = chi2.ppf(level, 1)
    ax.plot(kappas, dev, color="#2a78d6", linewidth=2, label="Profile deviance", zorder=3)
    if show_quadratic and np.isfinite(fit.kappa):
        se_kappa = fit._se_beta_kappa()[1]
        if np.isfinite(se_kappa) and se_kappa > 0:
            ax.plot(kappas, (kappas - fit.kappa) ** 2 / se_kappa ** 2, color="#eb6834", linewidth=2, linestyle="--",
                    label="Wald approximation", zorder=2)
    ax.axhline(cutoff, color="#6b6a65", linewidth=1, linestyle=":", label=f"χ²₁ {level:.0%} cutoff ({cutoff:.2f})", zorder=1)
    if np.isfinite(fit.kappa):
        ax.axvline(fit.kappa, color="#6b6a65", linewidth=1, label=f"κ̂ = {fit.kappa:.3f}", zorder=1)
    # Keep the region around the cutoff readable even when the deviance grows large away from the estimate
    top = max(1.5 * cutoff, min(float(np.max(dev)), 5 * cutoff)) * 1.05
    ax.set_xlim(kappas[0], kappas[-1])
    ax.set_ylim(0, top)
    ax.set_xlabel("κ")
    ax.set_ylabel("Deviance")
    ax.grid(True, color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    return ax
