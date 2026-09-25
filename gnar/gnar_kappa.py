from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2

from gnar.utils.data_utils import check_kappa, check_kappa_graph
from gnar.utils.neighbour_sets import node_degrees
from gnar.utils.gnar_profile_likelihood import (node_sums, residualised_sums, fixed_kappa_ols, estimate_kappa,
                                                profile_loglik, deviance, has_network_information)
from gnar.utils.gnar_inference import (beta_kappa_cov, cov_fast, var_fast, wald_ci, wald_test, profile_ci,
                                       beta_at_reference)


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


class GNARKappaFit:
    """
    Fit of the kappa-normalised GNAR(1, [1]) standard model

        X_{i,t} = alpha_i X_{i,t-1} + beta N_i^(-kappa) S_{i,t-1} + u_{i,t},    S_{i,t-1} = sum_{q in N(i)} X_{q,t-1},

    returned by fit_gnar1. Parameters are ordered theta = (alpha_1, ..., alpha_d, beta, kappa).

    Attributes:
        alpha (np.ndarray): Estimates of alpha_i. Shape (d,)
        beta (float): Estimate of beta (NaN if not identified).
        kappa (float): Estimate of kappa, or its fixed value (NaN if estimated but not identified).
        sigma_2 (float): Noise variance: RSS / (n_obs - p) with p = d + 2 (kappa estimated) or d + 1 (kappa fixed), or
            the fixed value passed to fit_gnar1. When kappa (or beta) is not identified it does not count in p.
        rss (float): Residual sum of squares RSS(kappa_hat).
        loglik (float): Profile log-likelihood at kappa_hat (see gnar.utils.gnar_profile_likelihood.profile_loglik).
        se (np.ndarray): Standard errors of theta_hat, from the full Jacobian at theta_hat, so they account for estimating
            kappa. With kappa fixed they are the usual OLS standard errors treating kappa as known, and the kappa entry is
            NaN. Shape (d + 2,)
        cov (np.ndarray): Covariance sigma^2 (J^T J)^(-1) of theta_hat, computed on first access. Shape (d + 2, d + 2)
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
        self._cov = None
        self.se = np.sqrt(self._covariance(diagonal=True))

    @property
    def num_params(self) -> int:
        # Number of estimated mean parameters: d alphas, beta and, if estimated, kappa
        return self.d + 1 + int(self.kappa_estimated)

    @property
    def cov(self) -> np.ndarray:
        if self._cov is None:
            self._cov = self._covariance(diagonal=False)
        return self._cov

    @property
    def param_names(self) -> list:
        return [f"alpha[{name}]" for name in self.names] + ["beta", "kappa"]

    def _sigma_2_known(self) -> float | None:
        # The noise variance used by the deviance: the fixed value, or None when it is profiled out
        return self.sigma_2 if self.sigma_2_fixed else None

    def _covariance(self, diagonal: bool) -> np.ndarray:
        # Covariance of theta_hat (or its diagonal) from the per-node sums, see cov_fast. The covariance with kappa known
        # is used when kappa is fixed, and for the identified parameters when it is not identified
        d, M, R, degrees = self.d, self._M, self._R, self.degrees
        f = var_fast if diagonal else cov_fast
        if not has_network_information(M, R, degrees):
            # No information on beta: only the alphas are estimated (beta is set to 0)
            inv_xx = np.divide(1.0, M.xx, out=np.full(d, np.nan), where=M.xx > 0)
            if diagonal:
                return np.concatenate([self.sigma_2 * inv_xx, [np.nan, np.nan]])
            cov = np.full((d + 2, d + 2), np.nan)
            cov[:d, :d] = np.diag(self.sigma_2 * inv_xx)
            return cov
        if not self.kappa_estimated:
            return f(M, R, degrees, self.beta, self.kappa, self.sigma_2, kappa_estimated=False)
        if np.isfinite(self.kappa):
            return f(M, R, degrees, self.beta, self.kappa, self.sigma_2, kappa_estimated=True)
        # kappa is not identified: the fitted values do not depend on it, so the alphas (and beta, if identified) take the
        # covariance of the kappa-known fit at kappa = 1
        beta_1 = fixed_kappa_ols(M, degrees, 1.0, R)[1]
        out = f(M, R, degrees, beta_1, 1.0, self.sigma_2, kappa_estimated=False)
        if diagonal:
            if not np.isfinite(self.beta):
                out[d] = np.inf
            out[d + 1] = np.inf
        else:
            if not np.isfinite(self.beta):
                out[d, :] = out[:, d] = np.nan
                out[d, d] = np.inf
            out[d + 1, d + 1] = np.inf
        return out

    def _se_beta_kappa(self) -> tuple[float, float]:
        # Standard errors of beta_hat and kappa_hat
        return float(self.se[self.d]), float(self.se[self.d + 1])

    def _require_estimated(self) -> None:
        if not self.kappa_estimated:
            raise ValueError("kappa was fixed in this fit; refit with kappa=None to make inference on kappa.")

    def profile_interval(self, level: float = 0.95) -> dict:
        """
        Profile-likelihood interval for kappa: {kappa : dev(kappa) <= chi-square(1) quantile at level}. See
        gnar.utils.gnar_inference.profile_ci.

        Returns:
            dict with keys "lower", "upper", "lower_open", "upper_open" (True when the deviance stays below the cutoff
            up to that bound, which is then returned as the end) and "disconnected".
        """
        self._require_estimated()
        if not np.isfinite(self.kappa):
            return {"lower": np.nan, "upper": np.nan, "lower_open": True, "upper_open": True, "disconnected": False}
        return profile_ci(self._M, self._R, self.degrees, self.kappa, self.rss, self.profile["kappa"], self.n_obs,
                          self._sigma_2_known(), level)

    def ci(self, level: float = 0.95, method: str = "profile") -> pd.DataFrame:
        """
        Confidence intervals for all parameters, and beta_0 = beta N_0^(-kappa) at the default reference degree (see
        beta_at).

        With method="wald" every interval is estimate +- z SE. With method="profile" (the default) kappa gets the
        profile-likelihood interval, which is preferable when identification is weak, and the other parameters Wald
        intervals; the "method" column says which. With kappa fixed its row is marked "fixed" and has no interval.

        Params:
            level: float. Confidence level. Defaults to 0.95.
            method: str. "profile" or "wald".

        Returns:
            pd.DataFrame with columns estimate, se, lower, upper, method, note, indexed by parameter.
        """
        if method not in ("profile", "wald"):
            raise ValueError("method must be 'profile' or 'wald'.")
        if not 0 < level < 1:
            raise ValueError("level must be between 0 and 1.")
        d = self.d
        ref = self.beta_at()
        estimate = np.concatenate([self.alpha, [self.beta, ref["beta_0"], self.kappa]])
        se = np.concatenate([self.se[:d + 1], [ref["se"], self.se[d + 1]]])
        lower, upper = wald_ci(estimate, se, level)
        methods, notes = ["wald"] * (d + 3), [""] * (d + 3)
        notes[d + 1] = f"N0 = {ref['N0']:.4g}"
        if not self.kappa_estimated:
            lower[-1] = upper[-1] = np.nan
            methods[-1] = "fixed"
        elif not np.isfinite(self.kappa):
            lower[-1] = upper[-1] = np.nan
            methods[-1], notes[-1] = "none", "not identified"
        elif method == "profile":
            interval = self.profile_interval(level)
            lower[-1], upper[-1], methods[-1] = interval["lower"], interval["upper"], "profile"
            flags = [text for key, text in (("lower_open", "open below"), ("upper_open", "open above"), ("disconnected", "disconnected"))
                     if interval[key]]
            notes[-1] = ", ".join(flags)
        index = self.param_names[:d + 1] + ["beta_0"] + ["kappa"]
        return pd.DataFrame({"estimate": estimate, "se": se, "lower": lower, "upper": upper, "method": methods, "note": notes}, index=index)

    def test_kappa(self, k0: float = 1.0) -> dict:
        """
        Test of kappa = k0 (by default the standard GNAR normalisation, k0 = 1).

        Returns the Wald statistic z = (kappa_hat - k0) / SE(kappa_hat) with its two-sided normal p-value, and the profile
        likelihood-ratio statistic dev(k0) with its chi-square(1) p-value. Neither is valid when kappa_hat is on a bound.

        Params:
            k0: float. Value of kappa under the null hypothesis. Defaults to 1.

        Returns:
            dict with keys k0, z, p_value, lr, lr_p_value.
        """
        self._require_estimated()
        k0 = check_kappa(k0, allow_none=False)
        se_kappa = self.se[self.d + 1]
        if np.isfinite(self.kappa) and np.isfinite(se_kappa) and se_kappa > 0:
            z, p_value = wald_test(self.kappa, se_kappa, k0)
            rss_0 = fixed_kappa_ols(self._M, self.degrees, k0, self._R)[2]
            lr = deviance(rss_0, self.rss, self.n_obs, self._sigma_2_known())
            lr_p_value = float(chi2.sf(lr, 1))
        else:
            z = p_value = lr = lr_p_value = np.nan
        return {"k0": k0, "z": z, "p_value": p_value, "lr": float(lr), "lr_p_value": lr_p_value}

    def beta_at(self, N0: float | None = None) -> dict:
        """
        Network coefficient at a reference degree, beta_0 = beta N_0^(-kappa), with its delta-method standard error

            Var(beta_0) ~= beta_0^2 [Var(beta)/beta^2 - 2 log N_0 Cov(beta, kappa)/beta + log^2 N_0 Var(kappa)]

        (see gnar.utils.gnar_inference.beta_at_reference). beta_0 is much less correlated with kappa_hat than beta when
        N_0 is a typical degree. On a regular graph beta_0 at the common degree is the identified quantity.

        Params:
            N0: float, optional. Reference degree; defaults to the geometric mean of the degrees of the nodes with
                N_i >= 1, exp(mean log N_i).

        Returns:
            dict with keys N0, beta_0, se.
        """
        connected = self.degrees[self.degrees > 0]
        if N0 is None:
            N0 = float(np.exp(np.mean(np.log(connected)))) if connected.size else 1.0
        if not N0 > 0:
            raise ValueError("N0 must be positive.")
        if self.kappa_estimated and not np.isfinite(self.kappa):
            # kappa is not identified; beta N^(-kappa) is identified at the common degree, from the fit at kappa = 1
            common = np.unique(connected)
            if common.size == 1 and np.isclose(N0, common[0]):
                beta_1 = fixed_kappa_ols(self._M, self.degrees, 1.0, self._R)[1]
                var_1 = var_fast(self._M, self._R, self.degrees, beta_1, 1.0, self.sigma_2, kappa_estimated=False)[self.d]
                return {"N0": float(N0), "beta_0": beta_1 / N0, "se": float(np.sqrt(var_1)) / N0}
            return {"N0": float(N0), "beta_0": np.nan, "se": np.inf}
        cov_bk = self.cov[self.d:, self.d:] if self._cov is not None else self._cov_beta_kappa
        beta_0, se = beta_at_reference(self.beta, self.kappa, cov_bk, N0, self.kappa_estimated)
        return {"N0": float(N0), "beta_0": beta_0, "se": se}

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        """
        Table of estimates, standard errors and confidence intervals (profile interval for kappa, Wald otherwise), with
        beta_0 = beta N_0^(-kappa) at the geometric-mean degree next to beta. print(fit) also shows the fit statistics,
        the test of kappa = 1 and any warnings.

        Params:
            level: float. Confidence level. Defaults to 0.95.

        Returns:
            pd.DataFrame
        """
        return self.ci(level=level, method="profile")

    def __repr__(self) -> str:
        status = "estimated" if self.kappa_estimated else "fixed"
        return f"GNARKappaFit(d={self.d}, n={self.n}, kappa={self.kappa:.6g} ({status}), beta={self.beta:.6g})"

    def __str__(self) -> str:
        status = "estimated" if self.kappa_estimated else "fixed"
        lines = [f"Kappa-normalised GNAR(1, [1]) fit: d = {self.d} nodes, T = {self.n} time points, n = {self.n_obs} observations",
                 f"kappa {status}" + (f" in [{self.kappa_bounds[0]:g}, {self.kappa_bounds[1]:g}]" if self.kappa_estimated else f" at {self.kappa:g}"),
                 f"sigma^2 = {self.sigma_2:.6g} ({'fixed' if self.sigma_2_fixed else 'estimated'}), RSS = {self.rss:.6g}, loglik = {self.loglik:.6g}",
                 "",
                 self.summary().to_string(float_format=lambda x: f"{x:.6g}")]
        if self.kappa_estimated and np.isfinite(self.kappa):
            test = self.test_kappa(1.0)
            lines += ["", f"Test of kappa = 1: Wald z = {test['z']:.4g} (p = {test['p_value']:.4g}), LR = {test['lr']:.4g} (p = {test['lr_p_value']:.4g})"]
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
    if sigma_2 is not None and not (np.isfinite(sigma_2) and sigma_2 > 0):
        raise ValueError("sigma_2 must be a positive number or None.")
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
    if sigma_2 is None and n_obs <= d + 2:
        raise ValueError(f"Too few observations ({n_obs}) for {d + 2} parameters.")

    profile = None
    if kappa_estimated:
        est = estimate_kappa(M, degrees, kappa_bounds, grid, sigma_2, R=R)
        found = list(est.warnings)
        if est.identified:
            kappa_hat = est.kappa
            alpha, beta, rss = fixed_kappa_ols(M, degrees, kappa_hat, R)
        else:
            # The fitted values do not depend on kappa, so alpha and RSS come from any kappa (here 1). beta is identified
            # only when every node with neighbours has degree 1 (then N_i^(-kappa) = 1), or trivially 0 without network
            # information
            kappa_hat = np.nan
            alpha, beta, rss = fixed_kappa_ols(M, degrees, 1.0, R)
            common = np.unique(degrees[degrees > 0])
            if has_network_information(M, R, degrees) and not (common.size == 1 and common[0] == 1):
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
        warnings.warn(message, _WARNING_CLASSES[code], stacklevel=2)

    return GNARKappaFit(
        alpha=alpha, beta=float(beta), kappa=float(kappa_hat), sigma_2=sigma_2_hat, rss=float(rss),
        loglik=profile_loglik(rss, n_obs, sigma_2), n_obs=n_obs, n=n, d=d, degrees=degrees, mu=mu, names=names,
        kappa_estimated=kappa_estimated, kappa_bounds=tuple(float(b) for b in kappa_bounds), sigma_2_fixed=sigma_2 is not None,
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
    # The curves rise away from kappa_hat, so the legend goes in the upper corner farther from it
    far_left = np.isfinite(fit.kappa) and fit.kappa > (kappas[0] + kappas[-1]) / 2
    ax.legend(frameon=False, fontsize="small", loc="upper left" if far_left else "upper right")
    return ax
