"""
Likelihood-ratio comparison of the nested chain of network autoregressive models

    AR  ⊂  G  ⊂  K  ⊂  S  ⊂  L

on an unweighted, undirected graph without isolated nodes, all with Gaussian noise N(0, sigma_2 I):

    AR : X_{i,t} = alpha X_{i,t-1} + e                                   (no network effect)
    G  : X_{i,t} = alpha X_{i,t-1} + beta   d_i^{-1}     sum_j X_{j,t-1} + e   (global GNAR(1, [1]))
    K  : X_{i,t} = alpha X_{i,t-1} + beta   d_i^{-kappa} sum_j X_{j,t-1} + e   (KappaGNAR, kappa in [0, 1])
    S  : X_{i,t} = alpha X_{i,t-1} + beta_i d_i^{-1}     sum_j X_{j,t-1} + e   (global alpha, local beta)
    L  : X_{i,t} = alpha_i X_{i,t-1} + beta_i d_i^{-1}   sum_j X_{j,t-1} + e   (local GNAR(1, [1]))

(X demeaned). All five models are fitted from the same per-node sufficient statistics, computed in one pass over the data
with one sparse product, so every fit costs O(N) and all models share the same demeaning, the same conditioning on the
first observation and the same likelihood. This makes their log-likelihoods directly comparable.

Reference distributions for tests against S or L
------------------------------------------------
Tests whose alternative is S or L impose a number of restrictions q of order N (all beta_i equal, all alpha_i equal, ...).
Each node-specific coefficient is estimated from a single short series with lagged regressors and an estimated mean, which
shifts that node's contribution to the test statistic by O(1/T) (the Hurwicz / Nickell bias of dynamic models with fixed
effects). Summed over q ~ N restrictions the shift is O(N / T), while the spread of the statistic under the null is only
O(sqrt(N)), so the plain F and chi2 references are badly mis-sized on large graphs unless T >> sqrt(N): in simulations
with N = 2,000 the plain F test of S -> L rejected 0-1% of the time at the 5% level (T = 50 and 200), and K -> S 15% at
T = 50. Once its mean is corrected, however, the null distribution is very close to the F distribution (same spread,
skewness and upper quantiles). The default for these tests is therefore a Bartlett correction: the mean of the F statistic
under the fitted null is estimated from a small parametric bootstrap (49 draws by default) and the F statistic is divided
by it before using the F(q, df) distribution. In the same simulations its size was 2-7% for a 5% test, as accurate as a
full parametric bootstrap with 199 draws at a quarter of the cost, and it gives smooth p-values far into the tail. Neither
correction is exact, because the bootstrap simulates from the estimated rather than the true null model.

References: Nickell (1981), Biases in dynamic models with fixed effects, Econometrica 49; Hahn and Kuersteiner (2002),
Asymptotically unbiased inference for a dynamic panel model with fixed effects when both n and T are large, Econometrica 70.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import chi2, f as f_dist

from gnar.kappa import KappaGNAR, _ProfileStatistics, _PROFILE_GRID_SIZE, _as_csr, _degrees, check_kappa_adjacency

MODELS = ("AR", "G", "K", "S", "L")
DESCRIPTIONS = {
    "AR": "global alpha, no network effect",
    "G": "global alpha, global beta, kappa = 1 (global GNAR)",
    "K": "global alpha, global beta, kappa estimated",
    "S": "global alpha, local beta",
    "L": "local alpha, local beta (local GNAR)",
}


@dataclass
class NodewiseFit:
    """
    A fitted AR, S or L model. alpha and beta are per-node arrays (constant for global coefficients); beta is the
    coefficient on the neighbour *average* d_i^{-1} sum_j X_{j,t-1}, as in GNAR.
    """
    name: str
    alpha: np.ndarray
    beta: np.ndarray
    sigma_2: float
    loglik: float
    n_coef: int
    mu: np.ndarray
    _A: sp.csr_array = field(repr=False)
    _degrees: np.ndarray = field(repr=False)

    def phi(self) -> sp.csr_array:
        """Sparse VAR(1) coefficient matrix diag(alpha) + diag(beta / d) A."""
        return (sp.diags_array(self.alpha) + sp.diags_array(self.beta / self._degrees) @ self._A).tocsr()


@dataclass
class LRTestResult:
    """Result of a likelihood-ratio test of a nested model (null) against a larger one (alternative)."""
    null: str
    alternative: str
    statistic: float          # LR = 2 (loglik_alternative - loglik_null)
    df: int                   # number of restrictions
    method: str               # reference distribution used
    p_value: float
    f_statistic: float = np.nan        # plain F statistic
    df_denominator: float = np.nan
    bartlett_factor: float = np.nan    # bootstrap estimate of the mean of F under the null (method "bartlett")
    n_boot: int = 0

    def __repr__(self) -> str:
        extra = ""
        if self.method in ("F", "bartlett"):
            extra = f", F = {self.f_statistic:.4g} on ({self.df}, {self.df_denominator:.0f}) df"
        if self.method == "bartlett":
            extra += f", Bartlett factor = {self.bartlett_factor:.4g}"
        boot = f", B = {self.n_boot}" if self.method in ("bootstrap", "bartlett") else ""
        return (f"LR test {self.null} -> {self.alternative}: LR = {self.statistic:.4g}, df = {self.df}{extra}, "
                f"reference = {self.method}{boot}, p = {self.p_value:.4g}")


@dataclass
class SequentialTestResult:
    """Result of the sequential (bottom-up) test along the chain AR -> G -> K -> S -> L."""
    table: pd.DataFrame
    selected: str
    alpha: float
    tests: str
    selected_vs_L_p: float = np.nan     # p-value of the selected model against L (NaN if L is selected)

    @property
    def adequate(self) -> bool:
        """False if the selected model is rejected against L at level alpha."""
        return not (self.selected_vs_L_p <= self.alpha)

    def __repr__(self) -> str:
        note = ""
        if self.selected != "L":
            verdict = "rejected" if not self.adequate else "not rejected"
            note = f" ({verdict} against L, p = {self.selected_vs_L_p:.4g})"
        return (f"Sequential test along AR -> G -> K -> S -> L ({self.tests} tests, alpha = {self.alpha:g})\n"
                f"{self.table.to_string()}\nSelected model: {self.selected}{note}")


def _fit_from_stats(name: str, st: _ProfileStatistics, degrees: np.ndarray) -> dict:
    """
    Fit one model from the sufficient statistics. Returns a dict with the residual sum of squares, the number of
    regression coefficients and the coefficients (alpha and beta per node, beta on the neighbour average; kappa for K).
    """
    N = degrees.size
    if name == "AR":
        a = st.sxy / st.sxx
        return {"rss": st.syy - a * st.sxy, "n_coef": 1, "alpha": np.full(N, a), "beta": np.zeros(N)}
    if name in ("G", "K"):
        if name == "G":
            kappa = 1.0
        else:
            grid = np.linspace(0.0, 1.0, _PROFILE_GRID_SIZE)
            kappa = KappaGNAR._maximise_profile(st, grid, st.solve(grid)[2])
        a, bt, rss = (v[0] for v in st.solve(kappa))
        # Coefficient on the neighbour average: beta d_i^{-kappa} = (beta d_i^{1 - kappa}) d_i^{-1}
        return {"rss": rss, "n_coef": 2 if name == "G" else 3, "alpha": np.full(N, a),
                "beta": bt * degrees ** (1 - kappa), "kappa": kappa, "beta_global": bt}
    if np.any(st.q <= 0):
        raise ValueError("Some nodes have identically zero neighbour sums: models with local beta cannot be fitted.")
    if name == "S":
        # For fixed alpha, beta_i = (u_i - alpha r_i) / q_i; substituting leaves a quadratic in alpha
        a = np.sum(st.c - st.u * st.r / st.q) / np.sum(st.x - st.r ** 2 / st.q)
        b_raw = (st.u - a * st.r) / st.q
        rss = st.syy - a * st.sxy - b_raw @ st.u
        return {"rss": rss, "n_coef": N + 1, "alpha": np.full(N, a), "beta": b_raw * degrees}
    if name == "L":
        det = st.x * st.q - st.r ** 2
        if np.any(det <= 0):
            raise ValueError("The regressors of some nodes are collinear: the local model cannot be fitted.")
        a = (st.q * st.c - st.r * st.u) / det
        b_raw = (st.x * st.u - st.r * st.c) / det
        rss = st.syy - a @ st.c - b_raw @ st.u
        return {"rss": rss, "n_coef": 2 * N, "alpha": a, "beta": b_raw * degrees}
    raise ValueError(f"Unknown model {name!r}; expected one of {MODELS}.")


class ModelComparison:
    """
    Fit and compare the nested chain of models AR ⊂ G ⊂ K ⊂ S ⊂ L (see the module docstring) with likelihood-ratio tests.

    All models are fitted by Gaussian (conditional) maximum likelihood with noise variance sigma_2 I, from shared
    per-node sufficient statistics, so fitting all five costs one sparse pass over the data plus O(N) per model.

    Parameters:
        A (np.ndarray or scipy sparse matrix): Binary, symmetric adjacency matrix without self-loops or isolated nodes;
            the graph must not be regular (kappa is then not identifiable).
        ts (np.ndarray or pd.DataFrame): Time series of shape (T, N).
        demean (bool): Whether to remove the sample mean of each node before fitting (as GNAR does). Defaults to True.

    Methods:
        table: Log-likelihood, number of parameters, sigma_2, AIC, BIC and key estimates of each model.
        lr_test: Likelihood-ratio test of one model against a larger one.
        sequential_test: Bottom-up sequential test along the chain, with family-wise error control.
        [name]: The fitted model (KappaGNAR for G and K, NodewiseFit for AR, S and L).
    """

    def __init__(self, A, ts: np.ndarray | pd.DataFrame, demean: bool = True) -> None:
        check_kappa_adjacency(A)
        self._A = _as_csr(A)
        self._degrees = _degrees(self._A)
        self._N = self._A.shape[0]
        values = ts.to_numpy(dtype=float) if isinstance(ts, pd.DataFrame) else np.asarray(ts, dtype=float)
        if values.ndim != 2 or values.shape[1] != self._N:
            raise ValueError(f"ts must have shape (T, {self._N}): one column per node of the graph.")
        if values.shape[0] < 5:
            raise ValueError("At least five time steps are required to compare the models (the local model needs "
                             "positive residual degrees of freedom).")
        if not np.all(np.isfinite(values)):
            raise ValueError("ts contains missing or infinite values.")
        if np.all(self._degrees == self._degrees[0]):
            raise ValueError("The graph is regular, so kappa (model K) is not identifiable.")

        self._ts = ts
        self._demean = demean
        self._T = values.shape[0]
        self._mu = values.mean(axis=0, keepdims=True) if demean else np.zeros((1, self._N))
        self._Y = values - self._mu
        self._stats = _ProfileStatistics(self._Y, self._A, np.log(self._degrees))
        self._fits = {name: _fit_from_stats(name, self._stats, self._degrees) for name in MODELS}
        self._models = {}

    # ------------------------------------------------------------------ fitted models
    @property
    def n_obs(self) -> int:
        """Number of observations in the likelihood, (T - 1) N."""
        return self._stats.n_obs

    def _loglik(self, rss: float) -> float:
        return float(self._stats.loglik(rss))

    def __getitem__(self, name: str):
        """The fitted model: a KappaGNAR for G (kappa fixed at 1) and K, a NodewiseFit for AR, S and L."""
        if name not in MODELS:
            raise KeyError(f"Unknown model {name!r}; expected one of {MODELS}.")
        if name not in self._models:
            fit = self._fits[name]
            if name in ("G", "K"):
                # Identification warnings for K (if any) are raised here, the first time K is accessed
                self._models[name] = KappaGNAR(self._A, ts=self._ts, kappa=1.0 if name == "G" else None,
                                               demean=self._demean)
            else:
                self._models[name] = NodewiseFit(name, fit["alpha"], fit["beta"], fit["rss"] / self.n_obs,
                                                 self._loglik(fit["rss"]), fit["n_coef"], self._mu, self._A,
                                                 self._degrees)
        return self._models[name]

    def table(self) -> pd.DataFrame:
        """
        One row per model: description, number of regression coefficients and of parameters (coefficients + sigma_2;
        the node means are not counted), log-likelihood, sigma_2, AIC, BIC, and the key estimates: the global alpha
        (NaN for L), the global b and beta (G and K) and kappa (1 for G, estimated for K).
        """
        rows = {}
        for name in MODELS:
            fit = self._fits[name]
            k = fit["n_coef"] + 1
            ll = self._loglik(fit["rss"])
            global_alpha = fit["alpha"][0] if name != "L" else np.nan
            b = beta = kappa = np.nan
            if name in ("G", "K"):
                kappa, beta = fit["kappa"], fit["beta_global"]
                b = self[name].b
            rows[name] = {"description": DESCRIPTIONS[name], "n_coef": fit["n_coef"], "n_params": k, "loglik": ll,
                          "sigma_2": fit["rss"] / self.n_obs, "AIC": -2 * ll + 2 * k,
                          "BIC": -2 * ll + k * np.log(self.n_obs), "alpha": global_alpha, "b": b, "beta": beta,
                          "kappa": kappa}
        return pd.DataFrame.from_dict(rows, orient="index")

    # ------------------------------------------------------------------ tests
    @staticmethod
    def _check_pair(null: str, alternative: str) -> None:
        for name in (null, alternative):
            if name not in MODELS:
                raise ValueError(f"Unknown model {name!r}; expected one of {MODELS}.")
        if MODELS.index(null) >= MODELS.index(alternative):
            raise ValueError(f"{null} is not nested in {alternative}: the null model must come before the alternative "
                             f"in the chain AR -> G -> K -> S -> L.")
        if (null, alternative) == ("AR", "K"):
            raise ValueError("AR -> K is not tested: under AR (beta = 0) kappa is not identified, so the likelihood-ratio "
                             "statistic has no standard reference distribution. Test AR -> G and then G -> K instead.")

    def lr_test(self, null: str, alternative: str, method: str = "auto", n_boot: int | None = None,
                seed: int | None = None) -> LRTestResult:
        """
        Likelihood-ratio test of the model `null` against the larger model `alternative` (both in AR, G, K, S, L, with
        null before alternative; AR -> K is not available because kappa is not identified under AR).

        The statistic is LR = 2 (loglik_alternative - loglik_null) = (T - 1) N log(RSS_null / RSS_alternative), with
        q = difference in the number of coefficients restrictions. Reference distributions (method):

            "F"         : F = [(RSS_null - RSS_alt) / q] / [RSS_alt / df], df = M - k_alt - N_means, compared with
                          F(q, df); M = (T - 1) N and N_means = N if demean else 0. Accurate for AR -> G. For tests against
                          S or L (q of order N) it is badly mis-sized on large graphs: see the module docstring.
            "chi2"      : LR ~ chi2(q). Same caveat as "F", and worse.
            "boundary"  : only for G -> K, whose null kappa = 1 lies on the boundary of [0, 1]:
                          LR ~ 1/2 chi2(0) + 1/2 chi2(1), i.e. p = P(chi2(1) > LR) / 2 for LR > 0 and p = 1 for LR = 0.
            "bartlett"  : Bartlett-corrected F: the mean of the F statistic under the fitted null model is estimated by a
                          parametric bootstrap with n_boot draws (default 49), and F / mean is compared with F(q, df).
                          Corrects the O(1/T)-per-node shift that ruins the plain F test when q grows with N.
            "bootstrap" : parametric bootstrap: p = (1 + #{LR* >= LR}) / (n_boot + 1), with n_boot draws (default 199).
            "auto"      : "F" for AR -> G, "boundary" for G -> K, "bartlett" for every test against S or L.

        Both bootstrap-based methods simulate n_boot data sets from the fitted null model, starting from the observed first
        time point (matching the conditional likelihood) and using the observed number of time steps, demean them as the
        data were demeaned, refit both models and recompute the statistic. They use their own random generator (seed), so
        they do not change np.random. Their size is approximately, not exactly, correct because they simulate from the
        estimated rather than the true null model (2-7% for a 5% test in simulations with N = 2,000).

        For the tests whose null is K (K -> S, K -> L), the null estimate of kappa may lie on the boundary; with q of order
        N the resulting change in the reference distribution is negligible.
        """
        self._check_pair(null, alternative)
        if method == "auto":
            if (null, alternative) == ("G", "K"):
                method = "boundary"
            elif alternative in ("S", "L"):
                method = "bartlett"
            else:
                method = "F"
        if method == "boundary" and (null, alternative) != ("G", "K"):
            raise ValueError("The boundary reference is only valid for the test G -> K (kappa = 1).")
        if method not in ("F", "chi2", "boundary", "bartlett", "bootstrap"):
            raise ValueError("method must be one of 'auto', 'F', 'chi2', 'boundary', 'bartlett', 'bootstrap'.")
        if n_boot is None:
            n_boot = 49 if method == "bartlett" else 199
        if method in ("bartlett", "bootstrap") and n_boot < 1:
            raise ValueError("n_boot must be at least 1.")

        rss0, rss1 = self._fits[null]["rss"], self._fits[alternative]["rss"]
        k0, k1 = self._fits[null]["n_coef"], self._fits[alternative]["n_coef"]
        q = k1 - k0
        lr = max(self.n_obs * np.log(rss0 / rss1), 0.0)
        result = LRTestResult(null, alternative, float(lr), int(q), method, np.nan)

        df2 = self.n_obs - k1 - (self._N if self._demean else 0)
        if method in ("F", "bartlett"):
            if df2 <= 0:
                raise ValueError("Not enough observations for the F test (non-positive residual degrees of freedom).")
            fstat = self._f_from_lr(lr, q, df2)
            result.f_statistic, result.df_denominator = float(fstat), float(df2)
        if method == "F":
            result.p_value = float(f_dist.sf(fstat, q, df2))
        elif method == "chi2":
            result.p_value = float(chi2.sf(lr, q))
        elif method == "boundary":
            result.p_value = 1.0 if lr <= 0 else float(0.5 * chi2.sf(lr, 1))
        elif method == "bartlett":
            factor = float(np.mean(self._f_from_lr(self._bootstrap_lr(null, alternative, n_boot, seed), q, df2)))
            result.bartlett_factor, result.n_boot = factor, n_boot
            result.p_value = float(f_dist.sf(fstat / factor, q, df2))
        else:
            boot = self._bootstrap_lr(null, alternative, n_boot, seed)
            result.p_value = float((1 + np.sum(boot >= lr)) / (n_boot + 1))
            result.n_boot = n_boot
        return result

    def _f_from_lr(self, lr, q: int, df2: float):
        """F statistic [(RSS_0 - RSS_1) / q] / [RSS_1 / df2], written in terms of LR = M log(RSS_0 / RSS_1)."""
        return np.expm1(np.asarray(lr) / self.n_obs) * df2 / q

    def _bootstrap_lr(self, null: str, alternative: str, n_boot: int, seed: int | None) -> np.ndarray:
        """Bootstrap distribution of the LR statistic under the fitted null model."""
        rng = np.random.default_rng(seed)
        fit0 = self._fits[null]
        phi = (sp.diags_array(fit0["alpha"]) + sp.diags_array(fit0["beta"] / self._degrees) @ self._A).tocsr()
        sd = np.sqrt(fit0["rss"] / self.n_obs)
        start = self._Y[0]
        log_deg = np.log(self._degrees)
        stats = np.empty(n_boot)
        for i in range(n_boot):
            Y = np.empty_like(self._Y)
            Y[0] = start
            noise = rng.standard_normal(Y[1:].shape) * sd
            for t in range(1, self._T):
                Y[t] = phi @ Y[t - 1] + noise[t - 1]
            if self._demean:
                Y = Y - Y.mean(axis=0, keepdims=True)
            st = _ProfileStatistics(Y, self._A, log_deg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                rss0 = _fit_from_stats(null, st, self._degrees)["rss"]
                rss1 = _fit_from_stats(alternative, st, self._degrees)["rss"]
            stats[i] = max(st.n_obs * np.log(rss0 / rss1), 0.0)
        return stats

    def sequential_test(self, alpha: float = 0.05, tests: str = "combined", method: str = "auto",
                        n_boot: int | None = None, seed: int | None = None) -> SequentialTestResult:
        """
        Bottom-up sequential test along the chain: test AR, then G, then K, then S, and stop at the first model that is not
        rejected; that model is selected (L if every model is rejected).

        Each step tests the current model M with a level-alpha test, chosen by `tests`:

            "combined" (default) : reject M if the adjacent test (M against the next model) or the omnibus test (M against
                                   L) rejects at level alpha / 2, i.e. step p-value = min(1, 2 min(p_adjacent, p_vs_L))
                                   (Bonferroni within the step; for S the two tests coincide). Keeps most of the power of
                                   the targeted adjacent test while still detecting deviations in other directions.
            "adjacent"           : M against the next model only. Most powerful against the next extension, but can stop
                                   early when M is wrong in another direction: e.g. with data from L whose node-specific
                                   coefficients are unrelated to degree, G -> K has nothing to detect, so the procedure stops
                                   at G although G -> L rejects strongly.
            "vs_L"               : M against L only (omnibus): detects any deviation, but a single restriction such as
                                   kappa = 1 is diluted among about 2N degrees of freedom.

        Because the hypotheses are nested (AR true => G true => K true => S true), this is closed testing on a chain: each
        step only needs to be a valid level-alpha test of its own null, no further adjustment is needed, and the
        family-wise error rate is at most alpha whichever `tests` is used (choose it before looking at the results). The
        adjusted p-value of each step is the running maximum of the step p-values, and a model is rejected iff its adjusted
        p-value is <= alpha. All steps are computed and reported; the steps after the first non-rejection are marked as
        not reached.

        Adequacy warning: whatever `tests` is, if the selected model is not L and its test against L rejects at level
        alpha, a UserWarning is raised and the p-value is stored in the result (selected_vs_L_p), since the selected model
        is then the simplest model not rejected by the procedure but is not supported by the data as a whole. Not rejecting
        a model does not show that it is correct, and confidence intervals of the selected model do not account for the
        selection.

        method, n_boot and seed are passed to lr_test for every test (with "auto", each test uses its default reference).
        When seed is given, each distinct test (null, alternative) gets its own seed derived from it. The bootstrap-based
        p-values carry Monte Carlo noise (with the default 49 draws, a p-value around 0.6 can move by about 0.1 between
        seeds; values near the usual levels move much less); increase n_boot for more stable p-values.
        """
        if tests not in ("combined", "adjacent", "vs_L"):
            raise ValueError("tests must be 'combined', 'adjacent' or 'vs_L'.")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1).")
        cache = {}

        def run(null: str, alt: str) -> LRTestResult:
            if (null, alt) not in cache:
                m = method if method != "boundary" or (null, alt) == ("G", "K") else "auto"
                pair_seed = None if seed is None else seed + 10 * MODELS.index(null) + MODELS.index(alt)
                cache[(null, alt)] = self.lr_test(null, alt, method=m, n_boot=n_boot, seed=pair_seed)
            return cache[(null, alt)]

        rows = []
        running_max = 0.0
        selected = None
        for i, null in enumerate(MODELS[:-1]):
            nxt = MODELS[i + 1]
            row = {}
            if tests == "adjacent":
                res = run(null, nxt)
                row.update(test=f"{null} -> {nxt}", statistic=res.statistic, df=res.df, reference=res.method,
                           p_value=res.p_value)
            elif tests == "vs_L":
                res = run(null, "L")
                row.update(test=f"{null} -> L", statistic=res.statistic, df=res.df, reference=res.method,
                           p_value=res.p_value)
            else:
                adj, omni = run(null, nxt), run(null, "L")
                step_p = adj.p_value if nxt == "L" else min(1.0, 2 * min(adj.p_value, omni.p_value))
                row.update(test=null if nxt == "L" else f"{null} -> {nxt} | {null} -> L", p_adjacent=adj.p_value,
                           p_vs_L=omni.p_value, p_value=step_p)
            running_max = max(running_max, row["p_value"])
            reject = running_max <= alpha
            reached = selected is None
            if reached and not reject:
                selected = null
            row.update(adjusted_p=running_max, reject=reject, reached=reached)
            rows.append(row)
        if tests == "combined":
            rows[-1]["test"] = "S -> L"
        selected = selected or "L"

        selected_vs_L_p = np.nan
        if selected != "L":
            selected_vs_L_p = run(selected, "L").p_value
            if selected_vs_L_p <= alpha:
                warnings.warn(f"The selected model {selected} is rejected against the local model L "
                              f"(p = {selected_vs_L_p:.3g}): it is the simplest model not rejected by the sequential "
                              f"procedure, but it does not fit the data as well as L.", UserWarning)
        return SequentialTestResult(pd.DataFrame(rows).set_index("test"), selected, alpha, tests, selected_vs_L_p)

    def __repr__(self) -> str:
        return f"ModelComparison(N={self._N}, T={self._T}, models={MODELS}, demean={self._demean})"
