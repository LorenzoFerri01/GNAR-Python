"""
Tests for estimation with kappa fixed (closed-form OLS) or estimated (profile likelihood), the optimiser and its
warnings, and plot_profile (pull request 2).

Acceptance tests covered here: 3 (closed form vs lstsq), 4 (profile optimum vs joint NLS), 10 (regular graph),
12 (isolated node), 13 (boundary warning), and part (c) of test 1 for the coefficients.
"""
import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import least_squares
from scipy.stats import chi2

from gnar import (GNAR, simulate_gnar1, stationary_params, fit_gnar1, plot_profile, KappaWarning, KappaBoundaryWarning,
                  KappaFlatProfileWarning, KappaIdentifiabilityWarning)
from gnar.utils.gnar_inference import beta_kappa_cov
from gnar.utils.gnar_linear_regression import design_matrix
from gnar.utils.gnar_profile_likelihood import (node_sums, residualised_sums, fixed_kappa_ols, rss_profile, rss_derivative,
                                                profile_loglik, deviance, _count_local_minima)
from gnar.utils.neighbour_sets import node_degrees, degree_terms
from tests.data.make_legacy_reference import build, load_case
from tests.kappa_graphs import path_graph, cycle_graph, star_graph, random_tree, erdos_renyi, with_isolated_node

HETERO = with_isolated_node(np.array([[0, 1, 0, 0, 0],
                                      [1, 0, 1, 1, 0],
                                      [0, 1, 0, 1, 0],
                                      [0, 1, 1, 0, 1],
                                      [0, 0, 0, 1, 0]], dtype=float))


def simulate(A, kappa, n, seed, b=0.4, sigma_2=1.0):
    """Simulate with the spec's defaults: alpha_i ~ U[0.1, 0.4] and beta from stationary_params with b = 0.4."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.1, 0.4, A.shape[0])
    beta = stationary_params(A, kappa, b, alpha)
    return simulate_gnar1(A, alpha, beta, kappa, n, sigma_2=sigma_2, rng=rng), alpha, beta


def quiet_fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", KappaWarning)
        return fit_gnar1(*args, **kwargs)


class TestClosedFormOLS:
    """Acceptance test 3: closed-form OLS versus np.linalg.lstsq on the full design, kappa fixed."""

    @pytest.mark.parametrize("graph", ["hetero", "tree", "er", "star"])
    @pytest.mark.parametrize("kappa", [0.0, 0.3, 1.0, 1.4])
    def test_matches_lstsq(self, graph, kappa):
        A = {"hetero": HETERO, "tree": random_tree(10, seed=1), "er": erdos_renyi(15, 0.3, seed=2), "star": star_graph(6)}[graph]
        X, _, _ = simulate(A, 0.7, 300, seed=3)
        fit = fit_gnar1(A, X, kappa=kappa, demean=False)
        D, y = design_matrix(X, A, kappa)
        coef, *_ = np.linalg.lstsq(D, y, rcond=None)
        rss = np.sum((y - D @ coef) ** 2)
        np.testing.assert_allclose(fit.alpha, coef[:-1], rtol=1e-8, atol=1e-12)
        assert fit.beta == pytest.approx(coef[-1], rel=1e-8)
        assert fit.rss == pytest.approx(rss, rel=1e-8)

    def test_profile_is_rss_of_fixed_fits(self):
        X, _, _ = simulate(HETERO, 0.5, 200, seed=4)
        M, degrees = node_sums(X, HETERO), node_degrees(HETERO)
        kappas = np.array([0.0, 0.4, 1.0, 1.3])
        expected = [fit_gnar1(HETERO, X, kappa=k, demean=False).rss for k in kappas]
        np.testing.assert_allclose(rss_profile(M, degrees, kappas), expected, rtol=1e-12)

    def test_rss_derivative(self):
        X, _, _ = simulate(HETERO, 0.5, 200, seed=4)
        M, degrees = node_sums(X, HETERO), node_degrees(HETERO)
        for kappa in [0.1, 0.6, 1.3]:
            h = 1e-6
            fd = (rss_profile(M, degrees, [kappa + h])[0] - rss_profile(M, degrees, [kappa - h])[0]) / (2 * h)
            assert rss_derivative(M, degrees, kappa) == pytest.approx(fd, rel=1e-5)
        # The estimate is a root of the derivative
        fit = fit_gnar1(HETERO, X, demean=False)
        assert abs(rss_derivative(M, degrees, fit.kappa)) < 1e-9 * fit.rss

    def test_node_sums(self):
        X = np.random.default_rng(0).standard_normal((50, 6))
        M = node_sums(X, HETERO)
        S = X @ HETERO
        np.testing.assert_allclose(M.xs, np.sum(X[:-1] * S[:-1], axis=0), rtol=1e-14)
        np.testing.assert_allclose(M.sy, np.sum(S[:-1] * X[1:], axis=0), rtol=1e-14)
        assert M.n == 50


class TestProfileOptimum:
    """Acceptance test 4: the profile optimum agrees with joint NLS by scipy.optimize.least_squares over all N + 2 parameters."""

    @pytest.mark.parametrize("graph, kappa", [("tree", 0.7), ("er", 1.2), ("hetero", 0.4)])
    def test_matches_joint_nls(self, graph, kappa):
        A = {"tree": random_tree(10, seed=5), "er": erdos_renyi(20, 0.3, seed=6), "hetero": HETERO}[graph]
        X, _, _ = simulate(A, kappa, 2000, seed=7)
        fit = fit_gnar1(A, X, demean=False)
        d = A.shape[0]

        def residuals(theta):
            D, y = design_matrix(X, A, theta[-1])
            return y - D @ theta[:-1]

        # Start away from the profile solution: kappa at the middle of the bounds and the OLS fit there
        start = fit_gnar1(A, X, kappa=0.75, demean=False)
        theta0 = np.concatenate([start.alpha, [start.beta, 0.75]])
        nls = least_squares(residuals, theta0, jac="3-point", x_scale="jac", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
        rss_nls = np.sum(nls.fun ** 2)
        assert abs(nls.x[-1] - fit.kappa) < 1e-5
        assert rss_nls == pytest.approx(fit.rss, rel=1e-8)
        np.testing.assert_allclose(nls.x[:d], fit.alpha, atol=1e-5)
        assert nls.x[d] == pytest.approx(fit.beta, abs=1e-5)


class TestRegularGraph:
    """Acceptance test 10: on a regular graph (a cycle) RSS(kappa) is constant and an identifiability warning is raised."""

    def test_cycle(self):
        A = cycle_graph(8)
        X = simulate_gnar1(A, 0.2, 0.3, 1.0, 500, rng=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_gnar1(A, X, demean=False)
        rss = fit.profile["rss"]
        assert np.ptp(rss) <= 1e-10 * np.min(rss)
        categories = [w.category for w in caught if issubclass(w.category, KappaWarning)]
        # Only the identifiability warning: the flat profile must not also trigger the boundary or multimodal warnings
        assert categories == [KappaIdentifiabilityWarning]
        assert [code for code, _ in fit.warnings] == ["identifiability"]
        assert np.isnan(fit.kappa) and np.isnan(fit.beta)
        # alpha and RSS are identified and do not depend on kappa
        fixed = fit_gnar1(A, X, kappa=0.3, demean=False)
        np.testing.assert_allclose(fit.alpha, fixed.alpha, rtol=1e-10)
        assert fit.rss == pytest.approx(fixed.rss, rel=1e-10)
        assert np.isinf(fit._cov_beta_kappa[1, 1])

    def test_all_degree_one(self):
        # A perfect matching: every node has degree 1, so N_i^(-kappa) = 1 and beta is identified but kappa is not
        A = np.kron(np.eye(3), np.array([[0.0, 1.0], [1.0, 0.0]]))
        X = simulate_gnar1(A, 0.2, 0.3, 0.5, 400, rng=3)
        with pytest.warns(KappaIdentifiabilityWarning):
            fit = fit_gnar1(A, X, demean=False)
        assert np.isnan(fit.kappa)
        assert fit.beta == pytest.approx(fit_gnar1(A, X, kappa=1.0, demean=False).beta, rel=1e-12)
        assert np.isfinite(fit._cov_beta_kappa[0, 0]) and np.isinf(fit._cov_beta_kappa[1, 1])

    def test_no_edges(self):
        X = np.random.default_rng(0).standard_normal((100, 4))
        with pytest.warns(KappaIdentifiabilityWarning):
            fit = fit_gnar1(np.zeros((4, 4)), X, demean=False)
        assert np.isnan(fit.kappa) and fit.beta == 0.0
        with pytest.warns(KappaIdentifiabilityWarning):
            fixed = fit_gnar1(np.zeros((4, 4)), X, kappa=1.0, demean=False)
        assert fixed.beta == 0.0


class TestIsolatedNode:
    """Acceptance test 12: with an isolated node the fit succeeds, the node is excluded from beta and kappa, and its alpha is its own AR(1) OLS."""

    def test_isolated_node(self):
        core = random_tree(8, seed=9)
        A = with_isolated_node(core)
        X, _, _ = simulate(A, 0.6, 1500, seed=10)
        fit = fit_gnar1(A, X, demean=False)
        fit_core = fit_gnar1(core, X[:, :-1], demean=False)
        # beta and kappa only use the connected nodes
        assert fit.kappa == pytest.approx(fit_core.kappa, abs=1e-8)
        assert fit.beta == pytest.approx(fit_core.beta, rel=1e-8)
        np.testing.assert_allclose(fit.alpha[:-1], fit_core.alpha, rtol=1e-7)
        # The isolated node's alpha is the AR(1) least-squares estimate of its own series
        x = X[:, -1]
        ar1 = np.linalg.lstsq(x[:-1, None], x[1:], rcond=None)[0][0]
        assert fit.alpha[-1] == pytest.approx(ar1, rel=1e-12)
        assert np.all(np.isfinite(fit._cov_beta_kappa))


class TestBoundary:
    """Acceptance test 13: true kappa = 1 with kappa_bounds = (0, 0.5) raises the boundary warning."""

    def test_boundary_warning(self):
        A = random_tree(10, seed=11)
        X, _, _ = simulate(A, 1.0, 2000, seed=12)
        with pytest.warns(KappaBoundaryWarning):
            fit = fit_gnar1(A, X, kappa_bounds=(0, 0.5), demean=False)
        # The optimum is on the bound, and is reported exactly there
        assert fit.kappa == 0.5
        assert fit.rss == pytest.approx(fit_gnar1(A, X, kappa=0.5, demean=False).rss, rel=1e-14)

    def test_lower_boundary(self):
        A = random_tree(10, seed=11)
        X, _, _ = simulate(A, 0.2, 2000, seed=13)
        with pytest.warns(KappaBoundaryWarning):
            fit = fit_gnar1(A, X, kappa_bounds=(0.8, 1.5), demean=False)
        assert fit.kappa == 0.8


class TestWarnings:

    def test_no_warnings_when_well_identified(self):
        A = erdos_renyi(20, 0.3, seed=6)
        X, _, _ = simulate(A, 0.7, 5000, seed=14)
        with warnings.catch_warnings():
            warnings.simplefilter("error", KappaWarning)
            fit = fit_gnar1(A, X, demean=False)
        assert fit.warnings == []

    def test_flat_profile(self):
        # A tiny beta and a short series leave kappa weakly identified
        A = random_tree(10, seed=15)
        rng = np.random.default_rng(16)
        X = simulate_gnar1(A, rng.uniform(0.1, 0.4, 10), 0.005, 0.7, 60, rng=rng)
        with pytest.warns(KappaFlatProfileWarning):
            fit = fit_gnar1(A, X, demean=False)
        assert "flat" in [code for code, _ in fit.warnings]

    def test_count_local_minima(self):
        assert _count_local_minima(np.array([3.0, 2.0, 1.0, 2.0, 3.0]), 0.0) == 1
        assert _count_local_minima(np.array([3.0, 1.0, 2.0, 1.0, 3.0]), 0.0) == 2
        # Ends count as minima; ties within the tolerance are ignored
        assert _count_local_minima(np.array([1.0, 2.0, 3.0]), 0.0) == 1
        assert _count_local_minima(np.array([2.0, 1.0, 1.0 + 1e-15, 1.0, 2.0]), 1e-12) == 1
        assert _count_local_minima(np.ones(5), 0.0) == 0


class TestLikelihood:

    def test_loglik_and_deviance(self):
        X, _, _ = simulate(HETERO, 0.5, 300, seed=17)
        fit = fit_gnar1(HETERO, X, demean=False)
        n = fit.n_obs
        assert n == 6 * 299
        # With sigma^2 profiled out: the Gaussian log-likelihood at sigma^2 = RSS / n
        s2 = fit.rss / n
        assert fit.loglik == pytest.approx(-n / 2 * np.log(2 * np.pi * s2) - fit.rss / (2 * s2), rel=1e-12)
        np.testing.assert_allclose(fit.profile["deviance"], n * np.log(fit.profile["rss"] / fit.rss), rtol=1e-9, atol=1e-9)
        assert np.min(fit.profile["deviance"]) >= 0
        assert fit.profile["loglik"].max() <= fit.loglik + 1e-9
        # sigma^2 = RSS / (n - N - 2) with kappa estimated, RSS / (n - N - 1) with kappa fixed
        assert fit.sigma_2 == pytest.approx(fit.rss / (n - 8), rel=1e-14)
        fixed = fit_gnar1(HETERO, X, kappa=0.5, demean=False)
        assert fixed.sigma_2 == pytest.approx(fixed.rss / (n - 7), rel=1e-14)

    def test_fixed_sigma(self):
        X, _, _ = simulate(HETERO, 0.5, 300, seed=17)
        fit = fit_gnar1(HETERO, X, sigma_2=1.0, demean=False)
        assert fit.sigma_2 == 1.0 and fit.sigma_2_fixed
        np.testing.assert_allclose(fit.profile["deviance"], fit.profile["rss"] - fit.rss, rtol=1e-9, atol=1e-9)
        assert fit.loglik == pytest.approx(-fit.rss / 2 - fit.n_obs / 2 * np.log(2 * np.pi), rel=1e-12)
        assert deviance(12.0, 10.0, 100, sigma_2=4.0) == pytest.approx(0.5)
        assert profile_loglik(10.0, 5) == pytest.approx(-2.5 * np.log(2.0) - 2.5 * (1 + np.log(2 * np.pi)))


class TestBetaKappaCov:

    def test_closed_form_matches_inverse_information(self):
        X, _, _ = simulate(HETERO, 0.6, 500, seed=18)
        fit = fit_gnar1(HETERO, X, demean=False)
        degrees = node_degrees(HETERO)
        _, log_N, w = degree_terms(degrees, fit.kappa)
        R = residualised_sums(node_sums(X, HETERO))
        # F~ = sum_i w_i u_i u_i^T with w_i = N_i^(-2 kappa) d_i and u_i = (1, -beta log N_i)
        wi = w * w * R.d
        u = np.vstack([np.ones(6), -fit.beta * log_N])
        F = (u * wi) @ u.T
        np.testing.assert_allclose(beta_kappa_cov(R, degrees, fit.beta, fit.kappa, fit.sigma_2), fit.sigma_2 * np.linalg.inv(F), rtol=1e-10)

    def test_fixed_kappa(self):
        X, _, _ = simulate(HETERO, 0.6, 500, seed=18)
        fixed = fit_gnar1(HETERO, X, kappa=0.6, demean=False)
        D, _ = design_matrix(X, HETERO, 0.6)
        ols = fixed.sigma_2 * np.linalg.inv(D.T @ D)
        assert fixed._cov_beta_kappa[0, 0] == pytest.approx(ols[-1, -1], rel=1e-10)
        assert np.isnan(fixed._cov_beta_kappa[1, 1])


class TestLegacyCoefficients:
    """Acceptance test 1(c), coefficients: fit_gnar1 with kappa fixed at 1 reproduces the legacy standard GNAR(1, [1]) OLS fit."""

    _ref = np.load(pathlib.Path(__file__).parent / "data" / "legacy_reference.npz")
    _meta = json.loads(str(_ref["meta"]))
    CASES = [i for i, spec in enumerate(_meta["specs"])
             if spec.get("method") == "OLS" and spec["model_type"] == "standard" and spec["p"] == 1 and spec["s"] == [1]
             and spec["net_type"] == "unweighted" and spec["graph"] != "path2" or
             (spec.get("method") == "OLS" and spec["model_type"] == "standard" and spec["p"] == 1 and spec["graph"] == "path2")]

    @pytest.mark.parametrize("i", CASES)
    def test_matches_frozen_legacy(self, i):
        spec = self._meta["specs"][i]
        inputs, expected = load_case(self._ref, i)
        fit = fit_gnar1(inputs["A"], inputs["ts"], kappa=1.0, demean=spec["demean"])
        coeffs = expected["coeffs"]
        # Legacy pygnar solves the stacked least-squares problem iteratively (lsqr); on these small problems it converges
        # to machine precision, so the closed form agrees to 1e-10
        np.testing.assert_allclose(fit.alpha, coeffs[0], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(fit.beta, coeffs[1], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(fit.mu, expected["mu"], rtol=1e-12, atol=1e-14)

    def test_matches_exact_lstsq(self):
        # On larger graphs lsqr stops at its 1e-6 tolerance, so the closed form is compared with the exact solution instead
        A = erdos_renyi(30, 0.2, seed=19)
        X, _, _ = simulate(A, 1.0, 400, seed=20)
        fit = fit_gnar1(A, X, kappa=1.0, demean=True)
        D, y = design_matrix(X - X.mean(axis=0), A, 1.0)
        coef = np.linalg.lstsq(D, y, rcond=None)[0]
        np.testing.assert_allclose(np.append(fit.alpha, fit.beta), coef, rtol=1e-10, atol=1e-12)

    def test_constant_node_matches_legacy(self):
        # A constant series is all zeros after demeaning; legacy pygnar then sets its alpha to 0
        rng = np.random.default_rng(21)
        A = path_graph(4)
        X = rng.standard_normal((80, 4))
        X[:, 2] = 3.0
        legacy = GNAR(A, p=1, s=np.array([1]), ts=X)
        fit = fit_gnar1(A, X, kappa=1.0, demean=True)
        np.testing.assert_allclose(fit.alpha, legacy.coeffs[0], rtol=1e-10, atol=1e-12)
        assert fit.beta == pytest.approx(legacy.coeffs[1, 0], rel=1e-10)


class TestFitInterface:

    def test_demean_is_required(self):
        with pytest.raises(TypeError):
            fit_gnar1(path_graph(3), np.zeros((10, 3)))

    def test_demean_and_names(self):
        X, _, _ = simulate(HETERO, 0.5, 200, seed=22)
        df = pd.DataFrame(X + np.arange(6), columns=list("abcdef"))
        fit = fit_gnar1(HETERO, df, demean=True)
        np.testing.assert_allclose(fit.mu, df.to_numpy().mean(axis=0, keepdims=True))
        assert list(fit.names) == list("abcdef")
        centred = fit_gnar1(HETERO, X - X.mean(axis=0), demean=False)
        # The estimate is polished on the analytic derivative, so rounding in the data barely moves it
        assert fit.kappa == pytest.approx(centred.kappa, abs=1e-11)
        assert "alpha" in str(fit) and "kappa" in repr(fit)

    def test_sparse_adjacency(self):
        from scipy.sparse import csr_matrix
        X, _, _ = simulate(HETERO, 0.5, 200, seed=23)
        dense, sparse = fit_gnar1(HETERO, X, demean=False), fit_gnar1(csr_matrix(HETERO), X, demean=False)
        assert dense.kappa == pytest.approx(sparse.kappa, abs=1e-10)
        np.testing.assert_allclose(dense.alpha, sparse.alpha, rtol=1e-10)

    @pytest.mark.parametrize("kwargs, error, match", [
        (dict(A=2 * path_graph(3)), NotImplementedError, "Weighted"),
        (dict(A=np.triu(path_graph(3))), ValueError, "symmetric"),
        (dict(ts=np.zeros((10, 4))), ValueError, "number of time series"),
        (dict(ts=np.full((10, 3), np.nan)), ValueError, "non-finite"),
        (dict(kappa_bounds=(1.0, 0.5)), ValueError, "kappa_bounds"),
        (dict(grid=2), ValueError, "grid"),
        (dict(sigma_2=-1.0), ValueError, "sigma_2"),
    ])
    def test_validation(self, kwargs, error, match):
        args = dict(A=path_graph(3), ts=np.random.default_rng(0).standard_normal((10, 3)))
        args.update(kwargs)
        with pytest.raises(error, match=match):
            quiet_fit(args.pop("A"), args.pop("ts"), demean=False, **args)


class TestGNARIntegration:

    def test_estimated_kappa_in_gnar(self):
        A = random_tree(10, seed=24)
        X, _, _ = simulate(A, 0.7, 1000, seed=25)
        G = GNAR(A, p=1, s=np.array([1]), ts=X, kappa=None, demean=False)
        fit = fit_gnar1(A, X, demean=False)
        assert G.kappa == pytest.approx(fit.kappa, abs=1e-12)
        np.testing.assert_allclose(G.coeffs[0], fit.alpha, rtol=1e-12)
        np.testing.assert_allclose(G.coeffs[1], fit.beta, rtol=1e-12)
        assert G.kappa_fit.kappa == G.kappa
        assert G._num_params() == 10 + 2
        # The neighbour set matrices are rebuilt at kappa_hat, so the one-step forecasts use the fitted model
        pred = G.predict(ts=X[-1:], h=1)[0]
        _, _, w = degree_terms(node_degrees(A), G.kappa)
        np.testing.assert_allclose(pred, fit.alpha * X[-1] + fit.beta * w * (X[-1] @ A), rtol=1e-12)
        assert "(estimated)" in str(G) and np.isfinite(G.bic())

    def test_refit_reestimates(self):
        A = random_tree(10, seed=24)
        X1, _, _ = simulate(A, 0.3, 1000, seed=26)
        X2, _, _ = simulate(A, 1.2, 1000, seed=27)
        G = GNAR(A, p=1, s=np.array([1]), ts=X1, kappa=None)
        first = G.kappa
        G.fit(X2)
        assert G.kappa != first
        assert G.kappa == pytest.approx(fit_gnar1(A, X2, demean=True).kappa, abs=1e-12)

    def test_unidentified_uses_kappa_one(self):
        A = cycle_graph(6)
        X = simulate_gnar1(A, 0.2, 0.3, 0.5, 300, rng=28)
        with pytest.warns(KappaIdentifiabilityWarning):
            G = GNAR(A, p=1, s=np.array([1]), ts=X, kappa=None)
        assert G.kappa == 1.0 and np.isnan(G.kappa_fit.kappa)
        exact = fit_gnar1(A, X, kappa=1.0, demean=True)
        np.testing.assert_allclose(G.coeffs, np.vstack([exact.alpha, np.full(6, exact.beta)]), rtol=1e-12)
        # Legacy pygnar's iterative lsqr stops at its 1e-6 tolerance here, so the comparison with it is looser
        legacy = GNAR(A, p=1, s=np.array([1]), ts=X)
        np.testing.assert_allclose(G.coeffs, legacy.coeffs, rtol=1e-5)

    def test_unsupported(self):
        X = np.random.default_rng(0).standard_normal((50, 3))
        with pytest.raises(NotImplementedError, match="OLS"):
            GNAR(path_graph(3), p=1, s=np.array([1]), ts=X, kappa=None, method="YW")
        with pytest.raises(NotImplementedError, match="standard"):
            GNAR(path_graph(3), p=1, s=np.array([1]), ts=X, kappa=None, model_type="global")


class TestPlotProfile:

    def test_plot(self):
        A = random_tree(10, seed=29)
        X, _, _ = simulate(A, 0.7, 1000, seed=30)
        fit = fit_gnar1(A, X, demean=False)
        ax = plot_profile(fit)
        labels = [line.get_label() for line in ax.get_lines()]
        assert any("Profile" in l for l in labels) and any("Wald" in l for l in labels)
        # The Wald curve is the quadratic (kappa - kappa_hat)^2 / SE^2
        wald = next(line for line in ax.get_lines() if "Wald" in line.get_label())
        se = np.sqrt(fit._cov_beta_kappa[1, 1])
        np.testing.assert_allclose(wald.get_ydata(), (fit.profile["kappa"] - fit.kappa) ** 2 / se ** 2)
        cutoff = next(line for line in ax.get_lines() if "cutoff" in line.get_label())
        assert cutoff.get_ydata()[0] == pytest.approx(chi2.ppf(0.95, 1))
        ax2 = plot_profile(fit, level=0.9, show_quadratic=False)
        assert not any("Wald" in line.get_label() for line in ax2.get_lines())
        matplotlib.pyplot.close("all")

    def test_fixed_kappa_has_no_profile(self):
        X = np.random.default_rng(0).standard_normal((50, 3))
        with pytest.raises(ValueError, match="fixed"):
            plot_profile(fit_gnar1(path_graph(3), X, kappa=1.0, demean=False))


class TestInformativeNodes:
    """kappa is identified by the degrees of the nodes that carry information on the network term, not by the graph alone."""

    def test_dead_neighbours(self):
        # Node 0 (degree 2) has two constant neighbours, so after demeaning its neighbour sum is 0 and it carries no
        # information; the informative nodes all have degree 1, so RSS(kappa) is constant and only beta is identified
        A = np.zeros((5, 5))
        for i, j in [(0, 1), (0, 2), (3, 4)]:
            A[i, j] = A[j, i] = 1
        X = np.random.default_rng(0).standard_normal((400, 5))
        X[:, 1], X[:, 2] = 2.0, -1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_gnar1(A, X, demean=True)
        assert [w.category for w in caught if issubclass(w.category, KappaWarning)] == [KappaIdentifiabilityWarning]
        # The message names the common degree of the informative nodes, at which beta_at gives the identified quantity
        assert "degree 1" in str(caught[-1].message)
        assert np.isnan(fit.kappa) and np.isfinite(fit.beta)
        assert fit.beta == pytest.approx(fit_gnar1(A, X, kappa=1.0, demean=True).beta, rel=1e-12)
        # The GNAR route agrees and falls back to kappa = 1 for the weights
        G = quiet_gnar(A, X)
        assert G.kappa == 1.0 and np.isnan(G.kappa_fit.kappa)

    def test_constant_hub(self):
        # A constant hub: beta_hat = 0 for every kappa, RSS(kappa) is constant
        A = star_graph(6)
        X = np.random.default_rng(1).standard_normal((300, 7))
        X[:, 0] = 5.0
        with pytest.warns(KappaIdentifiabilityWarning):
            fit = fit_gnar1(A, X, demean=True)
        assert np.isnan(fit.kappa)

    def test_beta_hat_zero_with_distinct_degrees(self):
        # Informative nodes of degrees 2 and 3 whose own series are constant after demeaning: c_i = 0, so beta_hat = 0
        # for every kappa although the degrees differ
        A = np.zeros((7, 7))
        for i, j in [(0, 2), (0, 3), (1, 4), (1, 5), (1, 6)]:
            A[i, j] = A[j, i] = 1
        X = np.random.default_rng(2).standard_normal((300, 7))
        X[:, 0], X[:, 1] = 1.0, -2.0
        with pytest.warns(KappaIdentifiabilityWarning, match="constant"):
            fit = fit_gnar1(A, X, demean=True)
        assert np.isnan(fit.kappa)

    def test_isolated_node_with_regular_core(self):
        # The isolated node's degree 0 must not make the cycle look heterogeneous
        A = with_isolated_node(cycle_graph(8))
        X = simulate_gnar1(A, 0.2, 0.3, 1.0, 500, rng=2)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_gnar1(A, X, demean=False)
        assert [w.category for w in caught if issubclass(w.category, KappaWarning)] == [KappaIdentifiabilityWarning]
        assert "degree 2" in str(caught[-1].message)
        assert np.isnan(fit.kappa)
        x = X[:, -1]
        assert fit.alpha[-1] == pytest.approx(np.linalg.lstsq(x[:-1, None], x[1:], rcond=None)[0][0], rel=1e-12)


class TestOptimiserEdgeCases:

    def test_polish_keeps_brents_minimum_on_a_bimodal_bracket(self):
        # With grid = 3 the Brent bracket is the whole range; the profile below has minima near 0.283 (global) and 0.987.
        # The derivative polish must refine Brent's minimum, not jump to the other stationary point
        from gnar.utils.gnar_profile_likelihood import NodeSums, estimate_kappa
        degrees = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89], dtype=float)
        c = np.array([0.03197, 6.0784, 0.85671, 3.0472, -0.0017539, -2.1379, 0.033632, 0.37035, 9.7669, -0.35388])
        dd = np.array([0.19293, 6.7033, 6.4925, 13.514, 0.41279, 3.0303, 2.0585, 13.764, 12.180, 6.0970])
        M0 = NodeSums(xx=np.ones(10), xy=np.zeros(10), xs=np.zeros(10), ss=dd, sy=c, yy=np.ones(10), n=1000)
        kappas = np.linspace(0, 1.5, 3001)
        explained = np.max([(np.exp(-k * np.log(degrees)) @ c) ** 2 / (np.exp(-2 * k * np.log(degrees)) @ dd) for k in kappas])
        M = M0._replace(yy=np.full(10, 1.2 * explained / 10))
        coarse = estimate_kappa(M, degrees, (0.0, 1.5), grid=3)
        fine = estimate_kappa(M, degrees, (0.0, 1.5), grid=201)
        assert coarse.kappa == pytest.approx(0.283174, abs=1e-5)
        assert coarse.kappa == pytest.approx(fine.kappa, abs=1e-7)
        assert "multimodal" in [code for code, _ in fine.warnings]

    def test_near_bound_behaviour(self):
        # The boundary warning fires within 1e-3 of a bound, and not when the interior optimum is 2e-3 away
        A = random_tree(10, seed=11)
        X, _, _ = simulate(A, 1.0, 2000, seed=12)
        k = fit_gnar1(A, X, demean=False).kappa
        with warnings.catch_warnings():
            warnings.simplefilter("error", KappaWarning)
            clear = fit_gnar1(A, X, kappa_bounds=(0.0, k + 0.002), demean=False)
        assert clear.kappa == pytest.approx(k, abs=1e-9)
        with pytest.warns(KappaBoundaryWarning):
            close = fit_gnar1(A, X, kappa_bounds=(0.0, k + 0.0005), demean=False)
        assert close.kappa == pytest.approx(k, abs=1e-9)

    def test_boundary_case_warns_only_about_the_boundary(self):
        # Acceptance test 13 setting: exactly the boundary warning, no spurious flat-profile or multimodal warning
        A = random_tree(10, seed=11)
        X, _, _ = simulate(A, 1.0, 2000, seed=12)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit_gnar1(A, X, kappa_bounds=(0, 0.5), demean=False)
        assert [w.category for w in caught if issubclass(w.category, KappaWarning)] == [KappaBoundaryWarning]

    def test_flat_profile_also_reports_multimodality(self):
        A = random_tree(10, seed=15)
        rng = np.random.default_rng(16)
        X = simulate_gnar1(A, rng.uniform(0.1, 0.4, 10), 0.005, 0.7, 60, rng=rng)
        fit = quiet_fit(A, X, demean=False)
        codes = [code for code, _ in fit.warnings]
        assert "flat" in codes and "multimodal" in codes


class TestWarningLocation:
    """Warnings point at the user's call, so Python's once-per-location filter does not hide them across fits."""

    def test_direct_and_gnar_calls(self):
        A = cycle_graph(6)
        X = simulate_gnar1(A, 0.2, 0.3, 0.5, 300, rng=28)
        for call in (lambda: fit_gnar1(A, X, demean=False), lambda: GNAR(A, p=1, s=np.array([1]), ts=X, kappa=None)):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                call()
            assert caught and all(w.filename == __file__ for w in caught if issubclass(w.category, KappaWarning))
        G = quiet_gnar(A, X)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            G.fit(X)
        assert caught and all(w.filename == __file__ for w in caught if issubclass(w.category, KappaWarning))


def quiet_gnar(A, X, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", KappaWarning)
        return GNAR(A, p=1, s=np.array([1]), ts=X, kappa=None, **kwargs)


class TestGNARKappaState:

    def test_failed_refit_leaves_model_unchanged(self):
        A = random_tree(10, seed=24)
        X, _, _ = simulate(A, 0.7, 1000, seed=25)
        G = GNAR(A, p=1, s=np.array([1]), ts=X, kappa=None)
        before = (G._n, G.mu.copy(), G.coeffs.copy(), G.kappa, G.predict(h=1).copy())
        with pytest.raises(NotImplementedError):
            G.fit(X[:500] + 10.0, method="YW")
        bad = X[:100] + 50.0
        bad[3, 2] = np.nan
        with pytest.raises(ValueError):
            G.fit(bad)
        after = (G._n, G.mu, G.coeffs, G.kappa, G.predict(h=1))
        assert after[0] == before[0] and after[3] == before[3]
        for a, b in zip(before[1:3] + before[4:], after[1:3] + after[4:]):
            np.testing.assert_array_equal(a, b)
        assert np.array_equal(G.kappa_fit.mu, G.mu)

    def test_unidentified_kappa_in_gnar(self):
        A = cycle_graph(6)
        X = simulate_gnar1(A, 0.2, 0.3, 0.5, 300, rng=28)
        G = quiet_gnar(A, X)
        fixed = GNAR(A, p=1, s=np.array([1]), ts=X, kappa=1.0)
        # kappa is not counted: the fit is the kappa = 1 fit
        assert G._num_params() == fixed._num_params() == 7
        assert G.bic() == pytest.approx(fixed.bic(), rel=1e-6)
        assert "kappa: not identified" in str(G) and "kappa=nan" in repr(G)
        assert G.kappa_fit.num_params == 7

    def test_kappa_bounds_sigma_2_and_mean(self):
        A = random_tree(10, seed=11)
        X, _, _ = simulate(A, 1.0, 2000, seed=12)
        with pytest.warns(KappaBoundaryWarning):
            G = GNAR(A, p=1, s=np.array([1]), ts=X + 3.0, kappa=None, kappa_bounds=(0, 0.5))
        assert G.kappa == 0.5
        # The mean is removed once, and the fit object reports it
        np.testing.assert_allclose(G.mu, (X + 3.0).mean(axis=0, keepdims=True), rtol=1e-14)
        np.testing.assert_array_equal(G.kappa_fit.mu, G.mu)
        # sigma_2 is the legacy residual covariance R^T R / (T - 2) at kappa_hat
        fit = G.kappa_fit
        x = X + 3.0 - G.mu
        _, _, w = degree_terms(node_degrees(A), G.kappa)
        R = x[1:] - fit.alpha * x[:-1] - fit.beta * w * (x[:-1] @ A)
        np.testing.assert_allclose(G.sigma_2, R.T @ R / (len(X) - 2), rtol=1e-10)
        with pytest.raises(ValueError, match="kappa_bounds"):
            GNAR(A, p=1, s=np.array([1]), ts=X, kappa=None, kappa_bounds=(1.0, 0.5))


class TestFitValidation:

    @pytest.mark.parametrize("kwargs, match", [
        (dict(sigma_2=0.0), "sigma_2"),
        (dict(sigma_2=True), "sigma_2"),
        (dict(sigma_2="1"), "sigma_2"),
        (dict(sigma_2=np.ones(3)), "sigma_2"),
        (dict(kappa_bounds=(0, 1, 2)), "kappa_bounds"),
        (dict(kappa_bounds=("0", "1")), "kappa_bounds"),
        (dict(kappa=0.5, kappa_bounds=(3, -1)), "kappa_bounds"),
        (dict(kappa=0.5, grid=1), "grid"),
        (dict(demean=None), "demean"),
        (dict(demean="no"), "demean"),
    ])
    def test_invalid_arguments(self, kwargs, match):
        args = dict(demean=False)
        args.update(kwargs)
        X = np.random.default_rng(0).standard_normal((20, 3))
        with pytest.raises(ValueError, match=match):
            quiet_fit(path_graph(3), X, **args)

    def test_minimum_observations(self):
        # With kappa fixed there are d + 1 parameters: a 2-node graph with T = 3 has n = 4 observations and one residual
        # degree of freedom
        X = np.random.default_rng(0).standard_normal((3, 2))
        fit = fit_gnar1(path_graph(2), X, kappa=1.0, demean=False)
        assert np.isfinite(fit.sigma_2)
        # With kappa estimated there are d + 2 = 4 parameters for the same 4 observations
        with pytest.raises(ValueError, match="Too few observations"):
            fit_gnar1(path_graph(2), X, demean=False)


class TestPlotProfileDetails:

    def test_axes_lines_and_limits(self):
        import matplotlib.pyplot as plt
        A = random_tree(10, seed=29)
        X, _, _ = simulate(A, 0.7, 1000, seed=30)
        fit = fit_gnar1(A, X, demean=False)
        plt.close("all")
        fig, ax = plt.subplots()
        assert plot_profile(fit, ax=ax, level=0.9) is ax
        assert plt.get_fignums() == [fig.number]
        lines = {line.get_label().split(" ")[0]: line for line in ax.get_lines()}
        np.testing.assert_allclose(lines["κ̂"].get_xdata(), [fit.kappa, fit.kappa])
        cutoff = chi2.ppf(0.9, 1)
        assert lines["χ²₁"].get_ydata()[0] == pytest.approx(cutoff)
        dev = fit.profile["deviance"]
        assert ax.get_ylim() == pytest.approx((0, max(1.5 * cutoff, min(dev.max(), 5 * cutoff)) * 1.05))
        plt.close("all")
