import warnings

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from gnar import GNAR, KappaGNAR
from gnar.kappa import _spectral_info, _symmetrised_weight_matrix, _as_csr
from gnar.utils.graphs import powerlaw_configuration_graph

matplotlib.use("Agg")


def _graph(N, seed=0, sparse=True):
    G = powerlaw_configuration_graph(N, seed=seed)
    if sparse:
        return nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), format="csr")
    return nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))


A_300 = _graph(300)
A_RING = np.zeros((8, 8))
for _i in range(8):
    A_RING[_i, (_i + 1) % 8] = A_RING[(_i + 1) % 8, _i] = 1


def _simulate(A, alpha=0.3, b=0.6, kappa=0.5, T=200, seed=0, mean=0.0, sigma_2=1.0):
    np.random.seed(seed)
    return KappaGNAR(A, alpha=alpha, b=b, kappa=kappa, mean=mean, sigma_2=sigma_2).simulate(T)


def _fit(A, ts, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return KappaGNAR(A, ts=ts, **kwargs)


class TestProfileShortcut:

    @pytest.mark.parametrize("kappa", [0.0, 0.25, 0.6, 1.0])
    def test_matches_brute_force_regression(self, kappa):
        ts = _simulate(A_300, kappa=0.5, T=80)
        F = _fit(A_300, ts, kappa=kappa)
        Y = ts - ts.mean(axis=0)
        A = A_300.toarray()
        Z = (Y[:-1] @ A.T) * A.sum(axis=1) ** -kappa          # d_i^{-kappa} sum_j A_ij y_{j,t-1}
        X = np.column_stack([Y[:-1].ravel(), Z.ravel()])
        coef, res, *_ = np.linalg.lstsq(X, Y[1:].ravel(), rcond=None)
        assert F.alpha == pytest.approx(coef[0], rel=1e-9)
        assert F.beta == pytest.approx(coef[1], rel=1e-9)
        assert F.sigma_2 == pytest.approx(res[0] / Y[1:].size, rel=1e-9)

    def test_kappa_one_matches_global_gnar(self):
        A = A_300.toarray()
        ts = _simulate(A_300, kappa=1.0, T=120)
        F = _fit(A_300, ts, kappa=1.0)
        G = GNAR(A, p=1, s=np.array([1]), ts=ts, model_type="global")
        assert F.alpha == pytest.approx(G.coeffs[0, 0], rel=1e-8)
        assert F.b == pytest.approx(G.coeffs[1, 0], rel=1e-8)
        assert F.sigma_2 == pytest.approx(np.trace(G.sigma_2) / A.shape[0], rel=1e-8)

    def test_profile_maximum(self):
        F = _fit(A_300, _simulate(A_300, kappa=0.4, T=100))
        prof = F.profile_
        assert F.loglik() >= prof["loglik"].max() - 1e-9
        assert prof.shape == (201, 4)
        assert list(prof.columns) == ["kappa", "loglik", "alpha", "beta"]


class TestEstimation:

    @pytest.mark.parametrize("kappa", [0.2, 0.5, 0.8])
    def test_recovery(self, kappa):
        ts = _simulate(A_300, alpha=0.3, b=0.6, kappa=kappa, T=1000, seed=1)
        F = _fit(A_300, ts, demean=False)
        assert F.kappa == pytest.approx(kappa, abs=0.05)
        assert F.alpha == pytest.approx(0.3, abs=0.01)
        assert F.b == pytest.approx(0.6, abs=0.02)
        assert F.sigma_2 == pytest.approx(1.0, abs=0.02)

    def test_mean_handling(self):
        ts = _simulate(A_300, T=300, mean=np.linspace(-2, 2, A_300.shape[0]))
        F = _fit(A_300, ts)
        np.testing.assert_allclose(F.mu.ravel(), ts.mean(axis=0))
        F0 = _fit(A_300, ts - ts.mean(axis=0), demean=False)
        np.testing.assert_allclose(F0.mu, 0)
        assert F0.alpha == pytest.approx(F.alpha) and F0.kappa == pytest.approx(F.kappa)

    def test_boundary_estimates_are_exact(self):
        # With true kappa = 1, about half of the estimates should land exactly on the boundary
        at_boundary = 0
        for seed in range(20):
            F = _fit(A_300, _simulate(A_300, kappa=1.0, T=60, seed=seed), demean=False)
            at_boundary += F.kappa == 1.0
            assert 0 <= F.kappa <= 1
        assert 3 <= at_boundary <= 17

    def test_boundary_zero(self):
        F = _fit(A_300, _simulate(A_300, kappa=0.0, T=2000, seed=2), demean=False)
        assert F.kappa == pytest.approx(0.0, abs=0.02)

    def test_fixed_kappa(self):
        ts = _simulate(A_300, T=100)
        F = _fit(A_300, ts, kappa=0.3)
        assert F.kappa == 0.3
        assert np.isnan(F.summary().loc["kappa", "std_error"])
        with pytest.raises(ValueError, match="fixed"):
            F.kappa_confint()

    def test_refit_resets_state(self):
        F = _fit(A_300, _simulate(A_300, kappa=0.2, T=200, seed=3))
        F.compute_autocov_mats(0)
        assert F._eig_cache is not None
        old = F.kappa
        F.fit(_simulate(A_300, kappa=0.9, T=200, seed=4))
        assert F._eig_cache is None and F._phi is None
        assert F.kappa != old

    def test_dataframe_names_and_predict(self):
        N = A_300.shape[0]
        ts = pd.DataFrame(_simulate(A_300, T=50), columns=[f"n{i}" for i in range(N)])
        F = _fit(A_300, ts)
        assert list(F.names) == list(ts.columns)
        fc = F.predict(h=2)
        assert isinstance(fc, pd.DataFrame) and list(fc.columns) == list(ts.columns) and fc.shape == (2, N)
        np.testing.assert_allclose(fc.to_numpy(), F.predict(ts.to_numpy()[-1:], h=2))

    def test_predict_from_numpy_fit(self):
        ts = _simulate(A_300, T=50)
        F = _fit(A_300, ts)
        np.testing.assert_allclose(F.predict(h=3), F.predict(ts[-1:], h=3))


class TestInference:

    def test_rho_derivative(self):
        A = _as_csr(A_300)
        F = _fit(A_300, _simulate(A_300, kappa=0.4, T=100))
        eps = 1e-6
        rho = lambda k: _spectral_info(_symmetrised_weight_matrix(A, k))[1]
        numeric = (rho(F.kappa + eps) - rho(F.kappa - eps)) / (2 * eps)
        assert F._rho_derivative() == pytest.approx(numeric, rel=1e-5)

    def test_gram_matches_numerical_jacobian(self):
        ts = _simulate(A_300, kappa=0.5, T=40)
        F = _fit(A_300, ts, demean=False)
        A = A_300.toarray()
        deg = A.sum(axis=1)
        y_lag = ts[:-1]
        S = y_lag @ A.T

        def fitted(theta):
            a, bt, k = theta
            return (a * y_lag + bt * S * deg ** -k).ravel()

        theta = np.array([F.alpha, F.beta, F.kappa])
        J = np.column_stack([(fitted(theta + e) - fitted(theta - e)) / 2e-6 for e in 1e-6 * np.eye(3)])
        np.testing.assert_allclose(F._fit["stats"].gram(F.beta, F.kappa, True), J.T @ J, rtol=1e-5)

    def test_summary_structure(self):
        F = _fit(A_300, _simulate(A_300, kappa=0.5, T=200))
        s = F.summary(level=0.9)
        assert list(s.index) == ["alpha", "b", "beta", "kappa"]
        assert list(s.columns) == ["estimate", "std_error", "ci_lower (90%)", "ci_upper (90%)"]
        assert np.all(s["ci_lower (90%)"] <= s["estimate"]) and np.all(s["estimate"] <= s["ci_upper (90%)"])
        # The b and beta intervals have the same relative width only if kappa is fixed
        assert s.loc["b", "estimate"] == pytest.approx(s.loc["beta", "estimate"] * F.rho)

    def test_kappa_confint_bounds_are_crossings(self):
        F = _fit(A_300, _simulate(A_300, kappa=0.5, T=100))
        lo, hi = F.kappa_confint(0.95)
        assert lo < F.kappa < hi
        cut = chi2.ppf(0.95, 1)
        for bound in (lo, hi):
            assert 2 * (F.loglik() - F._profile_loglik(bound)[0]) == pytest.approx(cut, rel=1e-6)
        lo90, hi90 = F.kappa_confint(0.9)
        assert lo < lo90 and hi90 < hi

    def test_confint_truncated_at_boundary(self):
        F = _fit(A_300, _simulate(A_300, kappa=1.0, T=100, seed=5), demean=False)
        assert F.kappa_confint()[1] == 1.0

    def test_invalid_level(self):
        F = _fit(A_300, _simulate(A_300, T=50))
        with pytest.raises(ValueError, match="level"):
            F.kappa_confint(1.5)

    def test_coverage(self):
        # Monte Carlo coverage of the Wald (alpha, b, kappa) and profile (kappa) 95% intervals, interior kappa
        true = {"alpha": 0.3, "b": 0.6, "kappa": 0.5}
        hits = {"alpha": 0, "b": 0, "kappa_wald": 0, "kappa_profile": 0}
        reps = 150
        for seed in range(reps):
            F = _fit(A_300, _simulate(A_300, T=60, seed=100 + seed, **true), demean=False)
            s = F.summary()
            for p in ("alpha", "b"):
                hits[p] += s.loc[p, "ci_lower (95%)"] <= true[p] <= s.loc[p, "ci_upper (95%)"]
            se = s.loc["kappa", "std_error"]
            hits["kappa_wald"] += abs(F.kappa - true["kappa"]) <= 1.96 * se
            hits["kappa_profile"] += s.loc["kappa", "ci_lower (95%)"] <= true["kappa"] <= s.loc["kappa", "ci_upper (95%)"]
        for name, h in hits.items():
            assert 0.89 <= h / reps <= 0.99, (name, h / reps)


class TestIdentificationAndValidation:

    def test_regular_graph_requires_fixed_kappa(self):
        np.random.seed(0)
        ts = KappaGNAR(A_RING, alpha=0.3, b=0.4, kappa=1.0).simulate(100)
        with pytest.raises(ValueError, match="regular"):
            KappaGNAR(A_RING, ts=ts)
        F = KappaGNAR(A_RING, ts=ts, kappa=1.0)
        assert F.kappa == 1.0

    def test_weak_identification_warns(self):
        np.random.seed(0)
        ts = np.random.normal(size=(30, A_300.shape[0]))       # no network effect at all
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            KappaGNAR(A_300, ts=ts)
        messages = [str(w.message) for w in caught]
        assert any("identified" in m for m in messages), messages

    def test_no_warning_when_well_identified(self):
        ts = _simulate(A_300, kappa=0.5, T=300)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            KappaGNAR(A_300, ts=ts)

    @pytest.mark.parametrize("ts, match", [
        (np.zeros((50, 5)), "shape"),
        (np.zeros((2, 300)), "three time steps"),
        (np.full((50, 300), np.nan), "missing"),
        (np.zeros((50, 300)), "identically zero"),
    ])
    def test_invalid_ts(self, ts, match):
        N = A_300.shape[0]
        if ts.shape[1] == 300:
            ts = np.zeros((ts.shape[0], N)) if not np.isnan(ts).any() else np.full((ts.shape[0], N), np.nan)
        with pytest.raises(ValueError, match=match):
            KappaGNAR(A_300, ts=ts)

    def test_ts_and_parameters_exclusive(self):
        ts = _simulate(A_300, T=20)
        with pytest.raises(ValueError, match="not both"):
            KappaGNAR(A_300, alpha=0.3, b=0.5, ts=ts)
        with pytest.raises(ValueError, match="required"):
            KappaGNAR(A_300)

    def test_kappa_default_when_building_from_parameters(self):
        assert KappaGNAR(A_300, alpha=0.3, b=0.5).kappa == 1.0

    def test_unfitted_methods_raise(self):
        K = KappaGNAR(A_300, alpha=0.3, b=0.5, kappa=0.5)
        for method in (K.summary, K.loglik, K.aic, K.bic, K.kappa_confint, K.plot_profile):
            with pytest.raises(ValueError, match="not fitted"):
                method()
        with pytest.raises(ValueError, match="not fitted"):
            K.predict()
        assert not K.is_fitted


class TestLikelihoodAndPlot:

    def test_loglik_aic_bic(self):
        ts = _simulate(A_300, T=100)
        F = _fit(A_300, ts)
        M = 99 * A_300.shape[0]
        expected = -0.5 * M * (np.log(2 * np.pi * F.sigma_2) + 1)
        assert F.loglik() == pytest.approx(expected)
        assert F.aic() == pytest.approx(-2 * expected + 2 * 4)
        assert F.bic() == pytest.approx(-2 * expected + 4 * np.log(M))
        F_fixed = _fit(A_300, ts, kappa=0.5)
        assert F_fixed.aic() == pytest.approx(-2 * F_fixed.loglik() + 2 * 3)

    def test_plot_profile(self):
        F = _fit(A_300, _simulate(A_300, T=100))
        fig = F.plot_profile(n_points=11)
        assert len(fig.axes) == 2
        F_fixed = _fit(A_300, _simulate(A_300, T=100), kappa=0.5)
        assert len(F_fixed.plot_profile(n_points=5).axes) == 2

    def test_str_fitted(self):
        s = str(_fit(A_300, _simulate(A_300, T=50)))
        assert "kappa estimated" in s and "log-likelihood" in s and "std_error" in s
        assert "fitted=True" in repr(_fit(A_300, _simulate(A_300, T=50)))


class TestScale:

    def test_large_graph_fit_is_fast(self):
        import time
        A = _graph(10_000, seed=1)
        np.random.seed(0)
        ts = KappaGNAR(A, alpha=0.3, b=0.6, kappa=0.4).simulate(200)
        t0 = time.perf_counter()
        F = _fit(A, ts)
        F.summary()
        assert time.perf_counter() - t0 < 5
        assert F.kappa == pytest.approx(0.4, abs=0.03)
