import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gnar import GNAR, KappaGNAR, ModelComparison, NodewiseModel
from gnar.comparison import MODELS
from gnar.utils.graphs import powerlaw_configuration_graph


def _graph(N, seed=0):
    G = powerlaw_configuration_graph(N, seed=seed)
    return nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), format="csr")


A_150 = _graph(150)
DEG_150 = np.asarray(A_150.sum(axis=1)).ravel()


def _simulate_nodewise(A, alpha, beta, T, rng, sigma_2=1.0, burn_in=50):
    """Simulate X_t = diag(alpha) X_{t-1} + diag(beta / d) A X_{t-1} + e_t (beta on the neighbour average)."""
    return NodewiseModel(A, alpha, beta, sigma_2=sigma_2).simulate(T, burn_in=burn_in, rng=rng)


def _simulate_kappa(A, alpha, b, kappa, T, rng):
    K = KappaGNAR(A, alpha=alpha, b=b, kappa=kappa)
    beta_avg = K.beta * np.asarray(A.sum(axis=1)).ravel() ** (1 - kappa)    # coefficient on the neighbour average
    return _simulate_nodewise(A, alpha, beta_avg, T, rng)


def _cmp(A, ts, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return ModelComparison(A, ts, **kwargs)


class TestFits:

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.ts = _simulate_kappa(A_150, 0.3, 0.5, 0.5, 40, rng)
        self.cmp = _cmp(A_150, self.ts)
        self.Y = self.ts - self.ts.mean(axis=0)
        self.M = self.Y[1:].size

    def _rss(self, name):
        return self.cmp._fits[name]["rss"]

    def test_ar(self):
        x, y = self.Y[:-1].ravel(), self.Y[1:].ravel()
        a = x @ y / (x @ x)
        assert self.cmp["AR"].alpha[0] == pytest.approx(a)
        assert self._rss("AR") == pytest.approx(np.sum((y - a * x) ** 2))

    def test_g_and_k_match_kappagnar(self):
        G = KappaGNAR(A_150, ts=self.ts, kappa=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            K = KappaGNAR(A_150, ts=self.ts)
        assert self.cmp.table().loc["G", "loglik"] == pytest.approx(G.loglik())
        assert self.cmp.table().loc["K", "loglik"] == pytest.approx(K.loglik())
        assert self.cmp.table().loc["K", "kappa"] == pytest.approx(K.kappa)
        assert self.cmp["K"].kappa == pytest.approx(K.kappa)

    def test_s_matches_brute_force(self):
        N = A_150.shape[0]
        Z = self.Y[:-1] @ A_150.T.toarray() / DEG_150             # neighbour averages, (T - 1, N)
        T1 = Z.shape[0]
        X = np.zeros((T1 * N, N + 1))
        X[:, 0] = self.Y[:-1].ravel()
        rows = np.arange(T1 * N)
        X[rows, 1 + np.tile(np.arange(N), T1)] = Z.ravel()        # node-specific neighbour-average columns
        coef, res, *_ = np.linalg.lstsq(X, self.Y[1:].ravel(), rcond=None)
        S = self.cmp["S"]
        assert S.alpha[0] == pytest.approx(coef[0], rel=1e-8)
        np.testing.assert_allclose(S.beta, coef[1:], rtol=1e-7)
        assert self._rss("S") == pytest.approx(res[0], rel=1e-9)

    def test_l_matches_gnar_local(self):
        A = A_150.toarray()
        G = GNAR(A, p=1, s=np.array([1]), ts=self.ts, model_type="local")
        L = self.cmp["L"]
        np.testing.assert_allclose(L.alpha, G.coeffs[0], rtol=1e-7)
        np.testing.assert_allclose(L.beta, G.coeffs[1], rtol=1e-7)
        assert L.sigma_2 == pytest.approx(np.trace(G.sigma_2) / A.shape[0], rel=1e-8)

    def test_loglik_increases_along_chain(self):
        ll = self.cmp.table()["loglik"].to_numpy()
        assert np.all(np.diff(ll) >= -1e-6)

    def test_table_structure(self):
        t = self.cmp.table()
        assert list(t.index) == list(MODELS)
        N = A_150.shape[0]
        assert list(t["n_coef"]) == [1, 2, 3, N + 1, 2 * N]
        assert np.allclose(t["AIC"], -2 * t["loglik"] + 2 * t["n_params"])
        assert np.allclose(t["BIC"], -2 * t["loglik"] + t["n_params"] * np.log(self.M))
        assert t.loc["G", "kappa"] == 1.0 and np.isnan(t.loc["L", "alpha"])

    def test_phi_of_nodewise_fit(self):
        L = self.cmp["L"]
        dense = np.diag(L.alpha) + np.diag(L.beta / DEG_150) @ A_150.toarray()
        np.testing.assert_allclose(L.phi().toarray(), dense)

    def test_demean_false(self):
        c = _cmp(A_150, self.ts - self.ts.mean(axis=0), demean=False)
        np.testing.assert_allclose(c.table()["loglik"], self.cmp.table()["loglik"])


class TestLRTest:

    def setup_method(self):
        rng = np.random.default_rng(1)
        self.cmp = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.5, 0.5, 40, rng))

    def test_statistic_and_f(self):
        res = self.cmp.lr_test("K", "S", method="F")
        t = self.cmp.table()
        assert res.statistic == pytest.approx(2 * (t.loc["S", "loglik"] - t.loc["K", "loglik"]))
        assert res.df == t.loc["S", "n_coef"] - t.loc["K", "n_coef"]
        N, M = A_150.shape[0], self.cmp.n_obs
        rss0, rss1 = self.cmp._fits["K"]["rss"], self.cmp._fits["S"]["rss"]
        df2 = M - (N + 1) - N
        assert res.df_denominator == df2
        assert res.f_statistic == pytest.approx(((rss0 - rss1) / res.df) / (rss1 / df2))
        assert res.method == "F" and 0 <= res.p_value <= 1

    def test_auto_methods(self):
        assert self.cmp.lr_test("AR", "G").method == "F"
        assert self.cmp.lr_test("G", "K").method == "boundary"
        for null, alt in [("AR", "S"), ("G", "S"), ("K", "S"), ("AR", "L"), ("G", "L"), ("K", "L"), ("S", "L")]:
            assert self.cmp.lr_test(null, alt, n_boot=5, seed=0).method == "bartlett"

    def test_bartlett(self):
        r = self.cmp.lr_test("K", "S", method="bartlett", n_boot=29, seed=4)
        plain = self.cmp.lr_test("K", "S", method="F")
        assert r.f_statistic == pytest.approx(plain.f_statistic)
        boot = self.cmp._bootstrap_lr("K", "S", 29, seed=4)
        factor = np.mean(self.cmp._f_from_lr(boot, r.df, r.df_denominator))
        assert r.bartlett_factor == pytest.approx(factor)
        from scipy.stats import f as f_dist
        assert r.p_value == pytest.approx(f_dist.sf(r.f_statistic / factor, r.df, r.df_denominator))
        assert r.n_boot == 29 and "Bartlett factor" in repr(r)
        assert self.cmp.lr_test("K", "S", method="bartlett", seed=4).n_boot == 49
        assert self.cmp.lr_test("K", "S", method="bartlett", n_boot=29, seed=4).p_value == r.p_value

    def test_boundary_p_value(self):
        res = self.cmp.lr_test("G", "K", method="boundary")
        chi = self.cmp.lr_test("G", "K", method="chi2")
        assert res.p_value == pytest.approx(chi.p_value / 2)

    def test_boundary_lr_zero_gives_p_one(self):
        # Data from G (kappa = 1): find a sample where kappa_hat = 1 exactly
        for seed in range(30):
            c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.5, 1.0, 30, np.random.default_rng(seed)))
            if c._fits["K"]["kappa"] == 1.0:
                res = c.lr_test("G", "K")
                assert res.statistic == 0.0 and res.p_value == 1.0
                return
        pytest.fail("kappa_hat never hit the boundary")

    @pytest.mark.parametrize("null, alt, match", [
        ("AR", "K", "not identified"),
        ("K", "G", "not nested"),
        ("S", "S", "not nested"),
        ("X", "L", "Unknown"),
    ])
    def test_invalid_pairs(self, null, alt, match):
        with pytest.raises(ValueError, match=match):
            self.cmp.lr_test(null, alt)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="boundary"):
            self.cmp.lr_test("K", "S", method="boundary")
        with pytest.raises(ValueError, match="method"):
            self.cmp.lr_test("K", "S", method="exact")

    def test_bootstrap(self):
        r1 = self.cmp.lr_test("G", "K", method="bootstrap", n_boot=19, seed=3)
        r2 = self.cmp.lr_test("G", "K", method="bootstrap", n_boot=19, seed=3)
        assert r1.p_value == r2.p_value and 1 / 20 <= r1.p_value <= 1
        assert r1.n_boot == 19 and "bootstrap" in repr(r1)
        assert self.cmp.lr_test("S", "L", method="bootstrap", n_boot=9, seed=0).p_value > 0

    def test_all_valid_pairs_run(self):
        for i, null in enumerate(MODELS):
            for alt in MODELS[i + 1:]:
                if (null, alt) != ("AR", "K"):
                    assert 0 <= self.cmp.lr_test(null, alt).p_value <= 1


class TestSize:
    """Monte Carlo false-rejection rates at the 5% level, one per kind of step in the chain."""
    REPS = 300

    def _rate(self, simulate, null, alt, method="auto", T=30):
        rejections = 0
        non_stationary = 0
        for seed in range(self.REPS):
            c = _cmp(A_150, simulate(T, np.random.default_rng(1000 + seed)))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                rejections += c.lr_test(null, alt, method=method).p_value <= 0.05
            non_stationary += any("non-stationary" in str(w.message) for w in caught)
        # With very short series a fitted null with node-specific coefficients can occasionally be non-stationary;
        # the bootstrap then warns. This must stay rare.
        assert non_stationary <= 0.02 * self.REPS
        return rejections / self.REPS

    def test_ar_to_g(self):
        rate = self._rate(lambda T, rng: _simulate_nodewise(A_150, 0.4, 0.0, T, rng), "AR", "G")
        assert 0.02 <= rate <= 0.09

    def test_g_to_k_boundary(self):
        rate = self._rate(lambda T, rng: _simulate_kappa(A_150, 0.3, 0.5, 1.0, T, rng), "G", "K")
        assert 0.02 <= rate <= 0.09

    def test_k_to_s(self):
        rate = self._rate(lambda T, rng: _simulate_kappa(A_150, 0.3, 0.5, 0.5, T, rng), "K", "S")
        assert 0.02 <= rate <= 0.09

    def test_s_to_l_bartlett_vs_plain_f(self):
        # With q ~ N restrictions and short T the plain F test is badly mis-sized (O(1/T) shift per node); the
        # Bartlett-corrected test (the default) is approximately correct
        betas = np.random.default_rng(0).uniform(0.2, 0.6, A_150.shape[0])
        sim = lambda T, rng: _simulate_nodewise(A_150, 0.3, betas, T, rng)
        rate_bartlett = self._rate(sim, "S", "L", T=15)
        rate_f = self._rate(sim, "S", "L", method="F", T=15)
        assert 0.02 <= rate_bartlett <= 0.09
        assert rate_f < 0.02                 # known distortion of the uncorrected test


class TestSequential:

    @staticmethod
    def _quiet_sequential(c, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = c.sequential_test(**kwargs)
        return res, [str(w.message) for w in caught]

    def _data_from_l(self):
        # Node-specific coefficients unrelated to degree: G is wrong, but not in the direction G -> K targets
        rng = np.random.default_rng(7)
        N = A_150.shape[0]
        alphas, betas = rng.uniform(-0.3, 0.6, N), rng.uniform(0.0, 0.35, N)
        return _cmp(A_150, _simulate_nodewise(A_150, alphas, betas, 300, rng))

    def test_combined_is_default_and_selects_k(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.6, 0.3, 200, np.random.default_rng(5)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")                 # K is adequate: no warning
            res = c.sequential_test(seed=0)
        assert res.tests == "combined" and res.selected == "K" and res.adequate
        t = res.table
        assert list(t.index) == ["AR -> G | AR -> L", "G -> K | G -> L", "K -> S | K -> L", "S -> L"]
        assert list(t["reached"]) == [True, True, True, False]
        expected = np.minimum(1, 2 * np.minimum(t["p_adjacent"], t["p_vs_L"]))
        expected.iloc[-1] = t["p_adjacent"].iloc[-1]       # for S the adjacent test is the test against L
        np.testing.assert_allclose(t["p_value"], expected)
        np.testing.assert_allclose(t["adjusted_p"], np.maximum.accumulate(t["p_value"]))
        assert list(t["reject"]) == list(t["adjusted_p"] <= 0.05)
        assert res.selected_vs_L_p == pytest.approx(t.loc["K -> S | K -> L", "p_vs_L"])

    def test_adjacent_mode(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.6, 0.3, 200, np.random.default_rng(5)))
        res = c.sequential_test(tests="adjacent", seed=0)
        assert list(res.table.index) == ["AR -> G", "G -> K", "K -> S", "S -> L"]
        assert res.selected == "K"

    def test_selects_ar_without_network(self):
        c = _cmp(A_150, _simulate_nodewise(A_150, 0.4, 0.0, 100, np.random.default_rng(6)))
        res = c.sequential_test(seed=0)
        assert res.selected == "AR"
        assert list(res.table["reached"]) == [True, False, False, False]

    @pytest.mark.parametrize("tests", ["combined", "vs_L"])
    def test_selects_l_when_g_is_wrong_in_another_direction(self, tests):
        res, messages = self._quiet_sequential(self._data_from_l(), tests=tests, seed=0)
        assert res.selected == "L" and np.isnan(res.selected_vs_L_p) and res.adequate
        assert not any("non-stationary" in m for m in messages)

    def test_adjacent_stops_early_and_warns(self):
        # Documented power caveat of adjacent tests: the procedure stops at G although G -> L rejects strongly; the
        # adequacy warning makes this visible
        c = self._data_from_l()
        res, messages = self._quiet_sequential(c, tests="adjacent", seed=0)
        assert any("selected model G is rejected against the local model L" in m for m in messages)
        assert res.selected == "G" and not res.adequate and res.selected_vs_L_p < 1e-6
        assert "rejected against L" in repr(res)

    def test_vs_l(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.6, 0.3, 200, np.random.default_rng(5)))
        res = c.sequential_test(tests="vs_L", seed=0)
        assert list(res.table.index) == ["AR -> L", "G -> L", "K -> L", "S -> L"]
        assert res.selected == "K"
        assert "Selected model: K (not rejected against L" in repr(res)

    def test_reproducible_with_seed(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.6, 0.3, 60, np.random.default_rng(5)))
        pd.testing.assert_frame_equal(c.sequential_test(seed=3).table, c.sequential_test(seed=3).table)

    def test_invalid_arguments(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.6, 0.3, 30, np.random.default_rng(5)))
        with pytest.raises(ValueError, match="tests"):
            c.sequential_test(tests="top_down")
        with pytest.raises(ValueError, match="alpha"):
            c.sequential_test(alpha=1.5)


class TestValidationAndScale:

    def test_regular_graph(self):
        A = nx.to_numpy_array(nx.cycle_graph(10))
        with pytest.raises(ValueError, match="regular"):
            ModelComparison(A, np.random.default_rng(0).normal(size=(20, 10)))

    def test_short_series(self):
        with pytest.raises(ValueError, match="five time steps"):
            ModelComparison(A_150, np.zeros((4, A_150.shape[0])))

    def test_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            ModelComparison(A_150, np.zeros((20, 3)))

    def test_getitem(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.6, 0.3, 30, np.random.default_rng(5)))
        assert isinstance(c["K"], KappaGNAR) and c["G"].kappa == 1.0
        with pytest.raises(KeyError):
            c["Q"]

    def test_dataframe_input(self):
        ts = _simulate_kappa(A_150, 0.3, 0.6, 0.3, 30, np.random.default_rng(5))
        c1 = _cmp(A_150, pd.DataFrame(ts))
        c2 = _cmp(A_150, ts)
        pd.testing.assert_frame_equal(c1.table(), c2.table())

    def test_large_graph_is_fast(self):
        A = _graph(10_000, seed=1)
        np.random.seed(0)
        ts = KappaGNAR(A, alpha=0.3, b=0.6, kappa=0.4).simulate(200)
        t0 = time.perf_counter()
        c = _cmp(A, ts)
        res = c.sequential_test(n_boot=9, seed=0)
        c.table()
        assert time.perf_counter() - t0 < 15
        assert res.selected == "K"


class TestNodewiseModel:

    def _random_coefficients(self, N, rng, signs="mixed"):
        alpha = rng.uniform(-0.5, 0.6, N)
        beta = rng.uniform(-0.8, 0.8, N) if signs == "mixed" else rng.uniform(0.1, 0.8, N) * (1 if signs == "pos" else -1)
        return alpha, beta

    @pytest.mark.parametrize("signs", ["mixed", "pos", "neg"])
    @pytest.mark.parametrize("N", [60, 300])          # dense path (<= 200 nodes) and sparse eigensolver path
    def test_spectral_radius_matches_dense(self, signs, N):
        A = _graph(N, seed=3)
        rng = np.random.default_rng(N)
        alpha, beta = self._random_coefficients(A.shape[0], rng, signs)
        m = NodewiseModel(A, alpha, beta)
        dense = np.max(np.abs(np.linalg.eigvals(m.phi().toarray())))
        assert m.phi_spectral_radius() == pytest.approx(dense, rel=1e-7)

    def test_spectral_radius_with_zero_betas(self):
        A = _graph(300, seed=3)
        rng = np.random.default_rng(0)
        alpha, beta = self._random_coefficients(A.shape[0], rng, "pos")
        beta[::7] = 0.0                                  # zeros break the symmetric route; the general solver is used
        m = NodewiseModel(A, alpha, beta)
        assert m.phi_spectral_radius() == pytest.approx(np.max(np.abs(np.linalg.eigvals(m.phi().toarray()))), rel=1e-7)

    @pytest.mark.parametrize("signs", ["mixed", "pos"])
    def test_row_budget_guarantees_stationarity(self, signs):
        A = _graph(300, seed=4)
        rng = np.random.default_rng(1)
        for _ in range(5):
            alpha = rng.uniform(-0.5, 0.5, A.shape[0])
            beta_shape = rng.uniform(-1.5, 1.5, A.shape[0]) if signs == "mixed" else rng.uniform(0.5, 1.5, A.shape[0])
            m = NodewiseModel(A, alpha, beta_shape, row_budget=0.95)
            assert m.row_sums().max() == pytest.approx(0.95)
            assert m.beta_scale < 1
            np.testing.assert_allclose(m.beta, m.beta_scale * beta_shape)       # one global factor: shape preserved
            assert m.phi_spectral_radius() <= 0.95 + 1e-9 and m.is_stationary()

    def test_row_budget_leaves_admissible_coefficients_unchanged(self):
        m = NodewiseModel(A_150, 0.3, np.full(A_150.shape[0], 0.5), row_budget=0.95)
        assert m.beta_scale == 1.0 and np.all(m.beta == 0.5)

    def test_row_budget_validation(self):
        with pytest.raises(ValueError, match="row_budget"):
            NodewiseModel(A_150, 0.3, 0.5, row_budget=1.2)
        with pytest.raises(ValueError, match="alpha"):
            NodewiseModel(A_150, 0.96, 0.1, row_budget=0.95)

    def test_without_budget_can_be_non_stationary(self):
        m = NodewiseModel(A_150, 0.6, 0.6)
        assert not m.is_stationary() and m.phi_spectral_radius() == pytest.approx(1.2)

    def test_simulate(self):
        m = NodewiseModel(A_150, 0.3, 0.4, mean=2.0, sigma_2=0.5)
        a = m.simulate(20, rng=5)
        b = m.simulate(20, rng=np.random.default_rng(5))
        np.testing.assert_array_equal(a, b)
        assert a.shape == (20, A_150.shape[0])
        x = m.simulate(5000, rng=1)
        assert x.mean() == pytest.approx(2.0, abs=0.02)

    def test_fitted_models_are_nodewise_models(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.5, 0.5, 40, np.random.default_rng(0)))
        for name in ("AR", "S", "L"):
            assert isinstance(c[name], NodewiseModel)
            assert c[name].is_stationary()
        assert "NodewiseFit(model=L" in repr(c["L"])

    def test_bootstrap_warns_for_non_stationary_null(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.5, 0.5, 40, np.random.default_rng(0)))
        c._fits["S"]["beta"] = c._fits["S"]["beta"] + 1.0            # force a non-stationary fitted null
        c._models.pop("S", None)
        c.__dict__.pop("_rho_cache", None)
        with pytest.warns(UserWarning, match="fitted null model S is non-stationary"):
            c._bootstrap_lr("S", "L", 2, seed=0)
