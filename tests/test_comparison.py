import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gnar import GNAR, KappaGNAR, ModelComparison
from gnar.comparison import MODELS
from gnar.utils.graphs import powerlaw_configuration_graph


def _graph(N, seed=0):
    G = powerlaw_configuration_graph(N, seed=seed)
    return nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), format="csr")


A_150 = _graph(150)
DEG_150 = np.asarray(A_150.sum(axis=1)).ravel()


def _simulate_nodewise(A, alpha, beta, T, rng, sigma_2=1.0, burn_in=50):
    """Simulate X_t = diag(alpha) X_{t-1} + diag(beta / d) A X_{t-1} + e_t (beta on the neighbour average)."""
    d = np.asarray(A.sum(axis=1)).ravel()
    N = d.size
    phi = (sp.diags_array(np.broadcast_to(alpha, N).astype(float))
           + sp.diags_array(np.broadcast_to(beta, N) / d) @ A).tocsr()
    x = np.zeros(N)
    out = np.empty((T, N))
    for t in range(burn_in + T):
        x = phi @ x + rng.standard_normal(N) * np.sqrt(sigma_2)
        if t >= burn_in:
            out[t - burn_in] = x
    return out


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
        for seed in range(self.REPS):
            c = _cmp(A_150, simulate(T, np.random.default_rng(1000 + seed)))
            rejections += c.lr_test(null, alt, method=method).p_value <= 0.05
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

    def test_selects_k(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.6, 0.3, 200, np.random.default_rng(5)))
        res = c.sequential_test()
        assert res.selected == "K"
        t = res.table
        assert list(t.index) == ["AR -> G", "G -> K", "K -> S", "S -> L"]
        assert list(t["reached"]) == [True, True, True, False]
        np.testing.assert_allclose(t["adjusted_p"], np.maximum.accumulate(t["p_value"]))
        assert list(t["reject"]) == list(t["adjusted_p"] <= 0.05)

    def test_selects_ar_without_network(self):
        c = _cmp(A_150, _simulate_nodewise(A_150, 0.4, 0.0, 100, np.random.default_rng(6)))
        res = c.sequential_test()
        assert res.selected == "AR"
        assert list(res.table["reached"]) == [True, False, False, False]

    def test_selects_l_when_everything_rejected(self):
        rng = np.random.default_rng(7)
        N = A_150.shape[0]
        alphas, betas = rng.uniform(-0.3, 0.6, N), rng.uniform(0.0, 0.35, N)
        c = _cmp(A_150, _simulate_nodewise(A_150, alphas, betas, 300, rng))
        assert c.sequential_test(tests="vs_L", seed=0).selected == "L"

    def test_adjacent_can_stop_early(self):
        # Documented power caveat: with node-specific coefficients unrelated to degree, G is wrong but the adjacent
        # G -> K test has nothing to detect, so the adjacent procedure may stop at G while G -> L clearly rejects
        rng = np.random.default_rng(7)
        N = A_150.shape[0]
        alphas, betas = rng.uniform(-0.3, 0.6, N), rng.uniform(0.0, 0.35, N)
        c = _cmp(A_150, _simulate_nodewise(A_150, alphas, betas, 300, rng))
        adjacent = c.sequential_test(seed=0)
        assert adjacent.selected == "G"
        assert c.lr_test("G", "L", seed=0).p_value < 1e-6

    def test_vs_l(self):
        c = _cmp(A_150, _simulate_kappa(A_150, 0.3, 0.6, 0.3, 200, np.random.default_rng(5)))
        res = c.sequential_test(tests="vs_L")
        assert list(res.table.index) == ["AR -> L", "G -> L", "K -> L", "S -> L"]
        assert res.selected == "K"
        assert "Selected model: K" in repr(res)

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
