import warnings

import numpy as np
import pandas as pd
import pytest

from gnar import GNAR, KappaGNAR
from gnar.kappa import kappa_spectral_radius, kappa_spectrum_bounds, kappa_weight_matrix


def _graph(edges, d):
    A = np.zeros((d, d))
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    return A


# Irregular, non-bipartite: a star centred at 0 with a tail 4-5 and a triangle 1-2-0
A_IRREG = _graph([(0, 1), (0, 2), (0, 3), (0, 4), (4, 5), (1, 2)], 6)
# Irregular, bipartite (a tree): star with a tail
A_TREE = _graph([(0, 1), (0, 2), (0, 3), (0, 4), (4, 5)], 6)
# 2-regular: 5-cycle
A_CYCLE = _graph([(i, (i + 1) % 5) for i in range(5)], 5)
KAPPAS = [0.0, 0.3, 0.5, 0.8, 1.0]


def _eig_D_kappa_A(A, kappa):
    d = A.sum(axis=1)
    return np.linalg.eigvals(np.diag(d ** -kappa) @ A)


def _quiet(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return KappaGNAR(**kwargs)


class TestWeightsAndSpectrum:

    @pytest.mark.parametrize("kappa", KAPPAS)
    @pytest.mark.parametrize("A", [A_IRREG, A_TREE, A_CYCLE])
    def test_spectral_radius_matches_dense_eigenvalues(self, A, kappa):
        assert kappa_spectral_radius(A, kappa) == pytest.approx(np.abs(_eig_D_kappa_A(A, kappa)).max())

    @pytest.mark.parametrize("kappa", KAPPAS)
    @pytest.mark.parametrize("A", [A_IRREG, A_TREE, A_CYCLE])
    def test_spectrum_bounds(self, A, kappa):
        eig = np.real(_eig_D_kappa_A(A, kappa))
        mu_min, mu_max = kappa_spectrum_bounds(A, kappa)
        assert mu_min == pytest.approx(eig.min())
        assert mu_max == pytest.approx(eig.max())

    @pytest.mark.parametrize("A", [A_IRREG, A_TREE, A_CYCLE])
    def test_rho_is_one_at_kappa_one(self, A):
        assert kappa_spectral_radius(A, 1.0) == pytest.approx(1.0)

    def test_regular_graph_rho(self):
        # For a k-regular graph, D^{-kappa} A = k^{-kappa} A and rho(A) = k
        for kappa in KAPPAS:
            assert kappa_spectral_radius(A_CYCLE, kappa) == pytest.approx(2 ** (1 - kappa))

    def test_bipartite_spectrum_symmetric(self):
        for kappa in KAPPAS:
            mu_min, mu_max = kappa_spectrum_bounds(A_TREE, kappa)
            assert mu_min == pytest.approx(-mu_max)

    def test_weight_matrix(self):
        kappa = 0.4
        W = kappa_weight_matrix(A_IRREG, kappa)
        d = A_IRREG.sum(axis=1)
        np.testing.assert_allclose(W, np.diag(d ** -kappa) @ A_IRREG)
        np.testing.assert_allclose(W.sum(axis=1), d ** (1 - kappa))


class TestValidation:

    def test_isolated_node(self):
        A = _graph([(0, 1), (1, 2)], 4)
        with pytest.raises(ValueError, match="isolated"):
            KappaGNAR(A, alpha=0.2, b=0.3, kappa=0.5)

    def test_not_symmetric(self):
        A = A_IRREG.copy()
        A[0, 5] = 1
        with pytest.raises(ValueError, match="symmetric"):
            KappaGNAR(A, alpha=0.2, b=0.3, kappa=0.5)

    def test_not_binary(self):
        with pytest.raises(ValueError, match="binary"):
            KappaGNAR(2 * A_IRREG, alpha=0.2, b=0.3, kappa=0.5)

    def test_self_loop(self):
        A = A_IRREG.copy()
        A[1, 1] = 1
        with pytest.raises(ValueError, match="self-loops"):
            KappaGNAR(A, alpha=0.2, b=0.3, kappa=0.5)

    def test_not_square(self):
        with pytest.raises(ValueError, match="square"):
            KappaGNAR(np.ones((3, 4)), alpha=0.2, b=0.3, kappa=0.5)

    @pytest.mark.parametrize("kappa", [-0.1, 1.1, np.array([0.5])])
    def test_kappa_out_of_range(self, kappa):
        with pytest.raises(ValueError, match="kappa"):
            KappaGNAR(A_IRREG, alpha=0.2, b=0.3, kappa=kappa)

    def test_b_and_beta_exclusive(self):
        with pytest.raises(ValueError, match="exactly one"):
            KappaGNAR(A_IRREG, alpha=0.2, b=0.3, kappa=0.5, beta=0.1)
        with pytest.raises(ValueError, match="exactly one"):
            KappaGNAR(A_IRREG, alpha=0.2, kappa=0.5)

    @pytest.mark.parametrize("sigma_2", [0, -1.0, np.eye(6)])
    def test_bad_sigma(self, sigma_2):
        with pytest.raises(ValueError, match="sigma_2"):
            KappaGNAR(A_IRREG, alpha=0.2, b=0.3, kappa=0.5, sigma_2=sigma_2)


class TestParametrisation:

    def test_b_beta_round_trip(self):
        K1 = KappaGNAR(A_IRREG, alpha=0.2, b=0.3, kappa=0.4)
        K2 = KappaGNAR(A_IRREG, alpha=0.2, beta=K1.beta, kappa=0.4)
        assert K2.b == pytest.approx(0.3)
        assert K1.beta == pytest.approx(0.3 / kappa_spectral_radius(A_IRREG, 0.4))
        np.testing.assert_allclose(K1.phi().toarray(), K2.phi().toarray())

    def test_phi_definition(self):
        K = KappaGNAR(A_IRREG, alpha=0.2, b=0.3, kappa=0.4)
        d = A_IRREG.sum(axis=1)
        expected = 0.2 * np.eye(6) + K.beta * np.diag(d ** -0.4) @ A_IRREG
        np.testing.assert_allclose(K.phi().toarray(), expected)
        # The network part of Phi has spectral radius |b|
        assert np.abs(np.linalg.eigvals(K.phi().toarray() - 0.2 * np.eye(6))).max() == pytest.approx(0.3)

    def test_kappa_one_matches_global_gnar(self):
        K = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=1.0, mean=np.arange(6.0))
        assert K.b == pytest.approx(K.beta)
        G = GNAR(A_IRREG, p=1, s=np.array([1]), model_type="global",
                 coeffs=np.tile([[0.3], [0.4]], (1, 6)), mean=np.arange(6.0))
        np.testing.assert_allclose(K.to_var().coeffs, G.to_var().coeffs, atol=1e-12)

    def test_regular_graph(self):
        K = KappaGNAR(A_CYCLE, alpha=0.2, b=0.3, kappa=0.5)
        assert K.is_regular()
        assert not KappaGNAR(A_IRREG, alpha=0.2, b=0.3, kappa=0.5).is_regular()
        # In the b parametrisation, kappa has no effect on a regular graph
        np.testing.assert_allclose(K.phi().toarray(), KappaGNAR(A_CYCLE, alpha=0.2, b=0.3, kappa=0.0).phi().toarray())

    def test_degrees(self):
        K = KappaGNAR(A_IRREG, alpha=0.2, b=0.3, kappa=0.5)
        np.testing.assert_array_equal(K.degrees, A_IRREG.sum(axis=1))


class TestStationarity:

    @pytest.mark.parametrize("A", [A_IRREG, A_TREE, A_CYCLE])
    def test_closed_form_matches_eigenvalues(self, A):
        rng = np.random.default_rng(0)
        for _ in range(300):
            alpha, b = rng.uniform(-1.5, 1.5, size=2)
            kappa = rng.uniform(0, 1)
            K = _quiet(A=A, alpha=alpha, b=b, kappa=kappa)
            assert K.is_stationary() == (np.abs(np.linalg.eigvals(K.phi().toarray())).max() < 1)

    def test_diamond_is_sufficient(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            alpha, b = rng.uniform(-1, 1, size=2)
            if abs(alpha) + abs(b) < 1:
                assert KappaGNAR(A_IRREG, alpha=alpha, b=b, kappa=rng.uniform(0, 1)).is_stationary()

    def test_bipartite_diamond_is_necessary(self):
        # For a bipartite graph, |alpha| + |b| < 1 is also necessary
        K = _quiet(A=A_TREE, alpha=0.5, b=-0.6, kappa=0.5)
        assert not K.is_stationary()
        # ...whereas for a non-bipartite graph the region is larger
        assert _quiet(A=A_IRREG, alpha=0.5, b=-0.6, kappa=0.5).is_stationary()

    def test_nonstationary_warns(self):
        with pytest.warns(UserWarning, match="non-stationary"):
            KappaGNAR(A_IRREG, alpha=0.6, b=0.5, kappa=0.5)

    def test_stationary_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            K = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.5)
            K.to_var()
        assert K.to_var().is_stationary()


class TestVARFunctionality:

    def setup_method(self):
        self.K = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.3, mean=np.linspace(-1, 1, 6), sigma_2=0.5)

    def test_to_var(self):
        V = self.K.to_var()
        np.testing.assert_allclose(V.coeffs, self.K.phi().toarray().T)
        np.testing.assert_allclose(V.mu, self.K.mu)
        assert V.sigma_2 == 0.5

    def test_simulate_matches_var(self):
        np.random.seed(0)
        a = self.K.simulate(100)
        np.random.seed(0)
        b = self.K.to_var().simulate(100)
        assert a.shape == (100, 6)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_simulate_moments(self):
        np.random.seed(1)
        sim = self.K.simulate(50_000)
        np.testing.assert_allclose(sim.mean(axis=0), self.K.mu.ravel(), atol=0.03)
        np.testing.assert_allclose(np.cov(sim.T, bias=True), self.K.compute_autocov_mats(0)[0], atol=0.03)

    def test_simulate_sigma_override(self):
        np.random.seed(2)
        sim = self.K.simulate(50_000, sigma_2=2.0)
        expected = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.3, sigma_2=2.0).compute_autocov_mats(0)[0]
        np.testing.assert_allclose(np.cov(sim.T, bias=True), expected, atol=0.1)

    def test_predict_one_step(self):
        x = np.random.default_rng(3).normal(size=(1, 6))
        expected = self.K.mu + (x - self.K.mu) @ self.K.phi().toarray().T
        np.testing.assert_allclose(self.K.predict(x, h=1), expected)

    def test_predict_multi_step_and_origins(self):
        x = np.random.default_rng(4).normal(size=(10, 6))
        preds = self.K.predict(x, h=3)
        assert preds.shape == (10, 6, 3)
        phi, mu = self.K.phi().toarray(), self.K.mu.ravel()
        expected = mu + np.linalg.matrix_power(phi, 3) @ (x[4] - mu)
        np.testing.assert_allclose(preds[4, :, 2], expected)

    def test_predict_dataframe(self):
        df = pd.DataFrame(np.random.default_rng(5).normal(size=(5, 6)), columns=list("abcdef"))
        out = self.K.predict(df, h=2)
        assert list(out.columns.get_level_values(0).unique()) == list("abcdef")
        assert out.shape == (5, 12)

    def test_autocov_stationary_equation(self):
        # Gamma(0) = Phi Gamma(0) Phi^T + sigma_2 I and Gamma(1) = Phi Gamma(0)
        G = self.K.compute_autocov_mats(2)
        phi = self.K.phi().toarray()
        np.testing.assert_allclose(G[0], phi @ G[0] @ phi.T + 0.5 * np.eye(6), atol=1e-10)
        np.testing.assert_allclose(G[1], phi @ G[0], atol=1e-10)
        np.testing.assert_allclose(G[2], phi @ G[1], atol=1e-10)

    def test_autocorr(self):
        R = self.K.compute_autocorr_mats(1)
        np.testing.assert_allclose(np.diag(R[0]), np.ones(6))

    def test_repr_str(self):
        assert "KappaGNAR" in repr(self.K)
        s = str(self.K)
        assert "rho_kappa" in s and "6 nodes and 6 edges" in s


class TestNames:

    def test_default_names(self):
        K = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.5)
        assert list(K.names) == list(range(6))
        assert list(K.degrees.index) == list(range(6))

    def test_custom_names(self):
        names = list("abcdef")
        K = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.5, names=names)
        assert list(K.names) == names
        assert K.degrees["a"] == 4
        assert list(K.to_var()._names) == names
        assert "a" in str(K)

    def test_simulate_as_frame(self):
        K = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.5, names=list("abcdef"))
        np.random.seed(0)
        df = K.simulate(20, as_frame=True)
        np.random.seed(0)
        arr = K.simulate(20)
        assert isinstance(df, pd.DataFrame) and list(df.columns) == list("abcdef")
        np.testing.assert_array_equal(df.to_numpy(), arr)

    def test_invalid_names(self):
        with pytest.raises(ValueError, match="number of names"):
            KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.5, names=list("abc"))
        with pytest.raises(ValueError, match="unique"):
            KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.5, names=list("aabcde"))


class TestPrintout:

    def test_constant_mean(self):
        assert "Mean: 0 (all nodes)" in str(KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.5))

    def test_small_graph_varying_mean_shows_all(self):
        s = str(KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.5, mean=np.arange(6.0), names=list("abcdef")))
        assert "f" in s and "5.0" in s

    def test_large_graph_varying_mean_is_compact(self):
        A = np.zeros((30, 30))
        for i in range(30):
            A[i, (i + 1) % 30] = A[(i + 1) % 30, i] = 1
        A[0, 15] = A[15, 0] = 1
        s = str(KappaGNAR(A, alpha=0.3, b=0.4, kappa=0.5, mean=np.arange(30.0)))
        assert "varies by node, from 0 to 29" in s
        assert len(s.splitlines()) < 15


# ---------------------------------------------------------------------- sparse / large-graph code paths
import networkx as nx
import scipy.sparse as sp

from gnar import VAR
from gnar.utils.graphs import powerlaw_configuration_graph


def _powerlaw_A(d=400, seed=0):
    G = powerlaw_configuration_graph(d, seed=seed)
    return nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), format="csr")


A_BIG = _powerlaw_A()                                   # > 200 nodes: sparse Lanczos path
A_BIG_TREE = nx.to_scipy_sparse_array(nx.balanced_tree(3, 5), format="csr")   # 364 nodes, bipartite


class TestSparseInput:

    @pytest.mark.parametrize("convert", [sp.csr_matrix, sp.csr_array, sp.coo_matrix, sp.lil_array])
    def test_sparse_matches_dense(self, convert):
        K_dense = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.4)
        K_sparse = KappaGNAR(convert(A_IRREG), alpha=0.3, b=0.4, kappa=0.4)
        assert K_sparse.rho == pytest.approx(K_dense.rho)
        np.testing.assert_allclose(K_sparse.phi().toarray(), K_dense.phi().toarray())

    def test_phi_is_sparse(self):
        assert sp.issparse(KappaGNAR(A_BIG, alpha=0.3, b=0.4, kappa=0.4).phi())

    def test_weight_matrix_type_follows_input(self):
        assert isinstance(kappa_weight_matrix(A_IRREG, 0.5), np.ndarray)
        W = kappa_weight_matrix(sp.csr_array(A_IRREG), 0.5)
        assert sp.issparse(W)
        np.testing.assert_allclose(W.toarray(), kappa_weight_matrix(A_IRREG, 0.5))

    @pytest.mark.parametrize("bad, match", [
        (lambda: sp.csr_array(2 * A_IRREG), "binary"),
        (lambda: sp.csr_array(np.triu(A_IRREG)), "symmetric"),
        (lambda: sp.csr_array(A_IRREG + np.eye(6)), "self-loops"),
        (lambda: sp.csr_array(np.pad(A_IRREG, ((0, 1), (0, 1)))), "isolated"),
        (lambda: A_IRREG.tolist(), "square NumPy array or SciPy sparse"),
    ])
    def test_sparse_validation(self, bad, match):
        with pytest.raises(ValueError, match=match):
            KappaGNAR(bad(), alpha=0.3, b=0.4, kappa=0.5)

    def test_explicit_zeros_ignored(self):
        rows, cols = np.nonzero(A_IRREG)
        # Two explicitly stored zeros at (0, 5) and (5, 0): still no edge there
        A = sp.csr_array((np.r_[np.ones(rows.size), 0.0, 0.0], (np.r_[rows, 0, 5], np.r_[cols, 5, 0])), shape=(6, 6))
        assert A.nnz == rows.size + 2
        K = KappaGNAR(A, alpha=0.3, b=0.4, kappa=0.5)
        np.testing.assert_array_equal(K.degrees.to_numpy(), A_IRREG.sum(axis=1))


class TestLargeGraphSpectrum:

    @pytest.mark.parametrize("kappa", KAPPAS)
    @pytest.mark.parametrize("A", [A_BIG, A_BIG_TREE])
    def test_lanczos_bounds_match_dense(self, A, kappa):
        d = A.sum(axis=1)
        s = d ** (-kappa / 2)
        ev = np.linalg.eigvalsh(s[:, None] * A.toarray() * s[None, :])
        mu_min, rho = kappa_spectrum_bounds(A, kappa)
        assert rho == pytest.approx(ev[-1], rel=1e-9)
        assert mu_min == pytest.approx(ev[0], rel=1e-9, abs=1e-9)

    def test_bipartite_large(self):
        for kappa in KAPPAS:
            mu_min, rho = kappa_spectrum_bounds(A_BIG_TREE, kappa)
            assert mu_min == pytest.approx(-rho, rel=1e-9)

    @pytest.mark.parametrize("A", [A_IRREG, A_TREE, A_BIG])
    def test_phi_spectral_radius(self, A):
        rng = np.random.default_rng(0)
        for _ in range(5):
            alpha, b = rng.uniform(-0.6, 0.6, size=2)
            K = _quiet(A=A, alpha=alpha, b=b, kappa=rng.uniform(0, 1))
            assert K.phi_spectral_radius() == pytest.approx(np.abs(np.linalg.eigvals(K.phi().toarray())).max(), rel=1e-8)


class TestClosedFormMoments:

    @pytest.mark.parametrize("A", [A_IRREG, A_TREE, A_CYCLE, A_BIG])
    @pytest.mark.parametrize("alpha, b, kappa", [(0.3, 0.4, 0.3), (0.2, -0.5, 0.8), (-0.4, 0.5, 0.0), (0.5, 0.45, 1.0)])
    def test_matches_var_lyapunov(self, A, alpha, b, kappa):
        K = KappaGNAR(A, alpha=alpha, b=b, kappa=kappa, sigma_2=0.7)
        np.testing.assert_allclose(K.compute_autocov_mats(3), K.to_var().compute_autocov_mats(3), atol=1e-9)

    def test_autocorr_and_variances(self):
        K = KappaGNAR(A_BIG, alpha=0.3, b=0.5, kappa=0.4, sigma_2=2.0)
        G = K.compute_autocov_mats(1)
        R = K.compute_autocorr_mats(1)
        np.testing.assert_allclose(np.diag(R[0]), 1.0)
        sd = np.sqrt(np.diag(G[0]))
        np.testing.assert_allclose(R[1], G[1] / np.outer(sd, sd), atol=1e-12)
        np.testing.assert_allclose(K.marginal_variances().to_numpy(), np.diag(G[0]), rtol=1e-10)
        np.testing.assert_allclose(G[0], G[0].T)

    def test_cache(self):
        K = KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.4)
        G1 = K.compute_autocov_mats(1)
        assert K._eig_cache is not None
        np.testing.assert_array_equal(K.compute_autocov_mats(1), G1)
        K.clear_cache()
        assert K._eig_cache is None
        np.testing.assert_allclose(K.compute_autocov_mats(1), G1)

    def test_nonstationary_raises(self):
        K = _quiet(A=A_IRREG, alpha=0.6, b=0.5, kappa=0.5)
        with pytest.raises(ValueError, match="stationary"):
            K.compute_autocov_mats(0)
        with pytest.raises(ValueError, match="stationary"):
            K.marginal_variances()

    def test_negative_max_lag(self):
        with pytest.raises(ValueError, match="max_lag"):
            KappaGNAR(A_IRREG, alpha=0.3, b=0.4, kappa=0.4).compute_autocov_mats(-1)


class TestSparseForecastingAndSimulation:

    def setup_method(self):
        self.K = KappaGNAR(A_BIG, alpha=0.3, b=0.5, kappa=0.4, mean=1.5, sigma_2=0.8)
        self.V = self.K.to_var()

    def test_predict_matches_var(self):
        x = np.random.default_rng(0).normal(size=(7, self.K.phi().shape[0]))
        np.testing.assert_allclose(self.K.predict(x, h=4), self.V.predict(x, h=4), atol=1e-12)
        np.testing.assert_allclose(self.K.predict(x[:1], h=4), self.V.predict(x[:1], h=4), atol=1e-12)

    def test_predict_dataframe_matches_var(self):
        d = self.K.phi().shape[0]
        df = pd.DataFrame(np.random.default_rng(1).normal(size=(5, d)), index=range(10, 15))
        out_k, out_v = self.K.predict(df, h=3), self.V.predict(df, h=3)
        pd.testing.assert_frame_equal(out_k, out_v, check_exact=False, atol=1e-12)
        pd.testing.assert_frame_equal(self.K.predict(df.iloc[[0]], h=3), self.V.predict(df.iloc[[0]], h=3),
                                      check_exact=False, atol=1e-12)

    def test_predict_validation(self):
        with pytest.raises(ValueError, match="shape"):
            self.K.predict(np.zeros((3, 5)))
        with pytest.raises(ValueError, match="horizon"):
            self.K.predict(np.zeros((1, self.K.phi().shape[0])), h=0)

    def test_simulate_matches_var(self):
        np.random.seed(3)
        a = self.K.simulate(200, burn_in=10)
        np.random.seed(3)
        b = self.V.simulate(200, burn_in=10)
        np.testing.assert_allclose(a, b, atol=1e-10)

    def test_simulate_burn_in_zero(self):
        np.random.seed(0)
        sim = self.K.simulate(3, burn_in=0)
        assert np.all(sim != self.K.mu)


class TestVARStationarityFlag:

    def test_flag_disables_check(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            V = VAR(p=1, coeffs=1.5 * np.eye(2), check_stationarity=False)
        assert not V.is_stationary()

    def test_default_still_warns(self):
        with pytest.warns(UserWarning, match="non-stationary"):
            VAR(p=1, coeffs=1.5 * np.eye(2))
