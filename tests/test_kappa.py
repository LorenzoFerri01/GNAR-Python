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
        np.testing.assert_allclose(K1.phi(), K2.phi())

    def test_phi_definition(self):
        K = KappaGNAR(A_IRREG, alpha=0.2, b=0.3, kappa=0.4)
        d = A_IRREG.sum(axis=1)
        expected = 0.2 * np.eye(6) + K.beta * np.diag(d ** -0.4) @ A_IRREG
        np.testing.assert_allclose(K.phi(), expected)
        # The network part of Phi has spectral radius |b|
        assert np.abs(np.linalg.eigvals(K.phi() - 0.2 * np.eye(6))).max() == pytest.approx(0.3)

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
        np.testing.assert_allclose(K.phi(), KappaGNAR(A_CYCLE, alpha=0.2, b=0.3, kappa=0.0).phi())

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
            assert K.is_stationary() == (np.abs(np.linalg.eigvals(K.phi())).max() < 1)

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
        np.testing.assert_allclose(V.coeffs, self.K.phi().T)
        np.testing.assert_allclose(V.mu, self.K.mu)
        assert V.sigma_2 == 0.5

    def test_simulate_matches_var(self):
        np.random.seed(0)
        a = self.K.simulate(100)
        np.random.seed(0)
        b = self.K.to_var().simulate(100)
        assert a.shape == (100, 6)
        np.testing.assert_array_equal(a, b)

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
        expected = self.K.mu + (x - self.K.mu) @ self.K.phi().T
        np.testing.assert_allclose(self.K.predict(x, h=1), expected)

    def test_predict_multi_step_and_origins(self):
        x = np.random.default_rng(4).normal(size=(10, 6))
        preds = self.K.predict(x, h=3)
        assert preds.shape == (10, 6, 3)
        phi, mu = self.K.phi(), self.K.mu.ravel()
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
        phi = self.K.phi()
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
