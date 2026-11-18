"""Regression tests for previously-fixed bugs."""
import warnings

import numpy as np
import pytest

from gnar import GNAR, VAR
from gnar.utils.gnar_linear_regression import format_X_y
from gnar.utils.neighbour_sets import compute_neighbour_sums, neighbour_set_mats


A_RING6 = np.zeros((6, 6))
for _i in range(6):
    A_RING6[_i, (_i + 1) % 6] = A_RING6[(_i + 1) % 6, _i] = 1
A_RING6[0, 3] = A_RING6[3, 0] = 1


def _standard_coeffs(d):
    alphas = np.vstack([np.linspace(0.15, 0.35, d), np.linspace(-0.1, 0.1, d)])
    betas = np.array([[0.25], [0.1], [-0.1]]) * np.ones((1, d))
    return np.vstack([alphas, betas])


class TestOLSNoiseCovariance:

    @pytest.mark.parametrize("model_type", ["global", "standard", "local"])
    @pytest.mark.parametrize("p", [1, 3])
    def test_sigma2_divides_by_n_minus_p(self, model_type, p):
        np.random.seed(0)
        n = 30
        ts = np.random.normal(size=(n, 6))
        s = np.ones(p, dtype=int)
        G = GNAR(A_RING6, p=p, s=s, ts=ts, model_type=model_type)
        X, y = format_X_y(compute_neighbour_sums(ts - G.mu, G._ns_mats, 1), p, s)
        res = np.sum(X * G.coeffs.T, axis=2) - y
        np.testing.assert_allclose(G.sigma_2, res.T @ res / (n - p), atol=1e-10)


class TestYuleWalkerUnequalStages:

    @pytest.mark.parametrize("s", [[2, 1], [1, 2], [1, 0], [0, 1]])
    def test_standard_yw_runs(self, s):
        np.random.seed(0)
        ts = np.random.normal(size=(200, 6))
        G = GNAR(A_RING6, p=2, s=np.array(s), ts=ts, model_type="standard", method="YW")
        assert G.coeffs.shape == (2 + sum(s), 6)
        # Betas shared across nodes
        np.testing.assert_allclose(G.coeffs[2:], G.coeffs[2:, :1] * np.ones((1, 6)))

    def test_standard_yw_matches_ols_large_sample(self):
        np.random.seed(1)
        G_true = GNAR(A_RING6, p=2, s=np.array([2, 1]), coeffs=_standard_coeffs(6), sigma_2=1.0)
        ts = G_true.simulate(20000)
        G_ols = GNAR(A_RING6, p=2, s=np.array([2, 1]), ts=ts, method="OLS")
        G_yw = GNAR(A_RING6, p=2, s=np.array([2, 1]), ts=ts, method="YW")
        np.testing.assert_allclose(G_yw.coeffs, G_ols.coeffs, atol=0.01)
        np.testing.assert_allclose(G_yw.coeffs, G_true.coeffs, atol=0.05)


class TestZeroStages:

    def test_neighbour_set_mats_r0(self):
        assert neighbour_set_mats(A_RING6, 0).shape == (0, 6, 6)

    @pytest.mark.parametrize("method", ["OLS", "YW"])
    @pytest.mark.parametrize("model_type", ["global", "standard", "local"])
    def test_pure_ar_fit_predict_simulate(self, method, model_type):
        np.random.seed(0)
        ts = np.random.normal(size=(100, 6))
        G = GNAR(A_RING6, p=2, s=np.array([0, 0]), ts=ts, method=method, model_type=model_type)
        assert G.coeffs.shape == (2, 6)
        assert G.predict(h=3).shape == (3, 6)
        assert G.simulate(10).shape == (10, 6)
        np.testing.assert_allclose(G.predict(ts[:10], h=3), G.to_var().predict(ts[:10], h=3), atol=1e-12)

    def test_pure_ar_matches_per_node_regression(self):
        np.random.seed(0)
        ts = np.random.normal(size=(100, 6))
        G = GNAR(A_RING6, p=1, s=np.array([0]), ts=ts, model_type="local", demean=False)
        for j in range(6):
            expected = np.linalg.lstsq(ts[:-1, [j]], ts[1:, j], rcond=None)[0][0]
            assert G.coeffs[0, j] == pytest.approx(expected)


class TestAutocovTruncation:

    @pytest.mark.parametrize("max_lag", [0, 1, 2, 4])
    def test_shape(self, max_lag):
        V = VAR(p=3, coeffs=np.vstack([0.2 * np.eye(2)] * 3))
        assert V.compute_autocov_mats(max_lag=max_lag).shape == (max_lag + 1, 2, 2)
        assert V.compute_autocorr_mats(max_lag=max_lag).shape == (max_lag + 1, 2, 2)

    def test_truncation_consistent(self):
        V = VAR(p=3, coeffs=np.vstack([0.2 * np.eye(2)] * 3))
        np.testing.assert_allclose(V.compute_autocov_mats(1), V.compute_autocov_mats(5)[:2])


class TestNumpyScalarSigma:

    @pytest.mark.parametrize("sigma", [np.int64(2), np.float32(2.0), np.array(2.0)])
    def test_numpy_scalar_sigma(self, sigma):
        G = GNAR(A_RING6, p=1, s=np.array([1]), coeffs=np.tile([[0.2], [0.1]], (1, 6)),
                 model_type="global", sigma_2=sigma)
        assert G.simulate(5).shape == (5, 6)
        assert G.simulate(5, sigma_2=sigma).shape == (5, 6)
        assert "Noise covariance" in str(G)
        V = VAR(p=1, coeffs=0.2 * np.eye(2), sigma_2=sigma)
        assert V.simulate(5).shape == (5, 2)
        assert V.compute_autocov_mats(1).shape == (2, 2, 2)


class TestValidation:

    def test_integer_valued_float_s_accepted(self):
        np.random.seed(0)
        ts = np.random.normal(size=(50, 6))
        G = GNAR(A_RING6, p=1, s=np.array([1.0]), ts=ts)
        assert np.issubdtype(G._s.dtype, np.integer)

    def test_non_integer_s_rejected(self):
        with pytest.raises(ValueError, match="integers"):
            GNAR(A_RING6, p=1, s=np.array([1.5]), ts=np.zeros((50, 6)))

    def test_non_integer_p_rejected(self):
        with pytest.raises(ValueError, match="integer"):
            GNAR(A_RING6, p=1.5, s=np.array([1]), ts=np.zeros((50, 6)))

    def test_ts_node_mismatch(self):
        with pytest.raises(ValueError, match="adjacency matrix has 6 nodes"):
            GNAR(A_RING6, p=1, s=np.array([1]), ts=np.zeros((50, 4)))

    def test_too_few_observations(self):
        with pytest.raises(ValueError, match="exceed the number of lags"):
            GNAR(A_RING6, p=3, s=np.array([1, 1, 1]), ts=np.zeros((3, 6)))
        with pytest.raises(ValueError, match="exceed the number of lags"):
            VAR(p=3, ts=np.zeros((3, 2)))

    def test_invalid_method_does_not_corrupt_model(self):
        np.random.seed(0)
        G = GNAR(A_RING6, p=1, s=np.array([1]), ts=np.random.normal(size=(50, 6)))
        coeffs = G.coeffs.copy()
        with pytest.raises(ValueError, match="Method"):
            G.fit(np.random.normal(size=(80, 6)), method="MLE")
        np.testing.assert_array_equal(G.coeffs, coeffs)
        assert G._n == 50


class TestStationarityWarning:

    def test_nonstationary_gnar_warns(self):
        with pytest.warns(UserWarning, match="non-stationary"):
            G = GNAR(A_RING6, p=1, s=np.array([1]), coeffs=np.tile([[0.9], [0.5]], (1, 6)), model_type="global")
        assert not G.is_stationary()

    def test_stationary_gnar_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            G = GNAR(A_RING6, p=2, s=np.array([2, 1]), coeffs=_standard_coeffs(6))
        assert G.is_stationary()


class TestShortBurnIn:

    @pytest.mark.parametrize("cls", ["GNAR", "VAR"])
    def test_burn_in_zero_has_no_zero_presample(self, cls):
        G = GNAR(A_RING6, p=3, s=np.array([1, 1, 1]), coeffs=np.tile([[0.2], [0.1], [0.1], [0.1], [0.05], [0.05]], (1, 6)),
                 model_type="global")
        model = G if cls == "GNAR" else G.to_var()
        np.random.seed(0)
        sim = model.simulate(5, burn_in=0)
        assert sim.shape == (5, 6)
        assert np.all(sim != 0)

    def test_gnar_var_simulations_identical(self):
        G = GNAR(A_RING6, p=2, s=np.array([2, 1]), coeffs=_standard_coeffs(6), mean=1.0)
        np.random.seed(5)
        a = G.simulate(50, burn_in=0)
        np.random.seed(5)
        b = G.to_var().simulate(50, burn_in=0)
        np.testing.assert_allclose(a, b, atol=1e-12)


class TestStandardOLSPrecision:

    def test_matches_exact_least_squares(self):
        np.random.seed(0)
        ts = np.cumsum(np.random.normal(size=(300, 6)), axis=0) * 0.3 + np.random.normal(size=(300, 6))
        p, s = 2, np.array([2, 1])
        G = GNAR(A_RING6, p=p, s=s, ts=ts, model_type="standard")
        X, y = format_X_y(compute_neighbour_sums(ts - G.mu, G._ns_mats, 2), p, s)
        n, d, _ = X.shape
        D = np.zeros((n * d, p * d))
        for i in range(d):
            D[i * n:(i + 1) * n, i::d] = X[:, i, :p]
        D = np.hstack([D, np.transpose(X[:, :, p:], (1, 0, 2)).reshape(d * n, -1)])
        exact = np.linalg.lstsq(D, y.T.reshape(-1), rcond=None)[0]
        np.testing.assert_allclose(G.coeffs[p:, 0], exact[p * d:], atol=1e-9)
        np.testing.assert_allclose(G.coeffs[:p].T.reshape(-1), exact[:p * d].reshape(p, d).T.reshape(-1), atol=1e-9)


class TestAutocovLyapunov:

    @staticmethod
    def _kron_reference(V):
        # Previous implementation: vec(Gamma) = (I - Phi (x) Phi)^{-1} vec(Sigma)
        phi, sigma_2 = V.companion_form()
        k = phi.shape[0]
        G = (np.linalg.inv(np.eye(k ** 2) - np.kron(phi, phi)) @ sigma_2.flatten()).reshape(k, k)
        return G[:V._d].T.reshape(V._p, V._d, V._d).transpose(0, 2, 1)

    @pytest.mark.parametrize("p", [1, 2, 3])
    def test_matches_kronecker_formula(self, p):
        rng = np.random.default_rng(p)
        d = 4
        coeffs = rng.normal(0, 0.15, (p * d, d))
        L = rng.normal(size=(d, d))
        V = VAR(p=p, coeffs=coeffs, sigma_2=L @ L.T + np.eye(d))
        assert V.is_stationary()
        np.testing.assert_allclose(V.compute_autocov_mats(p - 1), self._kron_reference(V), atol=1e-10)

    def test_large_var(self):
        # The Kronecker approach would need a (d^2 x d^2) matrix: 40000 x 40000 here
        d = 200
        rng = np.random.default_rng(0)
        V = VAR(p=1, coeffs=rng.normal(0, 0.3 / np.sqrt(d), (d, d)), sigma_2=1.0)
        G = V.compute_autocov_mats(2)
        phi = V.coeffs.T
        np.testing.assert_allclose(G[0], phi @ G[0] @ phi.T + np.eye(d), atol=1e-8)
        np.testing.assert_allclose(G[1], phi @ G[0], atol=1e-10)
