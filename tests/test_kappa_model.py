"""
Tests for kappa in the model specification and the simulator (pull request 1).

Acceptance tests covered here: 1 (backward compatibility, parts a and b), 2 (design matrix on a 4-node path) and
11 (stationarity checks).
"""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from gnar import GNAR, simulate_gnar1, stationary_params
from gnar.utils.data_utils import check_kappa_graph
from gnar.utils.gnar_linear_regression import design_matrix
from gnar.utils.neighbour_sets import neighbour_set_mats, kappa_weight_mat, degree_terms, node_degrees
from gnar.utils.simulating import gnar1_transition, row_sum_bound, spectral_radius, stationary_cov
from tests.data.make_legacy_reference import build, load_case
from tests.kappa_graphs import (path_graph, cycle_graph, cycle_plus_chord, star_graph, complete_bipartite, random_tree,
                                erdos_renyi, with_isolated_node)

REFERENCE = pathlib.Path(__file__).parent / "data" / "legacy_reference.npz"
_ref = np.load(REFERENCE)
_meta = json.loads(str(_ref["meta"]))

# A heterogeneous graph with an isolated node: degrees 1, 3, 2, 2, 2, 0
HETERO = with_isolated_node(np.array([[0, 1, 0, 0, 0],
                                      [1, 0, 1, 1, 0],
                                      [0, 1, 0, 1, 0],
                                      [0, 1, 1, 0, 1],
                                      [0, 0, 0, 1, 0]], dtype=float))
HETERO_ALPHA = np.array([0.1, 0.3, -0.2, 0.25, 0.15, 0.4])


def _observed(G: GNAR, ts_pred: np.ndarray, fitted: bool) -> dict:
    # Same quantities as make_legacy_reference.outputs, recomputed with the current code
    out = {
        "coeffs": np.asarray(G.coeffs, dtype=float),
        "sigma_2": np.atleast_1d(np.asarray(G.sigma_2, dtype=float)),
        "mu": np.asarray(G.mu, dtype=float),
        "predict_h3": np.asarray(G.predict(ts=ts_pred, h=3), dtype=float),
        "var_coeffs": np.asarray(G.to_var().coeffs, dtype=float),
        "num_params": np.array(G._num_params()),
        "str": np.array(str(G)),
        "repr": np.array(repr(G)),
    }
    np.random.seed(123)
    out["simulate"] = G.simulate(40, sigma_2=1.0, burn_in=10)
    if G.to_var().is_stationary():
        out["autocov"] = G.compute_autocov_mats(max_lag=3)
    if fitted:
        out["predict_last_h2"] = np.asarray(G.predict(h=2), dtype=float)
        out["bic"] = np.array(G.bic())
        out["aic"] = np.array(G.aic())
    return out


class TestBackwardCompatibility:
    """Acceptance test 1(a): with kappa = 1 (the default or passed explicitly) every existing model gives the outputs frozen from pygnar 8466a80."""

    @pytest.mark.parametrize("extra", [{}, {"kappa": 1.0}, {"kappa": 1}], ids=["default", "kappa=1.0", "kappa=1"])
    @pytest.mark.parametrize("i", range(len(_meta["specs"])))
    def test_matches_legacy(self, i, extra):
        spec = _meta["specs"][i]
        inputs, expected = load_case(_ref, i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if f"{i}/error" in _ref.files:
                # The legacy code raised on this configuration; it must still raise the same exception
                with pytest.raises(Exception) as err:
                    build(spec, inputs, **extra)
                assert type(err.value).__name__ == str(_ref[f"{i}/error"])
                return
            G = build(spec, inputs, **extra)
            observed = _observed(G, inputs["ts_pred"], fitted="ts" in inputs)
        assert set(observed) == set(expected)
        same_pandas = pd.__version__ == _meta["pandas"]
        for key, value in expected.items():
            if key in ("str", "repr"):
                if key == "str" and not same_pandas:
                    continue  # DataFrame formatting can change between pandas versions
                assert str(observed[key]) == str(value), key
            else:
                np.testing.assert_allclose(observed[key], value, rtol=1e-10, atol=1e-12, err_msg=key)


class TestNesting:
    """Acceptance test 1(b): the generic kappa code evaluated next to kappa = 1 reproduces the standard GNAR weights and fits."""

    KAPPA_NEAR_1 = float(np.nextafter(1.0, 2.0))

    @pytest.mark.parametrize("A", [path_graph(4), star_graph(5), HETERO, random_tree(12, seed=3)], ids=["path4", "star5", "hetero", "tree12"])
    def test_weights_continuous_at_one(self, A):
        legacy = neighbour_set_mats(A, 1)
        np.testing.assert_allclose(kappa_weight_mat(A, self.KAPPA_NEAR_1), legacy[0], rtol=1e-14, atol=0)
        # kappa = 1 through the kappa code uses 1 / N_i and is bit-identical to the legacy normalisation
        np.testing.assert_array_equal(kappa_weight_mat(A, 1.0), legacy[0])

    def test_fit_continuous_at_one(self):
        rng = np.random.default_rng(1)
        A = HETERO
        ts = rng.standard_normal((200, A.shape[0]))
        G_legacy = GNAR(A, p=1, s=np.array([1]), ts=ts)
        G_kappa = GNAR(A, p=1, s=np.array([1]), ts=ts, kappa=self.KAPPA_NEAR_1)
        np.testing.assert_allclose(G_kappa.coeffs, G_legacy.coeffs, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(G_kappa.sigma_2, G_legacy.sigma_2, rtol=1e-10, atol=1e-12)


class TestDegreeTerms:

    def test_isolated_nodes(self):
        mask, log_N, w = degree_terms(np.array([0.0, 1.0, 2.0, 4.0]), 0.5)
        np.testing.assert_array_equal(mask, [False, True, True, True])
        np.testing.assert_array_equal(log_N, [0.0, 0.0, np.log(2.0), np.log(4.0)])
        np.testing.assert_array_equal(w, [0.0, 1.0, np.exp(-0.5 * np.log(2.0)), np.exp(-0.5 * np.log(4.0))])

    def test_kappa_zero_is_sum(self):
        A = HETERO
        np.testing.assert_array_equal(kappa_weight_mat(A, 0.0), A)

    def test_sparse_and_int_inputs(self):
        A = path_graph(5)
        np.testing.assert_array_equal(node_degrees(csr_matrix(A)), node_degrees(A.astype(int)))
        np.testing.assert_array_equal(kappa_weight_mat(csr_matrix(A), 0.3), kappa_weight_mat(A, 0.3))


class TestDesignMatrix:
    """Acceptance test 2: design matrix on a 4-node path graph, compared with a hand computation."""

    # Path 0 - 1 - 2 - 3 with degrees (1, 2, 2, 1). The data are signed powers of two, so every product with a weight is
    # exact and each neighbour sum is rounded once, whatever the order of evaluation; the comparison can then be exact.
    TS = np.array([[1.0, -2.0, 0.5, 0.0],
                   [0.5, 1.0, -1.0, 2.0],
                   [-1.0, 0.0, 2.0, -0.5],
                   [2.0, 0.5, -0.5, 1.0]])

    @pytest.mark.parametrize("kappa", [0.3, 1.0])
    def test_hand_computation(self, kappa):
        A = path_graph(4)
        X, y = design_matrix(self.TS, A, kappa)
        # N_i^(-kappa) for the degree-2 nodes; the degree-1 nodes have weight exactly 1
        w2 = 0.5 if kappa == 1.0 else np.exp(-kappa * np.log(2.0))
        x = self.TS
        expected = np.zeros((12, 5))
        for t in range(3):
            # Node 0: own lag, and neighbour 1 with weight 1
            expected[t, 0] = x[t, 0]
            expected[t, 4] = x[t, 1]
            # Node 1: neighbours 0 and 2
            expected[3 + t, 1] = x[t, 1]
            expected[3 + t, 4] = w2 * (x[t, 0] + x[t, 2])
            # Node 2: neighbours 1 and 3
            expected[6 + t, 2] = x[t, 2]
            expected[6 + t, 4] = w2 * (x[t, 1] + x[t, 3])
            # Node 3: neighbour 2 with weight 1
            expected[9 + t, 3] = x[t, 3]
            expected[9 + t, 4] = x[t, 2]
        np.testing.assert_array_equal(X, expected)
        np.testing.assert_array_equal(y, np.concatenate([x[1:, i] for i in range(4)]))

    def test_matches_legacy_design_at_kappa_1(self):
        # At kappa = 1 the design is the one gnar_lr builds for the standard GNAR(1, [1]) model
        from gnar.utils.gnar_linear_regression import format_X_y
        from gnar.utils.neighbour_sets import compute_neighbour_sums
        A, ts = HETERO, np.random.default_rng(0).standard_normal((30, 6))
        Xn, yn = format_X_y(compute_neighbour_sums(ts, neighbour_set_mats(A, 1), 1), 1, np.array([1]))
        n, d = Xn.shape[0], Xn.shape[1]
        legacy = np.zeros([n * d, d])
        for i in range(d):
            legacy[i * n:(i + 1) * n, i::d] = Xn[:, i, :1]
        legacy = np.hstack([legacy, np.transpose(Xn[:, :, 1:], (1, 0, 2)).reshape(d * n, 1)])
        X, y = design_matrix(ts, A, 1.0)
        np.testing.assert_array_equal(X, legacy)
        np.testing.assert_array_equal(y, yn.T.reshape(-1))

    def test_isolated_node_has_no_network_regressor(self):
        X, _ = design_matrix(np.random.default_rng(0).standard_normal((10, 6)), HETERO, 0.7)
        np.testing.assert_array_equal(X[5 * 9:, 5 + 1], 0.0)


class TestTransition:

    @pytest.mark.parametrize("kappa", [0.0, 0.3, 1.0, 1.2])
    def test_matches_gnar_var_form(self, kappa):
        # The transition matrix of the new functions agrees with the VAR form of a GNAR object using the same kappa
        beta = 0.15
        coeffs = np.vstack([HETERO_ALPHA, np.full(6, beta)])
        G = GNAR(HETERO, p=1, s=np.array([1]), coeffs=coeffs, kappa=kappa)
        Phi = gnar1_transition(HETERO, HETERO_ALPHA, beta, kappa).toarray()
        np.testing.assert_allclose(Phi, G.to_var().coeffs.T, rtol=1e-15, atol=0)
        # Phi[i, q] = alpha_i 1{i = q} + beta N_i^(-kappa) A[i, q]
        N = HETERO.sum(axis=1)
        expected = np.diag(HETERO_ALPHA) + beta * np.where(N > 0, N, 1.0)[:, None] ** (-kappa) * HETERO
        np.testing.assert_allclose(Phi, expected, rtol=1e-14, atol=0)

    @pytest.mark.parametrize("kappa", [0.3, 1.0])
    def test_stationary_cov_matches_gnar_autocov(self, kappa):
        beta = 0.15
        coeffs = np.vstack([HETERO_ALPHA, np.full(6, beta)])
        G = GNAR(HETERO, p=1, s=np.array([1]), coeffs=coeffs, kappa=kappa)
        Gamma0 = stationary_cov(HETERO, HETERO_ALPHA, beta, kappa, sigma_2=2.0)
        np.testing.assert_allclose(Gamma0, 2.0 * G.compute_autocov_mats(0)[0], rtol=1e-10, atol=1e-12)
        Phi = gnar1_transition(HETERO, HETERO_ALPHA, beta, kappa).toarray()
        np.testing.assert_allclose(Gamma0, Phi @ Gamma0 @ Phi.T + 2.0 * np.eye(6), rtol=1e-10, atol=1e-12)

    def test_row_sum_bound_hand_computed(self):
        # Degrees (1, 3, 2, 2, 2, 0); rho = max_i |alpha_i| + |beta| N_i^(1 - kappa)
        rho = row_sum_bound(HETERO, HETERO_ALPHA, -0.2, 0.5)
        N = np.array([1, 3, 2, 2, 2, 0.0])
        expected = np.max(np.abs(HETERO_ALPHA) + 0.2 * np.where(N > 0, np.sqrt(N), 0.0))
        assert rho == pytest.approx(expected, rel=1e-14)

    @pytest.mark.parametrize("A", [HETERO, star_graph(6), erdos_renyi(15, 0.3, seed=2)], ids=["hetero", "star6", "er15"])
    @pytest.mark.parametrize("kappa", [0.0, 0.6, 1.3])
    def test_spectral_radius(self, A, kappa):
        d = A.shape[0]
        alpha = np.linspace(-0.3, 0.4, d)
        Phi = gnar1_transition(A, alpha, 0.2, kappa).toarray()
        assert spectral_radius(A, alpha, 0.2, kappa) == pytest.approx(np.max(np.abs(np.linalg.eigvals(Phi))), rel=1e-10)

    def test_spectral_radius_sparse_solver_regular(self):
        # Above the dense threshold a sparse symmetric eigensolver is used. On a regular graph the vector of ones is an
        # eigenvector, which must not stall the solver. Phi = 0.1 I + 0.3 * 2^(-kappa) A, with the eigenvalues of A in [-2, 2]
        from scipy.sparse import diags
        d = 2400
        A = diags([np.ones(d - 1), np.ones(d - 1), [1.0], [1.0]], [1, -1, d - 1, -(d - 1)], format="csr")
        assert spectral_radius(A, 0.1, 0.3, 0.5) == pytest.approx(0.1 + 0.3 * 2 ** -0.5 * 2, rel=1e-8)
        # With a negative alpha the most negative eigenvalue, -0.1 - 0.3 * 2^(-kappa) * 2, has the largest magnitude
        assert spectral_radius(A, -0.1, 0.3, 0.5) == pytest.approx(0.1 + 0.3 * 2 ** -0.5 * 2, rel=1e-8)

    def test_spectral_radius_sparse_solver_bipartite(self):
        # K_{m,n}: D^(-kappa/2) A D^(-kappa/2) has eigenvalues +-(mn)^((1 - kappa)/2) and 0, so with alpha = -0.1 the
        # spectral radius is 0.1 + beta (mn)^((1 - kappa)/2), attained by the most negative eigenvalue
        m, n, beta, kappa = 40, 2000, 0.02, 0.5
        A = csr_matrix(complete_bipartite(m, n))
        expected = 0.1 + beta * (m * n) ** ((1 - kappa) / 2)
        assert spectral_radius(A, -0.1, beta, kappa) == pytest.approx(expected, rel=1e-8)


class TestStationarity:
    """Acceptance test 11: the simulator raises when the spectral radius is at least 1, and stationary_params gives rho < 1."""

    def test_simulator_raises_when_not_stationary(self):
        A = star_graph(4)
        with pytest.raises(ValueError, match="not stationary") as err:
            simulate_gnar1(A, 0.5, 0.4, 0.0, n=10, rng=0)
        # The message reports the spectral radius and rho
        assert "rho" in str(err.value) and "spectral radius" in str(err.value)
        with pytest.raises(ValueError, match="not stationary"):
            stationary_cov(A, 0.5, 0.4, 0.0)

    def test_unit_root_raises(self):
        # alpha = 1 on an isolated node is a unit root
        A = with_isolated_node(path_graph(3))
        with pytest.raises(ValueError, match="not stationary"):
            simulate_gnar1(A, np.array([0.1, 0.1, 0.1, 1.0]), 0.1, 1.0, n=10, rng=0)

    def test_stationary_although_rho_above_one(self):
        # rho is only sufficient: on a star with kappa = 0, rho = 0.1 + 0.2 * 9 = 1.9 but the spectral radius is
        # 0.1 + 0.2 * 3 = 0.7, so the simulator must not raise
        A = star_graph(9)
        assert row_sum_bound(A, 0.1, 0.2, 0.0) == pytest.approx(1.9)
        assert spectral_radius(A, 0.1, 0.2, 0.0) == pytest.approx(0.7)
        X = simulate_gnar1(A, 0.1, 0.2, 0.0, n=50, rng=0)
        assert np.all(np.isfinite(X))

    @pytest.mark.parametrize("graph", ["er", "star", "cycle_chord", "bipartite", "tree", "hetero"])
    @pytest.mark.parametrize("kappa", [0.0, 0.3, 0.7, 1.0, 1.2, 1.5])
    def test_stationary_params_rho_below_one(self, graph, kappa):
        A = {"er": erdos_renyi(20, 0.3, seed=5), "star": star_graph(8), "cycle_chord": cycle_plus_chord(9),
             "bipartite": complete_bipartite(2, 6), "tree": random_tree(10, seed=4), "hetero": HETERO}[graph]
        rng = np.random.default_rng(11)
        alpha = rng.uniform(0.1, 0.4, A.shape[0])
        beta = stationary_params(A, kappa, 0.4, alpha)
        rho = row_sum_bound(A, alpha, beta, kappa)
        assert rho < 1
        # The most heavily weighted node has network weight exactly b
        N = node_degrees(A)
        assert np.max(abs(beta) * N[N > 0] ** (1 - kappa)) == pytest.approx(0.4, rel=1e-12)
        assert rho <= np.max(alpha) + 0.4 + 1e-12
        assert spectral_radius(A, alpha, beta, kappa) <= rho + 1e-12

    def test_stationary_params_checks(self):
        A = path_graph(3)
        with pytest.raises(ValueError, match="below 1"):
            stationary_params(A, 0.5, 0.7, np.array([0.1, 0.3, 0.2]))
        with pytest.raises(ValueError, match="no edges"):
            stationary_params(np.zeros((3, 3)), 0.5, 0.4, 0.1)
        # The sign of b gives the sign of beta
        assert stationary_params(A, 0.5, -0.4, 0.1) == pytest.approx(-stationary_params(A, 0.5, 0.4, 0.1))


class TestSimulateGnar1:

    ALPHA = HETERO_ALPHA
    BETA = 0.2
    KAPPA = 0.6

    def test_shape_and_reproducibility(self):
        X1 = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=30, rng=7)
        X2 = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=30, rng=np.random.default_rng(7))
        assert X1.shape == (30, 6)
        np.testing.assert_array_equal(X1, X2)
        assert not np.allclose(X1, simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=30, rng=8))

    def test_global_seed_controls_default(self):
        # rng=None uses NumPy's global random state, like GNAR.simulate
        np.random.seed(3)
        X1 = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=20)
        np.random.seed(3)
        X2 = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=20)
        np.testing.assert_array_equal(X1, X2)

    def test_recursion(self):
        # X_t = Phi X_{t-1} + u_t with the innovations drawn after X_0
        sigma_2 = 1.7
        X = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=25, sigma_2=sigma_2, rng=5)
        gen = np.random.default_rng(5)
        L = np.linalg.cholesky(stationary_cov(HETERO, self.ALPHA, self.BETA, self.KAPPA, sigma_2))
        x0 = L @ gen.standard_normal(6)
        u = np.sqrt(sigma_2) * gen.standard_normal((24, 6))
        Phi = gnar1_transition(HETERO, self.ALPHA, self.BETA, self.KAPPA).toarray()
        expected = np.zeros((25, 6))
        expected[0] = x0
        for t in range(1, 25):
            expected[t] = Phi @ expected[t - 1] + u[t - 1]
        np.testing.assert_allclose(X, expected, rtol=1e-12, atol=1e-14)

    def test_zero_start_and_burn_in(self):
        X = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=10, start="zero", rng=1)
        np.testing.assert_array_equal(X[0], 0.0)
        Xb = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=10, start="zero", burn_in=5, rng=1)
        Xf = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=15, start="zero", rng=1)
        np.testing.assert_array_equal(Xb, Xf[5:])

    def test_sparse_adjacency(self):
        X_dense = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=20, rng=2)
        X_sparse = simulate_gnar1(csr_matrix(HETERO), self.ALPHA, self.BETA, self.KAPPA, n=20, rng=2)
        np.testing.assert_allclose(X_sparse, X_dense, rtol=1e-14, atol=1e-15)

    def test_stationary_start_distribution(self):
        # X_0 ~ N(0, Gamma0): the Monte Carlo covariance of independent starting values matches Gamma0
        Gamma0 = stationary_cov(HETERO, self.ALPHA, self.BETA, self.KAPPA)
        gen = np.random.default_rng(12)
        X0 = np.array([simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=1, rng=gen)[0] for _ in range(3000)])
        np.testing.assert_allclose(X0.T @ X0 / len(X0), Gamma0, atol=0.12)

    def test_long_run_autocovariance(self):
        Gamma0 = stationary_cov(HETERO, self.ALPHA, self.BETA, self.KAPPA)
        Phi = gnar1_transition(HETERO, self.ALPHA, self.BETA, self.KAPPA).toarray()
        X = simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=40000, rng=4)
        np.testing.assert_allclose(X.T @ X / len(X), Gamma0, atol=0.06)
        # Lag-1 autocovariance E[X_t X_{t-1}^T] = Phi Gamma0
        np.testing.assert_allclose(X[1:].T @ X[:-1] / (len(X) - 1), Phi @ Gamma0, atol=0.06)

    def test_gnar_simulate_uses_kappa(self):
        # GNAR.simulate with kappa != 1 follows X_t = Phi X_{t-1} + e_t from a zero start, as for kappa = 1
        coeffs = np.vstack([self.ALPHA, np.full(6, self.BETA)])
        G = GNAR(HETERO, p=1, s=np.array([1]), coeffs=coeffs, kappa=self.KAPPA)
        np.random.seed(9)
        sim = G.simulate(30, burn_in=5)
        np.random.seed(9)
        e = np.random.normal(0, 1, (35, 6))
        Phi = gnar1_transition(HETERO, self.ALPHA, self.BETA, self.KAPPA).toarray()
        expected = np.zeros((35, 6))
        for t in range(1, 35):
            expected[t] = Phi @ expected[t - 1] + e[t]
        np.testing.assert_allclose(sim, expected[5:], rtol=1e-12, atol=1e-14)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError, match="start"):
            simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=5, start="bad")
        with pytest.raises(ValueError, match="n must"):
            simulate_gnar1(HETERO, self.ALPHA, self.BETA, self.KAPPA, n=0)
        with pytest.raises(ValueError, match="kappa"):
            simulate_gnar1(HETERO, self.ALPHA, self.BETA, None, n=5)


class TestKappaValidation:

    COEFFS = np.array([[0.2, 0.1, 0.3], [0.1, 0.1, 0.1]])
    A = path_graph(3)

    def test_kappa_float_model(self):
        G = GNAR(self.A, p=1, s=np.array([1]), coeffs=self.COEFFS, kappa=0)
        assert G.kappa == 0.0
        np.testing.assert_array_equal(G._ns_mats[0], self.A)
        assert "kappa=0" in repr(G) and "kappa: 0 (fixed)" in str(G)

    def test_kappa_one_display_unchanged(self):
        G = GNAR(self.A, p=1, s=np.array([1]), coeffs=self.COEFFS, kappa=1.0)
        assert "kappa" not in repr(G) and "kappa" not in str(G)

    def test_fit_with_fixed_kappa(self):
        # With kappa fixed the existing OLS and Yule-Walker estimators run on the kappa-weighted neighbour sums
        ts = np.random.default_rng(0).standard_normal((100, 6))
        for method in ["OLS", "YW"]:
            G = GNAR(HETERO, p=1, s=np.array([1]), ts=ts, kappa=0.4, method=method)
            assert G.coeffs.shape == (2, 6)
            assert G._num_params() == 7

    @pytest.mark.parametrize("kwargs, error, match", [
        (dict(p=2, s=np.array([1, 1])), NotImplementedError, "GNAR\\(1, \\[1\\]\\)"),
        (dict(p=1, s=np.array([2])), NotImplementedError, "GNAR\\(1, \\[1\\]\\)"),
        (dict(p=1, s=np.array([1]), model_type="local"), NotImplementedError, "local"),
        (dict(p=1, s=np.array([1]), net_type="weighted"), NotImplementedError, "unweighted"),
    ])
    def test_unsupported_models(self, kwargs, error, match):
        p = kwargs.pop("p")
        s = kwargs.pop("s")
        coeffs = np.zeros((p + int(np.sum(s)), 3))
        with pytest.raises(error, match=match):
            GNAR(self.A, p=p, s=s, coeffs=coeffs, kappa=0.5, **kwargs)

    def test_global_fixed_kappa_allowed(self):
        coeffs = np.array([[0.2, 0.2, 0.2], [0.1, 0.1, 0.1]])
        G = GNAR(self.A, p=1, s=np.array([1]), coeffs=coeffs, model_type="global", kappa=0.5)
        assert G.kappa == 0.5

    @pytest.mark.parametrize("kappa", [np.nan, np.inf, "1", True])
    def test_invalid_kappa(self, kappa):
        with pytest.raises(ValueError, match="kappa"):
            GNAR(self.A, p=1, s=np.array([1]), coeffs=self.COEFFS, kappa=kappa)

    def test_graph_checks(self):
        directed = np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        with pytest.raises(ValueError, match="symmetric"):
            GNAR(directed, p=1, s=np.array([1]), coeffs=self.COEFFS, kappa=0.5)
        loop = self.A + np.diag([1.0, 0, 0])
        with pytest.raises(ValueError, match="self-loops"):
            GNAR(loop, p=1, s=np.array([1]), coeffs=self.COEFFS, kappa=0.5)
        with pytest.raises(NotImplementedError, match="Weighted"):
            check_kappa_graph(2 * self.A)
        # Directed graphs and self-loops are still accepted with kappa = 1, as before
        GNAR(directed, p=1, s=np.array([1]), coeffs=self.COEFFS)

    def test_estimated_kappa_not_available_yet(self):
        ts = np.random.default_rng(0).standard_normal((50, 3))
        with pytest.raises(NotImplementedError, match="kappa=None"):
            GNAR(self.A, p=1, s=np.array([1]), ts=ts, kappa=None)
