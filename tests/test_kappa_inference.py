"""
Tests for the Jacobian covariance, intervals, the test of kappa = 1, beta at a reference degree and the theory utilities
(pull request 3).

Acceptance tests covered here: 5 (Jacobian vs finite differences), 6 (fast covariance vs dense inverse), 7 (recovery),
8 (calibration, slow), 9 (variance bounds), and part (c) of test 1 for the standard errors.
"""
import json
import pathlib
import warnings

import numpy as np
import pytest
from scipy.stats import chi2, norm

from gnar import (simulate_gnar1, stationary_params, fit_gnar1, asymptotic_cov, variance_bounds, closed_form_variances,
                  KappaWarning, KappaIdentifiabilityWarning)
from gnar.utils.gnar_inference import jacobian, cov_dense, cov_fast, var_fast, beta_at_reference
from gnar.utils.gnar_linear_regression import design_matrix, format_X_y
from gnar.utils.gnar_profile_likelihood import fixed_kappa_ols, deviance
from gnar.utils.neighbour_sets import neighbour_set_mats, compute_neighbour_sums, degree_terms, node_degrees
from tests.data.make_legacy_reference import load_case
from tests.kappa_graphs import (path_graph, cycle_graph, cycle_plus_chord, star_graph, complete_bipartite, random_tree,
                                erdos_renyi, with_isolated_node)

HETERO = with_isolated_node(np.array([[0, 1, 0, 0, 0],
                                      [1, 0, 1, 1, 0],
                                      [0, 1, 0, 1, 0],
                                      [0, 1, 1, 0, 1],
                                      [0, 0, 0, 1, 0]], dtype=float))


def draw_params(A, kappa, rng, b=0.4):
    """The spec's defaults: alpha_i ~ U[0.1, 0.4] and beta from stationary_params with b = 0.4."""
    alpha = rng.uniform(0.1, 0.4, A.shape[0])
    return alpha, stationary_params(A, kappa, b, alpha)


def simulate(A, kappa, n, seed):
    rng = np.random.default_rng(seed)
    alpha, beta = draw_params(A, kappa, rng)
    return simulate_gnar1(A, alpha, beta, kappa, n, rng=rng), alpha, beta


GRAPHS = {"hetero": HETERO, "tree": random_tree(10, seed=1), "er": erdos_renyi(15, 0.3, seed=2), "star": star_graph(6)}


class TestJacobian:
    """Acceptance test 5: the analytic Jacobian agrees with finite differences."""

    @pytest.mark.parametrize("graph", list(GRAPHS))
    def test_finite_differences(self, graph):
        A = GRAPHS[graph]
        X, _, _ = simulate(A, 0.7, 200, seed=3)
        fit = fit_gnar1(A, X, demean=False)
        d = A.shape[0]
        theta = np.concatenate([fit.alpha, [fit.beta, fit.kappa]])

        def mu(t):
            D, _ = design_matrix(X, A, t[-1])
            return D @ t[:-1]

        J = jacobian(X, A, fit.beta, fit.kappa)
        J_fd = np.zeros_like(J)
        for j in range(d + 2):
            h = 1e-6 * max(1.0, abs(theta[j]))
            up, down = theta.copy(), theta.copy()
            up[j] += h
            down[j] -= h
            J_fd[:, j] = (mu(up) - mu(down)) / (2 * h)
        assert np.linalg.norm(J - J_fd) / np.linalg.norm(J) < 1e-6
        assert np.max(np.abs(J - J_fd)) < 1e-6 * np.max(np.abs(J))
        # The kappa column vanishes for isolated nodes and for nodes of degree 1 (log N_i = 0)
        n = X.shape[0] - 1
        for i in np.flatnonzero(node_degrees(A) <= 1):
            np.testing.assert_array_equal(J[i * n:(i + 1) * n, d + 1], 0.0)


class TestFastCovariance:
    """Acceptance test 6: the fast covariance path agrees with the dense inverse sigma^2 (J^T J)^(-1)."""

    @pytest.mark.parametrize("graph", list(GRAPHS))
    def test_kappa_estimated(self, graph):
        A = GRAPHS[graph]
        X, _, _ = simulate(A, 0.8, 500, seed=4)
        fit = fit_gnar1(A, X, demean=False)
        dense = cov_dense(jacobian(X, A, fit.beta, fit.kappa), fit.sigma_2)
        fast = cov_fast(fit._M, fit._R, fit.degrees, fit.beta, fit.kappa, fit.sigma_2)
        np.testing.assert_allclose(fast, dense, rtol=1e-8, atol=1e-12 * np.max(np.abs(dense)))
        np.testing.assert_allclose(fit.cov, dense, rtol=1e-8, atol=1e-12 * np.max(np.abs(dense)))
        np.testing.assert_allclose(fit.se ** 2, np.diag(dense), rtol=1e-8)
        # The O(d) diagonal and the closed-form (beta, kappa) block agree with the full matrix
        np.testing.assert_allclose(var_fast(fit._M, fit._R, fit.degrees, fit.beta, fit.kappa, fit.sigma_2), np.diag(fast), rtol=1e-12)
        np.testing.assert_allclose(fit._cov_beta_kappa, fast[-2:, -2:], rtol=1e-10)

    @pytest.mark.parametrize("graph", list(GRAPHS))
    def test_kappa_fixed(self, graph):
        # With kappa fixed the standard errors are the usual OLS ones, treating kappa as known
        A = GRAPHS[graph]
        X, _, _ = simulate(A, 0.8, 500, seed=5)
        fit = fit_gnar1(A, X, kappa=0.8, demean=False)
        D, _ = design_matrix(X, A, 0.8)
        dense = cov_dense(D, fit.sigma_2)
        np.testing.assert_allclose(fit.cov[:-1, :-1], dense, rtol=1e-8, atol=1e-12 * np.max(np.abs(dense)))
        assert np.all(np.isnan(fit.cov[-1])) and np.isnan(fit.se[-1])


class TestLegacyStandardErrors:
    """Acceptance test 1(c), standard errors: with kappa fixed at 1 they equal the classical OLS standard errors
    sqrt(diag(s^2 (X^T X)^(-1))) of the legacy stacked design, with s^2 = RSS / (N(T - 1) - N - 1)."""

    _ref = np.load(pathlib.Path(__file__).parent / "data" / "legacy_reference.npz")
    _meta = json.loads(str(_ref["meta"]))
    CASES = [i for i, spec in enumerate(_meta["specs"])
             if spec.get("method") == "OLS" and spec["model_type"] == "standard" and spec["p"] == 1 and spec["s"] == [1]
             and spec["net_type"] == "unweighted"]

    @pytest.mark.parametrize("i", CASES)
    def test_matches_classical_ols(self, i):
        spec = self._meta["specs"][i]
        inputs, _ = load_case(self._ref, i)
        A, ts = inputs["A"], inputs["ts"]
        x = ts - ts.mean(axis=0) if spec["demean"] else ts
        # The stacked design of gnar_lr for the standard GNAR(1, [1]) model
        Xn, yn = format_X_y(compute_neighbour_sums(x, neighbour_set_mats(A, 1), 1), 1, np.array([1]))
        n, d = Xn.shape[0], Xn.shape[1]
        X = np.zeros([n * d, d])
        for j in range(d):
            X[j * n:(j + 1) * n, j] = Xn[:, j, 0]
        X = np.hstack([X, np.transpose(Xn[:, :, 1:], (1, 0, 2)).reshape(d * n, 1)])
        y = yn.T.reshape(-1)
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        s2 = np.sum((y - X @ coef) ** 2) / (d * n - d - 1)
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        fit = fit_gnar1(A, ts, kappa=1.0, demean=spec["demean"])
        np.testing.assert_allclose(fit.se[:d + 1], se, rtol=1e-10)
        assert fit.sigma_2 == pytest.approx(s2, rel=1e-10)


class TestRecovery:
    """Acceptance test 7: on an Erdos-Renyi graph (N = 20, p = 0.3) with T = 20 000, kappa_hat is within 4 SE of kappa."""

    @pytest.mark.parametrize("kappa", [0.3, 0.7, 1.2])
    def test_recovery(self, kappa):
        A = erdos_renyi(20, 0.3, seed=31)
        degrees = node_degrees(A)
        assert np.all(degrees > 0) and np.unique(degrees).size >= 2
        X, alpha, beta = simulate(A, kappa, 20000, seed=32)
        fit = fit_gnar1(A, X, demean=False)
        se_kappa = fit.se[-1]
        assert abs(fit.kappa - kappa) < 4 * se_kappa
        assert abs(fit.beta - beta) < 4 * fit.se[-2]
        assert fit.warnings == []


CALIBRATION = dict(reps=500, T=2000, kappa=0.7)


@pytest.fixture(scope="module")
def replicates():
    """500 replicates on a random tree with N = 10 and T = 2000 (acceptance test 8)."""
    reps, T, kappa = CALIBRATION["reps"], CALIBRATION["T"], CALIBRATION["kappa"]
    A = random_tree(10, seed=41)
    assert np.unique(node_degrees(A)).size >= 2
    alpha, beta = draw_params(A, kappa, np.random.default_rng(42))
    theta = np.concatenate([alpha, [beta, kappa]])
    est, se, prof, bound_hits = [], [], [], 0
    for seed in np.random.SeedSequence(43).spawn(reps):
        X = simulate_gnar1(A, alpha, beta, kappa, T, rng=np.random.default_rng(seed))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", KappaWarning)
            fit = fit_gnar1(A, X, demean=False)
        bound_hits += any(code == "boundary" for code, _ in fit.warnings)
        est.append(np.concatenate([fit.alpha, [fit.beta, fit.kappa]]))
        se.append(fit.se)
        interval = fit.profile_interval(0.95)
        prof.append(interval["lower"] <= kappa <= interval["upper"])
    return dict(A=A, alpha=alpha, beta=beta, theta=theta, est=np.array(est), se=np.array(se), prof=np.array(prof), bound_hits=bound_hits)


@pytest.mark.slow
class TestCalibration:
    """
    Acceptance test 8: on a random tree with N = 10 and T = 2000, over 500 replicates, the Monte Carlo SD of kappa_hat,
    beta_hat and each alpha_hat_i is within 15% of the mean SE and of the asymptotic_cov value, and the 95% Wald and profile
    intervals cover in [0.92, 0.98] (profile for kappa; Wald for kappa, beta and each alpha_i).
    """

    def test_no_boundary_estimates(self, replicates):
        assert replicates["bound_hits"] == 0

    def test_sd_matches_se_and_theory(self, replicates):
        sd = replicates["est"].std(axis=0, ddof=1)
        mean_se = replicates["se"].mean(axis=0)
        asym = np.sqrt(np.diag(asymptotic_cov(replicates["A"], replicates["alpha"], replicates["beta"], CALIBRATION["kappa"], n=CALIBRATION["T"])))
        np.testing.assert_array_less(np.abs(sd / mean_se - 1), 0.15)
        np.testing.assert_array_less(np.abs(sd / asym - 1), 0.15)

    def test_coverage(self, replicates):
        z = norm.ppf(0.975)
        covered = np.abs(replicates["est"] - replicates["theta"]) <= z * replicates["se"]
        wald = covered.mean(axis=0)
        assert np.all((wald >= 0.92) & (wald <= 0.98)), wald
        assert 0.92 <= replicates["prof"].mean() <= 0.98, replicates["prof"].mean()


BOUND_GRAPHS = {
    "er": erdos_renyi(15, 0.3, seed=51),
    "star": star_graph(8),
    "cycle_chord": cycle_plus_chord(10),
    "bipartite": complete_bipartite(3, 6),
    "tree": random_tree(12, seed=52),
}


class TestVarianceBounds:
    """Acceptance test 9: for 50 random stationary parameter sets on each of 5 graph types, the exact asymptotic variances lie inside variance_bounds."""

    @pytest.mark.parametrize("graph", list(BOUND_GRAPHS))
    def test_bounds(self, graph):
        A = BOUND_GRAPHS[graph]
        d = A.shape[0]
        rng = np.random.default_rng(53)
        slack = 1e-9
        for _ in range(50):
            kappa = rng.uniform(0.0, 1.5)
            alpha, beta = draw_params(A, kappa, rng)
            var = np.diag(asymptotic_cov(A, alpha, beta, kappa))
            bounds = variance_bounds(A, alpha, beta, kappa)
            for name, v in (("kappa", var[d + 1]), ("beta", var[d])):
                b = bounds[name]
                assert b["lower_rho"] <= b["lower"] * (1 + slack)
                assert b["lower"] * (1 - slack) <= v <= b["upper"] * (1 + slack), (name, b, v)
            a = bounds["alpha"]
            assert np.all(a["lower_rho"] <= a["lower"] * (1 + slack))
            assert np.all(a["lower"] * (1 - slack) <= var[:d]) and np.all(var[:d] <= a["upper"] * (1 + slack))
            assert bounds["gamma_bar"] <= 1 / (1 - bounds["rho"] ** 2) * (1 + slack)
            # The closed forms agree with the inverse of the information matrix
            closed = closed_form_variances(A, alpha, beta, kappa)
            assert closed["kappa"] == pytest.approx(var[d + 1], rel=1e-10)
            assert closed["beta"] == pytest.approx(var[d], rel=1e-10)

    def test_isolated_node_attains_alpha_lower_bound(self):
        rng = np.random.default_rng(54)
        alpha, beta = draw_params(HETERO, 0.5, rng)
        var = np.diag(asymptotic_cov(HETERO, alpha, beta, 0.5))
        bounds = variance_bounds(HETERO, alpha, beta, 0.5)
        # The isolated node's alpha is a plain AR(1): its variance equals 1 / Gamma0_ii
        assert var[5] == pytest.approx(bounds["alpha"]["lower"][5], rel=1e-10)

    def test_rho_bound_unavailable(self):
        # rho >= 1 although the process is stationary: the 1 / (1 - rho^2) bounds are not available
        with pytest.warns(UserWarning, match="rho"):
            bounds = variance_bounds(star_graph(9), 0.1, 0.2, 0.0)
        assert np.isnan(bounds["kappa"]["lower_rho"]) and np.isfinite(bounds["kappa"]["lower"])


class TestAsymptoticCov:

    def test_invariant_to_sigma_2_and_scaled_by_T(self):
        rng = np.random.default_rng(61)
        A = random_tree(8, seed=62)
        alpha, beta = draw_params(A, 0.6, rng)
        per_period = asymptotic_cov(A, alpha, beta, 0.6)
        np.testing.assert_allclose(asymptotic_cov(A, alpha, beta, 0.6, sigma_2=4.0), per_period, rtol=1e-9)
        np.testing.assert_allclose(asymptotic_cov(A, alpha, beta, 0.6, n=1001), per_period / 1000, rtol=1e-12)

    def test_matches_large_sample_fit(self):
        # T times the estimated covariance converges to the per-period asymptotic covariance
        A = random_tree(8, seed=62)
        X, alpha, beta = simulate(A, 0.6, 100000, seed=63)
        fit = fit_gnar1(A, X, demean=False)
        per_period = asymptotic_cov(A, alpha, beta, 0.6)
        np.testing.assert_allclose(fit.se ** 2 * (fit.n - 1), np.diag(per_period), rtol=0.05)

    def test_regular_graph(self):
        with pytest.warns(UserWarning, match="not identified"):
            cov = asymptotic_cov(cycle_graph(6), 0.2, 0.3, 0.5)
        assert np.isinf(cov[-1, -1]) and np.isinf(cov[-2, -2]) and np.all(np.isfinite(np.diag(cov)[:6]))
        with pytest.warns(UserWarning, match="not identified"):
            assert np.isinf(closed_form_variances(cycle_graph(6), 0.2, 0.3, 0.5)["kappa"])


@pytest.fixture(scope="module")
def fit():
    A = random_tree(10, seed=71)
    X, _, _ = simulate(A, 0.7, 2000, seed=72)
    return fit_gnar1(A, X, demean=False)


class TestIntervalsAndTests:

    def test_ci_table(self, fit):
        table = fit.ci()
        d = fit.d
        assert list(table.index) == [f"alpha[{j}]" for j in range(1, d + 1)] + ["beta", "beta_0", "kappa"]
        assert table.loc["kappa", "method"] == "profile" and table.loc["beta", "method"] == "wald"
        z = norm.ppf(0.975)
        assert table.loc["beta", "lower"] == pytest.approx(fit.beta - z * fit.se[d])
        wald = fit.ci(method="wald")
        assert wald.loc["kappa", "upper"] == pytest.approx(fit.kappa + z * fit.se[d + 1])
        # The profile interval's ends are where the deviance crosses the chi-square(1) cutoff
        cutoff = chi2.ppf(0.95, 1)
        for end in ("lower", "upper"):
            k = table.loc["kappa", end]
            dev = deviance(fixed_kappa_ols(fit._M, fit.degrees, k, fit._R)[2], fit.rss, fit.n_obs)
            assert dev == pytest.approx(cutoff, abs=1e-8)
        # With a well-identified kappa the profile and Wald intervals are close
        assert table.loc["kappa", "lower"] == pytest.approx(wald.loc["kappa", "lower"], abs=0.2 * fit.se[d + 1])

    def test_open_profile_interval(self):
        A = random_tree(10, seed=71)
        X, _, _ = simulate(A, 0.7, 2000, seed=72)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", KappaWarning)
            narrow = fit_gnar1(A, X, kappa_bounds=(0.65, 0.75), demean=False)
        interval = narrow.profile_interval()
        assert interval["lower_open"] and interval["upper_open"]
        assert (interval["lower"], interval["upper"]) == (0.65, 0.75)
        assert narrow.ci().loc["kappa", "note"] == "open below, open above"

    def test_kappa_test(self, fit):
        test = fit.test_kappa()
        se_kappa = fit.se[-1]
        assert test["z"] == pytest.approx((fit.kappa - 1) / se_kappa)
        assert test["p_value"] == pytest.approx(2 * norm.sf(abs(test["z"])))
        rss_1 = fit_gnar1(fit._A, fit._ts, kappa=1.0, demean=False).rss
        assert test["lr"] == pytest.approx(fit.n_obs * np.log(rss_1 / fit.rss))
        assert test["lr_p_value"] == pytest.approx(chi2.sf(test["lr"], 1))
        # Wald and likelihood-ratio statistics agree to first order
        assert test["lr"] == pytest.approx(test["z"] ** 2, rel=0.2)

    def test_fixed_kappa_inference(self):
        X, _, _ = simulate(HETERO, 0.5, 300, seed=73)
        fixed = fit_gnar1(HETERO, X, kappa=0.5, demean=False)
        with pytest.raises(ValueError, match="fixed"):
            fixed.test_kappa()
        with pytest.raises(ValueError, match="fixed"):
            fixed.profile_interval()
        table = fixed.ci()
        assert table.loc["kappa", "method"] == "fixed" and np.isnan(table.loc["kappa", "lower"])

    def test_beta_at(self, fit):
        connected = fit.degrees[fit.degrees > 0]
        ref = fit.beta_at()
        assert ref["N0"] == pytest.approx(np.exp(np.mean(np.log(connected))))
        assert ref["beta_0"] == pytest.approx(fit.beta * ref["N0"] ** -fit.kappa)
        # The gradient form equals the spec's delta-method formula
        V = fit.cov[-2:, -2:]
        L = np.log(ref["N0"])
        spec_var = ref["beta_0"] ** 2 * (V[0, 0] / fit.beta ** 2 - 2 * L * V[0, 1] / fit.beta + L ** 2 * V[1, 1])
        assert ref["se"] ** 2 == pytest.approx(spec_var, rel=1e-10)
        # At the w-weighted mean log degree, beta_0 is uncorrelated with kappa_hat
        _, log_N, w = degree_terms(fit.degrees, fit.kappa)
        wi = w * w * fit._R.d
        N_star = np.exp(np.sum(wi * log_N) / np.sum(wi))
        g = N_star ** -fit.kappa * np.array([1.0, -fit.beta * np.log(N_star)])
        assert abs(g @ V @ np.array([0.0, 1.0])) < 1e-10 * np.sqrt(V[0, 0] * V[1, 1])
        assert fit.summary().loc["beta_0", "estimate"] == pytest.approx(ref["beta_0"])

    def test_beta_at_fixed_kappa(self):
        X, _, _ = simulate(HETERO, 0.5, 300, seed=74)
        fixed = fit_gnar1(HETERO, X, kappa=0.5, demean=False)
        ref = fixed.beta_at(N0=2.0)
        assert ref["se"] == pytest.approx(2.0 ** -0.5 * fixed.se[-2])

    def test_regular_graph_inference(self):
        A = cycle_graph(8)
        X = simulate_gnar1(A, 0.2, 0.3, 1.0, 500, rng=75)
        with pytest.warns(KappaIdentifiabilityWarning):
            fit = fit_gnar1(A, X, demean=False)
        d = fit.d
        assert np.isinf(fit.se[d]) and np.isinf(fit.se[d + 1]) and np.all(np.isfinite(fit.se[:d]))
        # beta N^(-kappa) at the common degree 2 is identified
        ref = fit.beta_at()
        fixed = fit_gnar1(A, X, kappa=1.0, demean=False)
        assert ref["N0"] == pytest.approx(2.0)
        assert ref["beta_0"] == pytest.approx(fixed.beta / 2) and ref["se"] == pytest.approx(fixed.se[d] / 2)
        assert np.isnan(fit.test_kappa()["z"])
        assert fit.ci().loc["kappa", "note"] == "not identified"

    def test_str_report(self, fit):
        text = str(fit)
        assert "Test of kappa = 1" in text and "beta_0" in text and "profile" in text
