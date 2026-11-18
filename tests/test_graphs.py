import networkx as nx
import numpy as np
import pytest

from gnar import KappaGNAR
from gnar.utils.graphs import powerlaw_configuration_graph, powerlaw_degree_sequence


class TestDegreeSequence:

    def test_bounds_and_parity(self):
        for seed in range(20):
            deg = powerlaw_degree_sequence(501, gamma=2.5, k_min=2, k_max=22, rng=seed)
            assert deg.shape == (501,)
            assert deg.min() >= 2 and deg.max() <= 22
            assert deg.sum() % 2 == 0

    def test_default_k_max_is_structural_cutoff(self):
        deg = powerlaw_degree_sequence(400, gamma=1.5, rng=0)
        assert deg.max() <= 20

    def test_distribution(self):
        # Empirical frequencies match k^{-gamma} normalised on the support
        deg = powerlaw_degree_sequence(200_000, gamma=2.5, k_min=2, k_max=10, rng=1)
        support = np.arange(2, 11)
        expected = support ** -2.5 / np.sum(support ** -2.5)
        observed = np.array([np.mean(deg == k) for k in support])
        np.testing.assert_allclose(observed, expected, atol=0.005)

    def test_reproducible(self):
        np.testing.assert_array_equal(powerlaw_degree_sequence(100, 2.5, rng=3), powerlaw_degree_sequence(100, 2.5, rng=3))

    @pytest.mark.parametrize("kwargs", [dict(n=1, gamma=2.5), dict(n=50, gamma=0), dict(n=50, gamma=2.5, k_min=0),
                                        dict(n=50, gamma=2.5, k_min=5, k_max=4), dict(n=50, gamma=2.5, k_max=50)])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            powerlaw_degree_sequence(**kwargs)


class TestConfigurationGraph:

    @pytest.mark.parametrize("seed", range(5))
    def test_valid_for_kappa_gnar(self, seed):
        G, info = powerlaw_configuration_graph(500, seed=seed, return_info=True)
        assert nx.is_connected(G)
        assert nx.number_of_selfloops(G) == 0
        assert sorted(G.nodes) == list(range(G.number_of_nodes()))
        assert info["n_nodes"] == G.number_of_nodes() and info["n_edges"] == G.number_of_edges()
        assert info["nodes_removed"] == 500 - G.number_of_nodes()
        A = nx.to_numpy_array(G)
        K = KappaGNAR(A, alpha=0.3, b=0.4, kappa=0.5)
        assert not K.is_regular()

    def test_nearly_whole_graph_with_k_min_2(self):
        G, info = powerlaw_configuration_graph(1000, seed=0, return_info=True)
        assert info["nodes_removed"] < 0.05 * 1000

    def test_degree_cap(self):
        G = powerlaw_configuration_graph(400, k_max=20, seed=1)
        assert max(dict(G.degree).values()) <= 20

    def test_uncapped_has_larger_hubs(self):
        capped = powerlaw_configuration_graph(1000, gamma=2.2, seed=2)
        uncapped = powerlaw_configuration_graph(1000, gamma=2.2, k_max=999, seed=2)
        assert max(dict(uncapped.degree).values()) > max(dict(capped.degree).values())

    def test_reproducible(self):
        G1 = powerlaw_configuration_graph(300, seed=7)
        G2 = powerlaw_configuration_graph(300, seed=7)
        assert nx.utils.graphs_equal(G1, G2)

    def test_return_graph_only(self):
        assert isinstance(powerlaw_configuration_graph(100, seed=0), nx.Graph)
