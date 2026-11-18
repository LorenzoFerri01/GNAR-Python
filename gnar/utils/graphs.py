from __future__ import annotations

import networkx as nx
import numpy as np


def powerlaw_degree_sequence(
    n: int,
    gamma: float,
    k_min: int = 2,
    k_max: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """
    Sample n degrees from the discrete power law P(k) proportional to k^{-gamma} on {k_min, ..., k_max}. If the sum of the
    degrees is odd, one randomly chosen degree is changed by one (staying within [k_min, k_max]) so that the sequence is
    graphical for the configuration model.

    Params:
        n: int. Number of nodes.
        gamma: float. Power-law exponent (> 0).
        k_min: int. Minimum degree (>= 1).
        k_max: int. Maximum degree (<= n - 1). Defaults to floor(sqrt(n)), the structural cutoff.
        rng: np.random.Generator, int or None. Random number generator or seed.

    Returns:
        np.ndarray of shape (n,) with integer degrees and even sum.
    """
    k_max = int(np.floor(np.sqrt(n))) if k_max is None else int(k_max)
    if n < 2:
        raise ValueError("The number of nodes n must be at least 2.")
    if gamma <= 0:
        raise ValueError("The exponent gamma must be positive.")
    if not 1 <= k_min <= k_max <= n - 1:
        raise ValueError("The degree bounds must satisfy 1 <= k_min <= k_max <= n - 1.")
    rng = np.random.default_rng(rng)
    support = np.arange(k_min, k_max + 1)
    probs = support ** -float(gamma)
    degrees = rng.choice(support, size=n, p=probs / probs.sum())
    if degrees.sum() % 2 == 1:
        i = rng.integers(n)
        degrees[i] += 1 if degrees[i] < k_max else -1
        if degrees[i] < k_min:
            # Only possible when k_min == k_max; any graph with all degrees k_min then needs n * k_min even
            raise ValueError("Cannot obtain an even degree sum with k_min == k_max and odd n * k_min.")
    return degrees


def powerlaw_configuration_graph(
    d: int,
    gamma: float = 2.5,
    k_min: int = 2,
    k_max: int | None = None,
    seed: int | None = None,
    return_info: bool = False,
) -> nx.Graph | tuple[nx.Graph, dict]:
    """
    Generate an undirected simple graph with a power-law degree distribution using the erased configuration model:

        1. sample a degree sequence from P(k) proportional to k^{-gamma} on {k_min, ..., k_max};
        2. pair edge stubs uniformly at random (networkx.configuration_model);
        3. collapse multi-edges and remove self-loops;
        4. keep the largest connected component, relabelling its nodes 0, ..., n - 1.

    The result is connected and has no isolated nodes, so it can be used directly with KappaGNAR. Erasing edges slightly
    lowers the degrees of the largest hubs (and can push a few degrees below k_min); keeping k_max at or below the structural
    cutoff sqrt(d) makes this negligible. With k_min >= 2 the largest component is almost always (nearly) the whole graph,
    so the number of nodes returned can be slightly smaller than d.

    Params:
        d: int. Requested number of nodes.
        gamma: float. Power-law exponent. Defaults to 2.5.
        k_min: int. Minimum degree. Defaults to 2.
        k_max: int. Maximum degree. Defaults to floor(sqrt(d)). Use d - 1 for an uncapped degree distribution.
        seed: int. Random seed, for reproducibility.
        return_info: bool. If True, also return a dict describing the edges and nodes removed.

    Returns:
        nx.Graph, or (nx.Graph, dict) if return_info is True. The dict contains the requested and final number of nodes,
        the number of edges in the final graph, and the number of self-loops, multi-edges and nodes removed.
    """
    rng = np.random.default_rng(seed)
    degrees = powerlaw_degree_sequence(d, gamma, k_min, k_max, rng)
    multigraph = nx.configuration_model(degrees.tolist(), seed=int(rng.integers(2 ** 31)))

    graph = nx.Graph(multigraph)
    n_self_loops = nx.number_of_selfloops(graph)
    n_multi_edges = multigraph.number_of_edges() - graph.number_of_edges()
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))

    largest = max(nx.connected_components(graph), key=len)
    graph = nx.convert_node_labels_to_integers(graph.subgraph(largest).copy())

    if not return_info:
        return graph
    info = {
        "n_requested": d,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "self_loops_removed": n_self_loops,
        "multi_edges_removed": n_multi_edges,
        "nodes_removed": d - graph.number_of_nodes(),
    }
    return graph, info
