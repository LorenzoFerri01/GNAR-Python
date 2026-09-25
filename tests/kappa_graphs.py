"""Graph families for the kappa tests. Each function returns a binary, symmetric adjacency matrix with no self-loops."""
import networkx as nx
import numpy as np


def _adjacency(G: nx.Graph) -> np.ndarray:
    return nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=float)


def path_graph(d: int) -> np.ndarray:
    """Path 0 - 1 - ... - (d - 1)."""
    return _adjacency(nx.path_graph(d))


def cycle_graph(d: int) -> np.ndarray:
    """Cycle on d nodes (2-regular)."""
    return _adjacency(nx.cycle_graph(d))


def cycle_plus_chord(d: int) -> np.ndarray:
    """Cycle on d nodes plus the chord 0 - d // 2, so two nodes have degree 3 and the rest degree 2."""
    G = nx.cycle_graph(d)
    G.add_edge(0, d // 2)
    return _adjacency(G)


def star_graph(leaves: int) -> np.ndarray:
    """Star with a hub (node 0) and the given number of leaves, so leaves + 1 nodes."""
    return _adjacency(nx.star_graph(leaves))


def complete_bipartite(m: int, n: int) -> np.ndarray:
    """Complete bipartite graph K_{m,n}; with m != n the two sides have different degrees."""
    return _adjacency(nx.complete_bipartite_graph(m, n))


def random_tree(d: int, seed: int) -> np.ndarray:
    """Uniform random labelled tree on d >= 3 nodes, from a random Pruefer sequence (works for every networkx >= 3.0)."""
    rng = np.random.default_rng(seed)
    return _adjacency(nx.from_prufer_sequence(rng.integers(0, d, d - 2).tolist()))


def erdos_renyi(d: int, p: float, seed: int) -> np.ndarray:
    """Erdos-Renyi graph G(d, p)."""
    return _adjacency(nx.gnp_random_graph(d, p, seed=seed))


def with_isolated_node(A: np.ndarray) -> np.ndarray:
    """Append an isolated node (last index) to the graph."""
    d = A.shape[0]
    B = np.zeros((d + 1, d + 1))
    B[:d, :d] = A
    return B
