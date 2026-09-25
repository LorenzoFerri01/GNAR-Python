import numpy as np
import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run the slow Monte Carlo tests")


def pytest_collection_modifyitems(config, items):
    # Slow Monte Carlo tests (marked @pytest.mark.slow) are skipped unless --runslow is given
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="slow Monte Carlo test; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def path_graph_3():
    """3-node path graph: 0--1--2."""
    return np.array([[0, 1, 0],
                     [1, 0, 1],
                     [0, 1, 0]], dtype=float)


@pytest.fixture
def cycle_graph_5():
    """5-node cycle graph."""
    A = np.zeros((5, 5))
    for i in range(5):
        A[i, (i + 1) % 5] = 1
        A[(i + 1) % 5, i] = 1
    return A


@pytest.fixture
def complete_graph_3():
    """Complete graph K3."""
    return np.array([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0]], dtype=float)


@pytest.fixture
def diamond_graph_4():
    """Diamond graph: 0-1, 0-2, 1-3, 2-3."""
    return np.array([[0, 1, 1, 0],
                     [1, 0, 0, 1],
                     [1, 0, 0, 1],
                     [0, 1, 1, 0]], dtype=float)


@pytest.fixture
def weighted_path_3():
    """3-node path with weights 2, 3."""
    return np.array([[0, 2, 0],
                     [2, 0, 3],
                     [0, 3, 0]], dtype=float)


@pytest.fixture
def distance_path_3():
    """3-node path with distances 4, 2."""
    return np.array([[0, 4, 0],
                     [4, 0, 2],
                     [0, 2, 0]], dtype=float)


@pytest.fixture
def weighted_diamond_4():
    """Diamond with weights 2, 3, 5, 7."""
    return np.array([[0, 2, 3, 0],
                     [2, 0, 0, 5],
                     [3, 0, 0, 7],
                     [0, 5, 7, 0]], dtype=float)
