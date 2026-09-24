"""
Freeze the outputs of pygnar before the kappa extension, for the backward-compatibility test (acceptance test 1).

Run it against the unmodified package (commit 8466a80), for example from a git worktree:

    git worktree add ../GNAR-Python-legacy 8466a80
    PYTHONPATH=../GNAR-Python-legacy python tests/data/make_legacy_reference.py tests/data/legacy_reference.npz

For every case it stores the inputs, the specification and the outputs: coefficients, noise covariance, mean,
forecasts, a seeded simulation, the VAR form, autocovariances, BIC/AIC and the printed representations, or the type
of the exception raised when the legacy code fails on a configuration. The inputs
are stored as well, so the test does not depend on how they were generated. Graphs are kept small (at most 5 nodes)
so that the iterative least-squares solver used by the standard model converges to machine precision.
"""
import hashlib
import json
import sys
import warnings

import numpy as np
import pandas as pd
import scipy

import gnar
from gnar import GNAR

GRAPHS = {
    "path2": (np.array([[0, 1], [1, 0]], dtype=float), "unweighted"),
    "path3": (np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float), "unweighted"),
    "weighted_path3": (np.array([[0, 2, 0], [2, 0, 3], [0, 3, 0]], dtype=float), "weighted"),
    "distance_path3": (np.array([[0, 4, 0], [4, 0, 2], [0, 2, 0]], dtype=float), "distance"),
    "diamond4": (np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=float), "unweighted"),
    # The five-net of examples/five_net.py, as an integer array like in the example
    "five_net": (np.array([[0, 0, 0, 1, 1], [0, 0, 1, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0]]), "unweighted"),
}

# Hardcoded series from tests/test_fitting.py
TS_2NODE = np.array([[0.1, 0.3], [-0.2, 0.5], [0.4, -0.1], [0.3, 0.2], [-0.1, 0.4], [0.2, -0.3], [0.5, 0.1], [-0.3, 0.6],
                     [0.1, -0.2], [0.4, 0.3], [-0.2, 0.1], [0.3, -0.4], [0.1, 0.5], [-0.4, 0.2], [0.2, 0.1], [0.3, -0.1],
                     [-0.1, 0.3], [0.4, -0.2], [0.2, 0.4], [-0.3, 0.1]])
TS_3NODE = np.array([[0.5, -0.2, 0.3], [-0.1, 0.4, 0.2], [0.3, -0.3, 0.1], [0.2, 0.1, -0.2], [-0.4, 0.5, 0.3],
                     [0.1, -0.1, 0.4], [0.3, 0.2, -0.3], [-0.2, 0.3, 0.1], [0.4, -0.4, 0.2], [0.1, 0.1, -0.1],
                     [-0.3, 0.4, 0.3], [0.2, -0.2, 0.1], [0.3, 0.1, -0.4], [-0.1, 0.3, 0.2], [0.4, -0.1, 0.1],
                     [0.2, 0.2, -0.2], [-0.3, 0.4, 0.3], [0.1, -0.3, 0.2], [0.3, 0.1, -0.1], [-0.2, 0.2, 0.4]])


def outputs(G: GNAR, ts_pred: np.ndarray, fitted: bool) -> dict:
    """Everything a user can observe from a GNAR object."""
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


def cases():
    """Yield (spec, inputs) pairs; spec holds the GNAR keyword arguments that are not arrays."""
    rng = np.random.RandomState(20260924)
    # Fitted models: every graph, model type, method and lag structure used by the existing tests and examples
    for graph, (A, net_type) in GRAPHS.items():
        d = A.shape[0]
        ts = rng.normal(0, 1, (60, d)) + np.arange(d)
        ts_pred = rng.normal(0, 1, (8, d))
        for p, s in [(1, [1]), (2, [1, 1]), (2, [2, 1])]:
            for model_type in ["global", "standard", "local"]:
                for method in ["OLS", "YW"]:
                    spec = dict(graph=graph, net_type=net_type, p=p, s=s, model_type=model_type, method=method, demean=True)
                    yield spec, dict(A=A, ts=ts, ts_pred=ts_pred)
    # The hardcoded series of the existing fitting tests, with and without demeaning
    for graph, ts in [("path2", TS_2NODE), ("path3", TS_3NODE)]:
        A, net_type = GRAPHS[graph]
        for model_type in ["global", "standard", "local"]:
            for demean in [True, False]:
                spec = dict(graph=graph, net_type=net_type, p=1, s=[1], model_type=model_type, method="OLS", demean=demean)
                yield spec, dict(A=A, ts=ts, ts_pred=ts[-6:])
    # Models set up from parameters: the existing simulation and forecasting tests and examples/five_net.py
    path3 = GRAPHS["path3"][0]
    std_coeffs = np.array([[0.3, 0.2, 0.4], [0.1, 0.1, 0.1]])
    yield (dict(graph="path3", net_type="unweighted", p=1, s=[1], model_type="standard", mean=[1.0, 2.0, 3.0], sigma_2=1.0),
           dict(A=path3, coeffs=std_coeffs, ts_pred=rng.normal(0, 1, (6, 3))))
    yield (dict(graph="weighted_path3", net_type="weighted", p=1, s=[1], model_type="standard", mean=0, sigma_2=1.0),
           dict(A=GRAPHS["weighted_path3"][0], coeffs=std_coeffs, ts_pred=rng.normal(0, 1, (6, 3))))
    p2_coeffs = np.array([[0.5, 0.3, 0.4], [0.2, 0.1, 0.15], [0.1, 0.1, 0.1], [0.05, 0.05, 0.05]])
    yield (dict(graph="path3", net_type="unweighted", p=2, s=[1, 1], model_type="standard", mean=0, sigma_2=1.0),
           dict(A=path3, coeffs=p2_coeffs, ts_pred=rng.normal(0, 1, (6, 3))))
    five = np.array([[0.2] * 5, [0.2] * 5, [0.5] * 5, [-0.1] * 5])
    yield (dict(graph="five_net", net_type="unweighted", p=2, s=[1, 1], model_type="global", mean=0, sigma_2=1),
           dict(A=GRAPHS["five_net"][0], coeffs=five, ts_pred=rng.normal(0, 1, (10, 5))))


def load_case(ref, i: int) -> tuple[dict, dict]:
    """Inputs and frozen outputs of case i of a loaded reference file."""
    inputs = {k.split("/")[-1]: ref[f"inputs/{ref[k]}"] for k in ref.files if k.startswith(f"{i}/in/")}
    outputs = {k.split("/")[-1]: ref[k] for k in ref.files if k.startswith(f"{i}/out/")}
    return inputs, outputs


def build(spec: dict, inputs: dict, **extra) -> GNAR:
    """Rebuild the GNAR object of a case (used by the test with extra=dict(kappa=1.0))."""
    kwargs = dict(p=spec["p"], s=np.array(spec["s"]), model_type=spec["model_type"], net_type=spec["net_type"], **extra)
    if "coeffs" in inputs:
        mean = np.array(spec["mean"]) if isinstance(spec["mean"], list) else spec["mean"]
        return GNAR(inputs["A"], coeffs=inputs["coeffs"], mean=mean, sigma_2=spec["sigma_2"], **kwargs)
    return GNAR(inputs["A"], ts=inputs["ts"], demean=spec["demean"], method=spec["method"], **kwargs)


def main(path: str) -> None:
    arrays = {}
    specs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (spec, inputs) in enumerate(cases()):
            specs.append(spec)
            for key, value in inputs.items():
                # Inputs shared between cases are stored once, under their content hash
                digest = hashlib.sha1(np.ascontiguousarray(value).tobytes() + str(value.dtype).encode()).hexdigest()
                arrays[f"inputs/{digest}"] = value
                arrays[f"{i}/in/{key}"] = np.array(digest)
            try:
                G = build(spec, inputs)
            except Exception as err:
                # Some configurations fail in the legacy code; the test checks that they still fail the same way
                arrays[f"{i}/error"] = np.array(type(err).__name__)
                continue
            for key, value in outputs(G, inputs["ts_pred"], fitted="ts" in inputs).items():
                arrays[f"{i}/out/{key}"] = value
    meta = dict(specs=specs, numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__, gnar=gnar.__version__)
    arrays["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)
    print(f"Wrote {len(specs)} cases to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/data/legacy_reference.npz")
