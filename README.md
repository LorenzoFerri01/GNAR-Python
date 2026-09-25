# GNAR Python: A Python implementation of Generalised Network Autoregressive Processes

This repository provides a **Python implementation** of the **Generalised Network Autoregressive (GNAR) model**,  
as described in the paper:  
📄 **[Generalized Network Autoregressive Processes and the GNAR Package](https://doi.org/10.18637/jss.v096.i05)**  

GNAR processes are a class of autoregressive models that describe the behavior of **multivariate time series on graphs**. Each univariate time series represents a **node** on the graph, with information flowing between them via the **edges**. The graph imposes additional constraints on the parameters of the GNAR process, depending on the model class.  

- In global - $\alpha$ models, all parameters are shared between nodes.  
- In standard (or local - $\alpha$) models, only the $\beta$ (neighbour set) coefficients are shared whereas the $\alpha$ ( autoregressive coefficients) are node specific.  
- In local - $\alpha\beta$ models, all parameters are node specific.  

The package supports **unweighted**, **weighted** and **distance** networks. The `net_type` parameter controls how neighbour weights are computed:

- `"unweighted"` (default): Binary adjacency matrix. Each stage-$r$ neighbour receives equal weight $1/|N_r(j)|$.
- `"weighted"`: Edge weights represent connection strength (larger = stronger). Within each stage, weights are proportional to the sum of products of edge weights along all shortest paths.
- `"distance"`: Edge weights represent distances (larger = farther). Distances are converted to connection weights via $1/d_{ij}$, then treated as in the weighted case.

---

## Installation  

To install GNAR-Python, clone this repository and install it using `pip`:

```bash
git clone https://github.com/henrypalasciano/GNAR-Python.git
cd GNAR-Python
pip install .
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/henrypalasciano/GNAR-Python.git
```

---

## 📖 Example Usage  

Below is a simple example of how to use the **GNAR-Python** package to fit a Generalised Network Autoregressive (GNAR) model, generate forecasts, and simulate data.

```python
import numpy as np
from gnar.gnar import GNAR

# Generate synthetic time series data (100 time steps, 3 nodes)
ts = np.random.normal(0, 1, (100, 3))

# Define an adjacency matrix for the network
A = np.array([[0, 1, 0], 
              [1, 0, 1],
              [0, 1, 0]])

# Fit a standard GNAR(2,[1,1]) process to the time series data on an unweighted network
G = GNAR(A, p=2, s=np.array([1, 1]), ts=ts, demean=True, model_type="standard")
print(G)

# Compute model selection criteria
print("BIC:", G.bic())
print("AIC:", G.aic())

# Simulate 100 time steps from the fitted GNAR model
simulated_data = G.simulate(100)

# Forecast the next 5 time steps from some time series data (here h is the forecast horizon)
ts = np.random.normal(0, 1, (10, 3))
predictions = G.predict(ts=ts, h=5)

# Alternatively forecast directly from the last observation of the multivariate time series the model was fit to
predictions = G.predict(h=5)

# Visualise the graph
G.draw()

# --- Weighted and distance networks ---

# Weighted network: edge weights represent connection strength
A_weighted = np.array([[0, 2, 0],
                       [2, 0, 3],
                       [0, 3, 0]], dtype=float)
ts = np.random.normal(0, 1, (100, 3))
G_weighted = GNAR(A_weighted, p=2, s=np.array([1, 1]), ts=ts, net_type="weighted")

# Distance network: edge weights represent distances (closer nodes receive more weight)
A_distance = np.array([[0, 0.5, 0],
                       [0.5, 0, 1.2],
                       [0, 1.2, 0]], dtype=float)
G_distance = GNAR(A_distance, p=2, s=np.array([1, 1]), ts=ts, net_type="distance")
```

---

## Degree normalisation exponent κ

For GNAR(1, [1]) models on unweighted, undirected graphs, the neighbour sum of node $i$ can be scaled by its degree $N_i$ to the power $-\kappa$:

$$X_{i,t} = \alpha_i X_{i,t-1} + \beta N_i^{-\kappa} \sum_{q \in \mathcal{N}(i)} X_{q,t-1} + \varepsilon_{i,t}.$$

$\kappa = 1$ is the standard GNAR neighbour average and the default, so existing models are unchanged; $\kappa = 0$ means no normalisation. κ can be fixed or estimated by profile likelihood, with standard errors that account for estimating it:

```python
from gnar import simulate_gnar1, stationary_params, fit_gnar1, plot_profile

# Choose beta so that the most heavily weighted node has network weight 0.4, then simulate from the stationary distribution
alpha = np.full(A.shape[0], 0.3)
beta = stationary_params(A, kappa=0.5, b=0.4, alpha=alpha)
X = simulate_gnar1(A, alpha, beta, kappa=0.5, n=2000, rng=1)

fit = fit_gnar1(A, X, kappa=None, demean=False)   # kappa=None estimates kappa; a number fixes it
print(fit)                                        # estimates, standard errors, intervals and the test of kappa = 1
fit.test_kappa(1.0)                               # Wald and likelihood-ratio tests of kappa = 1
fit.beta_at()                                     # beta at the geometric-mean degree, with its standard error
plot_profile(fit)                                 # profile deviance with the chi-square cutoff and the Wald approximation

# The same estimate inside a GNAR object, for forecasting and simulation
G_kappa = GNAR(A, p=1, s=np.array([1]), ts=X, kappa=None, demean=False)
```

`asymptotic_cov` and `variance_bounds` give the exact asymptotic covariance and degree-based bounds on it from the true parameters and the graph. κ is identified only if the graph has at least two distinct degrees among nodes with neighbours; a warning is raised otherwise. See [examples/kappa_example.ipynb](examples/kappa_example.ipynb) for a worked example.

---

## 📂 Repository Structure  

The repository is organised as follows:

```plaintext
📂 GNAR-Python/
 ┣ 📂 gnar/            # Core GNAR model implementation
 ┣ 📂 examples/        # Example scripts demonstrating GNAR usage
 ┣ 📂 tests/           # Unit tests for model validation
 ┣ 📜 README.md        # Project documentation
 ┣ 📜 requirements.txt # List of dependencies
 ┣ 📜 setup.py         # Installation setup file
```

---

## Citing  

If you use **GNAR-Python** in your research, please cite the following paper:  

```bibtex
@article{GNAR,
  title={{Generalized Network Autoregressive Processes} and the {GNAR} Package},
  volume={96},
  url={https://www.jstatsoft.org/index.php/jss/article/view/v096i05},
  doi={10.18637/jss.v096.i05},
  number={5},
  journal={Journal of Statistical Software},
  author={Knight, M. and Leeming, K. and Nason, G. P. and Nunes, M.},
  year={2020},
  pages={1–36}
}
```

and Python implementation:

```bibtex
@software{GNAR-Python,
  author = {Palasciano, H. A.},
  title = {{GNAR Python}: A Python Implementation of Generalised Network Autoregressive Processes},
  year = {2024},
  url = {https://github.com/henrypalasciano/GNAR-Python},
  version = {1.0},
  doi = {10.5281/zenodo.15538154}
}
```

---

## Contact  

**Henry Antonio Palasciano**  
📧 Email: [henry.palasciano17@imperial.ac.uk](mailto:henry.palasciano17@imperial.ac.uk)

