# cosmicray-analysis-tools

Python library for cosmic-ray mass-composition analysis via template PDF fitting, following the methodology of [Aartsen et al. (2019)](https://arxiv.org/abs/1906.04317).

## Features

- Unbinned extended likelihood fit (`iminuit.cost.ExtendedUnbinnedNLL`)
- Binned template fit with Barlow-Beeston treatment (`iminuit.cost.Template`)
- Any number of mass-component templates
- HESSE covariance-propagated fraction uncertainties
- Optional MINOS asymmetric errors
- Per-event weight support (unbinned mode)

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.21
- matplotlib ≥ 3.4
- iminuit ≥ 2.16

## Installation

```bash
pip install .
```

For development (includes pytest and scipy):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from scipy.stats import truncnorm
from crtemplate_analysis.crtemplate_analysis import Template_Analysis

# 1. Define template PDFs (one per mass group)
lo, hi = 0.0, np.log(56)
rv_H  = truncnorm((lo - np.log(1))  / 0.75, (hi - np.log(1))  / 0.75, loc=np.log(1),  scale=0.75)
rv_Fe = truncnorm((lo - np.log(56)) / 0.75, (hi - np.log(56)) / 0.75, loc=np.log(56), scale=0.75)

# 2. Generate (or load) observed data
rng  = np.random.default_rng(42)
data = np.concatenate([rv_H.rvs(300, random_state=0), rv_Fe.rvs(100, random_state=1)])

# 3. Run the fit
fit = Template_Analysis(minos=False, binned=False, strategy=0)
fit.join_pdfs([rv_H.pdf, rv_Fe.pdf])
fit.template_likelihood(data, bins=np.linspace(lo, hi, 51), set_fitrange=(lo, hi))

# 4. Retrieve results
res = fit.get_results()
print(res['fractions'])        # {'N1': 0.75, 'N2': 0.25}  (approx.)
print(res['fraction_errors'])  # covariance-propagated uncertainties
```

A complete four-component example (H, He, O, Fe) is in [`example/example_template_fit.py`](example/example_template_fit.py).

## API reference

### `Template_Analysis(minos=False, binned=False, strategy=0)`

| Parameter  | Type | Description |
|------------|------|-------------|
| `minos`    | bool | Run MINOS after HESSE for asymmetric errors |
| `binned`   | bool | Use binned fit; unbinned if `False` |
| `strategy` | int  | iminuit strategy: 0 = fast, 1 = default, 2 = best |

### `.join_pdfs(template_pdfs)`

Register template PDFs. Each callable must accept an array of x values and return a normalised probability density.

### `.template_likelihood(data, bins, set_fitrange, weights=None)`

Build the cost function and run the fit.

| Parameter      | Type              | Description |
|----------------|-------------------|-------------|
| `data`         | array-like        | Observed events |
| `bins`         | int or array-like | Bin count or explicit edges |
| `set_fitrange` | tuple             | `(low, high)` fit range (used when `bins` is an int) |
| `weights`      | array-like        | Per-event weights (both binned and unbinned modes) |

**Weighted fits:**  in binned mode the weights are forwarded to `numpy.histogram`.
In unbinned mode, because `iminuit.cost.ExtendedUnbinnedNLL` does not expose a
weights parameter, a weighted extended NLL is used:

```
2 * ( Σ N_i  −  Σ_j w_j · log( Σ_i N_i · pdf_i(x_j) ) )
```

Example:

```python
weights = np.where(heavy_flag, 2.0, 1.0)   # up-weight heavy events
fit.template_likelihood(data, bins=50, set_fitrange=(lo, hi), weights=weights)
```

### `.get_results()`

Returns a dict with keys:

| Key               | Description |
|-------------------|-------------|
| `values`          | Fitted yields `{N1: ..., N2: ...}` |
| `errors`          | HESSE errors on yields |
| `valid`           | Whether the fit converged |
| `fval`            | Minimum function value |
| `fractions`       | Fitted fractions `f_i = N_i / Σ N` |
| `fraction_errors` | Covariance-propagated fraction uncertainties |

### `.draw(trues=None, parts=False, bins=None, ax=None)`

Plot the fit overlaid on a data histogram. Pass `trues` (list of true yields) to show true fractions in the legend; `parts=True` to draw individual components.

## Running tests

```bash
pytest
```

## To do

- Numba support for faster unbinned fits on large datasets
- Weighted event dataset validation
