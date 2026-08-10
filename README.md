# cosmicray-analysis-tools

Python library for cosmic-ray mass-composition analysis via template PDF fitting, following the methodology of [Aartsen et al. (2019)](https://arxiv.org/abs/1906.04317).

## Features

- Unbinned extended likelihood fit (`iminuit.cost.ExtendedUnbinnedNLL`)
- Binned template fit with Barlow-Beeston treatment (`iminuit.cost.Template`)
- Any number of mass-component templates
- HESSE covariance-propagated fraction uncertainties
- Optional MINOS asymmetric errors
- Per-event weight support (binned and unbinned modes)

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.21
- matplotlib ≥ 3.4
- iminuit ≥ 2.16

## Installation
local version
```bash
pip install .
```
or
just the package
```bash
pip install git+https://github.com/matthiasplum/cosmicray-analysis-tools
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
# Parameters for template PDFs
template_scale   = 0.75
mass_H, mass_Fe  = 1, 56
lo, hi           = 0.0, np.log(mass_Fe)

rv_H  = truncnorm((lo - np.log(mass_H))  / template_scale, (hi - np.log(mass_H))  / template_scale, loc=np.log(mass_H),  scale=template_scale)
rv_Fe = truncnorm((lo - np.log(mass_Fe)) / template_scale, (hi - np.log(mass_Fe)) / template_scale, loc=np.log(mass_Fe), scale=template_scale)

# 2. Generate (or load) observed data
# Define counts for H and Fe events
set_n_H  = 300
set_n_Fe = 100

total_samples = set_n_H + set_n_Fe # total number of events
true_fractions = {'N1': set_n_H / total_samples, 'N2':  set_n_Fe / total_samples} # Define true fractions dictionary based on counts

data = np.concatenate([rv_H.rvs(set_n_H, random_state=0), rv_Fe.rvs(set_n_Fe, random_state=1)])

# 3. Run the fit
fit = Template_Analysis(minos=False, binned=False, strategy=0)
fit.join_pdfs([rv_H.pdf, rv_Fe.pdf])
fit.template_likelihood(data, set_bins=np.linspace(lo, hi, 51), set_fitrange=(lo, hi))

# 4. Retrieve results
res = fit.get_results()

print("Template Analysis Results:")
print(f"{'':<4} {'Fraction':<10} {'Error':<10} {'True Value':<10}")
for key in res['fractions']:
    print(f"{key:<4} {res['fractions'][key]:<10.5f} {res['fraction_errors'][key]:<10.5f} {true_fractions[key]:<10.5f}")
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

### `.template_likelihood(data, set_bins, set_fitrange, weights=None)`

Build the cost function and run the fit.

| Parameter      | Type              | Description |
|----------------|-------------------|-------------|
| `data`         | array-like        | Observed events |
| `set_bins`     | int or array-like | Bin count or explicit edges |
| `set_fitrange` | tuple             | `(low, high)` fit range (used when `set_bins` is an int) |
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

### `.draw(trues=None, parts=False, bins=None, ax=None, colors=None, total_color="black")`

Plot the fit overlaid on a data histogram.

| Parameter     | Type              | Description |
|---------------|-------------------|-------------|
| `trues`       | list of float     | True yields — shown alongside fitted fractions in the legend |
| `parts`       | bool              | If `True`, plot each component as a dashed line |
| `bins`        | int or array-like | Bins for unbinned mode display; ignored in binned mode |
| `ax`          | `Axes`            | Axes to draw on; defaults to current axes |
| `colors`      | list of color     | Colors for individual component lines (requires `parts=True`) |
| `total_color` | color             | Color for the total fit line (default: `"black"`) |

### `.draw_fraction_contours(labels=None, sigma_levels=(1, 2), color="blue", true_color="red", trues=None, title=None, fig=None)`

Plot pairwise 2D confidence ellipses for all fitted fraction combinations.

| Parameter      | Type           | Description |
|----------------|----------------|-------------|
| `labels`       | list of str    | Component names for axis labels (e.g. `["H", "He", "O", "Fe"]`). Defaults to `["N1", "N2", ...]` |
| `sigma_levels` | tuple of int   | Sigma contours to draw (default: `(1, 2)`) |
| `color`        | color          | Color for the fit ellipses and best-fit marker (default: `"blue"`) |
| `true_color`   | color          | Color for the true-value marker (default: `"red"`) |
| `trues`        | list of float  | True yields in template order — converted to fractions and shown as a star on each panel |
| `title`        | str            | Figure suptitle |
| `fig`          | `Figure`       | Existing figure to draw into; a new one is created if not given |

Returns `(fig, axes)`. Ellipses use the physics convention Δχ² = n_σ² per contour.

```python
fig, axes = fit.draw_fraction_contours(
    labels=["H", "He", "O", "Fe"],
    sigma_levels=(1, 2),
    color="blue",
    trues=[200, 200, 200, 400],   # optional true yields
    title="Log(E/GeV): 7.45",
)
```

## Running tests

```bash
pytest
```

## To do

- Numba support for faster unbinned fits on large datasets
