# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.2.0] - 2026-08-10

### Added
- `CITATION.cff` for GitHub/Zenodo automatic citation support

### Added
- `draw_fraction_contours()` method: pairwise 2D confidence ellipses for all
  fitted fraction pairs, with optional true-value markers and configurable
  sigma levels, colors, and labels
- `colors` and `total_color` parameters on `draw()` for per-component and
  total-fit line colours
- Per-event weight support in unbinned mode via a hand-rolled weighted
  extended NLL (iminuit ≥ 2.x has no native weights parameter for
  `ExtendedUnbinnedNLL`)

### Changed
- Example script: `nFe` reduced from 500 to 400; contour plot added at the end

## [0.1.0] - 2026-06-24

### Added
- 27 pytest tests covering construction, unbinned fit, binned fit,
  four-component fit, and `draw()` (all modes)
- GitHub Actions CI workflow running the test suite on Python 3.9–3.12
- `pyproject.toml` (PEP 517/518) with classifiers, `python_requires>=3.9`,
  and a `dev` extras group (`pytest`, `pytest-cov`, `scipy`)
- `.gitignore` for `__pycache__`, build artifacts, and `.DS_Store`

### Fixed
- `get_results()` broken on iminuit ≥ 2.25: `ValueView` is no longer
  dict-constructible; values and errors are now extracted by iterating
  over `minuit.parameters`

### Removed
- `setup.py` — superseded by `pyproject.toml`

## [0.0.3] - 2026-06-24

### Changed
- Rewrote core fitting to use `iminuit.cost` built-in cost functions
  (`cost.Template` for binned, `cost.ExtendedUnbinnedNLL` for unbinned)
- Replaced HESSE-only errors with optional MINOS asymmetric errors
- Improved `draw()`: shows per-component fractions with propagated errors
  and optional true-value comparison in the legend box

### Removed
- `probfit` dependency (deprecated upstream)

## [0.0.2] - 2024-04-17

### Added
- Extended likelihood template fit method
- Both unbinned and binned fit modes selectable at runtime
- Per-event weights support in unbinned mode
- Constraint: number of events per template capped at total data count

### Fixed
- Reverted experimental feature-select option that set hyperparameters
  to bad starting values (introduced and reverted in the same cycle)

## [0.0.1] - 2022-09-21

### Added
- `Template_Analysis` class with `join_pdfs()`, `template_likelihood()`,
  `get_results()`, and `draw()`
- Support for any number of input template PDFs
- `setup.py` for package installation
- HESSE error estimation and covariance-propagated fraction uncertainties
- Four-component example script (H, He, O, Fe truncated-normal templates)
