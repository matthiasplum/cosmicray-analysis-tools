# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
