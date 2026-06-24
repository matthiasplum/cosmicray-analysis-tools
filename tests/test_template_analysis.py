"""
Tests for crtemplate_analysis.Template_Analysis.

The test dataset uses two truncated-normal PDFs (low-mean and high-mean)
mixed in a 3:1 ratio. Both binned and unbinned modes are covered. Fit
accuracy is verified at the ~20% level — tight enough to catch regressions
but loose enough to tolerate random-seed variation.
"""
import numpy as np
import pytest
from scipy.stats import truncnorm
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

from crtemplate_analysis.crtemplate_analysis import Template_Analysis

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SEED = 0
N_LOW, N_HIGH = 300, 100          # 3:1 ratio
FIT_RANGE = (0.0, 5.0)
BINS = np.linspace(*FIT_RANGE, 51)


def _make_rvs():
    lo, hi = FIT_RANGE
    rv_low  = truncnorm((lo - 1.5) / 0.6, (hi - 1.5) / 0.6, loc=1.5, scale=0.6)
    rv_high = truncnorm((lo - 3.5) / 0.6, (hi - 3.5) / 0.6, loc=3.5, scale=0.6)
    return rv_low, rv_high


def _make_data():
    rv_low, rv_high = _make_rvs()
    rng = np.random.default_rng(SEED)
    data = np.concatenate([
        rv_low.rvs(N_LOW,  random_state=rng.integers(1 << 31)),
        rv_high.rvs(N_HIGH, random_state=rng.integers(1 << 31)),
    ])
    return data


def _make_fitter(binned=False):
    rv_low, rv_high = _make_rvs()
    ta = Template_Analysis(minos=False, binned=binned, strategy=0)
    ta.join_pdfs([rv_low.pdf, rv_high.pdf])
    return ta


# ---------------------------------------------------------------------------
# Construction & API guards
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults(self):
        ta = Template_Analysis()
        assert ta.minos is False
        assert ta.binned is False
        assert ta.strategy == 0

    def test_custom_params(self):
        ta = Template_Analysis(minos=True, binned=True, strategy=2)
        assert ta.minos is True
        assert ta.binned is True
        assert ta.strategy == 2

    def test_get_results_before_fit_raises(self):
        ta = Template_Analysis()
        ta.join_pdfs([lambda x: np.ones_like(x)])
        with pytest.raises(RuntimeError):
            ta.get_results()

    def test_draw_before_fit_raises(self):
        ta = Template_Analysis()
        ta.join_pdfs([lambda x: np.ones_like(x)])
        with pytest.raises(RuntimeError):
            ta.draw()


class TestJoinPdfs:
    def test_num_pdfs_set(self):
        ta = Template_Analysis()
        rv_low, rv_high = _make_rvs()
        ta.join_pdfs([rv_low.pdf, rv_high.pdf])
        assert ta.num_pdfs == 2
        assert len(ta.template_pdfs) == 2

    def test_single_pdf(self):
        ta = Template_Analysis()
        rv_low, _ = _make_rvs()
        ta.join_pdfs([rv_low.pdf])
        assert ta.num_pdfs == 1


# ---------------------------------------------------------------------------
# Unbinned fit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def unbinned_fitted():
    data = _make_data()
    ta = _make_fitter(binned=False)
    ta.template_likelihood(data, BINS, FIT_RANGE)
    return ta


class TestUnbinnedFit:
    def test_fit_converged(self, unbinned_fitted):
        assert unbinned_fitted.minuit.valid

    def test_result_keys(self, unbinned_fitted):
        res = unbinned_fitted.get_results()
        for key in ("values", "errors", "valid", "fval", "fractions", "fraction_errors"):
            assert key in res

    def test_yields_positive(self, unbinned_fitted):
        res = unbinned_fitted.get_results()
        assert res["values"]["N1"] > 0
        assert res["values"]["N2"] > 0

    def test_yields_sum_to_total(self, unbinned_fitted):
        res = unbinned_fitted.get_results()
        total = res["values"]["N1"] + res["values"]["N2"]
        assert abs(total - (N_LOW + N_HIGH)) < 0.05 * (N_LOW + N_HIGH)

    def test_fractions_sum_to_one(self, unbinned_fitted):
        res = unbinned_fitted.get_results()
        fsum = sum(res["fractions"].values())
        assert abs(fsum - 1.0) < 1e-9

    def test_fraction_accuracy(self, unbinned_fitted):
        res = unbinned_fitted.get_results()
        f1_true = N_LOW / (N_LOW + N_HIGH)   # 0.75
        assert abs(res["fractions"]["N1"] - f1_true) < 0.20

    def test_errors_positive(self, unbinned_fitted):
        res = unbinned_fitted.get_results()
        assert res["errors"]["N1"] > 0
        assert res["errors"]["N2"] > 0

    def test_fraction_errors_positive(self, unbinned_fitted):
        res = unbinned_fitted.get_results()
        assert res["fraction_errors"]["N1"] > 0
        assert res["fraction_errors"]["N2"] > 0


# ---------------------------------------------------------------------------
# Binned fit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def binned_fitted():
    data = _make_data()
    ta = _make_fitter(binned=True)
    ta.template_likelihood(data, BINS, FIT_RANGE)
    return ta


class TestBinnedFit:
    def test_fit_converged(self, binned_fitted):
        assert binned_fitted.minuit.valid

    def test_yields_positive(self, binned_fitted):
        res = binned_fitted.get_results()
        assert res["values"]["N1"] > 0
        assert res["values"]["N2"] > 0

    def test_yields_sum_to_total(self, binned_fitted):
        res = binned_fitted.get_results()
        total = res["values"]["N1"] + res["values"]["N2"]
        assert abs(total - (N_LOW + N_HIGH)) < 0.05 * (N_LOW + N_HIGH)

    def test_fractions_sum_to_one(self, binned_fitted):
        res = binned_fitted.get_results()
        fsum = sum(res["fractions"].values())
        assert abs(fsum - 1.0) < 1e-9

    def test_fraction_accuracy(self, binned_fitted):
        res = binned_fitted.get_results()
        f1_true = N_LOW / (N_LOW + N_HIGH)
        assert abs(res["fractions"]["N1"] - f1_true) < 0.20

    def test_int_bins(self):
        """Accept an integer bin count in addition to an array of edges."""
        data = _make_data()
        ta = _make_fitter(binned=True)
        ta.template_likelihood(data, 50, FIT_RANGE)
        assert ta.minuit.valid


# ---------------------------------------------------------------------------
# Four-component fit (mirrors the example script)
# ---------------------------------------------------------------------------

def _make_four_component_fit():
    lo, hi = 0.0, np.log(56)
    means  = [np.log(1), np.log(4), np.log(14), np.log(56)]
    scale  = 0.75
    rvs = [
        truncnorm((lo - m) / scale, (hi - m) / scale, loc=m, scale=scale)
        for m in means
    ]
    counts = [200, 200, 200, 500]
    rng = np.random.default_rng(42)
    data = np.concatenate([
        rv.rvs(n, random_state=rng.integers(1 << 31))
        for rv, n in zip(rvs, counts)
    ])
    bins = np.linspace(lo, hi, 56)
    ta = Template_Analysis(minos=False, binned=False, strategy=0)
    ta.join_pdfs([rv.pdf for rv in rvs])
    ta.template_likelihood(data, bins, (lo, hi))
    return ta, counts


@pytest.fixture(scope="module")
def four_component_fitted():
    return _make_four_component_fit()


class TestFourComponents:
    def test_converged(self, four_component_fitted):
        ta, _ = four_component_fitted
        assert ta.minuit.valid

    def test_fractions_sum_to_one(self, four_component_fitted):
        ta, _ = four_component_fitted
        res = ta.get_results()
        assert abs(sum(res["fractions"].values()) - 1.0) < 1e-9

    def test_four_component_accuracy(self, four_component_fitted):
        ta, counts = four_component_fitted
        res = ta.get_results()
        total_true = sum(counts)
        for i, n_true in enumerate(counts):
            f_true = n_true / total_true
            f_fit  = res["fractions"][f"N{i+1}"]
            assert abs(f_fit - f_true) < 0.25, (
                f"Component {i+1}: fitted {f_fit:.3f}, true {f_true:.3f}"
            )


# ---------------------------------------------------------------------------
# draw() smoke test
# ---------------------------------------------------------------------------

class TestDraw:
    def _fitted_ta(self, binned):
        data = _make_data()
        ta = _make_fitter(binned=binned)
        ta.template_likelihood(data, BINS, FIT_RANGE)
        return ta

    def test_draw_unbinned_no_error(self):
        ta = self._fitted_ta(binned=False)
        fig, ax = plt.subplots()
        ta.draw(bins=len(BINS), ax=ax)
        plt.close(fig)

    def test_draw_binned_no_error(self):
        ta = self._fitted_ta(binned=True)
        fig, ax = plt.subplots()
        ta.draw(ax=ax)
        plt.close(fig)

    def test_draw_with_parts_and_trues(self):
        ta = self._fitted_ta(binned=False)
        fig, ax = plt.subplots()
        ta.draw(trues=[N_LOW, N_HIGH], parts=True, bins=len(BINS), ax=ax)
        plt.close(fig)

    def test_draw_default_axes(self):
        ta = self._fitted_ta(binned=False)
        fig, ax = plt.subplots()
        plt.sca(ax)
        ta.draw(bins=len(BINS))
        plt.close(fig)
