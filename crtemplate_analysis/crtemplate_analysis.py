'''
Modified on June 23, 2026
Rewritten to use iminuit built-in cost functions (iminuit.cost)

@author: Matthias Plum
@email:  matthias.plum@sdsmt.edu
'''
import numpy as np
import iminuit
from iminuit import cost
import matplotlib.pyplot as plt


class Template_Analysis:
    """
    Cosmic ray mass composition template fit using iminuit built-in cost functions.

    For binned fits, uses iminuit.cost.Template which is purpose-built for
    template fits to binned data with bin-wise Poisson uncertainties.

    For unbinned fits, uses iminuit.cost.ExtendedUnbinnedNLL which fits a
    mixture of unnormalised PDFs to raw event data. When per-event weights are
    provided, a hand-rolled weighted extended NLL is used instead because
    iminuit.cost.ExtendedUnbinnedNLL does not expose a weights parameter.

    Parameters
    ----------
    minos : bool
        If True, run MINOS after HESSE to get asymmetric errors.
    binned : bool
        If True, use binned fit (Template cost). If False, use unbinned fit
        (ExtendedUnbinnedNLL cost).
    strategy : int
        iminuit strategy: 0 = fast, 1 = default, 2 = best.
    """

    def __init__(self, minos=False, binned=False, strategy=0):
        self.minos = minos
        self.binned = binned
        self.strategy = strategy
        self.num_pdfs = 1
        self.template_pdfs = None
        self.cost_func = None
        self.minuit = None
        self._data = None
        self._weights = None
        self._custom_cost = False  # True when hand-rolled weighted NLL is used

    def join_pdfs(self, template_pdfs):
        """
        Register the template PDF functions.

        Parameters
        ----------
        template_pdfs : list of callable
            Each callable must accept an array of x values and return a
            normalised probability density evaluated at those points.
        """
        self.num_pdfs = len(template_pdfs)
        self.template_pdfs = template_pdfs

    def template_likelihood(self, data, set_bins, set_fitrange, weights=None):
        """
        Build the cost function, run migrad, hesse, and optionally minos.

        Parameters
        ----------
        data : array-like
            Observed data events.
        set_bins : int or array-like
            Number of bins (int) or explicit bin edges (array).
        set_fitrange : tuple
            (low, high) fit range used when set_bins is an int.
        weights : array-like, optional
            Per-event weights. Supported in both binned and unbinned modes.
            In binned mode the weights are passed directly to numpy.histogram.
            In unbinned mode a weighted extended NLL is used:
            2 * (sum(N) - sum(w_i * log(f(x_i)))).
        """
        data = np.asarray(data, dtype=float)
        weights = np.asarray(weights, dtype=float) if weights is not None else None
        self._data = data
        self._weights = weights
        self._custom_cost = False

        param_names = [f"N{i+1}" for i in range(self.num_pdfs)]
        # Effective event count: sum of weights when weighted, else number of events
        n_eff = float(weights.sum()) if weights is not None else float(len(data))

        if self.binned:
            if isinstance(set_bins, int):
                bin_edges = np.linspace(set_fitrange[0], set_fitrange[1], set_bins + 1)
            else:
                bin_edges = np.asarray(set_bins)

            counts, _ = np.histogram(data, bins=bin_edges, weights=weights)

            # Build template histograms for cost.Template (Barlow-Beeston method).
            # Template treats each array as finite-statistics MC counts and adds
            # Poisson uncertainty per bin. Passing pdf*width (~1 event total) would
            # make template uncertainty dominate and inflate yield errors enormously.
            # Scaling by a large N_mc makes template uncertainty negligible, recovering
            # the standard binned-likelihood result.
            N_mc = 1_000_000
            bin_widths = np.diff(bin_edges)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            templates = tuple(
                pdf(bin_centers) * bin_widths * N_mc for pdf in self.template_pdfs
            )

            self.cost_func = cost.Template(counts, bin_edges, templates,
                                           name=param_names)

        else:
            pdfs = self.template_pdfs

            if weights is not None:
                # iminuit.cost.ExtendedUnbinnedNLL has no weights parameter, so
                # we hand-roll the weighted extended NLL:
                #   2 * (sum(N) - sum(w_i * log(mixture_pdf(x_i))))
                w = weights
                data_local = data
                n_pdfs = self.num_pdfs

                def _weighted_enll(*yields):
                    N = np.array(yields)
                    total = float(N.sum())
                    pdf_vals = sum(N[i] * pdfs[i](data_local) for i in range(n_pdfs))
                    log_pdf = np.log(np.maximum(pdf_vals, 1e-300))
                    return 2.0 * (total - float(np.dot(w, log_pdf)))

                self.cost_func = _weighted_enll
                self._custom_cost = True

            else:
                def scaled_pdf(x, *yields):
                    N = np.array(yields)
                    # Integral of each normalised PDF over the fit range is 1 by
                    # definition, so the total expected count equals sum(N).
                    total = np.sum(N)
                    pdf_at_x = sum(N[i] * pdfs[i](x) for i in range(len(pdfs)))
                    return total, pdf_at_x

                self.cost_func = cost.ExtendedUnbinnedNLL(data, scaled_pdf,
                                                          name=param_names)

        # Starting values: equal split of effective event count
        start = {name: n_eff / self.num_pdfs for name in param_names}

        if self._custom_cost:
            # *args function has no named parameters; supply names explicitly
            self.minuit = iminuit.Minuit(self.cost_func,
                                         *list(start.values()),
                                         name=param_names)
        else:
            self.minuit = iminuit.Minuit(self.cost_func, **start)

        for name in param_names:
            self.minuit.limits[name] = (0, n_eff)

        self.minuit.strategy = self.strategy
        self.minuit.print_level = 0

        self.minuit.migrad()
        self.minuit.hesse()
        if self.minos:
            self.minuit.minos()

        print('fmin:')
        print(self.minuit.fmin)
        print('covariance matrix:')
        print(self.minuit.covariance)
        print('correlation matrix:')
        print(self.minuit.covariance.correlation())
        if self.minos:
            print(self.minuit.merrors)

    def get_results(self):
        """
        Return fit results as a dictionary.

        Returns
        -------
        dict with keys: values, errors, valid, fval, fractions, fraction_errors
            fractions and fraction_errors use full covariance propagation.
        """
        if self.minuit is None:
            raise RuntimeError("Run template_likelihood() before get_results().")
        fractions, frac_errors = self._fraction_errors()
        param_names = [f"N{i+1}" for i in range(self.num_pdfs)]
        params = self.minuit.parameters
        return {
            'values':          {k: self.minuit.values[k] for k in params},
            'errors':          {k: self.minuit.errors[k] for k in params},
            'valid':           self.minuit.valid,
            'fval':            self.minuit.fmin.fval,
            'fractions':       dict(zip(param_names, fractions)),
            'fraction_errors': dict(zip(param_names, frac_errors)),
        }

    def _fraction_errors(self):
        """
        Compute fraction f_i = N_i / sum(N) and propagate errors through
        the full HESSE covariance matrix.

        Returns
        -------
        fractions : ndarray, shape (num_pdfs,)
        frac_errors : ndarray, shape (num_pdfs,)
        """
        n = self.num_pdfs
        values = np.array([self.minuit.values[f"N{i+1}"] for i in range(n)])
        cov = np.array(self.minuit.covariance)
        N_total = values.sum()

        fractions = values / N_total

        # Jacobian J[i, j] = d(f_i)/d(N_j)
        #   = (N_total - N_i) / N_total^2   if i == j
        #   = -N_i / N_total^2              if i != j
        J = np.full((n, n), -values[:, None] / N_total**2)
        np.fill_diagonal(J, (N_total - values) / N_total**2)

        frac_cov = J @ cov @ J.T
        frac_errors = np.sqrt(np.maximum(np.diag(frac_cov), 0.0))

        return fractions, frac_errors

    def draw(self, trues=None, parts=False, bins=None, ax=None, colors=None, total_color="black"):
        """
        Plot the fit result overlaid on the data histogram.

        Parameters
        ----------
        trues : list of float, optional
            True yield values (displayed in legend for validation).
        parts : bool
            If True, plot individual template components as dashed lines.
        bins : int or array-like, optional
            Bins for unbinned mode display. Ignored in binned mode.
        ax : matplotlib.axes.Axes, optional
            Axes to draw on. Defaults to current axes.
        colors : list of color, optional
            Colors for the individual component lines (used when parts=True).
            Must have at least as many entries as there are templates.
            Defaults to the current matplotlib color cycle.
        total_color : color, optional
            Color for the total fit line. Default is "black".
        """
        if self.minuit is None:
            raise RuntimeError("Run template_likelihood() before draw().")
        if ax is None:
            ax = plt.gca()

        N_fit = [self.minuit.values[f"N{i+1}"] for i in range(self.num_pdfs)]
        fractions, frac_errors = self._fraction_errors()

        if self.binned:
            bin_edges = self.cost_func.xe
        else:
            # Use stored data (works for both built-in and custom cost functions)
            data = self._data if self._data is not None else self.cost_func.data
            if isinstance(bins, int):
                bin_edges = np.linspace(data.min(), data.max(), bins + 1)
            else:
                bin_edges = np.asarray(bins)

        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        widths = np.diff(bin_edges)

        fit_info = ["Fit Results:"]
        total_fit = np.zeros_like(centers)

        for i, pdf in enumerate(self.template_pdfs):
            component = N_fit[i] * pdf(centers) * widths
            total_fit += component

            f_fit = fractions[i]
            f_err = frac_errors[i]
            if trues is not None and i < len(trues):
                N_total_true = sum(trues)
                f_true = trues[i] / N_total_true if N_total_true > 0 else float('nan')
                fit_info.append(
                    f"N{i+1} = {N_fit[i]:.1f}  "
                    f"f{i+1} = {f_fit:.3f} ± {f_err:.3f}  (True: {f_true:.3f})"
                )
            else:
                fit_info.append(f"N{i+1} = {N_fit[i]:.1f}  f{i+1} = {f_fit:.3f} ± {f_err:.3f}")

            if parts:
                color = colors[i] if colors is not None else None
                ax.plot(centers, component, color=color,
                        label=f"Component N{i+1}", linestyle="--")

        ax.plot(centers, total_fit, color=total_color, lw=2, label="Total Fit")

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, "\n".join(fit_info), transform=ax.transAxes,
                fontsize=9, verticalalignment='top', bbox=props)
        ax.legend(loc="upper right")
