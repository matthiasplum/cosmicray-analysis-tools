'''
Modified on June 23, 2026
Modified to remove probfit dependencies

@author: Matthias Plum
@email:  matthias.plum@sdsmt.edu
'''
import numpy as np
import iminuit
import matplotlib.pyplot as plt

class CustomLikelihood:
    """Custom Extended Likelihood class to replace probfit's LH and drawing tools."""
    def __init__(self, pdfs, data, binned=False, bins=None, bound=None, weights=None):
        self.pdfs = pdfs
        self.data = np.asarray(data)
        self.binned = binned
        self.weights = np.asarray(weights) if weights is not None else None
        
        self.__num_pdfs = len(pdfs)
        self._param_names = [f"N{i+1}" for i in range(self.__num_pdfs)]
        
        if self.binned:
            if isinstance(bins, int):
                self.bin_edges = np.linspace(bound[0], bound[1], bins + 1)
            else:
                self.bin_edges = np.asarray(bins)
                
            self.counts, _ = np.histogram(self.data, bins=self.bin_edges, weights=self.weights)
            self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
            self.bin_widths = np.diff(self.bin_edges)
            self.pdf_values = np.array([pdf(self.bin_centers) * self.bin_widths for pdf in self.pdfs])
            
    def __call__(self, *args):
        N = np.array(args)
        if self.binned:
            mu = np.dot(N, self.pdf_values)
            mu = np.clip(mu, 1e-10, None)
            return np.sum(mu - self.counts * np.log(mu))
        else:
            N_tot = np.sum(N)
            pdf_eval = np.array([pdf(self.data) for pdf in self.pdfs])
            total_pdf = np.dot(N, pdf_eval)
            total_pdf = np.clip(total_pdf, 1e-10, None)
            
            if self.weights is not None:
                return N_tot - np.sum(self.weights * np.log(total_pdf))
            return N_tot - np.sum(np.log(total_pdf))

    def draw(self, args, errors, trues=None, parts=False, bins=None, ax=None):
        """Replicates probfit's draw functionality, displaying fitted vs true parameters."""
        if ax is None:
            ax = plt.gca()
            
        N_fit = [args[name] for name in self._param_names]
        N_err = [errors[name] for name in self._param_names]
        
        if self.binned:
            plot_bins = self.bin_edges
            centers = self.bin_centers
            widths = self.bin_widths
        else:
            if isinstance(bins, int):
                plot_bins = np.linspace(self.data.min(), self.data.max(), bins + 1)
            else:
                plot_bins = np.asarray(bins)
            centers = 0.5 * (plot_bins[:-1] + plot_bins[1:])
            widths = np.diff(plot_bins)

        # Build a text block for the parameters
        fit_info = ["**Fit Results:**"]

        # Plot total fit model
        total_fit = np.zeros_like(centers)
        for i, pdf in enumerate(self.pdfs):
            pdf_vals = pdf(centers) * widths
            component = N_fit[i] * pdf_vals
            total_fit += component
            
            label_text = f"Component N{i+1}"
            
            # Append true values to text if provided
            if trues is not None and i < len(trues):
                fit_info.append(f"N{i+1} = {N_fit[i]:.1f} ± {N_err[i]:.1f} (True: {trues[i]})")
            else:
                fit_info.append(f"N{i+1} = {N_fit[i]:.1f} ± {N_err[i]:.1f}")
            
            if parts:
                ax.plot(centers, component, label=label_text, linestyle="--")
                
        ax.plot(centers, total_fit, color="black", lw=2, label="Total Fit")
        
        # Overlay the results in a clean textbox on the plot
        text_str = "\n".join(fit_info)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        ax.legend(loc="upper right")


class Template_Analysis:

  def __init__(self, minos=False, binned=False, strategy=0):
    # Removed strict old iminuit version check as we use modern API
    self.minos      = minos
    self.binned     = binned
    self.strategy   = strategy
    self.num_pdfs   = 1
    self.template_pdfs = None
    self.likelihood = None
    self.minuit     = None

  def join_pdfs(self, template_pdfs):
    ### Store template PDFs
    self.num_pdfs = len(template_pdfs)
    self.template_pdfs = template_pdfs

  def template_likelihood(self, data, set_bins, set_fitrange, weights=None):
    ### Start fit-parameter (Assume worst case)
    pars = {}
    for i in range(self.num_pdfs):
      pars['N'+str(i+1)] = len(data) / self.num_pdfs
      
    # Create a custom probfit replacement cost function
    self.likelihood = CustomLikelihood(
        self.template_pdfs, data, binned=self.binned, 
        bins=set_bins, bound=set_fitrange, weights=weights
    )

    # Initialize modern Minuit syntax
    self.minuit = iminuit.Minuit(self.likelihood, name=list(pars.keys()), *pars.values())
    
    # Set limits and configurations
    for name in pars.keys():
        self.minuit.limits[name] = (0, len(data))
        
    self.minuit.strategy = self.strategy
    self.minuit.errordef = iminuit.Minuit.LIKELIHOOD
    self.minuit.print_level = 0
    
    self.minuit.migrad()
    self.minuit.hesse()
    if self.minos:
      self.minuit.minos()
      
    print('fmin:')
    print(self.minuit.fmin)
    print('error matrix:')
    print(self.minuit.covariance)
    ### or the correlation matrix
    print('correlation matrix:')
    print(self.minuit.covariance.correlation())
    if self.minos:
      print(self.minuit.merrors)