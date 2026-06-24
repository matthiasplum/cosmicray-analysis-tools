'''
Modified on June 23, 2026

@author: Matthias Plum
@email:  matthias.plum@sdsmt.edu
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Import directly from the modified file
from crtemplate_analysis.crtemplate_analysis import Template_Analysis

####Setup example 
x         = np.linspace(0,np.log(56),200)
bins      = np.linspace(0,np.log(56),56)
fit_range = (0,np.log(56))

### Binned likelihood
set_binned = False
### Fit strategy (0 = Fast, 2= Best)
set_strategy = 0
### run minos to get asymetrical error
run_minos = False
### set random seed
seed = 42

### Number of events per elementary group
nH  = 200
nHe = 200
nO  = 200
nFe = 500

### Template shape generation
bound_a   = 0.
bound_b   = np.log(56)

mean_H      = np.log(1)
scale_H     = 0.75
mean_He     = np.log(4)
scale_He    = 0.75
mean_O      = np.log(14)
scale_O     = 0.75
mean_Fe     = np.log(56)
scale_Fe    = 0.75

H_a, H_b    = (bound_a - mean_H) / scale_H, (bound_b - mean_H) / scale_H
He_a, He_b  = (bound_a - mean_He) / scale_He, (bound_b - mean_He) / scale_He
O_a, O_b    = (bound_a - mean_O) / scale_O, (bound_b - mean_O) / scale_O
Fe_a, Fe_b  = (bound_a - mean_Fe) / scale_Fe, (bound_b - mean_Fe) / scale_Fe

rv_H  = truncnorm(H_a, H_b,loc=mean_H, scale=scale_H)
rv_He = truncnorm(He_a, He_b,loc=mean_He, scale=scale_He)
rv_O  = truncnorm(O_a, O_b,loc=mean_O, scale=scale_O)
rv_Fe = truncnorm(Fe_a, Fe_b,loc=mean_Fe, scale=scale_Fe)

### Sampling data set
H_data  = rv_H.rvs(nH,random_state=seed)
He_data = rv_He.rvs(nHe,random_state=seed)
O_data  = rv_O.rvs(nO,random_state=seed)
Fe_data = rv_Fe.rvs(nFe,random_state=seed)

data = [H_data,He_data,O_data,Fe_data]

data_flat = np.concatenate(data)

###Plotting PDFs
fig1, ax = plt.subplots(1, 1)

ax.plot(x, rv_H.pdf(x),'r-', lw=2, alpha=0.6, label='H pdf')
ax.plot(x, rv_He.pdf(x),'y-', lw=2, alpha=0.6, label='He pdf')
ax.plot(x, rv_O.pdf(x),'g-', lw=2, alpha=0.6, label='O pdf')
ax.plot(x, rv_Fe.pdf(x),'b-', lw=2, alpha=0.6, label='Fe pdf')
ax.legend(loc=0)

### Ploting histogram and create subplot for fit results
fig2, (ax1,ax2) = plt.subplots(2, 1)

ax1.hist(data, bins=bins, density=False, stacked=True, histtype='step', color=['r','orange','g','b'])
ax2.hist(data_flat, bins=bins, density=False) # Changed density to False to line up with absolute counts

### Create list of the template PDFs or functions
template_pdfs = [rv_H.pdf,rv_He.pdf,rv_O.pdf,rv_Fe.pdf]

### Run template fitting method binned or unbinned
template = Template_Analysis(minos=run_minos,binned=set_binned,strategy=set_strategy)
template.join_pdfs(template_pdfs)

template.template_likelihood(data_flat, bins, fit_range)

# Create a list matching the template_pdfs order: [H, He, O, Fe]
true_values = [nH, nHe, nO, nFe] #None

if set_binned:
    print("Binned")
    template.draw(trues=true_values, parts=True, bins=len(bins), ax=ax2)
else:
    print("Unbinned")
    template.draw(trues=true_values, parts=True, bins=len(bins), ax=ax2)

plt.show()