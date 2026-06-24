# cosmicray-analysis-tools

This Python library should help perform cosmic-ray mass-composition analysis using a template PDF. Similar to https://arxiv.org/abs/1906.04317

##### Version 0.0.3
* remove deprecated probfit dependency
##### Version 0.0.2
* extended likelihood template fit method
* unbinned and binned fit method available
* The number of events per template is constrained to the maximum number of data
* add setup.py

##### ToDO:
* Add Numba support and general speedup code for unbinned dataset fits
* Test weighted event data sets 
