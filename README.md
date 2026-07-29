# Code examples and supporting data for the paper

# [_Phase retrieval based on intensity-only spatiotemporal wavefront shaping_]

## **Luis Alberto Razo-López [![](./ressources/logos/orcid.svg)](https://orcid.org/0000-0002-1308-8377) [![](./ressources/logos/gscholar.svg)](https://scholar.google.com/citations?user=UUp0d-0AAAAJ&hl=es&authuser=1), Marc Guillon [![](./ressources/logos/orcid.svg)](https://orcid.org/0000-0002-9945-4605) [![](./ressources/logos/gscholar.svg)](https://scholar.google.fr/citations?hl=es&authuser=1&user=xfrA-JUAAAAJ), Fabrice Lemoult [![](./ressources/logos/orcid.svg)](https://orcid.org/0000-0001-9757-0760) [![](./ressources/logos/gscholar.svg)](https://scholar.google.fr/citations?hl=es&authuser=1&user=Gy6ImbgAAAAJ), and S. M. Popoff [![](./ressources/logos//orcid.svg)](https://orcid.org/0000-0002-7199-9814) [![](./ressources/logos/gscholar.svg)](https://scholar.google.com/citations?user=2OuzjokAAAAJ)**

**Global requirements:**

- Numpy
- Seaborn
- Matplotlib
- Scipy
- numba
- colorsys
- tqdm
- torch

**Raw data files \[Optional\]:**

- Only required to regenerated the curated data files (in [/Data](Data/)) used to produce the results of the paper. Available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21677183.svg)](https://doi.org/10.5281/zenodo.21677183)


## Content of each file:

- [`Numerics.ipynb`](Numerics.ipynb) Contains the basics of spatiotemporal wavefront shaping: Creation of the temporal patterns, analysis of the envelops and Harmonics decomposition when modulating with a flat phase or with a random phase plate. This corresponds to figures 1, S1, and S3.
It also show the phase reconstruction and the comparison with respect to the ground truth in both real and fourier planes. Also analyses the correlation spectrum and the vortices positions, correspoding to figures S4, S5 and S6.

- [`Experiments.ipynb`](Experiments.ipynb) Contains the experimental validation of Numerics: spatiotemporal patterns, and their envelops and Harmonics decomposition when measuring without any diffuser, this corresponds to figures 2. Then we show how after reconstruction we computationally recover patterns identical to those measured, and we can generate wavefront to experimetally correct the beam, corresponding to figures 4 and 5.

- [`diverfun.py`](diverfun.py) Contains the common functions used in the notebooks for data processing and vizualisation.

- [`phaseRet.py`](phaseRet.py) Contain the functions of the gradient descent algorithm to find the near and far field phases.
Usage : 
        import phaseRet as pr
        pr.FindBigDiffuser([a0, S0, S1], Div=2, DiffSize=180, MaxLoss=0.5)
In this case: a0, S0, and S1 are the imput quantities of size different than 180. The diffuser to find have a size 180x180 and the maximal allowed loss is 0.5.
