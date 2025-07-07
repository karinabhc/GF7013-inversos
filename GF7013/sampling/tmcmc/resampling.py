"""
Computes the resampling for a step in the Transitional Markov Chain Montecarlo - TMCMC

The calculation is based on Minson et al. (2013).

- S. E. Minson, M. Simons, J. L. Beck (2013). Bayesian inversion for finite fault 
earthquake source models I—theory and algorithm, Geophysical Journal International, 
Volume 194, Issue 3, Pages 1701–1726, https://doi.org/10.1093/gji/ggt180.
  

Francisco Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM - Universidad de Chile 

"""
COMPLETAR = None
import numpy as NP
from copy import deepcopy
def resampling(m_ensemble, dbeta):
    """
    Resampling for a step in the TMCMC algorithm.
    """
    # determine the weight of weach sample according to change in beta
    if m_ensemble.use_log_likelihood:
        w = COMPLETAR
    else:
        w = COMPLETAR
    w = w / NP.sum(w)

    # determine the frequency of selection of each model using a multinomial distribution
    COMPLETAR = COMPLETAR
    # compute resampled ensemble for like**(beta + dbeta)
    m_resampled_ensemble = deepcopy(m_ensemble)
    COMPLETAR = COMPLETAR

    return m_resampled_ensemble

