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
#COMPLETAR = None
import numpy as np
from copy import deepcopy


def resampling(m_ensemble, dbeta):
    """
    Resampling for a step in the TMCMC algorithm.
    """
    # determine the weight of weach sample according to change in beta
    if m_ensemble.use_log_likelihood:
        w = np.exp(dbeta * m_ensemble.like) #Caso 2
    else:
        w = m_ensemble.like ** dbeta #Caso 1
    w = w / np.sum(w) # normalize weights

    # determine the frequency of selection of each model using a multinomial distribution
    indices = np.random.choice(m_ensemble.Nmodels, size=m_ensemble.Nmodels,
                               replace=True, p=w)
    # resample the ensemble
    # compute resampled ensemble for like**(beta + dbeta)
    m_resampled_ensemble = deepcopy(m_ensemble)
    m_resampled_ensemble.m_set = m_ensemble.m_set[indices]
    m_resampled_ensemble.fprior = m_ensemble.fprior[indices]
    m_resampled_ensemble.like = m_ensemble.like[indices]
    m_resampled_ensemble.f = m_ensemble.f[indices]

    return m_resampled_ensemble

