"""
Metropolis Algorithm

Generates samples from a pdf using the Metropolis Algorithm

Francisco Ortega Culaciati
ortega.francisco@uchile.cl
GF7013 - Metodos Inversos Avanzados
Departamento de Geofisica - FCFM - Universidad de Chile 

"""
COMPLETAR = None
import numpy as NP
from ...model_parameters import ensemble
from ..metropolis import metropolis

def metropolis_in_parallel_SERIAL(m0, likelihood_fun, pdf_prior, proposal, num_MCMC_steps,  
               use_log_likelihood = True):
    """
    Performs the Metropolis in Parallel Algorithm. THIS IS THE SERIAL VERSION OF THE 
    ALGORITHM, THUS IT RUNS ONE MCMC CHAIN AT EACH TIME.

    - m0 : initial ensemble of models for the MCMC chains.
    - likelihood_fun: an object that provides the functions likelihood_fun.likelihood(m)
                      and likelihood_fun.log_likelihood(m) that return the value or 
                      natural logarithm of the value of the likelihood function used
                      for the inverse problem.
    - pdf_prior: an object that provides the functions fprior.likelihood(m) and 
                      fprior.log_likelihood(m) that return the value or 
                      natural logarithm of the value of the prior probability
                      function on model parameters used for the inverse problem.
    - proposal: an object that provides the function proposal.propose(m) that returns
               a model m_test, proposed as the next step of the MCMC chain.
    - num_MCMC_steps: Number of MCMC steps of each Metropolis algorithm produced in 
                      parallel.
    - use_log_likelihood: if True, uses the logarithm of the likelihood of fprior and
                         of likelihood_fun to evaluate the acceptance probabilities
                         in Metropolis algorithm. Thus, needs fprior.log_likelihood(m)
                         and likelihood_fun.log_likelihood(m). If False, uses the actual
                         likelihood values, computed from fprior.likelihood(m) and 
                         likelihood_fun.likelihood(m) to evaluate acceptance probability.
    NOTE: the exponent beta of the likelihood function for TMCMC algorithm must be defined
          in the initial ensemble of models (use beta=1 if not performing TMCMC).
                          
    """

    
    m = COMPLETAR # this is the final ensemble after Metropolis in Parallel.

    acceptance_ratios = COMPLETAR # a 1D numpy array with the acceptance ratio of each
                                  # MCMC chain.
    return m, acceptance_ratios

