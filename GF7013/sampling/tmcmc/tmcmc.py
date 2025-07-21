"""
Transitional Markov Chain Monte Carlo (TMCMC)
based on the Metropolis in Parallel Algorithm (multiprocessing.Pool version)

Generates samples from a posterior pdf using the TMCMC algorithm based on 
the Metropolis in Parallel algorithm (parallel version using multiprocessing.Pool)

IMPORTANT: see tmcmc_metropolis_pool docstring for setting the environment 
           variables that control the number of threads used by each parallel process.

Francisco Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM - Universidad de Chile 
"""
import numpy as np
from copy import deepcopy
from ..metropolis_in_parallel import metropolis_in_parallel_POOL
from .calc_dbeta import calc_dbeta
from .resampling import resampling

def tmcmc_pool(m0_ensemble, likelihood_fun, pdf_prior, proposal, 
                                num_MCMC_steps, num_proc = None, chunksize = 1,
                                use_resampling = False):
    """
    Performs the Transitional Markov Chain Monte Carlo Algortihm (based on Metropolis).
    THIS IS A PARALLELIZED VERSION OF THE ALGORITHM, THUS IT RUNS AS MANY MCMC CHAINS AT 
    THE SAME TIME AS THE NUMBER OF COMPUTING PROCESSORS THAT ARE AVAILABLE. 
    THIS CODE USES multiprocessing.Pool
    https://docs.python.org/3/library/multiprocessing.html

    REMEMBER TO SET THIS ENVIRONMENT VARIABLES IN THE MAIN SCRIPT OF THE CODE BEFORE
    IMPORTING ANY OTHER PACKAGE
    # set the number of threads for numpy
      import os
      os.environ["OMP_NUM_THREADS"] = "1"
      os.environ["MKL_NUM_THREADS"] = "1"
      os.environ["MKL_DOMAIN_NUM_THREADS"] = "1"
      os.environ["OPENBLAS_NUM_THREADS"] = "1"
      os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
      os.environ["NUMEXPR_NUM_THREADS"] = "1"

    - m0_ensemble : initial ensemble of models for the MCMC chains.
    - likelihood_fun: an object that provides the functions likelihood_fun.likelihood(m_ensamble)
                      and likelihood_fun.log_likelihood(m_ensamble) that return the value or 
                      natural logarithm of the value of the likelihood function used
                      for the inverse problem.
    - pdf_prior: an object that provides the functions fprior.likelihood(m_ensamble) and 
                      fprior.log_likelihood(m_ensamble) that return the value or 
                      natural logarithm of the value of the prior probability
                      function on model parameters used for the inverse problem.
    - proposal: an object that provides the function proposal.propose(m_ensamble) that returns
               a model m_test, proposed as the next step of the MCMC chain.
    - num_MCMC_steps: Number of MCMC steps of each Metropolis algorithm produced in 
                      parallel.
    - num_proc: Number of cores to use within multiprocessing.Pool. If None, it will 
                use multiprocessing.cpu_count() to determine the number of cores.
    - chunksize: number of tasks given at the same time to each parallel process. See
                 multiprocessing.Pool documentation for further details.
    - use_resampling: if True, performs resampling of the current samples after
                      updating the beta value.
                         
    NOTE: the exponent beta of the likelihood function for TMCMC algorithm and if
          the algorithm uses likelihood or log_likelihood values are defined in the
          initial ensemble of models in the variables beta and use_log_likelihood
          The algorithm will start to iterate using beta given in m0_ensemble.
    """
    
    m_ensemble = deepcopy(m0_ensemble)
    # initialize values of fprior, like and f in the initial models ensemble
    # m_ensemble.f = m_ensemble.fprior * (m_ensemble.like ** m_ensemble.beta if not m_ensemble.use_log_likelihood else
                        #   np.exp(m_ensemble.beta *likelihood_fun.log_likelihood(m_ensemble)))

    # Inicializa likelihood y f antes de comenzar TMCMC
    if m_ensemble.use_log_likelihood:
        for i in range(m_ensemble.Nmodels):
            m_ensemble.like[i] = likelihood_fun.log_likelihood(np.array([m_ensemble.m_set[i, :]]))
        print(m_ensemble.like)
        m_ensemble.f = m_ensemble.fprior * np.exp(m_ensemble.beta * m_ensemble.like)
    # else:
    #     m_ensemble.like = np.ones_like(m_ensemble.like)
    #     m_ensemble.fprior = np.ones_like(m_ensemble.like)
    #     for i in range(m_ensemble.Nmodels):
    #         m_ensemble.like[i] = likelihood_fun.likelihood(np.array([m_ensemble.m_set[i, :]]))
    #     print(m_ensemble.like)

    #     m_ensemble.f = m_ensemble.fprior * (m_ensemble.like ** m_ensemble.beta)
        
        
    # do the iterations (initial beta value must be defined in the ensemble)
    beta = m_ensemble.beta
    acceptance_ratios = []

    while beta < 1:
        dbeta = calc_dbeta(m_ensemble)
        beta = min(1.0, beta + dbeta)
        print(f'beta_previo:{ m_ensemble.beta}') 
        m_ensemble.beta = beta  # actualizar en el ensemble
        print(f'beta_nuevo:{ m_ensemble.beta}\n')
        # for m_ensamble in m_ensemble: #para nuevo beta
        if m_ensemble.use_log_likelihood:
            m_ensemble.f = m_ensemble.fprior * np.exp(beta * likelihood_fun.log_likelihood(m_ensemble))
        else:
            m_ensemble.f = m_ensemble.fprior * (likelihood_fun.likelihood(m_ensemble) ** beta)
        if use_resampling:   # siii  use_resampling = true
            m_ensemble = resampling(m_ensemble)

        m_ensemble, acc_ratio = metropolis_in_parallel_POOL(   #Metropolis en paralelo
            m_ensemble, likelihood_fun, pdf_prior, proposal,
            num_MCMC_steps)

        acceptance_ratios.append(np.mean(acc_ratio))
    return m_ensemble, acceptance_ratios
