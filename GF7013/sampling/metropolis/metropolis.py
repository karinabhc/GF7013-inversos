"""
Metropolis Algorithm

Generates samples from a pdf using the Metropolis Algorithm

Francisco Ortega Culaciati
ortega.francisco@uchile.cl
GF7013 - Metodos Inversos Avanzados
Departamento de Geofisica - FCFM - Universidad de Chile 

"""
COMPLETAR = None
import numpy as np
from ...model_parameters import ensemble

def metropolis(m0, likelihood_fun, pdf_prior, proposal, num_samples, num_burnin, 
               use_log_likelihood = True, save_samples = True, beta = 1,
               LogOfZero = None, rng = None, seed = None):
    """
    Performs the Metropolis Algorithm.
    The first model in the MCMC chain is m0, and steps of the chain are proposed by
    the proposal distribution, m_test = proposal.propose(m).

    - m0 : initial model of the MCMC chain
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
    - num_samples: Number of samples to be produced by the Metropolis algorithm after
                   the burn-in iterations (if save_samples == True, these samples
                   are saved in an ensemble).
    - num_burnin: number of initial Burn-in iterations of the Metropolis algorithm.
    - use_log_likelihood: if True, uses the logarithm of the likelihood of fprior and
                         likelihood_fun to evaluate the acceptance probabilities
                         in Metropolis algorithm. Thus, needs fprior.log_likelihood(m)
                         and likelihood_fun.log_likelihood(m). If False, uses the actual
                         likelihood values, computed from fprior.likelihood(m) and 
                         likelihood_fun.likelihood(m) to evaluate acceptance probability.
    - save_samples: if True, saves num_samples (after burn-in) in an ensemble instance.
    - beta: the exponent of the likelihood function for TMCMC algorithm. default value is
            beta = 1 ,i.e., samples the posterior distribution instead of intermediate
            distributions defined in TMCMC (for beta < 1).
    - LogOfZero : number to assign to approximate the Logarithm of zero (that may be 
                  calculated for self.log_likelihood in some pdf's). 
                  Default is "-NP.finfo(float).max * (NP.finfo(float).eps**2)"
                  This is to avoid dealing with -NP.inf. 
                  This number must be the same as used in a probability function, 
                  pdf_uniform, for instance.
    - rng : an instance of random number generator. If None, it instantiates 
            np.random.default_rng() if seed is None, or np.random.default_rng(seed)
            if seed is given. 
    - seed: an integer number (or a value generated with a seed generator) used as a
            seed for the random number generator. If rng is not None, seed is not used.

                          
    """
    # setup the random number generator
    if rng is None:
        if seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed = seed)
    else:
        rng = rng

    # update the random number generator of the proposal distribution.
    proposal.reset_random_number_generator(rng=rng, seed=seed)

    # approximated value for the logarithm of 0
    if LogOfZero is None:
        LogOfZero = -np.finfo(float).max * (np.finfo(float).eps**2)
    

    if save_samples:
        # initialize container to save the samples
        samples = ensemble(Npar= len(m0), Nmodels= num_samples, 
                           use_log_likelihood= use_log_likelihood,
                           beta= beta)
    else:
        samples = None

    # determine the functions to use to compute likelihoods or their natural logarithm
    # NOTE BELOW that we are assigning the appropriate functions to the variables
    if use_log_likelihood:
        likelihood = likelihood_fun.log_likelihood
        fprior = pdf_prior.log_likelihood
        acceptance_criteria = __acceptance_criteria_log_likelihood
        # define how to compute posterior log_likelihood
        fpost = _log_fpost
    else:
        likelihood = likelihood_fun.likelihood
        fprior = pdf_prior.likelihood
        acceptance_criteria = __acceptance_criteria_likelihood
        # define how to compute posterior likelihood
        fpost = _fpost

    
    # evaluate pdf on initial model
    m = m0
    fm_prior = fprior(m)
    fm_like = likelihood(m)
    fm_post = fpost(fm_prior= fm_prior, fm_like= fm_like, beta= beta)

    # check that initial model  has nonzero probability for the prior distribution
    # this can effectively be done only when calculating log_likelihoods
    if use_log_likelihood:
        if fm_prior == LogOfZero:
            raise ValueError('Initial model has null prior probability')
    else: # normal likelihood for prior distribution
        if fm_prior <= 1000 * np.finfo('float').smallest_normal:
            raise ValueError('Initial model has null prior probability')

    # do Metropolis algorithm iterations.
    num_accepted_transitions = 0
    num_iterations = num_burnin + num_samples

    for k in range(0, num_iterations):

        # obtain the test model using the proposal distribution and compute its
        # likelihoods or log_likelihoods as corresponds.
        m_test = proposal.propose(m)
        fm_prior_test = fprior(m_test)
        # use Metropolis acceptance criteria to see if proposed model is a sample
        # of the prior probability distribution
        accept_prior = acceptance_criteria(fm_test= fm_prior_test, fm= fm_prior)
        # only if m_test is sample of prior, proceed to check if it is sample of posterior
        # otherwise the transition is rejected.
        if accept_prior:
            fm_like_test = likelihood(m_test)
            fm_post_test = fpost(fm_prior= fm_prior_test, fm_like= fm_like_test,
                                 beta= beta)

            # Use Metropolis criteria to accept transition to m_test with some probability
            accept = acceptance_criteria(fm_test= fm_post_test, fm= fm_post)

            if accept:
                m = COMPLETAR
                fm_prior = COMPLETAR
                fm_like = COMPLETAR
                fm_post = COMPLETAR
                num_accepted_transitions += 1
        # save samples after burn-in period if requested to do so
        if save_samples:
            if k >= num_burnin:
                samples.m_set[k - num_burnin,:] = COMPLETAR
                samples.fprior[k - num_burnin] = COMPLETAR
                samples.like[k - num_burnin] = COMPLETAR
                samples.f[k - num_burnin] = COMPLETAR

    # compute acceptance ratio of the MCMC chain
    acceptance_ratio = num_accepted_transitions/num_iterations
    # assemble results and return
    result = {}
    result['m'] = m
    result['fprior'] = fm_prior
    result['like'] = fm_like
    result['fpost'] = fm_post
    result['acceptance_ratio'] = acceptance_ratio
    result['samples'] = samples # it may be None if not asked to save
    return result

#### acceptance criteria when using likelihood
def __acceptance_criteria_likelihood(fm, fm_test):
    """
    Metropolis acceptance criteria when fm, fm_test are likelihoods (not log_likelihoods)
    Returns True if model is accepted, False if not.
    """
    # if likelihood of m_test is larger, do accept.
    accept = COMPLETAR
    # if likelihood is smaller, accept with probability Pac = fm_test/fm
    if not(accept):
        u = COMPLETAR
        if COMPLETAR:
            accept = True
    
    return accept

def __acceptance_criteria_log_likelihood(fm, fm_test):
    """
    Metropolis acceptance criteria when fm, fm_test are the natural logarithm of the 
    likelihoods (i.e., log_likelihoods)
    Returns True if model is accepted, False if not.
    """
    # if likelihood of m_test is larger, do accept.
    accept = COMPLETAR
    # if likelihood is smaller, accept with probability Pac = fm_test/fm
    if not(accept):
        u = COMPLETAR
        if COMPLETAR:
            accept = True
    
    return accept

# provide functions to calculate fpost when using (log)likelihoods for prior 
# and likelihood function
### calculate logfpost
def _log_fpost(fm_prior, fm_like, beta): 
    return fm_prior + beta * fm_like
### calculate fpost
def _fpost(fm_prior, fm_like, beta): 
    return fm_prior * (fm_like**beta)

