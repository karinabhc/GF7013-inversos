"""
Computes the dbeta for a step in the Transitional Markov Chain Montecarlo - TMCMC

The calculation is based on the Ching and Chen (2007) paper. See also Minson et al. (2013).

- Ching, J., and Y.-C. Chen (2007), Transitional Markov chain Monte Carlo method for 
  Bayesian model updating, model class selection, and model averaging, Journal of 
  Engineering Mechanics, 133(7), 816–832, doi:10.1061/(ASCE)0733-9399(2007)133: 7(816).

- S. E. Minson, M. Simons, J. L. Beck (2013). Bayesian inversion for finite fault 
earthquake source models I—theory and algorithm, Geophysical Journal International, 
Volume 194, Issue 3, Pages 1701–1726, https://doi.org/10.1093/gji/ggt180.
  

Francisco Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM - Universidad de Chile 


"""
import numpy as NP
from scipy import optimize as opt

def calc_dbeta(m_ensemble, effective_sample_size = 0.5, 
               tol = 1E-12, maxiter = 1000,
               dbetaMIN = 1E-6):
    """
    computes dbeta for TMCMC algorithm.
    - m_ensemble : ensemble of samples (an ensemble class).
    - effective_sample_size : target effective sample size of ensemble after updating 
                              beta.
    - tol: a tolerance to find the root in the problem to determine delta beta.
    - maxiter: maximum number of iterations to consider when searching for delta beta
    - dbetaMIN: smallest value of delta beta to consider.

    This code must try to use Brent constrained algorithm. If Brent constrained can not be 
    used, the use constrained minimize_scalar scipy algorithm.

    Note that the calculation of dbeta must consider both cases: use of likelihood or
    the natural logarithm of the likelihoods. The decision must be based in the value
    of the boolean variable m_ensemble.use_log_likelihood

    """
    beta = m_ensemble.beta
    dbetaMAX = 1.1 - beta
    # dbetaMAX = 100 # initially I can allow beta > 1 but must truncate to beta = 1 
                   # if resulting beta is > 1.
    bounds = NP.log([dbetaMIN, dbetaMAX])  # dbetaMIN y dbtaMAX son log_dbeta
    if can_use_brent_constrained(bounds[0], bounds[1], m_ensemble, effective_sample_size): 
        print('Calculating dbeta using CONSTRAINED Brent algorithm')
        d_beta = _dbeta_brent_constrained(m_ensemble,
                                        bounds=bounds,
                                        effective_sample_size = effective_sample_size,
                                        tol = tol, 
                                        maxiter = maxiter)
        
    else:
        print('Calculating dbeta using alternate bounded minimize_scalar algorithm')
        # if constrained version does not work, try to a less restrictive approach
        d_beta = _dbeta_bounded(m_ensemble, 
                              bounds = bounds, 
                              effective_sample_size = effective_sample_size, 
                              tol = tol, 
                              maxiter = maxiter)
    # d_beta = NP.exp(lndbeta)
    print(f"bounds ={bounds}, b0 = {beta:.8e}, dbeta: {d_beta:.8e}, b_new: {beta+d_beta:.8e}")
    return d_beta

# objective function for Brent algorithm
def _phi_brent_constrained(dbeta, m_ensemble, effective_sample_size):
    """
    defines the objective function to minimize when calculating delta_beta (dbeta)
    using the constrained brent algorithm
    """
    # remember that lndbeta = dbeta 
    if m_ensemble.use_log_likelihood:   # m_ensemble.like values are the natural logarithm of the likelihood function
        loglikes = NP.asarray(m_ensemble.like) #weights
        # w = NP.exp(dbeta * (loglikes - NP.max(loglikes)))
        
    else:        # m_ensemble.like values are of the likelihood function
        # dbeta = NP.exp(dbeta)
        loglikes = NP.log(NP.asarray(m_ensemble.like)+1E-300)  
    dbeta = NP.exp(dbeta)
    w = NP.exp((loglikes - NP.max(loglikes)))**dbeta # evita errores numericos (overflow)
    w /= NP.sum(w)  # Normalize weights
    ESS_target = 1.0 / NP.sum(w ** 2)
    phi = ESS_target - effective_sample_size * len(w) 
    return  phi 

# function to check if I can use or not the Brent constrained algorithm
def can_use_brent_constrained(dbetaMIN, dbetaMAX,  m_ensemble, effective_sample_size):    
    phi_min = _phi_brent_constrained(dbetaMIN, m_ensemble, effective_sample_size)
    phi_max = _phi_brent_constrained(dbetaMAX, m_ensemble, effective_sample_size)
    status = phi_min * phi_max < 0
    return status
   
# objective function for opt.minimize_scalar bounded algorithm
def _phi_minimize_scalar(dbeta, m_ensemble, effective_sample_size):
    """
    defines the objective function to minimize when calculating delta_beta (dbeta)
    using the unconstrained brent algorithm
    """
    # remember that lndbeta = dbeta 
    if m_ensemble.use_log_likelihood:
        loglikes = NP.asarray(m_ensemble.like)
    else:
        loglikes = NP.log(NP.asarray(m_ensemble.like))
    dbeta = NP.exp(dbeta)

    w = NP.exp((loglikes - NP.max(loglikes)))**dbeta # evita errores numericos (overflow)
    w /= NP.sum(w)  # Normalize weights
    ESS_target = 1.0 / NP.sum(w ** 2)
    phi = (ESS_target - effective_sample_size * len(w))**2
    return phi



def _dbeta_brent_constrained(m_ensemble, bounds, effective_sample_size,
                             tol=1E-12, maxiter=1000):
    """
    computes dbeta for TMCMC using constrained Brent algorithm (see scipy.optimize.brenth)
    """
    lndbeta = opt.brenth(_phi_brent_constrained, bounds[0], bounds[1],
                         args=(m_ensemble, effective_sample_size),xtol=tol,
                         maxiter=maxiter) # the objective function gets lndbeta (output is lndbeta opt)
    dbeta = NP.exp(lndbeta)
    beta = m_ensemble.beta + dbeta

    if beta < 1.0:
        return dbeta
    else:
        return 1 - m_ensemble.beta



def _dbeta_bounded(m_ensemble, bounds, effective_sample_size, 
                    tol = 1E-8, maxiter = 1000):
    """
    computes dbeta for TMCMC using unconstrained Brent algorithm(see scipy.optimize.minimize_scalar)

    """
    lndbeta = opt.minimize_scalar(_phi_minimize_scalar,
                                bounds=bounds, args=(m_ensemble, effective_sample_size),
                                method='Bounded, tol=tol, options={'maxiter': maxiter}).x # el .x es el valor de dbeta porque minimize_scalar devuelve un objeto OptimizeResult que contiene el valor de la funcion objetivo y el valor de dbeta
    dbeta = NP.exp(lndbeta) 
    beta = m_ensemble.beta + dbeta
    
    if beta < 1.0:
        return dbeta
    else:
        return 1 - m_ensemble.beta
