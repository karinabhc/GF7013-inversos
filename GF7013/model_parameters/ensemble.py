"""
Defines an ensemble of models to be used in sampling methods.

Francisco Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM - Universidad de Chile 

"""
# imports
import numpy as NP

class ensemble(object):
    """
    Defines a data structure of an ensemble of model_parameters, 
    The ensemble is defined as a set of model_parameters and their 
    associated likelihoods or probability density values (or their natural logarithm if 
    use_log_likelihood = True - default)

    The model parameter set is a numpy array whose corresponds to a different 
    model_parameters. Thus the number of columns (Npar) 
    """

    def __init__(self, Npar, Nmodels, use_log_likelihood = True, beta = 1):
        """
        Npar = number of parameters for each model_parameters
        Nmodels = number of model_parameters in the set.
        use_log_likelihood = True if likelihood/densities stored values correspond to the
                         natural logarithm of such values.
        beta = Value of the Likelihood exponent for TMCMC

        The data structure also defines the following variables:
            - m_set : numpy array (2D) that stores the sampled models (each model a row)
            - fprior: stores values of (log)likelihood/densities of the prior distribution
            - like: stores values of (log)likelihood/densities of the likelihood function
            - f: stores values of (log)likelihood/densities of the posterior distribution
        """
        self.Npar = Npar # number of parameters of each model
        self.Nmodels = Nmodels # number of models of the ensemble
        self.beta = beta # beta value of TMCMC for ensembles.
        # define the set of model_parameters as numpy array (each row is a model)
        self.m_set = NP.zeros((Nmodels, Npar))
        # define wether likelihood/densities or their logarithm are used.
        self.use_log_likelihood = use_log_likelihood
        # storage for the likelihood/densities 
        # values for all m conforming the rows of m_set
        self.fprior = NP.zeros(Nmodels) # prior distribution (log)likelihood
        self.like = NP.zeros(Nmodels) # (log)likelihood function 
        self.f = NP.zeros(Nmodels) # posterior distribution (log) likelihood


    
