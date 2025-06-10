# -*- python -*-
# -*- coding: utf-8 -*-
"""
Prof. Francisco Hernan Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM
Universidad de Chile

Defines a base class for a probability density function class.
It must provide 5 member functions. 
self.likelihood(x): gives the value of the probability function (unnormalized).
sef.log_likelihood(x): gives the value of the log of probability function (unnormalized).
self.pdf(x): gives the value of the normalized pdf
self.log_pdf(x): gives the log of the value of the normalized pdf.
self.draw(): produces realizations of the distribution. 
"""
import numpy as np

class pdf_base(object):
    """
    Defines a base class for a probability density function class.
    It must provide 3 basic functions. 
    self.likelihood(x): gives the value of the probability function (unnormalized).
    sef.log_likelihood(x): gives the value of log of probability function (unnormalized).
    self.pdf(x): gives the value of the normalized pdf
    self.log_pdf(x): gives the log of the value of the normalized pdf.    
    self.draw(N = None): produces realizations of the distribution. 
    """
    ####
    def __init__(self, par, rng = None):
        """
        par: a dictionary containing the parameters that define the pdf 
        rng: an instance of a numpy random number generator. If rng is None we use
            numpy.random.default_rng()
        """
        self.par = par
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng
        self.normalization = None # float variable to store normalization constant
        self.log_normalization = None # natural logarithm of normalization constant
        self.type = 'base' # a string with the name of the distribution
        
    ############# USER INTERFACE ########    
    ####
    def reset_random_number_generator(self, rng = None, seed = None):
        """
        When using parallel computing, it is necessary to reset the random number 
        generator with different seeds in each computing core. This can be done by passing
        a new instance of a random number generator (e.g., np.random.default_rng()) or by 
        creating an instance of np.random.default_rng() if seed is None or if seed is 
        given, by instantiating np.random.default_rng(seed = seed).
        """
        if rng is None:
            if seed is None:
                self.rng = np.random.default_rng()
            else:
                self.rng = np.random.default_rng(seed = seed)
        else:
            self.rng = rng
    ####
    def likelihood(self, x):
        """
        computes the likelihood of the vector value x. 
        x must be a numpy array of 1 D, or a column vector 2D numpy array.
        """ 
        return self._likelihood(x)
    
    ####
    def log_likelihood(self, x):
        """
        computes the base e logarithm of the likelihood of the vector value x. 
        x must be a numpy array of 1 D, or a column vector 2D numpy array.
        """
        return self._log_likelihood(x)
    
    ####
    def pdf(self, x):
        """
        computes the value of the probability density function (i.e., normalized pdf) 
        for x.
        """
        return self._pdf(x)

    ####
    def log_pdf(self, x):
        """
        computes the value of the probability density function (i.e., normalized pdf) 
        for x.
        """
        return self._log_pdf(x)

    ####
    def draw(self, Ns = None): 
        """
        produces a numpy array with a realization of the probability distribution.
        
        """
        return self._draw(Ns = Ns)

    ############### INTERNAL FUNCTIONS THAT MUST BE IMPLEMENTED AT CHILD CLASS.
    def _likelihood(self, x):
        """
        computes the likelihood of the vector value x. 
        x must be a numpy array of 1 D, or a column vector 2D numpy array.
        """
        raise NotImplementedError('User has not implemented likelihood(x)')

    ####
    def _log_likelihood(self, x):
        """
        computes the base e logarithm of the likelihood of the vector value x. 
        x must be a numpy array of 1 D, or a column vector 2D numpy array.
        """
        raise NotImplementedError('User has not implemented likelihood(x)')

    ####
    def _pdf(self, x):
        """
        computes the value of the probability density function (i.e., normalized pdf) 
        for x.
        """
        raise NotImplementedError('User has not implemented pdf(x)')

    ####
    def _log_pdf(self, x):
        """
        computes the base e logarithm of the value of the probability density 
        function (i.e., normalized pdf) for x.
        """
        raise NotImplementedError('User has not implemented pdf(x)')

    ####
    def _draw(self, Ns = None):
        """
        produces a numpy array with Ns realizations of the probability distribution.

        
        """
        raise NotImplementedError('User has not implemented draw(x)')

