# -*- coding: utf-8 -*-
"""
Prof. Francisco Hernan Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM
Universidad de Chile

Defines a class for a multivariate normal probability function.
It must provide 5 member functions, inherited from pdf_base

self.likelihood(x): gives the value of the probability function (unnormalized).
sef.log_likelihood(x): gives the value of the log of probability function (unnormalized).
self.pdf(x): gives the value of the normalized pdf
self.log_pdf(x): gives the log of the value of the normalized pdf.
self.draw(N = None): produces  realizations of the distribution. 
"""

from .pdf_base import pdf_base
import numpy as np

class pdf_normal(pdf_base):
    """
    Defines a class for a n-dimensional multivariate normal N(mu, cov) probability 
    function class.
    """
    ####
    def __init__(self, par, rng = None):
        """
        par: a dictionary containing the parameters that define the pdf.
        allowed keys are:
           'mu': a 1D array with the expected value (mean) of the normal pdf.
                 all values must be finite.
           'cov': a 2D array with the covariance matrix of the normal pdf.
                  must be a symmetric, nonsingular and positive definite matrix.
        rng: an instance of a numpy random number generator. If rng is None we use
            numpy.random.default_rng()
        """
        # The base class constructor assigns self.par = par
        super().__init__(par, rng)

        self.type = 'multivariate normal'

        
        # make relevant parameters easier to access
        self.mu = self.par['mu']
        self.cov = self.par['cov']
        self.N = len(self.mu)

        # compute inverse of covariance matrix needed for evaluation of likelihood/pdf
        self.inv_cov = np.linalg.inv(self.cov)
        # compute cholesky decomposition of self.cov for sample generation.
        self.right_chol_cov = np.linalg.cholesky(self.cov) # A.dot(A.T) = self.cov

        # compute normalization constant and base e logarithm of normalization.
        # (see self._pdf)
        sign, logdetCov = np.linalg.slogdet(self.cov)
        self.log_normalization = -0.5 * (self.N * np.log(2 * np.pi) + logdetCov)
        self.normalization = np.exp(self.log_normalization)

        if self.normalization < 1E3 * np.finfo(float).eps:
           print('Floating point overflow when calculating the '
                        +'normalization constant.')
           print('Use log_pdf or (log) Likelihood values instead of pdf.')
           self.normalization = None


    ####
    def __check_x(self, x):
        """
        Check that the array x has the correct shape and size
        array must be a single column 2D array or a 1D array
        """
        if x.ndim != 1:
            x = x.flatten()
            
        if len(x) != self.N:
           raise ValueError('x has size {:d}, must be a 1D array of length {:d}'.format(
                                                                          len(x), self.N))
        return x
        

    ####
    def _log_likelihood(self, x):
        """
        computes the base e logarithm of the likelihood of the vector value x. 
        x must be a numpy array of 1 D, or a column vector 2D numpy array.
        
        TODO: For the homework please compute directly the logarithm of the (unnormalized)
        likelihood ( DO NOT TAKE THE LOGARITHM OF THE PDF). You must program the formula.

        """
        x = self.__check_x(x)
        misfit = x - self.mu
        log_likelihood_value =-0.5 * misfit.T @ self.inv_cov @ misfit #prod matricial
        return log_likelihood_value
        
    ####
    def _likelihood(self, x):
        """
        computes the (unnormalized) likelihood of the vector value x. 
        x must be a numpy array of 1 D.

        TODO: A hint: use what you already coded!. 
        
        """
        LogLike = self._log_likelihood(x)

        return  np.exp(LogLike)
    
    ####
    def _log_pdf(self, x):
        """
        computes the value of the probability density function (i.e., normalized pdf) 
        for x.
        """
        return self._log_likelihood(x) + self.log_normalization
        
    ####
    def _pdf(self, x):
        """
        computes the value of the probability density function (i.e., normalized pdf) 
        for x.
        """
        return  self._likelihood(x) * self.normalization

    ####
    def _draw(self, Ns = None):
        """
        produces numpy array with Ns realizations of the probability distribution.
        If Ns is None, returns a 1D numpy array with the shape of self.mu .
        if Ns is a positive integer, returns a 2D numpy array where the number of rows
        is the length of self.mu and the number of columns is Ns (i.e., each column of the
        array is a sample of the multivariate normal pdf).

        HINT: Aquí puede usar el método relacional y el generador de números aleatorios 
        de python para la distribución Normal Estándar o Canónica N(0,1) dado por 
        la función de numpy rng.standard_normal().

        """
        
        if Ns is None:
            u = self.rng.standard_normal(self.N)
            sample = self.mu + self.right_chol_cov @ u
        else:
          # u = self.rng.standard_normal((self.N, Ns))
          # sample = self.mu.reshape(-1, 1) + self.right_chol_cov @ u
               
            #VERSION PASITO A PASITO 
            # Para varias muestras se crea una matriz para almacenarlas
            sample = np.zeros((self.N, Ns))
            
            for j in range(Ns):
                # Generamos un vector de N(0,1)
                u = self.rng.standard_normal(self.N)
                
                # transformación y = mu + A·x para cada columna jejeje
                sample[:, j] = self.mu + np.dot(self.right_chol_cov, u)
        
        return sample
           
