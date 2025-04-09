# -*- coding: utf-8 -*-
"""
Prof. Francisco Hernan Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM
Universidad de Chile


Defines a base class for a probability density function class.
It must provide 5 basic functions inherited from pdf_base

self.likelihood(x): gives the value of the probability function (unnormalized).
sef.log_likelihood(x): gives the value of the log of probability function (unnormalized).
self.pdf(x): gives the value of the normalized pdf
self.log_pdf(x): gives the log of the value of the normalized pdf.
self.draw(): produces a numpy array with realizations of the distribution.  
"""
from .pdf_base import pdf_base
import numpy as np

COMPLETAR = None

class pdf_uniform_nD(pdf_base):
    """
    Defines a class for a n-dimensinoal uniform probability density function.
    """
    ####
    def __init__(self, par, LogOfZero = None, rng = None):
        """
        par: a dictionary containing the parameters that define the pdf.
        allowed keys are:
           'lower_lim': a 1D array with lower limits of random variable.
           'upper_lim': a 1D array with upper limits of random variable.
        The limits are defined to be finite, and so that if x is the random
        vector (1D array), 
              lower_lim[i] <= x[i] <= upper_lim[i]
        also, it must have that 
              lower_lim[i] < upper_lim[i], i.e., the limits must define a finite interval

        LogOfZero : number to assign to approximate the Logarithm of zero (that must be 
                    calculated for self.log_likelihood if the variable falls outside of
                    the region with non-null probability). 
                    Default is "-NP.finfo(float).max * (NP.finfo(float).eps**2)"
                    This is to avoid dealing with -NP.inf.

        rng: an instance of a numpy random number generator. If rng is None we use
            numpy.random.default_rng() 

        NOTA:  considere que las diferentes componentes del vector aleatorio de n
        dimensiones son independientes entre si. Utilize esto para programar el método 
        inverso en la función self._draw()

        """
        # The base class constructor assigns self.par = par
        super().__init__(par, rng)
        
        # make relevant parameters easier to access
        self.ll = self.par['lower_lim']
        self.ul = self.par['upper_lim']
        self.N = len(self.ll)
        # approximated value for the logarithm of 0
        if LogOfZero is None:
            self.LogZero = -np.finfo(float).max * (np.finfo(float).eps**2)
        else:
            self.LogZero = LogOfZero

        # compute normalization constant
        self.normalization = COMPLETAR
        self.log_normalization = COMPLETAR
    
    ####
    def __check_x(self, x):
        """
        Check that the array x has the correct shape and size
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
        x must be a numpy array of 1 D.
        """
        x = self.__check_x(x)

        if COMPLETAR:
            return 0
        else:
            return self.LogZero

    ####
    def _likelihood(self, x):
        """
        computes the likelihood of the vector value x. 
        x must be a numpy array of 1 D.
        
        """
        x = self.__check_x(x)
 
        if COMPLETAR:
            return 1.0
        else:
            return 0.0


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
        if self.normalization is None:
            return None
        else:
            x = self.__check_x(x) 
            return  self._likelihood(x) * self.normalization

    ####
    def _draw(self, Ns):
        """
        produces numpy array with Ns realizations of the probability distribution.
        If Ns is None, returns a 1D numpy array with the shape of self.ll .
        if Ns is a positive integer, returns a 2D numpy array where the number of rows
        is the length of self.ll and the number of columns is Ns (i.e., each column of the
        array is a sample of the multivariate uniform distribution).

        Nota: Use el método inverso para generar muestras de la distribución.

        
        """
        sample = COMPLETAR
        
        return sample
           
