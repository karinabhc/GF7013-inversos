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
import numpy as NP

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
        
        self.normalization = 1.0 / np.prod(self.ul - self.ll)
        self.log_normalization = self.LogZero + np.log(self.normalization)
    
    ####
    def __internal_checks(self, tol_zero = 1000):
        """
        determine the number of dimensions of the random variable based in the limits
        also check limits consistency.
        """
        if 'lower_lim' in self.par.keys():
            llshape = self.par['lower_lim'].shape
            if len(llshape) != 1:
                raise ValueError("'lower_lim' must be a 1D numpy array")
            else:
               NumLL = len(self.par['lower_lim'])
        else:
            raise ValueError("'lower_lim' must be a key of par dictionary.")

        if 'upper_lim' in self.par.keys():
            ulshape = self.par['upper_lim'].shape
            if len(ulshape) != 1:
                raise ValueError("'upper_lim' must be a 1D numpy array")
            else:
               NumUL = len(self.par['upper_lim'])
        else:
            raise ValueError("'upper_lim' must be a key of par dictionary.")
   
        if NumUL == NumLL:
            self.N = NumUL
        else:
            raise ValueError("The size of 'upper_lim' and 'lower_lim' must be the same.") 
               
        # check that intervals are finite
        dif = NP.abs(self.par['lower_lim'] - self.par['upper_lim'])
        if NP.min(dif) < NP.finfo(float).eps * tol_zero :
           raise ValueError('at least one interval of the boundaries is not finite.')
        # check right order of limits
        if ( self.par['lower_lim'] > self.par['upper_lim'] ).any():
            raise ValueError('At least one lower_limit is larger than one upper_limit.')
       
        # compute normalization constant.
        self.normalization = 1 / NP.prod(dif) # is the size of the uniform volume.
        if self.normalization < 1E3 * NP.finfo(float).eps:
           print('Floating point overflow when calculating the '
                        +'normalization constant.')
           print('Use Likelihood values instead of pdf.')
           self.normalization = None

    ##############       
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
        if (x >= self.ll).all() and (x <= self.ul).all():
            return 0.0
        else:
            return self.LogZero

    ####
    def _likelihood(self, x):
        """
        computes the likelihood of the vector value x. 
        x must be a numpy array of 1 D.
        SI UN ELEMENTO ESTA FUERA DEL RANGO ES 0
        """
        x = self.__check_x(x)
 
        if (x >= self.ll).all() and (x <= self.ul).all():
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
        # generar xi = a + (b-a) * u, donde u es una variable aleatoria uniforme entre 0 y 1
        # y xi es la variable aleatoria uniforme entre a y b
        if Ns is None:
            # Generate a single sample (1D array) using the inverse method  with the shape of self.ll 
            u = self.rng.uniform(0.0, 1.0, size=self.N)
            return self.ll + (self.ul - self.ll) * u
        else:
            # Generate Ns samples (2D array) using the inverse method where the number of rows
            #is the length of self.ll and the number of columns is Ns 
            #u = self.rng.uniform(0.0, 1.0, size=(Ns, self.N))
            #return self.ll + (self.ul - self.ll) * u
            # Generate a 2D array of shape (Ns, self.N) using the inverse method
            samples = [self.ll + (self.ul - self.ll) * self.rng.uniform(0.0, 1.0, size=self.N)
                                                                            for _ in range(Ns)]
            samples = np.array(samples, dtype=float).T
            print(samples.shape)
            return samples