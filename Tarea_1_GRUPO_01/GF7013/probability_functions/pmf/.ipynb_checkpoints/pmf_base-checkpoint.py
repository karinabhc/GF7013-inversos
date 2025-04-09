# -*- coding: utf-8 -*-
"""
Prof. Francisco Hernan Ortega Culaciati
ortega.francisco@uchile.cl
f.ortega.culaciati@gmail.com
Departamento de Geofisica - FCFM
Universidad de Chile

Defines a base class for a probability mass function class.
It must provide 2 member functions. 
self.eval(x): gives the value of the probability mass function at each defined discrete 
              state.
self.draw(Ns = None): produces realizations of a probability mass. 
"""
import numpy as np

class pmf_base(object):
    """
    Defines a base class for a probability density function class.
    It must provide 3 basic functions. 
    self.eval(x): gives the value of the probability mass function at each defined  
                  discrete state.
    self.draw(M = None): produces a single value (M = None) or numpy array (M = integer) 
                         with realizations of the probability mass. 
    """
    ####
    def __init__(self, par, rng = None):
        """
        par: a dictionary containing the parameters that define the pdf 
        """
        self.par = par
        if rng is None:
            rng = np.random.default_rng()
        self.rng = rng

    ############# USER INTERFACE ########    
    ####
    def eval(self, x):
        """
        returns the value of the probability mass function at the defined discrete 
        state x.
        """
        return self._eval(x)

    ####
    def draw(self, Ns = None):
        """
        produces realizations of the probability distribution.
        If Ns is None, makes a realization of a discrete state according to its 
        probability. When Ns is a positive integer, it produces a 1D array with Ns 
        randomly selected discrete states (according to their probability).  
        """
        return self._draw(Ns = Ns)


    ############### INTERNAL FUNCTIONS THAT MUST BE IMPLEMENTED AT CHILD CLASS.
    def _eval(self, x):
        """
        implementation of self.eval(x)
        """
        raise NotImplementedError('User has not implemented _eval(x)')

    ####
    def _draw(self, Ns = None):
        """
        Implementation of self.draw(Ns)
        """
        raise NotImplementedError('User has not implemented _draw(x)')

