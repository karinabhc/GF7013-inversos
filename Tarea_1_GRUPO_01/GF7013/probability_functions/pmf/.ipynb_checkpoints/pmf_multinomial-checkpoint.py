# -*- coding: utf-8 -*-
"""
Prof. Francisco Hernan Ortega Culaciati
f.ortega.culaciati@gmail.com
Departamento de Geofisica - FCFM
Universidad de Chile


Defines a  class for a 1D multinomial probability mass function (pmf) class for
the multinomial distribution.
It is intended for assessing discrete states of a discrete random variable.
It must provide 3 basic functions. 
self.eval(x): gives the value of the probability mass function at each defined discrete 
              state.
self.draw(Ns = None): produces realizations of a probability mass. 
"""
from .pmf_base import pmf_base
import numpy as NP

class pmf_multinomial(pmf_base):
    """
    Defines a class for 1D multinomial probability mass function (pmf) class.
    """
    ####
    def __init__(self, par, rng = None):
        """
        par: a dictionary containing the parameters that define the pmf.
        allowed keys are:
           'values': a 1D numpy array with the values of the discrete cases that are considered.
           'importance': a 1D numpy array with the importance (i.e., unnormalized mass 
                         probabilities) for each value (same size as 'values').
           'method': the desired method to generate multinomial realizations. If 'method'
                     key is not present in par dictionary, it uses the default
                     method 'numpy'. In this homework you should code the 'analog' 
                     method (VER APUNTES DE CLASES). 
        rng: an instance of a random number generator. Default is NP.random.default_rng()
            if seed is None.
        """
            
        # The base class constructor assigns self.par = par
        super().__init__(par, rng)

        # determine the number of dimensions of value and importance.
        self.__internal_checks()

        # make relevant parameters easier to access
        self.values = self.par['values']
        self.importance = self.par['importance']
        if 'method' not in par.keys():
            self.method = 'numpy'
        else:
            self.method = par['method']

        self.N = len(self.values)
        # compute mass probabilities of all states
        # compute normalization constant and base e logarithm of normalization.
        self.normalization = NP.sum(self.par['importance'])

        if self.normalization < 1E3 * NP.finfo(float).eps:
           raise ValueError("all 'importance' elements are null."
                            + " At least one needs to be > 0.")
        self.prob = self.importance / self.normalization
        # define probabilities dictionary to allow evaluation of the probability mass per
        # a given state.
        self.prob_dict = dict(zip(self.values, self.prob))


    ####
    def __internal_checks(self, tol_zero = 1000):
        """
        - check that 'values' and 'imporance' exists and have the same shape
        - check that importances are non negative
        - check that at least one importance is different from zero
        - compute the probabilities by normalizing the total probability to 1.
        """
        if 'values' not in self.par.keys():
            raise ValueError("'values' must be a key of par dictionary.")

        if 'importance' not in self.par.keys():

            raise ValueError("'importance' must be a key of par dictionary.")

        # check importance and values have the same number of elements
        if self.par['values'].shape[0] != self.par['importance'].shape[0]:
            raise ValueError("'values' and 'importance' must be of equal length.")
        # check that at least have 2 states
        if len(self.par['importance']) <= 1:
            raise ValueError("at least 2 states need to be defined for the multinomial")
        # check all importance values are all non-negative
        test = self.par['importance'] < 0
        if test.any():
            raise ValueError("'importance' must contain only non-negative elements.")

        
    ####
    def _draw(self, Ns = None):
        """
        produces realizations of the probability mass.
        If Ns is None, makes a realization of an element of self.value according to its 
        probability. When Ns is a positive integer, it produces a 1D array with Ns 
        randomly selected values (according to their probability).  
        """
        if self.method == 'numpy':
            if Ns is None:
                realization = self.rng.multinomial(1, self.prob, size=None)
                i = NP.where(realization > 0)[0][0]
                sample = self.values[i]
            else:
                # use a single recursion to compute Ns samples.
                sample = NP.array([self._draw(Ns=None) for i in range(Ns)])
        elif self.method == 'analog':
            # VER NOTAS DE CLASES
            COMPLETAR = None

        else:
            raise ValueError("par['method'] should be 'numpy' or 'analog'")


        return sample

    ####
    def _eval(self, x):
        """
        returns the value of the probability mass function at the defined discrete 
        state x.
        """
        if x in self.values:
            return self.prob_dict[x]
        else:
            raise ValueError("the state x does not have a probability associated.")

