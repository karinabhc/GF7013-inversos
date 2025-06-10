# -*- coding: utf-8 -*-
"""
Prof. Francisco Hernan Ortega Culaciati
ortega.francisco@uchile.cl
f.ortega.culaciati@gmail.com
Departamento de Geofisica - FCFM
Universidad de Chile

June 2, 2022

Defines a class for a normal (gaussian) proposal distribution class for the Metropolis 
Algorithm.
It must provide 1 member functions. 
self.propose(m): returns m_test to be proposed as the next transition in Metropolis Alg.
                 
"""
from ...probability_functions.pdf import pdf_normal
import numpy as np

COMPLETAR = None

class proposal_normal(object):
    """
    Defines a class for a proposal distribution that uses an unbiased n-dimensional 
    multivariate normal probability density function to propose perturbations on 
    current models.
    """
    ####
    def __init__(self, cov):
        """      
        - 'cov': a 2D array with the covariance matrix of the normal pdf.
                  must be a symmetric, nonsingular and positive definite matrix.
            
        """

        # instantiate the normal pdf that will be used to draw samples
        pdf_normal_parameters = {}
        pdf_normal_parameters['cov'] = cov
        Npar = cov.shape[0]
        pdf_normal_parameters['mu'] = COMPLETAR # unbiassed (i.e., null mean)

        self.pdf = pdf_normal(par= pdf_normal_parameters)

    def reset_random_number_generator(self, rng = None, seed = None):
        """
        resets the random number generator of the pdf_normal. This is intented to be used
        when doing parallel computation, where each process must have a different
        seed of the random number generator.
        """
        self.pdf.reset_random_number_generator(rng = rng, seed = seed)

    ####
    def propose(self, m):
        """
        returns m_test to be proposed as the next transition in Metropolis Algorithm,
        as m +  dm, where dm is a realization of a multivariate normal 
        distribution. 

        """
        # propose the perturbation
        dm = COMPLETAR
        # check wether the perturbation has the same shape of m, if not, try to reshape
        if not(m.shape == dm.shape):
            try:
                dm = dm.reshape(m.shape)
            except ValueError:
                msg = ' m, and dm do not have the same number of elements !!! \n'
                msg += '--> Check the dimensions of the Covariance Matrix of the \n'
                msg += '    proposal distribution.'
                      
                raise ValueError(msg)

        # compute and return the proposed model
        return m + dm

        
    