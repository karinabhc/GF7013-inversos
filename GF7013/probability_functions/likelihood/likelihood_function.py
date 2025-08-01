# -*- coding: utf-8 -*-
"""
Prof. Francisco Hernan Ortega Culaciati
f.ortega.culaciati@gmail.com
Departamento de Geofisica - FCFM
Universidad de Chile

June 2, 2022

Defines a class to build likelihood functions for Bayesian inverse problems.
It must provide 2 member functions. 
self.likelihood(m): gives the value of the possibly unnormalized probability function.
self.log_likelihood(m)
"""
import numpy as NP

class likelihood_function(object):
    """
    Defines a class to build likelihood functions for Bayesian inverse problems.
    It provides 2 member functions. 
    self.likelihood(m): gives the value of the likelihood function.
    self.log_likelihood(m): gives the value of the natural logarithm of the likelihood 
                            function.
                        
    """
    def __init__(self, forward, pdf_data):
        """
        - forward: an object that defines the forward model. It must provide the member
                   function forward.eval(m) that receives a vector of model parameters, 
                   and returns its predictions. m must be a 1D array and model prediction,
                   must be also a 1D array. 
        - pdf_data: an object instantiated from a pdf class, that represents prior
                    information on the observed quantities. Must provide at least 2 
                    member functions: 
                        - pdf_data.likelihood(d) 
                        - pdf_data.log_likelihood(d)
                    that return the likelihood or log-likelihood value of the prior
                    distribution of the data. This is used to construc the likelihood
                    function as a function of the predictions of model parameters m. 
        """
        self.forward = forward
        self.pdf_data = pdf_data
        
    def likelihood(self, m):
        """
        computes the value of the likelihood function for model m.
        """
        dpred = self.forward.eval(m)
        return self.pdf_data.likelihood(dpred)

    def log_likelihood(self, m):
        """
        computes the value of the natural logarithn of the likelihood function for 
        model m.
        """
        dpred = self.forward.eval(m)
        return self.pdf_data.log_likelihood(dpred)

