#! /usr/bin/env python
import sys, os
# get folder of this module
this_module_folder = os.path.dirname(os.path.abspath(__file__))
# add GF7013 package to PYTHONPATH
GF7013_path = os.path.abspath(os.path.join(this_module_folder, '../../../..'))
sys.path.append(GF7013_path)

from GF7013.model_parameters import ensemble
from GF7013.sampling.metropolis import metropolis, proposal
from GF7013.sampling import probability_distributions as pdfs

import numpy as NP
import matplotlib.pyplot as plt

# define the pdf to sample (must have the likelihood/log_likelood function defined)
# THIS WILL BE USED AS THE LIKELHOOD FUNCTION!!!
class pdf_bimodal(object):
    def __init__(self,  x_0, sigma_0, p_0, x_1, sigma_1, p_1):
        self.args_f0 = (x_0, sigma_0, p_0)
        self.args_f1 = (x_1, sigma_1, p_1)

    def __f(self, x, mean, sigma, p):
        alpha = 1 / sigma / NP.sqrt(2*NP.pi)
        return alpha * NP.exp(-0.5*(NP.sum(NP.abs((x - mean)/sigma)**p)))

    def likelihood(self, x):
        return self.__f(x, *self.args_f0) + self.__f(x, *self.args_f1)
    
    def log_likelihood(self, x):
        return NP.log(self.likelihood(x))


### MAIN CODE OF THE EXAMPLE.
if __name__ == '__main__':
    NumBins = 100
    # define the fdp to sample using SIR resampling
    x_0 = -2.5
    sigma_0 = 2.0
    p_0 = 2

    x_1 = 14 #9 #7.5
    sigma_1 = 0.5#0.1 #0.75
    p_1 = 1

    # crear instancia de la pdf que usará como funcion de verosimilitud
    f = pdf_bimodal(x_0, sigma_0, p_0, x_1, sigma_1, p_1)

    # evaluate the pdf for later plot
    x_min = -15
    x_max = 22
    Num_x = 10000
    x_eval = NP.linspace(x_min, x_max, Num_x)
    f_values = NP.array([f.likelihood(aux) for aux in x_eval])
    dx = x_eval[1] - x_eval[0]
    f_area = NP.sum(f_values)*dx # rectangle integration
    
    # define prior distribution
    x_min_ini = -12
    x_max_ini = 22
    prior_pdf_pars = {}
    prior_pdf_pars['lower_lim'] = NP.array([x_min_ini])
    prior_pdf_pars['upper_lim'] = NP.array([x_max_ini])
    fprior = pdfs.pdf_uniform_nD(par=prior_pdf_pars)
    
    # define proposal distribution
    sigma_prop = (x_max_ini - x_min_ini)/100
    proposal_params = {}
    proposal_params['cov'] = NP.array([[sigma_prop]]) # 2D array!!
    proposal_pdf = proposal.proposal_normal(par=proposal_params)

    # define initial model:
    m0 = NP.array([-10]) # 1D array!!!

    # define initial models as samples from U(x_min, x_max)
    NumSamples = int(1E5)
    NumBurnIn = int(0.1 * NumSamples)
    use_log_likelihood = False
    #############
    # YOU CAN TRY DIFFERENT VALUES OF BETA and see what happens!
    beta = 1
    #############
    results = metropolis(m0= m0, 
                         likelihood_fun = f, 
                         pdf_prior = fprior, 
                         proposal = proposal_pdf, 
                         num_samples = NumSamples,
                         num_burnin = NumBurnIn,
                         use_log_likelihood = use_log_likelihood,
                         save_samples = True,
                         beta = beta)

    print(results['samples'].m_set.flatten())
    print(results['samples'].f)

    f_values_beta = NP.array([f.likelihood(aux)**beta for aux in x_eval])
    dx = x_eval[1] - x_eval[0]
    f_area_beta = NP.sum(f_values_beta)*dx # rectangle integration
    fig = plt.figure(1)
    fig.set_size_inches((8,10))
    ax1 = fig.add_subplot(211)
    ax1.plot(x_eval, f_values/f_area, label = 'Bimodal PDF', color = 'cyan')
    ax1.plot(x_eval, f_values_beta/f_area_beta, '--k', label = 'Bimodal PDF')
    ax1.hist(results['samples'].m_set.flatten(), density = True, bins = 300, 
            color = 'red')
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax2.plot(results['samples'].m_set.flatten(), range(NumSamples), '.-r')

    plt.show()

    
