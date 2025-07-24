#! /usr/bin/env python
import sys, os
# get folder of this module
this_module_folder = os.path.dirname(os.path.abspath(__file__))
# add GF7013 package to PYTHONPATH
GF7013_path = os.path.abspath(os.path.join(this_module_folder, '../../../..'))
sys.path.append(GF7013_path)

from GF7013.model_parameters import ensemble
from GF7013.sampling.metropolis_in_parallel import metropolis_in_parallel_SERIAL
from GF7013.sampling.metropolis.proposal_normal import proposal_normal

from GF7013.probability_functions import pdf as pdfs
from GF7013.probability_functions.likelihood import likelihood_function

import numpy as NP
import matplotlib.pyplot as plt
# define the pdf to sample (must have the likelihood/log_likelood function defined)
# THIS WILL BE USED AS THE LIKELHOOD FUNCTION!!!
from GF7013.bin.Tarea2.tests.test_metropolis import pdf_bimodal


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
    Num_x = 10_000
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
    cov = NP.array([[sigma_prop]]) # 2D array!!
    proposal_pdf = proposal_normal(cov=cov)


    # ======================================================
    # Preparing and performing MIP Serial
    # ======================================================


    # define initial model:
    Npar = 1
    NumSamples = int(1E5)
    beta = 1
    numStepChains = 300
    m0 = ensemble(Npar = Npar, Nmodels=NumSamples,
                  use_log_likelihood=False,
                  beta=beta)    
    m0.m_set = fprior.draw(NumSamples).T
    m, acceptance_ratios = metropolis_in_parallel_SERIAL(m0,likelihood_fun=f,
                                            pdf_prior=fprior,
                                            proposal=proposal_pdf,
                                            num_MCMC_steps=numStepChains,
                                            use_log_likelihood=False
                                            )

    print(m.m_set.flatten())
    print(m.f)

    f_values_beta = NP.array([f.likelihood(aux)**beta for aux in x_eval])
    dx = x_eval[1] - x_eval[0]
    f_area_beta = NP.sum(f_values_beta)*dx # rectangle integration
    
    
    fig = plt.figure(1, layout='constrained')
    fig.set_size_inches((8,10))
    ax1 = fig.add_subplot(211)
    ax1.plot(x_eval, f_values/f_area, label = 'Bimodal PDF', color = 'cyan')
    ax1.plot(x_eval, f_values_beta/f_area_beta, '--k', label = 'Bimodal PDF')
    ax1.hist(m.m_set.flatten(), density = True, bins = 300, color = 'red')
    ax1.legend()
    
    ax1.set_xlabel("modelos muestreados (m)")
    ax1.set_ylabel("Densidad de Probabilidad")
    ax1.legend()
    ax1.set_title("Distribución Muestreada vs. PDF Teórica")

    ax2 = fig.add_subplot(212, sharex=ax1)
    sc = ax2.scatter(m.m_set.flatten(), range(NumSamples),
                     c=NP.arange(NumSamples),cmap='rainbow',s=1)
    plt.colorbar(sc, ax=ax2, label='Índice de Muestra')
    ax2.set_xlabel("Modelos Muestreados (m)")
    ax2.set_ylabel("Número de Muestras")
    ax2.set_title(" Muestras de Modelos vs. Número de Muestras")
    plt.show()
