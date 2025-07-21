#! /usr/bin/env python
import sys, os
import numpy as NP
import matplotlib.pyplot as plt

# Set environment variables (important for multiprocessing with NumPy)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DOMAIN_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Setup import path
this_module_folder = os.path.dirname(os.path.abspath(__file__))
GF7013_path = os.path.abspath(os.path.join(this_module_folder, '../../../..'))
sys.path.append(GF7013_path)

# Import from GF7013 package
from GF7013.model_parameters import ensemble
from GF7013.sampling.tmcmc.tmcmc import tmcmc_pool
from GF7013.sampling.metropolis.proposal_normal import proposal_normal
from GF7013.probability_functions import pdf as pdfs
from GF7013.bin.Tarea2.tests.test_metropolis import pdf_bimodal

class pdf_bimodal_ensemble(pdf_bimodal):   
    def __f(self, x, mean, sigma, p):
        alpha = 1 / sigma / NP.sqrt(2*NP.pi)
        result = alpha * NP.exp(-0.5*(NP.sum(NP.abs((x.m_set - mean)/sigma)**p, axis=1)))
        return result
    
    def likelihood(self, m):
        if isinstance(m, ensemble):
            return self.__f(m, *self.args_f0) + self.__f(m, *self.args_f1)
        else:
            return super().likelihood(m)

    def log_likelihood(self, m):
        return NP.log(self.likelihood(m))


def run_tmcmc(use_log_likelihood=False):
    #instancia de la pdf que usará como funcion bimodal de verosimilitud
    f = pdf_bimodal(x_0=-2.5, sigma_0=2.0, p_0=2.0,
                    x_1=14.0, sigma_1=0.5, p_1=1.0)
    f2 = pdf_bimodal_ensemble(x_0=-2.5, sigma_0=2.0, p_0=2.0,
                    x_1=14.0, sigma_1=0.5, p_1=1.0)
    print(f)
    # Define evaluation grid
    x_eval = NP.linspace(-15, 22, 10_000)
    if use_log_likelihood:
        f_values = NP.exp(NP.array([f.log_likelihood(x) for x in x_eval]))
    else:
        f_values = NP.array([f.likelihood(x) for x in x_eval])
    dx = x_eval[1] - x_eval[0]
    f_area = NP.sum(f_values) * dx

    # uniform prior
    x_min_ini = -12
    x_max_ini = 22
    prior_pdf_pars = {}
    prior_pdf_pars['lower_lim'] = NP.array([x_min_ini])
    prior_pdf_pars['upper_lim'] = NP.array([x_max_ini])
    fprior = pdfs.pdf_uniform_nD(par=prior_pdf_pars)

    # proposal distribution
    sigma_prop = (x_max_ini - (x_min_ini)) / 100
    cov = NP.array([[sigma_prop]])  # 2D array!!
    proposal_pdf = proposal_normal(cov=cov)

    # Ensemble
    Npar = 1
    Nmodels = 10_000
    beta0 = 0.0 ########### initial beta value
    m0 = ensemble(Npar=Npar, Nmodels=Nmodels,
                  use_log_likelihood=use_log_likelihood,
                  beta=beta0)
    

    # TMCMC
    m, acc_ratios = tmcmc_pool(m0, likelihood_fun=f2,
                               pdf_prior=fprior,
                               proposal=proposal_pdf,
                               num_MCMC_steps=100,
                               num_proc=4,
                               chunksize=1,
                               use_resampling=False)

    #print(f"TMCMC terminado con use_log_likelihood={use_log_likelihood}")  # para verificar que se ejecuta
    #print(f"Beta final: {m.beta:.4f}") #para verificar que se actualiza beta
    #print(f"Razones de aceptación: {acc_ratios}") # para verificar que se calculan las razones de aceptación

    # Plot results
    f_values_beta = f_values ** m.beta if not use_log_likelihood else NP.exp(m.beta * (NP.log(f_values + 1e-300)))
    f_area_beta = NP.sum(f_values_beta) * dx

    fig, axs = plt.subplots(2, 1, figsize=(8, 10), layout='constrained')
    axs[0].plot(x_eval, f_values / f_area, label='PDF bimodal original', color='cyan')
    axs[0].plot(x_eval, f_values_beta / f_area_beta, '--k', label=f'PDF posterior beta={m.beta:.2f}')
    axs[0].hist(m.m_set.flatten(), density=True, bins=300, color='tab:red', alpha=0.5)
    axs[0].set_title(f'Distribución muestreada - TMCMC (log = {use_log_likelihood})')
    axs[0].set_xlabel("Modelo (m)")
    axs[0].set_ylabel("Densidad de probabilidad")
    axs[0].legend()

    axs[1].scatter(m.m_set.flatten(), range(Nmodels),
                   c=NP.arange(Nmodels), cmap='rainbow', s=1)
    axs[1].set_xlabel("Modelo (m)")
    axs[1].set_ylabel("Número de muestra")
    axs[1].set_title("Muestras de modelo vs. número de muestra")

    plt.show()


if __name__ == '__main__':
    run_tmcmc(use_log_likelihood=False)
    run_tmcmc(use_log_likelihood=True)
