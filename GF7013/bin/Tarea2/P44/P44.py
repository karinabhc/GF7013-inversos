"""
Script similar a P32, pero con modelos iniciales generados como muestras de fprior
"""
import matplotlib.pyplot as plt
import numpy as np



# importar modulos relevantes del paquete GF7013
pythonpackagesfolder = '../../../../' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

# modelo directo de la recta
#from GF7013.models.ajuste_ortogonal_recta import recta
from GF7013.bin.Tarea2.P1.datos import obtener_datos_elipses
from GF7013.models.ajuste_ortogonal_recta.forward import forward
from GF7013.probability_functions.pdf.pdf_uniform_nD import pdf_uniform_nD  # Para la distribución a priori
from GF7013.probability_functions.likelihood.likelihood_function import likelihood_function
from GF7013.probability_functions.pdf.pdf_normal import pdf_normal  # Para la distribución de los residuos
from GF7013.sampling.metropolis.proposal_normal import proposal_normal
from GF7013.sampling.metropolis_in_parallel import metropolis_in_parallel_POOL, metropolis_in_parallel_SERIAL
import matplotlib.gridspec as gridspec

from GF7013.model_parameters import ensemble


N = 50
semi_eje_mayor = 20
semi_eje_menor = 2
alpha = 45
delta_x = 0
delta_y = 0
desviacion_estandar_x = 1.0
desviacion_estandar_y = 1.0

x_obs, y_obs, sigma_x, sigma_y = obtener_datos_elipses(
                                        N = N,
                                        a = semi_eje_mayor,
                                        b = semi_eje_menor,
                                        alpha = alpha,
                                        deltax = delta_x,
                                        deltay = delta_y,
                                        sigma_x = desviacion_estandar_x,
                                        sigma_y = desviacion_estandar_y)



a_min = -15
a_max = 15
theta_min = -360
theta_max = 360 

lower_lim = np.array([a_min, theta_min])
upper_lim = np.array([a_max, theta_max])

par = {
    'lower_lim': lower_lim,
    'upper_lim': upper_lim
}

fprior = pdf_uniform_nD(par)

############## Creo que de aqui pa bajo ojito
modelo_forward = forward(x_obs=x_obs, y_obs=y_obs, sigma_x=sigma_x, sigma_y=sigma_y)

# Crear la distribución de probabilidad para los datos (residuos normalizados)
# Como forward.eval() devuelve residuos normalizados, usamos una normal estándar
# El número de parámetros debe coincidir con el número de observaciones
n_obs = len(x_obs)


# Parámetros teóricos
par = {'mu': np.zeros(n_obs), 'cov': np.eye(n_obs)}
pdf = pdf_normal(par)

# Crear la función de verosimilitud usando tu clase del paquete
likelihood_fun = likelihood_function(forward=modelo_forward, pdf_data=pdf)

# Definir la distribución de propuesta (normal multivariada)
# Matriz de covarianza para la propuesta
sigma_a = 0.5  # desviación estándar para el parámetro 'a'
sigma_theta = 0.5  # desviación estándar para el parámetro 'theta'
cov_matrix = np.array([[sigma_a**2, 0], 
                       [0, sigma_theta**2]])




proposal = proposal_normal(cov=cov_matrix)


# Parámetros del Metropolis
NumSamples = int(5e4)
NumBurnIn = int(0.3 * NumSamples)
#NumBurnIn = 0
numStepChains=300
use_log_likelihood = True

#m0 = np.array([0.0, 0.0])  # Modelo inicial, valores iniciales para [a, theta]
m0 = ensemble(
              Npar = 2, Nmodels=NumSamples, 
              use_log_likelihood=use_log_likelihood,
              beta=1
            ) # fprior.draw()

# generamos los modelos iniciales como muestras de fprior
m0.m_set[:] = fprior.draw()


m,acceptance_ratios = metropolis_in_parallel_POOL(m0,likelihood_fun=likelihood_fun,
                                            pdf_prior=fprior,
                                            proposal=proposal,
                                            num_MCMC_steps=numStepChains,
                                            use_log_likelihood=use_log_likelihood
                                            )

# Extraer muestras
samples = m.m_set
print(samples)
a_samples = samples[:, 0]
theta_samples = samples[:, 1]

# --- Graficar evolución de parámetros ---
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

sc=ax.scatter(a_samples,theta_samples,alpha=0.7,c=np.arange(len(a_samples)),cmap='rainbow')
plt.colorbar(sc, ax=ax, label='Índice de Muestra')
ax.set_xlabel('a')
ax.set_ylabel('theta [grados]')
ax.grid(True)

plt.suptitle("Evolución de la cadena de Metropolis (30% Burn-in)", fontsize=16)
plt.tight_layout()
plt.show()

# --- Graficar histogramas de parámetros ---
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       wspace=0.05, hspace=0.05)
ax_join = fig.add_subplot(gs[1, 0])
counts, xedges, yedges, im = ax_join.hist2d(a_samples, theta_samples, bins=50, cmap='viridis')
fig.colorbar(im, ax=ax_join, label='Cuentas')
ax_join.set_title('Histograma 2D (PDF conjunta)')
ax_join.set_xlabel('a')
ax_join.set_ylabel('theta [grados]')

# Marginal para a
ax_a = fig.add_subplot(gs[0, 0], sharex=ax_join)
a_hist, a_bins, _ = ax_a.hist(a_samples, bins=100, density=True, alpha=0.6, color='blue', label='Histograma de a')
ax_a.set_title('Marginal para a')
ax_a.set_ylabel('Densidad')
ax_a.legend()
ax_a.tick_params(labelbottom=False)
pos = ax_a.get_position()
ax_a.set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height])

# Marginal para theta
ax_theta = fig.add_subplot(gs[1, 1], sharey=ax_join)
theta_hist, theta_bins, _ = ax_theta.hist(theta_samples, bins=100, density=True, alpha=0.6, color='green', orientation='horizontal', label='Histograma de theta')
ax_theta.set_title('Marginal para theta')
ax_theta.set_xlabel('Densidad')
ax_theta.legend()
ax_theta.tick_params(labelleft=False)   
fig.suptitle("Distribución de Parámetros: Histograma Conjunto y Marginales", fontsize=14, y=0.965)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



