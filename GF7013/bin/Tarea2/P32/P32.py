import matplotlib.pyplot as plt
import numpy as np

from GF7013.bin.Tarea2.P1.datos import obtener_datos_elipses

# importar modulos relevantes del paquete GF7013
pythonpackagesfolder = '../../../..' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

# modelo directo de la recta
#from GF7013.models.ajuste_ortogonal_recta import recta
from GF7013.models.ajuste_ortogonal_recta.forward import forward
from GF7013.probability_functions.pdf.pdf_uniform_nD import pdf_uniform_nD  # Para la distribución a priori
from GF7013.probability_functions.likelihood.likelihood_function import likelihood_function
from GF7013.probability_functions.pdf.pdf_normal import pdf_normal  # Para la distribución de los residuos
from GF7013.sampling.metropolis.proposal_normal import proposal_normal
from GF7013.sampling.metropolis.metropolis import metropolis, _fpost,_log_fpost
import matplotlib.gridspec as gridspec


N = 25
semi_eje_mayor = 8
semi_eje_menor = 2
alpha = 45
delta_x = 0
delta_y = 4
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


normas = np.sqrt(x_obs**2 + y_obs**2)
max_norma = np.max(normas)

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
mu = np.array([-1.0, 4.0])
cov = np.array([[2.0, 1.0], [1.0, 4.0]])
par = {'mu': mu, 'cov': cov}
pdf = pdf_normal(par)

# Crear la función de verosimilitud usando tu clase del paquete
likelihood_fun = likelihood_function(forward=modelo_forward, pdf_data=pdf)

# Definir la distribución de propuesta (normal multivariada)
# Matriz de covarianza para la propuesta
sigma_a = 1.0  # desviación estándar para el parámetro 'a'
sigma_theta = 1.0  # desviación estándar para el parámetro 'theta'
cov_matrix = np.array([[sigma_a**2, 0], 
                       [0, sigma_theta**2]])

proposal = proposal_normal(cov=cov_matrix)


# Parámetros del Metropolis
NumSamples = int(5e4)
NumBurnIn = int(0.1 * NumSamples)
use_log_likelihood = True

m0 = np.array([0.0, 0.0])  # Modelo inicial, valores iniciales para [a, theta]

cadena = metropolis(m0=m0, 
                    likelihood_fun=likelihood_fun, 
                    pdf_prior=fprior, 
                    proposal=proposal, 
                    num_samples=NumSamples,
                    num_burnin=NumBurnIn,
                    use_log_likelihood=use_log_likelihood,
                    save_samples=True,
                    beta=1)

fpost =_fpost(fm_prior= fprior, fm_like=likelihood_fun, beta=1)
log_fpost =_log_fpost(fm_prior= fprior, fm_like=likelihood_fun, beta=1)

# Extraer muestras
samples = cadena['samples']
a_samples = samples.m_set[:, 0]
theta_samples = samples.m_set[:, 1]

# --- Graficar evolución de parámetros ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(a_samples, alpha=0.7)
axs[0].set_ylabel('a')
axs[0].grid(True)
axs[0].set_title('Evolución del parámetro a')

axs[1].plot(theta_samples, alpha=0.7, color='orange')
axs[1].set_ylabel('theta [grados]')
axs[1].set_xlabel('Iteración')
axs[1].grid(True)
axs[1].set_title('Evolución del parámetro theta')

plt.suptitle("Evolución de la cadena de Metropolis")
plt.tight_layout()
plt.show()
    

# --- Graficar evolución de parámetros ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(a_samples, label='a')
axs[0].set_ylabel('a')
axs[0].grid(True)
axs[1].plot(theta_samples, label='theta', color='orange')
axs[1].set_ylabel('theta')
axs[1].set_xlabel('Iteración')
axs[1].grid(True)
plt.suptitle("Evolución de la cadena de Metrópolis")
plt.tight_layout()
plt.show()

# --- Histogramas conjunto y marginales ---
Nsamples = int(1e5)
samples = fprior.draw(Nsamples)  
a_samples = samples[0, :]
theta_samples = samples[1, :]

fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       wspace=0.05, hspace=0.05)

# Histograma conjunto
ax_join = fig.add_subplot(gs[1, 0])
counts, xedges, yedges, im = ax_join.hist2d(a_samples, theta_samples, bins=100, cmap='viridis')
fig.colorbar(im, ax=ax_join, label='Número de muestras')
ax_join.set_title(r'Histograma Marginal 2D de (a, $\theta$) (FDP Conjunta)')
ax_join.set_xlabel('a')
ax_join.set_ylabel(r'$\theta$')

# Marginal para a
ax_x1 = fig.add_subplot(gs[0, 0], sharex=ax_join)
x1_hist, _ , _ = ax_x1.hist(a_samples, bins=100, density=True, alpha=0.6, color='blue', label='Histograma de a', edgecolor='k')
ax_x1.set_title('Marginal para a')
ax_x1.set_ylabel('frecuencia')
ax_x1.tick_params(labelbottom=False)

pos = ax_x1.get_position()
ax_x1.set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height])
ax_x1.set_xlim(xedges[0], xedges[-1])
ax_x1.set_xlim(ax_join.get_xlim())

# Marginal para theta
ax_x2 = fig.add_subplot(gs[1, 1], sharey=ax_join)
x2_hist, x2_bins, _ = ax_x2.hist(theta_samples, bins=100, density=True, alpha=0.6, color='green', orientation='horizontal', label=r'Histograma de $\theta$', edgecolor='k')
ax_x2.set_title(r'Marginal para $\theta$')
ax_x2.set_xlabel('frecuencia')
ax_x2.tick_params(labelleft=False)

fig.suptitle(r"Distribución uniforme de N dimensiones: Histograma conjunto y marginales con Ns $={1.0e+05}$", fontsize=14, y=0.965)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
