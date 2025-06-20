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
print(fprior.draw())

############## Creo que de aqui pa bajo ojito
modelo_forward = forward(x_obs=x_obs, y_obs=y_obs, sigma_x=sigma_x, sigma_y=sigma_y)

# Crear la distribución de probabilidad para los datos (residuos normalizados)
# Como forward.eval() devuelve residuos normalizados, usamos una normal estándar
# El número de parámetros debe coincidir con el número de observaciones
n_obs = len(x_obs)


# Parámetros teóricos
#mu = np.array([-1.0, 4.0])
#cov = np.array([[2.0, 1.0], [1.0, 4.0]])
par = {'mu': np.zeros(n_obs), 'cov': np.eye(n_obs)}
pdf = pdf_normal(par)

# Crear la función de verosimilitud usando tu clase del paquete
likelihood_fun = likelihood_function(forward=modelo_forward, pdf_data=pdf)

# Definir la distribución de propuesta (normal multivariada)
# Matriz de covarianza para la propuesta
#sigma_a = 0.5  # desviación estándar para el parámetro 'a'
#sigma_theta = 0.5  # desviación estándar para el parámetro 'theta'
#cov_matrix = np.array([[sigma_a**2, 0],0000 
#                       [0, sigma_theta**2]])

alpha=1/100
delta = (fprior.par['upper_lim'] - fprior.par['lower_lim']) 
cov_matrix = np.diag((alpha*delta))  # 2D array!!
#deltas =fprior.par['upper_lim'] - fprior.par['lower_lim']
#cov_matrix = np.diag(alpha * deltas**2)  # Matriz de cov

proposal = proposal_normal(cov=cov_matrix)


# Parámetros del Metropolis
NumSamples = int(1e5)
NumBurnIn = int(0.1 * NumSamples)
use_log_likelihood = True

#m0 = np.array([0.0, 0.0])  # Modelo inicial, valores iniciales para [a, theta]
m0 = fprior.draw()

cadena = metropolis(m0=m0, 
                    likelihood_fun=likelihood_fun, 
                    pdf_prior=fprior, 
                    proposal=proposal, 
                    num_samples=NumSamples,
                    num_burnin=NumBurnIn,
                    use_log_likelihood=use_log_likelihood,
                    save_samples=True,
                    beta=1)

#fpost =_fpost(fm_prior= fprior, fm_like=likelihood_fun, beta=1)
#log_fpost =_log_fpost(fm_prior= fprior, fm_like=likelihood_fun, beta=1)

# Extraer muestras
samples = cadena['samples']
a_samples = samples.m_set[:, 0]
theta_samples = samples.m_set[:, 1]

# --- Graficar evolución de parámetros ---
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].scatter(np.arange(len(a_samples)), a_samples, alpha=0.7)
axs[0].set_ylabel('a')
axs[0].grid(True)
axs[0].set_title('Evolución del parámetro a')

axs[1].scatter(np.arange(len(a_samples)),theta_samples, alpha=0.7, color='orange')
axs[1].set_ylabel('theta [grados]')
axs[1].set_xlabel('Iteración')
axs[1].grid(True)

axs[1].set_title('Evolución del parámetro theta')

plt.suptitle("Evolución de la cadena de Metropolis")
plt.tight_layout()
plt.show()
    



