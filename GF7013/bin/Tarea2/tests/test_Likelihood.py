import sys
import os
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(root_dir)

from GF7013.probability_functions.pdf.pdf_normal import pdf_normal
from GF7013.probability_functions.pdf.pdf_uniform_nD import pdf_uniform_nD
from GF7013.probability_functions.likelihood.likelihood_function import likelihood_function
from GF7013.model_parameters.ensemble import ensemble
from GF7013.models.ajuste_ortogonal_recta.forward import forward
from GF7013.bin.Tarea2.P1.datos import obtener_datos_elipses


# (I) SIMULAR DATOS
np.random.seed(0)

# N = 25
# semi_eje_mayor = 8
# semi_eje_menor = 2
# alpha = 45
# delta_x = 0
# delta_y = 4
# desviacion_estandar_x = 1.0
# desviacion_estandar_y = 1.0

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



modelo_directo = forward(x_obs, y_obs, sigma_x, sigma_y)

# MODELOS
normas = np.sqrt(x_obs**2 + y_obs**2)
max_norma = np.max(normas)
a_min, a_max = -2 * max_norma, 2 * max_norma
theta_min, theta_max = -180, 180
Na, Ntheta = 100, 100

a_vals = np.linspace(a_min, a_max, Na)
theta_vals = np.linspace(theta_min, theta_max, Ntheta)
m = np.array([[a, theta] for a in a_vals for theta in theta_vals])
Nm = m.shape[0]

##################################################################################
#ENSEMBLE
# (II y III) Ensemble con verosimilitudes NORMALES JEJE
ensemble_normal = ensemble(Npar=2, Nmodels=Nm, use_log_likelihood=False)
ensemble_log = ensemble(Npar=2, Nmodels=Nm, use_log_likelihood=True)

# Llenar el conjunto de modelos
ensemble_normal.m_set[:, :] = m
ensemble_log.m_set[:, :] = m

# Instancia de la fdp de los datos (distancias)
sigma_deltas = np.sqrt(sigma_x**2 + sigma_y**2)
cov = np.diag(sigma_deltas**2)  # matriz de covarianza diagonal
params = {'mu': np.zeros(N), 'cov': cov}
pdf_datos = pdf_normal(par=params)

##################################################################################
#VEROSIMILITUD
# Instancia de la función de verosimilitud
L = likelihood_function(modelo_directo, pdf_datos)

# Evaluar verosimilitudes y log-verosimilitudes
ensemble_normal.like[:] = np.array([L.likelihood(mi) for mi in m])
ensemble_log.like[:] = np.array([L.log_likelihood(mi) for mi in m])

##################################################################################
#FPRIOR
lower_lim = np.array([a_min, theta_min])
upper_lim = np.array([a_max, theta_max])

par_fprior = {
    'lower_lim': lower_lim,
    'upper_lim': upper_lim
}

pdf_fprior = pdf_uniform_nD(par_fprior)

ensemble_normal.fprior[:] = np.array([pdf_fprior._likelihood(m) for m in ensemble_normal.m_set])
ensemble_log.fprior[:] = np.array([pdf_fprior._log_likelihood(m) for m in ensemble_normal.m_set])
##################################################################################
#FPOST
ensemble_normal.f[:] = ensemble_normal.fprior[:]*ensemble_normal.like[:]
ensemble_log.f[:] = ensemble_log.fprior[:] + ensemble_log.like[:]   

##################################################################################
# (IV) RESULTADOS
T, A = np.meshgrid(theta_vals, a_vals)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 filas, 3 columnas

def plot_grid(ax, Z, title, cmap="viridis"):
    im = ax.pcolor(T, A, Z.reshape(Na, Ntheta), shading='auto', cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(r'$\theta$ [°]')
    ax.set_ylabel('a')
    plt.colorbar(im, ax=ax)

# Fila 1: valores normales
plot_grid(axes[0, 0], ensemble_normal.fprior, 'fprior (normal)', cmap='Blues')
plot_grid(axes[0, 1], ensemble_normal.like, 'like (normal)', cmap='Reds')
plot_grid(axes[0, 2], ensemble_normal.f, 'f = fprior * like (normal)', cmap='Greens')

# Fila 2: valores en log
plot_grid(axes[1, 0], ensemble_log.fprior, 'log(fprior)', cmap='Blues')
plot_grid(axes[1, 1], ensemble_log.like, 'log(like)', cmap='Reds')
plot_grid(axes[1, 2], ensemble_log.f, 'log(f) = log(fprior + like)', cmap='Greens')

plt.tight_layout()
plt.show()

