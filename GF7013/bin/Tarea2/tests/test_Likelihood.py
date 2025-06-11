import sys
import os
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(root_dir)

from GF7013.probability_functions.pdf.pdf_normal import pdf_normal
from GF7013.probability_functions.likelihood.likelihood_function import likelihood_function
from GF7013.model_parameters.ensemble import ensemble
from GF7013.models.ajuste_ortogonal_recta.forward import forward

# (I) SIMULAR DATOS
np.random.seed(0)
N = 100
x_obs = np.random.uniform(-10, 10, size=N)
y_obs = np.random.uniform(-10, 10, size=N)
sigma_x = np.ones(N)
sigma_y = np.ones(N)

modelo_directo = forward(x_obs, y_obs, sigma_x, sigma_y)

# RANGO DE MODELOS
normas = np.sqrt(x_obs**2 + y_obs**2)
max_norma = np.max(normas)
a_min, a_max = -2 * max_norma, 2 * max_norma
theta_min, theta_max = -180, 180
Na, Ntheta = 100, 100

a_vals = np.linspace(a_min, a_max, Na)
theta_vals = np.linspace(theta_min, theta_max, Ntheta)
m = np.array([[a, theta] for a in a_vals for theta in theta_vals])
Nm = m.shape[0]

# (II y III) Ensemble con verosimilitudes normales
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

# Instancia de la función de verosimilitud
L = likelihood_function(modelo_directo, pdf_datos)

# Evaluar verosimilitudes y log-verosimilitudes
ensemble_normal.like[:] = np.array([L.likelihood(mi) for mi in m])
ensemble_normal.f[:] = ensemble_normal.like[:]  # Suponiendo f ≡ like
ensemble_log.like[:] = np.array([L.log_likelihood(mi) for mi in m])
ensemble_log.f[:] = ensemble_log.like[:]        # Suponiendo f ≡ log_like

# (IV) Graficar resultados
A, T = np.meshgrid(theta_vals, a_vals)


fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 filas, 3 columnas

def plot_grid(ax, Z, title, cmap="viridis"):
    im = ax.pcolor(T, A, Z.reshape(Na, Ntheta), shading='auto', cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel('θ [°]')
    ax.set_ylabel('a')
    plt.colorbar(im, ax=ax)

# Fila 1: valores normales
plot_grid(axes[0, 0], ensemble_normal.fprior, 'fprior (normal)', cmap='Blues')
plot_grid(axes[0, 1], ensemble_normal.like, 'like (normal)', cmap='Reds')
plot_grid(axes[0, 2], ensemble_normal.f, 'f = fprior * like (normal)', cmap='Greens')

# Fila 2: valores en log
plot_grid(axes[1, 0], ensemble_log.fprior, 'log(fprior)', cmap='Blues')
plot_grid(axes[1, 1], ensemble_log.like, 'log(like)', cmap='Reds')
plot_grid(axes[1, 2], ensemble_log.f, 'log(f) = log(fprior * like)', cmap='Greens')

plt.tight_layout()
plt.show()

