import numpy as np
import matplotlib.pyplot as plt
from GF7013.probability_functions.pdf.pdf_uniform_nD import pdf_uniform_nD
from GF7013.bin.Tarea2.P1.datos import obtener_datos_elipses
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


a_min = -2 * max_norma
a_max = 2 * max_norma
theta_min = -180
theta_max = 180  

lower_lim = np.array([a_min, theta_min])
upper_lim = np.array([a_max, theta_max])

par = {
    'lower_lim': lower_lim,
    'upper_lim': upper_lim
}

fprior = pdf_uniform_nD(par)

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

