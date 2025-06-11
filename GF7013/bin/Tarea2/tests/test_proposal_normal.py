"""
Modulo para verificar el funcionamiento de la distribución de proposición
definida en proposal_normal.py (definida en el módulo metrópolis del módulo
sampling). 

La idea es verificar que la distribución de proposición definida (para generar 
m_test) sigue correctamente una distribución normal centrada en m0 (modelo inicial).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ....sampling.metropolis.proposal_normal import proposal_normal

# Parámetros iniciales 
m0 = np.array([-1.0, 4.0])  # modelo inicial a perturbar
cov = np.array([[2.0, 1.0], [1.0, 4.0]])

# creación de instancia de proposal_normal
proposal = proposal_normal(cov)

# verificando la distribución de las propuestas 
Ns = 100_000  # numero de muestras
m_proposed = np.zeros((Ns, m0.shape[0]))
m_proposed[0, :] += m0

for k in range(1, Ns):
    m_test = proposal.propose(m0)
    m_proposed[k, :] = m_test

x1 = m_proposed[:, 0]
x2 = m_proposed[:, 1]

# graficando los resultados
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                       wspace=0.05, hspace=0.05)

# Histograma conjunto
ax_join = fig.add_subplot(gs[1, 0])
counts, xedges, yedges, im = ax_join.hist2d(x1, x2, bins=50, cmap='viridis')
fig.colorbar(im, ax=ax_join, label='Cuentas')
ax_join.set_title('2D Histogram (Joint PDF)')
ax_join.set_xlabel('X1')
ax_join.set_ylabel('X2')

# Marginal para X1
ax_x1 = fig.add_subplot(gs[0, 0], sharex=ax_join)
x1_hist, x1_bins, _ = ax_x1.hist(x1, bins=100, density=True, alpha=0.6, color='blue', label='Histograma')

# Usar centros de bins para evaluar la PDF
x1_centers = 0.5 * (x1_bins[:-1] + x1_bins[1:])
x1_pdf = (1 / np.sqrt(2 * np.pi * cov[0, 0])) * np.exp(-0.5 * ((x1_centers - m0[0]) ** 2) / cov[0, 0])
ax_x1.plot(x1_centers, x1_pdf, color='red', label='PDF teórica')
ax_x1.set_title('Marginal para X1')
ax_x1.set_ylabel('Densidad')
ax_x1.legend()
ax_x1.tick_params(labelbottom=False)

pos = ax_x1.get_position()
ax_x1.set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height])

# Marginal para X2
ax_x2 = fig.add_subplot(gs[1, 1], sharey=ax_join)
x2_hist, x2_bins, _ = ax_x2.hist(x2, bins=100, density=True, alpha=0.6, color='green', orientation='horizontal', label='Histograma')
x2_centers = 0.5 * (x2_bins[:-1] + x2_bins[1:])
x2_pdf = (1 / np.sqrt(2 * np.pi * cov[1, 1])) * np.exp(-0.5 * ((x2_centers - m0[1]) ** 2) / cov[1, 1])
ax_x2.plot(x2_pdf, x2_centers, color='red', label='PDF teórica')
ax_x2.set_title('Marginal para X2')
ax_x2.set_xlabel('Densidad')
ax_x2.legend()
ax_x2.tick_params(labelleft=False)

fig.suptitle("Distribución Normal Bivariada: Histograma conjunto y marginales", fontsize=14, y=0.965)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

