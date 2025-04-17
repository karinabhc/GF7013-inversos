import numpy as np
import matplotlib.pyplot as plt
import sys, os 

path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path,'..','..','..')
path = os.path.abspath(path)
sys.path.append(path)


import GF7013
from GF7013.probability_functions import pdf

# Distribución uniforme 2D
lower_lim = np.array([-2.0, 3.2])
upper_lim = np.array([5.0, 7.0])
par = {
    'lower_lim': lower_lim,
    'upper_lim': upper_lim
}


dist = pdf.pdf_uniform_nD(par)

# Generar 1e5 muestras
Ns = int(1e5)
samples = dist.draw(Ns=Ns)

# Separar componentes x1 y x2
x1 = samples[0,:]
x2 = samples[1,:]

# Generar histogramas
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.25, wspace=0.25)

# Histograma marginal de x1
ax_x1 = fig.add_subplot(gs[0, 0])
ax_x1.hist(x1, bins=50, color='skyblue', edgecolor='k')
ax_x1.set_ylabel('Frecuencia')
ax_x1.set_title('Marginal de $x_1$')
ax_x1.grid(True)

# Histograma marginal de x2
ax_x2 = fig.add_subplot(gs[1, 1])
ax_x2.hist(x2, bins=50, orientation='horizontal', color='salmon', edgecolor='k')
ax_x2.set_xlabel('Frecuencia')
ax_x2.set_title('Marginal de $x_2$')
ax_x2.grid(True)

# Histograma 2D
ax_joint = fig.add_subplot(gs[1, 0])
counts, xedges, yedges, im = ax_joint.hist2d(x1, x2, bins=50, cmap='viridis')
fig.colorbar(im, ax=ax_joint, label='Cuentas')
ax_joint.set_xlabel('$x_1$')
ax_joint.set_ylabel('$x_2$')
ax_joint.set_title('Histograma 2D (aproximación fdp conjunta)')
ax_joint.grid(True)

plt.tight_layout()
plt.show()

