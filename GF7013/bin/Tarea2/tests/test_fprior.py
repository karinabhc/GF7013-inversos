import numpy as np
import matplotlib.pyplot as plt
from GF7013.probability_functions.pdf.pdf_uniform_nD import pdf_uniform_nD



np.random.seed(0)
N = 100
x_obs = np.random.uniform(-10, 10, size=N)
y_obs = np.random.uniform(-10, 10, size=N)


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

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Histograma conjunto (a, theta)
ax = axes[0, 0]
hist = ax.hist2d(a_samples, theta_samples, bins=[100, 100], cmap='viridis')
plt.colorbar(hist[3], ax=ax)
ax.set_title("Histograma conjunto de (a, θ)")
ax.set_xlabel("a")
ax.set_ylabel("θ (grados)")

# Histograma marginal de a
axes[0, 1].hist(a_samples, bins=100, color='steelblue', edgecolor='k')
axes[0, 1].set_title("Histograma marginal de a")
axes[0, 1].set_xlabel("a")

# Histograma marginal de theta
axes[1, 0].hist(theta_samples, bins=100, color='salmon', edgecolor='k')
axes[1, 0].set_title("Histograma marginal de θ")
axes[1, 0].set_xlabel("θ (grados)")

# Ocultar el subplot inferior derecho
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()