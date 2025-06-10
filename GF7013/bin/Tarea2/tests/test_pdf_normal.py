import sys
import os
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(root_dir)

from GF7013.probability_functions.pdf.pdf_normal import pdf_normal


#parámetros
mu = np.array([-1.0, 4.0])
cov = np.array([[2.0, 1.0], [1.0, 4.0]])

# distribución normal multivariada

par = {'mu': mu, 'cov': cov}
pdf = pdf_normal(par)


# Generación de muestras

Ns = 100000
samples = pdf.draw(Ns).T  # Shape: (Ns, 2)

mu_emp = np.mean(samples, axis=0)
cov_emp = np.cov(samples.T)

print("Media teórica:", mu)
print("Media empírica:", mu_emp)
print("\nCovarianza teórica:\n", cov)
print("Covarianza empírica:\n", cov_emp)


# Histograma conjunto
plt.figure(figsize=(6, 5))
plt.hist2d(samples[:, 0], samples[:, 1], bins=80, cmap='plasma')
plt.colorbar(label="Frecuencia")
plt.xlabel("m1")
plt.ylabel("m2")
plt.title("Histograma conjunto de muestras")
plt.tight_layout()
plt.savefig("hist2d_m1_m2.png")


# Histogramas marginales
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

axs[0].hist(samples[:, 0], bins=80, color='steelblue', edgecolor='black')
axs[0].set_title("Marginal de $m_1$")
axs[0].set_xlabel("m1")

axs[1].hist(samples[:, 1], bins=80, color='salmon', edgecolor='black')
axs[1].set_title("Marginal de $m_2$")
axs[1].set_xlabel("m2")

plt.tight_layout()
plt.savefig("hist_marginales.png")
plt.show()


