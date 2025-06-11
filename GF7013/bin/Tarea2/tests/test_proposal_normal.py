"""
Modulo para verificar el funcionamiento de la distribución de proposición definida  en proposal_normal.py (definida en el módulo metrópolis del módulo sampling).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ....sampling.metropolis.proposal_normal import proposal_normal

# Parámetros teóricos
cov = np.array([[2.0, 1.0], [1.0, 4.0]])
mu= np.array([0., 0.])

proposal = proposal_normal(cov)

# try 
Ns = 100_000  # number of samples
m0 = np.array([-1.0, 4.0])

# Generación de muestras
m_proposed = np.zeros((Ns, m0.shape[0]))
# m_proposed[0, :] += m0

for k in range(Ns):
    m_test = proposal.propose(m0)
    m_proposed[k, :] = m_test
    # m0 = m_test

