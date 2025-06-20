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

