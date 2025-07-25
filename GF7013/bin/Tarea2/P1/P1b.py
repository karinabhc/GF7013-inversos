"""
Francisco Ortega Culaciati
ortega.francisco@uchile.cl
GF7013 - Metodos Inversos Avanzados
Departamento de Geofisica - FCFM - Universidad de Chile 
"""

import numpy as np
import matplotlib.pyplot as plt
from .datos import obtener_datos_elipses
# importar modulos relevantes del paquete GF7013
pythonpackagesfolder = '../../../..' # no modificar si no se mueve de carpeta este notebook
import sys
sys.path.append(pythonpackagesfolder)

# modelo directo de la recta
from GF7013.models.ajuste_ortogonal_recta import recta

# generar datos sinteticos

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



# ejemplo de como graficar una linea recta ()
# definiendo la linea ( PUEDE JUGAR CON VARIOS VALORES PARA VER COMO CAMBIA LAS DISTANCIAS
# ENTRE LAS OBSERVACIONES Y LA RECTA)
a = -10
theta = -45 # grados sexagesimales despues de hacer P1.1)
m_plot = np.array([a, theta]) # modelo de recta a graficar.

fig = plt.figure(1)
fig.clear()
fig.set_size_inches((6, 4))
ax = fig.add_subplot(111)
# en el comando siguiente, si el color es 'none' no se grafica el elemento.
recta.plot_recta(ax, *m_plot, x_obs = x_obs, y_obs = y_obs, color_dist='c', color='b') 
ax.errorbar(x = x_obs, y=y_obs, xerr=sigma_x, yerr=sigma_y, fmt='.r', capsize = 2)
ax.axis('equal')
ax.grid('on')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title(' Dibujo de recta paramétrica y distancias a los puntos de observación.')
fig.canvas.draw()
plt.show()
