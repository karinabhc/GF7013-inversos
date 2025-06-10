"""
Francisco Ortega Culaciati
ortega.francisco@uchile.cl
GF7013 - Metodos Inversos Avanzados
Departamento de Geofisica - FCFM - Universidad de Chile 
"""

import numpy as NP
def obtener_datos_elipses(N = 100, a = 8, b = 4, alpha = -45, deltax = 8, deltay = 8, 
                          sigma_x = 0.5, sigma_y = 0.5):
    """
    Genera N pares ordenados de observaciones que tienen la forma de una elipse de 
    semieje mayor a, semieje menor b, rotada en un angulo alpha, y trasladada en 
    deltax y deltay.

    devuelve [ x, y, sigma_x, sigma_y]
 
    por Francisco Ortega Culaciati
        ortega.francisco@uchile.cl
        GF5013 - Metodos Inversos Aplicados a la Geofisica
        Departamento de Geofisica - FCFM - Universidad de Chile 
    """
    angulos_rad = NP.linspace(0, 360, N+1) * NP.pi/180.0
    alpha_rad = alpha * NP.pi / 180
    # remover el ultimo ya que 360 = 0
    angulos_rad = angulos_rad[0:N]
    # calcular elipse sin rotar
    x =  a * NP.cos(angulos_rad)
    y =  b * NP.sin(angulos_rad)
    # rotar en alpha
    x_rot = x * NP.cos(alpha_rad) - y * NP.sin(alpha_rad)
    y_rot = x * NP.sin(alpha_rad) + y * NP.cos(alpha_rad)
    # trasladar
    x = x_rot + deltax
    y = y_rot + deltay

    # generar arreglos de desviaciones estandar
    sigma_x = NP.ones(N) * sigma_x 
    sigma_y = NP.ones(N) * sigma_y

    return x, y, sigma_x, sigma_y   


def obtener_datos_recta_ruido(N = 100, sigma_x = 0.5, sigma_y= 0.5, 
                              alpha = -45, 
                              deltax = 8, deltay = 8):
    x_lim = 25
    x = NP.linspace(-x_lim, x_lim, N)
    y = 0 * x
    alpha_rad = alpha * NP.pi/180
    # rotar en alpha
    x_rot = x * NP.cos(alpha_rad) - y * NP.sin(alpha_rad)
    y_rot = x * NP.sin(alpha_rad) + y * NP.cos(alpha_rad)
    # trasladar
    x = x_rot + deltax
    y = y_rot + deltay
    # agregar ruido 
    x = x + NP.random.randn(len(x)) * sigma_x
    y = y + NP.random.randn(len(y)) * sigma_y
    # generar arreglos de desviaciones estandar
    sigma_x = NP.ones(N) * sigma_x 
    sigma_y = NP.ones(N) * sigma_y
    
    return x, y, sigma_x, sigma_y
