"""
Francisco Ortega Culaciati
ortega.francisco@uchile.cl
GF7013 - Metodos Inversos Avanzados
Departamento de Geofisica - FCFM - Universidad de Chile 
"""
import numpy as NP
from .recta import calc_xy_pred, recta_par

###
def plot_recta(ax,  a, theta, s = None, x_obs = None, y_obs = None, 
               color = 'b', color_dist = 'c', alpha = 1):
    """
    Función para graficar una recta paramétrica.
    La recta se grafica en valores dados de s, salvo que se provea x_obs y y_obs, 
    en cuyo caso el parametro s dado no se utiliza, y se grafica la recta en los puntos
    de predicción de las observaciones x_obs, y_obs.

    - ax: matplotlib axis donde se desea graficar la recta. 
    - s: valores para el parámetro de la recta
    - a: distancia de la recta al origen del sistema de coordenadas
    - theta: orientación de la recta (grados sexagesimales).
    - x_obs, y_obs: 1D numpy.array, valores de las observaciones. 
    - color: color de la recta
    - color_dist: si no es None, grafica con el color indicado la linea que une el valor
                  observado con el valor predicho en la recta. 
    - alpha: controla la transparencia con que se grafica la recta. 

    """ 
    
    
    if x_obs is None:
        x_pred, y_pred, n, t = recta_par(s, a, theta)
        ax.plot(x_pred, y_pred, 'o', color = color, 
                alpha = alpha)

    else:
        # graficar datos observados
        ax.plot(x_obs, y_obs, 'or', markersize = 4)
        # calcular prediccion de los datos observados
        x_pred, y_pred, s = calc_xy_pred(a, theta, x_obs, y_obs)
        # graficar recta
        ax.plot(x_pred, y_pred, '-', color = color, 
                alpha = alpha)
        if color_dist is not None:
            for i in range(0, len(x_pred)):
                ax.plot([x_pred[i], x_obs[i]],
                        [y_pred[i], y_obs[i]],
                        linestyle = '-', 
                        color = color_dist, 
                        alpha = alpha)
