"""
Francisco Ortega Culaciati
ortega.francisco@uchile.cl
GF7013 - Metodos Inversos Avanzados
Departamento de Geofisica - FCFM - Universidad de Chile 
"""
import numpy as np

####
def recta_par(s, a, theta):
    """
     - s: es arreglo numpy de 1 dimension, 
     - a: es un numero real (float): distancia de la recta al origen del sistema de 
          coordenadas
     - theta: en grados sexagesimales (float) es la orientacion de la recta medida en 
             sentido antihorario desde el eje x (lado positivo).

     devuelve x(s), y(s), n(theta) [vector normal], t(theta) [vector tangente]
    """
    theta = theta * np.pi / 180.0
    # vector tangente:
    t = np.array([np.cos(theta), np.sin(theta)])
    # vector normal
    n = np.array([-np.sin(theta), np.cos(theta)])
    # x, y
    xy = np.array([a*n + svalue*t for svalue in s])
    
    return xy[:,0], xy[:,1], n, t


####
def calc_xy_pred(a, theta, x_obs, y_obs):
    """
    Calculo de la prediccion de un punto en la recta, son las 
    coordenadas (x_pred, y_pred) de la recta que son las mas cercanas al par 
    ordenado observado.

    - a: es un numero real (float): distancia de la recta al origen del sistema de
         coordenadas
    - theta: en grados sexagesimales (float): la orientacion de la recta medida en
             sentido antihorario desde el eje x (lado positivo).
    - x_obs: numpy.array de 1D con valores X observados
    - y_obs: numpy.array de 1D con valores Y observados

    devuelve [x_pred, y_pred, s]
    """
    # calcular t
    _, _, n, t = recta_par(s=np.array([0]), a=a, theta=theta)
    
    # ahora necesito obtener la proyeccion ortogonal de cada punto en la recta.
    s = np.array([np.array([x_obs[i], y_obs[i]]).dot(t) for i in range(0, len(x_obs))])
    x_pred, y_pred, n, t = recta_par(s, a, theta)
    
    return x_pred, y_pred, s


####
def calc_dist_sigma(m, x_obs, y_obs, sigma_x, sigma_y):
    """
    calcula el vector de distancias deltas (numpy.array de 1D) con las distancias de 
    los pares ordenados observados a la recta definida por m = [a, theta] (theta en 
    grados sexagesimales). Asimismo calcula el vector sigma_deltas con las desviaciones
    estandar de dichas distancias.

    devuelve [deltas, sigma_deltas, s, e_x, e_y], aqui s es el parámetro de la recta donde
    se calcula la predicción para cada punto de observacion. 

    """
    a, theta = m
    # calcular prediccion ortogonal de la recta
    x_pred, y_pred, s = calc_xy_pred(a, theta, x_obs, y_obs) 
    e_x = x_pred - x_obs
    e_y = y_pred - y_obs
    # calcular distancias
    deltas = np.sqrt(e_x**2 + e_y**2)
    # calcular desviaciones estandar de las distancias
    sigma_deltas = np.sqrt( (e_x/deltas * sigma_x)**2  + (e_y/deltas * sigma_y)**2 )
    
    return deltas, sigma_deltas, s, e_x, e_y
