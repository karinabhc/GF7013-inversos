"""
Defines a class that implements the forward model for the 
orthogonal straight line fit. As this problem is nonlinear, returns the misfit vector 
corresponding to the distance between observations and straight line divided by the 
standard deviation of such distance.

A forward model class must implement the method .eval(m) that returns the forward
calculation of model m.

Francisco Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofisica - FCFM - Universidad de Chile 

"""
COMPLETAR = None
from . import recta

class forward(object):
    """
    Defines a class that implements the forward model for the 
    orthogonal straight line fit. As this problem is nonlinear, returns the misfit vector 
    corresponding to the distance between observations and straight line divided by the 
    standard deviation of such distance.
    """

    def __init__(self, x_obs, y_obs, sigma_x, sigma_y):
        """
        The arguments are those of .F_JF.getF():
        x_obs, y_obs, sigma_x, sigma_y: all 1D numpy.arrays define observed x and y 
        (x_obs, y_obs) and their respective standard deviations (sigma_x, sigma_y).
        """
        self.x_obs = x_obs
        self.y_obs = y_obs
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y

    def eval(self, m):
        """
        Computes a prediction of the model parameters (see description in this
        module docstring).
        - m = NP.array([a, theta]) with theta in degrees (both float quantities). 
          -> a is the distance between straight line to origin of coordinate system and 
          -> theta is the orientation of the straight line measured counter-clockwise 
            measured from x axis. 
        """

        dpred = COMPLETAR

        return dpred
