# -*- python -*-
# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofísica - FCFM
Universidad de Chile


Make some testing of the multivariate normal distribution
Modifications: 

"""
import numpy as np
import sys, os 
import matplotlib.pyplot as plt

# add GF7013 location to PYTHONPATH
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path,'..','..','..')
path = os.path.abspath(path)
sys.path.append(path)
print('El paquete GF7013 debería estar dentro de la carpeta:', path)

# now import GF7013
import GF7013 
from GF7013.probability_functions import pdf

# defina funciones acá

#### SECCION 3.4 
def test_draw(pdf_object, n_samples=1000, seed=42):
    """
    Prueba si la generación de muestras de la distribución normal multivariada
    funciona correctamente, usando una semilla específica para garantizar reproducibilidad.
    
    Parámetros:
    - pdf_object: Instancia de pdf_normal con mu y cov especificados
    - n_samples: Número de muestras para generar
    - seed: Semilla para el generador de números aleatorios
    
    Retorna:
    - True si la generación de muestras parece correcta
    - False si se detectan problemas
    """
    # Guardar el estado actual del generador de números aleatorios
    original_rng = pdf_object.rng
    
    # Crear un nuevo generador con la semilla específica
    test_rng = np.random.default_rng(seed)
    
    # Reemplazar temporalmente el generador
    pdf_object.rng = test_rng
    
    try:
        # Generar muestras con la semilla controlada
        samples = pdf_object.draw(n_samples)
        
        # Calcular estadísticas de las muestras
        if samples.shape[0] == pdf_object.N:
            samples_t = samples.T  # Transponer si es necesario
        else:
            samples_t = samples
            
        sample_mean = np.mean(samples_t, axis=0)
        sample_cov = np.cov(samples_t, rowvar=False)
        
        # Verificar media
        mean_error = np.linalg.norm(sample_mean - pdf_object.mu)
        mean_threshold = 3 * np.sqrt(np.trace(pdf_object.cov) / n_samples)
        
        # Verificar covarianza
        cov_error = np.linalg.norm(sample_cov - pdf_object.cov, 'fro')
        cov_threshold = 3 * np.linalg.norm(pdf_object.cov, 'fro') / np.sqrt(n_samples)
        
        result = (mean_error < mean_threshold) and (cov_error < cov_threshold)            
        # print(f"Test con semilla {seed} y {n_samples} muestras:")
        # print(f"Error en media: {mean_error:.6f} (umbral: {mean_threshold:.6f}) - {'OK' if mean_error < mean_threshold else 'FALLA'}")
        # print(f"Error en covarianza: {cov_error:.6f} (umbral: {cov_threshold:.6f}) - {'OK' if cov_error < cov_threshold else 'FALLA'}")
        # print(f"Resultado: {'PASA' if result else 'FALLA'}\n")
        
        return result
        
    finally:
        # Restaurar el generador original pase lo que pase
        pdf_object.rng = original_rng


if __name__ == '__main__':
        # Parameters of the multivariate normal distribution
    mu = np.array([0.5, 3.0])
    C = np.array([[2, 1], [1, 4]])

    norm=pdf.pdf_normal({'mu': mu, 'cov': C})
    # Generate 1E5 samples
    Ns= int(1e5)
    samples = norm.draw(Ns=Ns)

    x1 = samples[0,:]
    x2 = samples[1,:]

    fig = plt.figure(figsize=(12, 8), layout='constrained')
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1], hspace=0.25, wspace=0.25)

    # Create 2D histogram
    ax_join = fig.add_subplot(gs[1, 0])
    counts, xedges, yedges, im = ax_join.hist2d(x1, x2, bins=50, cmap='viridis')
    fig.colorbar(im, ax=ax_join,label='Cuentas')
    ax_join.set_title('2D Histogram (Joint PDF)')
    ax_join.set_xlabel('X1')
    ax_join.set_ylabel('X2')

    # Marginal histogram for X1
    ax_x1 = fig.add_subplot(gs[0, 0])
    x1_hist, x1_bins, _ = ax_x1.hist(x1, bins=100, density=True, alpha=0.6, color='blue', label='Histogram')
    x1_pdf = (1 / np.sqrt(2 * np.pi * C[0, 0])) * np.exp(-0.5 * ((x1_bins - mu[0]) ** 2) / C[0, 0])
    ax_x1.plot(x1_bins, x1_pdf, color='red', label='PDF')
    ax_x1.set_title('Marginal Histogram for X1')
    ax_x1.set_xlabel('X1')
    ax_x1.set_ylabel('Density')
    ax_x1.legend()

    # Marginal histogram for X2
    ax_x2 = fig.add_subplot(gs[1, 1])
    x2_hist, x2_bins, _ = ax_x2.hist(x2, bins=100, density=True, alpha=0.6, color='green', label='Histogram')
    x2_pdf = (1 / np.sqrt(2 * np.pi * C[1, 1])) * np.exp(-0.5 * ((x2_bins - mu[1]) ** 2) / C[1, 1])
    ax_x2.plot(x2_bins, x2_pdf, color='red', label='PDF')
    ax_x2.set_title('Marginal Histogram for X2')
    ax_x2.set_xlabel('X2')
    ax_x2.set_ylabel('Density')
    ax_x2.legend()

    # plt.tight_layout()
    plt.show()