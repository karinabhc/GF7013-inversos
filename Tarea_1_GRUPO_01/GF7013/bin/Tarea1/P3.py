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
def create_histogram2D(samples, suptitle, figsize=(12, 8), bins_2d=50, bins_xs=100, cmap='viridis'):
    """
    Función para crear los histogramas 2D y 1D de las muestras x1 y x2 en  subplots diferentes
    """
    x1 = samples[0,:]
    x2 = samples[1,:]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize, layout='constrained')
    
    # Create 2D histogram (lower left [1, 0])
    counts, xedges, yedges, im = axs[1, 0].hist2d(x1, x2, bins=bins_2d, cmap=cmap)
    fig.colorbar(im, ax=axs[1, 0],label='N Muestras', location='bottom', orientation='horizontal', fontsize=15)
    axs[1, 0].set_title('Histograma Marginal 2D (FDP Conjunta)', fontsize=16)
    axs[1, 0].set_xlabel(r'$X_1$', fontsize=15)
    axs[1, 0].set_ylabel(r'$X_2$', fontsize=15)

    # Marginal histogram for $X_1$ (upper left [0, 0])
    x1_hist, x1_bins, _ = axs[0, 0].hist(x1, bins=bins_xs, density=True, alpha=0.6, color='blue', label='Histograma')
    x1_pdf = (1 / np.sqrt(2 * np.pi * C[0, 0])) * np.exp(-0.5 * ((x1_bins - mu[0]) ** 2) / C[0, 0])
    
    # FDP $X_1$
    axs[0, 0].plot(x1_bins, x1_pdf, color='red', label='FDP')
    axs[0, 0].set_title(r'Histograma Marginal $X_1$', fontsize=16)
    axs[0, 0].set_xlabel(r'$X_1$', fontsize=15)
    axs[0, 0].set_ylabel('Densidad', fontsize=15)
    axs[0, 0].legend()

    # Marginal histogram for $X_2$ (lower right [1, 1])
    x2_hist, x2_bins, _ = axs[1, 1].hist(x2, bins=bins_xs, density=True, alpha=0.6, color='green', label='Histograma', orientation='horizontal')
    x2_pdf = (1 / np.sqrt(2 * np.pi * C[1, 1])) * np.exp(-0.5 * ((x2_bins - mu[1]) ** 2) / C[1, 1])
    
    axs[1, 1].plot(x2_pdf, x2_bins, color='red', label='FDP')

    axs[1, 1].set_title(r'Histograma Marginal $X_2$', fontsize=16)
    axs[1, 1].set_ylabel(r'$X_2$', fontsize=15)
    axs[1, 1].set_xlabel('Densidad', fontsize=15)
    axs[1, 1].legend()
    
    # Marginal histogram for $X_1$ and $X_2$ (upper right [0, 1])
    x1_hist, x1_bins, _ = axs[0, 1].hist(x1, bins=bins_xs, density=True, alpha=0.6, color='blue', label=r'Histograma $X_1$')
    x1_pdf = (1 / np.sqrt(2 * np.pi * C[0, 0])) * np.exp(-0.5 * ((x1_bins - mu[0]) ** 2) / C[0, 0])
    x2_hist, x2_bins, _ = axs[0, 1].hist(x2, bins=bins_xs, density=True, alpha=0.6, color='green', label=r'Histograma $X_2$')
    x2_pdf = (1 / np.sqrt(2 * np.pi * C[1, 1])) * np.exp(-0.5 * ((x2_bins - mu[1]) ** 2) / C[1, 1])
    
    # FDPs
    axs[0, 1].plot(x1_bins, x1_pdf, color='red', label=r'FDP $X_1$')
    axs[0, 1].plot(x2_bins, x2_pdf, color='red', label=r'FDP $X_2$',  ls='--')

    axs[0, 1].set_title('Histograma Marginales', fontsize=16)
    axs[0, 1].set_xlabel('X', fontsize=15)
    axs[0, 1].set_ylabel('Densidad', fontsize=15)
    axs[0, 1].legend()
    
    
    fig.suptitle(suptitle, fontsize=20)
    fig.subplots_adjust(left=0.01)
    return fig

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
    
    result = (mean_error < mean_threshold) and (cov_error < cov_threshold)        # retorna True o False    
    # print(f"Test con semilla {seed} y {n_samples} muestras:")
    # print(f"Error en media: {mean_error:.6f} (umbral: {mean_threshold:.6f}) - {'OK' if mean_error < mean_threshold else 'FALLA'}")
    # print(f"Error en covarianza: {cov_error:.6f} (umbral: {cov_threshold:.6f}) - {'OK' if cov_error < cov_threshold else 'FALLA'}")
    # print(f"Resultado: {'PASA' if result else 'FALLA'}\n")
    
    # Restaurar el generador original
    pdf_object.rng = original_rng
    
    return result # retornamos el resultado del test (si aprueba o no)


if __name__ == '__main__':
    # Parameters of the multivariate normal distribution
    mu = np.array([0.5, 3.0])
    C = np.array([[2, 1], [1, 4]])

    norm=pdf.pdf_normal({'mu': mu, 'cov': C})
    
    # Generate 1E5 samples
    Ns1= int(1e5)
    samples1 = norm.draw(Ns=Ns1)
    
    # Generate 1E4 samples
    Ns2= int(1e4)
    samples2 = norm.draw(Ns=Ns2)
    # Generate 1E6 samples
    Ns3= int(1e6)
    samples3 = norm.draw(Ns=Ns3)
    
    # Generate the plots for each set of samples
    create_histogram2D(samples=samples1, suptitle=f'Distribución Normal Multivariada con N={Ns1:.1e} muestras', figsize=(10, 8))
    # create_histogram2D(samples=samples2, suptitle=f'Distribución Normal Multivariada con N={Ns2:.1e} muestras')
    # create_histogram2D(samples=samples3, suptitle=f'Distribución Normal Multivariada con N={Ns3:.1e} muestras')

    
    
    # prueba test_draw
    resultado_test = test_draw(norm, n_samples=100_000)
    info_test = f'El resultado de test_draw fue {resultado_test}'
    try:
        assert(resultado_test, info_test) 
        print('Felicidades, test_draw aprobó el test.')
    except:
        print(info_test)
        
    
    
    plt.show()
