# -*- python -*-
# -*- coding: utf-8 -*-
"""
Francisco Hernán Ortega Culaciati
ortega.francisco@uchile.cl
Departamento de Geofísica - FCFM
Universidad de Chile


Make some testing of the multinomial distribution 

Modifications: 

"""
import numpy as np
import sys, os 
import matplotlib.pyplot as plt

# add GF7013 location to PYTHONPATH
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path,'..','..','..')
# path = os.path.join('..','..','..')
path = os.path.abspath(path)
sys.path.append(path)
print('El paquete GF7013 debería estar dentro de la carpeta:', path)

# now import GF7013
# import GF7013 
from GF7013.probability_functions import pmf

# defina funciones acá
# =====================================
# FUNCIÓN PARA OBTENER MUESTRAS Y CONTEO
# =====================================
def freq(metodo, values, importance, rng, Ns=10000):
    par = {
        'values': values,
        'importance': importance,
        'method': metodo
    }
    multi = pmf.pmf_multinomial(par=par, rng=rng)
    samples = multi.draw(Ns=Ns)

    count_values = np.zeros(multi.values.shape)
    values2index = dict(zip(multi.values, range(len(multi.values))))
    for value in samples:
        i = values2index[value]
        count_values[i] += 1

    normalized_count_values = count_values / np.sum(count_values)
    return multi, normalized_count_values



if __name__ == '__main__':
    # a) RNG
    rng = np.random.default_rng()

    # b) Valores
    values = np.array(['Spurious', 'Guess', 'No Idea', 'True value?', 'invalid?'])

    # c) Importancia (masa no normalizada)
    importance = np.array([1, 3, 2.2, 5, 0.1])
    # Número de muestras
    Ns = 10_000

    # Frecuencia con método numpy
    multi_numpy, freq_numpy = freq('numpy', values, importance, rng, Ns)

    # Frecuencia con método analog
    multi_analog, freq_analog = freq('analog', values, importance, rng, Ns)

    # Probabilidad teórica
    prob = np.array([multi_numpy.eval(x) for x in values])

    # ===================
    # GRÁFICO DE COMPARACIÓN
    # ===================
    width = 0.25
    N = len(values)
    index = np.arange(N)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.bar(index - width, prob, width, color='blue', label='Probabilidad teórica')
    ax.bar(index, freq_numpy, width, color='red', label='numpy')
    ax.bar(index + width, freq_analog, width, color='green', label='análogo')

    ax.set_xticks(index)
    ax.set_xticklabels(values, rotation=30,  fontsize=16)
    ax.set_title(f'Ns = {Ns:d} muestras normalizada cuenta v/s probabilidades',  fontsize=20)
    ax.set_xlabel('Estados discretos', fontsize=17)
    ax.set_ylabel('Probabilidad / Frecuencia',  fontsize=17)
    ax.legend(fontsize=16)
    
    plt.tight_layout()
    plt.show()
