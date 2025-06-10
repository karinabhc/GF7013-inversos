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
path = os.path.join('..','..','..')
path = os.path.abspath(path)
sys.path.append(path)
print('El paquete GF7013 debería estar dentro de la carpeta:', path)

# now import GF7013
import GF7013 
# or directly import pmf
from GF7013.probability_functions import pmf

# In order to instantiate a multinomial probability, I need three things:
# a) a random number generator instance
rng = np.random.default_rng()

# b) a list of values representing a set of states for the random variable x.
values = np.array(['Spurious', 'Guess', 'No Idea', 'True value?', 'invalid?'])

# c) a list with the corresponding importance (unnormalized mass or probability) of 
#    each state described in values.
importance = np.array([1, 3, 2.2, 5, 0.1])

# assemble the dictionary of model parameters
par = {}
par['values'] = values
par['importance'] = importance
par['method'] = 'numpy'

# instantiate the multinomial pmf
multi = pmf.pmf_multinomial(par = par, rng = rng)
# print the probability of each discrete state
print('The probabilities for each discrete state are')
for i, value in enumerate(multi.values):
    print(value, 'with probability', multi.prob[i])
print('Total Probability is:', np.sum(multi.prob))
# draw some samples (TO DO: PLAY WITH DIFFERENT number of samples Ns)
Ns = 10000
samples = multi.draw(Ns = Ns)
# make a histogram of the draws
count_values = np.zeros(multi.values.shape)
values2index = dict(zip(multi.values, range(len(multi.values))))
print(values2index)

for value in samples:
    i = values2index[value]
    count_values[i] += 1

normalized_count_values = count_values / np.sum(count_values)

width = 0.25
N = len(multi.values)
index = np.arange(N)

prob = np.array([multi.eval(x) for x in multi.values])


fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.bar(index, normalized_count_values, width, color ='red', label = 'Normalized Sample Count')
ax.bar(index + width, prob, width, color = 'blue', label = 'Probability of each state' )
ax.set_xticks(index + width/2, multi.values)
ax.set_title(f'Ns = {Ns:d} samples normalized count v/s probability')
ax.set_xlabel('Label of each discrete state')
ax.legend()
plt.show()
