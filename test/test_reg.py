import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as _erf
from copy import deepcopy
from scipy.optimize import minimize
from copy import deepcopy

import regressors as reg


model = reg.Model()

erf_weights = np.array([[.25, 1], [.25, 1]])
model.add_initialize(2, erf_weights, reg.Erf, 'erf')

at_weights = np.array([[.25, 1], [.25, 1]])
model.add_initialize(2, at_weights, reg.Arctan, 'arctan')

domain = np.linspace(-5, 5, 1000)
data = 2.1 * np.arctan(2 * (domain)) + np.random.normal(0, .3, 1000)
m, M = data.min(), data.max()
data -= m
data /= (M - m)

model.fit_to_esf(data, domain)

model.fit_to_esf(data, domain)

plt.plot(reg.Arctan(*model.weights['arctan'][1]).esf(np.linspace(-5, 5, 100)))
plt.show()


plt.imshow(model.kernel(50, 50))
plt.show()

plt.plot(model.psf(np.linspace(-50, 50, 1000)))
plt.show()

plt.plot(data)
plt.plot(model.esf(domain))
plt.show()
#--------------------------------------------------

#--------------------------------------------------
model = reg.Model()

model.add(2, reg.Erf, 'erf')
model.add(2, reg.Arctan, 'arctan')
model.add(2, reg.Erf, 'erf')

# weights = {
#     'erf': np.array([[2, 1], [1, 1], [.2, .2], [1, 44]]),
#     'arctan': np.array([[.2, 1], [2, 22]]),
# }
# model.initialize(weights=weights)

model.initialize()

model.weights_vec

model.weights_vec_id

model.weights

domain = np.linspace(-5, 5, 1000)
data = 2.1 * np.arctan(2 * (domain)) + np.random.normal(0, .3, 1000)
m, M = data.min(), data.max()
data -= m
data /= (M - m)

model.fit_to_esf(data, domain)
model.fit_to_esf(data, domain)

plt.plot(data)
plt.plot(model.esf(domain))
plt.show()
