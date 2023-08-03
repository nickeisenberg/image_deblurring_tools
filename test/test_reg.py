import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as _erf
from copy import deepcopy
from scipy.optimize import minimize
from copy import deepcopy
import cv2

import regressors as reg
import create

# make data
domain = np.linspace(-20, 20, 1000)
data = 2.1 * np.arctan(2 * (domain)) + np.random.normal(0, .3, 1000)
m, M = data.min(), data.max()
data -= m
data /= (M - m)

# make model
model = reg.Model()

erf_weights = np.array([[.25, 1], [.25, 1]])
model.add_initialize(2, erf_weights, reg.Erf, 'erf')

at_weights = np.array([[.25, 1], [.25, 1]])
model.add_initialize(2, at_weights, reg.Arctan, 'arctan')

model.fit_to_esf(data, domain, iters=10, verbose=False)

plt.plot(data)
plt.plot(model.esf(domain))
plt.show()

# make kernal and blur image
image = create.Image(1000, 1000)
image.square(1, [100, 500], [100, 400])
image.circle(2, (600, 600), 150)

plt.imshow(model.kernel(domain, domain))
plt.show()

plt.imshow(image.img)
plt.show()

img_b = cv2.filter2D(
    src=image.img,
    ddepth=-1,
    kernel=model.kernel(domain, domain))

plt.imshow(img_b)
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

len(model.weights_vec_id)

cons = []
for i in range(len(model.weights_vec_id)):
    cons.append((-np.inf, np.inf)), (0, np.inf)
    cons.append((0, np.inf))
cons

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
