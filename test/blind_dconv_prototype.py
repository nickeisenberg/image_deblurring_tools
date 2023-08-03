import regressors as reg
import create
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.linalg as alg
from scipy.optimize import minimize
import numpy as np
import cv2

# create an image to blur
image = create.Image(1000, 1000)
image.square(1, [200, 400], [600, 900])
image.circle(2, (600, 300), 200)

img = image.img
img_b = create.Blur(img).gaussian(0, 100)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(img_b)
plt.show()

# initialize the model
def kernel(P):
    model = reg.Model()
    model.add(1, reg.Erf, 'erf')
    weights = {
        'erf': np.array([[P[0], P[1]]])
    }
    model.initialize(weights=weights)
    X, Y = np.meshgrid(np.linspace(-100, 100, 1000), np.linspace(-100, 100, 1000))
    ker = model.psf(X **2 + Y ** 2)
    return ker

def blur(P):
    img_b_ = cv2.filter2D(
        src=img,
        ddepth=-1,
        kernel=kernel(P))
    return img_b_

def loss(P):
    return alg.norm(blur(P) - img_b)

P = minimize(loss, [1, 1]).x

P = minimize(loss, P).x
P = minimize(loss, P).x

fig, ax = plt.subplots(1, 2)
ax[0].imshow(blur(P))
ax[1].imshow(img_b)
plt.show()

plt.plot(blur(P)[600][:200])
plt.plot(img_b[600][:200])
plt.show()




