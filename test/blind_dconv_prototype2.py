from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.linalg as alg
from scipy.optimize import minimize
import numpy as np
import cv2
from skimage.restoration import richardson_lucy

import regressors as reg
import create

shape = create.Image(20, 20)

shape.square(2, [5, 15], [5, 15])

plt.imshow(shape.img)
plt.show()

blur = create.Blur(shape.img)

img_b = blur.gaussian(0, .3)

plt.imshow(img_b)
plt.show()

def psf(P):
    dom = np.linspace(-10, 10, 20)
    X, Y = np.meshgrid(dom, dom)
    Z = X ** 2 + Y ** 2
    return reg.Erf(P[0], P[1]).psf(Z)

def loss(P, target=img_b):
    param = P[:2]
    img = P[2:].reshape((20, 20))
    blur_est = cv2.filter2D(
        src=img,
        ddepth=-1,
        kernel=psf(param)
    )
    return alg.norm(blur_est - target, 2)

guess = np.ones(2 + 400)

opt_params = minimize(loss, guess).x

img_recon = opt_params[2:].reshape((20, 20))

plt.imshow(img_recon)
plt.show()
