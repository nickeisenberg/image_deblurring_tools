from scipy.optimize import minimize
import matplotlib.pyplot as plt
import scipy.linalg as alg
from scipy.optimize import minimize
import numpy as np
import cv2

import regressors as reg
import create

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

model = reg.Model()
model.add(2, reg.Erf, 'erf')
model.initialize()

model.fit_nonblind(
    img, 
    img_b,
    np.linspace(-10, 10, 1000),
    np.linspace(-10, 10, 1000),
)

blur_est = cv2.filter2D(
    src=img,
    ddepth=-1,
    kernel=model.kernel(np.linspace(-10, 10, 1000), np.linspace(-10, 10, 1000))
)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(blur_est)
ax[1].imshow(img_b)
plt.show()

plt.plot(img_b[600])
plt.plot(blur_est[600])
plt.show()
