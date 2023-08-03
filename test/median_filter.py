import numpy as np
import matplotlib.pyplot as plt
import create
from copy import deepcopy
import numba

@numba.njit(parallel=True)
def median_filt(image, win_r, win_c):
    filtered_image = np.zeros(image.shape)
    row_pad = int((win_r - 1) / 2)
    col_pad = int((win_c - 1) / 2)
    for row_idx in np.arange(row_pad, image.shape[0] - row_pad):
        for col_idx in np.arange(col_pad, image.shape[0] - col_pad):
            med = np.median(
                image[
                    row_idx - row_pad: row_idx + row_pad + 1,
                    col_idx - col_pad: col_idx + col_pad + 1
                ]
            )
            filtered_image[row_idx, col_idx] = med
    return filtered_image

shape = create.Image(1000, 1000)
shape.square(2, [200, 700], [100, 400])
shape.circle(1, (700, 700), 200)
img_n = shape.img + np.random.normal(0, .2, (1000, 1000))

fig, ax = plt.subplots(1, 3)
ax[0].imshow(shape.img)
ax[1].imshow(img_n)
ax[2].imshow(median_filt(img_n, 9, 9))
plt.show()
