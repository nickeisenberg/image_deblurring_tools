import numpy as np
import matplotlib.pyplot as plt
import cv2

class Image:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.img = np.zeros((height, width))
        self.grid = np.meshgrid(np.arange(width), np.arange(height))

    def square(self, val, width_loc, height_loc):
        self.img[width_loc[0]: width_loc[1], height_loc[0]: height_loc[1]] += val
    
    def circle(self, val, center, radius):
        for C, R in zip(self.grid[0], self.grid[1]):
            for r, c in zip(R, C):
                if (r - center[0]) ** 2 + (c - center[1]) ** 2 <= radius ** 2:
                    self.img[r, c] += val


class Blur:

    def __init__(self, img):
        self.img = img

    # def gaussian(self, mu, std, dim_len, return_kernel=False):
    def gaussian(self, mu, std, return_kernel=False):
        dom_w = np.arange(
            -int(self.img.shape[1] / 2), int(self.img.shape[1] / 2), 1
        )
        dom_h = np.arange(
            -int(self.img.shape[0] / 2), int(self.img.shape[0] / 2), 1
        )
        X, Y = np.meshgrid(dom_w, dom_h)
        r = np.sqrt((X - mu) ** 2 + (Y - mu) ** 2)

        kernel = (1 / np.sqrt(2 * np.pi * std ** 2))
        kernel *= np.exp(-(r ** 2) / (2 * std))
        kernel /= kernel.sum()

        if return_kernel:
            return cv2.filter2D(
                src=self.img,
                ddepth=-1,
                kernel=kernel), kernel
        else:
            return cv2.filter2D(
                src=self.img,
                ddepth=-1,
                kernel=kernel)
    
