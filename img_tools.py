import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as _erf

class Erf:
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def esf(self, x):
        return self.a * .5 * (_erf(self.b * x) + 1)

    def psf(self, r):
        return (self.a * self.b / np.pi) * np.exp(-(self.b ** 2) * (r ** 2))

class Arctan:
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def esf(self, x):
        return self.a* (np.arctan(self.b * x) + (np.pi / 2)) / np.pi
    
    def psf(self, r):
        return (self.a * self.b ** 2) / (
            2 * np.pi * (1 + (self.b * r) ** 2) ** (3 / 2)
        )

class Bennent:
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def esf(self, x):
        return self.a * .5 * (
            1 + ((self.b * x) / np.sqrt((self.b * x) ** 2 + 1))
        )


    def psf(self, r):
        return self.a * (self.b ** 2 / (self.b ** 2 + r))

class PSF:

    def __init__(self, no_erf, no_arc, no_ben, weights=None):
        self.no_erf = no_erf
        self.no_arc = no_arc
        self.np_ben = no_ben
        self.weights = weights
    
    def __init__esf(self):
        return None

    def make_esf_and_fit(self):
        return None

    def create_kernal(self):
        return None

    def blur_image(self):
        return None


