import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as _erf
from copy import deepcopy

class Erf:
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def esf(self, x):
        return self.a * .5 * (_erf(self.b * x) + 1)

    def psf(self, r):
        return (self.a * self.b / np.pi) * np.exp(-(self.b ** 2) * ((r) ** 2))

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

class Bennett:
    def __init__(self, a=1, b=1):
        self.a = a
        self.b = b

    def esf(self, x):
        return self.a * .5 * (
            1 + ((self.b * x) / np.sqrt((self.b * x) ** 2 + 1))
        )

    def psf(self, r):
        return self.a * (self.b ** 2 / (self.b ** 2 + (r) ** 2))

class Model:

    def __init__(self):
        self.reg = {}
        self.reg_counts = {}
        self.weights = {}
        self.weights_vec = np.array([])

    def add(self, count, reg_class, reg_name):
        self.reg[reg_name] = reg_class
        try:
            self.reg_counts[reg_name] += count
        except:
            self.reg_counts[reg_name] = count
        return None

    def initialize(self, weights=None):
        
        if weights == None:
            total = np.array([*self.reg_counts.values()]).sum()
            for rt, count in self.reg_counts.items():
                self.weights[rt] = np.ones((count, 2)) * [1 / total, 1]
                self.weights_vec = np.vstack(
                    [*self.weights.values()]
                ).reshape(-1)

        else:
            self.weights = weights
            self.weights_vec = np.vstack(
                [*weights.values()]
            ).reshape(-1)

        return None

    def add_initialize(self, count, weights, reg_class, reg_name):

        self.reg[reg_name] = reg_class

        try:
            self.reg_counts[reg_name] += count
        except:
            self.reg_counts[reg_name] = count

        try:
            self.weights[reg_name] = np.vstack(
                (self.weights[reg_name], weights)
            )
        except:
            self.weights[reg_name] = weights

        try:
            self.weights_vec = np.hstack(
                (self.weights_vec, weights.reshape(-1))
            )
        except:
            self.weights_vec = weights.reshape(-1)


    def esf(self, x):
        esf_ = 0 
        for rt, W in self.weights.items():
            if self.reg_counts[rt] == 0:
                continue
            for w in W:
                esf_ += self.reg[rt](*list(w)).esf(x)
        return esf_

    def psf(self, r):
        psf_ = 0 
        for rt, W in self.weights.items():
            if self.reg_counts[rt] == 0:
                continue
            for w in W:
                psf_ += self.reg[rt](*list(w)).psf(r)
        return psf_

# Testing the classes

#--------------------------------------------------
model = Model()

erf_weights = np.array([[.25, 1], [.25, 1]])
model.add_initialize(2, erf_weights, Erf, 'erf')

at_weights = np.array([[.25, 1], [.25, 1]])
model.add_initialize(2, at_weights, Arctan, 'arctan')

model_esf = model.esf(np.linspace(-10, 10, 1000))
model_psf = model.psf(np.linspace(-10, 10, 1000))

plt.plot(model_psf)
plt.plot(model_esf)
plt.show()
#--------------------------------------------------

#--------------------------------------------------
model = Model()

model.add(2, Erf, 'erf')
model.add(2, Arctan, 'arctan')

weights = {
    'erf': np.array([[2, 1], [1, 1]]),
    'arctan': np.array([[.2, 1], [2, 22]]),
}
model.initialize(weights=weights)

model_esf = model.esf(np.linspace(-10, 10, 1000))
model_psf = model.psf(np.linspace(-10, 10, 1000))

model.reg

model.weights_vec

model._weights_vec_id
