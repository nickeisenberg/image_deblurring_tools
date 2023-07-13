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

    def add(self, count, reg_class, reg_name):
        """
        reg_type: erf, arctan, bennett, custom
        """
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

model = Model()

model.add(3,  Bennett, 'bennett',)
model.add(2, Erf, 'erf')
model.add(10, Arctan, 'arctan')

model.initialize()

model_esf = model.esf(np.linspace(-10, 10, 1000))
model_psf = model.psf(np.linspace(-10, 10, 1000))

plt.plot(model_psf)
plt.plot(model_esf)
plt.show()
