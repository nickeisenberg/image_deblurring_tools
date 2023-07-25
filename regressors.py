import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as _erf
from copy import deepcopy
from scipy.optimize import minimize
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
        self.weights_vec_id = np.array([])

    def add(self, count, reg_class, reg_name):
        self.reg[reg_name] = reg_class
        try:
            self.reg_counts[reg_name] += count
        except:
            self.reg_counts[reg_name] = count

    def initialize(self, weights=None):
        
        if weights == None:
            ids = []
            for rn, count in self.reg_counts.items():
                ids.append(np.repeat(rn, count))
            self.weights_vec_id = np.hstack(ids)
            total = np.array([*self.reg_counts.values()]).sum()
            for rt, count in self.reg_counts.items():
                self.weights[rt] = np.ones((count, 2)) * [1 / total, 1]
                self.weights_vec = np.vstack(
                    [*self.weights.values()]
                ).reshape(-1)

        else:
            self.weights = weights
            ids = []
            ws = []
            for rn, w in self.weights.items():
                ids.append(np.repeat(rn, self.reg_counts[rn]))
                ws.append(w)
            self.weights_vec_id = np.hstack(ids)
            self.weights_vec = np.array(ws).reshape(-1)

        return None

    def add_initialize(self, count, weights, reg_class, reg_name):

        self.reg[reg_name] = reg_class

        try:
            self.reg_counts[reg_name] += count
            self.weights[reg_name] = np.vstack(
                (self.weights[reg_name], weights)
            )
        except:
            self.reg_counts[reg_name] = count
            self.weights[reg_name] = weights

        try:
            self.weights_vec = np.hstack(
                (self.weights_vec, weights.reshape(-1))
            )
            self.weights_vec_id = np.hstack(
                (self.weights_vec_id, np.repeat(reg_name, count))
            )
        except:
            self.weights_vec = weights.reshape(-1)
            self.weights_vec_id = np.repeat(reg_name, count)

    def esf(self, x):
        esf_ = 0 
        for w, rn in zip(
            self.weights_vec.reshape((-1, 2)), self.weights_vec_id
        ):
            if self.reg_counts[rn] == 0:
                continue
            esf_ += self.reg[rn](*list(w)).esf(x)
        return esf_

    def psf(self, r):
        psf_ = 0 
        for w, rn in zip(
            self.weights_vec.reshape((-1, 2)), self.weights_vec_id
        ):
            if self.reg_counts[rt] == 0:
                continue
            psf_ += self.reg[rn](*list(w)).psf(r)
        return psf_

    def fit_to_esf(self, data, domain=None):
        if domain is None:
            domain = np.linspace(
                -int(data.size / 2), int(data.size / 2), data.size
            )

        def _loss(P, data=data, domain=domain, ids=self.weights_vec_id):
            residual = np.zeros(domain.size)
            for w, rn in zip(P.reshape((-1, 2)), ids):
                residual += self.reg[rn](*list(w)).esf(domain)
            residual = np.abs(residual - data)
            return np.sum(residual)

        self.weights_vec = minimize(_loss, self.weights_vec).x
        return None
