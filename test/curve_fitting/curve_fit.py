import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import minimize
import tensorflow as tf
from copy import deepcopy

#-scipy as a benchmark test------------------------

domain = np.linspace(-10, 10, 1000)
data = 1.2 * np.arctan(2 * (domain - 3))
data += np.random.normal(0, .2, domain.size)
def reg(P, x=domain):
    return P[0] * np.arctan(P[1] * (x - P[2]))
def loss(P):
    return np.mean(np.abs(reg(P) - data) ** 2)
guess = [1, 1, 0]
now = time.time()
opt = minimize(loss, guess).x
scipy_time = time.time() - now

scipy_final_loss = loss(opt)

scipy_fit = deepcopy(reg(opt, domain)) 

plt.plot(data)
plt.plot(scipy_fit)
plt.show()

#--------------------------------------------------

domain = torch.tensor(np.linspace(-10, 10, 1000))
data = 1.2 * torch.arctan(2 * (domain - 3)) 
data += torch.tensor(np.random.normal(0, .2, domain.size()[0]))

def Arctan(P, x=domain):
    return P[0] * torch.arctan(P[1] * (x - P[2]))

class CurveFit:
    def __init__(self, data, domain, regressor, lr=.01):
        self.data = data
        self.domain = domain
        self.regressor = regressor
        self.P = torch.tensor([1., 1., 0.], requires_grad=True)
        self.lr = lr

    def loss(self):
        guess = self.regressor(self.P, self.domain)
        error = torch.mean(
            (guess - self.data) ** 2
        )
        return error

    def update(self):
        _loss = self.loss()
        _loss.backward()
        with torch.no_grad():
            self.P -= self.P.grad * self.lr
            _ = self.P.grad.zero_()

fit = CurveFit(data, domain, Arctan, lr=.05) 

now = time.time()
for i in range(5000):
    _ = fit.update()
    if fit.loss() <= scipy_final_loss * 1:
        print('done')
        break
torch_time = time.time() - now

torch_loss = fit.loss()

torch_fit = deepcopy(Arctan(fit.P.detach().numpy(), domain))

#--------------------------------------------------

domain = tf.constant(np.linspace(-10, 10, 1000), dtype=tf.float64)
data = 1.2 * tf.math.atan(2 * (domain - 3)) 
data += tf.constant(np.random.normal(0, .2, domain.shape[0]), dtype=tf.float64)

def Arctan(P, x=domain):
    return P[0] * tf.math.atan(P[1] * (x - P[2]))

class CurveFit:
    def __init__(self, data, domain, regressor, lr=.01):
        self.data = data
        self.domain = domain
        self.regressor = regressor
        self.P = tf.Variable([1., 1., 0.], dtype=tf.float64)
        self.lr = lr

    def loss(self, P):
        guess = self.regressor(P, self.domain)
        error = tf.reduce_mean(
            (guess - self.data) ** 2
        )
        return error

    def update(self):
        with tf.GradientTape() as tape:
            _loss = self.loss(self.P)
        grads = tape.gradient(_loss, self.P)
        _ = self.P.assign_sub(grads * self.lr)

fit = CurveFit(data, domain, Arctan, lr=.05)

now = time.time()
for i in range(5000):
    _ = fit.update()
    if fit.loss(fit.P) <= scipy_final_loss * 1:
        print('done')
        break
tf_time = time.time() - now

tf_loss = fit.loss(fit.P)

tf_fit = deepcopy(Arctan(fit.P, domain))

#-final results------------------------------------
print(f'scipy time: {scipy_time}')
print(f'torch time: {torch_time}')
print(f'tf time: {tf_time}')
print(f'scipy loss: {scipy_final_loss}')
print(f'torch loss: {torch_loss}')
print(f'tf loss: {tf_loss}')

plt.plot(data, label='data')
plt.plot(scipy_fit, label='scipy')
plt.plot(torch_fit, label='torch')
plt.plot(tf_fit, label='tensorflow')
plt.legend()
plt.show()
