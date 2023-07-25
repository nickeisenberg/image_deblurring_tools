import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.math as tfm
import time
import numba

domain = np.linspace(-10, 10, 1000)
data = 1.2 * np.arctan(2 * (domain - 3)) + np.random.normal(0, .2, domain.size)

#-Usings scipy-------------------------------------
def reg(P, x=domain):
    return P[0] * np.arctan(P[1] * (x - P[2]))

def loss(P):
    return np.mean(np.abs(reg(P) - data))

guess = [1, 1, 0]

now = time.time()
opt = minimize(loss, guess).x
scipy = time.time() - now

plt.plot(data)
plt.plot(reg(opt, domain))
plt.show()

#-Using tensorflow---------------------------------
def tf_reg(P, x=tf.Variable(domain)):
    return P[0] * tfm.atan(P[1] * (x - P[2]))

def tf_loss(P):
    return tf.reduce_mean(tf.abs(data - tf_reg(P)))

P = tf.Variable(initial_value=tf.constant(np.array(guess).astype(np.float64)))

now = time.time()
lr = .1
for i in range(2000):
    with tf.GradientTape() as tape:
        _loss = tf_loss(P)
    grads = tape.gradient(_loss, P)
    _ = P.assign_sub(grads * lr)
    if _loss < loss(opt) + .0001:
        print(i)
        break
tensorflow = time.time() - now

plt.plot(data)
plt.plot(tf_reg(P))
plt.show()

print(scipy)
print(tensorflow)
