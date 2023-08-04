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
# there are two ways to do this.
# both ways are slower that scipy optimize minimze but the second method 
# gives more freedom and allows the regressor coefficients to not be confined
# to a singal flattened array.

#1) similare set uop with scipy optimize minimize with P a flat array
def tf_reg(P, x=tf.Variable(domain)):
    return P[0] * tfm.atan(P[1] * (x - P[2]))
def tf_loss(P):
    return tf.reduce_mean(tf.abs(data - tf_reg(P)))

P = tf.Variable(initial_value=tf.constant(np.array(guess).astype(np.float64)))

now = time.time()
lr = .05
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

#2) In this approach, we do not need to have the regressor so in a single
# flattened array.
def tf_reg(p0, p1, p2, x=tf.Variable(domain)):
    return p0 * tfm.atan(p1 * (x - p2))
def tf_loss(p0, p1, p2):
    return tf.reduce_mean(tf.abs(data - tf_reg(p0, p1, p2)))

p0 = tf.Variable(initial_value=tf.constant(np.array(guess[0]).astype(np.float64)))
p1 = tf.Variable(initial_value=tf.constant(np.array(guess[1]).astype(np.float64)))
p2 = tf.Variable(initial_value=tf.constant(np.array(guess[2]).astype(np.float64)))

now = time.time()
lr = .1
for i in range(2000):
    with tf.GradientTape() as tape:
        _loss = tf_loss(p0, p1, p2)
    grads = tape.gradient(_loss, [p0, p1, p2])
    _ = p0.assign_sub(grads[0] * lr)
    _ = p1.assign_sub(grads[1] * lr)
    _ = p2.assign_sub(grads[2] * lr)
    if _loss < loss(opt) + .0001:
        print(i)
        break
tensorflow = time.time() - now

plt.plot(data)
plt.plot(tf_reg(p0, p1, p2))
plt.show()

print(scipy)
print(tensorflow)
