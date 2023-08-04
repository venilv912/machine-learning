import numpy as np
import matplotlib.pyplot as plt
import math

# setting training examples
x = np.array([1.0, 2.0])
y = np.array([300.0, 500.0])


# defining cost function
def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0.
    for i in range(m):
        f = w*x[i] + b
        error = (f - y[i])**2
        cost = cost + error
    cost = (1/(2*m)) * cost

    return cost


# computing gradients
def gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0.
    dj_db = 0.
    cost = 0.
    for i in range(m):
        f = w*x[i] + b
        error = (f - y[i])
        dj_db = dj_db + error
        dj_dw = dj_dw + (error * x[i])
    dj_dw = (1/m)*dj_dw
    dj_db = (1/m)*dj_db

    return dj_dw, dj_db


# defining gradient descent
def gradient_descent(x, y, w_in, b_in, alpha, n_iters, cost_function, gradient):
    J_hist = []     # this array stores past values of cost function
    p_hist = []     # this array stores past values of w & b
    w = w_in
    b = b_in

    for i in range (n_iters):
        # calculating gradient
        dj_dw, dj_db = gradient(x, y, w, b)

        # updating parameters
        w = w - alpha*dj_dw
        b = b - alpha*dj_dw

        # saving past values of J
        if i < 100000:
            J_hist.append(cost_function(x, y, w, b))
            p_hist.append([w, b])

        # print cost 10 times in whole duration with equal time intervals/
        if i%math.ceil(n_iters/10) == 0:
            print(f"Iterations {i:4}: Cost {J_hist[-1]:0.2e} ", f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e} ", f"w: {w: 0.3e}, b: {b: 0.5e}")

    return w, b, J_hist, p_hist


# Initializing parameters
w_i = 0
b_i = 0

iters = 10000
alpha = 1.0e-2

w_final, b_final, J_hist, p_hist = gradient_descent(x, y, w_i, b_i, alpha, iters, cost_function, gradient)

print(f"(w,b) by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")