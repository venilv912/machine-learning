import numpy as np
import matplotlib.pyplot as plt

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