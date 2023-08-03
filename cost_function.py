import numpy as np
import matplotlib.pyplot as plt


# training examples
x = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y = np.array([250, 300, 480, 430, 630, 730])


# defining cost function
def cost_function(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f = w*x[i] + b
        error = (f - y[i])**2
        cost = cost + error
    cost = (1/(2*m)) * cost

    return cost


# plotting given data points
plt.scatter(x, y, marker = '*', c = 'r')
plt.title("Prices of Houses")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sq. ft)')
plt.show()


# setting model parameters
w = 200
b = 10


# defining model function
def model_function(x, w, b):
    m = x.shape[0]
    f = np.zeros(m)
    for i in range(m):
        f[i] = w*x[i] + b
    
    return f


# assigning function values
temp_f = model_function(x, w, b)

cost_f = cost_function(x, y, w, b)


# plotting model funcion graph on basis of some cost function
plt.plot(x, temp_f, c = 'b', label = 'our prediction')
plt.scatter(x, y, marker='x', c='r',label='Actual Values')
plt.title(f'Plot for cost function = {cost_f}')
plt.xlabel('Size (in 1000 sq. ft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()


# We have to minimize the cost function