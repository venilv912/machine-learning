import numpy as np
import matplotlib.pyplot as plt

# setting training examples
x = np.array([1.0, 2.0])        # x is input variable (in 1000 sq. feet)
y = np.array([300.0, 500.0])    # y is target variable (in 1000s of dollars)
print(f"x = {x}")
print(f"y = {y}")
m = x.shape[0]
print(f"Here we have taken number of training examples {m}.\n\n")


# Plotting these data points
plt.scatter(x, y, marker='x', c='r')
plt.title("Prices of Houses")
plt.ylabel('Price (in 1000s of dollars)')
plt.xlabel('Size (1000 sq. ft)')
plt.show()


# adjusting model parameters
w = 200
b = 100
print(f"w = {w}")
print(f"b = {b}\n\n")


# defining model function
def model_function(x, w, b):
    m = x.shape[0]
    f = np.zeros(m)
    for i in range(m):
        f[i] = w*x[i] + b
    
    return f


# Plottin graph by value of model function
temp_f = model_function(x, w, b)

plt.plot(x, temp_f, c = 'b', label = 'our prediction')
plt.scatter(x, y, marker='x', c='r',label='Actual Values')
plt.title('Price of House')
plt.xlabel('Size (in 1000 sq. ft)')
plt.ylabel('Price (in 1000s of dollars)')
plt.legend()
plt.show()


# Predicting the price
x_p = 1.2
cost_p = w*x_p + b

print(f"Cost = ${cost_p:.0f}")