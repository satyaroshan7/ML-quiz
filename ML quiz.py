#!/usr/bin/env python
# coding: utf-8

# Q1

# When a (n, k) matrix A and a (k, m) matrix B are multiplied matrix-wise, the resultant matrix C has (n, m) dimensions. Each element in the ith row of matrix A is multiplied by the corresponding element in the jth column of matrix B, and the values are added together to obtain the element at position (i, j) in matrix C.
# 1.)For each element in the resulting matrix C, you need to perform k multiplications (one for each element in the corresponding row of A and column of B).
# 2.)After obtaining the k products, you need to perform k-1 additions to sum them up.
# So, for each element in the resulting matrix C, you need k multiplications and (k-1) additions.
# Since matrix C has dimensions (n, m), there are n * m elements in total. Therefore, the total number of multiplications is n * m * k, and the total number of additions is n * m * (k-1).
# Number of multiplications: [n * m * k]
# Number of additions: [n * m * (k-1)]

# Q2

# Using Python Lists

# In[1]:


def matrix_multiply_python(A, B):
    n = len(A)
    k = len(A[0])
    m = len(B[0])

    # Initialize the result matrix C with zeros
    C = [[0] * m for _ in range(n)]

    # Perform matrix multiplication
    for i in range(n):
        for j in range(m):
            for l in range(k):
                C[i][j] += A[i][l] * B[l][j]

    return C


# Using NumPy

# In[2]:


import numpy as np

def matrix_multiply_numpy(A, B):
    return np.dot(A, B)


# Time Comparision

# In[5]:


import time

# Create sample matrices
n, k, m = 50, 250, 150
matrix_A = [[1] * k for _ in range(n)]
matrix_B = [[2] * m for _ in range(k)]

# Timing matrix multiplication using Python lists
start_time_python = time.time()
result_python = matrix_multiply_python(matrix_A, matrix_B)
time_python = time.time() - start_time_python

# Convert the matrices to NumPy arrays for NumPy multiplication
array_A = np.array(matrix_A)
array_B = np.array(matrix_B)

# Timing matrix multiplication using NumPy
start_time_numpy = time.time()
result_numpy = matrix_multiply_numpy(array_A, array_B)
time_numpy = time.time() - start_time_numpy

# Verify the correctness of the results
assert np.allclose(result_numpy, result_python)

# Print the timing results
print("Time (Python):", time_python)
print("Time (NumPy):", time_numpy)


#  the NumPy solution is faster than the Python list-based solution for matrix multiplication. This is because NumPy is implemented in C and optimized for numerical operations, resulting in efficient handling of matrices.   

# Q4
# 

# with respect to x
# 2xy + y^3cos(x)
# 
# with respect to y
# x^2 + 3y^2sin(x)

# Q5
# 

# In[36]:


import jax
import jax.numpy as jnp

def function_to_minimize(x):
    return x[0]**2 * x[1] + x[1]**3 * jnp.sin(x[0])

# Define the gradient function using JAX
gradient_function = jax.grad(function_to_minimize)

# Choose random values for x and y
x_value = 2.0
y_value = 3.0

# Compute the analytical gradient
analytical_gradient = jnp.array([
    2 * x_value * y_value + y_value**3 * jnp.cos(x_value),
    x_value**2 + 3 * y_value**2 * jnp.sin(x_value)
])

# Compute the numerical gradient using JAX
numerical_gradient = gradient_function(jnp.array([x_value, y_value]))

# Compare the analytical and numerical gradients
print("Analytical Gradient:", analytical_gradient)
print("Numerical Gradient:", numerical_gradient)

# Check if the gradients match within a small tolerance
assert jnp.allclose(analytical_gradient, numerical_gradient, atol=1e-6)


# Q6

# In[19]:


from sympy import symbols, diff, sin

# Define symbols
x, y = symbols('x y')

# Define the function
f = x**2 * y + y**3 * sin(x)

# Compute the partial derivatives
df_dx = diff(f, x)
df_dy = diff(f, y)

# Display the partial derivatives
print("Partial derivative with respect to x:")
print(df_dx)

print("\nPartial derivative with respect to y:")
print(df_dy)


# Q9
# 
# 

# y=x

# In[20]:


import matplotlib.pyplot as plt
import numpy as np

# Define the domain
x_values = np.arange(0.5, 100.5, 0.5)

# Define the function y = x
y_values = x_values

# Plot the function
plt.plot(x_values, y_values, label='y = x')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = x')

# Add grid
plt.grid(True)

# Show the legend
plt.legend()

# Show the plot
plt.show()


# y=x^2

# In[22]:


import matplotlib.pyplot as plt
import numpy as np

# Define the function
def squared_function(x):
    return x**2

# Generate x values from 0.5 to 100.0 in steps of 0.5
x_values = np.arange(0.5, 100.5, 0.5)

# Calculate corresponding y values using the function
y_values = squared_function(x_values)

# Plot the function
plt.plot(x_values, y_values, label=r'y = x^2')
plt.title('Plot of y = x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


# y=x^3/100

# In[23]:


import matplotlib.pyplot as plt
import numpy as np

# Define the function
def my_function(x):
    return (x**3) / 100

# Generate x values from 0.5 to 100.0 in steps of 0.5
x_values = np.arange(0.5, 100.1, 0.5)

# Compute y values using the function
y_values = my_function(x_values)

# Plot the function
plt.plot(x_values, y_values, label=r'y = \frac{x^3}{100}')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = \\frac{x^3}{100}')

# Display the legend
plt.legend()

# Show the plot
plt.show()


# y=sin(x)

# In[24]:


import numpy as np
import matplotlib.pyplot as plt

# Define the domain
x_values = np.arange(0.5, 100.1, 0.5)

# Compute the corresponding y values
y_values = np.sin(x_values)

# Plot the function
plt.plot(x_values, y_values, label='y = sin(x)')

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of y = sin(x)')

# Show the legend
plt.legend()

# Show the plot
plt.show()


# y=sinx/x

# In[25]:


import matplotlib.pyplot as plt
import numpy as np

# Define the function
def func(x):
    return np.sin(x) / x

# Define the domain
x_values = np.arange(0.5, 100.5, 0.5)

# Calculate the corresponding y values
y_values = func(x_values)

# Plot the function
plt.plot(x_values, y_values, label=r'\frac{\sin(x)}{x}')
plt.title('Plot of y = \\frac{\\sin(x)}{x}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()


# y=log(x)

# In[27]:


import numpy as np
import matplotlib.pyplot as plt

# Define the function
def logarithm(x):
    return np.log(x)

# Generate x values from 0.5 to 100.0 in steps of 0.5
x_values = np.arange(0.5, 100.5, 0.5)

# Calculate corresponding y values using the function
y_values = logarithm(x_values)

# Plot the function
plt.plot(x_values, y_values, label='y = log(x)')
plt.title('Plot of y = log(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.show()


# y=e^x

# In[29]:


import numpy as np
import matplotlib.pyplot as plt

# Define the domain
x_values = np.arange(0.5, 100.5, 0.5)

# Calculate the corresponding y values
y_values = np.exp(x_values)

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label=r'y = e^x')

# Labeling the plot
plt.title('Plot of y = e^x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# Show the plot
plt.show()


# Q10

# In[30]:


import numpy as np
import pandas as pd

# Generate a matrix of size 20x5 with random numbers between 1 and 2
matrix = np.random.uniform(1, 2, size=(20, 5))

# Create a Pandas DataFrame with column names "a", "b", "c", "d", "e"
df = pd.DataFrame(matrix, columns=['a', 'b', 'c', 'd', 'e'])

# Display the DataFrame
print("DataFrame:")
print(df)

# Find the column with the highest standard deviation
max_std_column = df.std().idxmax()

# Find the row with the lowest mean
min_mean_row = df.mean(axis=1).idxmin()

# Display the results
print("\nColumn with highest standard deviation:", max_std_column)
print("Row with lowest mean:", min_mean_row)


# Q12

# In[32]:


import numpy as np

# Create a 2x3 array
arr1 = np.array([[1, 2, 3],
                 [4, 5, 6]])

# Create a 1x3 array
arr2 = np.array([7, 8, 9])

# Perform element-wise addition using broadcasting
result = arr1 + arr2

# Print the original arrays and the result
print("Array 1:")
print(arr1)

print("\nArray 2:")
print(arr2)

print("\nResult after broadcasting:")
print(result)


# Q13

# In[35]:


import numpy as np

def custom_argmin(arr):
    min_index = 0
    min_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] < min_value:
            min_value = arr[i]
            min_index = i

    return min_index

# Test the custom function
array_to_test = np.array([5, 3, 9, 2, 6, 4])
custom_result = custom_argmin(array_to_test)

# Verify with np.argmin
np_result = np.argmin(array_to_test)

# Print the results
print("Custom argmin result:", custom_result)
print("NumPy argmin result:", np_result)

# Verify that the results match
assert custom_result == np_result


# In[ ]:




