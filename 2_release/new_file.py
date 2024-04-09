import numpy as np
import matplotlib.pyplot as plt

# Function for the exponential form: y = c * e^(dx)
def exponential_function(x, c, d):
    return c * np.exp(d * x)

# Function for the power-law form: y = a * x^d
def power_law_function(x, a, d):
    return a * x**d

def exponential_powerLaw_equiv(x, a, d):
    return a * np.exp(d * np.log(x))

def power_law_function_equiv(x, a, d):
    return a * x**(d * (x/np.log(x)))

# Generate x values
x_values = np.linspace(0.001, 2, 100)  # Avoiding x=0 for the logarithmic scaling

# Calculate corresponding y values
A, alpha = 30, -0.5

y_power = power_law_function(x_values, A, alpha)
y_exp = exponential_powerLaw_equiv(x_values, A, alpha)

factor = 2
y_exp_2 = exponential_function(x_values, factor * A, alpha)
y_power_2 = power_law_function_equiv(x_values, factor * A, alpha)


# Plotting the exponential and power-law functions
plt.figure(figsize=(10, 6))

plt.plot(x_values, y_exp, label=r'$y = c \cdot e^{dx}$')
plt.plot(x_values, y_power, label=r'$y = a \cdot x^d$', linestyle='dashed')
plt.plot(x_values, y_exp_2, label=r'$y = c \cdot e^{dx}$')
plt.plot(x_values, y_power_2, label=r'$y = a \cdot x^d$', linestyle='dashed')

plt.title('Exponential and Power-Law Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()