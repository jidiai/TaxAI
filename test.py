import numpy as np

# Parameters
tau = 0.2
xi = 0.5
tau_a = 0.1
xi_a = 0.5
gamma = 0.5
theta = 1
beta = 0.95
W = 1  # Wage rate
r = 0.04  # Interest rate
tau_s = 0.1
# Agent-specific values
a_i = 5
e_i = 0.5

# Calculate marginal tax rates
tilde_tau_i = 1 - (1 - tau) * (r * a_i + W * e_i) ** (-xi)
tilde_tau_a_i = 1 - (1 - tau_a) * a_i ** (-xi_a)

# Calculate labor supply h_i
h_i = ((1 - tilde_tau_i) / (1 + tau_s) * W * e_i) ** (1 / gamma)

# Time iteration method
tolerance = 0.01
max_iter = 1000

c_init = np.ones(max_iter)  # Initialize the consumption guess
c_iter = np.zeros(max_iter)

for i in range(max_iter - 1):
    # Compute the next consumption value using the Euler equation
    tilde_tau_next = 1 - (1 - tau) * (r * a_i + W * e_i) ** (-xi)
    tilde_tau_a_next = 1 - (1 - tau_a) * a_i ** (-xi_a)
    chi_next = 0  # You can replace this with an appropriate value or function for the borrowing constraint

    c_iter[i + 1] = ((c_init[i] ** (-theta)) /(beta * (1 - tilde_tau_a_next + (1 - tilde_tau_next) * r + chi_next))) ** (-1 / theta)

    # Check for convergence
    if np.abs(c_iter[i + 1] - c_init[i]) < tolerance:
        break

    # Update the consumption guess
    c_init[i + 1] = c_iter[i + 1]

# The converged consumption value
c_i = c_iter[i]

print("Agent's consumption (c_i):", c_i)
print("Agent's labor supply (h_i):", h_i)
