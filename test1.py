import numpy as np

# Parameters
gamma = 2
theta = 1
beta = 0.95
tau = 0.2
tau_s = 0.1
tau_a = 0.1
xi = 0.5
xi_a = 0.5
T = 10  # Number of periods
W = 1   # Wage rate
r = 0.04  # Interest rate

# Grids
a_grid = np.linspace(0, 10, 100)
e_grid = np.linspace(0, 1, 100)

# Utility function
def utility(c, h):
    return np.log(c) - h**(1+gamma) / (1+gamma)


# Budget function
def budget(c, h, W, r, a, e, tau, tau_s, tau_a, xi, xi_a):
    tilde_tau = 1 - (1 - tau) * (r * a + W * e * h) ** (-xi)
    tilde_tau_a = 1 - (1 - tau_a) * a ** (-xi_a)
    return (1 - tilde_tau) * W * e * h + (1 - tilde_tau_a) * r * a - (1 + tau_s) * c

# Value function
def value_function(W, r, a, e, V_next):
    c_grid = np.linspace(1e-6, 10, 100)
    h_grid = np.linspace(1e-6, 1, 100)
    V = np.zeros((len(c_grid), len(h_grid)))
    policy_c = np.zeros_like(V)
    policy_h = np.zeros_like(V)

    for i, c in enumerate(c_grid):
        for j, h in enumerate(h_grid):
            if budget(c, h, W, r, a, e, tau, tau_s, tau_a, xi, xi_a) >= 0:
                V[i, j] = utility(c, h) + beta * V_next
                policy_c[i, j] = c
                policy_h[i, j] = h
            else:
                V[i, j] = -1e10

    max_idx = np.unravel_index(np.argmax(V, axis=None), V.shape)
    return V[max_idx], policy_c[max_idx], policy_h[max_idx]

# Dynamic programming loop
V = np.zeros((T, len(a_grid), len(e_grid)))
policy_c = np.zeros_like(V)
policy_h = np.zeros_like(V)

for t in range(T - 1, -1, -1):
    for i, a in enumerate(a_grid):
        for j, e in enumerate(e_grid):
            if t == T - 1:
                V_next = 0
            else:
                V_next = V[t + 1, i, j]
            V[t, i, j], policy_c[t, i, j], policy_h[t, i, j] = value_function(W, r, a, e, V_next)

print("Optimal policy functions:")
print("Consumption (c):", policy_c)
print("Labor supply (h):", policy_h)
