import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters
S_max = 200
K = 100
T = 1.0
r = 0.05
sigma = 0.1

M = 2000
N = 1000000

# Discretization
dS = S_max / M
dt = T / N
S = np.linspace(0, S_max, M+1)
V = np.maximum(S - K, 0)

# Coefficients for explicit scheme
j = np.arange(1, M)
a = 0.5 * dt * (sigma**2 * j**2 - r * j)
b = 1 - dt * (sigma**2 * j**2 + r)
c = 0.5 * dt * (sigma**2 * j**2 + r * j)

# Time stepping
for n in range(N):
    V_old = V.copy()
    V[1:M] = a * V_old[0:M-1] + b * V_old[1:M] + c * V_old[2:M+1]
    V[0] = 0
    V[M] = S_max - K * np.exp(-r * (T - n * dt))

# Analytical B-S formula for European call
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K + 1e-10) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Compute analytical solution at t = 0
V_exact = black_scholes_call(S, K, T, r, sigma)

print(V)
print(V_exact)

# Plot comparison
plt.figure(figsize=(9, 5))
plt.plot(S, V, label='FDM (Explicit)', linewidth=2)
plt.plot(S, V_exact, '--', label='Analytical Black-Scholes', linewidth=2)
plt.xlabel('Asset Price $S$')
plt.ylabel('Option Value $V(S, 0)$')
plt.title('FDM vs. Analytical Black-Scholes (European Call)')
plt.legend()
plt.grid(True)
plt.show()