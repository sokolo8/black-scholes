from black_scholes.fdm import solve_fd
from black_scholes import analytical_european_options
import matplotlib.pyplot as plt
import numpy as np


# Parameters
S_max = 200
X = 100
T = 1.0
r = 0.05
sigma = 0.2
call_or_put = "call"

M = 100
N = 1000

S = np.linspace(0, S_max, M+1)

V_ex, error_ex = solve_fd("explicit_european", S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
V_im, error_im = solve_fd("implicit_european", S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
V_cn, error_cn = solve_fd("crank_european", S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
V_exact = analytical_european_options(S, X, T, r, sigma, call_or_put=call_or_put)

plt.figure(figsize=(9, 5))
plt.plot(S, V_ex, label='Explicit, European', linewidth=2)
plt.plot(S, V_im, label='Implicit, European', linewidth=2)
plt.plot(S, V_cn, label='Crank-Nicolson, European', linewidth=2)
plt.plot(S, V_exact, '--', label='Analytical, European', linewidth=2)
plt.xlabel('Asset Price $S$')
plt.ylabel('Option Value $V(S, 0)$')
plt.title(f'European Option vs. European (B-S) — {call_or_put.capitalize()}')
plt.legend()
plt.grid(True)
plt.show()