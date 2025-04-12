import numpy as np
from scipy.stats import norm
from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

def analytical_european_options(S0, X, T, r, sigma, call_or_put="call"):

    d1 = (np.log(S0 / X + 1e-10) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if call_or_put == "call":
        price = S0 * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif call_or_put == "put":
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price

def explicit_european_options(S_max, X, T, r, sigma, M, N, call_or_put="call"):

    dt = T / N
    S = np.linspace(0, S_max, M+1)

    stability_condition = dt - 1 / (sigma ** 2 * M ** 2)

    if stability_condition >= 0:
        print("Stability condition is not satisfied.")

    if call_or_put == "call":
        V = np.maximum(S - X, 0)
    elif call_or_put == "put":
        V = np.maximum(X - S, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    i = np.arange(1, M)
    a = 0.5 * dt * (sigma**2 * i**2 - r * i)
    b = 1 - dt * (sigma**2 * i**2 + r)
    c = 0.5 * dt * (sigma**2 * i**2 + r * i)

    for n in range(1, N+1):
        V_old = V.copy()
        V[1:M] = a * V_old[:M-1] + b* V_old[1:M] + c* V_old[2:]

        if call_or_put == "call":
            V[0] = 0
            V[M] = S_max - X * np.exp(-r * n * dt)
        elif call_or_put == "put":
            V[0] = X * np.exp(-r * n * dt)
            V[M] = 0
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
    return V

def implicit_european_options(S_max, X, T, r, sigma, M, N, call_or_put="call"):

    dt = T / N
    S = np.linspace(0, S_max, M+1, endpoint=True)

    if call_or_put == "call":
        V = np.maximum(S - X, 0)
    elif call_or_put == "put":
        V = np.maximum(X - S, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    i = np.arange(1, M)
    alpha = -0.5 * dt * (sigma**2 * i**2 - r * i)
    beta = 1 + dt * (sigma**2 * i**2 + r)
    gamma = -0.5 * dt * (sigma**2 * i**2 + r * i)

    A = lil_matrix((M-1, M-1))

    for i in range(M-1):
        A[i, i] = beta[i]

    for i in range(M-2):
        A[i, i+1] = gamma[i]
        A[i+1, i] = alpha[i + 1]

    for n in range(1, N+1):

        rhs = V[1:M]
        if call_or_put == "call":
            rhs[0] -= alpha[0] * 0
            rhs[-1] -= gamma[-1] * (S_max - X * np.exp(-r * (n * dt)))
        elif call_or_put == "put":
            rhs[0] -= alpha[0] * X * np.exp(-r * (n * dt))
            rhs[-1] -= gamma[-1] * 0
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        V[1:M] = spsolve(csc_matrix(A), rhs)

        # Boundary values
        if call_or_put == "call":
            V[0] = 0
            V[M] = S_max - X * np.exp(-r * (n * dt))
        elif call_or_put == "put":
            V[0] = X * np.exp(-r * (n * dt))
            V[M] = 0
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
    return V

def crank_nicolson_european_options(S_max, X, T, r, sigma, M, N, call_or_put="call"):

    dt = T / N
    S = np.linspace(0, S_max, M + 1)

    if call_or_put == "call":
        V = np.maximum(S - X, 0)
    elif call_or_put == "put":
        V = np.maximum(X - S, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    i = np.arange(1, M)

    alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
    beta = -0.5 * dt * (sigma**2 * i**2 + r)
    gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)

    A = lil_matrix((M-1, M-1))
    B = lil_matrix((M-1, M-1))

    for i in range(M-1):
        A[i, i] = 1 - beta[i]
        B[i, i] = 1 + beta[i]

    for i in range(M-2):
        A[i, i+1] = -gamma[i]
        A[i+1, i] = -alpha[i + 1]

        B[i, i+1] = gamma[i]
        B[i+1, i] = alpha[i + 1]

    A = csc_matrix(A)
    B = csc_matrix(B)

    for n in range(1, N+1):
        rhs = B @ V[1:M]

        if call_or_put == "call":
            rhs[0] += alpha[1] * 0 + alpha[1] * 0
            rhs[-1] += gamma[-1] * (S_max - X * np.exp(-r * (n-1) * dt)) + gamma[-1] * (S_max - X * np.exp(-r * n * dt))
        elif call_or_put == "put":
            rhs[0] += alpha[0] * (X * np.exp(-r * (n-1) * dt)) + alpha[0] * (X * np.exp(-r * n * dt))
            rhs[-1] += gamma[-1] * 0 + gamma[-1] * 0

        V[1:M] = spsolve(A, rhs)

        if call_or_put == "call":
            V[0] = 0
            V[M] = S_max - X * np.exp(-r * n * dt)
        elif call_or_put == "put":
            V[0] = X * np.exp(-r * n * dt)
            V[M] = 0

    return V

def explicit_american_options(S_max, X, T, r, sigma, M, N, call_or_put="call"):

    dt = T / N
    S = np.linspace(0, S_max, M+1)

    stability_condition = dt - 1 / (sigma ** 2 * M ** 2)

    if stability_condition >= 0:
        print("Stability condition is not satisfied.")

    if call_or_put == "call":
        V = np.maximum(S - X, 0)
    elif call_or_put == "put":
        V = np.maximum(X - S, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    i = np.arange(1, M)
    a = 0.5 * dt * (sigma**2 * i**2 - r * i)
    b = 1 - dt * (sigma**2 * i**2 + r)
    c = 0.5 * dt * (sigma**2 * i**2 + r * i)

    for n in range(1, N+1):
        V_old = V.copy()
        V[1:M] = a * V_old[:M-1] + b * V_old[1:M] + c * V_old[2:]

        if call_or_put == "call":
            V[0] = 0
            V[M] = S_max - X * np.exp(-r * n * dt)
        elif call_or_put == "put":
            V[0] = X * np.exp(-r * n * dt)
            V[M] = 0
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
        # American option constraint (early exercise)
        if call_or_put == "call":
            V = np.maximum(V, S - X)
        elif call_or_put == "put":
            V = np.maximum(V, X - S)

    return V

def implicit_psor_american_options(S_max, X, T, r, sigma, M, N, call_or_put="call", omega=1.2, tol=1e-08, max_iter=10000):

    dt = T / N
    S = np.linspace(0, S_max, M+1)

    if call_or_put == "call":
        V = np.maximum(S - X, 0)
    elif call_or_put == "put":
        V = np.maximum(X - S, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    i = np.arange(1, M)
    alpha = -0.5 * dt * (sigma**2 * i**2 - r * i)
    beta = 1 + dt * (sigma**2 * i**2 + r)
    gamma = -0.5 * dt * (sigma**2 * i**2 + r * i)

    for n in range(1, N+1):
        payoff = X - S[1:M] if call_or_put == "put" else S[1:M] - X
        b = V[1:M].copy()

        if call_or_put == "call":
            b[-1] -= gamma[-1] * (S_max - X * np.exp(-r * n * dt))
        elif call_or_put == "put":
            b[0] -= alpha[0] * (X * np.exp(-r * n * dt))
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        V_old = V[1:M].copy()
        for it in range(max_iter):
            V_new = V_old.copy()
            for j in range(M-1):
                lhs = b[j]
                if j > 0:
                    lhs -= alpha[j] * V_new[j-1]
                if j < M-2:
                    lhs -= gamma[j] * V_old[j+1]
                V_new[j] = max(payoff[j], (1 - omega) * V_old[j] + omega / beta[j] * lhs)
            if np.linalg.norm(V_new - V_old, ord=np.inf) < tol:
                break
            V_old = V_new.copy()

        V[1:M] = V_new

        if call_or_put == "call":
            V[0] = 0
            V[M] = np.maximum(S_max - X * np.exp(-r * (n * dt)), S[M] - X)
        elif call_or_put == "put":
            V[0] = np.maximum(X * np.exp(-r * (n * dt)), X - S[0])
            V[M] = 0
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        
    return V

def solve_fd(method, S_max, X, T, r, sigma, M, N, call_or_put="call"):
    S = np.linspace(0, S_max, M+1)

    V_exact = analytical_european_options(S, X, T, r, sigma, call_or_put=call_or_put)

    if method == "explicit_european":
        V = explicit_european_options(S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
        return V, abs(V - V_exact)
    elif method == "implicit_european":
        V = implicit_european_options(S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
        return V, abs(V - V_exact)
    elif method == "crank_european":
        V = crank_nicolson_european_options(S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
        return V, abs(V - V_exact)
    elif method == "explicit_american":
        return explicit_american_options(S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
    elif method == "implicit_psor_american":
        return implicit_psor_american_options(S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Parameters
    S_max = 200
    X = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    call_or_put = "put"

    M = 100
    N = 1000

    S = np.linspace(0, S_max, M+1)

    V, _ = solve_fd("implicit_european", S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
    V_exact = analytical_european_options(S, X, T, r, sigma, call_or_put=call_or_put)

    plt.figure(figsize=(9, 5))
    plt.plot(S, V, label='FDM (Implicit), European', linewidth=2)
    plt.plot(S, V_exact, '--', label='Analytical, European', linewidth=2)
    plt.xlabel('Asset Price $S$')
    plt.ylabel('Option Value $V(S, 0)$')
    plt.title(f'European Option vs. European (B-S) — {call_or_put.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()