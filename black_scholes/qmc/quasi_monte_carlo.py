import numpy as np
from scipy.stats import norm


def midpoint_rule_qmc(N, S0, X, T, r, sigma, which="call"):

    U = (np.arange(N) + 0.5) / N
    Z = norm.ppf(U)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if which == "call":
        discounted_payoffs = np.exp(-r * T) * np.maximum(ST - X, 0)
    elif which == "put":
        discounted_payoffs = np.exp(-r * T) * np.maximum(X - ST, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    mc_price = np.mean(discounted_payoffs)

    return mc_price
