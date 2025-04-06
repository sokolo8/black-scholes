import numpy as np
from scipy.stats import norm

# Analytical Black-Scholes formula for European Call or Put option

def black_scholes_european_options(S0, X, T, r, sigma, which="call"):

    d1 = (np.log(S0 / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if which == "call":
        price = S0 * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif which == "put":
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return price

# Standard Monte Carlo Technique Start from random sampling from N(0, 1)

def standard_mc(N, S0, X, T, r, sigma, which="call"):

    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    if which == "call":
        discounted_payoffs = np.exp(-r * T) * np.maximum(ST - X, 0)
    elif which == "put":
        discounted_payoffs = np.exp(-r * T) * np.maximum(X - ST, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    mc_price = np.mean(discounted_payoffs)

    sample_std_dev = np.std(discounted_payoffs, ddof=1)
    standard_error = sample_std_dev / np.sqrt(N)
    standard_error_95 = 1.96 * standard_error # Calculate standard error at 95% confidence level

    return mc_price, standard_error_95

# Antithetic variates variance reduction

def antithetic_variates_mc(N, S0, X, T, r, sigma, which="call"):
    Z = np.random.randn(N // 2)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    ST_anti = S0 * np.exp((r - 0.5 * sigma**2) * T - sigma * np.sqrt(T) * Z)

    if which == "call":
        discounted_payoffs = np.exp(-r * T) * 0.5 * (np.maximum(ST - X, 0) + np.maximum(ST_anti - X, 0))
    elif which == "put":
        discounted_payoffs = np.exp(-r * T) * 0.5 * (np.maximum(X - ST, 0) + np.maximum(X - ST_anti, 0))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    mc_price = np.mean(discounted_payoffs)

    sample_std_dev = np.std(discounted_payoffs, ddof=1)
    standard_error = sample_std_dev / np.sqrt(N // 2)
    standard_error_95 = 1.96 * standard_error # Calculate standard error at 95% confidence level

    return mc_price, standard_error_95

# Control variates variance reduction

def control_variates_mc(N, S0, X, T, r, sigma, which="call"):

    Z = np.random.randn(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Step 2: Calculate call option payoffs (X) and control variable (Y = S_T)

    if which == "call":
        payoffs = np.maximum(ST - X, 0)
    elif which == "put":
        payoffs = np.maximum(X - ST, 0)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    Y = ST
    Y_mean = S0 * np.exp(r * T)  # known expected value of S_T under risk-neutral measure

    # Step 3: Estimate lambda*
    cov = np.cov(payoffs, Y, ddof=1)[0, 1]
    var_Y = np.var(Y, ddof=1)
    lambda_star = cov / var_Y

    # Step 4: Apply control variate adjustment
    payoffs_adj = payoffs + lambda_star * (Y_mean - Y)
    discounted_payoffs_adj = np.exp(-r * T) * payoffs_adj

    # Step 5: Estimate option price using the adjusted estimator
    mc_price = np.mean(discounted_payoffs_adj)

    standard_error = np.std(discounted_payoffs_adj, ddof=1) / np.sqrt(N)
    standard_error_95 = 1.96 * standard_error # Calculate standard error at 95% confidence level

    return mc_price, standard_error_95

# Stratified sampling method

def stratified_sampling_mc(N, S0, X, T, r, sigma, which="call", M=1000):

    L = N // M  # Number of samples per stratum
    ave = 0.0
    var = 0.0

    for m in range(1, M + 1):
        # Stratified uniform samples in [(m-1)/M, m/M]
        U = (m - 1 + np.random.rand(L)) / M
        Z = norm.ppf(U)

        # Simulate asset prices
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

        if which == "call":
            discounted_payoffs = np.exp(-r * T) * np.maximum(ST - X, 0)
        elif which == "put":
            discounted_payoffs = np.exp(-r * T) * np.maximum(X - ST, 0)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        ave1 = np.mean(discounted_payoffs)
        #var1 = np.var(discounted_payoffs, ddof=1)  # Sample variance
        var1 = np.var(discounted_payoffs, ddof=1) / L

        # Aggregate the results
        ave += ave1 / M
        var += var1 / M**2

    mc_price = ave
    standard_error_95 = 1.96 * np.sqrt(var) # Calculate standard error at 95% confidence level

    return mc_price, standard_error_95


if __name__ == '__main__':

    S0 = 110      # Initial stock price
    X = 100       # eXercise price
    T = 1.0       # Time to maturity (in years)
    r = 0.05      # Risk-free interest rate
    sigma = 0.2   # Volatility
    call_or_put = "call"
    N = 10 ** 5

    print(standard_mc(N, S0, X, T, r, sigma, which="call"))