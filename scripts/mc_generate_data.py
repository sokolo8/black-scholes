import numpy as np
import pandas as pd
from black_scholes import analytical_european_options
from black_scholes.mc import standard_mc, antithetic_variates_mc, control_variates_mc, stratified_sampling_mc
from black_scholes.qmc import midpoint_rule_qmc

# Define parameters

S0 = 100      # Initial stock price
X = 100       # eXercise price
T = 1.0       # Time to maturity (in years)
r = 0.05      # Risk-free interest rate
sigma = 0.2   # Volatility
call_or_put = "call"
N_values = [int(1.1 ** i) for i in range(90, 190)]    # Number of Monte Carlo simulations

bs_price = analytical_european_options(S0, X, T, r, sigma, call_or_put=call_or_put)

results = {
    "N": [], 
    "std_mc_price": [], 
    "antith_mc_price": [], 
    "control_mc_price": [], 
    "strat_mc_price": [], 
    "midpoint_rule_qmc_price": [], 
    "std_mc_err": [], 
    "antith_mc_err": [], 
    "control_mc_err": [], 
    "strat_mc_err": [], 
    "std_mc_abs_err": [], 
    "antith_mc_abs_err": [], 
    "control_mc_abs_err": [], 
    "strat_mc_abs_err": [], 
    "midpoint_rule_qmc_abs_err": []
}

for N in N_values:

    print(f"N = {int(np.log(N) / np.log(1.1))}") # Control code execution

    std_mc_price, std_mc_err  = standard_mc(N, S0, X, T, r, sigma, call_or_put=call_or_put)
    antith_mc_price, antith_mc_err = antithetic_variates_mc(N, S0, X, T, r, sigma, call_or_put=call_or_put)
    control_mc_price, control_mc_err = control_variates_mc(N, S0, X, T, r, sigma, call_or_put=call_or_put)
    strat_mc_price, strat_mc_err = stratified_sampling_mc(N, S0, X, T, r, sigma, call_or_put=call_or_put)
    midpoint_rule_qmc_price = midpoint_rule_qmc(N, S0, X, T, r, sigma, call_or_put=call_or_put)

    std_mc_abs_err = abs(bs_price - std_mc_price)
    antith_mc_abs_err = abs(bs_price - antith_mc_price)
    control_mc_abs_err = abs(bs_price - control_mc_price)
    strat_mc_abs_err = abs(bs_price - strat_mc_price)
    midpoint_rule_qmc_abs_err = abs(bs_price - midpoint_rule_qmc_price)
    
    results["N"].append(N)

    results["std_mc_price"].append(std_mc_price)
    results["antith_mc_price"].append(antith_mc_price)
    results["control_mc_price"].append(control_mc_price)
    results["strat_mc_price"].append(strat_mc_price)
    results["midpoint_rule_qmc_price"].append(midpoint_rule_qmc_price)

    results["std_mc_err"].append(std_mc_err)
    results["antith_mc_err"].append(antith_mc_err)
    results["control_mc_err"].append(control_mc_err)
    results["strat_mc_err"].append(strat_mc_err)

    results["std_mc_abs_err"].append(std_mc_abs_err)
    results["antith_mc_abs_err"].append(antith_mc_abs_err)
    results["control_mc_abs_err"].append(control_mc_abs_err)
    results["strat_mc_abs_err"].append(strat_mc_abs_err)
    results["midpoint_rule_qmc_abs_err"].append(midpoint_rule_qmc_abs_err)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(f"data/mc_results_{call_or_put}.csv", index=False)
