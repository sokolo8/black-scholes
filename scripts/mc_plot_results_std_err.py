import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter
import pandas as pd

call_or_put = "put"
S0 = 100      # Initial stock price
X = 100       # eXercise price
T = 1.0       # Time to maturity (in years)
r = 0.05      # Risk-free interest rate
sigma = 0.2   # Volatility
results = pd.read_csv(f"data/mc_results_{call_or_put}.csv")

std_mc_err = results["std_mc_err"]
antith_mc_err = results["antith_mc_err"]
control_mc_err = results["control_mc_err"]
strat_mc_err = results["strat_mc_err"]

N_values = [int(1.1 ** i) for i in range(90, 190)]    # Number of Monte Carlo simulations

marker = [r'$\otimes$', r'$\ast$', '+', 'x', r'$\star$', r'$\circ$']
markersize = [3, 4, 4, 3.5, 4, 3]
markersize_inset = [1.0, 2.0, 2.0, 1.2, 2.0, 1.0]
markeredgewidth = [0.25, 0.25, 0.3, 0.3, 0.2, 0.05]
markeredgewidth_inset = [0.1, 0.1, 0.11, 0.11, 0.09, 0.0025]

plt.switch_backend('pgf')

# Main plot
fig, ax = plt.subplots(figsize=(4.0, 2.5))
ax.set_xscale('log')
ax.set_yscale('log')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": "Times New Roman",
})

major_ticks_x = [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8]
ax.set_xticks(major_ticks_x)
minor_ticks_x = np.concatenate([np.linspace(2, 8, 4, endpoint=True) * 10**exp for exp in range(2, 8)])
ax.xaxis.set_minor_locator(FixedLocator(minor_ticks_x))
ax.xaxis.set_minor_formatter(NullFormatter())

major_ticks_y = [1e-05, 1e-04, 1e-03, 1e-02, 1e-01]
ax.set_yticks(major_ticks_y)
minor_ticks_y = np.concatenate([np.linspace(2, 8, 4, endpoint=True) * 10**exp for exp in range(-5, 0)])
ax.yaxis.set_minor_locator(FixedLocator(minor_ticks_y))
ax.yaxis.set_minor_formatter(NullFormatter())

plt.tick_params(axis='both', which='major', direction='in', labelsize=8, length=5, width=0.2, bottom=True, top=True, left=True, right=True)
plt.tick_params(axis='both', which='minor', direction='in', length=2.2, width=0.1, bottom=True, top=True, left=True, right=True)

# Loop for main plot

plt.plot(N_values, std_mc_err, marker=marker[0], markersize=markersize[0], markeredgewidth=markeredgewidth[0], label=r'\bf{standard MC}', alpha=1.0, linewidth=0.35)
plt.plot(N_values, antith_mc_err, marker=marker[1], markersize=markersize[1], markeredgewidth=markeredgewidth[1], label=r'\bf{antithetic MC}', alpha=1.0, linewidth=0.35)
plt.plot(N_values, control_mc_err, marker=marker[2], markersize=markersize[2], markeredgewidth=markeredgewidth[2], label=r'\bf{control MC}', alpha=1.0, linewidth=0.35)
plt.plot(N_values, strat_mc_err, marker=marker[3], markersize=markersize[3], markeredgewidth=markeredgewidth[3], label=r'\bf{stratified MC}', alpha=1.0, linewidth=0.35)

plt.axis((3 * 10 ** 3, 10 ** 8, 1e-05, 5 * 1e-01))
plt.xlabel(r'\bf{Numer of samplings} $N$', fontsize=8)
plt.ylabel(r'$1.96\times$\bf{Standard error}', fontsize=8)
plt.legend(loc='lower left', fontsize=6.4, frameon=False, bbox_to_anchor=(0.02, 0.02))

if call_or_put == "put":
    plt.title(r'\bf{European Put}' + rf'\;\;$S_0={S0}$, $X={X}$, $T={T}$, $r={r}$, $\sigma={sigma}$', fontsize=9)
elif call_or_put == "call":
    plt.title(r'\bf{European Call}' + rf'\;\; $S_0={S0}$, $X={X}$, $T={T}$, $r={r}$, $\sigma={sigma}$', fontsize=9)

plt.tight_layout(pad=0.2)
plt.subplots_adjust(right=0.9)
plt.savefig(f"plots/mc_results_std_err_{call_or_put}.pdf")

plt.close()
