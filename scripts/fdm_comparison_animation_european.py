from black_scholes.fdm import solve_fd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import pandas as pd

# Parameters
S_max = 200
X = 100
T = 1.0
r = 0.05
call_or_put = "call"

M = 40
N = 10

S = np.linspace(0, S_max, M+1)

sigma_values = np.linspace(0.01, 0.8, 100)

data = []

for sigma in sigma_values:

    _, error_ex = solve_fd("explicit_european", S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
    _, error_im = solve_fd("implicit_european", S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)
    _, error_cn = solve_fd("crank_european", S_max, X, T, r, sigma, M, N, call_or_put=call_or_put)

    data.append({
        "sigma": sigma,
        "error_explicit": error_ex,
        "error_implicit": error_im,
        "error_cn": error_cn
    })

df = pd.DataFrame(data)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots()

ax.set_xlabel(r"Stock Price $S$")
ax.set_ylabel(r"Error $\left | V_{\mathrm{exact}} - V_{\mathrm{FDM}} \right |$")
ax.set_title(r"Absolute Error for $\sigma = {:.2f}$".format(sigma))
ax.legend(title=r"\textbf{Method}", fontsize=10)

ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xticklabels([r"$0$", r"$50$", r"$100$", r"$150$", r"$200$"])

ax.set_yticks([0, 0.5, 1.0, 1.5])
ax.set_yticklabels([r"$0$", r"$0.5$", r"$1.0$", r"$1.5$"])

ax.minorticks_on()

ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

ax.tick_params(axis='both', which='minor', length=3.5, width=0.4)
ax.tick_params(axis='both', which='major', length=5.5, width=0.6)

line_exp, = ax.plot([], [], label='Explicit')
line_imp, = ax.plot([], [], label='Implicit')
line_cn,  = ax.plot([], [], label='Crank-Nicolson')
title = ax.set_title("")

ax.set_xlim(S.min(), S.max())
ax.set_ylim(0, max(np.max(np.abs(df["error_explicit"].iloc[0])),
                  np.max(np.abs(df["error_implicit"].iloc[0])),
                  np.max(np.abs(df["error_cn"].iloc[0]))) * 1.2)

legend = ax.legend(loc='upper right', frameon=True)

def init():
    line_exp.set_data([], [])
    line_imp.set_data([], [])
    line_cn.set_data([], [])
    return line_exp, line_imp, line_cn

def update(frame):
    row = df.iloc[frame]
    sigma = row["sigma"]
    error_ex = row["error_explicit"]
    error_im = row["error_implicit"]
    error_cn = row["error_cn"]

    line_exp.set_data(S, np.abs(error_ex))
    line_imp.set_data(S, np.abs(error_im))
    line_cn.set_data(S, np.abs(error_cn))

    if call_or_put == "put":
        title.set_text(r'European Put' + rf'\;\;$X={X}$, $T={T}$, $r={r}$, $\sigma={sigma:.2f}$')
    elif call_or_put == "call":
        title.set_text(r'European Call' + rf'\;\; $X={X}$, $T={T}$, $r={r}$, $\sigma={sigma:.2f}$')
    return line_exp, line_imp, line_cn, title

ani = animation.FuncAnimation(
    fig, update, frames=len(df), init_func=init,
    interval=100, blit=True
)

plt.tight_layout(pad=2)
ani.save(f"plots/animations/fdm_error_vs_sigma_{call_or_put}.mp4", writer='ffmpeg', dpi=250)
