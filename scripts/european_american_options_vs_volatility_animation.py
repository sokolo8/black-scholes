from black_scholes.fdm import analytical_european_options
from black_scholes.fdm import implicit_psor_american_options
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

M = 100
N = 50

S = np.linspace(0, S_max, M+1)

sigma_values = np.linspace(0.01, 0.5, 100)

data = []

for sigma in sigma_values:

    call_price_eu = analytical_european_options(S, X, T, r, sigma, call_or_put="call")
    put_price_eu = analytical_european_options(S, X, T, r, sigma, call_or_put="put")
    put_price_am = implicit_psor_american_options(S_max, X, T, r, sigma, M, N, call_or_put="put")

    data.append({
        "sigma": sigma,
        "call_price_eu": call_price_eu,
        "put_price_eu": put_price_eu,
        "put_price_am": put_price_am
    })

df = pd.DataFrame(data)

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

fig, ax = plt.subplots()

ax.set_xlabel(r"Stock Price $S$")
ax.set_ylabel(r"Option value")

ax.set_xticks([0, 50, 100, 150, 200])
ax.set_xticklabels([r"$0$", r"$50$", r"$100$", r"$150$", r"$200$"])

ax.set_yticks([0, 50, 100, 150, 200])
ax.set_yticklabels([r"$0$", r"$50$", r"$100$", r"$150$", r"$200$"])

ax.minorticks_on()

ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

ax.tick_params(axis='both', which='minor', length=3.5, width=0.4)
ax.tick_params(axis='both', which='major', length=5.5, width=0.6)

line_call_eu, = ax.plot([], [], label='European/American Call')
line_put_eu, = ax.plot([], [], label='European Put')
line_put_am, = ax.plot([], [], label='American Put')
title = ax.set_title("")

ax.set_xlim(S.min(), S.max())
ax.set_ylim(0, max(np.max(np.abs(df["call_price_eu"].iloc[0])),
                  np.max(np.abs(df["put_price_eu"].iloc[0])),
                  np.max(np.abs(df["put_price_am"].iloc[0]))) * 1.2)

legend = ax.legend(loc='upper right', frameon=True)

def init():
    line_call_eu.set_data([], [])
    line_put_eu.set_data([], [])
    line_put_am.set_data([], [])
    return line_call_eu, line_put_eu, line_put_am

def update(frame):
    row = df.iloc[frame]
    sigma = row["sigma"]
    price_call_eu = row["call_price_eu"]
    price_put_eu = row["put_price_eu"]
    price_put_am = row["put_price_am"]

    line_call_eu.set_data(S, np.abs(price_call_eu))
    line_put_eu.set_data(S, np.abs(price_put_eu))
    line_put_am.set_data(S, np.abs(price_put_am))

    title.set_text(r'European and American Options' + rf'\;\;$X={X}$, $T={T}$, $r={r}$, $\sigma={sigma:.2f}$')

    return line_call_eu, line_put_eu, line_put_am, title

ani = animation.FuncAnimation(
    fig, update, frames=len(df), init_func=init,
    interval=50, blit=True
)

plt.tight_layout(pad=2)
ani.save(f"plots/animations/european_american_options_vs_sigma.mp4", writer='ffmpeg', dpi=250)
