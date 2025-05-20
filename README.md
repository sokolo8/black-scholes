# Advanced Numerical Simulations for Black-Scholes and Beyond

This project is a comprehensive and modular Python framework for simulating and analyzing financial derivatives under the **Black-Scholes model** and its extensions.
This repository was created as part of a transition from **theoretical physics** into **quantitative finance**.

---

## Monte Carlo: [CLICK HERE 🔗](./docs/mc.ipynb)
## Finite Difference Methods: [CLICK HERE 🔗](./docs/fdm.ipynb)
## SDE Simulations: [CLICK HERE 🔗](./docs/gbm.ipynb)
## Extensions of B-S Model: Stochastic Volatility: [CLICK HERE 🔗](./docs/heston.ipynb)
## Extensions of B-S Model: Jump Diffusion: [CLICK HERE 🔗](./docs/merton_and_bates.ipynb)
## Calibration of Heston Model for S&P 500 (SPX) Options: [CLICK HERE 🔗](./docs/heston_calibration.ipynb)

---

## Overview

This repository is structured to:
- Recreate and analyze the **Black-Scholes pricing model**
- Implement **Monte Carlo methods**: standard MC, variance reduction techniques, quasi-MC
- Implement **Finite difference methods**: Explicit, implicit, Crank-Nicolson
- Extend to **American options** using free boundary solvers: Explicit, Implicit (with PSOR)
- Simulate **stochastic volatility** under Heston model and calibrate it using historical data for SPX options
- Simulate **jump-diffusion processes** under Merton and Bates models

---

## Theoretical background: Black-Scholes Model

The **Black-Scholes model** assumes the price of a stock $S(t)$ follows a geometric Brownian motion under the risk-neutral measure:

$$
dS_t = r S_t \,dt + \sigma S_t \,dW_t
$$

where:
- $r$ is the risk-free interest rate
- $\sigma$ is the volatility
- $W_t$ is a standard Brownian motion

The price of a European call option $C(t, S)$ satisfies the **Black-Scholes PDE**:

$$
\frac{\partial C}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} + r S \frac{\partial C}{\partial S} - r C = 0
$$

with terminal condition:

$$
C(T, S) = \max(S - X, 0)
$$

---

## Numerical Methods Implemented

### 1. Monte Carlo Simulation [(done) 🔗](./docs/mc.ipynb)

We use risk-neutral valuation:

$$
C_0 = e^{-rT} \mathbb{E}^\mathbb{Q}[\max(S_T - X, 0)]
$$

Estimated using:
- **Standard Monte Carlo**
- **Antithetic Variates**: use $Z$ and $-Z$ to reduce variance
- **Control Variates**: use known analytical solutions to reduce error
- **Stratified Sampling**: divide the sampling space for lower variance
- **Quasi Monte Carlo**: nearly quadratic advantage in convergence rate over standard MC

---

### 2. Finite Difference Methods (PDE Solvers) [(done) 🔗](./docs/fdm.ipynb)

We discretize the Black-Scholes PDE using:

- **Explicit Scheme** (conditionally stable)
- **Implicit Scheme** (unconditionally stable)
- **Crank-Nicolson Scheme** (unconditionally stable, second-order accurate)

#### American Options
We solve a **Linear Complementarity Problem (LCP)**:

$$
\begin{cases}
V \geq f \\
A V \geq b \\
(V - f)^T (A V - b) = 0
\end{cases}
$$

using a **Projected Successive Over-Relaxation (PSOR)** algorithm.

---

### 3. SDE Simulation [(done) 🔗](./docs/gbm.ipynb)

Simulate paths of $S(t)$ under the SDE:

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

Using:

- **Euler-Maruyama**
- **Milstein Method**

Compare simulated $S_T$ distribution with theoretical **log-normal PDF**:

$$
S_T \sim \text{LogNormal}\left(\log S_0 + (r - \frac{1}{2} \sigma^2) T, \sigma^2 T \right)
$$

---

### 4.1 Heston Model (Stochastic Volatility) [(done) 🔗](./docs/heston.ipynb)

$$
\begin{aligned}
dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S \\\\
dv_t &= \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v
\end{aligned}
$$

- Volatility becomes a stochastic process
- Captures **volatility smiles** and **leverage effects**

### 4.2 Calibration of Heston Model for S&P 500 (SPX Options) [(done) 🔗](./docs/heston_calibration.ipynb)

---

### 5 Jump-diffusion models (Merton and Bates) [(done) 🔗](./docs/merton_and_bates.ipynb)

#### 5.1 Merton model

$$
dS_t = (\mu - \lambda k) S_t dt + \sigma S_t dW_t + S_{t-} dJ_t
$$

- Jumps modeled with **Poisson process**
- Captures **sudden price changes** (earnings, news shocks)
- Explains fat tails

#### 5.2 Bates Stochastic Volatility Jump Diffusion (SVJD) model

$$
\begin{cases}
dS_t / S_t = (\mu - \lambda k)\, dt + \sqrt{\nu_t} \, dW_t^S + \left(\prod_{j=1}^{dN_t}Y_j - 1 \right)\\
d\nu_t = \kappa (\theta - v_t) \, dt + \xi \sqrt{\nu_t} \, dW_t^v\\
dW_t^S \cdot dW_t^v = \rho dt\\
P\left(dN_t=1\right) = \lambda dt
\end{cases}
$$

- Combines Heston stochastic volatility model with sudden jumps modeled by Merton

---

## Directory Structure

```bash
black-scholes/
├── LICENSE
├── README.md
├── black_scholes
│   ├── __init__.py
│   ├── fdm
│   │   ├── __init__.py
│   │   └── finite_diff_methods.py
│   ├── mc
│   │   ├── __init__.py
│   │   └── monte_carlo.py
│   └── qmc
│       ├── __init__.py
├── data
│   ├── mc_results_call.csv
│   └── mc_results_put.csv
├── docs
│   ├── data
│   │   └── spx_quotedata.csv
│   ├── documentation.ipynb
│   ├── fdm.ipynb
│   ├── figures
│   │   ├── european_american_options_vs_sigma.gif
│   │   ├── fdm_error_vs_sigma_call.gif
│   │   ├── fdm_error_vs_sigma_put.gif
│   │   ├── heston_calibr.png
│   │   ├── log-SPX-retn-vs-normal.jpg
│   │   ├── mc_results_abs_err_call.svg
│   │   ├── mc_results_abs_err_put.svg
│   │   ├── mc_results_std_err_call.svg
│   │   ├── mc_results_std_err_put.svg
│   │   └── vix.png
│   ├── gbm.ipynb
│   ├── heston.ipynb
│   ├── heston_calibration.ipynb
│   ├── iframe_figures
│   │   └── figure_345.html
│   ├── mc.ipynb
│   └── merton_and_bates.ipynb
├── plots
│   ├── animations
│   │   ├── european_american_options_vs_sigma.mp4
│   │   ├── european_options_vs_sigma_exact.mp4
│   │   ├── fdm_error_vs_sigma_call.mp4
│   │   └── fdm_error_vs_sigma_put.mp4
│   ├── mc_results_abs_err_call.pdf
│   ├── mc_results_abs_err_put.pdf
│   ├── mc_results_std_err_call.pdf
│   └── mc_results_std_err_put.pdf
├── requirements.txt
├── scripts
│   ├── european_american_options_vs_volatility_animation.py
│   ├── fdm_comparison_animation_european.py
│   ├── mc_generate_data.py
│   ├── mc_plot_results_abs_err.py
│   └── mc_plot_results_std_err.py
└── setup.py