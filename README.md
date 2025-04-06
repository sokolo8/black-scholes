# Advanced Numerical Simulations for Black-Scholes and Beyond **STILL IPMROVING THE PROJECT**

This project is a comprehensive and modular Python framework for simulating and analyzing financial derivatives under the **Black-Scholes model** and its **advanced generalizations** such as the **Heston stochastic volatility** model and the **Merton jump-diffusion** model.
It is aimed at practitioners and researchers in **quantitative finance**, **financial engineering**, and **computational physics**, and was designed as part of a transition from theoretical physics into **quantitative risk analysis**.

---

## Full Documentation: [CLICK HERE 🔗](./docs/documentation.ipynb)
## Part 1: [CLICK HERE 🔗](./docs/part1.ipynb)
## Part 2: [CLICK HERE 🔗](./docs/part2.ipynb)

---

## Overview

This repository is structured to:
- Recreate and analyze the **Black-Scholes pricing model** from first principles
- Implement **advanced numerical methods**: Monte Carlo, PDE-based, and SDE simulations
- Explore **variance reduction techniques**
- Extend to **American options** using free boundary solvers
- Simulate **stochastic volatility** and **jumps**
- Provide a strong foundation for **model calibration**

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

The price of a European call option $C(t, S) $ satisfies the **Black-Scholes PDE**:

$$
\frac{\partial C}{\partial t} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 C}{\partial S^2} + r S \frac{\partial C}{\partial S} - r C = 0
$$

with terminal condition:

$$
C(T, S) = \max(S - K, 0)
$$

---

## Numerical Methods Implemented

### 1. Monte Carlo Simulation

We use risk-neutral valuation:

$$
C_0 = e^{-rT} \mathbb{E}^\mathbb{Q}[\max(S_T - K, 0)]
$$

Simulated using:

- Standard Monte Carlo
- **Antithetic Variates**: use $Z$ and $-Z$ to reduce variance
- **Control Variates**: use known analytical solutions to reduce error
- **Stratified Sampling**: divide the sampling space for lower variance

---

### 2. Finite Difference Methods (PDE Solvers)

We discretize the Black-Scholes PDE using:

- **Explicit Scheme** (conditionally stable)
- **Implicit Scheme** (unconditionally stable)
- **Crank-Nicolson Scheme** (second-order accurate)

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

### 3. SDE Simulation

Simulate paths of $S(t)$ under the SDE:

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

Using:

- **Euler-Maruyama**
- **Milstein Method** (improved accuracy)

Compare simulated $S_T$ distribution with theoretical **log-normal PDF**:

$$
S_T \sim \text{LogNormal}\left(\log S_0 + (r - \frac{1}{2} \sigma^2) T, \sigma^2 T \right)
$$

---

### 4. Advanced Models

#### Heston Model (Stochastic Volatility):

$$
\begin{aligned}
dS_t &= \mu S_t dt + \sqrt{v_t} S_t dW_t^S \\\\
dv_t &= \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v
\end{aligned}
$$

- Volatility becomes a stochastic process
- Captures **volatility smiles** and **leverage effects**

#### Merton Jump-Diffusion Model:

$$
dS_t = (\mu - \lambda k) S_t dt + \sigma S_t dW_t + S_{t-} dJ_t
$$

- Jumps modeled with **Poisson process**
- Captures **sudden price changes** (earnings, news shocks)

---

## Directory Structure

```bash
bs-numerical-simulations/
├── src/
│   ├── monte_carlo/            # Monte Carlo and variance reduction
│   ├── finite_difference/      # PDE solvers: Explicit, Implicit, CN, PSOR
│   ├── sde_simulation/         # SDE path simulation (Euler, Milstein)
│   ├── advanced_models/        # Heston and Merton simulation
│   └── utils/                  # Helpers, Black-Scholes formula
│
├── notebooks/                  # Theory and documentation notebooks
│   ├── black_scholes_theory_intro.ipynb
│   ├── monte_carlo_option_pricing_theory.ipynb
│   ├── fdm_black_scholes_theory.ipynb
│   ├── psor_theory_american_options.ipynb
│   ├── sde_simulation_theory.ipynb
│   └── heston_merton_models_theory.ipynb
│
├── README.md
├── requirements.txt
└── LICENSE