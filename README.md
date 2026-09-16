Extracting critical exponents from simulations of statistical field theories is traditionally done with a complicated, but ad-hoc procedure.
The goal of this project, in the spirit of [Harada 2011](https://arxiv.org/abs/1102.4149), is to systematize that procedure using Bayesian probability. The benefits we're hoping to get are:

- better uncertainty quantification on the critical parameters
- relatedly: optimal use of simulation budget
- near-optimal choice of simulations, in a cost aware fashion

Try out `ising/discrepancy_m_explorer.ipynb` to see visualizations of the likelihood, and `ising/plots/budget_al_harada_square_costaware_lcycle/evolution_stepper.ipynb` to see the evolution of the posterior at more simulations are run.

# The Bayesian setup

As in any Bayesian problem, we start from the knowns and unknowns.
Concretely, let's consider the 2D Ising model, where $t = \frac{T-T_c}{T_c}$ is the reduced temperature and $z = t L^{1/\nu}$ is the collapse coordinate.

**Knowns**

- $\langle m \rangle = L^{-\beta/\nu} f_m(t L^{1/\nu} + \ldots) + g_m(t L^{1/\nu} + \ldots)$, and similar for $m^2$ and $m^4$.
- Critical exponents are positive; $T_c$ is positive.
- Scaling functions $f,g$ are smooth, with some characteristic length scale
- We can estimate observables at any $(t,L)$ by Monte Carlo, up to
    a statistical uncertainty $\sigma$ which we can estimate.

**Unknowns**

- The values of the critical parameters $\theta = (T_c,\nu,\beta)$ (and various other exponents)
- The universal scaling functions $f_c$ for each observable $c\in\{m,m^2,m^4\}$.

In short, our data $D$ are Monte Carlo estimates $y_i$ of observables at various $(t,L)$, and we want to infer $\theta = \{T_c,\nu,\beta, f_m, f_{m^2}, f_{m^4}, \ldots \}$.

## Prior

Without data, we have no reason to think that the critical parameters are correlated, so we write a factorized prior: $p(\theta) = p(T_c)\, p(\nu)\, p(\beta)p(f_m)\ldots$

We can be more specific about the priors on the critical parameters. For instance, let's $p(T_c) = \mathrm{Uniform}(2.0,\, 2.5)$, $p(\nu) = \mathrm{Uniform}(0.5,\, 1.5)$, $p(\beta) = \mathrm{Uniform}(0.05,\, 0.25)$.

The prior we choose for the scaling function itself is a Gaussian process prior: $p(f_c) = \mathrm{GP}(0,\, k_{\ell_f,\eta_f})$, where $k_{\ell_f,\eta_f}(z,z') = \eta_f^2 \exp\Bigl(-\frac{(z-z')^2}{2\ell_f^2}\Bigr)$.

Since we are Bayesian, all other model parameters, like the length scale of the Gaussian process should be put in the prior, but let's leave this for now, for simplicity.

## Likelihood (naive model)

This is $p(D \bigm| \theta) = \prod_{i=1}^{n} p(y_i \bigm| \theta)$.

Let's define $p(y_i \bigm| \theta) = p(y_i \bigm| T_c, \nu, \beta, f_m, f_{m^2}, f_{m^4}) = \mathcal{N}(y_i \bigm| \mu_i, \sigma_i^2)$, where $\mu_i = L^{-\beta/\nu} f_i(z_i)$ and $\sigma_i = \sqrt{var(y_{i,mc})}$.

We then infer $p(\theta \bigm| D) = p(T_c,\nu, \beta, f_m, \ldots \bigm| D) \propto p(T_c) p(\nu) p(\beta) p(f_m) \ldots p(D \bigm| T_c, \nu, \beta, f_m, \ldots)$ to obtain the posterior. 

Usefully, we can integrate out $p(f_c)$ to obtain a marginal likelihood $p(D \bigm| T_c, \nu, \beta) = \int p(D \bigm| T_c, \nu, \beta, f_m)df_m \ldots df_{m^4}$. This allows us to sample from the posterior $p(T_c, \beta, \nu \bigm| D) \propto p(T_c) p(\beta) p(\nu) p(D \bigm| T_c, \beta, \nu)$ using e.g. MCMC.

Writing this out, $\log p(D\mid\psi) = \sum_{c\in\{m,m^2,m^4\}} \Biggl[ -\tfrac12\, {\Phi^{(c)}}^\top \bigl(K^{(c)}\bigr)^{-1} \Phi^{(c)} -\tfrac12\log\det K^{(c)} -\tfrac{n}{2}\log(2\pi) \Biggr]$, where $K^{(c)}_{ij} = k_{l_f,\eta_f}(z_i,z_j) + \sigma_i^2 \delta_{ij}$.


# A less naive model

We know that this model is wrong, because $m = L^{-\beta/\nu} f_m(z)$ is only correct for $L\to\infty, t \to 0$, at fixed $z$. But that's fine: we should just add the corrections that we know!

What I currently do is a little ad-hoc, so there's room for improvement.

I define $\mu_i = (1-\pi(t_i,L_i)) L^{-\beta/\nu} f_i(z_i) + \pi(t_i,L_i) g_i(z_i)$, where $\pi(t,L) = \frac{a(t,L)}{1 + a(t,L)}$ is the mixture gate and $a(t,L) = L^{-\omega} + \kappa\, |t|^{\omega\nu}$ dictates which term to prefer.

The idea is that a point $m(L,t)$ is either dominated by the universal scaling function $f_m(z)$ or the correction $g_m(z)$.


As $t\to 0$ and $L\to\infty$, $a\to 0$ so $\pi\to 0$ and $\Phi = f + \varepsilon$. When $a$ is large (small $L$ and/or large $|z|$), $\pi\to 1$ and the point is explained by $g$ instead of $f$: out-of-window observations do not pertub the universal scaling function we infer. When $\kappa=0$, $a$ reduces to pure $L^{-\omega}$ gating, $\pi = \frac{L^{-\omega}}{1+L^{-\omega}}$.


## Generative model

$$
\begin{aligned}
T_c \sim \mathrm{Uniform}(2.0,\, 2.5) \\
\nu \sim \mathrm{Uniform}(0.5,\, 1.5) \\
\beta \sim \mathrm{Uniform}(0.05,\, 0.25) \\
% \omega \sim \mathrm{Uniform}(1,\, 10) \\
% \kappa \sim \mathrm{Uniform}(0,\, 20) \\[0.5em]
f \sim \mathrm{GP}\bigl(0,\, k_{\ell_f,\eta_f}\bigr) \\
% g^{(c)} \sim \mathrm{GP}\bigl(0,\, k_{\ell_g,\sigma_g}\bigr)
  % \qquad c\in\{m,m^2,m^4\} \\[0.5em]
t_i = \frac{T_i - T_c}{T_c} \\
z_i = t_i L_i^{1/\nu} \\
% a_i = L_i^{-\omega} + \kappa\, |t_i|^{\omega\nu} \\
% \pi_i = \frac{a_i}{1+a_i} \\[0.5em]
\varepsilon_i \sim \mathcal{N}\bigl(0,\, \sigma_{\Phi_c,i}^2\bigr) \\
\Phi_i = L^{-\beta/\nu} f(z_i) + \varepsilon^{(c)}_i
\end{aligned}
$$
