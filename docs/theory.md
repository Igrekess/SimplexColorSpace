# Theory

## From primes to color

The Sieve of Eratosthenes, starting from `s = 1/2`, produces a geometric
distribution on prime gaps. This distribution has a unique fixed point at
`mu* = 15`, where exactly 3 primes ({3, 5, 7}) are "active" — they dominate
the sieve dynamics.

These 3 primes become the 3 chromatic channels. The prime `p = 2` becomes
the luminance channel.

**Nothing is chosen. Everything is computed.**

## The simplex

A color's chromaticity lives on the 2-simplex:

```
pi = (pi_3, pi_5, pi_7)    with    pi_3 + pi_5 + pi_7 = 1
```

The coordinates are the gamma-weighted LMS cone responses, normalized:

```
w_i = gamma_i * LMS_i
pi_i = w_i / (w_1 + w_2 + w_3)
```

where `gamma_p` are the effective dimensions derived from the sieve:

```
gamma_3 = 0.808    (L-cone channel)
gamma_5 = 0.696    (M-cone channel)
gamma_7 = 0.595    (S-cone channel)
```

## The conservation law

On the simplex, two quantities are defined:

- **Saturation** S = D_KL(pi || uniform) — how far from achromatic
- **Luminance entropy** L = H(pi) — Shannon entropy of the simplex point

The GFT identity states:

```
S + L = log2(3)    (exactly, algebraically)
```

This means: the total informational budget is fixed at ~1.585 bits. If you
increase saturation, luminance entropy decreases by exactly the same amount.
This is not a heuristic — it's a mathematical identity that holds for any
point on the simplex.

## The metric

The natural metric on the simplex is the Fisher information metric, weighted
by gamma_p. For two colors with coordinates (ell_1, pi_1) and (ell_2, pi_2):

```
dE^2 = (3/4) * d_lum^2 + (1/4) * d_chrom^2
```

where:

- `d_lum = 2|arcsin(sqrt(ell_1)) - arcsin(sqrt(ell_2))|`
  (Fisher distance on the Bernoulli manifold, p=2 channel)

- `d_chrom = 2 * arccos(sum_i sqrt(pi_1_i * pi_2_i))`
  (Bhattacharyya distance on the simplex)

- Weights 3/4 and 1/4 come from N/(N+1) with N = 3 active primes

The metric is unique (Cencov's theorem): it is the only Riemannian metric
on statistical manifolds that is invariant under sufficient statistics.

## Derivation chain

```
s = 1/2                              (unique input, T0)
  -> geometric distribution           (Cauchy + max-entropy, L0)
  -> holonomy angles sin^2(theta_p)   (T6)
  -> fixed point mu* = 15             (T7)
  -> active primes {3, 5, 7}          (N = 3)
  -> effective dimensions gamma_p     (T7 at mu*)
  -> Fisher metric on simplex         (Cencov uniqueness theorem)
  -> Bhattacharyya geodesic on D^2    (geodesic of Fisher metric)
  -> 3/4 luminance + 1/4 chroma       (N/(N+1) balance)
  -> dE_SCS                          (color difference formula)
```

Zero fitted parameters at every step.

## Reference

Y. Senez, *"The Sieve Color Space: A First-Principles Color Space
from the Sieve of Eratosthenes"* (2026). See `article/` directory.
