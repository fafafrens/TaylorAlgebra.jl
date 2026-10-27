# TaylorAlgebra.jl

Minimal truncated power series in one variable, built for solving ODE/PDE
fixed-point problems by polynomial truncation with Newton + automatic
differentiation.

```julia
using TaylorAlgebra
a = Series([1.0, 2.0, 3.0])        # 1 + 2x + 3x², order 2
b = exp(a) / (1 + a^2)          # same order
derivative(b)                   # order 1   (top coefficient unknown → dropped)
X * b                           # order 3   (x·b, exact)
4X^2 * b                        # order 4
a(0.3)                          # evaluate
a ∘ b                           # compose
shift(a, 0.5)                   # re-expand around x = 0.5
inverse(Series([0.0, 1.0, 1.0]))   # compositional inverse
```

## Design

- **Storage is an `SVector{N,T}`.** The number of coefficients `N` is part of
  the type, so every operation is unrolled and allocation-free (measured up to
  N ≈ 65), and the element type is generic: `ForwardDiff` duals pass through,
  so Jacobians of series-valued residuals come for free.
- **Strict order bookkeeping.** Series of different orders combine to the
  smaller one (by promotion, like `Number`s). `derivative` lowers the order by
  one because the top coefficient of a derivative is not known; `X * a` and
  `integrate` raise it by one because their top coefficient is exact.
  `pad`, `truncate` and `Series{N}(a)` change the order explicitly. A truncation
  closure is then *written*, not hidden: pad the unknowns to the closure order
  and the residual of the differential equation comes out at the order to
  which it is actually determined.
- **Monomials.** `X` is the monomial x, `Monomial{P}(s)` is s·xᴾ. Products
  with a series are coefficient shifts, exact and O(N), not truncated Cauchy
  products. Literal powers (`X^2`) stay at compile time via `literal_pow`;
  `X^n` with a run-time `n` dispatches dynamically.
- **Algorithms.** Cauchy product as a generator sum inside one `ntuple`;
  division, inverse, real powers (Miller's recurrence), `exp` and `log` as
  O(N²) recursions on a stack `MVector`; `shift` by Horner's form of a(κ + y); `inverse` by fixed-point iteration (exact after N−1 steps);
  `radius` by a Domb–Sykes ratio fit or the root test.

## Example: closure of a differential hierarchy

For an ODE `F(w, w', w'') = 0` solved with `w = Σ_{n<M} W_n xⁿ` and the
closure `W_M = 0`:

```julia
W   = pad(Series(y), Val(M + 1))     # unknowns W_0..W_{M-1}, padded: that IS the closure
res = F(W, derivative(W), derivative(W, Val(2)))   # comes out at order M−1: M equations
```

Newton on `y ↦ res.c` with `ForwardDiff.jacobian` is then a square system.

## Status

Written for a functional-renormalisation-group fixed-point problem
(expansion of the effective potential and wave-function renormalisation in
the field). Not registered. Tests: `] test TaylorAlgebra`.

## Benchmarks

`benchmark/benchmarks.jl` is a BenchmarkTools suite (also usable with
PkgBenchmark). `julia --project=benchmark benchmark/run.jl` runs it and
compares against `benchmark/baseline.json`, flagging any benchmark slower than
1.5× the baseline; `--save` rewrites the baseline. Timing checks are kept out
of `Pkg.test` because they are machine-dependent; the allocation checks, which
are deterministic, are in the tests.
