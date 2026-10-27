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

## Example: the harmonic oscillator by the closure method

For an ODE solved by a polynomial ansatz, the closure of the coefficient
hierarchy is written as padding, and the residual comes out at the order to
which it is actually determined. For the quantum harmonic oscillator,
ψ = e^{−x²/2} h(x) with h'' − 2x h' + (ε − 1) h = 0 and ε = 2E:

```julia
using TaylorAlgebra, ForwardDiff

function residual(y, ::Val{n}) where {n}
    ε = y[end]
    c = vcat(y[1:n], 2.0^n)                 # c_0 … c_{n−1} unknown, c_n = 2ⁿ fixed
    h = Series{n+1}(c)                      # h = Σ c_j xʲ, degree n
    h = pad(h, Val(n + 3))                  # closure: c_{n+1} = c_{n+2} = 0
    h′ = derivative(h)
    h″ = derivative(h′)
    r = h″ - 2X * h′ + (ε - 1) * h          # order n: n+1 equations, n+1 unknowns
    return r.c
end

y = vcat(zeros(n), 2n + 0.6)                # rough guess for (c_0 … c_{n−1}, ε)
for _ in 1:40
    y -= ForwardDiff.jacobian(z -> residual(z, Val(n)), y) \ residual(y, Val(n))
end
```

Newton converges to ε = 2n + 1 (E = n + ½) and the coefficients of the
Hermite polynomial Hₙ, with the wrong-parity coefficients vanishing. The full
script, `examples/harmonic_oscillator.jl`, runs to n = 10 (H₁₀ has coefficients
spanning six orders of magnitude, still exact); the first rows:

```
 n      ε      E = ε/2   h(x) = H_n(x)
 0   1.00000   0.50000   +1 x^0
 1   3.00000   1.50000   +2 x^1
 2   5.00000   2.50000   +4 x^2 -2 x^0
 3   7.00000   3.50000   +8 x^3 -12 x^1
 4   9.00000   4.50000   +16 x^4 -48 x^2 +12 x^0
 5  11.00000   5.50000   +32 x^5 -160 x^3 +120 x^1
```

`examples/hydrogen_atom.jl` does the same for the hydrogen atom, where the
origin is a regular singular point of the radial equation and one closure
coefficient suffices: it finds κ = 1/n for every (n, l), the Bohr levels
E = −1/(2n²) with their degeneracy in l, and the Laguerre coefficients:

```
 n  l  nr     κ        E [Ha]    v(ρ)
 1  0   0   1.00000   -0.50000    +1 ρ^0
 2  0   1   0.50000   -0.12500    +1 ρ^0 -0.5 ρ^1
 2  1   0   0.50000   -0.12500    +1 ρ^0
 3  0   2   0.33333   -0.05556    +1 ρ^0 -0.666667 ρ^1 +0.0740741 ρ^2
 3  1   1   0.33333   -0.05556    +1 ρ^0 -0.166667 ρ^1
 ...
```

## Benchmarks

`benchmark/benchmarks.jl` is a BenchmarkTools suite (also usable with
PkgBenchmark). `julia --project=benchmark benchmark/run.jl` runs it and
compares against `benchmark/baseline.json`, flagging any benchmark slower than
1.5× the baseline; `--save` rewrites the baseline. Timing checks are kept out
of `Pkg.test` because they are machine-dependent; the allocation checks, which
are deterministic, are in the tests.
