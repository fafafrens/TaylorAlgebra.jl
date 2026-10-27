# Lambert's W function by series reversion.
#
# W(x) is defined by W e^W = x. The series of w e^w around w = 0 has no constant
# term and unit slope, so `inverse` (compositional inverse) gives W(x) as a
# series in x directly. Lagrange's inversion formula gives the coefficients in
# closed form, W(x) = Σ_{n≥1} (−n)^{n−1}/n! xⁿ, which is the check.
#
# Run in any environment that has TaylorAlgebra, e.g.
#   julia --project=benchmark examples/lambert_w.jl

using TaylorAlgebra, Printf

w = X * one(Series{12,Float64})          # the series w
W = inverse(w * exp(w))                  # W(x): inverse of w e^w

println(" n   coefficient      (−n)^(n−1)/n!")
for n in 1:order(W)
    @printf("%2d  %+.10e  %+.10e\n", n, coefficient(W, n), (-n)^(n - 1) / factorial(n))
end
println("\ncheck: W(0.2) = ", W(0.2), ",  W e^W = ", W(0.2) * exp(W(0.2)))
println("radius of convergence: ", radius(W; nlast=6)[1], "  (exact 1/e = ", 1 / ℯ, ")")
