# The Wilson–Fisher fixed point in the local potential approximation (LPA),
# expanded around the minimum of the potential.
#
# One real scalar in d = 3 with the Litim regulator. For the dimensionless
# potential u(ρ), ρ = φ²/2 (loop coefficient scaled to 1, which changes nothing
# in the exponents):
#     ∂t u = −3u + ρ u' + 1/(1 + u' + 2ρ u'').
# For v = u' (one ρ-derivative):
#     ∂t v = −2v + ρ v' − (3v' + 2ρ v'')/(1 + v + 2ρ v')².
#
# Expansion around the minimum κ, where v(κ) = 0: v = Σ_{j=1}^{M} v_j yʲ with
# y = ρ − κ, so ρ = κ + y is an explicit series. The equation is second order
# with a regular point at κ, so the closure takes two coefficients,
# v_{M+1} = v_{M+2} = 0 (padding by two); unknowns v_1 … v_M and κ, residual at
# order M: M + 1 equations. Exponents: eigenvalues of ∂(∂t v_j)/∂v_i at the
# fixed point with v_0 included as a variable and κ held fixed; with ∂t = k∂k
# the most negative one is −1/ν and the smallest positive one is ω. Literature (LPA, Litim): ν = 0.6496,
# ω = 0.6557 (Litim, Nucl. Phys. B 631 (2002) 128).
#
# Run in any environment that has TaylorAlgebra and ForwardDiff, e.g.
#   julia --project=benchmark examples/wilson_fisher_lpa.jl

using TaylorAlgebra, ForwardDiff, LinearAlgebra, Printf

"∂t v as a series in y = ρ − κ, for v a series in y."
function flow(v::Series, κ)
    ρ = κ + X * one(v)                      # ρ = κ + y
    v′ = derivative(v)
    v″ = derivative(v′)
    return -2v + ρ * v′ - (3v′ + 2 * (ρ * v″)) / (1 + v + 2 * (ρ * v′))^2
end

function residual(y, ::Val{M}) where {M}
    κ = y[end]
    v = Series{M+1}(vcat(zero(eltype(y)), y[1:M]))   # v_0 = 0: κ is the minimum
    v = pad(v, Val(M + 3))                           # closure: v_{M+1} = v_{M+2} = 0
    return flow(v, κ).c                              # order M: M + 1 equations
end

function solve(M, y0=vcat(0.12, zeros(M - 1), 1.9); it=50, tol=1e-12)   # guess: λ ≈ 0.12, κ ≈ 1.9
    y = vcat(y0[1:end-1], zeros(M - length(y0) + 1), y0[end])            # warm start from a lower M
    for _ in 1:it
        f = residual(y, Val(M))
        maximum(abs, f) < tol && break
        δ = ForwardDiff.jacobian(z -> residual(z, Val(M)), y) \ f
        s = 1.0                                       # damping: halve the step while it would
        while s > 1e-8 && ((y - s * δ)[end] < 0 || (y - s * δ)[1] < 0)   # make κ or λ negative
            s /= 2
        end
        y -= s * δ
    end
    return y
end

"""
Eigenvalues of the linearised coefficient flow at the fixed point (κ fixed,
v_0 free). With ∂t = k∂k a relevant direction has a negative eigenvalue: the
most negative one is −1/ν, the smallest positive one is ω.
"""
function spectrum(y, ::Val{M}) where {M}
    κ = y[end]
    f(c) = flow(pad(Series{M+1}(c), Val(M + 3)), κ).c       # c = (v_0, …, v_M)
    return sort(eigvals(ForwardDiff.jacobian(f, vcat(0.0, y[1:M]))); by=real)
end
exponents(ev) = (-real(ev[1]), real(ev[findfirst(e -> real(e) > 0, ev)]))   # 1/ν, ω

println(" M      κ         λ = v_1       ν         ω      (LPA Litim: ν = 0.6496, ω = 0.6557)")
y = solve(2)
for M in 2:2:14
    global y = solve(M, y)
    θ, ω = exponents(spectrum(y, Val(M)))
    @printf("%2d  %.7f  %.7f  %.6f  %.6f\n", M, y[end], y[1], 1 / θ, ω)
end

# The whole spectrum at the largest M: one relevant direction (−1/ν), then the
# irrelevant ones ω, ω_2, …, growing roughly like the canonical dimension j − 2
# of the coupling v_j. The highest ones come in complex pairs: truncation
# artefacts of the top coefficients, not converged in M.
println("\neigenvalues at M = 14:")
foreach(e -> @printf("  %+9.4f %+8.4f im\n", real(e), imag(e)), spectrum(y, Val(14)))

# The fixed-point potential. The series is in y = ρ − κ; `shift` re-expands it
# around the origin so it is a polynomial in ρ, and `integrate` gives u with
# u(0) = 0. Trustworthy out to a few units in y (the series has a complex pair
# of singularities, so the ratio and root estimates of the radius disagree).
κ = y[end]
w = shift(Series(vcat(0.0, y[1:14])), -κ)   # u'(ρ) as a series in ρ
u = integrate(w)                            # u(ρ), u(0) = 0
@printf("\nu'(0) = %.5f, u(κ) − u(0) = %.5f, radius estimate (root test) %.2f\n",
    w(0.0), u(κ), radius(u; nlast=6, method=:root)[1])

if Base.find_package("Plots") !== nothing   # optional plot, skipped when Plots is not installed
    @eval using Plots
    ρ = range(0, κ + 4, 400)
    display(plot(ρ, [w.(ρ) u.(ρ)]; xlabel="ρ", label=["u'(ρ)" "u(ρ)"]))
    vline!([κ]; label="κ")
end