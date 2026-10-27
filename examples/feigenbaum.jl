# Feigenbaum's universal function by polynomial truncation.
#
# Period doubling in unimodal maps is governed by the fixed point of the
# doubling operator, the Feigenbaum–Cvitanović functional equation
#     g(x) = −α g(g(−x/α)),      g(0) = 1,  g even,
# whose solution g and constant α = 2.502907875… are universal. In y = x²,
# g(x) = G(y): the equation reads  G(y) + α G( G(y/α²)² ) = 0.
#
# Ansatz: G a polynomial of degree M in y with G_0 = 1. Unknowns G_1 … G_M
# and α; the residual, a composition of series, is truncated at order M by
# the algebra itself (compositions keep the smaller order): M + 1 equations
# for M + 1 unknowns, Newton with a ForwardDiff Jacobian. The same structure
# as a truncated RG fixed point: a nonlinear operator, a polynomial ansatz,
# a closure, and an eigenvalue-like constant that comes out of the solve.
#
# Run in any environment that has TaylorAlgebra and ForwardDiff, e.g.
#   julia --project=benchmark examples/feigenbaum.jl

using TaylorAlgebra, ForwardDiff, Printf

function residual(y, ::Val{M}) where {M}
    α = y[end]
    G = Series{M+1}(vcat(one(eltype(y)), y[1:M]))   # G = 1 + G_1 y + … + G_M y^M
    t = X * one(G)                                   # the series y
    inner = G ∘ (t / α^2)                            # G(y/α²)
    r = G + α * (G ∘ inner^2)                        # G(y) + α G(G(y/α²)²), order M
    return r.c
end

function solve(M; it=50, tol=1e-13)
    y = vcat(-1.5, 0.1, zeros(M - 2), 2.5)           # rough guess: G_1, G_2, α
    for _ in 1:it
        f = residual(y, Val(M))
        maximum(abs, f) < tol && break
        y -= ForwardDiff.jacobian(z -> residual(z, Val(M)), y) \ f
    end
    return y[end], Series(vcat(1.0, y[1:M]))
end

println(" M       α            G_1           G_2        (exact: 2.502907875, −1.527632997, 0.1048151943)")
for M in 2:2:12
    α, G = solve(M)
    @printf("%2d  %.10f  %+.10f  %+.10f\n", M, α, coefficient(G, 1), coefficient(G, 2))
end
