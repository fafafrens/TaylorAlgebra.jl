# The hydrogen atom by the closure method.
#
# Radial equation in atomic units (ρ = r/a₀, E = −κ²/2 Hartree) for u = rR:
#     u'' + (2/ρ − l(l+1)/ρ² − κ²) u = 0.
# With u = ρ^{l+1} e^{−κρ} v(ρ) the polynomial factor obeys
#     ρ v'' + 2(l+1−κρ) v' + 2(1−κ(l+1)) v = 0.
#
# Ansatz: v a polynomial of degree nr (the radial quantum number). The origin
# is a regular singular point, so order j of the residual involves c_{j+1} only
# and ONE padded coefficient, c_{nr+1} = 0, closes the hierarchy (compare the
# oscillator, a regular point, which needs two). Unknowns: c_1 … c_nr and κ with
# c_0 = 1; residual at order nr: nr + 1 equations for nr + 1 unknowns, Newton
# with a ForwardDiff Jacobian. The algebra finds κ = 1/n with n = nr + l + 1,
# i.e. the Bohr levels E_n = −1/(2n²), and v ∝ the associated Laguerre
# polynomial L^{2l+1}_{nr}(2κρ).
#
# Run in any environment that has TaylorAlgebra and ForwardDiff, e.g.
#   julia --project=benchmark examples/hydrogen_atom.jl

using TaylorAlgebra, ForwardDiff, Printf

function residual(y, ::Val{nr}, l) where {nr}
    κ = y[end]
    c = vcat(one(eltype(y)), y[1:nr])       # c_0 = 1 fixed, c_1 … c_nr unknown
    v = Series{nr+1}(c)                     # v = Σ c_j ρʲ, degree nr
    v = pad(v, Val(nr + 2))                 # closure: c_{nr+1} = 0
    v′ = derivative(v)
    v″ = derivative(v′)
    r = X * v″ + 2 * ((l + 1) * v′ - κ * X * v′) + 2 * (1 - κ * (l + 1)) * v   # order nr
    return r.c
end

function solve(nr, l; it=40, tol=1e-12)
    n = nr + l + 1
    y = vcat([(-1.0)^j / factorial(j) for j in 1:nr], 0.9 / n)     # rough guess
    for _ in 1:it
        f = residual(y, Val(nr), l)
        maximum(abs, f) < tol && break
        y -= ForwardDiff.jacobian(z -> residual(z, Val(nr), l), y) \ f
    end
    return y[end], Series(vcat(1.0, y[1:nr]))
end

println(" n  l  nr     κ        E [Ha]    v(ρ)")
for n in 1:4, l in 0:n-1
    nr = n - l - 1
    κ, v = solve(nr, l)
    terms = [@sprintf("%+.6g ρ^%d", coefficient(v, j), j) for j in 0:nr]
    @printf("%2d  %d  %2d  %8.5f  %9.5f    %s\n", n, l, nr, κ, -κ^2 / 2, join(terms, " "))
end
