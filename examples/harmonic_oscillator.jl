# The quantum harmonic oscillator by the closure method.
#
#   −ψ''/2 + x²ψ/2 = Eψ,   ψ = e^{−x²/2} h(x),   ε = 2E
#   ⇒  h'' − 2x h' + (ε − 1) h = 0.
#
# Ansatz: h a polynomial of degree n. Padding it by two is the closure
# h_{n+1} = h_{n+2} = 0 (the equation is second order with a regular point at
# the origin, so order j of the residual involves c_{j+2}). Unknowns: the lower
# coefficients c_0 … c_{n−1} and ε, with c_n = 2ⁿ fixed (Hermite convention).
# The residual then comes out at order n: n + 1 equations for n + 1 unknowns,
# solved by Newton with a ForwardDiff Jacobian. The series algebra finds
# ε = 2n + 1 and h = H_n on its own.
#
# Run in any environment that has TaylorAlgebra and ForwardDiff, e.g.
#   julia --project=benchmark examples/harmonic_oscillator.jl

using TaylorAlgebra, ForwardDiff, Printf

function residual(y, ::Val{n}) where {n}
    ε = y[end]
    c = vcat(y[1:n], 2.0^n)                 # c_0 … c_{n−1} unknown, c_n = 2ⁿ fixed
    h = Series{n+1}(c)                      # h = Σ c_j xʲ, degree n
    h = pad(h, Val(n + 3))                  # closure: c_{n+1} = c_{n+2} = 0
    h′ = derivative(h)
    h″ = derivative(h′)
    r = h″ - 2X * h′ + (ε - 1) * h          # h'' − 2x h' + (ε − 1) h, order n
    return r.c
end

function solve(n; it=40, tol=1e-12)
    y = vcat([0.3 * (-1.0)^j / (j + 1) for j in 0:n-1], 2n + 0.6)   # rough guess
    for _ in 1:it
        f = residual(y, Val(n))
        maximum(abs, f) < tol && break
        y -= ForwardDiff.jacobian(z -> residual(z, Val(n)), y) \ f
    end
    return y[end], Series(vcat(y[1:n], 2.0^n))
end

println(" n      ε      E = ε/2   h(x) = H_n(x)")
for n in 0:5
    ε, h = solve(n)
    terms = [@sprintf("%+g x^%d", coefficient(h, j), j) for j in n:-1:0 if abs(coefficient(h, j)) > 1e-10]
    @printf("%2d  %8.5f  %8.5f   %s\n", n, ε, ε / 2, join(terms, " "))
end
