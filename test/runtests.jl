using Test, TaylorAlgebra, StaticArrays, ForwardDiff

const a = Series([0.3, 1.0, -0.5, 0.2, 0.1, -0.05, 0.02, 0.01])
const b = Series([1.5, 0.2, 0.1, -0.3, 0.05, 0.0, 0.1, -0.02])
maxabs(t::Series) = maximum(abs, t.c)

@testset "construction, order, access" begin
    @test order(a) == 7 && length(a) == 8
    @test coefficient(a, 0) == 0.3 && coefficient(a, 7) == a[8]
    @test Series{4}([1.0, 2.0]) == Series([1.0, 2.0, 0.0, 0.0])
    @test Series{3}(2.0) == Series([2.0, 0.0, 0.0])
    @test pad(a, Val(10)) == Series{10}(a) && order(pad(a, 10)) == 9
    @test truncate(a, 4) == Series(a.c[1:4]) && truncate(a, b) == a
    @test zero(a) + a == a && one(a) * a == a && iszero(a - a)
    @test a ≈ a + 1e-17
end

@testset "ring operations and promotion" begin
    @test (a + b) - b ≈ a
    @test a * b ≈ b * a
    @test (a * b) * inv(b) ≈ a
    @test a / b ≈ a * inv(b)
    @test 2.0 * a == a + a == a * 2.0 && a / 2 == 0.5 * a
    @test a + 1 == 1 + a && coefficient(a + 1, 0) == 1.3
    @test order(a + Series{3}([1.0, 1.0, 1.0])) == 2            # smaller order wins
    @test order(2.0 + Series{3}(1.0)) == 2
    @test Series{3}([1, 2, 3]) * Series{3}([1.0, 0.0, 0.0]) == Series([1.0, 2.0, 3.0])   # Int×Float
end

@testset "powers, sqrt, exp, log" begin
    @test a^2 == a * a && a^3 == a * a * a && a^0 == one(a)
    @test a^-1 ≈ inv(a) && a^(-2) ≈ inv(a)^2
    @test a^2.0 ≈ a * a && a^(1 // 2) * a^(1 // 2) ≈ a
    @test sqrt(a)^2 ≈ a
    @test collect(exp(Series{8}([0.0, 1.0])).c) ≈ [1 / factorial(n) for n in 0:7]
    @test collect(log(Series{8}([1.0, 1.0])).c) ≈ [0.0; [(-1)^(n + 1) / n for n in 1:7]]
    @test log(exp(a)) ≈ a && exp(log(a)) ≈ a
    @test exp(2a) ≈ exp(a)^2
    @test a^b ≈ exp(b * log(a)) && a^Series{8}(2.0) ≈ a^2
end

@testset "derivative, integrate, monomial" begin
    @test order(derivative(a)) == 6 && order(integrate(a)) == 8
    @test derivative(integrate(a)) == a
    @test coefficient(integrate(a, 7.0), 0) == 7.0
    @test derivative(a * b) ≈ derivative(a) * b + a * derivative(b)
    @test derivative(exp(a)) ≈ derivative(a) * exp(a)
    @test derivative(a, Val(2)) == derivative(derivative(a)) && derivative(a, 0) == a
    @test X * a == Series([0.0; a.c...]) && a * X == X * a
    @test order(X^2 * a) == 9 && X^2 * a == X * (X * a) && X * X == X^2
    @test 4.0 * X * a == 4.0 * (X * a) && 2X^2 * a == 2.0 * (X^2 * a)
    @test (2X)^2 isa Monomial{2} && (2X)^2 * one(a) == 4X^2 * one(a)
    @test a + 2X == a + Series{8}(2X) && coefficient(X^2 - a, 2) == 1 - a[3]
    @test a - X^20 == a                                       # beyond the order: dropped
end

@testset "evaluate, compose, inverse, shift" begin
    @test evaluate(a, 0.0) == a[1] && a(0.0) == a[1]
    @test a(0.3) ≈ sum(a.c[n+1] * 0.3^n for n in 0:7)
    y = X * one(Series{7,Float64})
    @test (a ∘ y) ≈ a && compose(a, y) == a(y)
    κ = 0.7
    @test shift(a, κ)(-κ) ≈ a(0.0) && shift(shift(a, κ), -κ) ≈ a
    @test a ∘ (κ + y) ≈ shift(a, κ)
    e = exp(Series{10}([0.0, 1.0])) - 1                      # e^x − 1  ↔  log(1 + x)
    @test inverse(e) ≈ log(Series{10}([1.0, 1.0]))
    g = Series([0.0, 2.0, -1.0, 0.5, 0.3, -0.2, 0.1, 0.05]); h = inverse(g)
    @test (g ∘ h) ≈ X * one(Series{6,Float64}) && (h ∘ g) ≈ X * one(Series{6,Float64})
    @test_throws ArgumentError inverse(a)
end

@testset "radius" begin
    @test radius(inv(Series{20}([1.0, -0.5])))[1] ≈ 2.0
    r, s = radius(inv(Series{20}([1.0, 3.0]))); @test r ≈ 1 / 3 && s == -1
    @test radius(Series{20}([1.0, -1.0])^(-0.5); nlast=6)[1] ≈ 1.0
    @test 0.9 < radius(Series{20}([1.0, -1.0])^(-0.5); nlast=6, method=:root)[1] < 1.3
end

# Physicists' Hermite polynomials H_0 … H_5, for the next two test sets.
H(x, n) = (1.0, 2x, 4x^2 - 2, 8x^3 - 12x, 16x^4 - 48x^2 + 12, 32x^5 - 160x^3 + 120x)[n+1]

@testset "discovering the harmonic oscillator" begin
    # Schrödinger equation −ψ''/2 + x²ψ/2 = Eψ with ψ = e^{−x²/2} h(x), ε = 2E:
    #     h'' − 2x h' + (ε − 1) h = 0.
    # Closure method as for a truncated fixed-point hierarchy: h is a polynomial of
    # degree n, padded by two (h_{n+1} = h_{n+2} = 0: the equation is second order
    # with no singular point at the origin, so order j involves c_{j+2} and the
    # closure takes two coefficients). Unknowns c_0..c_{n−1} and ε with the top
    # coefficient fixed to 2ⁿ (Hermite convention): n+1 unknowns, residual at
    # orders 0..n: n+1 equations, Newton with ForwardDiff. The algebra must find
    # ε = 2n + 1, i.e. E_n = n + 1/2, and h = H_n, including the vanishing of the
    # coefficients of the wrong parity.
    function oscillator(y, ::Val{n}) where {n}
        ε = y[end]
        c = vcat(y[1:n], 2.0^n)                 # c_0 … c_{n−1} unknown, c_n = 2ⁿ fixed
        h = Series{n+1}(c)                      # h = Σ c_j xʲ, degree n
        h = pad(h, Val(n + 3))                  # closure: c_{n+1} = c_{n+2} = 0
        h′ = derivative(h)
        h″ = derivative(h′)
        residual = h″ - 2X * h′ + (ε - 1) * h   # h'' − 2x h' + (ε − 1) h = 0, order n
        return residual.c
    end
    for n in 0:5
        y = vcat([0.3 * (-1.0)^j / (j + 1) for j in 0:n-1], 2n + 0.6)   # rough guess
        for _ in 1:40
            f = oscillator(y, Val(n))
            maximum(abs, f) < 1e-12 && break
            y -= ForwardDiff.jacobian(z -> oscillator(z, Val(n)), y) \ f
        end
        ε = y[end]; h = Series(vcat(y[1:n], 2.0^n))
        @test ε ≈ 2n + 1                                        # E_n = n + 1/2
        @test all(isapprox(h(x), H(x, n); atol=1e-9) for x in (-1.3, 0.0, 0.5, 2.0))  # h = H_n (H_n(0) can be 0)
        @test all(abs(coefficient(h, j)) < 1e-10 for j in (n - 1):-2:0)   # parity
    end
end

@testset "discovering the Hermite polynomials" begin
    # Generating function: exp(2xt − t²) = Σ_n H_n(x) tⁿ/n!, with t the series variable.
    t = X * one(Series{8,Float64})
    for x in (0.0, 0.5, -1.3, 2.0)
        g = exp(2x * t - t^2)
        @test all(factorial(n) * coefficient(g, n) ≈ H(x, n) for n in 0:5)
    end
    # Rodrigues: H_n(x0) = (−1)ⁿ e^{x0²} dⁿ/dxⁿ e^{−x²} at x0. The n-th derivative at a
    # point is n! times the n-th coefficient of the series expanded around that point.
    for x0 in (0.0, 0.7, -1.3)
        y = X * one(Series{9,Float64})
        f = exp(-(x0 + y)^2)                         # e^{−x²} expanded around x0
        @test all((-1)^n * exp(x0^2) * factorial(n) * coefficient(f, n) ≈ H(x0, n) for n in 0:5)
        @test all(coefficient(derivative(f, Val(n)), 0) ≈ factorial(n) * coefficient(f, n) for n in 0:5)
    end
    # Three-term recurrence H_{n+1} = 2x H_n − 2n H_{n−1}, read off the generating function.
    x = 0.9; g = exp(2x * t - t^2); h(n) = factorial(n) * coefficient(g, n)
    @test all(h(n + 1) ≈ 2x * h(n) - 2n * h(n - 1) for n in 1:6)
end

@testset "discovering the hydrogen atom" begin
    # Radial equation in atomic units, E = −κ²/2:  u'' + (2/ρ − l(l+1)/ρ² − κ²) u = 0.
    # With u = ρ^{l+1} e^{−κρ} v(ρ) the polynomial factor obeys
    #     ρ v'' + 2(l+1−κρ) v' + 2(1−κ(l+1)) v = 0.
    # Same closure method as a truncated fixed-point hierarchy: v is a polynomial of
    # degree nr (v_{nr+1} = 0 by padding), unknowns v_1..v_nr and κ (v_0 = 1), the
    # residual comes out at order nr (nr+1 equations), Newton with ForwardDiff. The
    # algebra must then find κ = 1/n with n = nr + l + 1, i.e. the Bohr levels
    # E = −1/(2n²), and the coefficients of the associated Laguerre polynomial.
    function hydrogen(y, ::Val{nr}, l) where {nr}
        κ = y[end]
        c = vcat(one(eltype(y)), y[1:nr])       # c_0 = 1 fixed, c_1 … c_nr unknown
        v = Series{nr+1}(c)                     # v = Σ c_j ρʲ, degree nr
        v = pad(v, Val(nr + 2))                 # closure: c_{nr+1} = 0
        v′ = derivative(v)
        v″ = derivative(v′)
        residual = X * v″ + 2 * ((l + 1) * v′ - κ * X * v′) + 2 * (1 - κ * (l + 1)) * v   # order nr
        return residual.c
    end
    for (nr, l) in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (3, 2))
        n = nr + l + 1
        y = vcat([(-1.0)^j / factorial(j) for j in 1:nr], 0.9 / n)      # rough guess
        for _ in 1:30
            f = hydrogen(y, Val(nr), l)
            maximum(abs, f) < 1e-13 && break
            y -= ForwardDiff.jacobian(z -> hydrogen(z, Val(nr), l), y) \ f
        end
        κ = y[end]
        @test κ ≈ 1 / n                                   # E_n = −1/(2n²)
        c = vcat(1.0, y[1:nr])                            # textbook recurrence of the coefficients
        @test all(c[j+2] ≈ 2(κ * (j + l + 1) - 1) / ((j + 1) * (j + 2l + 2)) * c[j+1] for j in 0:nr-1)
    end
end

@testset "automatic differentiation passes through" begin
    f(v) = begin
        t = Series(v)
        (exp(t) / (1 + t^2) + sqrt(t + 2) * derivative(t) + X * log(t)).c |> sum
    end
    v = [0.5, 1.0, -0.3, 0.2]
    J = ForwardDiff.gradient(f, v)
    ε = 1e-6
    Jfd = [(f(v .+ ε .* (1:4 .== i)) - f(v .- ε .* (1:4 .== i))) / 2ε for i in 1:4]
    @test J ≈ Jfd rtol = 1e-6
end

@testset "no allocation" begin
    # F and N in the signature force specialization on the function and the argument
    # types; without them Julia (1.10) calls f dynamically and boxes the result.
    function alloc(f::F, args::Vararg{Any,N}) where {F,N}
        f(args...)                                   # compile first
        return @allocated f(args...)
    end
    @test alloc(*, a, b) == 0 && alloc(/, a, b) == 0 && alloc(inv, a) == 0
    @test alloc(^, a, 2.5) == 0 && alloc(exp, a) == 0 && alloc(log, a) == 0
    @test alloc(derivative, a) == 0 && alloc(integrate, a) == 0
    @test alloc(*, X, a) == 0 && alloc(shift, a, 0.7) == 0 && alloc(compose, a, b) == 0
end
