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
