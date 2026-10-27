# BenchmarkTools suite for TaylorAlgebra. Used by run.jl (regression check against
# a saved baseline) and usable with PkgBenchmark (`benchmarkpkg`, `judge`).
using BenchmarkTools, TaylorAlgebra, ForwardDiff

const SUITE = BenchmarkGroup()

for N in (8, 16, 32)
    a = Series(rand(N) .+ 0.5); b = Series(rand(N) .+ 0.5)
    g = SUITE["N=$N"] = BenchmarkGroup()
    g["add"]        = @benchmarkable $a + $b
    g["mul"]        = @benchmarkable $a * $b
    g["div"]        = @benchmarkable $a / $b
    g["inv"]        = @benchmarkable inv($a)
    g["pow_int3"]   = @benchmarkable $a^3
    g["pow_real"]   = @benchmarkable $a^2.5
    g["sqrt"]       = @benchmarkable sqrt($a)
    g["exp"]        = @benchmarkable exp($a)
    g["log"]        = @benchmarkable log($a)
    g["derivative"] = @benchmarkable derivative($a)
    g["integrate"]  = @benchmarkable integrate($a)
    g["monomial"]   = @benchmarkable 4X * $a
    g["evaluate"]   = @benchmarkable $a(0.3)
    g["compose"]    = @benchmarkable compose($a, $b)
    g["shift"]      = @benchmarkable shift($a, 0.7)
end

# A residual of the kind the package was written for: nonlinear expression with
# derivatives and a monomial, in Float64 and with ForwardDiff duals (Jacobian).
function residual(y, ::Val{M}=Val(12)) where {M}     # M static: Series{M}(y) is a compile-time size
    W = pad(Series{M}(y), Val(M + 1))
    Wx = derivative(W); Wxx = derivative(Wx)
    a0 = inv((W + 1)^2)
    return (2.0 * (a0 * Wx) + 4X * derivative(a0 * Wx) + 2W - 0.5 * X * Wx).c
end
let y = 0.1 .* rand(12) .- 0.15
    SUITE["residual"] = BenchmarkGroup()
    SUITE["residual"]["float"]    = @benchmarkable residual($y)
    SUITE["residual"]["jacobian"] = @benchmarkable ForwardDiff.jacobian(residual, $y)
end
