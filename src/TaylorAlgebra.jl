"""
    TaylorAlgebra

Minimal truncated power series (Taylor algebra) in one variable,
    a = Σ_{n=0}^{M} c[n+1] xⁿ,     N = M + 1 coefficients,
stored as an `SVector{N,T}`: the order is part of the type, every operation is
allocation-free and unrolled, and the coefficient type is generic (ForwardDiff
duals pass through, so Newton Jacobians of series-valued residuals come by
automatic differentiation).

Operations: + − * / (series and scalars), inv, ^ (integer, real by Miller's
recurrence, series exponent), sqrt, exp, log, derivative (order −1), integrate
(order +1), s * X^p * a (monomial times a, order +p), evaluate(a, x) = a(x),
compose(a, b) = a ∘ b = a(b), inverse (compositional inverse), shift (re-expansion
around a point), radius (convergence-radius estimate), coefficient, order, pad,
truncate.

Order bookkeeping is strict: series of different orders combine to the smaller
order, the derivative lowers the order (its top coefficient is unknown), X * a
and integrate raise it, and pad / truncate / Series{N}(a) change it explicitly.
This makes a truncation closure explicit: pad the unknowns, and the residual of
a differential equation comes out at the order to which it is actually known.
"""
module TaylorAlgebra

using StaticArrays

export Series, X, Monomial, pad, order, coefficient, derivative, integrate,
       evaluate, compose, inverse, shift, radius


struct Series{N,T}
    c::SVector{N,T}
end
Series(v::AbstractVector{T}) where {T} = Series{length(v),T}(SVector{length(v),T}(v))
Series{N}(v::AbstractVector{T}) where {N,T} = Series{N,T}(SVector{N,T}(ntuple(i -> i ≤ length(v) ? v[i] : zero(T), Val(N))))
Series{N}(s::T) where {N,T<:Number} = Series{N,T}(SVector{N,T}(ntuple(i -> i == 1 ? s : zero(T), Val(N))))
"Change the order explicitly: truncate (N < M) or zero-pad (N > M, exact only for a polynomial)."
Series{N}(a::Series{M,T}) where {N,M,T} = Series{N,T}(SVector{N,T}(ntuple(i -> i ≤ M ? a.c[i] : zero(T), Val(N))))
"""
    pad(a, Val(N)), pad(a, b::Series), pad(a, n::Integer)
    truncate(a, …)   (same forms)

Both change the number of coefficients to N (the length of `b`, or `n`):
zero-padding when N is larger, dropping the top coefficients when N is smaller.
They are the same operation; the two names are there to say in the code which
direction is intended (pad = "taken as an exact polynomial", i.e. a closure;
truncate = "drop what is not needed"). The Integer form builds Val(n) at run
time and dispatches dynamically: fine for setup code, not for hot loops.
"""
@inline pad(a::Series, ::Val{N}) where {N} = Series{N}(a)
@inline pad(a::Series, ::Series{N}) where {N} = Series{N}(a)
pad(a::Series, n::Integer) = Series{n}(a)
Base.truncate(a::Series, x) = pad(a, x)          # extends Base.truncate(io, n)
Base.zero(::Type{Series{N,T}}) where {N,T} = Series{N,T}(zero(SVector{N,T}))
Base.zero(::Series{N,T}) where {N,T} = zero(Series{N,T})
Base.one(::Type{Series{N,T}}) where {N,T} = Series{N}(one(T))
Base.one(::Series{N,T}) where {N,T} = one(Series{N,T})
Base.iszero(a::Series) = iszero(a.c)
Base.:(==)(a::Series{N}, b::Series{N}) where {N} = a.c == b.c
Base.isapprox(a::Series{N}, b::Series{N}; kw...) where {N} = isapprox(a.c, b.c; kw...)
Base.:(==)(a::Series, b::Series) = ==(promote(a, b)...)                 # mixed orders: at the smaller one
Base.isapprox(a::Series, b::Series; kw...) = isapprox(promote(a, b)...; kw...)

order(::Series{N}) where {N} = N - 1
Base.length(::Series{N}) where {N} = N
Base.getindex(a::Series, i) = a.c[i]                      # storage index, 1-based
"coefficient(a, n): the coefficient of xⁿ (n = 0 … order). Equal to a⁽ⁿ⁾(0)/n!, but read directly."
@inline coefficient(a::Series, n::Integer) = a.c[n+1]
Base.eltype(::Series{N,T}) where {N,T} = T

# --- promotion: a Number is a constant series of the same order ---
Base.convert(::Type{Series{N,T}}, s::Number) where {N,T} = Series{N}(T(s))
Base.convert(::Type{Series{N,T}}, a::Series{N}) where {N,T} = Series{N,T}(SVector{N,T}(a.c))
Base.promote_rule(::Type{Series{N,T}}, ::Type{S}) where {N,T,S<:Number} = Series{N,promote_type(T, S)}
Base.promote_rule(::Type{Series{N,T}}, ::Type{Series{N,S}}) where {N,T,S} = Series{N,promote_type(T, S)}
# different orders: the result is only known to the smaller one
Base.promote_rule(::Type{Series{N,T}}, ::Type{Series{M,S}}) where {N,M,T,S} = Series{min(N, M),promote_type(T, S)}
Base.convert(::Type{Series{N,T}}, a::Series{M}) where {N,M,T} = Series{N}(convert(Series{M,T}, a))

@inline Base.:+(a::Series{N}, b::Series{N}) where {N} = Series(a.c + b.c)
@inline Base.:-(a::Series{N}, b::Series{N}) where {N} = Series(a.c - b.c)
@inline Base.:-(a::Series) = Series(-a.c)
for op in (:+, :-, :*, :/)                           # mixed Number/Series and mixed orders through promotion
    @eval @inline Base.$op(a::Series{N}, b::Series{M}) where {N,M} = $op(promote(a, b)...)
end
for op in (:+, :-)
    @eval @inline Base.$op(a::Series, s::Number) = $op(promote(a, s)...)
    @eval @inline Base.$op(s::Number, a::Series) = $op(promote(s, a)...)
end
# scaling by a Number stays O(N) (promoting the scalar would cost a full O(N²) product)
@inline Base.:*(s::Number, a::Series) = Series(s * a.c)
@inline Base.:*(a::Series, s::Number) = s * a
@inline Base.:/(a::Series, s::Number) = Series(a.c / s)

# Truncated Cauchy product, coefficient k = Σ_{i≤k} a_i b_{k+1−i}. One ntuple over
# k (compile-time N) with a generator sum inside: every coefficient is a pure
# expression, no mutable buffer. Measured at N = 25: 10 ns, vs 43 ns for an
# explicit inner loop and 99 ns for an MVector accumulator. Do NOT write the
# inner sum as ntuple(…, Val(k)): k is a run-time value inside the closure, so
# Val(k) dispatches dynamically (≈100× slower, allocates).
@inline function Base.:*(a::Series{N,T}, b::Series{N,S}) where {N,T,S}
    R = promote_type(T, S)
    c = ntuple(k -> (@inbounds sum(a.c[i] * b.c[k+1-i] for i in 1:k)), Val(N))
    return Series{N,R}(SVector{N,R}(c))
end

# Series inverse by the standard recursion. Each coefficient needs the previous
# ones, so a sequential buffer is natural; the MVector lives on the stack (no
# heap allocation) because it does not escape the function.
@inline function Base.inv(a::Series{N,T}) where {N,T}
    m = zero(MVector{N,T})
    @inbounds begin
        m[1] = inv(a.c[1])
        for n in 2:N
            s = zero(T)
            for k in 2:n
                s += a.c[k] * m[n-k+1]
            end
            m[n] = -s * m[1]
        end
    end
    return Series{N,T}(SVector(m))
end

# Fused division: c_n = (a_n − Σ_{k=2}^{n} b_k c_{n+1−k}) / b_1. Same recursion
# cost as one inverse, instead of inverse + product (measured 174 vs 270 ns at N = 25).
@inline function Base.:/(a::Series{N,T}, b::Series{N,S}) where {N,T,S}
    R = promote_type(T, S)
    m = zero(MVector{N,R})
    @inbounds begin
        ib1 = inv(b.c[1])
        m[1] = a.c[1] * ib1
        for n in 2:N                          # same index pattern as inv: provably in bounds,
            s = zero(R)                       # so the buffer stays on the stack even with
            for k in 2:n                      # --check-bounds=yes (Pkg.test, CI)
                s += b.c[k] * m[n-k+1]
            end
            m[n] = (a.c[n] - s) * ib1
        end
    end
    return Series{N,R}(SVector(m))
end
# Small positive integer powers by repeated product: measured faster than binary
# exponentiation and than the Miller recurrence for n ≤ 5 at N ≤ 25, because the
# generator product is very cheap. Negative integers go through Miller.
@inline function Base.:^(a::Series{N,T}, n::Integer) where {N,T}
    n < 0 && return a^float(n)
    n == 0 && return Series{N}(one(T))
    r = a
    for _ in 2:n
        r = r * a
    end
    return r
end

# a^2, a^3 … with a literal exponent: make explicit that they go to the product
# loop above (Base's fallback would call ^(a, 2) anyway; this pins it and keeps
# negative literals, a^-1, on the Miller path with the exponent as a Float).
@inline Base.literal_pow(::typeof(^), a::Series, ::Val{P}) where {P} = P ≥ 0 ? a^P : a^float(P)

# Any real power by J. C. P. Miller's recurrence (needs a_0 ≠ 0):
#   p_0 = a_0^n,   p_m = (1/(m a_0)) Σ_{k=1}^{m} (k n − m + k) a_k p_{m−k}.
# One O(N²) recursion for fractional and negative exponents (e.g. the on-shell
# frequency Ω = (…)^{1/2} of the dynamic sector); a^-3 is ~20% cheaper than inv(a)^3.
@inline function Base.:^(a::Series{N,T}, n::Real) where {N,T}
    R = promote_type(T, typeof(n))
    m = zero(MVector{N,R})
    @inbounds begin
        ia0 = inv(a.c[1])
        m[1] = a.c[1]^n
        for j in 2:N                          # m[j] is the coefficient of x^{j-1}
            s = zero(R)
            for k in 1:j-1
                s += (k * n - (j - 1) + k) * a.c[k+1] * m[j-k]
            end
            m[j] = s * ia0 / (j - 1)
        end
    end
    return Series{N,R}(SVector(m))
end
@inline Base.:^(a::Series, n::Rational) = a^float(n)      # Rational arithmetic in the loop is 4× slower
@inline Base.sqrt(a::Series) = a^0.5
"a^b with a series exponent: exp(b log a), needs a_0 > 0."
@inline Base.:^(a::Series, b::Series) = exp(b * log(a))

# exp and log by the same O(N²) recursions (from b' = a' b and a' = a b'):
#   exp:  b_0 = e^{a_0},   n b_n = Σ_{k=1}^{n} k a_k b_{n−k}
#   log:  b_0 = ln a_0,    n a_0 b_n = n a_n − Σ_{k=1}^{n−1} k b_k a_{n−k}      (a_0 > 0)
@inline function Base.exp(a::Series{N,T}) where {N,T}
    R = typeof(exp(zero(T)))
    m = zero(MVector{N,R})
    @inbounds begin
        m[1] = exp(a.c[1])
        for j in 2:N                          # m[j] is the coefficient of x^{j-1}
            s = zero(R)
            for k in 1:j-1
                s += k * a.c[k+1] * m[j-k]
            end
            m[j] = s / (j - 1)
        end
    end
    return Series{N,R}(SVector(m))
end
@inline function Base.log(a::Series{N,T}) where {N,T}
    R = typeof(log(one(T)))
    m = zero(MVector{N,R})
    @inbounds begin
        ia0 = inv(a.c[1])
        m[1] = log(a.c[1])
        for j in 2:N
            n = j - 1
            s = zero(R)
            for k in 1:n-1
                s += k * m[k+1] * a.c[n-k+1]
            end
            m[j] = (n * a.c[j] - s) * ia0 / n
        end
    end
    return Series{N,R}(SVector(m))
end

# Order bookkeeping: ∂x LOWERS the order by one (the top coefficient of the
# derivative is unknown), X·a RAISES it by one (its top coefficient is a_{N-1},
# exactly known). Mixed orders then promote to the smaller one, so a residual
# built from series of order M comes out at order M−1 and the truncation
# closure (W_{M} = Z_{M} = 0) is the explicit zero-padding of the unknowns, not a
# fake coefficient inserted here.
"derivative(a): ∂x a, N → N−1 coefficients (the top one is unknown)."
@inline derivative(a::Series{N,T}) where {N,T} = Series{N-1,T}(SVector{N-1,T}(ntuple(i -> T(i) * a.c[i+1], Val(N - 1))))
"derivative(a, Val(n)): n-th derivative, N → N−n. The Integer form dispatches dynamically."
@inline derivative(a::Series, ::Val{0}) = a
@inline derivative(a::Series, ::Val{n}) where {n} = derivative(derivative(a), Val(n - 1))
derivative(a::Series, n::Integer) = derivative(a, Val(Int(n)))

"""
    X * a,  a * X,  X^p * a,  X * X,  s * X

The monomial x and its multiples: `Monomial{P}(s)` stands for s·x^P, X = 1·x,
so `4.0 * X` is 4x and `2X^2` is 2x². Multiplying a series by it shifts the
coefficients up by P places and scales them: N → N+P coefficients, exact. Not a
truncated product of two series, so no order is lost. Written so that the code
reads like the formulas, e.g.  4 * X * derivative(a0 * Wx)  for 4x ∂x(a0 w_x).

The exponent must be known at compile time, since it sets the result's order.
A literal exponent (`X^2`) is, because Julia lowers it to literal_pow with Val(2);
so is `X * X` and `X^Val(p)`. `X^p` with a run-time integer p dispatches
dynamically (fine for setup, not in hot code).
"""
struct Monomial{P,S<:Number}
    s::S
end
Monomial{P}(s::S) where {P,S<:Number} = Monomial{P,S}(s)
const X = Monomial{1}(1)
@inline function Base.:*(m::Monomial{P,S}, a::Series{N,T}) where {P,S,N,T}
    R = promote_type(S, T)
    return Series{N+P,R}(SVector{N+P,R}(ntuple(i -> i > P ? R(m.s * a.c[i-P]) : zero(R), Val(N + P))))
end
@inline Base.:*(a::Series, m::Monomial) = m * a
@inline Base.:*(s::Number, m::Monomial{P}) where {P} = Monomial{P}(s * m.s)
@inline Base.:*(m::Monomial, s::Number) = s * m
@inline Base.:/(m::Monomial{P}, s::Number) where {P} = Monomial{P}(m.s / s)
@inline Base.:-(m::Monomial{P}) where {P} = Monomial{P}(-m.s)
@inline Base.:*(m::Monomial{P}, n::Monomial{Q}) where {P,Q} = Monomial{P+Q}(m.s * n.s)
@inline Base.literal_pow(::typeof(^), m::Monomial{P}, ::Val{Q}) where {P,Q} = Monomial{P*Q}(m.s^Q)
@inline Base.:^(m::Monomial{P}, ::Val{Q}) where {P,Q} = Monomial{P*Q}(m.s^Q)
Base.:^(m::Monomial, q::Integer) = m^Val(Int(q))
Base.show(io::IO, m::Monomial{P}) where {P} = print(io, m.s, " X", P == 1 ? "" : "^$P", "  (monomial ", m.s, "·x", P == 1 ? "" : "^$P", ")")
# A monomial also converts to a series once an order is given (coefficient s at
# position P+1, dropped if P exceeds the order): a + 2X, X^2 - a go through
# promotion like Numbers do. Only + and −: for * the shift above is exact and
# O(N), a promoted product would be a truncated O(N²) Cauchy product.
Series{N}(m::Monomial{P,S}) where {N,P,S} = Series{N,S}(SVector{N,S}(ntuple(i -> i == P + 1 ? m.s : zero(S), Val(N))))
Base.convert(::Type{Series{N,T}}, m::Monomial{P}) where {N,T,P} = Series{N}(Monomial{P}(T(m.s)))
Base.promote_rule(::Type{Series{N,T}}, ::Type{Monomial{P,S}}) where {N,T,P,S} = Series{N,promote_type(T, S)}
for op in (:+, :-)
    @eval @inline Base.$op(a::Series, m::Monomial) = $op(promote(a, m)...)
    @eval @inline Base.$op(m::Monomial, a::Series) = $op(promote(m, a)...)
end
"integrate(a, c0 = 0): ∫₀ˣ a dx' + c0, N → N+1 coefficients (exact); inverse of derivative up to the constant."
@inline function integrate(a::Series{N,T}, c0::Number=zero(T)) where {N,T}
    R = promote_type(T, typeof(c0))
    return Series{N+1,R}(SVector{N+1,R}(ntuple(i -> i == 1 ? R(c0) : R(a.c[i-1]) / (i - 1), Val(N + 1))))
end

"""
    inverse(a)

The inverse function (compositional inverse, series reversion): b with
a(b(y)) = y, for a_0 = 0 and a_1 ≠ 0; order one less than a. Not to be confused
with inv(a) = 1/a. Same name as in TaylorSeries.jl. Fixed-point iteration b ← (y − (a − a_1 x)(b)) / a_1, which fixes one more
coefficient per step, so N − 1 steps are exact; O(N⁴) but cheap at these sizes.
"""
function inverse(a::Series{N,T}) where {N,T}
    iszero(a.c[1]) || throw(ArgumentError("inverse needs a_0 = 0"))
    R = typeof(inv(one(T)))
    y = X * one(Series{N-1,R}); ia1 = inv(a.c[2])
    rest = a - a.c[2] * y                                  # a_2 x² + a_3 x³ + …
    b = ia1 * y
    for _ in 1:N-2
        b = ia1 * (y - compose(rest, b))
    end
    return b
end

"Evaluate the series at a point (Horner)."
@inline function evaluate(a::Series{N}, x) where {N}
    v = a.c[N]
    @inbounds for n in N-1:-1:1
        v = v * x + a.c[n]
    end
    return v
end
"""
    compose(a, b)

Composition a(b(y)) as a series in y, truncated to the smaller order. Horner's
scheme with series arithmetic, O(N³); b_0 need not vanish (the series is a
polynomial, so no convergence question arises at this level).
"""
@inline function compose(a::Series{N}, b::Series{M,S}) where {N,M,S}
    v = Series{M}(a.c[N])
    @inbounds for n in N-1:-1:1
        v = v * b + a.c[n]
    end
    return v
end
# A series is callable: a(x) evaluates, a(b) composes; and a ∘ b composes.
(a::Series)(x::Number) = evaluate(a, x)
(a::Series)(b::Series) = compose(a, b)
Base.:∘(a::Series, b::Series) = compose(a, b)
# the step is a separate function: a closure capturing a variable that is
# reassigned in the loop (v) would box it.
@inline _shift_step(v::SVector{N,R}, κ, an) where {N,R} =
    SVector{N,R}(ntuple(i -> κ * v[i] + (i > 1 ? v[i-1] : zero(R)) + (i == 1 ? R(an) : zero(R)), Val(N)))
"""
    shift(a, κ)

The same polynomial re-expanded around x = κ: returns b with
a(x) = b(x − κ), i.e. b_n = Σ_{m≥n} C(m,n) κ^{m−n} a_m. Exact (same order),
O(N²): Horner's form b = a(κ + y), one step v ← κ·v + y·v + a_n per coefficient,
each step a single unrolled ntuple on the SVector. No mutable buffer: an MVector
that a bounds-error path could capture is moved to the heap under
--check-bounds=yes (Pkg.test, CI), while this stays in registers.
`shift(a, κ)` evaluated at y = −κ gives back a(0).
"""
@inline function shift(a::Series{N,T}, κ::Number) where {N,T}
    R = promote_type(T, typeof(κ))
    v = SVector{N,R}(ntuple(i -> i == 1 ? R(a.c[N]) : zero(R), Val(N)))
    for n in N-1:-1:1
        v = _shift_step(v, κ, a.c[n])
    end
    return Series{N,R}(v)
end

"""
    radius(a; nlast = 4, method = :ratio) -> (r, s)

Estimate of the convergence radius of the function the series represents, from
the growth of its top coefficients.

`method = :ratio` (Domb–Sykes plot): the ratios a_n/a_{n−1} behave like
(1/x_s)(1 + (α−1)/n) near a singularity x_s of strength α, so a straight-line
fit of the last `nlast` ratios against 1/n has intercept 1/x_s. Returns
r = |x_s| and s = sign(x_s): s = −1 means the nearest singularity is on the
negative axis (alternating coefficients). Accurate for a single real
singularity; meaningless when the signs follow a longer pattern (complex
pair), or when a coefficient vanishes (NaN).

`method = :root` (Cauchy–Hadamard): r = |a_n|^{−1/n} averaged over the last
`nlast` coefficients, s = 0. Robust to the sign pattern, slowly convergent
(O(log n / n) bias): a rough but safe estimate.

Meaningful only when the top coefficients are converged, i.e. not for the few
highest ones of a closed hierarchy; use on a series one or two orders below the
closure.
"""
function radius(a::Series{N}; nlast::Integer=4, method::Symbol=:ratio) where {N}
    nlast = min(nlast, N - 2)
    nlast ≥ 2 || throw(ArgumentError("need at least 4 coefficients"))
    if method === :root
        rs = [abs(a.c[n+1])^(-1 / n) for n in N-nlast:N-1]
        return sum(rs) / length(rs), 0
    end
    method === :ratio || throw(ArgumentError("method must be :ratio or :root"))
    ns = N-nlast:N-1                               # ratio index n (coefficient of x^n over x^{n-1})
    u = [1 / n for n in ns]
    q = [a.c[n+1] / a.c[n] for n in ns]
    ū = sum(u) / length(u); q̄ = sum(q) / length(q)
    slope = sum((u .- ū) .* (q .- q̄)) / sum((u .- ū) .^ 2)
    intercept = q̄ - slope * ū
    return 1 / abs(intercept), sign(intercept)
end

Base.show(io::IO, a::Series) = print(io, "Series of order ", order(a), ": ", collect(a.c))

end # module
