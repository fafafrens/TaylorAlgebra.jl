# Run the suite and compare with the saved baseline.
#   julia --project=benchmark benchmark/run.jl            # compare (and create baseline if missing)
#   julia --project=benchmark benchmark/run.jl --save     # overwrite the baseline with this run
# Reports min time per benchmark, the ratio to the baseline, and flags ratios above
# THRESHOLD as regressions (exit code 1 so it can gate a CI job if wanted).
using BenchmarkTools, JSON, Printf
include(joinpath(@__DIR__, "benchmarks.jl"))

const BASELINE  = joinpath(@__DIR__, "baseline.json")
const THRESHOLD = 1.5

tune!(SUITE)
res = run(SUITE; verbose=false)
flat = Dict{String,Float64}(); allocs = Dict{String,Int}()     # min time in ns, allocations
for (g, grp) in res, (k, t) in grp
    flat["$g/$k"] = minimum(t).time
    allocs["$g/$k"] = minimum(t).allocs
end

if "--save" in ARGS || !isfile(BASELINE)
    open(BASELINE, "w") do io
        JSON.print(io, Dict("julia" => string(VERSION), "times_ns" => flat), 2)
    end
    println("baseline written to $BASELINE")
end
base = JSON.parsefile(BASELINE)["times_ns"]

println(rpad("benchmark", 24), lpad("min [ns]", 12), lpad("baseline", 12), lpad("ratio", 8), "  allocs")
regressions = String[]
for k in sort(collect(keys(flat)))
    t = flat[k]; b = get(base, k, NaN); r = t / b
    flag = r > THRESHOLD ? "  <-- REGRESSION" : ""
    r > THRESHOLD && push!(regressions, k)
    @printf("%-24s%12.1f%12.1f%8.2f  %d%s\n", k, t, b, r, allocs[k], flag)
end
any(>(0), values(allocs)) && println("\nnote: allocating benchmarks: ", join([k for (k, v) in allocs if v > 0], ", "))
if !isempty(regressions)
    println("\n", length(regressions), " regression(s) above ", THRESHOLD, "x: ", join(regressions, ", "))
    exit(1)
end
println("\nno regression above ", THRESHOLD, "x")
