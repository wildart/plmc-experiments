using BSON, Random

include("src/plotting-geo.jl")
# pygui(true)

import Base: convert
using Statistics, Clustering, LMCLUS, ClusterComplex

using Logging
# lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Info);
lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Debug);

# raw = BSON.load("data/CLIM_1951_1980_1x1.bson")
BSON.@load "data/CLIM_1951_1980_1x1.bson" NAMES UNIT MASK

# using DataFrames
# CLIM = DataFrame(UNIT, map(Symbol, NAMES))

seed = 923487298

# Original
println("LMCLUS")

p = LMCLUS.Parameters(11)
p.random_seed = seed
p.best_bound = 0.4
p.number_of_clusters = 34
p.sampling_factor = 0.1
p.min_cluster_size = 150
# p.sampling_heuristic = 2
p.basis_alignment = false
p.bounded_cluster = false
p.mdl = false

C = lmclus(UNIT', p)
# C = with_logger(lgr) do
#     lmclus(convert(Matrix{Float64}, UNIT'), p)
# end
saveplot("../gen/clim-lmclus.pdf", mapclusters(C, MASK))


# MDL
println("LMCLUS-MDL")
p.basis_alignment = true
p.bounded_cluster = true
p.mdl = true
# C2 = lmclus(UNIT', p)
# saveplot("../gen/clim-lmcmdl.pdf", mapclusters(C2, MASK))

p.best_bound = 0.35
p.mdl_compres_ratio = 2.44 #1.95 # 2.4
p.mdl_quant_error = 1e-3 # 1e-3
C3 = lmclus(UNIT', p)
# nclusters(C3)
saveplot("../gen/clim-lmcmdl.pdf", mapclusters(C3, MASK))

# C2 = with_logger(lgr) do
#     lmclus(convert(Matrix{Float64}, UNIT'), p)
# end
# disable_logging(Logging.BelowMinLevel)
C3 = open("lmclus.log", "w") do io
    lgr = Logging.ConsoleLogger(io, Base.CoreLogging.Debug);
    # lgr = Logging.ConsoleLogger(io, LMCLUS.TRACE)
    with_logger(lgr) do
        lmclus(UNIT', p)
    end
end

# k-Means
println("k-Means")
k = 34 #nclusters(C)
Random.seed!(seed);
C4 = kmeans(UNIT', k)
saveplot("../gen/clim-kmeans.pdf", mapclusters(C4, MASK))


# ORCLUS
println("ORCLUS")
maxoutdim = 1 # maximum dimension of subspace
using RCall
@rlibrary orclus

z = orclus(UNIT, k, maxoutdim, k*3);
C5 = CustomClusteringResult(convert(Vector{Int}, z[:cluster]), convert(Matrix{Float32}, convert(Matrix, z[:centers])'))
saveplot("../gen/clim-orclus.pdf", mapclusters(C5, MASK))


#---------------------------
# Compression
#
res = Dict()
res[:LMCLUS] = C
res[:LMCMDL]  = C3
res[:kMeans] = LMCLUSResult(C4)
res[:ORCLUS] = LMCLUSResult(C5, maxoutdim)

algs = keys(res) |> collect |> sort
ɛs = [10.0^(-i) for i in 1:7]
ds = -1:1:6

using DataFrames, StatsPlots, DataFramesMeta
results = DataFrame(a=Symbol[], d=Int[], ɛ=Float64[], MDL=Float64[], CR=Float64[])
raw = 32 * length(UNIT)
for a in algs, ɛ in ɛs, d in ds
    # mdltype = LMCLUS.MDL.SizeIndependent
    mdltype = LMCLUS.MDL.OptimalQuant
    mdl = LMCLUS.MDL.calculate!(LMCLUS.MDL.OptimalQuant, res[a], UNIT', 32, 16, ɛ=ɛ, d=d)
    push!(results, (a, d, ɛ, mdl, raw/mdl))
end

# make tables
crates = unstack(@where(results[!,Not(:MDL)], :d .== 1), :a, :CR)
open("../gen/clim-crates.tex", "w") do io
    println(io, latexify(crates[!,Not(:d)]; env=:table, latex=false, fmt="%.4g"))
end

mdims = DataFrame(a=algs,
                  md=[map(outdim, res[a].manifolds) |> mean for a in algs],
                  cl=[nclusters(res[a]) for a in algs])
mdims = unstack(stack(mdims, [:cl, :md], variable_name=:desc), :a, :value)
mdims[1,:desc] = "Clustering Size"
mdims[2,:desc] = "Avg. Dimension"
open("../gen/clim-mfld-avgdim.tex", "w") do io
    println(io, latexify(mdims; env=:table, latex=false, fmt="%.3f"))
end

# fltr = @where(results, :d .>= 0, :ɛ .== 1e-3)
# @df fltr StatsPlots.plot(:d, :MDL, group=:a, leg=:topleft, xticks=ds, ylab="MDL", xlab="d", yscale=:log2)
# @df fltr StatsPlots.plot(:d, :CR, group=:a, leg=:topright, xticks=ds, ylab="CR", xlab="d")

fltr = @where(results, :d .== 1, :ɛ .>= 1e-5, :ɛ .<= 1e-1)
p = @df fltr StatsPlots.plot(:ɛ, :CR, group=:a, xscale=:log10, leg=:bottomright, ylab="CR", xlab="ɛ")
lens!(p, [5e-4, 6e-4], [2.5, 2.64], inset = (1, bbox(0.1, 0.1, 0.5, 0.5)))
StatsPlots.Plots.pdf(p, "../gen/clim-crates.pdf")

# fltr = @where(results, :d .== 0, :ɛ .>= 1e-5, :ɛ .<= 1e-1)
# p = @df fltr StatsPlots.plot(:ɛ, :CR, group=:a, xscale=:log10, leg=:bottomright, ylab="CR", xlab="ɛ")
# lens!(p, [2.5e-4, 4e-4], [2.5, 2.64], inset = (1, bbox(0.1, 0.1, 0.5, 0.5)))

# z = unstack(@where(results[!,Not(:MDL)], :d .== 1), :a, :CR)[!,Not([:d, :ɛ])] |> dropmissing |> Matrix
# z = unstack(@where(results[!,Not(:CR)], :d .== 1), :a, :MDL)[!,Not([:d, :ɛ])] |> dropmissing |> Matrix

# lms = @where(results[!,Not(:CR)], :d .> 0)[!, :MDL] |> extrema
# i = 4
# z = unstack(@where(results[!,Not(:CR)],  :d .> 0, :a .== algs[i]), :d, :MDL)[!,Not([:a, :ɛ])] |> dropmissing |> Matrix
# heatmap(z, xlab="d", ylab="ɛ", title="$(algs[i])", zlim=lms)
