using Statistics
using DataFrames

NAMES = Dict(:tm=>"TwoMoons", :crc=>"Circles", :sph=>"Sphere", :oip15=>"OIP15", :oip300=>"OIP300")

include("src/plotting.jl")
include("src/results.jl")

#-----------------------------

dss = sort!(collect(values(NAMES)))
parts = Dict(:Partition=>"spa-rng-part", :Generator=>"spa-rng-gen")
results = DataFrame(Method = String[], RelDominance=Union{Float64,Missing}[],
                    Cells=Union{Int,Missing}[], Dataset=String[], Component=Symbol[])
# dsn = first(dss)
for (sym, flt) in parts
    for (j,dsn) in enumerate(dss)
        RelDom = getResults(dsn, "RelDom", filter=flt, null=missing)
        length(RelDom) == 0 && continue
        mthdIdx = Dict( first(k)=>i for (i,k) in enumerate(keys(RelDom)) )
        Cells = getResults(dsn, "Cells", filter=flt, null=missing)
        for (rd,cl) in zip(RelDom, Cells)
            m = first(rd)[1]
            for (v1,v2) in zip(last(rd), last(cl))
                push!(results, (m, v1, v2, dsn, sym))
            end
        end
    end
end

#-----------------------------

nparts = collect(keys(parts))
mthds = unique(results[!,:Method])
dfs = Dict(m => dropmissing(results[results.Method .== m, :], :RelDominance) for m in mthds)
plots = []
for (i,mthd) in enumerate(mthds)
    df = [dfs[mthd][dfs[mthd].Component .== part, :] for part in nparts]
    lbl = i == 1 ? "Relative Dominance" : ""
    lgnd = i == 1 ? (0.8,0.1) : :none
    p = @df df[1] violin(:Dataset, :RelDominance, side=:right, marker=(0.2, :blue, stroke(0)), title=mthd, ylabel=lbl, label=nparts[1], legend=lgnd);
    @df df[2] violin!(p, :Dataset, :RelDominance, side=:left, marker=(0.2, :red, stroke(0)), label=nparts[2]);
    push!(plots, p)
end
p = plot(plots..., layout=@layout [a b]);
global PLOT_FORMAT = :pdf
saveplot("../gen/spa-rng.pdf", p)


#-----------------------------

using HypothesisTests
using DataFramesMeta

#http://www.real-statistics.com/one-way-analysis-of-variance-anova/homogeneity-variances/dealing-with-heterogeneous-variances/
correctvar(x) = log.(x)
# correctvar(x) = log10.(x)
# correctvar(x) = log.(x .+ 0.1)
# correctvar(x) = sqrt.(x)
# correctvar(x) = sqrt.(x .+ 0.5)
# correctvar(x) = sqrt.(x) .+ sqrt.(x .+ 1)

rdtest = DataFrame(Method = String[], Dataset=String[], Pvalue=Float64[])
celltest = DataFrame(Method = String[], Dataset=String[], Pvalue=Float64[])
cleandata = collect ∘ skipmissing
for ds in dss
    for mthd in mthds
        res = [
            (@linq dfs[mthd] |>
             where(:Dataset .== ds, :Component .== part) |>
             select(:RelDominance, :Cells) |>
             transform( x = correctvar(:RelDominance) ))
            for part in nparts
        ]
        size(res[1],1) == 0 && continue
        # size.(res) |> println
        # extrema.(res) |> println
        t = FlignerKilleenTest(res[1][!,:x], res[2][!,:x])
        pv = round(pvalue(t), digits=3)
        push!(rdtest, (mthd, ds, pv))
        outcome = if pv > 0.05 "fail to reject" else "reject" end
        println(ds, ", ", mthd, ": $pv ($outcome)")
        t = FlignerKilleenTest(cleandata(res[1][!,:Cells]), cleandata(res[2][!,:Cells]))
        pv = round(pvalue(t), digits=3)
        push!(celltest, (mthd, ds, pv))
    end
end

open("../gen/spa-rng-rd-table.tex", "w") do io
    tmpdf = unstack(rdtest, :Method, :Pvalue)
    show(io, MIME("text/latex"), tmpdf, eltypes=false)
end
open("../gen/spa-rng-cell-table.tex", "w") do io
    tmpdf = unstack(celltest, :Method, :Pvalue)
    show(io, MIME("text/latex"), tmpdf, eltypes=false)
end

#-----------------------------
