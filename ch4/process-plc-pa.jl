using Statistics
using DataFrames
using DataFramesMeta
using Measurements

PARAMS = Dict(
# for datasets
:seed => 837387533,
:m => 500,
:c => 2,
:ε => 0.15, #0.05-0.25
:r => 1,
:d => 2,
:repulse => (-0.25, -0.25),
:t => [0.0; 0.0],
# for k-Means
:k => 20,
# for PLMC
:ϵ => 4.0,
:iter => 100, # sample size
)
NAMES = Dict(:tm=>"TwoMoons", :crc=>"Circles", :sph=>"Sphere")

include("src/plotting.jl")
include("src/results.jl")

# global PLOT_FORMAT = :pdf

#-----------------------------
# Plot datasets
#
using NMoons

ps = []
#εs = [0.10, 0.20, 0.25, 0.3, 0.35, 0.4]
εs = [0.10, 0.20, 0.3, 0.4]
ms = [250, 500, 1000, 1500]
for (i,m) in enumerate(ms)
    for (j,ε) in enumerate(εs)
        X, L = nmoons(Float64, m, PARAMS[:c], ε=ε, r=PARAMS[:r], d=PARAMS[:d],
                      repulse=PARAMS[:repulse], translation=PARAMS[:t], seed=PARAMS[:seed])
        t = i == 1 ? "ε=$ε" : ""
        yl = j == 1 ? "m=$m" : ""
        push!(ps, scatter(X[1,:], X[2,:], legend=:none, ms=1.0, c=:black, showaxis=false, title=t, ylabel=yl));
    end
end
p = plot(ps..., layout=(length(ms),length(εs)), size=(800,600), margin=0mm);
saveplot("../gen/pa-data.png", p)
#-----------------------------

dss = sort!(collect(values(NAMES)))
n = length(dss)
results = DataFrame(Method = String[], RelDominance=Union{Float64,Missing}[],
                    Cells=Union{Int,Missing}[], Clusters=Int[],
                    Noise=Float64[], Size=Int[], Dataset=String[])
# dsn = first(dss)
flt="-pa-"
params = getParams(flt)
for (j,dsn) in enumerate(dss)
    RelDom = getResults(dsn, "RelDom", filter=flt, null=missing)
    length(RelDom) == 0 && continue
    Cells = getResults(dsn, "Cells", filter=flt, null=missing)
    Clusters = getResults(dsn, "Clusters", filter=flt, null=missing)
    for (rd,c,cl) in zip(RelDom, Cells, Clusters)
        k = first(rd)
        p = params[k]
        m = k[1]
        for (v1,v2, v3) in zip(last(rd), last(c), last(cl))
            push!(results, (m, v1, v2, v3, p["ε"], p["m"], dsn))
        end
    end
end
# remove VR & WTN
# results = results[map(n -> startswith(n, "PLM"), results.Method),:]


mthds = Dict( k=>n for (n,k) in zip(unique(results[!,:Method]) |> sort!, ["const", "prop", "dyn", "vr", "wtn"]))
εs = unique(results[!,:Noise]) |> sort!
ms = unique(results[!,:Size]) |> sort!

#-----------------------------
# mk, mthd = collect(mthds)[4]
# i, m = 1, collect(ms)[1]
# ε = first(εs)
for (mk, mthd) in mthds
println(mthd)
ps = []
for (i,m) in enumerate(ms)
    tmpdf = @linq results |> where(:Method .== mthd, :Size .== m)
    size(tmpdf)[1] == 0 && continue
    # pldata = dropmissing(tmpdf)
    pldata = coalesce.(tmpdf, 0)
    yl = isodd(i) ? "Relative Dominance" : ""
    p = @df pldata groupedboxplot(:Noise, :RelDominance, group=:Dataset, size=(800,600), markersize=3,
                                title="m=$m", xlabel="Noise, ε", ylabel=yl, bar_width=0.0075, legend=:topright);
    gdf = groupby(pldata, [:Dataset, :Noise])
    linedf = combine(gdf, :RelDominance => median => :RelDominance)
    lc = map(last, filter(i->isodd(first(i)), map(p -> (p[1],p[2][:fillcolor]), enumerate(p.series_list)))) |> permutedims
    @df linedf plot!(p, :Noise, :RelDominance, group=:Dataset, seriestype=:line, seriescolor=lc, label="");
    push!(ps, p)
end
p = plot(ps..., leg=false);

# legend
lc = map(last, filter(i->isodd(first(i)), map(p -> (p[1],p[2][:fillcolor]), enumerate(ps[1].series_list))))[1:n];
xs, ys =(1:n)*0.50, fill(0.5,n)
ann = [(x+0.075,y,Plots.text(nm, c, :left, pointsize=10)) for (x,y,c,nm) in zip(xs, ys, lc, dss)]
scatter!(p, xs, ys, marker=6, ylims=(0,1.0), xlims=(minimum(xs)-0.05, maximum(xs)+0.50),
        inset = bbox(0.03, length(ps) % 4 == 0 ? 0.0 : -0.35, 200pt, 50pt, :center), subplot = length(ps)+1,
        c=lc, leg=false, annotations=ann, framestyle=:none);

saveplot("../gen/pa-$mk.pdf", p)
end
#-----------------------------

#-----------------------------
# mk, mthd = collect(mthds)[3]
# i, m = 1, collect(ms)[1]
# ε = first(εs)
tmpdf = @linq results |> where(:Size .== maximum(ms)) |> select(:Dataset, :RelDominance)
rdmax = combine(groupby(coalesce.(tmpdf, 0), :Dataset), :RelDominance => maximum => :RDM)
for (dsk, dsn) in NAMES
# dsk, dsn = collect(NAMES)[2]
ylim = ceil(rdmax[rdmax.Dataset .== dsn,:RDM][], digits=2)
ps = []
for (i,m) in enumerate(ms)
    tmpdf = @linq results |> where(:Dataset .== dsn, :Size .== m)
    # pldata = dropmissing(tmpdf)
    pldata = coalesce.(tmpdf, 0)
    yl = isodd(i) ? "Relative Dominance" : ""
    p = @df pldata groupedboxplot(:Noise, :RelDominance, group=:Method, size=(800,600), markersize=3,
                                  title="m=$m", ylims=(0,ylim),
                                  xlabel="Noise, ε", ylabel=yl, bar_width=0.0075, legend=:topright);
    gdf = groupby(pldata, [:Method, :Noise])
    linedf = combine(gdf, :RelDominance => median => :RelDominance)
    lc = map(last, filter(i->isodd(first(i)), map(p -> (p[1],p[2][:fillcolor]), enumerate(p.series_list)))) |> permutedims
    @df linedf plot!(p, :Noise, :RelDominance, group=:Method, seriestype=:line, seriescolor=lc, label="");
    push!(ps, p)
end
p = plot(ps..., leg=false);

# legend
m = length(mthds)
mh = m >> 1
lc = map(last, filter(i->isodd(first(i)), map(p -> (p[1],p[2][:fillcolor]), enumerate(ps[1].series_list))))[1:m];
xs, ys = [fill(-0.2,m-mh); fill(0.5,mh)], [(1:(m-mh))*0.2; (1:mh)*0.2]
ann = [(x+0.05,y,Plots.text(nm, c, :left, pointsize=9)) for (x,y,c,nm) in zip(xs, ys, lc, sort!(collect(values(mthds))))]
scatter!(p, xs, ys, marker=5, ylims=(0,1.0), xlims=(minimum(xs)-0.05, maximum(xs)+0.50),
        inset = bbox(0.05, -0.01, 200pt, 40pt, :center), subplot = 5,
        c=lc, leg=false, annotations=ann, framestyle=:none);

saveplot("../gen/pa-$dsk.pdf", p)
end
#-----------------------------

#-----------------------------
# Clustering sizes (by size)
#
gdf = groupby(results[map(n -> startswith(n, "PLM"), results.Method),:], [:Method, :Dataset, :Size])
# gdf = groupby(results, [:Method, :Dataset, :Size])
stats = sort(combine(gdf, :Clusters => (x->measurement(median(x),mad(x))) => :Clusters, :Cells => (x->measurement(median(x),mad(x))) => :Cells), [:Method, :Dataset, :Size])
ymcl = findmax(map(v->Measurements.value(v)+Measurements.uncertainty(v), stats[!,:Clusters])) |> first |> ceil
ymcc = findmax(map(v->Measurements.value(v)+Measurements.uncertainty(v), stats[!,:Cells])) |> first |> ceil

let p = nothing
    gres = groupby(stats, [:Method])
    p = @df gres[3] plot(:Size, :Clusters, group=:Dataset, seriestype=:line, ylims=(0,ymcl), leg=:topleft, xlab="Size, m", ylab="# of Clusters");
    @df @where(gres[1], :Dataset .== "Sphere") plot!(p, :Size, :Clusters, seriestype=:line, label="Constant", c=:black);
    @df @where(gres[2], :Dataset .== "Sphere") plot!(p, :Size, :Clusters, seriestype=:line, line=(:dash, 1.5), label="Proportionate", c=:black);
    # p
    saveplot("../gen/pa-clusters-size.pdf", p)
end

#
# Number of cells per dataset (by size)
#
gSizes = groupby(stats, [:Dataset])
# gs = first(gSizes)
# ltyp = :solid
let p = nothing, lc = nothing
    for (ltyp, gk) in zip([:solid, :dash, :dashdot], keys(gSizes))
        # global p, lc
        ds = first(values(gk))
        res = transform(gSizes[gk], :Method => (x -> ds*",".*chop.(x, head=5, tail=0)) => :Desc)
        if ltyp == :solid
            p = @df res plot(:Size, :Cells, group=:Desc, seriestype=:line, line=(ltyp,2), ylims=(0,ymcc),
                            xlims=(200, 1550), leg=:topleft, xlab="Size, m", ylab="# of Cells")
            lc = map(last, filter(i->isodd(first(i)), map(p -> (p[1],p[2][:fillcolor]), enumerate(p.series_list)))) |> permutedims
        else
            @df res plot!(p, :Size, :Cells, group=:Desc, seriestype=:line, line=(ltyp,2), seriescolor=lc)
        end
    end
    # p
    saveplot("../gen/pa-cells-size.pdf", p)
end

#-----------------------------
# Clustering sizes (by noise)

gdf = groupby(results[map(n -> startswith(n, "PLM"), results.Method),:], [:Method, :Dataset, :Noise])
stats = sort(combine(gdf, :Clusters => (x->measurement(median(x),mad(x))) => :Clusters, :Cells => (x->measurement(median(x),mad(x))) => :Cells), [:Method, :Dataset, :Noise])
ymcl = findmax(map(v->Measurements.value(v)+Measurements.uncertainty(v), stats[!,:Clusters])) |> first |> ceil
ymcc = findmax(map(v->Measurements.value(v)+Measurements.uncertainty(v), stats[!,:Cells])) |> first |> ceil

let p = nothing
    ym = findmax(map(v->Measurements.value(v)+Measurements.uncertainty(v), stats[!,:Clusters])) |> first |> ceil
    gres = groupby(stats, [:Method])
    p = @df gres[3] plot(:Noise, :Clusters, group=:Dataset, seriestype=:line, ylims=(0,ymcl), leg=:topright, xlab="Noise, ε", ylab="# of Clusters");
    @df @where(gres[1], :Dataset .== "Sphere") plot!(p, :Noise, :Clusters, seriestype=:line, label="Constant", c=:black);
    @df @where(gres[2], :Dataset .== "Sphere") plot!(p, :Noise, :Clusters, seriestype=:line, line=(:dash, 1.5), label="Proportionate", c=:black);
    # p
    saveplot("../gen/pa-clusters-noise.pdf", p)
end

gSizes = groupby(stats, [:Dataset])
let p = nothing, lc = nothing
    for (ltyp, gk) in zip([:solid, :dash, :dashdot], keys(gSizes))
        ds = first(values(gk))
        res = transform(gSizes[gk], :Method => (x -> ds*",".*chop.(x, head=5, tail=0)) => :Desc)
        if ltyp == :solid
            p = @df res plot(:Noise, :Cells, group=:Desc, seriestype=:line, line=(ltyp,2), ylims=(0,ymcc),
                            leg=:topright, xlab="Noise, ε", ylab="# of Cells")
            lc = map(last, filter(i->isodd(first(i)), map(p -> (p[1],p[2][:fillcolor]), enumerate(p.series_list)))) |> permutedims
        else
            @df res plot!(p, :Noise, :Cells, group=:Desc, seriestype=:line, line=(ltyp,2), seriescolor=lc)
        end
    end
    # p
    saveplot("../gen/pa-cells-noise.pdf", p)
end



#-----------------------------
ε = 0.2
resnoise = @linq results[map(n -> startswith(n, "PLM"), results.Method),:] |> where(:Noise .== ε)
gdf = groupby(results[map(n -> startswith(n, "PLM"), results.Method),:], [:Method, :Dataset, :Size])
stats = sort(combine(gdf, :Clusters => (x->measurement(median(x),mad(x))) => :Clusters, :Cells => (x->measurement(median(x),mad(x))) => :Cells), [:Method, :Dataset, :Size])
clusts = sort(unstack(stats[!,Not(:Cells)], :Dataset, :Clusters), [:Size, :Method])
cells = sort(unstack(stats[!,Not(:Clusters)], :Dataset, :Cells), [:Size, :Method])
insertcols!(clusts, 1, :Id => 1:size(clusts,1))
insertcols!(cells, 1, :Id => 1:size(cells,1))
formated = innerjoin(clusts, cells, on = :Id, makeunique=true)


f = open("../gen/pa-clusters-table.tex", "w")
# f = stdout

res = """\\begin{tabular}{llll}
% header 1
    & \\multicolumn{3}{|c|}{\\textbf{Dataset, \$\\varepsilon=$ε\$}} \\\\ \\cline{2-4}
% header 2
    \\multicolumn{1}{c}{\\textbf{Method}} &"""
for ds in dss
    global res
    res *= "\n    \\multicolumn{1}{|c|}{$ds} &"
end
res = res[1:end-1] * " \\\\ \\cline{1-4}\n"
println(f, res)

cm = 0
for r in eachrow(formated)
    global cm
    m = r[:Size]
    res = if cm != m
        cm = m
"""
    % m = $m
    \\multicolumn{1}{r}{\\textbf{m = $m}} & & & \\multicolumn{1}{l}{} \\\\ \\cline{1-4}
"""
    else
        ""
    end

    res *= "    \\multicolumn{1}{r|}{$(r[:Method])} &"
    for ds in dss
        cls = round(r[Symbol(ds)] |> Measurements.value, digits=2)
        clc = round(Int, r[Symbol(ds*"_1")] |> Measurements.value)
        res *= "\n    \\multicolumn{1}{c}{$cls ($clc)} &"
    end
    res = res[1:end-1] * "  \\\\ \\cline{1-4}\n"
    println(f, res)
end
println(f, "\\end{tabular}%")
close(f)

#-----------------------------
