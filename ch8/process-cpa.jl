# Process CPA results

#-----------------------------
# Load results
#

include("src/results.jl")

flt = r"tm\d-d\d+-cpa"

nmi = getResults("Moons", :NMI, filter=flt)
ridx = getResults("Moons", :RAND, filter=flt)
assign = getResults("Moons", :ASSIGN, filter=flt)
eps = getResults("Moons", :EPS, filter=flt)

results = DataFrame(Components=Int[], Dimension=Int[], Eps=Float64[], Method=String[],
                    NMI=Float64[], RAND=Float64[], ClustNum=Int[])
# (ds,ts) = keys(eps) |> first
for (dsn,ts) in keys(eps)
    k = (dsn,ts)
    m = parseCD(dsn)
    m === nothing && continue
    c, d = m
    #println("$ds: $c, $d")
    # ss,rr,aa = first(nmi[k]), first(ridx[k]), first(assign[k])
    for (e, ss, rr, aa) in zip(eps[k], nmi[k], ridx[k], assign[k])
        for mn in keys(ss)
            push!(results, (c, d, e, mn, ss[mn], rr[mn][2], length(unique(aa[mn]))))
        end
    end
end

using Feather
Feather.write("results/cpa.feather", results)
exit()

#-----------------------------
# NMI by c & d
#

using Statistics
using Measurements: measurement, value, uncertainty
using DataFramesMeta
using Feather
results = Feather.read("results/cpa.feather")

include("src/plotting.jl")

ds = [2, 3, 7, 14]
cs = [2, 3, 5, 7]
εs = [0.1, 0.2, 0.3]

# d,c = 2,2
for d in ds, c in cs
    println("Moons: $c, $d")
    dres = @where(results, :Dimension .== d, :Components .== c)
    size(dres,1) == 0 && continue
    gdf = groupby(dres, [:Eps, :Method])
    stats = combine(gdf, :NMI => (x->measurement(median(x),mad(x))) => :NMI,
                         :RAND => (x->measurement(median(x),mad(x))) => :RAND,
                         :ClustNum => (x->measurement(median(x),mad(x))) => :ClustNum)
    sstats = sort(stats, [:NMI, :Method, :Eps])

    p = @df sstats groupedbar(:Method, :NMI, group=:Eps, ylims=(0,1.05),
                            ylab="NMI", xrotation = 45, # title="Moons$c (d=$d)",
                            legend=(0.97, 1), legendtitle="ε")

    saveplot("../gen/cpa-moons$c-$d.pdf", p)
end


# NMI Table

gdf = groupby(results, [:Eps, :Method, :Components, :Dimension])
stats = combine(gdf, :NMI => (x->measurement(median(x),mad(x))) => :NMI,
                     :RAND => (x->measurement(median(x),mad(x))) => :RAND,
                     :ClustNum => (x->measurement(median(x),mad(x))) => :ClustNum)

# sort(@where(stats, :Method .== "kmeans"), [:Components, :Dimension, :Eps])

# stats2 = @linq stats |> transform(sEps =  "ε".*string.(:Eps))
# nmistats = unstack(stats2[!, Not([:RAND, :ClustNum,:Eps])], :sEps, :NMI)
# clnstats = unstack(stats2[!, Not([:RAND, :NMI,:Eps])], :sEps, :ClustNum)
# allstats = innerjoin(nmistats, clnstats, makeunique=true,
#                      on = [:Method => :Method, :Dimension => :Dimension, :Components => :Components])
# sstats = sort(allstats, [:Dimension, :Method])

# ε = 0.1
for ε in εs
    io = open("../gen/cpa-moons-nmi-ε$ε.tex", "w")
    # calculate stats
    allstats = @where(stats, :Eps .== ε)[!, Not(:Eps)]
    allsstats = unstack(allstats[!, Not([:RAND, :ClustNum])], :Method, :NMI) |> df -> sort(df, [:Components, :Dimension])
    cols = size(allsstats,2)
    cline = "\\cline{1-$cols}"
    # rearrange
    cnames = names(allsstats)
    cnames[1:2] .= ["c", "d"]
    rename!(allsstats, Symbol.(cnames))
    insert!(cnames, 6, pop!(cnames))
    allsstats = allsstats[!, cnames]
    # r = allsstats[6, Not([:c, :d])]
    tomark = [let rv = coalesce.(collect(r), 0); findall(rv .== maximum(rv)).+2; end for r in eachrow(allsstats[!, Not([:c, :d])])]
    # header
    println(io, "\\begin{tabular}{|"* "c|"^cols * "}")
    println(io, cline)
    plmcidx = findall(n->startswith(n, "plmc"), cnames)
    println(io, join(cnames[1:plmcidx[1]-1], " & ") * " & \\multicolumn{$(length(plmcidx))}{c|}{plmc} \\\\")
    println(io, "\\cline{$(plmcidx[1])-$cols}")
    println(io, " & "^(plmcidx[1]-1) * join(map(s->replace(s, "plmc+"=>""), cnames[plmcidx]), " & ") * " \\\\")
    println(io, cline)
    # body
    # j, r = 1, allsstats[6,:]
    for (j,r) in enumerate(eachrow(allsstats))
        for (i,n) in enumerate(cnames)
            if i < 3
                print(io, r[n])
            else
                if ismissing(r[n])
                    print(io, "NA")
                else
                    mdv = round(r[n] |> value, digits=3)
                    mde = round(r[n]|> uncertainty, digits=3)
                    sv = "\$$mdv \\pm $mde\$"
                    print(io, i ∈ tomark[j] ? "\\bm{$sv}" : sv)
                end
            end
            print(io, i == cols ? "" : " & ")
        end
        println(io, " \\\\")
    end
    println(io, cline)
    println(io, "\\end{tabular}")
    close(io)
end

#-----------------------------
# Plot some clusterings

include("src/results.jl")
include("src/cover.jl")
include("src/plotting.jl")

using PLMC
using Plots, TDA

function findscore(J)
    # J = map(j->floor(Int, j), J)
    J[1] = J[end] = maximum(J)
    mx1 = PLMC.findglobalmin(J, 1e-3)
    mx2 = PLMC.findglobalmin2(J)
    return max(mx1, mx2)
end

flt = r"tm\d-d2-cpa"
data = getResults("Moons", :DATA, filter=flt)
params = getResults("Moons", :PARAMETERS, filter=flt)

ii = [3, 84, 282, 36]
ks = keys(data) |> collect |> v->sort(v, by=first)
# k, i = ks[3], ii[3]
for (i,k) in zip(ii, ks)
    c, d = parseCD(k[1])
    println("Moons: $c, $d")
    X = data[k][i]
    P = params[k][i]
    flt, MC =  cover(X, :mfa, P)
    # plot(MC, X, leg=:none)
    agg = agglomerate(flt)
    C, AGG, J = plmc(agg, MC, X, flt, score=PLMC.nml, find=findscore)
    # C, AGG, J = plmc(PLMC.MDL, X, flt, MC, β=2, βsize=1, score=PLMC.nml, find=findscore)
    # C, AGG, J = plmc(PLMC.InformationBottleneck, X, flt, MC, β=2, score=PLMC.nml, find=findscore)
    # C2, _ = plmc(AGG, C.models, X, aggidx=15)
    p = plot(C, X, lw=2, label="", ms=3)
    plot!(p, C, lw=2, pc=false, label="")
    # saveplot("../gen/res-moons$c-$d.pdf", p)
    saveplot("../gen/res-moons$c-$d-ib.pdf", p)
end
