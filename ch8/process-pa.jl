# Process PA results

using Statistics
using Measurements: measurement, value, uncertainty
using DataFramesMeta

include("src/results.jl")
include("src/plotting.jl")

ds = [2, 3, 7, 14]
cs = [2, 3, 5, 7]
εs = [0.1, 0.2, 0.3]

#-----------------------------
# Load results
#

parameter = :Cover
parameterType = Int

# flt = r"tm2-d\d+-pa"
flt = r"tm\d-d\d+-pa"
# flt = Regex("tm\\d-d\\d+-pa-$(lowercase(String(parameter)))")

nmi = getResults("Moons", :NMI, filter=flt)
ridx = getResults("Moons", :RAND, filter=flt)
assign = getResults("Moons", :ASSIGN, filter=flt)
cvr = getResults("Moons", :COVER, filter=flt)
eps = getResults("Moons", :EPS, filter=flt)

results = DataFrame(Components=Int[], Dimension=Int[], Eps=Float64[], Method=String[],
                    NMI=Float64[], RAND=Float64[], ClustNum=Int[])
@eval results.$parameter = $parameterType[]

# (dsn,ts) = keys(eps) |> first
for (dsn,ts) in keys(eps)
    k = (dsn,ts)
    m = parseCD(dsn)
    m === nothing && continue
    c, d = m
    println("$dsn: $c, $d")
    # ss,rr,aa,cc = first(nmi[k]), first(ridx[k]), first(assign[k]), first(cvr[k])
    for (e, ss, rr, aa, cc) in zip(eps[k], nmi[k], ridx[k], assign[k], cvr[k])
        for mn in keys(ss)
            clc = length(unique(aa[mn]))
            push!(results, (c, d, e, mn, ss[mn], rr[mn][2], clc, c*cc))
        end
    end
end

#-----------------------------
# NMI by parameter & ε
#

# d,c = 2,2
for d in ds, c in cs
    println("Moons: $c, $d")
    dres = @where(results, :Dimension .== d, :Components .== c)
    size(dres,1) == 0 && continue
    gdf = groupby(dres[!, Not([:Components, :Dimension])], [:Eps, :Method, parameter])
    stats = combine(gdf, :NMI => (x->measurement(median(x),mad(x))) => :NMI,
                         :RAND => (x->measurement(median(x),mad(x))) => :RAND,
                         :ClustNum => (x->measurement(median(x),mad(x))) => :ClustNum)
    sstats = sort(stats, [:Method, :Eps])

    cvrs = unique(sstats[!,parameter])
    # ε = 0.1
    for ε in εs
        plotdf = sort(@where(sstats, :Eps .== ε), [parameter])
        tmp = plotdf[!,:NMI] |> maximum
        ymaxnmi =round( value(tmp)+uncertainty(tmp), digits=1)
        tmp = plotdf[!,:ClustNum] |> maximum
        ymaxcl = value(tmp)+uncertainty(tmp)
        p1 = @df plotdf plot(cols(parameter), :NMI, group=:Method, leg=:none,
                            ylab="NMI", xlab="$parameter", xticks=cvrs,
                            ylims=(0.0,1.0), yticks=0.0:0.1:1);
        p2 = @df plotdf plot(cols(parameter), :ClustNum, group=:Method, leg=:topleft,
                            ylims=(0,ymaxcl), ymirror = true,
                            ylab="Clusters", xlab="$parameter", xticks=cvrs, yticks=2:2:ymaxcl);
        p = plot(p1,p2, layout = @layout([a b]), plot_title="Moons$c (d=$d, ε=$ε)") #, size=figsize)

        saveplot("../gen/pa-moons$c-d$d-ε$ε.pdf", p)
    end
end
