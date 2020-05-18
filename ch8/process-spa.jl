# Process SPA results

using Statistics
using Measurements: measurement, value, uncertainty
using DataFramesMeta
#using Latexify

include("src/results.jl")
include("src/plotting.jl")

ds = [2, 3, 7, 14]
cs = [2, 3, 5, 7]
εs = [0.1, 0.2, 0.3]

#-----------------------------
# Load results
#

# fltmthd = nothing
# fltmthd = Regex("(mfa|kmeans):(top|mdl|ib)\$")
fltmthd = Regex("(mfa|kmeans):(top|mdl)\$")

spaid = "spa"
datesuffix = "" # 159630
fileflt = "tm\\d+-d\\d+-$spaid"
rxfiles = map(sfx->Regex("$fileflt-$sfx-$datesuffix"), ["data", "cover"])


#-----------------------------
# MDL data
#

# cno = 7
for cno in cs
Pdata = getParams(replace(fileflt, "\\d+" => string(cno), count=1)|>Regex)
length(Pdata) == 0 && continue
P = Pdata |> values |> first
mrn = cno*P[:cover] # columns
mdldftypes = [Int, String, Bool, Int, Float64, fill(Float64, mrn)...]
mdldnames = [:Dimension, :Method, :Mode, :Itr, :Eps, map(i->Symbol("M$i"), 1:mrn)...]
mdldf = DataFrame(mdldftypes, mdldnames)
# flt = rxfiles[2]
for (i,flt) in enumerate(rxfiles)
    mdl = getResults("Moons", :TMDL, filter=flt)
    ep = getResults("Moons", :EPS, filter=flt)
    itr = getResults("Moons", :ITR, filter=flt)

    # k = (keys(mdl) |> collect)[3]
    for k in keys(mdl)
        # global cno
        m = parseCD(k[1])
        m === nothing && continue
        c, d = m
        c != cno && continue
        # e,ml,ii = first(ep[k]), first(mdl[k]), first(itr[k])
        for (e, mmdls, ii) in zip(ep[k], mdl[k], itr[k])
            # println("$i, $ii, $e, $jj")
            for (mth, mmdl) in mmdls
                if length(mmdl) != 8
                    append!(mmdl, fill(mmdl[end], mrn-length(mmdl)))
                end
                push!(mdldf, (d, mth, i-1, ii, e, mmdl...))
            end
        end
    end
end
size(mdldf,1) == 0 && continue
#
# d,ε = 2,0.1
cylims = Dict(2 =>(7050, 7420), 3 =>(11220, 11820), 5 =>(20050, 21000), 7 =>(29250, 30250))
for d in ds, ε in εs
    mdldimeps = @where(mdldf, :Dimension .== d, :Eps .== ε)
    size(mdldimeps,1) == 0 && continue
    mdlmthd = groupby(mdldimeps[!, Not([:Itr, :Eps, :Dimension])], [:Method])
    # mk = keys(mdlmthd)[1]
    ps = []
    for mk in keys(mdlmthd)
        mdlmode = groupby(mdlmthd[mk][!, Not(:Method)], :Mode)
        pdata1 = Matrix(mdlmode[1][!, Not(:Mode)])
        pdata2 = Matrix(mdlmode[2][!, Not(:Mode)])
        #extrema([extrema(pdata1)...,extrema(pdata2)...])
        p1 = plot(pdata1', leg=:none, ylabel="MDL", title="Data: $(mk[:Method])", widen=false);
        p2 = plot(pdata2', leg=:none, title="Cover: $(mk[:Method])", showaxis=:x, yticks=:none, widen=false);
        p = plot(p1, p2, link=:y, ylims=cylims[cno]);
        push!(ps, p)
    end
    pp = plot(ps..., layout =@layout([a;b]), titlefontsize=12);
    saveplot("../gen/$spaid-mdl-c$cno-d$d-ε$ε.pdf", pp)
end
end

#-----------------------------
# NMI by mode & ε
#

CLUSTERING = [:plmc]
COVERS = [:kmeans, :mfa]
META = [:top, :mdl, :ib]
mthds = ["$cl+$cv:$m" for cl in CLUSTERING for cv in COVERS for m in META]

results = DataFrame(Components=Int[], Dimension=Int[], Eps=Float64[], Method=String[],
                    NMI=Float64[], RAND=Float64[], ClustNum=Int[], Mode=Bool[], Id=Int[])
for (i,flt) in enumerate(rxfiles)
    nmi = getResults("Moons", :NMI, filter=flt)
    ridx = getResults("Moons", :RAND, filter=flt)
    assign = getResults("Moons", :ASSIGN, filter=flt)
    ep = getResults("Moons", :EPS, filter=flt)
    itr = getResults("Moons", :ITR, filter=flt)

    # (ds,ts) = keys(nmi) |> first
    for (ds,ts) in keys(nmi)
        k = (ds,ts)
        println("Processing: $k")
        m = parseCD(ds)
        m === nothing && continue
        c, d = m
        #println("$ds: $c, $d")
        # e,ss,rr,aa,ii = first(ep[k]), first(nmi[k]), first(ridx[k]), first(assign[k]), first(itr[k])
        for (e, ss, rr, aa, ii) in zip(ep[k], nmi[k], ridx[k], assign[k], itr[k])
            # mn = first(keys(ss))
            for mn in mthds
                fltmthd !== nothing && match(fltmthd,mn) === nothing && continue
                r = if mn ∈ mthds
                    (c, d, e, mn, ss[mn], rr[mn][2], length(unique(aa[mn])), i-1, ii)
                else
                    (c, d, e, mn, missing, missing, missing, i-1, ii)
                end
                push!(results, r)
            end
        end
    end
end
#categorical!(results, :Method)

# using Feather
# Feather.write("spa.feather", results)
# results = Feather.read("spa.feather")

# d,c,ε = 14,2,0.3
for d in ds, c in cs, ε in εs
    plotdf = @where(results, :Dimension .== d, :Components .== c, :Eps .== ε)
    size(plotdf,1) == 0 && continue
    println("Moons: $c, $d, $ε")

    p = @df plotdf[plotdf.Mode .== false,:] begin
        dotplot(:Method, :NMI, side=:right, marker=(3, 0.25, :blue, stroke(0)),
                label="Data", ylab="NMI", yticks=0.0:0.1:1, ylim=(0.0,1.0))
    end
    @df plotdf[plotdf.Mode .== true,:] begin
        dotplot!(p, :Method, :NMI, side=:left, marker=(3, 0.25, :red, stroke(0)),
                label="Cover", leg=:bottomright)
    end

    saveplot("../gen/$spaid-moons$c-d$d-ε$ε.pdf", p)
end

#-----------------------------

using HypothesisTests

#http://www.real-statistics.com/one-way-analysis-of-variance-anova/homogeneity-variances/dealing-with-heterogeneous-variances/
correctvar(x) = log.(x)
# correctvar(x) = log10.(x)
# correctvar(x) = log.(x .+ 0.1)
# correctvar(x) = sqrt.(x)
# correctvar(x) = sqrt.(x .+ 0.5)
# correctvar(x) = sqrt.(x) .+ sqrt.(x .+ 1)
cleandata = collect ∘ skipmissing

fkttypes = [Int,Int,String,Float64,Float64]
fktnames = [:Components,:Dimension,:Method,Symbol("\$\\varepsilon\$"),Symbol("P-value")]
fkt = DataFrame(fkttypes, fktnames)
# d,c,ε,m = 2,2,0.1,Symbol("plmc+mfa:top")
for d in ds, c in cs, ε in εs
    testdf = @where(results, :Dimension .== d, :Components .== c, :Eps .== ε)
    size(testdf,1) == 0 && continue
    res = [ unstack(testdf[testdf.Mode .== md, [:NMI, :Method, :Id, :Eps]], :Method, :NMI) for md in [0, 1] ]
    for m in mthds
        m ∉ names(res[1]) && continue
        t = FlignerKilleenTest((cleandata(r[!,m]) for r in res)...)
        # println("$m\n", t)
        push!(fkt, (c,d,m,ε,round(pvalue(t), digits=3)))
    end
end

function generatespatable(fname, testdf, α = 0.05)
    open(fname, "w") do io
        colnum = size(testdf, 2)
        colnms = names(testdf)
        cline = "\\cline{1-$colnum}"
        println(io, "\\begin{tabular}{c|c|$(rpad("",colnum-2,"c"))}")
        println(io, cline)
        println(io, join(map(s->replace(s, "plmc+"=>""), colnms), " & ")*" \\\\ \n"*cline)
        # r = eachrow(testdf) |> first
        for r in eachrow(testdf)
            for (i,n) in enumerate(colnms)
                sv = if i < 3
                    r[n]
                else
                    tmp = rpad("$(r[n])", 5, '0')
                    r[n] >= α ? "\\textbf{$tmp}" : tmp
                end
                print(io, sv, colnum != i ? " & " : "\\\\")
            end
            if r[1] == "7"
                println(io, "\n$cline")
            else
                println(io)
            end
        end
        println(io,"\\end{tabular}")
    end
end

fkttab = unstack(fkt, :Method, fktnames[end])
fkttabd2 = sort(@where(fkttab, :Dimension .== 2)[!,Not(:Dimension)], [2])
generatespatable("../gen/$spaid-moons-fkt-nmi-d2.tex", fkttabd2)
fkttabc2 = sort(@where(fkttab, :Components .== 2)[!,Not(:Components)], [2])
generatespatable("../gen/$spaid-moons-fkt-nmi-c2.tex", fkttabc2)
# println(io, latexify(fkttab; env=:table, latex=false, fmt="%.3f"))


# Add Δ medians

fkttab = unstack(fkt, :Method, fktnames[end])

# add med.diff
gdf = groupby(results[!, Not([:RAND, :ClustNum])], [:Eps, :Method, :Components, :Dimension, :Mode])
stats = combine(gdf, :NMI => (x->measurement(median(x),mad(x))) => :NMI)
sstats = sort(unstack(stats, :Method, :NMI), [:Components, :Dimension])
# m = "plmc+mfa:top"
for m in mthds
    m ∉ names(fkttab) && continue
    meds = unstack(sstats[!, [:Components, :Dimension, :Eps, :Mode, Symbol(m)]], :Mode, Symbol(m))
    mdiff = meds[!,end] .- meds[!,end-1]
    newname = Symbol(m*"@MDiff")
    @eval fkttab.$newname = $mdiff
end

# io = stdout
io = open("../gen/$spaid-moons-fkt-nmi.tex", "w")
#
α = 0.05
colnum = size(fkttab, 2)
colnms = names(fkttab)
clinegen(s) = "\\cline{$s-$colnum}"
println(io, "\\begin{tabular}{|c|c|c|$(rpad("",colnum-3,"c"))}")
println(io, clinegen(1))
reported = filter(c->c ∈ mthds, colnms)
joined = [["c", "d", "\$\\varepsilon\$"]; map(s->"\\multicolumn{2}{c|}{"*replace(s, "plmc+"=>"")*"}", reported)]
println(io, join(joined, " & ")*" \\\\ \n"*clinegen(4))
println(io, " & "^3 * join(["\\multicolumn{1}{c}{\$\\Delta med\$} & \\multicolumn{1}{c|}{FK.Pv}" for i=1:length(reported)], " & ") * " \\\\")
println(io, clinegen(1))
# r = eachrow(fkttab) |> first
# n = reported[1]
for r in eachrow(fkttab)
    for (i,n) in enumerate([colnms[1:3]; reported])
        sv = if i < 4
            r[n]
        else
            mdv = round(r["$n@MDiff"] |> value, digits=3)
            mde = round(r["$n@MDiff"] |> uncertainty, digits=3)
            fk = r[n]
            tmp = rpad("$fk", 5, '0')
            sfk = fk >= α ? "\\textbf{$tmp}" : tmp
            "\\multicolumn{1}{c}{\$$mdv \\pm  $mde\$} & \\multicolumn{1}{c|}{$sfk}"
        end
        print(io, sv, i != length(reported)+3 ? " & " : "\\\\")
    end
    if r[1] == "7"
        println(io, "\n"*clinegen(1))
    else
        println(io)
    end
end
println(io, clinegen(1))
println(io,"\\end{tabular}")
#
close(io)
