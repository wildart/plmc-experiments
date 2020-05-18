# PenDigits

using Statistics
using Measurements: measurement, value, uncertainty
using DataFramesMeta

include("src/results.jl")

res = getResults("PEN", :fileinfo, filter=r"^pendigits-\d+-\d+\.bson")
res = getResults("PEN", :fileinfo, filter=flt)

#-------------------------------------------------------------------------------

flt = r"^pendigits-(2-s.*35|8-d.*35|16-f.*35|\d+-k.*)-\d+\.bson"
flt = r"^pendigits-(2-d.*35|8-d.*35|16-f.*35|\d+-k.*)-\d+\.bson"
# flt = r"^pendigits-\d+-(s.*35|k.*)-\d+\.bson"
# flt = r"^pendigits-16-16066\d+\.bson"
nmi = getResults("PEN", :NMI, filter=flt)
nmi_test = getResults("PEN", :NMI_TEST, filter=flt)
assign = getResults("PEN", :ASSIGN, filter=flt)
iter = getResults("PEN", :ITER, filter=flt)
params = getParams(flt)

results = DataFrame(Algorithm=String[], Dimension=Int[], Clusters=Int[], NMI=Float64[], NMITest=Float64[], Iter=Int[])
# (dsn,ts) = keys(nmi) |> collect |> last
for (dsn,ts) in keys(nmi)
    k = (dsn,ts)
    p = params[k]
    d = p[:dimension]
    # ni, nt, a, it = first(nmi[k]), first(nmi_test[k]), first(assign[k]), first(iter[k])
    for (ni, nt, a, it) in zip(nmi[k], nmi_test[k], assign[k], iter[k])
        for i in keys(ni)
            push!(results, (i, d, length(unique(a[i])), ni[i], nt[i], it))
        end
    end
end

gdf = groupby(results, [:Algorithm, :Dimension])
rstats = combine(gdf,
    :Clusters => (x->measurement(median(x),mad(x))) => :Clusters,
    :NMI => (x->measurement(median(x),mad(x))) => :Training,
    :NMITest => (x->measurement(median(x),mad(x))) => :Testing,
    :NMITest => extrema => Symbol("Min/Max"))
# sort(rstats, [:Dimension, :Testing])

sstats = combine(gdf,
    :Clusters => (x->measurement(mean(x),std(x))) => :Clusters,
    :NMI => (x->measurement(mean(x),std(x))) => :Training,
    :NMITest => (x->measurement(mean(x),std(x))) => :Testing)


#-------------------------------------------------------------------------------
io = stdout
# io = open("../gen/pendigits-nmi.tex", "w")
#
toppart = ["kmeans", "kmeans-nmi", "gmm", "gmm-jl"]
filter = ["-jl", "-nmi"] #  String[]
filter = ["-nmi"] #  String[]
colnum = size(rstats, 2)
colnms = names(rstats)
scols = length(colnms)-3
nline = " \\\\"
clinegen(s) = " \\cline{$s-$colnum}"
println(io, "\\begin{tabular}{r$("|c"^(colnum-1))}")
println(io, "\\hline")
println(io, join(map(c->"\\multirow{2}{*}{$c} & ", colnms[1:scols])) * "\\multicolumn{$(colnum-3)}{c}{NMI}" * nline * clinegen(scols+1))
println(io, " & "^scols *join(colnms[scols+1:end], " & ") * nline * clinegen(1))
for d in unique(rstats[!, :Dimension])
for algs in [toppart, setdiff(rstats[!, :Algorithm], toppart) |> sort]
    # a = toppart[2]
    for a in algs
        any(sfx->endswith(a,sfx), filter) && continue # filter
        rr = @where(rstats, :Algorithm .== a, :Dimension .== d)
        size(rr,1) == 0 && continue
        r = rr[1,:]
        print(io, "$(rpad(r[1], 11, ' ')) & $(lpad(r[2],3,' '))")
        for (i,v) in enumerate(r[3:end])
            prcs = i == 1 ? 1 : 3
            if isa(v, Measurement)
                mdv = round(v |> value, digits=prcs)
                smdv = rpad("$mdv", prcs+2, '0')
                if prcs == 1
                    smdv = lpad(smdv, 4, ' ')
                end
                mde = round(v |> uncertainty, digits=prcs)
                print(io, " & \$$(smdv) \\pm  $(rpad("$mde", prcs+2, '0'))\$")
            elseif isa(v, Tuple)
                mdv1 = rpad("$(round(v[1], digits=prcs))", prcs+2, '0')
                mdv2 = rpad("$(round(v[2], digits=prcs))", prcs+2, '0')
                print(io, " & \$($mdv1, $mdv2)\$")
            end
        end
        println(io, nline)
    end
    println(io, "\\hline")
end
println(io, "\\hline")
end
println(io,"\\end{tabular}")
#
close(io)


#-------------------------------------------------------------------------------
sort(combine(gdf, :NMITest => extrema => :BestNMI,
                  :NMI => (x->measurement(median(x),mad(x))) => :Training,
                  :NMITest => (x->measurement(median(x),mad(x))) => :Testing,
                  :Clusters => (x->measurement(median(x),mad(x))) => :Clusters), [:Dimension, :Testing])
# @where(results, :Algorithm .=="plmc+dpgmm", :Dimension .== 2, :Iter .== 11)
# @where(results, :Algorithm .=="plmc+dpgmm", :Dimension .== 2)
@where(sort(results, [:NMITest, :Clusters]), :Dimension .== 2)
# sort(rstats, [:Dimension, :Testing])
