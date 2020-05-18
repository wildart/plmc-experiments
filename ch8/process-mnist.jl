# Process MNIST results

using Statistics
using Measurements: measurement, value, uncertainty
using DataFramesMeta

include("src/results.jl")

res = getResults("MNIST", :fileinfo, filter=r"^mnist-\d+-\d+-clustering-\d+\.bson")

#-------------------------------------------------------------------------------

# flt = r"^mnist-\d+-\d+-\d+-clustering-\d+\.bson"
flt = r"fmnist-\d+-\d+-\d+-clustering-\d+\.bson"
nmi = getResults("MNIST", :NMI, filter=flt)
nmi_sample = getResults("MNIST", :NMI_SAMPLE, filter=flt)
nmi_test = getResults("MNIST", :NMI_TEST, filter=flt)
# assign = getResults("MNIST", :ASSIGN, filter=flt)

results = DataFrame(Algorithm=String[], NMI=Float64[], NMISample=Float64[], NMITest=Float64[])
# (dsn,ts) = keys(nmi) |> first
for (dsn,ts) in keys(nmi)
    k = (dsn,ts)
    # ni,ns,nt = first(nmi[k]), first(nmi_sample[k]), first(nmi_test[k])
    # a = first(assign[k])
    for (ni, ns, nt) in zip(nmi[k], nmi_sample[k], nmi_test[k]) #, assign[k]
        for i in keys(ni)
            push!(results, (i, ni[i], ns[i], nt[i])) #, length(unique(aa))
        end
    end
end


gdf = groupby(results, [:Algorithm])
rstats = combine(gdf, :NMI => (x->measurement(median(x),mad(x))) => :Training,
                     :NMISample => (x->measurement(median(x),mad(x))) => :Sample,
                     :NMITest => (x->measurement(median(x),mad(x))) => :Testing)

sstats = combine(gdf, :NMI => (x->measurement(mean(x),std(x))) => :NMI,
                     :NMISample => (x->measurement(mean(x),std(x))) => :NMISample,
                     :NMITest => (x->measurement(mean(x),std(x))) => :NMITest)


#-------------------------------------------------------------------------------
# io = stdout
io = open("../gen/fmnist-nmi.tex", "w")
#
toppart = ["kmeans", "gmm"]
colnum = size(rstats, 2)
colnms = names(rstats)
nline = " \\\\"
clinegen(s) = " \\cline{$s-$colnum}"
println(io, "\\begin{tabular}{r$("|c"^(colnum-1))}")
println(io, "\\hline")
println(io, "\\multirow{2}{*}{$(colnms[1])} & \\multicolumn{$(colnum-1)}{c}{NMI}" * nline * clinegen(2))
println(io, " & "*join(colnms[2:end], " & ") * nline * clinegen(1))
for p in [toppart, setdiff(rstats[!, :Algorithm], toppart) |> sort]
    for a in p
        r = @where(rstats, :Algorithm .== a)[1,:]
        print(io, r[1])
        for v in r[2:end]
            mdv = round(v |> value, digits=3)
            mde = round(v |> uncertainty, digits=3)
            print(io, " & \$$(rpad("$mdv", 5, '0')) \\pm  $(rpad("$mde", 5, '0'))\$")
        end
        println(io, nline)
    end
    println(io, "\\hline")
end
println(io,"\\end{tabular}")
#
close(io)
