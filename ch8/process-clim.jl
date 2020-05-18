# Process climate data results

include("src/results-plmc.jl")
# include("src/plotting.jl")

BSON.@load "../ch6/data/CLIM_1951_1980_1x1.bson" UNIT MASK
X = convert(Matrix{Float64}, UNIT')

csz = [36, 72, 108, 144]
cfname = "plmc-km"
cfname = "plmc-mfa"
suffix = ["top", "mdl", "inf"]

cs, sfx = csz[2], suffix[1]

C,AGG,J = loadClustering("results/$cfname-$cs-$sfx.bson")

# using Plots
# plot(J, leg=:none)
# PLMC.markminima(J) |> first |> findall
# AGG.clusters |> length
C2, _ = plmc(AGG, C.models, X, aggidx=119)

include("src/conda.jl")
include("../ch6/src/plotting-geo.jl")
saveplot("../gen/clim-plmc-mfa-$cs-$sfx.pdf", mapclusters(C, MASK))
saveplot("../gen/clim-plmc-mfa-$cs-$sfx.pdf", mapclusters(C2, MASK))
saveplot("../gen/clim-km-34.pdf", mapclusters(C3, MASK))

# C3 = kmeans(X, 34)

PLMC.score(PLMC.nll, C, X) |> n -> round(Int, -n)
PLMC.score(PLMC.nll, C2, X) |> n -> round(Int, -n)
PLMC.score(PLMC.nll, C3, X) |> n -> round(Int, -n)

#            &  LL     & CL
# KM:TOP & 36  & 1389416 & 4
# MFA:TOP & 36 & 1168924 & 5
# KM:MDL & 36  & 1429104 & 15
# MFA:MDL & 36 & 1206325 & 22
# KM:IB & 36   & 1437135 & 14
# MFA:IB & 36  & 1184756 & 9
# KM & 15      & 1349280 & 15
# KM:TOP & 72  & 1494392 & 28
# MFA:TOP & 72 & 1267293 & 5
# KM:MDL & 72  & 1506659 & 26
# MFA:MDL & 72 & 1318469 & 39
# KM:IB & 72   & 1517704 & 35
# MFA:IB & 72   & 1311217 & 27
# KM & 26      & 1405087 & 26
# KM & 34      & 1445777 & 34
# KM:TOP & 108 & 1544956 & 43
# MFA:TOP & 108 & 1378801 & 69
# KM:MDL & 108 & 1564755 & 44
# MFA:MDL & 108 & 1378183 & 52
# KM:IB & 108  & 1581016 & 63
# MFA:IB & 108 & 1390779 & 78
# KM & 44      & 1473011 & 44
# KM:TOP & 144 & 1576948 & 48
# MFA:TOP & 144 & 1394862 & 88
# KM:MDL & 144 & 1610799 & 68
# MFA:MDL & 144 & 1398835 & 51
# KM:IB & 144  & 1619750 & 71
# MFA:IB & 144  & 1378679 & 33
# KM & 68      & 1521896 & 68
# KM & 85      & 1556931 & 85

#-------------------------------------------------------------------------------

using DataFrames
climdf = DataFrame(Method=String[], Cover=Int[], LL=Int[], Clusters=Int[])
push!(climdf, ("PLMC:KM+TOP",36 ,1389416,4))
push!(climdf, ("PLMC:MFA+TOP",36,1168924,5))
push!(climdf, ("PLMC:KM+MDL",36 ,1429104,15))
push!(climdf, ("PLMC:MFA+MDL",36,1206325,22))
push!(climdf, ("PLMC:KM+IB",36  ,1437135,14))
push!(climdf, ("PLMC:MFA+IB",36 ,1184756,9))
push!(climdf, ("k-Means",15     ,1349280,15))
push!(climdf, ("PLMC:KM+TOP",72 ,1494392,28))
# push!(climdf, ("PLMC:MFA+TOP",72,1267293,5))
push!(climdf, ("PLMC:KM+MDL",72 ,1506659,26))
push!(climdf, ("PLMC:MFA+MDL",72,1318469,39))
push!(climdf, ("PLMC:KM+IB",72  ,1517704,35))
push!(climdf, ("PLMC:MFA+IB",72  ,1311217,27))
push!(climdf, ("k-Means",26     ,1405087,26))
push!(climdf, ("k-Means",34     ,1445777,34))
push!(climdf, ("PLMC:KM+TOP",108,1544956,43))
push!(climdf, ("PLMC:MFA+TOP",108,1378801,69))
push!(climdf, ("PLMC:KM+MDL",108,1564755,44))
push!(climdf, ("PLMC:MFA+MDL",108,1378183,52))
push!(climdf, ("PLMC:KM+IB",108 ,1581016,63))
push!(climdf, ("PLMC:MFA+IB",108,1390779,78))
push!(climdf, ("k-Means",44     ,1473011,44))
push!(climdf, ("PLMC:KM+TOP",144,1576948,48))
push!(climdf, ("PLMC:MFA+TOP",144,1394862,88))
push!(climdf, ("PLMC:KM+MDL",144,1610799,68))
push!(climdf, ("PLMC:MFA+MDL",144,1398835,51))
push!(climdf, ("PLMC:KM+IB",144 ,1619750,71))
push!(climdf, ("PLMC:MFA+IB",144 ,1378679,33))
push!(climdf, ("k-Means",68     ,1521896,68))
push!(climdf, ("k-Means",85     ,1556931,85))

include("src/plotting.jl")
p = @df sort(climdf, [:Clusters]) plot(:Clusters, :LL, group=:Method,
    leg=:bottomright, xlab="Clusters", ylab="LL", lw=2, legendfontsize=7,
    xticks=(10:10:90))

climdf2 = DataFrame(Method=String[], LL=Int[], Clusters=Int[])
push!(climdf2, ("LMCLUS",1455666,36))
push!(climdf2, ("LMCMDL",1431362,32))

scatter!(p, climdf2[[1],:Clusters], climdf2[[1],:LL], label=climdf2[1,:Method])
scatter!(p, climdf2[[2],:Clusters], climdf2[[2],:LL], label=climdf2[2,:Method])

saveplot("../gen/cpa-clim.pdf", p)
