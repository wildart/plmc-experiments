# PenDigits

using Logging
lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Info);
# lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Debug);
# Logging.disable_logging(Logging.BelowMinLevel);

include("src/conda.jl")
include("src/cover.jl")
include("src/clustering.jl")
include("src/metaclustering.jl")
include("src/classifiers-plmc.jl")
include("src/results.jl")

# process args
startindex = 1
reduce_dim = 0
covtype = "full"
covsize = 25
if length(ARGS) > 0
    startindex = parse(Int, ARGS[1])
    println("Start from the iteration: $startindex")
    if length(ARGS) > 1
        reduce_dim = parse(Int, ARGS[2])
        reduce_dim > 1 && println("UMAP dimentionality reduction: $reduce_dim")
        if length(ARGS) > 2
            covtype = ARGS[3] == "s" ? "spherical" : ARGS[3] == "d" ? "diag" : "full"
            println("GMM covariance type: $covtype")
            if length(ARGS) > 3
                covsize = parse(Int, ARGS[4])
            end
        end
    end
end
println("Cover size: $covsize")

## parameters
PARAMS = Dict(
    :iterations => 100,
    :experiment_seed => 837387533,
    :sample => 1000,
    :dimension => 2,
    # COVER
    :cover_seed => 837387533,
    :cover => 25, # X: 35, Y: 25
    :iters => 10,
    :min_cluster_size => 72,
    # k-MEANS
    :c => 5, # 3
    :kmsample => 3,
    :components => 10,
    :prune => true,
    # GMM
    :covtype => "full", # X:full  Y:spherical
)

#===============================================================================
    Load data
===============================================================================#

using DelimitedFiles
TMP = readdlm("data/pendigits.tra", ',', Float64)
TST = readdlm("data/pendigits.tes", ',', Float64)
# TMP = readdlm("data/pendigits.tra", ',', Float32)
# TST = readdlm("data/pendigits.tes", ',', Float32)
X, L = TMP[:,1:16]', Int.(TMP[:,end]).+1
XT, LT = TST[:,1:16]', Int.(TST[:,end]).+1
# X64 = convert(Matrix{Float64}, X)
# XT64 = convert(Matrix{Float64}, XT)

# UMAP
using PyCall
if reduce_dim > 1
    UM = PyCall.pyimport("umap").UMAP(random_state=PARAMS[:experiment_seed], n_components=reduce_dim,
                                    n_neighbors=25, min_dist=0.0).fit(X')
    Y = UM.transform(X')'
    YT = UM.transform(XT')'
    X = convert(Matrix{Float64}, Y)
    XT = convert(Matrix{Float64}, YT)

#     PARAMS[:cover] = 25
end
PARAMS[:dimension] = d = size(X,1)
PARAMS[:covtype] = covtype
PARAMS[:cover] = covsize

#===============================================================================
  Start clustering
===============================================================================#

algos = Dict(
    1=>"plmc+kmeans",
    2=>"plmc+gmm",
    # 3=>"kmeans",
    4=>"gmm",
    5=>"dpgmm",
    # 6=>"kmeans-nmi",
    7=>"plmc+gmm-jl",
    8=>"gmm-jl",
    # 9=>"plmc+kmeans+mdl",
    # 10=>"plmc+gmm+mdl",
    # 11=>"plmc+gmm-jl+mdl"
    12=>"plmc+dpgmm",
)

RESULTS = Results()
for i in startindex:PARAMS[:iterations]

    tm = time()
    # PARAMS[:cover_seed] = 837387533 # reset
    PARAMS[:cover_seed] += 1

    Cgmmdl, _ = 2 ∈ keys(algos) ? with_logger(lgr) do
        metaclusters(X, L, PARAMS, PLMC.nml, GMMCover)
    end : (nothing, nothing)

    Cgmmdl_jl, _ = 7 ∈ keys(algos) ? with_logger(lgr) do
        metaclusters(X, L, PARAMS, PLMC.nml, GMMCoverJL)
    end : (nothing, nothing)

    Ckmmdl, _ = 1 ∈ keys(algos) ? with_logger(lgr) do
        metaclusters(convert(Matrix{Float64}, X), L, PARAMS, PLMC.nml, kMeansCover)
    end : (nothing, nothing)

    Ckm_nmi  = 6 ∈ keys(algos) ? with_logger(lgr) do
        clustering(X, L, :kmeansnmi, PARAMS)
    end : nothing

    Ckm  = 3 ∈ keys(algos) ?  with_logger(lgr) do
        clustering(X, L, :kmeansk, PARAMS)
    end : nothing

    Cgmm  = 4 ∈ keys(algos) ? with_logger(lgr) do
        clustering(X, L, :pygmm, PARAMS)
    end : nothing

    Cgmm_jl  = 8 ∈ keys(algos) ? with_logger(lgr) do
        clustering(X, L, :gmm, PARAMS)
    end : nothing

    Cdpgmm  = 5 ∈ keys(algos) ? with_logger(lgr) do
        clustering(X, L, :pydpgmm, PARAMS)
    end : nothing

    Cgmmdl2, _ = 10 ∈ keys(algos) ? with_logger(lgr) do
        metaclusters(X, L, PARAMS, PLMC.nml, GMMCover, PLMC.MDL)
    end : (nothing, nothing)

    Cgmmdl_jl2, _ = 11 ∈ keys(algos) ?  with_logger(lgr) do
        metaclusters(X, L, PARAMS, PLMC.nml, GMMCoverJL, PLMC.MDL)
    end : (nothing, nothing)

    Ckmmdl2, _ = 9 ∈ keys(algos) ? with_logger(lgr) do
        metaclusters(convert(Matrix{Float64}, X), L, PARAMS, PLMC.nml, kMeansCover, PLMC.MDL)
    end : (nothing, nothing)

    Cdpgmmdl, tmp = 12 ∈ keys(algos) ? with_logger(lgr) do
        metaclusters(X, L, PARAMS, PLMC.nml, DPGMMCover)
    end : (nothing, nothing)

    models = Dict(
        1=> Ckmmdl === nothing ? nothing : fit(NaiveBayesClassifier, Ckmmdl),
        2=> Cgmmdl === nothing ? nothing : fit(NaiveBayesClassifier, Cgmmdl),
        3=> Ckm === nothing ? nothing : fit(KNNClassifier, X, assignments(Ckm)),
        4=> Cgmm === nothing ? nothing : fit(NaiveBayesClassifier, Cgmm),
        5=> Cdpgmm === nothing ? nothing : fit(NaiveBayesClassifier, Cdpgmm),
        6=> Ckm_nmi === nothing ? nothing : fit(KNNClassifier, X, assignments(Ckm_nmi)),
        7=> Cgmmdl_jl === nothing ? nothing : fit(NaiveBayesClassifier, Cgmmdl_jl),
        8=> Cgmm_jl === nothing ? nothing : fit(NaiveBayesClassifier, Cgmm_jl),
        9=> Ckmmdl2 === nothing ? nothing : fit(NaiveBayesClassifier, Ckmmdl2),
        10=> Cgmmdl2 === nothing ? nothing : fit(NaiveBayesClassifier, Cgmmdl2),
        11=> Cgmmdl_jl2 === nothing ? nothing : fit(NaiveBayesClassifier, Cgmmdl_jl2),
        12=> Cdpgmmdl === nothing ? nothing : fit(NaiveBayesClassifier, Cdpgmmdl),
    )

    RESULTS.ASSIGN = String => Vector{Int}
    RESULTS.NMI = String => Float64
    RESULTS.NMI_TEST = String => Float64

    for (i,a) in algos
        M = models[i]
        M === nothing && continue
        RESULTS.NMI = a => mutualinfo(predict(M, X), L)
        RESULTS.NMI_TEST = a => mutualinfo(predict(M, XT), LT)
        RESULTS.ASSIGN = a => M.labels
    end

    RESULTS.ITER = i

    println("$i] elapsed: ", (time() - tm))
end

resfile = "pendigits-$d-$covtype$covsize"
# resfile = "pendigits-$d-kmeans"
saveresults(resfile, "PEN", PARAMS, RESULTS, true)

# [algos[k]=>(mutualinfo(predict(m, X), L), length(unique(m.labels))) for (k,m) in Dict(
#     1=>fit(NaiveBayesClassifier, Ckmmdl),
#     2=>fit(NaiveBayesClassifier, Cgmmdl),
#     7=>fit(NaiveBayesClassifier, Cgmmdl_jl),
#     12 => fit(NaiveBayesClassifier, Cdpgmmdl),
# )]
