# MNIST Data

using Logging
lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Info);
# lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Debug);
# Logging.disable_logging(Logging.BelowMinLevel);

include("src/conda.jl")
include("src/cover.jl")
include("src/clustering.jl")
include("src/metaclustering.jl")
include("src/results.jl")
include("src/classifiers-plmc.jl")

# process args
fashion = 0
datadir = "data/umap-md-0.0"
sample_size = 500
if length(ARGS) > 0
    fashion = parse(Int, ARGS[1])
    println("Fashion-MNIST: $fashion")
    if length(ARGS) > 1
        datadir = ARGS[2]
        println("Location: $datadir")
        if length(ARGS) > 2
            sample_size = parse(Int, ARGS[3])
            println("Sample/digit: $sample_size")
        end
    end
end

## parameters
PARAMS = Dict(
    :iterations => 100,
    :experiment_seed => 837387533,
    :sample => 1000,
    # COVER
    :cover_seed => 837387533,
    :cover => 30,
    :iters => 10,
    :min_cluster_size => 32,
    # k-MEANS
    :c => 3,
    :kmsample => 3,
    :components => 10,
    # GMM
    :covtype => "full",
)


RESULTS = Results()

# d, n = 2, 20 #6 #4 # 14 # mnist
d, n = 16, 8 #6 # 14 # fmnist
dsname = fashion == 0 ? "mnist" : "fmnist"
ds = "$datadir/$dsname-umap-d$d-n$n.bson"

# Load data
ctx = BSON.load(ds)
X, T = ctx[:D], ctx[:T]
L, LT = ctx[:L].+1, ctx[:LT].+1

PARAMS[:sample] = sample_size
PARAMS[:prune] = true #false
# PARAMS[:wcprior] = 1.0

algos = Dict(1=>"plmc+kmeans", 2=>"plmc+gmm", 3=>"kmeans", 4=>"gmm", 5=>"dpgmm")

for i in 1:PARAMS[:iterations]

    tm = time()

    # generate sampe
    S, LS = datasample(X, L, PARAMS[:sample], seed=PARAMS[:experiment_seed]-i-1, digits=collect(1:10))

    Ckmmdl, C2, AGG, J, NMIs, flt = with_logger(lgr) do
        # metaclusters(S, LS, PARAMS, PLMC.nml, MFACover)
        # metaclusters(S, LS, PARAMS, PLMC.nml, GMMCover)
        # metaclusters(S, LS, PARAMS, PLMC.refinedmdl)
        metaclusters(S, LS, PARAMS, PLMC.nml)
    end

    Cgmmdl, _ = with_logger(lgr) do
        metaclusters(S, LS, PARAMS, PLMC.nml, GMMCover)
    end

    Ckm  = with_logger(lgr) do
        clustering(S, LS, :kmeansk, PARAMS)
    end
    # Ckmm = ModelClusteringResult(constructmodels(S, Ckm)...)

    Cgmm  = with_logger(lgr) do
        clustering(S, LS, :pygmm, PARAMS)
    end

    Cdpgmm  = with_logger(lgr) do
        clustering(S, LS, :pydpgmm, PARAMS)
    end

    # mfa  = with_logger(lgr) do
    #     fit_mm(FactorAnalysis, S; m=10, k=2)
    # end
    # Cmfa = ModelClusteringResult(convert.(MvNormal, components(mfa)), predict(mfa, S))

    # gmm  = with_logger(lgr) do
    #     fit_mm(FullNormal, S, 10)
    # end
    # Cgmm = ModelClusteringResult(components(gmm), predict(gmm, S))

    models = Dict(
        1=>fit(NaiveBayesClassifier, Ckmmdl),
        2=>fit(NaiveBayesClassifier, Cgmmdl),
        3=>fit(KNNClassifier, S, assignments(Ckm)),
        4=>fit(NaiveBayesClassifier, Cgmm),
        5=>fit(NaiveBayesClassifier, Cdpgmm),
    )

    # mdl3 = fit(KNNClassifier, S, assignments(Ckmm))
    # mdl2 = fit(NaiveBayesClassifier, C2)

    RESULTS.NMI_SAMPLE = String => Float64
    RESULTS.NMI = String => Float64
    RESULTS.NMI_TEST = String => Float64

    # [a => mutualinfo(predict(eval(Symbol("mdl$i")), S), LS) for (i,a) in algos]
    for (i,a) in algos
        M = models[i]
        RESULTS.NMI_SAMPLE = a => mutualinfo(predict(M, S), LS)
        RESULTS.NMI = a => mutualinfo(predict(M, X), L)
        RESULTS.NMI_TEST = a => mutualinfo(predict(M, T), LT)
    end

    RESULTS.ITER = i

    println("$i] elapsed: ", (time() - tm))
end

saveresults("$dsname-$d-$n-$sample_size-clustering", "MNIST", PARAMS, RESULTS, true)
