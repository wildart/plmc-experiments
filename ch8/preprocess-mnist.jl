include("src/conda.jl")
using PyCall
umap = PyCall.pyimport("umap")

using BSON
using MLDatasets

const seed = 42
const min_dists = [0.1, 0.2] # 0.0, 0.05,

# process args
fashion = 0
if length(ARGS) > 0
    fashion = parse(Int, ARGS[1])
    println("Fashion-MNIST: $fashion")
end

for min_dist in min_dists
    features, labels = if fashion == 0
        MLDatasets.MNIST.traindata(Float64)
    else
        MLDatasets.FashionMNIST.traindata(Float64)
    end
    tfeatures, tlabels = if fashion == 1
        MLDatasets.MNIST.testdata(Float64)
    else
        MLDatasets.FashionMNIST.testdata(Float64)
    end
    X = reshape(features,784,60000)
    T = reshape(tfeatures,784,10000)

    prefix = fashion == 0 ? "" : "f"

    for d in [2,8,16], n in vcat(3:10, 12:2:30) #
        println("d: $d, n: $n")

        UM = umap.UMAP(random_state=seed, n_components=d, n_neighbors=n, min_dist=min_dist).fit(X')
        Z = UM.transform(X')'
        ZT = UM.transform(T')'

        open("data/umap-md-$min_dist/$(prefix)mnist-umap-d$d-n$n.bson", "w") do io
            ctx = Dict{Symbol,Any}()
            ctx[:D] = convert(Matrix{Float64},Z)
            ctx[:T] = convert(Matrix{Float64},ZT)
            ctx[:L] = labels
            ctx[:LT] = tlabels
            BSON.bson(io, ctx)
        end
        UM = nothing
        Z = nothing
        ZT = nothing
        GC.gc()

    end
end
