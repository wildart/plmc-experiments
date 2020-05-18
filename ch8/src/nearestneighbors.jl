using ManifoldLearning

# import FLANN
# function knn_flann(X::AbstractMatrix{T}, k::Int=12) where T<:Real
#     params = FLANN.FLANNParameters()
#     E, D = FLANN.knn(X, X, k+1, params)
#     sqrt.(@view D[2:end, :]), @view E[2:end, :]
# end

using NearestNeighbors: NearestNeighbors

struct KDTree <: ManifoldLearning.AbstractNearestNeighbors
    k::Integer
    fitted::NearestNeighbors.KDTree
end
StatsBase.fit(::Type{KDTree}, X::AbstractMatrix{T}, k::Integer) where {T<:Real} = KDTree(k, NearestNeighbors.KDTree(X))
Base.show(io::IO, NN::KDTree) = print(io, "KDTree(k=$(NN.k))")
function ManifoldLearning.knn(NN::KDTree, X::AbstractVecOrMat{T}; self=false) where {T<:Real}
    m, n = size(X)
    k = NN.k
    @assert n > k "Number of observations must be more then $(k)"

    idxs, dist = NearestNeighbors.knn(NN.fitted, X, k+1, true)
    D = Array{T}(undef, k, n)
    E = Array{Int32}(undef, k, n)
    for i in eachindex(idxs)
        E[:, i] = idxs[i][2:end]
        D[:, i] = dist[i][2:end]
    end
    return D, E
end
