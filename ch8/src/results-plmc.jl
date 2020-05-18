using Statistics
using PLMC
using BSON
using Random
using ClusterComplex
using ComputationalHomology

function saveClustering(fname, C, AGG, J)
    ctx = Dict(
        :clusters => C.clusters,
        :eps => C.ϵ,
        :assignments => C.models.assignments,
        :mean => hcat(mean.(C.models.models)...),
        :cov => cat(cov.(C.models.models)..., dims=3),
        :scores => J,
        :agglomeration => Dict(
            :clusters => AGG.clusters,
            :costs => AGG.costs,
            :mergers => AGG.mergers,
        )
    )
    open(fname, "w") do io
        BSON.bson(io, ctx)
    end
end

function loadClustering(fname)
    ctx = BSON.load(fname)
    μ = ctx[:mean]
    Σ = ctx[:cov]
    mvs = [MvNormal(μ[:,i], Σ[:,:,i],) for i in 1:size(Σ,3)]
    models = ModelClusteringResult(mvs, ctx[:assignments])
    C = PLMClusteringResult(models, ctx[:clusters], SimplicialComplex(), ctx[:eps])
    J = ctx[:scores]
    agg = ctx[:agglomeration]
    cls = agg[:clusters]
    mrgs= agg[:mergers]
    AGG = PLMC.Agglomeration(cls, mrgs, agg[:costs])
    C, AGG, J
end
