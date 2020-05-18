using PLMC: PLMClusteringResult
using ClusterComplex: ModelClusteringResult
using MultivariateMixtures: FinateMixtureModel
using Distributions: MixtureModel
import StatsBase.predict

include("classifiers.jl")

function fit(::Type{NaiveBayesClassifier}, mc::ModelClusteringResult)
    n = length(mc.models)
    models = Dict(i=>mc.models[i] for i in 1:n)
    NaiveBayesClassifier(models, mc.assignments)
end

function fit(::Type{NaiveBayesClassifier}, plc::PLMClusteringResult)
    n = length(plc.clusters)
    cnts = counts(plc.models)
    # println("cnt=",cnts)
    models = Dict{Int,MixtureModel{Multivariate,Continuous,Distribution{Multivariate,Continuous},Float64}}()
    for i in 1:n
        p = plc.clusters[i]
        w = cnts[p]./sum(cnts[p])
        # println("$i: p=$p, w=$w")
        any(isnan, w) && continue
        mm = MixtureModel(plc.models.models[p], w)
        push!(models, i=>mm)
    end
    # models2 = Dict(i=>MixtureModel(plc.models, plc.clusters[i]) for i in 1:n)
    NaiveBayesClassifier(models, assignments(plc))
end

function fit(::Type{MLClassifier}, mc::ModelClusteringResult)
    n = length(mc.models)
    models = Dict(i=>mc.models[i] for i in 1:n)
    MLClassifier(models, mc.assignments)
end

function fit(::Type{MLClassifier}, plc::PLMClusteringResult)
    n = length(plc.clusters)
    models = Dict(i=>MixtureModel(plc.models, plc.clusters[i]) for i in 1:n)
    MLClassifier(models, assignments(plc))
end

predict(classifier::Union{MixtureModel,FinateMixtureModel}, data::AbstractVector) =
    findmax(componentwise_logpdf(classifier, data))[end]
predict(classifier::Union{MixtureModel,FinateMixtureModel}, data::AbstractMatrix; dims=2) =
    map(ci->ci[2], findmax(componentwise_logpdf(classifier, (dims == 1 ? data' : data)), dims=2)[end])
