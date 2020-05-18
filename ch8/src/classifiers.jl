using NearestNeighbors: NearestNeighbors
using Distributions
using StatsBase

import StatsBase: fit, predict

abstract type AbstractClassifier end

struct KNNClassifier{T} <: AbstractClassifier
    model::NearestNeighbors.KDTree
    labels::Vector{T}
end

function fit(::Type{KNNClassifier}, data::AbstractMatrix{T},
             labels::AbstractVector{S}) where {T, S}
    kdtree = NearestNeighbors.KDTree(data; leafsize = 10)
    KNNClassifier{S}(kdtree, labels)
end

function predict(classifier::KNNClassifier, data::AbstractVector; dims=2)
    idx = first(NearestNeighbors.nn(classifier.model, data))
    return classifier.labels[idx]
end

struct NaiveBayesClassifier{D<:ContinuousDistribution, T} <: AbstractClassifier
    model::Dict{T, D}
    labels::Vector{T}
end

function fit(::Type{NaiveBayesClassifier{D}}, data::AbstractMatrix{T},
             labels::AbstractVector{S}; dims=1) where {D <: ContinuousDistribution, T, S}
    models = Dict{T, D}()
    for l in unique(labels)
        pdata = selectdim(data, dims, labels.==l)
        models[l] = fit_mle(D, pdata)
    end
    NaiveBayesClassifier{D,S}(models, labels)
end

function predict(classifier::NaiveBayesClassifier, data::AbstractVector)
    n = length(classifier.labels)
    p = log.(counts(classifier.labels) ./ n)
    lbls = keys(classifier.model)
    lpp = [l=>p[l]+logpdf(classifier.model[l], data) for l in lbls]
    v, idx = findmax(map(last, lpp))
    return first(lpp[idx])
end

function predict(classifier::NaiveBayesClassifier, data::AbstractMatrix; dims=2)
    n = length(classifier.labels)
    p = log.(counts(classifier.labels) ./ n)
    lbls = keys(classifier.model) |> collect
    lpp = hcat((p[l].+logpdf(classifier.model[l], data) for l in lbls)...)
    return [lbls[findmax(r)[2]] for r in eachrow(lpp)]
end

struct MLClassifier{D<:ContinuousDistribution, T} <: AbstractClassifier
    model::Dict{T, D}
    labels::Vector{T}
end

function fit(::Type{MLClassifier{D}}, data::AbstractMatrix{T},
             labels::AbstractVector{S}; dims=1) where {D <: ContinuousDistribution, T, S}
    models = Dict{T, D}()
    for l in unique(labels)
        pdata = selectdim(data, dims, labels.==l)
        models[l] = fit_mle(D, pdata)
    end
    MLClassifier{D,S}(models, labels)
end

function predict(classifier::MLClassifier, data::AbstractVector)
    lpp = [l=>logpdf(m, data) for (l,m) in classifier.model]
    v, idx = findmax(map(last, lpp))
    return first(lpp[idx])
end

function predict(classifier::MLClassifier, data::AbstractMatrix; dims=2)
    lbls = keys(classifier.model) |> collect
    lpp = hcat((logpdf(classifier.model[l], data) for l in lbls)...)
    return [lbls[findmax(r)[2]] for r in eachrow(lpp)]
end

function predict(classifier::AbstractClassifier, data::AbstractMatrix; dims=2)
    n = size(data, dims)
    X = dims == 1 ? data' : data
    return [predict(classifier, selectdim(data, dims, i)) for i in 1:n]
end

function score(classifier::C, data::AbstractMatrix{T}, labels::AbstractVector; dims=2) where {C <: AbstractClassifier, T}
    predicted = predict(classifier, data, dims=dims)
    sum(predicted .== labels) / size(data, dims)
end
