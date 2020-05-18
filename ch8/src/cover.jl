using MultivariateStats
using MultivariateMixtures
using Random: MersenneTwister, seed!
using Clustering: kmeans, assignments, ClusteringResult
using LMCLUS
using ClusterComplex: clustercomplex, ModelClusteringResult, MvNormal, model
using Statistics
using ComputationalHomology: filtration
using LinearAlgebra
import StatsBase: fit
using PyCall

sklearnmixture = pyimport("sklearn.mixture")

abstract type AbstractCover end
struct kMeansCover <: AbstractCover end
struct MFACover <: AbstractCover end
struct LMCLUSCover <: AbstractCover end
struct GMMCover <: AbstractCover end
struct GMMCoverJL <: AbstractCover end
struct DPGMMCover <: AbstractCover end

function fit(::Type{kMeansCover}, data::AbstractMatrix{T}, sz::Int;
             seed=-1, r=T(Inf), maxoutdim=1, iters=1,
             minclustersize=1, prune=true, kwargs...) where {T <: AbstractFloat}
    d = size(data,1)
    if seed >= 0
        seed!(seed);
    end
    C = nothing
    for i in 1:iters
        tmp = kmeans(data, sz)
        if C === nothing || C.totalcost < tmp.totalcost
            C = tmp
        end
    end
    cnts = counts(C)
    @debug "Clusters" Cost=C.totalcost Size=cnts
    # cplx, w, MC = clustercomplex(data, C, r, maxoutdim=min(maxoutdim, d-1), expansion=:inductive)

    ms, assign = constructmodels(data, C, prune=prune, minclustersize=minclustersize)
    cplx, w, MC = clustercomplex(data, ms, r, maxoutdim=min(maxoutdim, d-1), expansion=:inductive, assignments=assign)
    return filtration(cplx, w), MC
end

function fit(::Type{MFACover}, data::AbstractMatrix{T}, m::Int, k::Int=1;
             seed=-1, tol=1e-3, maxiter=1000,
             r=T(Inf), maxoutdim=1, minclustersize=1, prune=true, kwargs...) where {T}
    d = size(data,1)
    if seed >= 0
        seed!(seed);
    end
    C = kmeans(data, m)
    @debug "Clusters" Size=counts(C)

    # collect cluster statistics
    mvs = [model(data, points(C, i)) for i in 1:nclusters(C)]
    Σs = cat(cov.(mvs)..., dims=3)
    μs = cat(mean.(mvs)..., dims=2)
    # Σs = cat((cov(view(data, : , points(C, i)), dims=2) for i in 1:nclusters(C))..., dims=3)
    # μs = C.centers

    # construct mixture
    mfa = fit_mm(FactorAnalysis, data; m=m, k=k, μs=μs, Σs=Σs, tol=tol, maxiter=maxiter)

    # generate model clustering
    ms = [MultivariateMixtures.MvNormal(mean(m), cov(m)) for m in MultivariateMixtures.components(mfa)]
    if prune
        # map(diffentropy∘cov, ms) |> println
        # filter!(m->diffentropy(cov(m)) < 0, ms) # remove models with large covariance
        prunemodiles!(ms, data, minclustersize)
    end
    cplx, w, MC = clustercomplex(data, ms, r, maxoutdim=min(maxoutdim, d-1), expansion=:inductive, assignments=assignments(data, ms))
    return filtration(cplx, w), MC
end

function fit(::Type{LMCLUSCover}, data::AbstractMatrix{T}, p::LMCLUS.Parameters;
             seed=-1, r=T(Inf), maxoutdim=1) where {T}
    d = size(data,1)
    rng = seed >= 0 ? MersenneTwister(Int(seed)) : MersenneTwister()

    # setup clustering parameters
    C = LMCLUSResult(lmclus(data, p, [rng])...)
    @debug "Clusters" Size=counts(C)

    # fix clustering
    # fix_defected!(C, data, p, rng)

    # construct models
    ms = Array{MvNormal}(undef, nclusters(C))
    assign = assignments(C)
    # println(nclusters(C), counts(C))
    nids = Int[]
    amap = Dict{Int,Int}()
    outliers = Int[]
    c = 0
    for i in 1:nclusters(C)
        ms[i] = model(data, points(C.manifolds[i]))
        de = diffentropy(cov(ms[i]))
        @debug "Entropy" i=i E=de
        if de > 0
            append!(outliers, points(C.manifolds[i]))
            push!(nids, i)
        else
            c+=1
        end
        amap[i] = c
    end
    deleteat!(ms, nids)

    # reassign outliers
    aoutliers = assignments(view(data, :, outliers), ms)
    for (i, a) in zip(outliers, aoutliers)
        assign[i] = a
    end
    # unique(assign) |> println
    assign = map(i->amap[i], assign)
    # unique(assign) |> println

    # construct model-based clusters
    cplx, w, MC = clustercomplex(data, ms, r, maxoutdim=min(maxoutdim, d-1), expansion=:inductive, assignments=assign)
    return filtration(cplx, w), MC
end

function fit(::Type{GMMCover}, data::AbstractMatrix{T}, m::Int;
             seed=-1, tol=1e-3, maxiter=100, init=3, covtype="diag",
             r=T(Inf), maxoutdim=1, minclustersize=1, prune=true, kwargs...) where {T<:AbstractFloat}
    d = size(data,1)
    if seed >= 0
        seed!(seed);
    end
    gmm = sklearnmixture.GaussianMixture(n_components=m, covariance_type=covtype,
                                         n_init=init, random_state=seed, max_iter=maxiter)
    gmm.fit(data')
    A = gmm.predict(data').+1
    ms = MvNormal[]
    for i in 1:size(gmm.means_,1)
        mu = convert(Vector{T}, gmm.means_[i,:])
        sigma = if covtype == "diag"
            Diagonal(convert(Vector{T}, gmm.covariances_[i,:]))
        elseif covtype == "spherical"
            T(gmm.covariances_[i])
        else
            sigma = Symmetric(convert(Matrix{T}, gmm.covariances_[i,:,:]'))
            if !issuccess(cholesky(sigma, check=false))
                d = size(sigma,1)
                sigma[1:d+1:end] .+= 1e-5
            end
            # Distributions.PDMat(F)
            sigma
        end
        mvn = try
            MvNormal(mu, sigma)
        catch
            println("S=", sigma)
            MvNormal(mu, Distributions.PDMat(cholesky(sigma, check=false)))
        end
        push!(ms, mvn)
    end
    # C = ModelClusteringResult(ms, A)
    @debug "Clusters" Size=counts(A)
    prune && prunemodiles!(ms, data, minclustersize);

    cplx, w, MC = clustercomplex(data, ms, r, maxoutdim=min(maxoutdim, d-1), expansion=:inductive, assignments=assignments(data, ms))
    return filtration(cplx, w), MC
end

function fit(::Type{GMMCoverJL}, data::AbstractMatrix{T}, m::Int;
             seed=-1, tol=1e-4, maxiter=100, init=3, r=T(Inf),
             covtype="diag", maxoutdim=1, minclustersize=1, prune=true, kwargs...) where{T}
    d = size(data,1)
    if seed >= 0
        seed!(seed);
    end
    CT = covtype =="spherical" ? IsoNormal : (covtype =="diag" ? DiagNormal : FullNormal)
    gmm = fit_mm(CT, data, m, tol=tol, maxiter=maxiter)

    A = MultivariateMixtures.predict(gmm, data)
    @debug "Clusters" Size=counts(A)
    ms = components(gmm)
    prune && prunemodiles!(ms, data, minclustersize);

    cplx, w, MC = clustercomplex(data, ms, r, maxoutdim=min(maxoutdim, d-1),
                                 expansion=:inductive, assignments=assignments(data, ms))
    return filtration(cplx, w), MC
end

function fit(::Type{DPGMMCover}, data::AbstractMatrix{T}, m::Int;
             seed=-1, tol=1e-3, maxiter=100, init=3, covtype="diag",
             r=T(Inf), maxoutdim=1, minclustersize=1, prune=true, kwargs...) where {T<:AbstractFloat}
    d = size(data,1)
    if seed >= 0
        seed!(seed);
    end
    gmm = sklearnmixture.BayesianGaussianMixture( n_components=m, covariance_type=covtype, #  max_iter=200,
                                                 n_init=1, init_params="random", # mean_precision_prior=1e-2,
                                                 weight_concentration_prior=1/m, random_state=seed)
    gmm.fit(data')
    A = gmm.predict(data').+1
    ms = MvNormal[]
    for i in unique(A)
        mu = convert(Vector{T}, gmm.means_[i,:])
        sigma = if covtype == "diag"
            Diagonal(convert(Vector{T}, gmm.covariances_[i,:]))
        elseif covtype == "spherical"
            T(gmm.covariances_[i])
        else
            sigma = Symmetric(convert(Matrix{T}, gmm.covariances_[i,:,:]'))
            if !issuccess(cholesky(sigma, check=false))
                d = size(sigma,1)
                sigma[1:d+1:end] .+= 1e-5
            end
            # Distributions.PDMat(F)
            sigma
        end
        mvn = try
            MvNormal(mu, sigma)
        catch
            # println("S=", sigma)
            MvNormal(mu, Distributions.PDMat(cholesky(sigma, check=false)))
        end
        push!(ms, mvn)
    end
    # C = ModelClusteringResult(ms, A)
    @debug "Clusters" Size=counts(A)
    prune && prunemodiles!(ms, data, minclustersize);

    cplx, w, MC = clustercomplex(data, ms, r, maxoutdim=min(maxoutdim, d-1), expansion=:inductive, assignments=assignments(data, ms))
    return filtration(cplx, w), MC
end

function prunemodiles!(ms, data, minclustersize=1)
    A = assignments(data, ms)
    rmvidxs = findall(counts(A) .< minclustersize)
    deleteat!(ms, rmvidxs);
end

function constructmodels(data::AbstractMatrix{T}, cidxs::Vector{Vector{Int}};
                         minclustersize=1, prune=true) where {T}
    # construct models
    ms = MvNormal[]
    nids = Int[]
    # i, idx = 75, idxs[75]
    for (i,idx) in enumerate(cidxs)
        m = if length(idx) > minclustersize
            model(data, idx)
        else
            nothing
        end
        de = (m === nothing) ? -Inf : diffentropy(cov(m))
        @debug "Entropy" i=i E=de size=length(idx)
        if isinf(de) #|| (prune &&  (size(data,1) > 3 ? de > 0 : de < 0))
            push!(nids, i)
        else
            push!(ms, m)
        end
    end
    # remove unfit models
    @debug "Remove outliers" outliers=nids
    # deleteat!(ms, nids)

    # reassign ponts
    return ms, assignments(data, ms)
end

function constructmodels(data::AbstractMatrix{T}, clst::ClusteringResult;
                         minclustersize=1, prune=true) where {T}
    assign = assignments(clst)
    cidxs = [findall(assign .== i) for i in 1:nclusters(clst)]
    constructmodels(data, cidxs, prune=prune, minclustersize=minclustersize)
end

function diffentropy(Σ)
    d = size(Σ,1)
    try
        (log(det(Σ))+d*log(2π)+1)/2
    catch
        @warn "|Σ|" det(Σ)
        Inf
    end
end

function cover(X, method, PARAMS)
    d = size(X,1)
    cnum = PARAMS[:cover]*PARAMS[:c] # (expected) number of clusters
    if method == :kmeans
        return fit(kMeansCover, X, cnum; seed=PARAMS[:cover_seed])
    elseif method == :mfa
        return fit(MFACover, X, cnum, PARAMS[:factors]; seed=PARAMS[:cover_seed], tol=PARAMS[:mfa_tol])
    elseif method == :lmclus
        # setup clustering parameters
        p = LMCLUS.Parameters(d-1)
        p.best_bound = PARAMS[:best_bound]
        p.number_of_clusters = cnum
        p.sampling_heuristic = PARAMS[:sampling_heuristic]
        p.sampling_factor = PARAMS[:sampling_factor]
        p.basis_alignment = PARAMS[:basis_alignment]
        p.bounded_cluster = PARAMS[:bounded_cluster]
        p.min_cluster_size = PARAMS[:min_cluster_size]
        p.bounded_cluster = true
        p.mdl = true

        return fit(LMCLUSCover, X, p; seed=PARAMS[:cover_seed])
    else
        error("Uknwown method: $method")
    end
end

# Detect defected clusters by looking at in-manifold subspace distributions of distances
function fix_defected!(C, X, p, rng; threshold=1.1)
    # println("Old: ", length(C.manifolds))
    # manifold with gaps inside
    c = [distance_to_manifold(view(X, :,points(c)), c, ocss=true) |> maximum for c in C.manifolds]
    defected = findall(c .> threshold)
    # noise
    noise = findall(s->criteria(s) == 0.0, C.separations)
    # form new dataset
    append!(defected, noise)
    defected_idxs = vcat((points(C.manifolds[i]) for i in unique(defected))...)
    deleteat!(C.manifolds, unique(defected))
    deleteat!(C.separations, unique(defected))
    # cluster again
    p.min_cluster_size = p.min_cluster_size >> 1 # half cluster size
    Cfix, Sfix = lmclus(view(X,:,defected_idxs), p, [rng])
    for (c,s) in zip(Cfix,Sfix)
        c.points = defected_idxs[points(c)]
        push!(C.manifolds, c)
        push!(C.separations, s)
    end
    # println("New: ", length(C.manifolds))
    return C
end

