using Clustering: kmeans, dbscan, mutualinfo, assignments, randindex, ClusteringResult
using ClusterComplex: CustomClusteringResult
using PyCall
using Random: seed!
using LinearAlgebra

sklearncluster = pyimport("sklearn.cluster")
sklearnmixture = pyimport("sklearn.mixture")

function clustering(X, L, method, PARAMS)
    N = size(X,2)
    C = if method == :kmeansnmi
        Ctmp = nothing
        maxmi = 0
        for k in 2:4*PARAMS[:c]
            for i in 0:(get(PARAMS, :kmsample, 1)-1)
                seed!(PARAMS[:cover_seed]+i);
                C = kmeans(X, k)
                mi = mutualinfo(C, L)
                @debug "k-Means" k=k NMI=mi
                if mi > maxmi
                    maxmi = mi
                    Ctmp = C
                end
            end
        end
        Ctmp
    elseif method == :kmeans
        Ctmp = nothing
        mincost = Inf
        for k in 2:4*PARAMS[:c]
            for i in 0:(get(PARAMS, :kmsample, 1)-1)
                seed!(PARAMS[:cover_seed]+i);
                C = kmeans(X, k)
                mc = C.totalcost
                @debug "k-Means" k=k Cost=mc
                if mc < mincost
                    mincost = mc
                    Ctmp = C
                end
            end
        end
        Ctmp
    elseif method == :kmeansk
        Ctmp = nothing
        mincost = Inf
        for i in 0:(get(PARAMS, :kmsample, 1)-1)
            seed!(PARAMS[:cover_seed]+i);
            C = kmeans(X, PARAMS[:components])
            mc = C.totalcost
            @debug "k-Means" k=PARAMS[:components] Cost=mc
            if mc < mincost
                mincost = mc
                Ctmp = C
            end
        end
        Ctmp
    elseif method == :meanshift
        bandwidth = sklearncluster.estimate_bandwidth(X', quantile=PARAMS[:quantile])
        ms = sklearncluster.MeanShift(bandwidth=bandwidth)
        ms.fit(X')
        CustomClusteringResult(map(x->x+1, ms.labels_), ms.cluster_centers_')
    elseif  method == :dbscan
        seed!(PARAMS[:cover_seed]);
        cs = dbscan(X, PARAMS[:radius], min_cluster_size=PARAMS[:min_cluster_size])
        assign = zeros(Int, N)
        for (i,c) in enumerate(cs)
            assign[c.core_indices] .= i
        end
        noise = maximum(assign)+1
        assign[assign .== 0] .= noise
        CustomClusteringResult(assign)
    elseif  method == :spectral
        assign = nothing
        maxmi = 0
        for k in 2:4*PARAMS[:c]
            ms = sklearncluster.SpectralClustering(n_clusters=k, random_state=PARAMS[:cover_seed])
            ms.fit(X')
            A = map(x->x+1, ms.labels_)
            mi = mutualinfo(A, L)
            if mi > maxmi
                maxmi = mi
                assign = A
            end
        end
        CustomClusteringResult(assign)
    elseif method == :pygmm
        covtype = get(PARAMS, :covtype, "full")
        gmm = sklearnmixture.GaussianMixture(n_components=PARAMS[:components], covariance_type=covtype,
                                             n_init=get(PARAMS, :kmsample, 1), random_state=PARAMS[:cover_seed])
        gmm.fit(X')
        A = gmm.predict(X').+1
        # CustomClusteringResult(A, ms.means_')
        ms = [ MvNormal(gmm.means_[i,:],
                            covtype == "diag" ?      Diagonal(sqrt.(gmm.covariances_[i,:])) :
                            covtype == "spherical" ? sqrt(gmm.covariances_[i]) :
                                                     Symmetric(gmm.covariances_[i,:,:]')
                        ) for i in 1:size(gmm.means_,1)]
        ModelClusteringResult(ms, A)
    elseif method == :pydpgmm
        covtype = get(PARAMS, :covtype, "full")
        gmm = sklearnmixture.BayesianGaussianMixture(covariance_type=covtype, n_components=PARAMS[:cover], #  max_iter=200,
                                                     n_init=get(PARAMS, :kmsample, 1), init_params="random", # mean_precision_prior=1e-2,
                                                     weight_concentration_prior_type="dirichlet_process", # verbose=2,
                                                     weight_concentration_prior=get(PARAMS, :wcprior, 1/PARAMS[:components]),
                                                     random_state=PARAMS[:cover_seed])
        gmm.fit(X')
        A = gmm.predict(X').+1
        # println(gmm.get_params())
        # CustomClusteringResult(A, ms.means_')
        ms = [ MvNormal(gmm.means_[i,:],
                covtype == "diag" ?      Diagonal(sqrt.(gmm.covariances_[i,:])) :
                covtype == "spherical" ? sqrt(gmm.covariances_[i]) :
                                         Symmetric(gmm.covariances_[i,:,:]')
                ) for i in unique(A)]
        ModelClusteringResult(ms, A)
    elseif method == :gmm
        Random.seed!(PARAMS[:cover_seed]);
        covtype = get(PARAMS, :covtype, "full")
        CT = covtype =="spherical" ? IsoNormal : (covtype =="diag" ? DiagNormal : FullNormal)
        mmx = fit_mm(CT, X, PARAMS[:components])
        A = MultivariateMixtures.predict(mmx, X)
        ModelClusteringResult(components(mmx), A)
    else
        error("Uknwown method: $method")
    end
    return C
end

function saveindecies!(name, L, C, nmi, ridx, assign)
    nmi[name] = try
        mutualinfo(L, C)
    catch
        0.0
    end
    ridx[name] = try
        collect(randindex(L, C))
    catch
        zeros(4)
    end
    assign[name] = try
        assignments(C)
    catch
        Int[]
    end
end

function saveindecies(name, L, C)
    nmi = try
        mutualinfo(L, C)
    catch
        0.0
    end
    ridx = try
        collect(randindex(L, C))
    catch
        zeros(4)
    end
    assign = try
        assignments(C)
    catch
        Int[]
    end
    (name=>nmi, name=>ridx, name=>assign)
end

function confusion(L1, L2; sorted=false)
    n, m = length(unique(L1)), length(unique(L2))
    cfm = zeros(Int, n, m)
    for (i,j) in zip(L1,L2)
        cfm[i,j] += 1
    end
    idxs = collect(1:m)
    if sorted
        idxs = Int[]
        cidxs = Set(1:m)
        for i in 1:n
            tmp = cidxs |> collect
            maxc = findmax(cfm[i,tmp]) |> last
            push!(idxs, pop!(cidxs, tmp[maxc]))
        end
        if length(cidxs) > 0
            append!(idxs, collect(cidxs))
        end
    end
    return cfm[:,idxs]
end
confusion(C::T, L; sorted=false) where {T<:ClusteringResult} = confusion(assignments(C), L, sorted=sorted)
confusion(C1::T1, C2::T2; sorted=false) where {T1<:ClusteringResult, T2<:ClusteringResult} =
    confusion(assignments(C1), assignments(C2), sorted=sorted)
