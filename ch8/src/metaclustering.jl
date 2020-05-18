using PLMC, Clustering, Random, Distributions
using StatsBase

function findscore(J)
    # J = map(j->floor(Int, j), J)
    J[1] = J[end] = maximum(J)
    mx1 = PLMC.findglobalmin(J, 1e-3)
    mx2 = PLMC.findglobalmin2(J)
    return max(mx1, mx2)
end

function metaclusters(data, lbls, PARAMS, score, coverType=kMeansCover, heuristic=PLMC.Topological)
    # construct cover
    flt, MC = fit(coverType, data, PARAMS[:cover], seed=PARAMS[:cover_seed], iters=PARAMS[:iters],
                  minclustersize=PARAMS[:min_cluster_size], covtype=PARAMS[:covtype],
                  prune = get(PARAMS, :prune, true))
    # perform agglomeration
    C, AGG, J = plmc(heuristic, data, flt, MC, score=score, find=findscore) # find=PLMC.findglobalmin)
    C2 = nothing
    mi = 0.0
    NMIs = Float64[]
    for i in 1:length(AGG.clusters)
        tmp, _ = plmc(AGG, C.models, data, aggidx=i)
        tmpmi = mutualinfo(tmp, lbls)
        push!(NMIs, tmpmi)
        if mi < tmpmi
            C2 = tmp
            mi = tmpmi
        end
        @debug "Agglomeration" i=i NMI=tmpmi Score=J[i] Clusters = AGG.clusters[i]
    end
    @debug "Metaclustering" C=C
    C, C2, AGG, J, NMIs, flt
end

function StatsBase.predict(cl::PLMClusteringResult, data::AbstractMatrix)
    mdls = modelclass(cl.models, cl.clusters)
    assignments(data, mdls)
end

function datasample(features, labels, ssize=500; seed=1, digits=collect(0:9))
    Random.seed!(seed)
    idxs = Int[]
    lidxs = Int[]
    for d in digits
        idx = findall(labels.==d)
        append!(idxs, idx[rand(1:length(idx), ssize)])
        append!(lidxs, fill(d+1, ssize))
    end
    p = randperm(length(idxs))
    convert(Matrix{Float64}, features[:,idxs[p]]), lidxs[p]
end
