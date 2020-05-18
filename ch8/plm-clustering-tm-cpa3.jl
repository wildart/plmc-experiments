# Comparative Performance Analysis

include("src/conda.jl") # setup conda env settings

include("src/results.jl")
include("src/cover.jl")
include("src/clustering.jl")

## packages
using NMoons
using PLMC
using Random

## parameters
PARAMS = Dict(
    :iterations => 100,
    :experiment_seed => 837387533,
    # DATASET
    :data_seed => 837387533,
    :c => 2, # [2, 3, 5, 7]
    :ε => 0.1, # [0.1, 0.2, 0.3]
    :d => 2, # [2, 3, 7, 14]
    :m => 500,
    :r => 1,
    :repulse => (-0.25, -0.25),
    :t => [0.5; 0.5], # [0.25, -0.25, 0.0, ....]
    # COVER
    :cover_seed => 837387533,
    :cover => 4, # multiplier
    # for mfa
    :factors => 2,
    :mfa_tol => 1e-3,
    # for LMCLUS
    :best_bound => 1.0,
    :sampling_heuristic => 2,
    :sampling_factor => 0.5,
    :basis_alignment => true,
    :bounded_cluster => true,
    :min_cluster_size => 50,
    # CLUSTERING
    # for mean-shift
    :quantile => 0.1,
    # for dbscan
    :radius => 0.1,
)

# process args
startindex = 1
plmconly = false
if length(ARGS) > 1
    PARAMS[:c] = parse(Int, ARGS[1])
    PARAMS[:d] = parse(Int, ARGS[2])
    if length(ARGS) > 2
        startindex = parse(Int, ARGS[3])
        println("Start from the iteration: $startindex")
        if length(ARGS) > 3
            plmconly = parse(Bool, ARGS[4])
            plmconly && println("Run only PLMC clustering")
        end
    end
end

# adjust data generation parameters
if PARAMS[:c] == 3
    PARAMS[:repulse] = (-0.3, 0.3) # 3
elseif PARAMS[:c] == 5
    PARAMS[:repulse] = (0.3, 0.15) # 5
elseif PARAMS[:c] == 7
    PARAMS[:repulse] = (0.45, 0.45) # 7
end
if PARAMS[:d] > 2
    append!(PARAMS[:t], zeros(PARAMS[:d]-2))
end

CLUSTERING = plmconly ? [:plmc] : [:kmeans, :meanshift, :dbscan, :spectral, :plmc]
COVERS = [:kmeans, :lmclus, :mfa]

function findscore(J)
    # J = map(j->floor(Int, j), J)
    J[1] = J[end] = maximum(J)
    mx1 = PLMC.findglobalmin(J, 1e-3)
    mx2 = PLMC.findglobalmin2(J)
    return max(mx1, mx2)
end

RESULTS = Results()

Random.seed!(PARAMS[:experiment_seed]);
for i in startindex:PARAMS[:iterations]
    print("$i ")
    tm = time()
    PARAMS[:data_seed] += 1

    for ε in [0.1, 0.2, 0.3]
        ## generate
        PARAMS[:ε] = ε
        θ = Dict((i=>j) => rand()*(π) for i in 1:PARAMS[:d] for j in 1:PARAMS[:d] if i < j)
        X, L = nmoons(Float64, PARAMS[:m], PARAMS[:c], ε=PARAMS[:ε], r=PARAMS[:r], d=PARAMS[:d],
                    repulse=PARAMS[:repulse], translation=PARAMS[:t],
                    seed=PARAMS[:data_seed], rotations=θ)
        RESULTS.DATA = X

        ## setup results storage
        RESULTS.NMI = String => Float64
        RESULTS.RAND = String => Vector{Float64}
        RESULTS.ASSIGN = String => Vector{Int}
        RESULTS.ASSIGN = "dataset" => L

        for cl in CLUSTERING
            print("$cl ")
            C = nothing
            if cl != :plmc
                C = clustering(X, L, cl, PARAMS)
                RESULTS.NMI, RESULTS.RAND, RESULTS.ASSIGN = saveindecies("$cl", L, C)
            else
                for cv in COVERS
                    print("+$cv[")
                    flt, MC =  try
                        cover(X, cv, PARAMS)
                    catch
                        print("COV-ERR] ")
                        nothing, nothing
                    end
                    flt === nothing && continue
                    RESULTS.ASSIGN = "$cv" => assignments(MC)

                    # topological aglomeration
                    agg = try
                        agglomerate(flt)
                    catch
                        nothing
                    end
                    if agg !== nothing
                        print("Tmdl")
                        C = try
                            plmc(agg, MC, X, flt, score=PLMC.nml, find=findscore)
                        catch ex
                            print("-ERR")
                            nothing
                        end
                        RESULTS.NMI, RESULTS.RAND, RESULTS.ASSIGN = saveindecies("$cl+$cv:top", L, C)
                    end
                    # mdl aglomeration
                    print(" MDL")
                    C = try
                        plmc(PLMC.MDL, X, flt, MC, β=2, βsize=1, score=PLMC.nml, find=findscore)
                    catch
                        print("-ERR")
                        nothing
                    end
                    RESULTS.NMI, RESULTS.RAND, RESULTS.ASSIGN = saveindecies("$cl+$cv:mdl", L, C)

                    # ib aglomeration
                    print(" IBS")
                    C = try
                        plmc(PLMC.InformationBottleneck, X, flt, MC, β=2, score=PLMC.nml, find=findscore)
                    catch
                        print("-ERR")
                        nothing
                    end
                    RESULTS.NMI, RESULTS.RAND, RESULTS.ASSIGN = saveindecies("$cl+$cv:ib", L, C)

                    print("] ")
                end
            end
        end

        RESULTS.PARAMETERS = copy(PARAMS)
        RESULTS.EPS = ε
        RESULTS.ITR = i
    end

    # Save results
    if i % 50 == 0
        fname = "plm-clustering-tm$(PARAMS[:c])-d$(PARAMS[:d])-cpa"
        saveresults(fname, "Moons$(PARAMS[:c])-D$(PARAMS[:d])", PARAMS, RESULTS, true)
    end
    println("elapsed: ", (time() - tm))
end
