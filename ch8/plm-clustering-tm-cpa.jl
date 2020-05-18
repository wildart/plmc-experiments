# Comparative Performance Analysis

include("src/conda.jl") # setup conda env settings

include("src/results.jl")
include("src/cover.jl")
include("src/clustering.jl")

## packages
using NMoons
using PLMC
using Random

using Logging
lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Info);
# lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Debug);

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
    :cover => 3, # multiplier
    # for mfa
    :factors => 2,
    :trace => 0.2,
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
if length(ARGS) > 1
    PARAMS[:c] = parse(Int, ARGS[1])
    PARAMS[:d] = parse(Int, ARGS[2])
    if length(ARGS) > 2
        startindex = parse(Int, ARGS[3])
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

CLUSTERING = [:kmeans, :meanshift, :dbscan, :spectral, :plmc]
COVERS = [:kmeans, :lmclus, :mfa]

NMI = Dict{String,Float64}[]
RAND = Dict{String,Vector{Float64}}[]
A = Dict{String,Vector{Int}}[]
D = Matrix[]
EPS = Float64[]
for i in startindex:PARAMS[:iterations], ε in [0.1, 0.2, 0.3]
    print("\n$i: ")
    PARAMS[:ε] = ε
    push!(EPS, ε)

    ## generate
    θ = Dict((i=>j) => rand()*(π) for i in 1:PARAMS[:d] for j in 1:PARAMS[:d] if i < j)
    X, L = nmoons(Float64, PARAMS[:m], PARAMS[:c], ε=PARAMS[:ε], r=PARAMS[:r], d=PARAMS[:d],
                  repulse=PARAMS[:repulse], translation=PARAMS[:t],
                  seed=PARAMS[:data_seed]+i, rotations=θ)

    ## setup results storage
    nmi = Dict{String,Float64}()
    ridx = Dict{String,Vector{Float64}}()
    assign = Dict{String,Vector{Int}}()
    assign["dataset"] = L

    for cl in CLUSTERING
        print("$cl ")
        C = nothing
        if cl != :plmc
            C = clustering(X, cl, PARAMS)
            saveindecies!("$cl", L, C, nmi, ridx, assign)
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
                assign["cover"] = assignments(MC)

                # topological aglomeration
                agg = try
                    agglomerate(flt)
                catch
                    nothing
                end
                if agg !== nothing
                    print("Tmdl")
                    C = try
                        plmc(agg, MC, X, flt)
                    catch
                        print("-ERR")
                        nothing
                    end
                    saveindecies!("$cl+$cv:Tmdl", L, C, nmi, ridx, assign)

                    # print(" Tpll")
                    # C = try
                    #     plmc(agg, MC, X, flt, score=:posll)
                    # catch
                    #     print("-ERR")
                    #     nothing
                    # end
                    # saveindecies!("$cl+$cv:Tpll", L, C, nmi, ridx, assign)

                    # print(" Tch")
                    # C = try
                    #     plmc(agg, MC, X, flt, score=:change)
                    # catch
                    #     print("-ERR")
                    #     nothing
                    # end
                    # saveindecies!("$cl+$cv:Tch", L, C, nmi, ridx, assign)
                end
                # mdl aglomeration
                print(" MDL")
                C = try
                    plmc(PLMC.MDL, X, flt, MC, β=3)
                catch
                    print("-ERR")
                    nothing
                end
                saveindecies!("$cl+$cv:MDL", L, C, nmi, ridx, assign)

                # ib aglomeration
                print(" IBS")
                C = try
                    plmc(PLMC.InformationBottleneck, X, flt, MC, β=2, hard=false)
                catch
                    print("-ERR")
                    nothing
                end
                saveindecies!("$cl+$cv:IB-S", L, C, nmi, ridx, assign)

                # print(" IBH")
                # C = try
                #     plmc(PLMC.InformationBottleneck, X, flt, MC, β=2)
                # catch
                #     print("-ERR")
                #     nothing
                # end
                # saveindecies!("$cl+$cv:IB-H", L, C, nmi, ridx, assign)

                print("] ")
            end
        end
    end

    push!(NMI, nmi)
    push!(RAND, ridx)
    push!(A, assign)
    push!(D, X)

    # Save results
    saveresults("plm-clustering-tm$(PARAMS[:c])-d$(PARAMS[:d])-cpa", "Moons$(PARAMS[:c])-D$(PARAMS[:d])", PARAMS, true,
                NMI=NMI, RAND=RAND, ASSIGN=A, DATA=D, EPS=EPS)
end
