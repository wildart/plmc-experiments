# Climatological Data

using Logging
# lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Info);
lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Debug);
Logging.disable_logging(Logging.BelowMinLevel);

include("src/cover.jl")
include("src/results-plmc.jl")

## load data
BSON.@load "../ch6/data/CLIM_1951_1980_1x1.bson" NAMES UNIT MASK
X = convert(Matrix{Float64}, UNIT')

## parameters
PARAMS = Dict(
    :iterations => 100,
    :experiment_seed => 837387533,
    # COVER
    :cover_seed => 837387533,
    :cover => 105,
    :c => 1,
    # for mfa
    :factors => 2,
    :mfa_tol => 1e-3,
    # for LMCLUS
    :best_bound => 0.2,
    :sampling_heuristic => 2,
    :sampling_factor => 0.3,
    :basis_alignment => true,
    :bounded_cluster => true,
    :min_cluster_size => 150,
)

function findscore(J)
    J = map(j->floor(Int, j), J)
    # J[1] = J[end] = maximum(J)
    print("\nJ: $J\n")
    mx1 = PLMC.findglobalmin(J, 1e-3)
    mx2 = PLMC.findglobalmin2(J)
    return max(mx1, mx2)
end

csz = [36, 72, 108, 144]
hs =  [PLMC.MDL, PLMC.InformationBottleneck, PLMC.Topological]
heuristic = length(ARGS) > 0 ? PLMC.InformationBottleneck : PLMC.Topological

# cs = csz[1]
for cs in csz
    println(cs)
    # flt, MC = cover(X, :lmclus, PARAMS)
    # flt, MC = fit(kMeansCover, X, cs; seed=PARAMS[:cover_seed])
    # flt, MC = fit(kMeansCover, X, PARAMS[:cover]; seed=PARAMS[:cover_seed])
    # flt, MC = fit(MFACover, X, PARAMS[:cover], PARAMS[:factors];
    #               seed=PARAMS[:cover_seed], tol=PARAMS[:mfa_tol], maxiter=500)
    flt, MC = with_logger(lgr) do
        fit(MFACover, X, cs, PARAMS[:factors];
            seed=PARAMS[:cover_seed], tol=PARAMS[:mfa_tol], maxiter=500)
    end

    # cplx, w, MC2 = clustercomplex(X, MC.models, 3.0, assignments=assignments(MC))
    for heuristic in hs
        println(heuristic)
        C, AGG, J = with_logger(lgr) do
            # plmc(PLMC.Topological, X, flt, MC, score=PLMC.nml, find=PLMC.findfirstminimum)
            plmc(heuristic, X, flt, MC, β=2, βsize=1, score=PLMC.nml, find=findscore)
        end

        suffix = (split(string(heuristic), ".") |> last |> lowercase)[1:3]
        # saveClustering("results/plmc-km-$cs-$suffix.bson", C, AGG, J)
        saveClustering("results/plmc-mfa-$cs-$suffix.bson", C, AGG, J)
    end
end
