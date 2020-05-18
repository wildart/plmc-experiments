# Stochastic Perturbation Analysis: Dataset RNG

include("src/results.jl")

## libraries
using Random: seed!, MersenneTwister
using NMoons
using ClusterComplex: clustercomplex, dominance2, dataset
using LMCLUS
using ComputationalHomology: filtration, complex
using JSON

using Logging
lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Info);
# lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Debug);

## parameters
PARAMS = Dict(
# for datasets
:seed => 837387533,
:m => 500,
:c => 2,
:ε => 0.15, #0.05-0.25
:r => 1,
:d => 2,
:repulse => (-0.25, -0.25),
:t => [0.0; 0.0],
# for LMCLUS
:best_bound => 0.3,
:sampling_heuristic => 2,
:sampling_factor => 0.1,
:basis_alignment => true,
:bounded_cluster => true,
:min_cluster_size => 30,
# for PLMC
:ϵ => 10.0,
:iter => 1000, # sample size
)
NAMES = Dict(:tm=>"TwoMoons", :crc=>"Circles", :sph=>"Sphere")

# Initialize result storage
RD = Dict([(n => zeros(PARAMS[:iter])) for n in keys(NAMES)])
CC = Dict([(n => zeros(Int, PARAMS[:iter])) for n in keys(NAMES)])

# Generate seeds for dataset creation
seed!(PARAMS[:seed]);
seeds = rand(UInt, PARAMS[:iter])

for (n, ds) in NAMES
    println("Dataset: $ds")

    for i in 1:PARAMS[:iter]

        # generate data
        X, H, L = if n == :tm
            Xtm, Ltm = nmoons(Float64, PARAMS[:m], PARAMS[:c], ε=PARAMS[:ε], r=PARAMS[:r], d=PARAMS[:d],
                              repulse=PARAMS[:repulse], translation=PARAMS[:t], seed=seeds[i])
            Htm = (PARAMS[:c],0)
            Xtm, Htm, Ltm
        elseif n == :crc
            Xcrc, Lcrc = spheres(Float64, PARAMS[:m], PARAMS[:c], ε=PARAMS[:ε], d=PARAMS[:d], s=1, seed=seeds[i])
            Hcrc = (PARAMS[:c],PARAMS[:c])
            Xcrc, Hcrc, Lcrc
        elseif n == :sph
            Xsph, Lsph = spheres(Float64, PARAMS[:m], PARAMS[:c]-1, ε=PARAMS[:ε], d=PARAMS[:d]+1, s=2, seed=seeds[i])
            Hsph = (PARAMS[:c]-1,0,PARAMS[:c]-1)
            Xsph, Hsph, Lsph
        end

        # Set algorithm RNG
        rng = MersenneTwister(PARAMS[:seed])

        # Set clustering parameters
        p = LMCLUS.Parameters(size(X,1))
        p.best_bound = PARAMS[:best_bound]
        p.sampling_heuristic = PARAMS[:sampling_heuristic]
        p.sampling_factor = PARAMS[:sampling_factor]
        p.basis_alignment = PARAMS[:basis_alignment]
        p.bounded_cluster = PARAMS[:bounded_cluster]
        p.min_cluster_size = PARAMS[:min_cluster_size]

        # Clustering algorithm: LMCLUS
        C = with_logger(lgr) do
            LMCLUSResult(lmclus(X, p, [rng])...)
        end

        try
            # Conctruct model class from original clustering
            cplx, w, MC = clustercomplex(X, C, PARAMS[:ϵ], maxoutdim=min(2, size(X,1)-1), expansion=:inductive)
            # println("Created $(MC)")

            # Construct filtration
            flt = filtration(cplx, w)

            # relative dominance
            RD[n][i], R0, R1, K = with_logger(lgr) do
                dominance2(flt, H)
            end
            CC[n][i] = length(complex(flt, R0))
        catch
            RD[n][i] = NaN
            CC[n][i] = 0
        end
        println("Relative dominance $i: $(RD[n][i]) [$(CC[n][i])]")
    end

end

# Save results
saveresults("plm-complex-lmclus-spa-rng-gen", "PLM + LMCLUS", PARAMS, RelDom=RD, Cells=CC)
