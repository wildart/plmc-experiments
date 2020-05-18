# Parametric Analysis

include("src/results.jl")

## libraries
using Random: seed!
using NMoons
using ClusterComplex: clustercomplex, dominance2, dataset
using ComputationalHomology: filtration, complex, vietorisrips
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
:ϵ => 0.45,
#:ϵ => 0.25, #OIP15
# :ϵ => 0.28, #OIP300
# sample size
:iter => 1,
)
NAMES = Dict(:tm=>"TwoMoons", :crc=>"Circles", :sph=>"Sphere")

# Initialize result storage
RD = Dict([(n => zeros(PARAMS[:iter])) for n in keys(NAMES)])
CC = Dict([(n => zeros(Int, PARAMS[:iter])) for n in keys(NAMES)])
CL = Dict([(n => zeros(Int, PARAMS[:iter])) for n in keys(NAMES)])

for m in [250, 500, 1000, 1500]
PARAMS[:m] = m
println("Size: $m")
for ε in [0.10, 0.15, 0.175, 0.20, 0.225, 0.25, 0.30, 0.35, 0.40] # 0.10-0.40
PARAMS[:ε] = ε
println("Noise: $ε")

for (n, ds) in NAMES
    println("Dataset: $ds")

    # generate data
    X, H, L = if n == :tm
        Xtm, Ltm = nmoons(Float64, PARAMS[:m], PARAMS[:c], ε=PARAMS[:ε], r=PARAMS[:r], d=PARAMS[:d],
                          repulse=PARAMS[:repulse], translation=PARAMS[:t], seed=PARAMS[:seed])
        Htm = (PARAMS[:c],0)
        Xtm, Htm, Ltm
    elseif n == :crc
        Xcrc, Lcrc = spheres(Float64, PARAMS[:m], PARAMS[:c], ε=PARAMS[:ε], d=PARAMS[:d], s=1, seed=PARAMS[:seed])
        Hcrc = (PARAMS[:c],PARAMS[:c])
        Xcrc, Hcrc, Lcrc
    elseif n == :sph
        Xsph, Lsph = spheres(Float64, PARAMS[:m], PARAMS[:c]-1, ε=PARAMS[:ε], d=PARAMS[:d]+1, s=2, seed=PARAMS[:seed])
        Hsph = (PARAMS[:c]-1,0,PARAMS[:c]-1)
        Xsph, Hsph, Lsph
    end

    seed!(PARAMS[:seed]); # seed algorithm
    for i in 1:PARAMS[:iter]

        # vietoris-rips complex construction
        cplx, w = vietorisrips(X, PARAMS[:ϵ], maxoutdim=size(X,1)-1)
        CL[n][i] = 0

        try
            # Construct filtration
            flt = filtration(cplx, w)

            # relative dominance
            RD[n][i], R0, R1, K = with_logger(lgr) do
                dominance2(flt, H)
            end
            CC[n][i] = length(complex(flt, R0))
            # println("Relative dominance $i: $(RD[n][i]) [$(CC[n][i])]")
        catch
            RD[n][i] = NaN
            CC[n][i] = 0
        end
    end

end

# Save results
saveresults("plm-complex-vr-pa", "Vietoris-Rips", PARAMS, RelDom=RD, Cells=CC, Clusters=CL)

end
end
