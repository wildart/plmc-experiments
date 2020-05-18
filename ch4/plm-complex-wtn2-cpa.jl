# Comparative Performance Analysis

include("src/results.jl")

## libraries
using Random: seed!
using NMoons
using ClusterComplex: clustercomplex, dominance2, dataset
using ComputationalHomology: filtration, complex, witness
using JSON

using Logging
lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Info);
# lgr = Logging.ConsoleLogger(stderr, Base.CoreLogging.Debug);

## parameters
PARAMS = Dict(
:seed => 837387533,
:m => 500,
:c => 2,
:ε => 0.15, #0.05-0.25
:r => 1,
:d => 2,
:repulse => (-0.25, -0.25),
:t => [0.0; 0.0],
# for witness construction
:l => 50,
:ν => 2,
# sample size
:iter => 100,
)
NAMES = Dict(:tm=>"TwoMoons", :crc=>"Circles", :sph=>"Sphere", :oip15=>"OIP15", :oip300=>"OIP300")

## generate
Xtm, Ltm = nmoons(Float64, PARAMS[:m], PARAMS[:c], ε=PARAMS[:ε], r=PARAMS[:r], d=PARAMS[:d],
                  repulse=PARAMS[:repulse], translation=PARAMS[:t], seed=PARAMS[:seed])
Htm = (PARAMS[:c],0)
# X, L = Xtm, Ltm; scatter(X[1,:], X[2,:], ms=1.0, legend=:none, color=L)

Xcrc, Lcrc = spheres(Float64, PARAMS[:m], PARAMS[:c], ε=PARAMS[:ε], d=PARAMS[:d], s=1, seed=PARAMS[:seed])
Hcrc = (PARAMS[:c],PARAMS[:c])
# X, L = Xcrc, Lsph; scatter(X[1,:], X[2,:], ms=1.0, legend=:none, color=L)

Xsph, Lsph = spheres(Float64, PARAMS[:m], PARAMS[:c]-1, ε=PARAMS[:ε], d=PARAMS[:d]+1, s=2, seed=PARAMS[:seed])
Hsph = (PARAMS[:c]-1,0,PARAMS[:c]-1)
# X, L = Xsph, Lsph; scatter3d(X[1,:], X[2,:], X[3,:], ms=1.0, legend=:none, color=L)

# Initialize result storage
RD = Dict([(n => zeros(PARAMS[:iter])) for n in keys(NAMES)])
CC = Dict([(n => zeros(Int, PARAMS[:iter])) for n in keys(NAMES)])

seed!(PARAMS[:seed]); # seed algorithm
for (n, ds) in NAMES
    println("Dataset: $ds")

    # Load data
    X, H, L = try
        eval(Symbol("X$n")), eval(Symbol("H$n")), eval(Symbol("L$n"))
    catch
        dataset(ds)
    end

    # Xn = standardize(UnitRangeTransform, X)
    for i in 1:PARAMS[:iter]
        # witness complex construction
        cplx, w = witness(X, PARAMS[:l], ν=PARAMS[:ν], maxoutdim=min(2, size(X,1)-1))

        # Construct filtration
        flt = filtration(cplx, w)

        # relative dominance
        RD[n][i], R0, R1, K = with_logger(lgr) do
            dominance2(flt, H)
        end
        CC[n][i] = length(complex(flt, R0))
        println("Relative dominance $i: $(RD[n][i]) [$(CC[n][i])]")
    end

end

# Save results
saveresults("plm-complex-wtn2-cpa", "Witness2", PARAMS, RelDom=RD, Cells=CC)
