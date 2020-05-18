using LinearManifoldCluster
using LinearManifoldCluster.Distributions
using LinearAlgebra
using Random
using LMCLUS
using DataFrames
using Measurements

include("src/plotting.jl")
StatsPlots.Plots.pyplot()

n = 1000
Φ = 1.0
E = 0.1
Pm = 24      # Precision encoding constant for model
Pd = 16      # Precision encoding constant for data
θ = E        # distance threshold
tot = 1000
tol = 1e-18
ɛs = [10.0^(-i) for i in 1:8]
K = 100

#-------------------------------------------------------------------------------
N = 2
M = 1
D = Matrix{Float64}(I, N, N)
Random.seed!(230898573857);
X1 = LinearManifoldCluster.generate_cluster(Uniform, n, zeros(N), D, M, E, E)
X2 = LinearManifoldCluster.generate_cluster(Normal, n, zeros(N), D, M, E, E)

X1 .-= [0.0; -0.5]
X1 .*= [fill(3.0,M); fill(1.0, N-M)]
X2 .-= [0.0; +0.5]
X2 .*= [fill(3.0,M); fill(1.0, N-M)]

p = scatter(X1[1,:], X1[2,:], xlims=(-1.1, 1.1), ylims=(-1.1, 1.1),
            xlab="x", ylab="y", ms=1.5, msw=0.3, label="Uniform")
scatter!(p, X2[1,:], X2[2,:], label="Normal", ms=1.5, msw=0.3)
saveplot("../gen/lmclus-0d.pdf", p)

#-------------------------------------------------------------------------------
