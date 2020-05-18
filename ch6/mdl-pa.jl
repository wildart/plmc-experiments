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
X1 = LinearManifoldCluster.generate_cluster(Uniform, n, zeros(N), D, M, Φ, E)
X2 = LinearManifoldCluster.generate_cluster(Normal, n, zeros(N), D, M, Φ, E)

X1 .-= [0.0; -0.5]
X1 .*= [fill(3.0,M); fill(1.0, N-M)]
X2 .-= [0.0; +0.5]
X2 .*= [fill(3.0,M); fill(1.0, N-M)]

p = scatter(X1[1,:], X1[2,:], xlims=(-3.2, 3.2), ylims=(-1.1, 1.1),
            xlab="x", ylab="y", ms=1.5, msw=0.5, label="Uniform")
scatter!(p, X2[1,:], X2[2,:], label="Normal", ms=1.5, msw=0.5)
saveplot("../gen/lmclus-1d.pdf", p)

#-------------------------------------------------------------------------------
println("Error & Dimension")
Random.seed!(230898573857);
results = DataFrame(N=Int[], ɛ=String[], MDL=Float64[])
for N in 2:10
    println("N=$N")
    μ = zeros(N)
    M = 1
    D = Matrix{Float64}(I, N, N)
    MF = Manifold(M, μ, D, collect(1:n), θ, 0.0)

    for i in 1:K
        X = LinearManifoldCluster.generate_cluster(Normal, n, μ, D, M, Φ, E)
        X .*= [fill(3.0,M); fill(1.0, N-M)]
        for (i,ɛ) in enumerate(ɛs)
            bits = LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, MF, X, Pm, Pd, ɛ=ɛ, tol=tol)
            push!(results, (N, "1e-$i", bits))
        end
        push!(results, (N, "Raw", LMCLUS.MDL.calculate(LMCLUS.MDL.Raw, MF, X, Pm, Pd)))
    end
end

gdf = groupby(results, [:N, :ɛ])
mdl = combine(gdf, :MDL => (x->measurement(mean(x),std(x))) => :MDL)
mdl.Dimension = "N =".*map(n->lpad(string(n),2),mdl[:,:N])
p = @df mdl plot(:ɛ, :MDL, group=:Dimension, seriestype=:line, yaxis = (:log2),
                 ylab="MDL, bits", xlab="Quantization error, ɛ", legend = :topleft)

saveplot("../gen/mdl-1d.pdf", p)

#-------------------------------------------------------------------------------
println("Encoding Constants")
Random.seed!(230898573857);
results = DataFrame(N=Int[], Consts=String[], MDL=Float64[])
consts = [(4*i, 4*j) for i in 4:2:8 for j in 4:2:8]
ɛ=1e-3
M = 1
for N in 2:10
    μ = zeros(N)
    println("N=$N")
    D = Matrix{Float64}(I, N, N)
    MF = Manifold(M, μ, D, collect(1:n), θ, 0.0)

    for i in 1:K
        for (Pm,Pd) in consts
            X = LinearManifoldCluster.generate_cluster(Normal, n, μ, D, M, Φ, E)
            X .*= [fill(3.0,M); fill(1.0, N-M)]
            bits = LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, MF, X, Pm, Pd, ɛ=ɛ, tol=tol)
            push!(results, (N, "M$(Pm)D$(Pd)", bits))
        end
    end
end

gdf = groupby(results, [:N, :Consts])
mdl =  combine(gdf, :MDL => (x->measurement(mean(x),std(x))) => :MDL)
mdl.Dimension = "N =".*map(n->lpad(string(n),2),mdl[:,:N])
p = @df mdl plot(:Consts, :MDL, group=:Dimension, seriestype=:line, yaxis = (:log2),
                 ylab="MDL, bits", xlab="Encoding constants", legend = :outertopright)

saveplot("../gen/mdl-1d-const.pdf", p)

#-------------------------------------------------------------------------------
println("Dimensionality")
Random.seed!(230898573857);
results = DataFrame(M=Int[], ɛ=String[], MDL=Float64[])
N = 10
μ = zeros(N)
D = Matrix{Float64}(I, N, N)
for M in 1:9
    println("M=$M")
    MF = Manifold(M, μ, D, collect(1:n), θ, 0.0)

    for (j,ɛ) in enumerate(ɛs)
        for i in 1:K
            X = LinearManifoldCluster.generate_cluster(Normal, n, μ, D, M, Φ, E)
            X .*= [fill(3.0,M); fill(1.0, N-M)]
            bits = LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, MF, X, Pm, Pd, ɛ=ɛ, tol=tol)
            push!(results, (M, "ɛ = 1e-$j", bits))
        end
    end
end

gdf = groupby(results, [:M, :ɛ])
mdl =  combine(gdf, :MDL => (x->measurement(mean(x),std(x))) => :MDL)
p = @df mdl plot(:M, :MDL, group=:ɛ, seriestype=:line, yaxis = (:log2),
                 ylab="MDL, bits", xlab="Cluster dimension, M", legend = :outertopright)

saveplot("../gen/mdl-nd.pdf", p)

#-------------------------------------------------------------------------------
println("5D")
N = 10
M = 5
μ = zeros(N)
D = Matrix{Float64}(I, N, N)
Pm, Pd = 24, 16
Random.seed!(230898573857);
X = LinearManifoldCluster.generate_cluster(Normal, n, μ, D, M, Φ, E)
X .*= [fill(3.0,M); fill(1.0, N-M)]
results = DataFrame(M=Int[], ɛ=String[], MDL=Float64[])
for M in 1:9
    println("M=$M")
    MF = Manifold(M, μ, D, collect(1:n), θ, 0.0)

    for (i,ɛ) in enumerate(ɛs)
        bits = LMCLUS.MDL.calculate(LMCLUS.MDL.OptimalQuant, MF, X, Pm, Pd, ɛ=ɛ, tol=tol)
        push!(results, (M, "ɛ = 1e-$i", bits))
    end
end

p = @df results plot(:M, :MDL, group=:ɛ, seriestype=:line, yaxis = (:log2),
                 ylab="MDL, bits", xlab="Cluster dimension, M", legend = :outertopright)

saveplot("../gen/mdl-5d.pdf", p)

#-------------------------------------------------------------------------------
