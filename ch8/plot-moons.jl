include("src/plotting.jl")

#-----------------------------
# Plot datasets
#
using NMoons

seed = 837387533
d = 2
cs = [2, 3, 5, 7]
εs = [0.1, 0.2, 0.3]
t = [0.5; 0.5]

ps = []
for (i,c) in enumerate(cs)
    # adjust data generation parameters
    repulse = if c == 2
        (-0.25, -0.25)
    elseif c == 3
        (-0.3, 0.3) # 3
    elseif c == 5
        (0.3, 0.15) # 5
    elseif c == 7
        (0.45, 0.45) # 7
    end
    for (j,ε) in enumerate(εs)
        θ = Dict((i=>j) => rand()*(π) for i in 1:d for j in 1:d if i < j)
        X, L = nmoons(Float64, 500, c, ε=ε, r=1, d=d, repulse=repulse,
                      translation=t, seed=seed, rotations=θ)
        tt = i == 1 ? "ε=$ε" : ""
        yl = j == 1 ? "c=$c" : ""
        push!(ps, scatter(X[1,:], X[2,:], c=L, legend=:none, ms=2, msw=0.5, showaxis=false, title=tt, ylabel=yl));
    end
end
p = plot(ps..., layout=(length(cs),length(εs)), size=(800,800), margin=0mm);
saveplot("../gen/cpa-moons.png", p)
#-----------------------------
