using Statistics
using StatsPlots
using StatsPlots: pt, mm
using ClusterComplex
# using TDA
# gr(size=(400,300))

# using PLMC

global PLOT_FORMAT = :png

function saveplot(file, p)
    if  length(file)>0
        fn, fext = splitext(file)
        fext = if length(fext) == 0
            if PLOT_FORMAT == :png
                ".png"
            elseif PLOT_FORMAT == :pdf
                ".pdf"
            end
        else
            fext
        end
        if fext == ".png"
            StatsPlots.Plots.png(p, fn*fext)
        elseif fext == ".pdf"
            StatsPlots.Plots.pdf(p, fn*fext)
        else
            @error "Unsupported format: $PLOT_FORMAT"
        end
    end
end

function plotmdl(MDL, LL, COMP, title; file="")
    k = length(MDL)
    p = plot(MDL, label="MDL", title=title,
             ylabel="Bits", xlabel="Clusters", legend=:topright,
             xticks=(collect(1:k), map(string, collect(k:-1:1))))
    plot!(p, LL, label="LL")
    plot!(p, COMP, label="COMP")
    saveplot(file, p)
    return p
end

function plotclusterboundaries(MC, χ=2.0; data=nothing,
                               annotate=false, file="", color=true)
    ann = annotate ? map(c->let mu = mean(c[2]);(mu[1], mu[2], "C$(c[1])");end, enumerate(models(MC))) : []
    if data === nothing
        p = plot(MC, χ, annotations=ann, colors=color, legend=:none)
    else
        p = plot(MC, data, markersize=1.5, legend=:none)
        plot!(p, MC, χ, colors=color, annotations=ann)
    end
    saveplot(file, p)
    return p
end

function plotplmc(PLC, χ=2.0; data=nothing, title="",
                  annotate=false, file="", color=true, fmt=:svg)
    ann = annotate ? map(c->let mu = mean(c[2]);(mu[1], mu[2], "C$(c[1])");end, enumerate(models(PLC))) : []
    if data === nothing
        p = plot(PLC, annotations=ann, legend=:none, fmt=fmt, title=title)
    else
        p = plot(PLC, data, markersize=1.5, legend=:none, fmt=fmt, title=title)
        plot!(p, PLC, annotations=ann)
    end
    saveplot(file, p)
    return p
end

function mdlplot(ds; remove_first_pts = 8, fmt=:svg)
    # Setup plot parameters
    rnames, res, xticks = mdlplotparams(ds, remove_first_pts = remove_first_pts)
    # Generate plot
    plot(res, title="Minimum Description Length, Dataset: $(NAMES[ds])", lab=rnames, ylab="Bits", xlab="Clusters",
         leg=:topright, xticks=xticks, fmt=fmt)
end
