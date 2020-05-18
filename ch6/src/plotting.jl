using Statistics
using StatsPlots
using StatsPlots: pt, mm
# gr(size=(400,300))

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

function mdlplot(ds; remove_first_pts = 8, fmt=:svg)
    # Setup plot parameters
    rnames, res, xticks = mdlplotparams(ds, remove_first_pts = remove_first_pts)
    # Generate plot
    plot(res, title="Minimum Description Length, Dataset: $(NAMES[ds])", lab=rnames, ylab="Bits", xlab="Clusters",
         leg=:topright, xticks=xticks, fmt=fmt)
end
