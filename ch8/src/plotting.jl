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

function plotscores(scrs)
    y1 = scrs[:,1]
    y2 = scrs[:,2]
    x = collect(length(y1):-1:1)
    p = plot(x, y1, label="Score", xlabel="Clusters", bottom_margin=0.5mm,
             ylabel="MDL", right_margin=10mm, leg=(0.7,0.1))
    plt = twinx()
    plot!(plt, x, y2, color=:red, label="NMI",
          leg=(0.87,0.1), ylabel="NMI" , ylim=(0.0,1.0))
    return p
end
