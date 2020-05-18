#--------------------------------
# Setup plotting (requires Conda)

# Julia packages
using Colors
using Clustering
using PyPlot, PyCall
# pygui(true)

# Python imports
crs = PyCall.pyimport("cartopy.crs")
ctick = PyCall.pyimport("cartopy.mpl.ticker")
ma = PyCall.pyimport("numpy.ma")

#-------------------------------

global PLOT_FORMAT = :png

function saveplot(file, fig; dpi=100)
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
        fig.savefig(fn*fext, bbox_inches="tight", dpi=dpi)
        PyPlot.close(fig)
    end
end

function mapclusters(clust::T, mask, cm; figsize=(6, 5),
                     res=(180,360), fNA=-1e30) where {T <: ClusteringResult}
    lats = ((res[1]-1):-1:0) .- (res[1]/2 - 0.5)|> collect
    lons = (res[2]/2 - 0.5) .+ (-(res[2]-1):1:0) |> collect

    fig = PyPlot.figure(figsize=figsize)
    # ax = PyPlot.axes(projection=crs.PlateCarree())
    ax = PyPlot.axes(projection=crs.Miller())
    ax.set_global()
    ax.coastlines()
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180], crs=crs.PlateCarree())
    ax.set_yticks([-78, -60, -25, 25, 60, 80], crs=crs.PlateCarree())
    lon_formatter = ctick.LongitudeFormatter(number_format=".0f", degree_symbol="")
    #number_format=".1f", degree_symbol='', dateline_direction_label=True
    lat_formatter = ctick.LatitudeFormatter(number_format=".0f", degree_symbol="")
    # number_format='.1f', degree_symbol=''
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_extent([-180, 180, -65, 88])

    valmap = fill(fNA, prod(res))
    sz = sortperm(counts(clust), rev=true)
    valmap[mask] .= map(i->sz[i], assignments(clust))
    # valmap[mask] .= assignments(clust)

    valmap = reverse(reshape(valmap, res[2], res[1])', dims=1)
    mmo = pycall(ma.masked_equal, Any, valmap, fNA)

    # PyPlot.contourf(lons, lats, mmo, 60, transform=ccrs.PlateCarree(), cmap=cm)
    ax.pcolor(lons, lats, mmo, transform=crs.PlateCarree(), cmap=cm)
    ax.coastlines()
    return fig
end

function mapclusters(clust::T, mask; ncolors=0, col=colorant"white",
                     kwargs...) where {T <: ClusteringResult}
    n = max(nclusters(clust), ncolors)
    cm = ColorMap("clusters", distinguishable_colors(n+4, col)[3:end])
    mapclusters(clust, mask, cm; kwargs...)
end
