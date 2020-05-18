using Statistics
using JSON
using Dates
using Serialization

global RESULTS_DIR="results/"

function saveresults(fname, method, params; data...)
    ts = round(Int64, time())
    respath = "$RESULTS_DIR/$fname-$ts"
    res = Dict[]
    for (n, ds) in NAMES
        dt = Dict{String,Any}("name"=>ds)
        for (pn,pd) in data
            dt["$pn"] = pd[n]
        end
        push!(res, dt)
    end
    open("$respath.json", "w") do io
        write(io, JSON.json(Dict("method"=>method, "time"=>ts, "parameters"=>params, "results"=>res)))
    end
end

function getResults(dset, param; filter="", null=nothing)
    data = Dict{Tuple{String,DateTime},Any}()
    for f in readdir(RESULTS_DIR)
        findfirst(filter,f) === nothing && continue
        fpath = "$RESULTS_DIR/$f"
        !endswith(fpath, "json") && continue
        #println("Parsing $f")
        res = JSON.parsefile(fpath, null=null)
        ts = unix2datetime(res["time"])
        mtd = res["method"]
        rdata = res["results"]
        ri = findfirst(r->r["name"] == dset, rdata)
        ri === nothing && continue
        !haskey(rdata[ri], param) && continue
        push!(data, (mtd, ts)=>rdata[ri][param])
    end
    return data
end

function getParams(filter="")
    data = Dict{Tuple{String,DateTime},Any}()
    for f in readdir(RESULTS_DIR)
        findfirst(filter,f) === nothing && continue
        fpath = "$RESULTS_DIR/$f"
        !endswith(fpath, "json") && continue
        #println("Parsing $f")
        res = JSON.parsefile(fpath)
        ts = unix2datetime(res["time"])
        mtd = res["method"]
        pdata = res["parameters"]
        push!(data, (mtd, ts)=>pdata)
    end
    return data
end

function getClusters(dset)
    data = Dict{String,ClusteringResult}()
    for f in map(fn->split(fn, '.')|>first, readdir(RESULTS_DIR)) |> unique
        fpath = "$RESULTS_DIR/$f"
        !isfile("$fpath.json") && continue
        res = JSON.parsefile("$fpath.json")
        mtd = res["method"]
        !isfile("$fpath.jld") && continue
        cls = open(deserialize, "$fpath.jld")
        push!(data, mtd=>cls[dset])
    end
    return data
end

function mdlplotparams(ds; remove_first_pts = 8)
    MDL = getResults(ds, "mdl")
    rnames, res = collect(keys(MDL)), hcat(values(MDL)...)
    res = res[remove_first_pts:end, :]
    k = size(res,1)
    xticks=(collect(1:k), map(string, collect(k:-1:1)))
    return rnames, res, xticks
end

function mdlplot(ds; remove_first_pts = 8, fmt=:svg)
    # Setup plot parameters
    rnames, res, xticks = mdlplotparams(ds, remove_first_pts = remove_first_pts)
    # Generate plot
    plot(res, title="Minimum Description Length, Dataset: $(NAMES[ds])", lab=rnames, ylab="Bits", xlab="Clusters",
         leg=:topright, xticks=xticks, fmt=fmt)
end

function mad(itr)
    med = median(itr)
    median( abs(x - med) for x in itr )
end
