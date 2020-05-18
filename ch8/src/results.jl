using Statistics
using JSON
using BSON
using Dates
using DataFrames

global RESULTS_DIR="results/"

# r = Results()
# r.TEST1 = 1
# r.TEST1 = 2
# r.TEST2 = Symbol => Int
# r.TEST2 = :a => 1
# r.TEST2 = :b => 1
# r.TEST3 = :a => 1
# r.TEST3 = Symbol => Int
# r.TEST3 = :b => 1
# r.TEST4 = Symbol => Vector{Vector{Int}}
# r.TEST4 = :b => [[1]]
# r.TEST4 = :c => [[1]]
mutable struct Results
    storage::Dict{Symbol,Vector{<:Any}}
    Results() = new(Dict{Symbol,Vector{<:Any}}())
end
function Base.setproperty!(r::Results, p::Symbol, v)
    f = getfield(r, :storage)
    if !haskey(f, p)
        f[p] = typeof(v)[]
    end
    push!(f[p], v)
end
function Base.setproperty!(r::Results, p::Symbol, v::Pair)
    f = getfield(r, :storage)
    T1 = isa(v.first, DataType) ? v.first : typeof(v.first)
    T2 = isa(v.second, DataType) ? v.second : typeof(v.second)
    if !haskey(f, p)
        f[p] = Dict{T1,T2}[]
    end
    (isa(v.first, DataType) || length(f[p]) == 0) && push!(f[p], Dict{T1,T2}())
    !isa(v.second, DataType) && push!(f[p][end], v)
end
Base.getproperty(r::Results, p::Symbol) = p == :storage ? getfield(r, :storage) : getfield(r, :storage)[p]

function saveresults(fname, method, params, res::Dict, bson=false)
    ts = round(Int64, time())
    respath = "$RESULTS_DIR/$fname-$ts"
    open("$respath.$(bson ? 'b' : 'j')son", "w") do io
        ctx = Dict(:method=>method, :time=>ts,
                   :parameters=>copy(params),
                   :results=>res)
        if bson
            BSON.bson(io, ctx)
        else
            write(io, JSON.json(ctx))
        end
    end
end

saveresults(fname, method, params, res::Results, bson=false) =
    saveresults(fname, method, params, res.storage, bson)

function saveresults(fname, method, params, bson=false; data...)
    res = Dict{Symbol,Any}()
    for (pn,pd) in data
        res[pn] = pd
    end
    saveresults(fname, method, params, res, bson)
end

function getResults(dset, param; bson=true, filter=r".*", null=nothing)
    ext = "$(bson ? 'b' : 'j')son"
    data = Dict{Tuple{String,DateTime},Any}()
    # println("Read dir: $RESULTS_DIR")
    for f in readdir(RESULTS_DIR)
        match(filter, f) === nothing && continue
        fpath = "$RESULTS_DIR/$f"
        !endswith(fpath, ext) && continue
        # println("Parsing $f")
        res = if bson
            BSON.load(fpath)
        else
            JSON.parsefile(fpath, null=null)
        end
        ts = unix2datetime(res[:time])
        mtd = res[:method]
        !startswith(mtd, dset) && continue
        rdata = res[:results]

        # add file info
        if param == :fileinfo
            fdata = get(rdata,:DATA,[])
            finfo = (mtd, ts)=>(f, length(fdata))
            # println(finfo)
            push!(data, finfo)
        else
            # add requested parameter
            !haskey(rdata, param) && continue
            push!(data, (mtd, ts)=>rdata[param])
        end
    end
    return data
end

function getParams(filter=r"")
    data = Dict{Tuple{String,DateTime},Any}()
    for f in readdir(RESULTS_DIR)
        findfirst(filter,f) === nothing && continue
        fpath = "$RESULTS_DIR/$f"
        ext = splitext(fpath) |> last
        ext ∉ [".json", ".bson"] && continue
        println("Parsing $f")
        res = if ext == ".bson"
            BSON.load(fpath)
        else
            JSON.parsefile(fpath, null=null)
        end
        ts = unix2datetime(res[:time])
        mtd = res[:method]
        pdata = res[:parameters]
        push!(data, (mtd, ts)=>pdata)
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

function mad(itr)
    med = median(itr)
    median( abs(x - med) for x in itr )
end

function changenames!(df, namesub)
    for (oldn, newn) in namesub
        idxs = map(m -> endswith(m, oldn), df.Method)
        df[idxs, :Method] .= map(n->n[1:end-length(oldn)]*newn, df[idxs, :Method])
    end
end

function parseCD(dsname)
    sizedim = r"(\d+)-D(\d+)$"
    m = match(sizedim, dsname)
    !(m !== nothing && length(m.captures) > 1) && return nothing
    c, d = map(c->parse(Int,c), m.captures)
    (c, d)
end
