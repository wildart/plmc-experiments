using Statistics
using DataFrames
using DataFramesMeta

include("src/plotting.jl")
include("src/results.jl")

cols = ["Vietoris-Rips", "Witness0", "Witness2", "PLM + LMCLUS", "PLM + K-Means"]
parts = Dict(1=>"\\% success", 2=>"relative dominance, median", 3=>"relative dominance, MAD",
             4=>"number of cells, median", 5=>"number of cells, MAD")
dss = ["TwoMoons", "Circles", "Sphere", "OIP15", "OIP300"]
# dss = sort!(collect(values(NAMES)))
# dsn = first(dss)

#-----------------------------

results = DataFrame(Method = String[], Dataset = String[], RelDominance=Union{Float64,Missing}[], Cells=Union{Int,Missing}[])
for dsn in dss
    RelDom = getResults(dsn, "RelDom", filter="cpa", null=missing)
    length(RelDom) == 0 && continue
    Cells = getResults(dsn, "Cells", filter="cpa", null=missing)
    for (rd,cl) in zip(RelDom, Cells)
        m = first(rd)[1]
        for (v1,v2) in zip(last(rd), last(cl))
            push!(results, (m, dsn, v1, v2))
        end
    end
end

global PLOT_FORMAT = :pdf
for (dc, dcn) in  [:RelDominance => "Relative Dominance", :Cells => "# of cells" ]
    suffix = lowercase(string(dc))[1:5]

    p = []
    for (i,dsn) in enumerate(dss)
        flt = dc == :Cells ? cols[1] : ""
        res = @linq results |> where(:Dataset .== dsn, :Method .!= flt)
        push!(p, @df dropmissing(res, dc) boxplot(:Method, cols(dc), legend=:none, ylabel= i != 5 ? dcn : "" ) );
    end

    pall = plot(p..., layout=@layout[a ; b ; c; d e], size=(800,800), title=reshape(dss,1,5));
    saveplot("../gen/cpa-results-$suffix.$PLOT_FORMAT", pall)
end

#-----------------------------

function proccessResult(data, statistic)
    len = length(data)
    mis = count(ismissing, data)
    len-mis > 0.0 ? statistic(skipmissing(data)) :  0.0
end


fname = length(ARGS) == 0 ? "../gen/cpa-results-table.tex" : ARGS[1]
f = open(fname, "w")
# f = stdout

println(f, """\\begin{tabular}{llllll}
% header 1
    & \\multicolumn{5}{|c|}{\\textbf{Construction}} \\\\ \\cline{2-6}
% header 2
    \\multicolumn{1}{c}{\\textbf{Dataset}} &
    \\multicolumn{1}{|c|}{Vietoris-Rips} &
    \\multicolumn{1}{|c|}{Witness, \$\\nu = 0\$} &
    \\multicolumn{1}{|c|}{Witness, \$\\nu = 2\$} &
    \\multicolumn{1}{|c|}{PLM + LMCLUS} &
    \\multicolumn{1}{|c|}{PLM + \$k\$-means} \\\\ \\cline{1-6}
""")

for ds in dss
    # data = results[ds]
    res = """
% dataset $ds
    \\multicolumn{1}{l}{\\textbf{$ds}} & & & & & \\multicolumn{1}{l}{} \\\\ \\cline{1-6}
"""

    for part in sort(collect(keys(parts)))
        vals = fill("-", size(cols)...)

        for m in cols
            i = findfirst(isequal(m), cols)
            data = @linq results |>
                    where(:Dataset .== ds, :Method .== m) |>
                    select(:Cells, :RelDominance)
            if size(data,1) > 0
                expdata = data[!,:RelDominance]
                len = length(expdata)
                v = if part == 1
                    i == 1 ? 100.0 : len - count(ismissing, expdata)
                elseif part == 2
                    proccessResult(expdata, median)
                elseif part == 3
                    proccessResult(expdata, mad)
                elseif part == 4
                    pr = proccessResult(expdata, median)
                    if pr == 0.0 || len - count(ismissing, expdata) == 0
                        0.0
                    else
                        expdata = skipmissing(data[!,:Cells]) |> collect
                        proccessResult(expdata, median)
                    end
                elseif part == 5
                    pr = proccessResult(expdata, median)
                    if pr == 0.0 || len - count(ismissing, expdata) == 0
                        0.0
                    else
                        expdata = skipmissing(data[!,:Cells]) |> collect
                        proccessResult(expdata, mad)
                    end
                end
                vals[i] = string(part == 4 ? round(Int, v) : round(v, digits=2))
            end
        end


        res *= """
% row $part
    \\multicolumn{1}{r|}{$(parts[part])} &"""
        for v in vals
            res *= "\n    \\multicolumn{1}{c}{$v} &"
        end
        res = res[1:end-1] * "  \\\\ \\cline{1-6}\n"
    end
    println(f, res)
end

println(f, "\\end{tabular}%")

close(f)
