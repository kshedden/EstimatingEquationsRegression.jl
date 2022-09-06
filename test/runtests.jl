using Test, DataFrames, CSV, StatsBase, LinearAlgebra, Distributions
using Printf, StableRNGs, FiniteDifferences
using GEE, GLM, Random

function data1()
    X = [
        3.0 3 2
        1 2 1
        2 3 3
        1 5 4
        4 0 0
        3 2 2
        4 2 0
        6 1 4
        0 0 0
        2 0 4
        8 3 3
        4 4 0
        1 2 1
        2 1 5
    ]
    y = [3.0, 1, 4, 4, 1, 3, 1, 2, 1, 1, 2, 4, 2, 2]
    z = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0]
    g = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3]
    df = DataFrame(y = y, z = z, x1 = X[:, 1], x2 = X[:, 2], x3 = X[:, 3], g = g)

    return y, z, X, g, df
end

function save()
    y, z, X, g, da = data1()
    CSV.write("data1.csv", da)
end

include("runtests_gee.jl")
include("runtests_geee.jl")
include("runtests_qif.jl")

