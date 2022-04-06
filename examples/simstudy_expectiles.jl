using GEE, StatsModels, DataFrames, LinearAlgebra, Statistics

ngrp = 1000
m = 10
p = 3

function gen_response(ngrp, m, p)

    xmat = randn(ngrp * m, p)
    ey = xmat[:, 1] - xmat[:, 3]
    y = copy(ey)

    ii = 0
    id = zeros(ngrp * m)
    for i = 1:ngrp
        y[ii+1:ii+m] .+= randn(m) #randn() .+ randn(m).*sqrt.(1 .+ xmat[ii+1:ii+m, 2].^2)
        id[ii+1:ii+m] .= i
        ii += m
    end

    df = DataFrame(:y => y, :id => id)
    for k = 1:p
        df[:, Symbol("x$(k)")] = xmat[:, k]
    end
    df[:, :x2x] = sqrt.(1 .+ df[:, :x2] .^ 2)

    return df
end

function simstudy()
    nrep = 100
    tau = [0.25, 0.5, 0.75]
    q = length(tau)
    zs = zeros(nrep, q * (p + 1))
    for k = 1:nrep
        df = gen_response(ngrp, m, p)
        m1 = geee(@formula(y ~ x1 + x2x + x3), df, df[:, :id], tau)
        println(k)
        zs[k, :] = coef(m1) ./ sqrt.(diag(vcov(m1)))
        println([dispersion(m1.model, j) for j in eachindex(tau)])
    end
    println(mean(zs, dims = 1))
    println(std(zs, dims = 1))
end

simstudy()
