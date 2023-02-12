using Distributions
using GEE
using GLM: IdentityLink, LogLink, LogitLink
using LinearAlgebra
using Statistics
using Printf

# Overall sample size
n = 1000

# Covariates, covariate 1 is intercept, covariate 2 is the only covariate that predicts
# the response
p = 10

# Group size
m = 10

# Number of groups
q = div(n, m)

# Effect size
es = 0.5

function gendat(es, dist)
    X = randn(n, p)
    X[:, 1] .= 1

    # Induce correlations between the null variables and the non-null variable
    r = 0.5
    for k=3:p
        X[:,k] = r*X[:, 2] + sqrt(1-r^2)X[:, k]
    end

    g = kron(1:q, ones(Int, m))
    lp = es*X[:, 2]

    # Drop two null variables
    ii = [i for i in 1:p if !(i in [3, 4])]
    X0 = X[:, ii]

    # Drop a non-null variable
    ii = [i for i in 1:p if i != 2]
    X1 = X[:, ii]

    y = if dist == :Gaussian
        e = randn(q)[g] + randn(n)
        ey = lp
        ey + e
    elseif dist == :Binomial
        e = (randn(q)[g] + randn(n)) / sqrt(2)
        u = cdf(Normal(0, 1), e)
        ey = exp.(lp)
        ey ./= (1 .+ ey)
        qq = quantile.(Poisson.(ey), u)
        clamp.(quantile.(Poisson.(ey), u), 0, 1)
    elseif dist == :Poisson
        e = (randn(q)[g] + randn(n)) / sqrt(2)
        u = cdf(Normal(0, 1), e)
        ey = exp.(lp)
        quantile.(Poisson.(ey), u)
    else
        error(dist)
    end

    return X, X0, X1, y, g
end

function fitmodels(X0, X, y, g, link, varfunc, corstruct)
    m0 = gee(X0, y, g, link, varfunc, corstruct)
    m1 = gee(X, y, g, link, varfunc, corstruct; dofit = false)
    mx = gee(X, y, g, link, varfunc, corstruct)
    st = scoretest(m1, m0)
    return st
end

function runsim(nrep, link, varfunc, corstruct, dist)
    stats = (score_null=Float64[], score_alt=Float64[])
    for i = 1:nrep
        X, X0, X1, y, g = gendat(es, dist)

        # The null is true
        st = fitmodels(X0, X, y, g, link, varfunc, corstruct)
        push!(stats.score_null, st.Stat)

        # The null is false
        st = fitmodels(X1, X, y, g, link, varfunc, corstruct)
        push!(stats.score_alt, st.Stat)
    end
    return stats
end

nrep = 1000

function main()

    for (dist, link, varfunc, corstruct) in [[:Gaussian, IdentityLink(), ConstantVar(), ExchangeableCor()],
                                             [:Binomial, LogitLink(), BinomialVar(), ExchangeableCor()],
                                             [:Poisson, LogLink(), IdentityVar(), ExchangeableCor()]]
        stats = runsim(nrep, link, varfunc, corstruct, dist)
        println(dist)
        for p in [0.5, 0.9, 0.95]

            println(@sprintf("p=%.2f", p))

            q = mean(stats.score_null .>= quantile(Chisq(2), p))
            println(@sprintf("%12.4f Score test under the null", q))

            q = mean(stats.score_alt .>= quantile(Chisq(1), p))
            println(@sprintf("%12.4f Score test under the alternative", q))
        end
    end
end

main()
