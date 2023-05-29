```@meta
EditURL = "<unknown>/expectiles_simstudy.jl"
```

Simulation study to assess the sampling properties of
GEEE expectile estimation.

````julia
using EstimatingEquationsRegression, StatsModels, DataFrames, LinearAlgebra, Statistics

# Number of groups of correlated data
ngrp = 1000

# Size of each group
m = 10

# Regression parameters, excluding intercept which is zero.
beta = Float64[1, 0, -1]
p = length(beta)

# Jointly estimate these expectiles
tau = [0.25, 0.5, 0.75]

# Null parameters
ii0 = [5, 7] #[3, 5, 7, 11]

# Non-null parameters
ii1 = [i for i in 1:3*p if !(i in ii0)]

function gen_response(ngrp, m, p)

    # Explanatory variables
    xmat = randn(ngrp * m, p)

    # Expected value of response variable
    ey = xmat * beta

    # This will hold the response values
    y = copy(ey)

    # Generate correlated data for each block
    ii = 0
    id = zeros(ngrp * m)
    for i = 1:ngrp
        y[ii+1:ii+m] .+= randn() .+ randn(m) .* sqrt.(1 .+ xmat[ii+1:ii+m, 2] .^ 2)
        id[ii+1:ii+m] .= i
        ii += m
    end

    # Make a dataframe from the data
    df = DataFrame(:y => y, :id => id)
    for k = 1:p
        df[:, Symbol("x$(k)")] = xmat[:, k]
    end

    # The quantiles and expectiles scale with this value.
    df[:, :x2x] = sqrt.(1 .+ df[:, :x2] .^ 2)

    return df
end

function simstudy()

    # Number of simulation replications
    nrep = 100

    # Number of expectiles to jointly estimate
    q = length(tau)

    # Z-scores
    zs = zeros(nrep, q * (p + 1))

    # Coefficients
    cf = zeros(nrep, q * (p + 1))

    for k = 1:nrep
        df = gen_response(ngrp, m, p)
        m1 = geee(@formula(y ~ x1 + x2x + x3), df, df[:, :id], tau)
        zs[k, :] = coef(m1) ./ sqrt.(diag(vcov(m1)))
        cf[k, :] = coef(m1)
    end

    println("Mean of coefficients:")
    println(mean(cf, dims = 1))

    println("\nMean Z-scores for null coefficients:")
    println(mean(zs[:, ii0], dims = 1))

    println("\nSD of Z-scores for null coefficients:")
    println(std(zs[:, ii0], dims = 1))

    println("\nMean Z-scores for non-null coefficients:")
    println(mean(zs[:, ii1], dims = 1))

    println("\nSD of Z-scores for non-null coefficients:")
    println(std(zs[:, ii1], dims = 1))
end

simstudy()
````

````
Mean of coefficients:
[-0.248305956022642 1.0004069284159738 -0.36059161765156067 -0.9982692685219554 -0.00855612062452041 1.0004346155138808 0.009494716362008876 -0.9986438680013795 0.2340745944987212 1.0011417581971853 0.3777067833861307 -0.9986768531766281]

Mean Z-scores for null coefficients:
[-0.10986953515956976 0.16402799757990244]

SD of Z-scores for null coefficients:
[0.9905867027660789 0.9527365381274587]

Mean Z-scores for non-null coefficients:
[-2.8816904486561934 54.91652015807419 -5.730980377483325 -54.51745020697771 57.84672482537326 -57.493074715982395 2.6900399914749573]

SD of Z-scores for non-null coefficients:
[0.9799839461657127 1.9607986340991055 0.8682616798726036 2.1094969568132775 1.851063585310018 2.049256335983278 1.0604846148894322]

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

