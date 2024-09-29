```@meta
EditURL = "src/expectiles_simstudy.jl"
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
[-0.24995023808975353 1.0005251763314522 -0.36108615443586545 -1.0006499779277593 -0.011154181505994393 0.999937289130272 0.009013751683341347 -1.0006378416362887 0.22809140422684654 0.9986785701305956 0.37893310834980665 -1.000850006873464]

Mean Z-scores for null coefficients:
[-0.13055215930957512 0.14954143511555817]

SD of Z-scores for null coefficients:
[1.040579462885651 0.9746134735148885]

Mean Z-scores for non-null coefficients:
[-2.8960835247116234 55.09565002816378 -5.78297900714302 -54.720908633503186 57.9308563590596 -57.712703391198744 2.6540159000209216]

SD of Z-scores for non-null coefficients:
[1.0704370797571143 2.100947536928869 0.9978569740366025 2.289133212027854 1.9808910446704002 2.087562819446072 1.0682434069834734]

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

