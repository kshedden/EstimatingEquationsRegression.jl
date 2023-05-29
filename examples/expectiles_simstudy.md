```@meta
EditURL = "<unknown>/expectiles_simstudy.jl"
```

Simulation study to assess the sampling properties of
GEEE expectile estimation.

````julia
using GEE, StatsModels, DataFrames, LinearAlgebra, Statistics

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
[-0.2413802308128318 1.0021291111725876 -0.37072225035695583 -0.9986702622969833 -4.654797525465118e-5 1.001519580652245 -0.0022555816644241663 -0.9985059665879515 0.24305445514279447 1.0009375500032527 0.3648920314029845 -0.9979214450148914]

Mean Z-scores for null coefficients:
[-0.0028125749863164317 -0.030326031664663]

SD of Z-scores for null coefficients:
[1.0458180545298608 1.030641335147947]

Mean Z-scores for non-null coefficients:
[-2.798411761379705 54.67369216268675 -5.897615885432134 -54.78565503742246 57.80885196269524 -57.97452674482646 2.8244360344752617]

SD of Z-scores for non-null coefficients:
[1.105139405674583 2.1557462081594054 1.0014579663624599 2.34613842974393 2.0809209538107507 2.0575392039901748 1.0454037335717357]

````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

