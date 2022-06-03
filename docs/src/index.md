# GEE.jl Manual

Fit marginal regression models in Julia using Generalized Estimating
Equations (GEE) and Quadratic Inference Functions (QIF).

## Installation

The package is not yet registered and must be installed from the
[github repository](https://github.com/kshedden/GEE.jl).

## Marginal regression models

Marginal regression models express the conditional mean of a response
variable (also known as the outcome or dependent variable) in terms of
a linear predictor formed from a set of explanatory variables
(covariates).  The data used to fit a marginal regression model are
partitioned into groups or clusters such that observations in the same
group may be correlated, but observations in different groups are
independent.

To fit a marginal regression model using this package, you must have a
variable in your dataset that defines the group to which each
observation belongs.  Before fitting the model, the dataset must be
sorted so that the group values are non-decreasing.  If your data are
in a dataframe `df` and the group variable is `id`, then you can sort
the dataframe using `sort(df, :id)`.

## Fitting a model with GEE

GEE is a technique for fitting marginal regression models that is an
extension of generalized linear modeling (GLM). GLM is supported in
Julia by the [GLM.jl](http://github.com/juliastats/GLM.jl) package.

The easiest way to fit a model using GEE is with the `gee` function
using a dataframe and formula.  A linear model can be fit with GEE
using syntax like this:

```jldoctest
julia> using DataFrames, GEE, RDatasets

julia> gas = dataset("plm", "Gasoline");

julia> gee(@formula(LGasPCar ~ LIncomeP + LRPMG), gas, gas[:, :Country])
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}, Matrix{Float64}}

LGasPCar ~ 1 + LIncomeP + LRPMG

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   2.25114     1.77475    1.27    0.2046  -1.2273     5.72958
LIncomeP     -0.341722    0.303557  -1.13    0.2603  -0.936683   0.253239
LRPMG         0.101071    0.233894   0.43    0.6657  -0.357353   0.559494
─────────────────────────────────────────────────────────────────────────
```

The above model has a linear mean structure, so the conditional mean
of `LGasPCar` (logarithm of motor gasoline consumption per car) is
modeled as a linear function of the explanatory variables `LIncomeP`
(logarithm of real per-capita income) and `LRPMG` (logarithm of real
motor gasoline price), also including an intercept.  This mean
structure is the same as used in a conventional linear model fit using
ordinary least squares (OLS).  However the standard errors are different
due to the non-independence of repeated measures within countries.

Three arguments in the example above are assigned their default values.
Below, these arguments are provided explicitly.
The distribution defaults to `Normal`, the working covariance model
defaults to `IndependenceCor`, and the link function defaults to
`IdentityLink`.

```jldoctest
julia> using DataFrames, GEE, RDatasets, Distributions

julia> gas = dataset("plm", "Gasoline");

julia> gee(@formula(LGasPCar ~ LIncomeP + LRPMG), gas, gas[:, :Country], Normal(), IndependenceCor(), IdentityLink())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}, Matrix{Float64}}

LGasPCar ~ 1 + LIncomeP + LRPMG

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   2.25114     1.77475    1.27    0.2046  -1.2273     5.72958
LIncomeP     -0.341722    0.303557  -1.13    0.2603  -0.936683   0.253239
LRPMG         0.101071    0.233894   0.43    0.6657  -0.357353   0.559494
─────────────────────────────────────────────────────────────────────────
```

An alternative interface based on quasi-likelihood uses explicit link
and variance functions instead of a distribution family (GEE only uses
the family to determine the link function and variance function).

```jldoctest
julia> using DataFrames, GEE, RDatasets

julia> gas = dataset("plm", "Gasoline");

julia> gee(@formula(LGasPCar ~ LIncomeP + LRPMG), gas, gas[:, :Country], IdentityLink(), ConstantVar(), IndependenceCor())
StatsModels.TableRegressionModel{GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}, Matrix{Float64}}

LGasPCar ~ 1 + LIncomeP + LRPMG

Coefficients:
─────────────────────────────────────────────────────────────────────────
                 Coef.  Std. Error      z  Pr(>|z|)  Lower 95%  Upper 95%
─────────────────────────────────────────────────────────────────────────
(Intercept)   2.25114     1.77475    1.27    0.2046  -1.2273     5.72958
LIncomeP     -0.341722    0.303557  -1.13    0.2603  -0.936683   0.253239
LRPMG         0.101071    0.233894   0.43    0.6657  -0.357353   0.559494
─────────────────────────────────────────────────────────────────────────
```

It is also possible to fit a model directly from arrays rather than
using a dataframe and formula:

```jldoctest
julia> using DataFrames, GEE, Distributions, RDatasets

julia> gas = dataset("plm", "Gasoline");

julia> X = ones(size(gas, 1), 3);

julia> X[:, 2] = gas[:, :LIncomeP]; X[:, 3] = gas[:, :LRPMG];

julia> y = gas[:, :LGasPCar]; g = gas[:, :Country];

julia> fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), AR1Cor())
GeneralizedEstimatingEquationsModel{GEE.GEEResp{Float64}, GEE.DensePred{Float64}}:

Coefficients:
─────────────────────────────────────────────────────────────────
        Coef.  Std. Error      z  Pr(>|z|)  Lower 95%   Upper 95%
─────────────────────────────────────────────────────────────────
x1   1.55677    1.01554     1.53    0.1253  -0.433647   3.54719
x2  -0.43769    0.171676   -2.55    0.0108  -0.774169  -0.101211
x3  -0.164282   0.0451351  -3.64    0.0003  -0.252745  -0.0758187
─────────────────────────────────────────────────────────────────
```

Additional examples can be found [here](examples.md).

## Fitting a model with QIF

## Expectile GEE

This package supports expectile regression of clustered data using the
Generalized Expectile Estimating Equations (GEEE) framework.  The
methodology and algorithms are described
[here](https://arxiv.org/abs/1810.09214).
