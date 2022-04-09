# GEE.jl Manual

Fit regression models in Julia using Generalized Estimating Equations
(GEE).

## Installation

The package is not yet registered and must be installed from the
github repository.

## Fitting a model with GEE

GEE is an extension of generalized linear modeling (GLM), which is
supported in Julia using the
[GLM.jl](http://github.com/juliastats/GLM.jl) package.  GEE is used
when the observations are not independent, and can be partioned into
groups or clusters such that observations in the same group may be
correlated, but observations in different groups are always
independent.

To fit a model using GEE, you must have a variable in your dataset
that defines the group to which each observation belongs.  Before
fitting a model with GEE using this package, the dataset must be
sorted so that the group values are non-decreasing.  If your data are
in a dataframe `df` and the group variable is `id`, then you can sort
the dataframe for use by this package using `sort(df, :id)`.

The easiest way to fit a model using GEE is with the `gee` function
using a dataframe and formula.  A linear model can be fit using syntax
like this:

```gee(@formula(y ~ x + z), df, df[:, :id])```

The above model has a linear mean structure, so the conditional mean
of `y` is modeled as a linear function of the explanatory variables
`x` and `z`, also including an intercept.  This mean structure is the
same as might be used in a linear model fit using ordinary least
squares (OLS).

There are three additional important arguments which in the example
above are given their default values.  Below these arguments are
provided explicitly.  The distribution defaults to `Normal`, the
working covariance model defaults to `IndependenceCor`, and the link
function defaults to `IdentityLink`.

```gee(@formula(y ~ x + z), df, df[:, :id], Normal(), IndependenceCor(), IdentityLink())```

It is also possible to fit a model directly from arrays rather than
using a dataframe and formula.  The syntax in this case is shown
below:

```fit(GeneralizedEstimatingEquationsModel, X, y, g, Normal(), AR1Cor())```

[Examples](examples.md)

## Expectile GEE

This package supports expectile regression of clustered data using the
Generalized Expectile Estimating Equations (GEEE) framework.  The
methodology and algorithms are described
[here](https://arxiv.org/abs/1810.09214).
