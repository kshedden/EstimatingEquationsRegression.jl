#=
A simple example using a dataset from the Stata documentation.

Use the link below to obtain the data.

http://www.stata-press.com/data/r13/nlswork2.dta

Compare to the results here:

https://www.stata.com/manuals13/xtxtgee.pdf
=#

using StatFiles
using GEE
using GLM
using DataFrames
using StatsModels
using Distributions
using Statistics

# Fit a model to the nlswork2  data
d1 = DataFrame(load("nlswork2.dta"))

d1 = d1[:, [:ln_wage, :grade, :age, :idcode]]
d1 = d1[completecases(d1), :]
d1 = disallowmissing(d1)

d1[!, :ln_wage] = Float64.(d1[:, :ln_wage])
d1[!, :grade] = Float64.(d1[:, :grade])
d1[!, :age] = Float64.(d1[:, :age])
d1[:, :age2] = d1[:, :age].^2

# Fit a linear model with GEE using independent working correlation.
m1 = gee(
    @formula(ln_wage ~ grade + age + age2),
    d1,
    d1[:, :idcode],
    Normal(),
    IndependenceCor(),
    IdentityLink(),
)
disp1 = dispersion(m1.model)

m2 = gee(
    @formula(ln_wage ~ grade + age + age2),
    d1,
    d1[:, :idcode],
    Normal(),
    ExchangeableCor(),
    IdentityLink(),
)
disp2 = dispersion(m2.model)
