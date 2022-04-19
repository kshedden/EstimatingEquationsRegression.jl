#=
A simple example using a dataset from the Stata
documentation.  Use the link below to obtain
the data.

http://www.stata-press.com/data/r13/nlswork2.dta
=#

using StatFiles, GEE, GLM, DataFrames, StatsModels, Distributions, Statistics

# Fit a model to the nlswork2  data
d1 = DataFrame(load("nlswork2.dta"))

d1 = d1[:, [:ln_wage, :grade, :age, :idcode]]
d1 = d1[completecases(d1), :]

d1[:, :age_c] = d1[:, :age] .- mean(d1[:, :age])

m1 = gee(
    @formula(ln_wage ~ grade + age + age_c^2),
    d1,
    d1[:, :idcode],
    Normal(),
    IndependenceCor(),
    IdentityLink(),
)
disp1 = dispersion(m1.model)

m2 = gee(
    @formula(ln_wage ~ grade + age + age_c^2),
    d1,
    d1[:, :idcode],
    Normal(),
    ExchangeableCor(),
    IdentityLink(),
)
disp2 = dispersion(m2.model)
