#=

http://www.stata-press.com/data/r13/nlswork2.dta

=#

using StatFiles, GEE, GLM, DataFrames, StatsModels, Distributions, Statistics

# Fit a model to the clogitid data
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
disp1 = dispersion(m1)

m2 = gee(
    @formula(ln_wage ~ grade + age + age_c^2),
    d1,
    d1[:, :idcode],
    Normal(),
    ExchangeableCor(),
    IdentityLink(),
)
disp2 = dispersion(m2)
