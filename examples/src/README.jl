# # Estimating Equations Regression in Julia

# This package fits regression models to data using estimating equations.
# Estimating equations are useful for carrying out regression analysis
# when the data are not independent, or when there are certain forms
# of heteroscedasticity.  This package currently support three methods:
#
# * "Generalized Estimating Equations" (GEE)
#
# * "Quadratic Inference Functions" (QIF)
#
# * "Generalized Expectile Estimating Equations" (GEEE)

ENV["GKSwstype"] = "nul" #hide
using EstimatingEquationsRegression, Random, RDatasets, StatsModels, Plots

## The example below fits linear GEE models to test score data that are clustered
## by classroom, using two different working correlation structures.
da = dataset("SASmixed", "SIMS")
da = sort(da, :Class)
f = @formula(Gain ~ Pretot)
## m1 uses independence working correlation by default.
m1 = fit(GeneralizedEstimatingEquationsModel, f, da, da[:, :Class])
m2 = fit(GeneralizedEstimatingEquationsModel, f, da, da[:, :Class],
         IdentityLink(), ConstantVar(), ExchangeableCor())
corparams(m2) # within-classroom correlation

# Plot the fitted values with a 95% pointwise confidence band.
x = range(extrema(da[:, :Pretot])..., 20)
xm = [ones(20) x]
se = sum((xm * vcov(m2)) .* xm, dims=2).^0.5 # standard errors
yy = xm * coef(m2) # fitted values
plt = plot(x, yy; ribbon=2*se, color=:grey, xlabel="Pretot", ylabel="Gain",
           label=nothing, size=(400,300))
plt = plot!(plt, x, yy, label=nothing)
Plots.savefig(plt, "assets/readme1.svg")

# ![Example plot 1](assets/readme1.svg)

# See the examples in the examples folder and the unit tests in the test folder.

# ## References
#
# Longitudinal Data Analysis Using Generalized Linear Models. KY Liang, S Zeger (1986).
# https://www.biostat.jhsph.edu/~fdominic/teaching/bio655/references/extra/liang.bka.1986.pdf
#
# Efficient estimation for longitudinal data by combining large-dimensional moment condition.
# H Cho, A Qu (2015). https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-9/issue-1/Efficient-estimation-for-longitudinal-data-by-combining-large-dimensional-moment/10.1214/15-EJS1036.full

# A new GEE method to account for heteroscedasticity, using assymetric least-square regressions.
# A Barry, K Oualkacha, A Charpentier (2018). https://arxiv.org/abs/1810.09214
